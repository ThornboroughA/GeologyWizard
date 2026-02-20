from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...models import (
    BoundaryKinematics,
    BoundarySegment,
    BoundaryStateClass,
    BoundaryStateRecord,
    BoundaryType,
    FrameDiagnostics,
    GeoEvent,
    PlausibilityCheck,
    PlateFeature,
    PlateKinematics,
    PlateLifecycleState,
    ProjectConfig,
    TimelineFrame,
    UncertaintySummary,
)
from ...utils import stable_hash
from ..pygplates_adapter import PygplatesModelCache
from ..tectonic_backends import derive_seed_bundle
from .boundary_machine import update_boundary_state
from .events import update_event_ledger
from .force_solver import solve_plate_kinematics
from .lifecycle import LifecycleUpdate, boundary_average_oceanic_age, initialize_oceanic_grid, update_oceanic_grid
from .state import BoundaryStateV2, PlateStateV2, RunStateV2
from .supercontinent import update_supercontinent_state


@dataclass
class BackendFrameResultV2:
    frame: TimelineFrame
    diagnostics: FrameDiagnostics
    strain_field: np.ndarray
    coverage_ratio: float
    kinematic_digest: str
    uncertainty_digest: str
    fallback_used: bool
    preview_height_field: np.ndarray
    oceanic_age_field: np.ndarray
    crust_type_field: np.ndarray
    crust_thickness_field: np.ndarray
    tectonic_potential_field: np.ndarray
    plausibility_checks: list[PlausibilityCheck]


def _wrap_lon(value: float) -> float:
    wrapped = ((value + 180.0) % 360.0) - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _clamp_lat(value: float) -> float:
    return max(-89.0, min(89.0, value))


def _deg_per_myr_from_cm_yr(cm_yr: float) -> float:
    return cm_yr * 0.09


def _vector_from_speed_azimuth(speed_cm_yr: float, azimuth_deg: float) -> tuple[float, float]:
    rad = math.radians(azimuth_deg)
    return speed_cm_yr * math.cos(rad), speed_cm_yr * math.sin(rad)


def _distance_deg(a: PlateStateV2, b: PlateStateV2) -> float:
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)
    alat = math.radians(a.lat)
    blat = math.radians(b.lat)
    hav = math.sin(dlat / 2.0) ** 2 + math.cos(alat) * math.cos(blat) * math.sin(dlon / 2.0) ** 2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(hav))))


def _polygon_ring(state: PlateStateV2, time_ma: int, points: int = 24) -> list[list[float]]:
    ring: list[list[float]] = []
    for idx in range(points):
        theta = 2.0 * math.pi * (idx / points)
        anisotropy = 1.0 + 0.12 * math.sin(theta * 3.0 + state.plate_id * 0.51 + time_ma * 0.004)
        radius = state.radius_deg * anisotropy
        nlat = _clamp_lat(state.lat + radius * math.sin(theta))
        lon_scale = max(0.24, math.cos(math.radians(state.lat)))
        nlon = _wrap_lon(state.lon + (radius * math.cos(theta) / lon_scale))
        ring.append([round(nlon, 5), round(nlat, 5)])
    ring.append(ring[0])
    return ring


def _ordered_pairs(plates: list[PlateStateV2], mode: str) -> list[tuple[PlateStateV2, PlateStateV2]]:
    ordered = sorted(plates, key=lambda plate: (plate.lon, plate.lat))
    pairs: set[tuple[int, int]] = set()

    def add_pair(left: PlateStateV2, right: PlateStateV2) -> None:
        pid = tuple(sorted((left.plate_id, right.plate_id)))
        pairs.add(pid)

    for idx, left in enumerate(ordered):
        right = ordered[(idx + 1) % len(ordered)]
        add_pair(left, right)

    if mode == "hybrid_rigor":
        for left in ordered:
            nearest = min(
                (plate for plate in ordered if plate.plate_id != left.plate_id),
                key=lambda candidate: _distance_deg(left, candidate),
            )
            add_pair(left, nearest)

    lookup = {plate.plate_id: plate for plate in plates}
    return [(lookup[a], lookup[b]) for a, b in sorted(pairs)]


def _map_state_class_to_boundary_type(state_class: BoundaryStateClass) -> BoundaryType:
    if state_class in {BoundaryStateClass.ridge, BoundaryStateClass.rift}:
        return BoundaryType.divergent
    if state_class in {BoundaryStateClass.subduction, BoundaryStateClass.collision, BoundaryStateClass.suture}:
        return BoundaryType.convergent
    return BoundaryType.transform


def _continuity_score(previous: PlateStateV2 | None, current: PlateStateV2, max_plate_velocity: float) -> float:
    if previous is None:
        return 1.0

    speed_delta = abs(current.velocity_cm_yr - previous.velocity_cm_yr) / max(1.0, max_plate_velocity)
    heading_delta = abs(((current.azimuth_deg - previous.azimuth_deg + 180.0) % 360.0) - 180.0) / 180.0
    radius_delta = abs(current.radius_deg - previous.radius_deg) / max(1.0, previous.radius_deg)
    score = 1.0 - min(1.0, 0.45 * speed_delta + 0.4 * heading_delta + 0.15 * radius_delta)
    return round(score, 4)


def _collect_time_local_edits(edits: list[dict], time_ma: int) -> list[dict]:
    matches: list[dict] = []
    for edit in edits:
        payload = edit.get("payload", {})
        duration = int(payload.get("durationMyr", 30))
        if abs(int(edit.get("timeMa", time_ma)) - time_ma) <= duration:
            matches.append(edit)
    return matches


def _apply_plate_edits(plate: PlateStateV2, edits: list[dict]) -> tuple[float, float]:
    speed_gain = 0.0
    azimuth_delta = 0.0
    for edit in edits:
        if edit.get("editType") not in {"rift_initiation", "rift_start"}:
            continue
        payload = edit.get("payload", {})
        target_plate = payload.get("plateId")
        if target_plate not in (None, plate.plate_id):
            continue
        speed_gain += float(payload.get("speedGain", 0.6))
        azimuth_delta += float(payload.get("azimuthDelta", 18.0))
    return speed_gain, azimuth_delta


def _apply_boundary_edits(boundary: BoundaryStateV2, edits: list[dict]) -> BoundaryStateV2:
    for edit in edits:
        payload = edit.get("payload", {})
        target_segment = payload.get("segmentId")
        if target_segment and target_segment != boundary.segment_id:
            continue

        if edit.get("editType") == "boundary_override":
            forced_type = payload.get("boundaryType")
            if forced_type in {"convergent", "divergent", "transform"}:
                if forced_type == "convergent":
                    boundary.state_class = BoundaryStateClass.subduction
                elif forced_type == "divergent":
                    boundary.state_class = BoundaryStateClass.ridge
                else:
                    boundary.state_class = BoundaryStateClass.transform

        if edit.get("editType") == "subducting_side_override":
            forced_side = payload.get("subductingSide")
            if forced_side in {"left", "right", "none"}:
                boundary.subducting_side = forced_side

    return boundary


def _build_strain_field(boundaries: list[BoundarySegment], kinematics: list[BoundaryKinematics], width: int = 256, height: int = 128) -> np.ndarray:
    lon = np.linspace(-180.0, 180.0, width, dtype=np.float32)
    lat = np.linspace(-90.0, 90.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(lon, lat)

    field = np.zeros((height, width), dtype=np.float32)
    kin_by_id = {item.segmentId: item for item in kinematics}

    for boundary in boundaries:
        kin = kin_by_id[boundary.segmentId]
        coords = boundary.geometry["coordinates"]
        cx = (coords[0][0] + coords[1][0]) * 0.5
        cy = (coords[0][1] + coords[1][1]) * 0.5
        sigma = 8.0
        influence = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2.0 * sigma * sigma))
        field += influence * float(kin.strainRate)

    max_value = float(field.max())
    if max_value > 0:
        field /= max_value
    return field.astype(np.float32)


def _build_plausibility_checks(
    *,
    project_id: str,
    time_ma: int,
    config: ProjectConfig,
    plate_kinematics: list[PlateKinematics],
    boundary_states: list[BoundaryStateV2],
    lifecycle: LifecycleUpdate,
    short_lived_orogeny_count: int,
    uncoupled_volcanic_belts: int,
    coverage_ratio: float,
    frame_coverage_ratio: float,
) -> list[PlausibilityCheck]:
    checks: list[PlausibilityCheck] = []

    velocity_jump_max = max(
        [
            abs(item.velocityCmYr - item.divergenceCmYr + item.convergenceCmYr) * 0.05
            for item in plate_kinematics
        ]
        + [0.0]
    )
    checks.append(
        PlausibilityCheck(
            checkId="continuity.max_velocity_jump_cm_per_yr",
            severity="warning" if velocity_jump_max > 2.0 else "info",
            timeRangeMa=(time_ma, time_ma),
            regionOrPlateIds=[f"plate:{item.plateId}" for item in plate_kinematics[:6]],
            observedValue=round(velocity_jump_max, 4),
            expectedRangeOrRule="warn > 2.0, error > 5.0",
            explanation="Largest per-plate kinematic jump proxy for this frame.",
            suggestedFix="Increase temporal smoothing or lower forcing perturbations when persistent.",
        )
    )

    motion_mismatch_count = sum(item.motion_mismatch_count > 0 for item in boundary_states)
    checks.append(
        PlausibilityCheck(
            checkId="boundary.motion_mismatch_count",
            severity="error" if motion_mismatch_count > 0 else "info",
            timeRangeMa=(time_ma, time_ma),
            regionOrPlateIds=[f"boundary:{item.segment_id}" for item in boundary_states if item.motion_mismatch_count > 0][:8],
            observedValue=motion_mismatch_count,
            expectedRangeOrRule="must equal 0",
            explanation="Boundary semantic class should match relative motion sign and shear dominance.",
            suggestedFix="Adjust transition hysteresis or force coupling until mismatches clear.",
        )
    )

    elapsed_myr = max(1, config.startTimeMa - time_ma)
    type_flip_rate = (
        sum(item.transition_count for item in boundary_states)
        / max(1.0, elapsed_myr / 100.0)
    )
    checks.append(
        PlausibilityCheck(
            checkId="boundary.type_flip_rate_per_100myr",
            severity="warning" if type_flip_rate > 6 else "info",
            timeRangeMa=(config.startTimeMa, time_ma),
            regionOrPlateIds=[f"boundary:{item.segment_id}" for item in boundary_states[:8]],
            observedValue=round(type_flip_rate, 4),
            expectedRangeOrRule="warn > 6, error > 10",
            explanation="Rapid repeated boundary-class flipping indicates unstable semantics.",
            suggestedFix="Increase transition persistence windows or reduce forcing oscillation.",
        )
    )

    checks.append(
        PlausibilityCheck(
            checkId="lifecycle.unexplained_plate_births",
            severity="error" if lifecycle.unexplained_plate_births > 0 else "info",
            timeRangeMa=(time_ma, time_ma),
            observedValue=lifecycle.unexplained_plate_births,
            expectedRangeOrRule="must equal 0",
            explanation="Plate creation must be tied to deterministic lifecycle transitions.",
            suggestedFix="Route births through explicit rift/spreading topology actions.",
        )
    )

    checks.append(
        PlausibilityCheck(
            checkId="lifecycle.unexplained_plate_deaths",
            severity="error" if lifecycle.unexplained_plate_deaths > 0 else "info",
            timeRangeMa=(time_ma, time_ma),
            observedValue=lifecycle.unexplained_plate_deaths,
            expectedRangeOrRule="must equal 0",
            explanation="Plate removal must happen through deterministic subduction/collision closure.",
            suggestedFix="Tie removals to closed-basin topology events with area accounting.",
        )
    )

    checks.append(
        PlausibilityCheck(
            checkId="lifecycle.net_area_balance_error",
            severity="error" if lifecycle.net_area_balance_error > 0.01 else "info",
            timeRangeMa=(time_ma, time_ma),
            observedValue=round(lifecycle.net_area_balance_error, 6),
            expectedRangeOrRule="error > 0.01",
            explanation="Area balance check across oceanic + continental raster fractions.",
            suggestedFix="Normalize lifecycle updates to preserve closure per timestep.",
        )
    )

    checks.append(
        PlausibilityCheck(
            checkId="events.short_lived_orogeny_count",
            severity="warning" if short_lived_orogeny_count > 0 else "info",
            timeRangeMa=(time_ma, time_ma),
            observedValue=short_lived_orogeny_count,
            expectedRangeOrRule="warn when > 0 for duration < 10 Myr",
            explanation="Orogenic pulses shorter than geologically plausible windows are flagged.",
            suggestedFix="Increase collision persistence or post-collision decay tail.",
        )
    )

    checks.append(
        PlausibilityCheck(
            checkId="events.uncoupled_volcanic_belts",
            severity="warning" if uncoupled_volcanic_belts > 0 else "info",
            timeRangeMa=(time_ma, time_ma),
            observedValue=uncoupled_volcanic_belts,
            expectedRangeOrRule="must equal 0",
            explanation="Arc volcanism should remain coupled to active subduction states.",
            suggestedFix="Enforce coupling between volcanic synthesis and subduction state machine.",
        )
    )

    spatial_gap = 1.0 - max(0.0, min(1.0, frame_coverage_ratio))
    checks.append(
        PlausibilityCheck(
            checkId="coverage.spatial_gap_fraction",
            severity="warning" if spatial_gap > 0.25 else "info",
            timeRangeMa=(time_ma, time_ma),
            observedValue=round(spatial_gap, 4),
            expectedRangeOrRule="warn > 0.25",
            explanation="Fraction of modeled space with weak tectonic coverage signals.",
            suggestedFix="Improve boundary sampling density or topology availability.",
        )
    )

    temporal_gap = 0.0 if coverage_ratio >= 0.82 else float(config.timeIncrementMyr)
    checks.append(
        PlausibilityCheck(
            checkId="coverage.temporal_gap_myr",
            severity="warning" if temporal_gap > 0 else "info",
            timeRangeMa=(time_ma, time_ma),
            observedValue=temporal_gap,
            expectedRangeOrRule="warn when fallback dominates frame",
            explanation="Fallback-temporal coverage proxy based on combined coverage ratio.",
            suggestedFix="Load richer topology/reconstruction inputs or tighten fallback controls.",
        )
    )

    return checks


class TectonicStateV2Backend:
    mode_name = "tectonic_state_v2"

    def initialize(self, config: ProjectConfig, seeds: Any, pygplates_cache: PygplatesModelCache) -> RunStateV2:
        rng = random.Random(seeds.plates)
        plate_states: list[PlateStateV2] = []

        continental_count = max(3, int(round(config.plateCount * 0.42)))
        golden = math.pi * (3.0 - math.sqrt(5.0))

        for plate_idx in range(config.plateCount):
            lat = math.degrees(math.asin(max(-1.0, min(1.0, 1.0 - (2.0 * (plate_idx + 0.5) / config.plateCount)))))
            lon = _wrap_lon(math.degrees(golden * plate_idx) + rng.uniform(-8.5, 8.5))
            radius = rng.uniform(7.0, 16.0)
            velocity = rng.uniform(0.8, min(config.maxPlateVelocityCmYr, 9.0))
            azimuth = rng.uniform(0.0, 360.0)
            craton_factor = min(1.0, max(0.0, 0.3 + rng.random() * 0.7))
            is_continental = plate_idx < continental_count
            plate_states.append(
                PlateStateV2(
                    plate_id=plate_idx + 1,
                    name=f"Plate {plate_idx + 1}",
                    lat=lat,
                    lon=lon,
                    radius_deg=radius,
                    velocity_cm_yr=velocity,
                    azimuth_deg=azimuth,
                    craton_factor=craton_factor,
                    is_continental=is_continental,
                )
            )

        width = config.coreGridWidth or (720 if config.simulationMode.value == "hybrid_rigor" else 512)
        height = config.coreGridHeight or (360 if config.simulationMode.value == "hybrid_rigor" else 256)
        oceanic = initialize_oceanic_grid(width=width, height=height, seed=config.seed, plate_states=plate_states)

        return RunStateV2(
            config=config,
            plate_states=plate_states,
            previous_plate_states={},
            boundaries={},
            oceanic=oceanic,
            events={},
            seeds=seeds,
            pygplates=pygplates_cache,
        )

    def advance(self, state: RunStateV2, config: ProjectConfig, time_ma: int, step_myr: int, edits: list[dict]) -> None:
        plate_lookup = {plate.plate_id: plate for plate in state.plate_states}
        local_edits = _collect_time_local_edits(edits, time_ma)

        continental = [plate for plate in state.plate_states if plate.is_continental]
        state.supercontinent = update_supercontinent_state(
            state=state.supercontinent,
            continental_plates=continental,
            time_ma=time_ma,
            step_myr=step_myr,
        )

        next_states: list[PlateStateV2] = []
        for plate in state.plate_states:
            neighbors = [candidate for candidate in state.plate_states if candidate.plate_id != plate.plate_id]
            neighbor_distance = min(_distance_deg(plate, candidate) for candidate in neighbors)
            plate_boundaries = [
                boundary
                for boundary in state.boundaries.values()
                if plate.plate_id in (boundary.left_plate_id, boundary.right_plate_id)
            ]

            next_speed, next_azimuth = solve_plate_kinematics(
                plate=plate,
                boundaries=plate_boundaries,
                neighbor_mean_distance_deg=neighbor_distance,
                max_velocity_cm_yr=config.maxPlateVelocityCmYr,
                supercontinent_bias_strength=config.supercontinentBiasStrength,
                supercontinent_phase=state.supercontinent.phase,
                supercontinent_center=state.supercontinent.centroid,
                seed=state.seeds.plates,
                time_ma=time_ma,
            )

            speed_gain, azimuth_delta = _apply_plate_edits(plate, local_edits)
            next_speed = max(0.25, min(config.maxPlateVelocityCmYr, next_speed + speed_gain))
            next_azimuth = (next_azimuth + azimuth_delta) % 360.0

            deg_per_myr = _deg_per_myr_from_cm_yr(next_speed) * step_myr
            rad = math.radians(next_azimuth)
            dlat = deg_per_myr * math.sin(rad)
            lon_scale = max(0.25, math.cos(math.radians(plate.lat)))
            dlon = (deg_per_myr * math.cos(rad)) / lon_scale

            collision_factor = sum(
                1
                for boundary in plate_boundaries
                if boundary.state_class in {BoundaryStateClass.collision, BoundaryStateClass.suture}
            )
            radius_adjust = -0.02 * collision_factor + 0.01 * math.sin((plate.plate_id * 1.3) + time_ma * 0.02)

            next_states.append(
                PlateStateV2(
                    plate_id=plate.plate_id,
                    name=plate.name,
                    lat=_clamp_lat(plate.lat + dlat),
                    lon=_wrap_lon(plate.lon + dlon),
                    radius_deg=max(4.0, min(23.0, plate.radius_deg + radius_adjust)),
                    velocity_cm_yr=next_speed,
                    azimuth_deg=next_azimuth,
                    craton_factor=plate.craton_factor,
                    is_continental=plate.is_continental,
                )
            )

        state.previous_plate_states = plate_lookup
        state.plate_states = next_states

    def build_frame(
        self,
        *,
        project_id: str,
        config: ProjectConfig,
        state: RunStateV2,
        time_ma: int,
        edits: list[dict],
    ) -> BackendFrameResultV2:
        assert state.oceanic is not None

        plate_features: list[PlateFeature] = []
        plate_kinematics: list[PlateKinematics] = []
        convergence_by_plate: dict[int, float] = {plate.plate_id: 0.0 for plate in state.plate_states}
        divergence_by_plate: dict[int, float] = {plate.plate_id: 0.0 for plate in state.plate_states}

        for plate in state.plate_states:
            ring = _polygon_ring(plate, time_ma)
            plate_features.append(
                PlateFeature(
                    plateId=plate.plate_id,
                    name=plate.name,
                    geometry={"type": "Polygon", "coordinates": [ring]},
                    validTime=(float(config.startTimeMa), float(config.endTimeMa)),
                    reconstructionPlateId=plate.plate_id,
                )
            )

        boundaries: list[BoundarySegment] = []
        boundary_kinematics: list[BoundaryKinematics] = []
        boundary_state_records: list[BoundaryStateRecord] = []

        boundary_motion_mismatch_count = 0
        boundary_midpoints: dict[str, tuple[float, float]] = {}
        boundary_geometries: dict[str, dict] = {}

        local_edits = _collect_time_local_edits(edits, time_ma)

        for left, right in _ordered_pairs(state.plate_states, config.simulationMode.value):
            left_v = _vector_from_speed_azimuth(left.velocity_cm_yr, left.azimuth_deg)
            right_v = _vector_from_speed_azimuth(right.velocity_cm_yr, right.azimuth_deg)
            rel_v = (right_v[0] - left_v[0], right_v[1] - left_v[1])

            dx = right.lon - left.lon
            dy = right.lat - left.lat
            norm = max(1e-6, math.hypot(dx, dy))
            tangent = (dx / norm, dy / norm)
            normal = (-tangent[1], tangent[0])

            normal_velocity = rel_v[0] * normal[0] + rel_v[1] * normal[1]
            tangential_velocity = rel_v[0] * tangent[0] + rel_v[1] * tangent[1]
            relative_velocity = math.hypot(rel_v[0], rel_v[1])

            seg_id = f"seg_{min(left.plate_id, right.plate_id)}_{max(left.plate_id, right.plate_id)}"
            coords = [[round(left.lon, 5), round(left.lat, 5)], [round(right.lon, 5), round(right.lat, 5)]]

            avg_oceanic_age = boundary_average_oceanic_age(state.oceanic, coords)

            boundary_state = state.boundaries.get(seg_id)
            if boundary_state is None:
                boundary_state = BoundaryStateV2(
                    segment_id=seg_id,
                    left_plate_id=left.plate_id,
                    right_plate_id=right.plate_id,
                    state_class=BoundaryStateClass.passive_margin,
                    last_transition_ma=time_ma,
                    type_persistence_myr=config.timeIncrementMyr,
                )

            boundary_state, mismatch = update_boundary_state(
                boundary=boundary_state,
                normal_velocity_cm_yr=normal_velocity,
                tangential_velocity_cm_yr=tangential_velocity,
                relative_velocity_cm_yr=relative_velocity,
                average_oceanic_age_myr=avg_oceanic_age,
                left_plate_is_continental=left.is_continental,
                right_plate_is_continental=right.is_continental,
                step_myr=config.timeIncrementMyr,
                time_ma=time_ma,
            )
            boundary_state = _apply_boundary_edits(boundary_state, local_edits)
            state.boundaries[seg_id] = boundary_state

            boundary_type = _map_state_class_to_boundary_type(boundary_state.state_class)

            if boundary_type == BoundaryType.convergent:
                convergence_by_plate[left.plate_id] += abs(normal_velocity)
                convergence_by_plate[right.plate_id] += abs(normal_velocity)
            elif boundary_type == BoundaryType.divergent:
                divergence_by_plate[left.plate_id] += abs(normal_velocity)
                divergence_by_plate[right.plate_id] += abs(normal_velocity)

            boundary = BoundarySegment(
                segmentId=seg_id,
                leftPlateId=left.plate_id,
                rightPlateId=right.plate_id,
                boundaryType=boundary_type,
                digitizationDirection="forward",
                subductingSide=boundary_state.subducting_side,
                isActive=True,
                geometry={"type": "LineString", "coordinates": coords},
            )

            kinematics = BoundaryKinematics(
                segmentId=seg_id,
                relativeVelocityCmYr=round(relative_velocity, 4),
                normalVelocityCmYr=round(normal_velocity, 4),
                tangentialVelocityCmYr=round(tangential_velocity, 4),
                strainRate=round(abs(normal_velocity) * 0.022 + abs(tangential_velocity) * 0.01, 5),
                recommendedBoundaryType=boundary_type,
            )

            boundaries.append(boundary)
            boundary_kinematics.append(kinematics)
            boundary_state_records.append(
                BoundaryStateRecord(
                    segmentId=seg_id,
                    stateClass=boundary_state.state_class,
                    lastTransitionMa=boundary_state.last_transition_ma,
                    typePersistenceMyr=boundary_state.type_persistence_myr,
                    polarityFlipCount=boundary_state.polarity_flip_count,
                    transitionCount=boundary_state.transition_count,
                    subductionFlux=round(boundary_state.subduction_flux, 4),
                    averageOceanicAgeMyr=round(boundary_state.average_oceanic_age_myr, 4),
                    motionMismatch=bool(mismatch),
                )
            )

            boundary_midpoints[seg_id] = (
                (coords[0][0] + coords[1][0]) * 0.5,
                (coords[0][1] + coords[1][1]) * 0.5,
            )
            boundary_geometries[seg_id] = {"type": "LineString", "coordinates": coords}

            if mismatch:
                boundary_motion_mismatch_count += 1

        lifecycle = update_oceanic_grid(
            grid=state.oceanic,
            boundary_states=list(state.boundaries.values()),
            boundary_midpoints=boundary_midpoints,
            step_myr=config.timeIncrementMyr,
            seed=config.seed,
        )

        state.supercontinent = update_supercontinent_state(
            state=state.supercontinent,
            continental_plates=[plate for plate in state.plate_states if plate.is_continental],
            time_ma=time_ma,
            step_myr=config.timeIncrementMyr,
        )

        event_result = update_event_ledger(
            ledger=state.events,
            boundaries=list(state.boundaries.values()),
            boundary_geometries=boundary_geometries,
            time_ma=time_ma,
            step_myr=config.timeIncrementMyr,
        )

        events: list[GeoEvent] = event_result.events

        # Expert event gain overrides remain deterministic and window-scoped.
        event_gain = 1.0
        for edit in local_edits:
            if edit.get("editType") in {"event_gain", "event_boost"}:
                payload = edit.get("payload", {})
                event_gain += float(payload.get("gain", payload.get("boost", 0.12)))

        if event_gain != 1.0:
            events = [
                event.model_copy(update={"intensity": round(min(1.0, event.intensity * event_gain), 4)})
                for event in events
            ]

        rotation_outliers = 0
        for plate in state.plate_states:
            previous = state.previous_plate_states.get(plate.plate_id)
            continuity = _continuity_score(previous, plate, config.maxPlateVelocityCmYr)
            if previous is not None:
                heading_delta = abs(((plate.azimuth_deg - previous.azimuth_deg + 180.0) % 360.0) - 180.0)
                if heading_delta > 55.0:
                    rotation_outliers += 1

            plate_kinematics.append(
                PlateKinematics(
                    plateId=plate.plate_id,
                    velocityCmYr=round(plate.velocity_cm_yr, 4),
                    azimuthDeg=round(plate.azimuth_deg, 4),
                    convergenceCmYr=round(convergence_by_plate[plate.plate_id], 4),
                    divergenceCmYr=round(divergence_by_plate[plate.plate_id], 4),
                    continuityScore=continuity,
                )
            )

        coverage_ratio = state.pygplates.coverage_hint if state.pygplates else 0.74
        frame_coverage_ratio = max(0.0, min(1.0, 1.0 - lifecycle.net_area_balance_error - (0.0 if boundary_motion_mismatch_count == 0 else 0.05)))
        combined_coverage = round((coverage_ratio + frame_coverage_ratio) * 0.5, 4)

        if config.simulationMode.value == "hybrid_rigor":
            k_u, e_u, t_u = 0.16, 0.2, 0.24
        else:
            k_u, e_u, t_u = 0.24, 0.3, 0.34

        if config.rigorProfile.value == "research":
            k_u = max(0.08, k_u - 0.05)
            e_u = max(0.1, e_u - 0.05)
            t_u = max(0.12, t_u - 0.04)

        uncertainty = UncertaintySummary(
            kinematic=round(min(1.0, k_u + (1.0 - combined_coverage) * 0.33), 4),
            event=round(min(1.0, e_u + (0.03 if len(events) < len(boundaries) * 0.3 else 0.0)), 4),
            terrain=round(min(1.0, t_u + (0.04 if lifecycle.oceanic_age_p99_myr > 280 else 0.0)), 4),
            coverage=round(1.0 - combined_coverage, 4),
        )

        lifecycle_state = PlateLifecycleState(
            unexplainedPlateBirths=lifecycle.unexplained_plate_births,
            unexplainedPlateDeaths=lifecycle.unexplained_plate_deaths,
            netAreaBalanceError=lifecycle.net_area_balance_error,
            continentalAreaFraction=lifecycle.continental_area_fraction,
            oceanicAreaFraction=lifecycle.oceanic_area_fraction,
            oceanicAgeP99Myr=lifecycle.oceanic_age_p99_myr,
            supercontinentPhase=state.supercontinent.phase,  # type: ignore[arg-type]
            supercontinentLargestClusterFraction=state.supercontinent.largest_cluster_fraction,
            supercontinentCycleCount=state.supercontinent.cycle_count,
            shortLivedOrogenyCount=event_result.short_lived_orogeny_count,
            uncoupledVolcanicBelts=event_result.uncoupled_volcanic_belts,
        )

        frame = TimelineFrame(
            timeMa=time_ma,
            plateGeometries=plate_features,
            boundaryGeometries=boundaries,
            eventOverlays=events,
            plateKinematics=plate_kinematics,
            boundaryKinematics=boundary_kinematics,
            boundaryStates=boundary_state_records,
            plateLifecycleState=lifecycle_state,
            strainFieldRef=None,
            oceanicAgeFieldRef=None,
            crustTypeFieldRef=None,
            crustThicknessFieldRef=None,
            tectonicPotentialFieldRef=None,
            uncertaintySummary=uncertainty,
            previewHeightFieldRef="",
        )

        continuity_violations = [
            f"plate_{item.plateId}_continuity_low"
            for item in plate_kinematics
            if item.continuityScore < 0.22 or item.velocityCmYr > config.maxPlateVelocityCmYr
        ]
        boundary_consistency_issues = [
            f"{state.segment_id}_motion_mismatch"
            for state in state.boundaries.values()
            if state.motion_mismatch_count > 0
        ]

        checks = _build_plausibility_checks(
            project_id=project_id,
            time_ma=time_ma,
            config=config,
            plate_kinematics=plate_kinematics,
            boundary_states=list(state.boundaries.values()),
            lifecycle=lifecycle,
            short_lived_orogeny_count=event_result.short_lived_orogeny_count,
            uncoupled_volcanic_belts=event_result.uncoupled_volcanic_belts,
            coverage_ratio=combined_coverage,
            frame_coverage_ratio=frame_coverage_ratio,
        )

        diagnostics = FrameDiagnostics(
            projectId=project_id,
            timeMa=time_ma,
            continuityViolations=continuity_violations,
            boundaryConsistencyIssues=boundary_consistency_issues,
            coverageGapRatio=round(1.0 - combined_coverage, 4),
            warnings=[
                "pygplates_models_not_loaded" if not (state.pygplates and state.pygplates.available) else "",
                "oceanic_age_p99_high" if lifecycle.oceanic_age_p99_myr > 280 else "",
            ],
            pygplatesStatus=state.pygplates.status if state.pygplates else "unavailable",
            metrics={
                "continuity.max_velocity_jump_cm_per_yr": max(
                    [
                        abs(item.velocityCmYr - item.divergenceCmYr + item.convergenceCmYr) * 0.05
                        for item in plate_kinematics
                    ]
                    + [0.0]
                ),
                "continuity.rotation_outlier_count": float(rotation_outliers),
                "boundary.motion_mismatch_count": float(boundary_motion_mismatch_count),
                "boundary.polarity_flip_count": float(sum(item.polarity_flip_count for item in state.boundaries.values())),
                "boundary.type_flip_rate_per_100myr": float(
                    sum(item.transition_count for item in state.boundaries.values())
                    / max(1.0, (config.startTimeMa - time_ma) / 100.0)
                ),
                "lifecycle.unexplained_plate_births": float(lifecycle.unexplained_plate_births),
                "lifecycle.unexplained_plate_deaths": float(lifecycle.unexplained_plate_deaths),
                "lifecycle.net_area_balance_error": float(lifecycle.net_area_balance_error),
                "events.short_lived_orogeny_count": float(event_result.short_lived_orogeny_count),
                "events.uncoupled_volcanic_belts": float(event_result.uncoupled_volcanic_belts),
                "coverage.spatial_gap_fraction": float(max(0.0, 1.0 - frame_coverage_ratio)),
                "coverage.temporal_gap_myr": float(0.0 if combined_coverage >= 0.82 else config.timeIncrementMyr),
            },
            checkIds=[item.checkId for item in checks],
        )
        diagnostics.warnings = [warning for warning in diagnostics.warnings if warning]

        strain_field = _build_strain_field(boundaries, boundary_kinematics)

        return BackendFrameResultV2(
            frame=frame,
            diagnostics=diagnostics,
            strain_field=strain_field,
            coverage_ratio=combined_coverage,
            kinematic_digest=stable_hash([item.model_dump(mode="json") for item in plate_kinematics]),
            uncertainty_digest=stable_hash(uncertainty.model_dump(mode="json")),
            fallback_used=not (state.pygplates and state.pygplates.available),
            preview_height_field=state.oceanic.terrain_height.astype(np.float32),
            oceanic_age_field=state.oceanic.oceanic_age_myr.astype(np.float32),
            crust_type_field=state.oceanic.crust_type.astype(np.uint8),
            crust_thickness_field=state.oceanic.crust_thickness_km.astype(np.float32),
            tectonic_potential_field=state.oceanic.tectonic_potential.astype(np.float32),
            plausibility_checks=checks,
        )


def build_backend_v2() -> TectonicStateV2Backend:
    return TectonicStateV2Backend()


__all__ = [
    "BackendFrameResultV2",
    "TectonicStateV2Backend",
    "build_backend_v2",
    "derive_seed_bundle",
]
