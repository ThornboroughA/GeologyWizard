from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..models import (
    BoundaryKinematics,
    BoundarySegment,
    BoundaryType,
    FrameDiagnostics,
    GeoEvent,
    GeoEventType,
    PlateFeature,
    PlateKinematics,
    ProjectConfig,
    RigorProfile,
    SeedBundle,
    SimulationMode,
    TimelineFrame,
    UncertaintySummary,
)
from ..utils import stable_hash
from .pygplates_adapter import PygplatesModelCache


@dataclass
class PlateState:
    plate_id: int
    name: str
    lat: float
    lon: float
    radius_deg: float
    velocity_cm_yr: float
    azimuth_deg: float
    craton_factor: float


@dataclass
class BoundaryHistory:
    boundary_type: BoundaryType
    streak_myr: int = 0
    cumulative_convergence: float = 0.0
    cumulative_divergence: float = 0.0


@dataclass
class BackendState:
    plate_states: list[PlateState]
    previous_plate_states: dict[int, PlateState] = field(default_factory=dict)
    boundary_history: dict[str, BoundaryHistory] = field(default_factory=dict)
    seeds: SeedBundle | None = None
    pygplates: PygplatesModelCache | None = None


@dataclass
class BackendFrameResult:
    frame: TimelineFrame
    diagnostics: FrameDiagnostics
    strain_field: np.ndarray
    coverage_ratio: float
    kinematic_digest: str
    uncertainty_digest: str
    fallback_used: bool


def derive_seed_bundle(seed: int) -> SeedBundle:
    return SeedBundle(
        plates=_seed_from(seed, "plates"),
        boundaries=_seed_from(seed, "boundaries"),
        events=_seed_from(seed, "events"),
        terrain=_seed_from(seed, "terrain"),
    )


def _seed_from(seed: int, label: str) -> int:
    digest = stable_hash({"seed": seed, "label": label})
    return int(digest[:16], 16)


def _wrap_lon(value: float) -> float:
    wrapped = ((value + 180.0) % 360.0) - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _clamp_lat(value: float) -> float:
    return max(-89.0, min(89.0, value))


def _deg_per_myr_from_cm_yr(cm_yr: float) -> float:
    # 1 cm/yr = 10 km/Myr ~= 0.09 degree/Myr near Earth surface.
    return cm_yr * 0.09


def _vector_from_speed_azimuth(speed_cm_yr: float, azimuth_deg: float) -> tuple[float, float]:
    rad = math.radians(azimuth_deg)
    vx = speed_cm_yr * math.cos(rad)
    vy = speed_cm_yr * math.sin(rad)
    return vx, vy


def _distance_deg(a: PlateState, b: PlateState) -> float:
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)
    alat = math.radians(a.lat)
    blat = math.radians(b.lat)

    hav = math.sin(dlat / 2.0) ** 2 + math.cos(alat) * math.cos(blat) * math.sin(dlon / 2.0) ** 2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(hav))))


def _polygon_ring(state: PlateState, time_ma: int, points: int = 18) -> list[list[float]]:
    ring: list[list[float]] = []
    for idx in range(points):
        theta = 2.0 * math.pi * (idx / points)
        anisotropy = 1.0 + 0.12 * math.sin(theta * 3.0 + state.plate_id * 0.41 + time_ma * 0.003)
        radius = state.radius_deg * anisotropy
        nlat = _clamp_lat(state.lat + radius * math.sin(theta))
        lon_scale = max(0.25, math.cos(math.radians(state.lat)))
        nlon = _wrap_lon(state.lon + (radius * math.cos(theta) / lon_scale))
        ring.append([round(nlon, 5), round(nlat, 5)])
    ring.append(ring[0])
    return ring


def _ordered_pairs(plates: list[PlateState], mode: SimulationMode) -> list[tuple[PlateState, PlateState]]:
    ordered = sorted(plates, key=lambda plate: (plate.lon, plate.lat))
    pairs: set[tuple[int, int]] = set()

    def add_pair(left: PlateState, right: PlateState) -> None:
        pid = tuple(sorted((left.plate_id, right.plate_id)))
        pairs.add(pid)

    for idx, left in enumerate(ordered):
        right = ordered[(idx + 1) % len(ordered)]
        add_pair(left, right)

    if mode == SimulationMode.hybrid_rigor:
        for left in ordered:
            nearest = min(
                (plate for plate in ordered if plate.plate_id != left.plate_id),
                key=lambda candidate: _distance_deg(left, candidate),
            )
            add_pair(left, nearest)

    plate_lookup = {plate.plate_id: plate for plate in plates}
    return [(plate_lookup[a], plate_lookup[b]) for a, b in sorted(pairs)]


def _continuity_score(previous: PlateState | None, current: PlateState, max_plate_velocity: float) -> float:
    if previous is None:
        return 1.0

    speed_delta = abs(current.velocity_cm_yr - previous.velocity_cm_yr) / max(1.0, max_plate_velocity)
    heading_delta = abs(((current.azimuth_deg - previous.azimuth_deg + 180.0) % 360.0) - 180.0) / 180.0
    radius_delta = abs(current.radius_deg - previous.radius_deg) / max(1.0, previous.radius_deg)
    score = 1.0 - min(1.0, 0.45 * speed_delta + 0.4 * heading_delta + 0.15 * radius_delta)
    return round(score, 4)


def _event_persistence_class(streak_myr: int) -> str:
    if streak_myr >= 25:
        return "long_lived"
    if streak_myr >= 8:
        return "sustained"
    return "transient"


def _boundary_type_from_velocity(normal_velocity: float, tangential_velocity: float) -> BoundaryType:
    if normal_velocity > 0.85:
        return BoundaryType.divergent
    if normal_velocity < -0.85:
        return BoundaryType.convergent
    if abs(tangential_velocity) > 0.55:
        return BoundaryType.transform
    return BoundaryType.transform


def _mode_uncertainty_base(mode: SimulationMode, rigor: RigorProfile) -> tuple[float, float, float]:
    if mode == SimulationMode.hybrid_rigor:
        kinematic = 0.2
        event = 0.24
        terrain = 0.28
    else:
        kinematic = 0.34
        event = 0.4
        terrain = 0.44

    if rigor == RigorProfile.research:
        kinematic = max(0.08, kinematic - 0.06)
        event = max(0.12, event - 0.05)
        terrain = max(0.14, terrain - 0.04)

    return (kinematic, event, terrain)


def _apply_edit_overrides(
    boundary: BoundarySegment,
    kinematics: BoundaryKinematics,
    time_ma: int,
    edits: list[dict],
) -> tuple[BoundarySegment, BoundaryKinematics]:
    boundary_override = None
    subducting_override = None

    for edit in edits:
        edit_type = edit.get("editType")
        payload = edit.get("payload", {})
        duration = int(payload.get("durationMyr", 20))
        if abs(int(edit.get("timeMa", time_ma)) - time_ma) > duration:
            continue
        target_segment = payload.get("segmentId")
        if target_segment and target_segment != boundary.segmentId:
            continue

        if edit_type == "boundary_override":
            forced_type = payload.get("boundaryType")
            if forced_type in {"convergent", "divergent", "transform"}:
                boundary_override = BoundaryType(forced_type)
        if edit_type == "subducting_side_override":
            forced_side = payload.get("subductingSide")
            if forced_side in {"left", "right", "none"}:
                subducting_override = forced_side

    if boundary_override is not None:
        boundary.boundaryType = boundary_override
        kinematics.recommendedBoundaryType = boundary_override

    if subducting_override is not None:
        boundary.subductingSide = subducting_override  # type: ignore[assignment]

    return boundary, kinematics


class BaseTectonicBackend:
    mode: SimulationMode

    def initialize(
        self,
        config: ProjectConfig,
        seeds: SeedBundle,
        pygplates_cache: PygplatesModelCache,
    ) -> BackendState:
        rng = random.Random(seeds.plates)
        plate_states: list[PlateState] = []
        golden = math.pi * (3.0 - math.sqrt(5.0))

        for plate_idx in range(config.plateCount):
            lat = math.degrees(math.asin(max(-1.0, min(1.0, 1.0 - (2.0 * (plate_idx + 0.5) / config.plateCount)))))
            lon = _wrap_lon(math.degrees(golden * plate_idx) + rng.uniform(-10.0, 10.0))
            radius = rng.uniform(7.5, 15.5)
            velocity = rng.uniform(1.4, min(config.maxPlateVelocityCmYr, 9.5))
            azimuth = rng.uniform(0.0, 360.0)
            craton_factor = min(1.0, max(0.0, 0.35 + rng.random() * 0.65))
            plate_states.append(
                PlateState(
                    plate_id=plate_idx + 1,
                    name=f"Plate {plate_idx + 1}",
                    lat=lat,
                    lon=lon,
                    radius_deg=radius,
                    velocity_cm_yr=velocity,
                    azimuth_deg=azimuth,
                    craton_factor=craton_factor,
                )
            )

        return BackendState(
            plate_states=plate_states,
            previous_plate_states={},
            boundary_history={},
            seeds=seeds,
            pygplates=pygplates_cache,
        )

    def advance(self, state: BackendState, config: ProjectConfig, time_ma: int, step_myr: int, edits: list[dict]) -> None:
        assert state.seeds is not None
        next_states: list[PlateState] = []
        plate_lookup = {plate.plate_id: plate for plate in state.plate_states}

        for plate in state.plate_states:
            local_seed = _seed_from(state.seeds.plates, f"{plate.plate_id}:{time_ma}")
            rng = random.Random(local_seed)

            nearest = min(
                (candidate for candidate in state.plate_states if candidate.plate_id != plate.plate_id),
                key=lambda candidate: _distance_deg(plate, candidate),
            )
            distance = _distance_deg(plate, nearest)

            interaction_bias = 0.0
            if distance < 22.0:
                interaction_bias -= 8.0
            elif distance > 45.0:
                interaction_bias += 4.0

            azimuth_adjust = interaction_bias + rng.uniform(-3.5, 3.5)
            speed_adjust = rng.uniform(-0.3, 0.3)

            for edit in edits:
                edit_type = edit.get("editType")
                payload = edit.get("payload", {})
                duration = int(payload.get("durationMyr", 30))
                if abs(int(edit.get("timeMa", time_ma)) - time_ma) > duration:
                    continue
                target_plate = payload.get("plateId")
                if target_plate not in (None, plate.plate_id):
                    continue

                if edit_type in {"rift_initiation", "rift_start"}:
                    azimuth_adjust += float(payload.get("azimuthDelta", 18.0))
                    speed_adjust += float(payload.get("speedGain", 0.6))

            new_azimuth = (plate.azimuth_deg + azimuth_adjust) % 360.0
            new_speed = max(0.4, min(config.maxPlateVelocityCmYr, plate.velocity_cm_yr + speed_adjust))
            deg_per_myr = _deg_per_myr_from_cm_yr(new_speed) * step_myr

            rad = math.radians(new_azimuth)
            dlat = deg_per_myr * math.sin(rad)
            lon_scale = max(0.2, math.cos(math.radians(plate.lat)))
            dlon = (deg_per_myr * math.cos(rad)) / lon_scale

            radius_adjust = rng.uniform(-0.08, 0.08)

            next_states.append(
                PlateState(
                    plate_id=plate.plate_id,
                    name=plate.name,
                    lat=_clamp_lat(plate.lat + dlat),
                    lon=_wrap_lon(plate.lon + dlon),
                    radius_deg=max(4.0, min(22.0, plate.radius_deg + radius_adjust)),
                    velocity_cm_yr=new_speed,
                    azimuth_deg=new_azimuth,
                    craton_factor=plate.craton_factor,
                )
            )

        state.previous_plate_states = plate_lookup
        state.plate_states = next_states

    def build_frame(
        self,
        *,
        project_id: str,
        config: ProjectConfig,
        state: BackendState,
        time_ma: int,
        edits: list[dict],
    ) -> BackendFrameResult:
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

        for left, right in _ordered_pairs(state.plate_states, self.mode):
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
            inferred_type = _boundary_type_from_velocity(normal_velocity, tangential_velocity)

            seg_id = f"seg_{min(left.plate_id, right.plate_id)}_{max(left.plate_id, right.plate_id)}"

            if inferred_type == BoundaryType.convergent:
                subducting_side = "left" if left.craton_factor < right.craton_factor else "right"
                convergence_by_plate[left.plate_id] += abs(normal_velocity)
                convergence_by_plate[right.plate_id] += abs(normal_velocity)
            elif inferred_type == BoundaryType.divergent:
                subducting_side = "none"
                divergence_by_plate[left.plate_id] += abs(normal_velocity)
                divergence_by_plate[right.plate_id] += abs(normal_velocity)
            else:
                subducting_side = "none"

            boundary = BoundarySegment(
                segmentId=seg_id,
                leftPlateId=left.plate_id,
                rightPlateId=right.plate_id,
                boundaryType=inferred_type,
                digitizationDirection="forward",
                subductingSide=subducting_side,  # type: ignore[arg-type]
                isActive=True,
                geometry={
                    "type": "LineString",
                    "coordinates": [[round(left.lon, 5), round(left.lat, 5)], [round(right.lon, 5), round(right.lat, 5)]],
                },
            )

            kinematics = BoundaryKinematics(
                segmentId=seg_id,
                relativeVelocityCmYr=round(relative_velocity, 4),
                normalVelocityCmYr=round(normal_velocity, 4),
                tangentialVelocityCmYr=round(tangential_velocity, 4),
                strainRate=round(abs(normal_velocity) * 0.02 + abs(tangential_velocity) * 0.008, 5),
                recommendedBoundaryType=inferred_type,
            )

            history = state.boundary_history.get(seg_id)
            if history is None:
                history = BoundaryHistory(boundary_type=inferred_type, streak_myr=config.timeIncrementMyr)
            else:
                if history.boundary_type == inferred_type:
                    history.streak_myr += config.timeIncrementMyr
                else:
                    history.boundary_type = inferred_type
                    history.streak_myr = config.timeIncrementMyr

            if inferred_type == BoundaryType.convergent:
                history.cumulative_convergence += max(0.0, -normal_velocity)
            if inferred_type == BoundaryType.divergent:
                history.cumulative_divergence += max(0.0, normal_velocity)
            state.boundary_history[seg_id] = history

            boundary, kinematics = _apply_edit_overrides(boundary, kinematics, time_ma, edits)

            boundaries.append(boundary)
            boundary_kinematics.append(kinematics)

        for plate in state.plate_states:
            previous = state.previous_plate_states.get(plate.plate_id)
            plate_kinematics.append(
                PlateKinematics(
                    plateId=plate.plate_id,
                    velocityCmYr=round(plate.velocity_cm_yr, 4),
                    azimuthDeg=round(plate.azimuth_deg, 4),
                    convergenceCmYr=round(convergence_by_plate[plate.plate_id], 4),
                    divergenceCmYr=round(divergence_by_plate[plate.plate_id], 4),
                    continuityScore=_continuity_score(previous, plate, config.maxPlateVelocityCmYr),
                )
            )

        events = _synthesize_events(
            boundaries=boundaries,
            boundary_kinematics=boundary_kinematics,
            boundary_history=state.boundary_history,
            time_ma=time_ma,
            config=config,
            edits=edits,
            mode=self.mode,
        )

        coverage_ratio = state.pygplates.coverage_hint if state.pygplates else 0.7
        if self.mode == SimulationMode.hybrid_rigor:
            coverage_ratio = min(0.995, coverage_ratio + 0.015)

        k_u, e_u, t_u = _mode_uncertainty_base(self.mode, config.rigorProfile)
        uncertainty = UncertaintySummary(
            kinematic=round(min(1.0, k_u + (1.0 - coverage_ratio) * 0.35), 4),
            event=round(min(1.0, e_u + (0.02 if len(events) < len(boundaries) * 0.4 else 0.0)), 4),
            terrain=round(min(1.0, t_u + max(0.0, 0.3 - len(events) * 0.01)), 4),
            coverage=round(1.0 - coverage_ratio, 4),
        )

        frame = TimelineFrame(
            timeMa=time_ma,
            plateGeometries=plate_features,
            boundaryGeometries=boundaries,
            eventOverlays=events,
            plateKinematics=plate_kinematics,
            boundaryKinematics=boundary_kinematics,
            strainFieldRef=None,
            uncertaintySummary=uncertainty,
            previewHeightFieldRef="",
        )

        diagnostics = FrameDiagnostics(
            projectId=project_id,
            timeMa=time_ma,
            continuityViolations=[
                f"plate_{kin.plateId}_continuity_low"
                for kin in plate_kinematics
                if kin.continuityScore < 0.22 or kin.velocityCmYr > config.maxPlateVelocityCmYr
            ],
            boundaryConsistencyIssues=[
                f"{boundary.segmentId}_subduction_side_missing"
                for boundary in boundaries
                if boundary.boundaryType == BoundaryType.convergent and boundary.subductingSide == "none"
            ],
            coverageGapRatio=round(1.0 - coverage_ratio, 4),
            warnings=[
                "pygplates_models_not_loaded" if not (state.pygplates and state.pygplates.available) else ""
            ],
            pygplatesStatus=state.pygplates.status if state.pygplates else "unavailable",
        )
        diagnostics.warnings = [warning for warning in diagnostics.warnings if warning]

        strain_field = _build_strain_field(boundaries, boundary_kinematics, width=256, height=128)

        return BackendFrameResult(
            frame=frame,
            diagnostics=diagnostics,
            strain_field=strain_field,
            coverage_ratio=coverage_ratio,
            kinematic_digest=stable_hash([kin.model_dump(mode="json") for kin in plate_kinematics]),
            uncertainty_digest=stable_hash(uncertainty.model_dump(mode="json")),
            fallback_used=not (state.pygplates and state.pygplates.available),
        )


class FastPlausibleBackend(BaseTectonicBackend):
    mode = SimulationMode.fast_plausible


class HybridRigorBackend(BaseTectonicBackend):
    mode = SimulationMode.hybrid_rigor


def build_backend(mode: SimulationMode) -> BaseTectonicBackend:
    if mode == SimulationMode.hybrid_rigor:
        return HybridRigorBackend()
    return FastPlausibleBackend()


def _build_strain_field(
    boundaries: list[BoundarySegment],
    boundary_kinematics: list[BoundaryKinematics],
    width: int,
    height: int,
) -> np.ndarray:
    lon = np.linspace(-180.0, 180.0, width, dtype=np.float32)
    lat = np.linspace(-90.0, 90.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(lon, lat)

    field = np.zeros((height, width), dtype=np.float32)
    kinematics_by_id = {kin.segmentId: kin for kin in boundary_kinematics}

    for boundary in boundaries:
        kin = kinematics_by_id[boundary.segmentId]
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


def _synthesize_events(
    *,
    boundaries: list[BoundarySegment],
    boundary_kinematics: list[BoundaryKinematics],
    boundary_history: dict[str, BoundaryHistory],
    time_ma: int,
    config: ProjectConfig,
    edits: list[dict],
    mode: SimulationMode,
) -> list[GeoEvent]:
    events: list[GeoEvent] = []
    kin_by_id = {kin.segmentId: kin for kin in boundary_kinematics}

    global_event_gain = 1.0
    for edit in edits:
        if edit.get("editType") in {"event_gain", "event_boost"}:
            payload = edit.get("payload", {})
            if abs(int(edit.get("timeMa", time_ma)) - time_ma) <= int(payload.get("durationMyr", 30)):
                global_event_gain += float(payload.get("gain", payload.get("boost", 0.12)))

    for boundary in boundaries:
        history = boundary_history.get(boundary.segmentId)
        if history is None:
            continue
        kin = kin_by_id[boundary.segmentId]
        streak = history.streak_myr
        persistence_class = _event_persistence_class(streak)
        event_confidence = min(0.98, 0.35 + streak * 0.015 + (0.08 if mode == SimulationMode.hybrid_rigor else 0.0))

        if boundary.boundaryType == BoundaryType.convergent:
            intensity = min(1.0, (abs(kin.normalVelocityCmYr) / max(1.0, config.maxPlateVelocityCmYr)) * 1.2)
            if streak >= 5:
                events.append(
                    GeoEvent(
                        eventId=f"evt_orogeny_{boundary.segmentId}_{time_ma}",
                        eventType=GeoEventType.orogeny,
                        timeStartMa=max(0.0, float(time_ma - streak)),
                        timeEndMa=float(time_ma),
                        intensity=round(min(1.0, intensity * global_event_gain), 4),
                        confidence=round(min(1.0, event_confidence), 4),
                        drivingMetrics={
                            "normalVelocityCmYr": abs(kin.normalVelocityCmYr),
                            "streakMyr": float(streak),
                            "strainRate": kin.strainRate,
                        },
                        persistenceClass=persistence_class,  # type: ignore[arg-type]
                        sourceBoundaryIds=[boundary.segmentId],
                        regionGeometry=boundary.geometry,
                    )
                )
            if streak >= 8 and abs(kin.normalVelocityCmYr) > 2.0:
                events.append(
                    GeoEvent(
                        eventId=f"evt_subduction_{boundary.segmentId}_{time_ma}",
                        eventType=GeoEventType.subduction,
                        timeStartMa=max(0.0, float(time_ma - streak)),
                        timeEndMa=float(time_ma),
                        intensity=round(min(1.0, (0.55 + intensity * 0.4) * global_event_gain), 4),
                        confidence=round(min(1.0, event_confidence + 0.08), 4),
                        drivingMetrics={
                            "normalVelocityCmYr": abs(kin.normalVelocityCmYr),
                            "streakMyr": float(streak),
                            "subductingSideLeft": 1.0 if boundary.subductingSide == "left" else 0.0,
                        },
                        persistenceClass=persistence_class,  # type: ignore[arg-type]
                        sourceBoundaryIds=[boundary.segmentId],
                        regionGeometry=boundary.geometry,
                    )
                )
                events.append(
                    GeoEvent(
                        eventId=f"evt_arc_{boundary.segmentId}_{time_ma}",
                        eventType=GeoEventType.arc,
                        timeStartMa=max(0.0, float(time_ma - streak)),
                        timeEndMa=float(time_ma),
                        intensity=round(min(1.0, (0.4 + intensity * 0.3) * global_event_gain), 4),
                        confidence=round(min(1.0, event_confidence + 0.04), 4),
                        drivingMetrics={
                            "normalVelocityCmYr": abs(kin.normalVelocityCmYr),
                            "tangentialVelocityCmYr": abs(kin.tangentialVelocityCmYr),
                        },
                        persistenceClass=persistence_class,  # type: ignore[arg-type]
                        sourceBoundaryIds=[boundary.segmentId],
                        regionGeometry=boundary.geometry,
                    )
                )

        elif boundary.boundaryType == BoundaryType.divergent and streak >= 4:
            intensity = min(1.0, (abs(kin.normalVelocityCmYr) / max(1.0, config.maxPlateVelocityCmYr)) * 1.1)
            events.append(
                GeoEvent(
                    eventId=f"evt_rift_{boundary.segmentId}_{time_ma}",
                    eventType=GeoEventType.rift,
                    timeStartMa=max(0.0, float(time_ma - streak)),
                    timeEndMa=float(time_ma),
                    intensity=round(min(1.0, intensity * global_event_gain), 4),
                    confidence=round(min(1.0, event_confidence), 4),
                    drivingMetrics={
                        "normalVelocityCmYr": abs(kin.normalVelocityCmYr),
                        "streakMyr": float(streak),
                    },
                    persistenceClass=persistence_class,  # type: ignore[arg-type]
                    sourceBoundaryIds=[boundary.segmentId],
                    regionGeometry=boundary.geometry,
                )
            )

        elif boundary.boundaryType == BoundaryType.transform and streak >= 7:
            intensity = min(1.0, abs(kin.tangentialVelocityCmYr) / max(1.0, config.maxPlateVelocityCmYr))
            events.append(
                GeoEvent(
                    eventId=f"evt_terrane_{boundary.segmentId}_{time_ma}",
                    eventType=GeoEventType.terrane,
                    timeStartMa=max(0.0, float(time_ma - streak)),
                    timeEndMa=float(time_ma),
                    intensity=round(min(1.0, intensity * global_event_gain), 4),
                    confidence=round(min(1.0, event_confidence - 0.08), 4),
                    drivingMetrics={
                        "tangentialVelocityCmYr": abs(kin.tangentialVelocityCmYr),
                        "streakMyr": float(streak),
                    },
                    persistenceClass=persistence_class,  # type: ignore[arg-type]
                    sourceBoundaryIds=[boundary.segmentId],
                    regionGeometry=boundary.geometry,
                )
            )

    return sorted(events, key=lambda event: (event.eventType.value, event.eventId))


def frame_coverage_ratio(frame: TimelineFrame) -> float:
    if not frame.plateGeometries:
        return 0.0

    area_proxy = 0.0
    for plate in frame.plateGeometries:
        ring = plate.geometry.get("coordinates", [[]])[0]
        if len(ring) < 3:
            continue
        # Shoelace on lon/lat proxy coordinates.
        total = 0.0
        for idx in range(len(ring) - 1):
            x1, y1 = ring[idx]
            x2, y2 = ring[idx + 1]
            total += (x1 * y2) - (x2 * y1)
        area_proxy += abs(total) * 0.5

    # Approximate normalization against full lon/lat rectangle area proxy.
    normalized = min(1.0, area_proxy / (360.0 * 180.0 * 0.62))
    return round(normalized, 4)


def collect_continuity_violations(frame: TimelineFrame) -> list[str]:
    violations: list[str] = []
    for plate in frame.plateKinematics:
        if plate.continuityScore < 0.2:
            violations.append(f"plate_{plate.plateId}_continuity_low")
    return violations


def collect_boundary_semantic_issues(frame: TimelineFrame) -> list[str]:
    issues: list[str] = []
    for boundary in frame.boundaryGeometries:
        if boundary.boundaryType == BoundaryType.convergent and boundary.subductingSide == "none":
            issues.append(f"{boundary.segmentId}_missing_subducting_side")
        if boundary.leftPlateId == boundary.rightPlateId:
            issues.append(f"{boundary.segmentId}_same_plate_ids")
    return issues


def aggregate_fallback_times(times: Iterable[int], coverages: dict[int, float], threshold: float = 0.82) -> list[int]:
    return sorted([time for time in times if coverages.get(time, 0.0) < threshold], reverse=True)
