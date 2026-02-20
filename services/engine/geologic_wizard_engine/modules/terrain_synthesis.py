from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..models import BoundaryKinematics, GeoEvent, PlateFeature, UncertaintySummary


def _normalize(arr: np.ndarray) -> np.ndarray:
    min_v = float(arr.min())
    max_v = float(arr.max())
    if abs(max_v - min_v) < 1e-9:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def _event_center(event: GeoEvent) -> tuple[float, float]:
    geom = event.regionGeometry
    coords = geom.get("coordinates", [])
    if not coords:
        return (0.0, 0.0)
    if geom.get("type") == "LineString":
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
    elif geom.get("type") == "Polygon":
        ring = coords[0] if coords else []
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
    else:
        return (0.0, 0.0)
    if not xs or not ys:
        return (0.0, 0.0)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _kinematic_center(segment_id: str) -> tuple[float, float]:
    # Deterministic pseudo-center used when only kinematic stats are known.
    seed = sum(ord(ch) for ch in segment_id)
    lon = ((seed * 7) % 360) - 180
    lat = ((seed * 13) % 180) - 90
    return float(lon), float(lat)


def build_tectonic_potential_fields(
    width: int,
    height: int,
    events: list[GeoEvent],
    boundary_kinematics: list[BoundaryKinematics],
) -> dict[str, np.ndarray]:
    lon = np.linspace(-math.pi, math.pi, width, dtype=np.float32)
    lat = np.linspace(-math.pi / 2.0, math.pi / 2.0, height, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    uplift = np.zeros((height, width), dtype=np.float32)
    subsidence = np.zeros((height, width), dtype=np.float32)
    volcanic = np.zeros((height, width), dtype=np.float32)
    crust_age = np.ones((height, width), dtype=np.float32) * 0.45

    for event in events:
        elon_deg, elat_deg = _event_center(event)
        elon = math.radians(elon_deg)
        elat = math.radians(elat_deg)
        sigma = 0.09 if event.eventType.value in {"orogeny", "subduction", "arc"} else 0.14
        signal = np.exp(-(((lon_grid - elon) ** 2) + ((lat_grid - elat) ** 2)) / (2.0 * sigma * sigma))

        if event.eventType.value in {"orogeny", "subduction"}:
            uplift += signal * (0.55 * event.intensity * event.confidence)
        elif event.eventType.value == "rift":
            subsidence += signal * (0.46 * event.intensity)
            crust_age -= signal * 0.24
        elif event.eventType.value == "arc":
            volcanic += signal * (0.6 * event.intensity)
            uplift += signal * (0.21 * event.intensity)
        elif event.eventType.value == "terrane":
            uplift += signal * (0.18 * event.intensity)

    for kin in boundary_kinematics:
        blon_deg, blat_deg = _kinematic_center(kin.segmentId)
        blon = math.radians(blon_deg)
        blat = math.radians(blat_deg)
        sigma = 0.15
        influence = np.exp(-(((lon_grid - blon) ** 2) + ((lat_grid - blat) ** 2)) / (2.0 * sigma * sigma))

        if kin.recommendedBoundaryType.value == "convergent":
            uplift += influence * (abs(kin.normalVelocityCmYr) * 0.011)
        elif kin.recommendedBoundaryType.value == "divergent":
            subsidence += influence * (abs(kin.normalVelocityCmYr) * 0.01)
            crust_age -= influence * 0.2

        crust_age += influence * (abs(kin.tangentialVelocityCmYr) * 0.0009)

    return {
        "uplift": _normalize(uplift),
        "subsidence": _normalize(subsidence),
        "volcanic": _normalize(volcanic),
        "crust_age": _normalize(np.clip(crust_age, 0.0, 1.0)),
    }


def synthesize_preview_height(
    time_ma: int,
    seed: int,
    plates: list[PlateFeature],
    events: list[GeoEvent],
    boundary_kinematics: list[BoundaryKinematics],
    uncertainty: UncertaintySummary,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lon = np.linspace(-math.pi, math.pi, width, dtype=np.float32)
    lat = np.linspace(-math.pi / 2.0, math.pi / 2.0, height, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    base = (
        0.44 * np.sin(lon_grid * (2.2 + (seed % 3)))
        + 0.24 * np.cos(lat_grid * (2.4 + (time_ma % 7) * 0.05))
        + 0.18 * np.sin((lon_grid + lat_grid) * (4.4 + (seed % 11) * 0.03))
    )

    for idx, plate in enumerate(plates):
        center = plate.geometry["coordinates"][0][0]
        plon = math.radians(center[0])
        plat = math.radians(center[1])
        sigma = 0.16 + (idx % 6) * 0.02
        gaussian = np.exp(-(((lon_grid - plon) ** 2) + ((lat_grid - plat) ** 2)) / (2.0 * sigma * sigma))
        base += gaussian * (0.03 + (idx % 5) * 0.017)

    fields = build_tectonic_potential_fields(width, height, events, boundary_kinematics)

    relief = base
    relief += fields["uplift"] * 0.62
    relief += fields["volcanic"] * 0.28
    relief -= fields["subsidence"] * 0.56
    relief += fields["crust_age"] * 0.13

    # Increase smoothing when uncertainty is high so terrain avoids noisy artifacts.
    uncertainty_weight = min(1.0, max(0.0, uncertainty.terrain))
    smooth_component = (
        0.5 * np.sin(lon_grid * 1.1)
        + 0.45 * np.cos(lat_grid * 1.25)
        + 0.25 * np.sin((lon_grid - lat_grid) * 1.5)
    )
    relief = relief * (1.0 - 0.25 * uncertainty_weight) + smooth_component * (0.25 * uncertainty_weight)

    return _normalize(relief).astype(np.float32), fields


def synthesize_refined_region(
    preview: np.ndarray,
    region: dict[str, Any] | None,
    width: int,
    height: int,
    refinement_level: int,
    seed: int,
    strain_field: np.ndarray | None = None,
) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)
    x = np.linspace(0, 1, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    expanded = np.kron(preview, np.ones((max(1, height // preview.shape[0]), max(1, width // preview.shape[1]))))
    expanded = expanded[:height, :width]

    detail = np.zeros_like(expanded)
    for octave in range(1, 4 + refinement_level):
        freq = 2 ** octave
        amp = 1.0 / (octave ** 1.45)
        detail += amp * np.sin((xx * freq + (seed % 19) * 0.01) * np.pi * 2.0)
        detail += amp * np.cos((yy * freq + (seed % 23) * 0.01) * np.pi * 2.0)

    if strain_field is not None:
        strain_up = np.kron(
            strain_field,
            np.ones((max(1, height // strain_field.shape[0]), max(1, width // strain_field.shape[1]))),
        )[:height, :width]
        detail += strain_up * 0.8

    if region and region.get("type") == "Polygon":
        coords = region.get("coordinates", [[]])[0]
        if coords:
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            cx_n = (cx + 180.0) / 360.0
            cy_n = (cy + 90.0) / 180.0
            mask = np.exp(-(((xx - cx_n) ** 2) + ((yy - cy_n) ** 2)) / 0.03)
            detail *= 0.58 + mask

    refined = expanded + 0.08 * detail
    return _normalize(refined).astype(np.float32)


def synthesize_preview_height_v2(
    *,
    previous_height: np.ndarray,
    uplift_rate: np.ndarray,
    subsidence_rate: np.ndarray,
    volcanic_flux: np.ndarray,
    erosion_capacity: np.ndarray,
    orogenic_root: np.ndarray,
    step_myr: int,
) -> np.ndarray:
    # Process-coupled terrain evolution: H(t+dt) = H(t) + uplift - subsidence - erosion(H, slope).
    slope_y, slope_x = np.gradient(previous_height)
    slope = np.sqrt((slope_x * slope_x) + (slope_y * slope_y))
    erosion = np.clip(erosion_capacity * slope * (0.2 + 0.015 * step_myr), 0.0, 0.08)
    uplift = np.clip(uplift_rate + orogenic_root * 0.1 + volcanic_flux * 0.22, 0.0, 0.2)
    subsidence = np.clip(subsidence_rate * (1.05 + 0.01 * step_myr), 0.0, 0.18)
    updated = np.clip(previous_height + uplift - subsidence - erosion, 0.0, 1.0)
    return updated.astype(np.float32)
