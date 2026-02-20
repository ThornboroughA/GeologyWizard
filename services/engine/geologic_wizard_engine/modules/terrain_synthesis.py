from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..models import GeoEvent, PlateFeature


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
    elif geom.get("type") == "Polygon" and coords:
        ring = coords[0]
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
    else:
        return (0.0, 0.0)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def synthesize_preview_height(
    time_ma: int,
    seed: int,
    plates: list[PlateFeature],
    events: list[GeoEvent],
    width: int,
    height: int,
) -> np.ndarray:
    lon = np.linspace(-math.pi, math.pi, width, dtype=np.float32)
    lat = np.linspace(-math.pi / 2.0, math.pi / 2.0, height, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    base = (
        0.52 * np.sin(lon_grid * (2.0 + (seed % 3)))
        + 0.35 * np.cos(lat_grid * (3.0 + (time_ma % 5) * 0.08))
        + 0.18 * np.sin((lon_grid + lat_grid) * (5.0 + (seed % 7) * 0.13))
    )

    for idx, plate in enumerate(plates):
        center = plate.geometry["coordinates"][0][0]
        plon = math.radians(center[0])
        plat = math.radians(center[1])
        sigma = 0.14 + (idx % 5) * 0.02
        gaussian = np.exp(-(((lon_grid - plon) ** 2) + ((lat_grid - plat) ** 2)) / (2.0 * sigma * sigma))
        base += gaussian * (0.05 + (idx % 3) * 0.02)

    for event in events:
        elon_deg, elat_deg = _event_center(event)
        elon = math.radians(elon_deg)
        elat = math.radians(elat_deg)
        sigma = 0.08 if event.eventType.value in ("orogeny", "subduction") else 0.12
        signal = np.exp(-(((lon_grid - elon) ** 2) + ((lat_grid - elat) ** 2)) / (2.0 * sigma * sigma))
        if event.eventType.value == "orogeny":
            base += signal * (0.35 * event.intensity)
        elif event.eventType.value == "rift":
            base -= signal * (0.25 * event.intensity)
        elif event.eventType.value == "terrane":
            base += signal * (0.12 * event.intensity)

    return _normalize(base).astype(np.float32)


def synthesize_refined_region(
    preview: np.ndarray,
    region: dict[str, Any] | None,
    width: int,
    height: int,
    refinement_level: int,
    seed: int,
) -> np.ndarray:
    # Start from a smoothly upsampled global field, then add deterministic high-frequency detail.
    expanded = np.kron(preview, np.ones((max(1, height // preview.shape[0]), max(1, width // preview.shape[1]))))
    expanded = expanded[:height, :width]

    y = np.linspace(0, 1, height, dtype=np.float32)
    x = np.linspace(0, 1, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Deterministic fractal-ish detail without extra dependencies.
    detail = np.zeros_like(expanded)
    for octave in range(1, 4 + refinement_level):
        freq = 2 ** octave
        amp = 1.0 / (octave ** 1.5)
        detail += amp * np.sin((xx * freq + (seed % 19) * 0.01) * np.pi * 2.0)
        detail += amp * np.cos((yy * freq + (seed % 23) * 0.01) * np.pi * 2.0)

    if region and region.get("type") == "Polygon":
        # Approximate regional emphasis by center-weighting when polygon exists.
        coords = region.get("coordinates", [[]])[0]
        if coords:
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            cx_n = (cx + 180.0) / 360.0
            cy_n = (cy + 90.0) / 180.0
            mask = np.exp(-(((xx - cx_n) ** 2) + ((yy - cy_n) ** 2)) / 0.03)
            detail *= 0.6 + mask

    refined = expanded + 0.08 * detail
    return _normalize(refined).astype(np.float32)
