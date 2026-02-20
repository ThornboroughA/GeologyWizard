from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .state import BoundaryStateV2, OceanicGridState, PlateStateV2

OCEANIC = np.uint8(0)
CONTINENTAL = np.uint8(1)


@dataclass
class LifecycleUpdate:
    unexplained_plate_births: int = 0
    unexplained_plate_deaths: int = 0
    net_area_balance_error: float = 0.0
    continental_area_fraction: float = 0.0
    oceanic_area_fraction: float = 0.0
    oceanic_age_p99_myr: float = 0.0
    subduction_flux_total: float = 0.0


@lru_cache(maxsize=32)
def _gaussian_kernel(radius_cells: int) -> np.ndarray:
    radius = max(1, radius_cells)
    ys, xs = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    sigma = max(1.0, radius * 0.55)
    kernel = np.exp(-((xs * xs) + (ys * ys)) / (2.0 * sigma * sigma)).astype(np.float32)
    return kernel / float(kernel.max())


def _wrap_index_x(x: int, width: int) -> int:
    return x % width


def _clamp_index_y(y: int, height: int) -> int:
    return max(0, min(height - 1, y))


def _center_from_lon_lat(lon: float, lat: float, width: int, height: int) -> tuple[int, int]:
    x = int(((lon + 180.0) / 360.0) * (width - 1))
    y = int(((lat + 90.0) / 180.0) * (height - 1))
    return x, y


def _apply_local(
    field: np.ndarray,
    *,
    center_x: int,
    center_y: int,
    kernel: np.ndarray,
    multiplier: float,
    op: str,
) -> None:
    height, width = field.shape
    radius = kernel.shape[0] // 2

    for ky in range(kernel.shape[0]):
        y = _clamp_index_y(center_y + (ky - radius), height)
        for kx in range(kernel.shape[1]):
            x = _wrap_index_x(center_x + (kx - radius), width)
            influence = float(kernel[ky, kx]) * multiplier
            if op == "add":
                field[y, x] += influence
            elif op == "set_min":
                field[y, x] = min(field[y, x], influence)
            elif op == "set_max":
                field[y, x] = max(field[y, x], influence)
            elif op == "blend_to_zero":
                field[y, x] = field[y, x] * max(0.0, 1.0 - influence)


def _apply_local_crust_type(
    crust_type: np.ndarray,
    *,
    center_x: int,
    center_y: int,
    kernel: np.ndarray,
    threshold: float,
    value: np.uint8,
) -> None:
    height, width = crust_type.shape
    radius = kernel.shape[0] // 2

    for ky in range(kernel.shape[0]):
        y = _clamp_index_y(center_y + (ky - radius), height)
        for kx in range(kernel.shape[1]):
            x = _wrap_index_x(center_x + (kx - radius), width)
            if float(kernel[ky, kx]) >= threshold:
                crust_type[y, x] = value


def initialize_oceanic_grid(
    *,
    width: int,
    height: int,
    seed: int,
    plate_states: list[PlateStateV2],
) -> OceanicGridState:
    lon = np.linspace(-math.pi, math.pi, width, dtype=np.float32)
    lat = np.linspace(-math.pi / 2.0, math.pi / 2.0, height, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    crust_type = np.zeros((height, width), dtype=np.uint8)
    oceanic_age = np.clip(
        55.0
        + 35.0 * np.sin(lon_grid * (1.7 + (seed % 7) * 0.03))
        + 22.0 * np.cos(lat_grid * (2.1 + (seed % 5) * 0.04)),
        0.0,
        200.0,
    ).astype(np.float32)

    continental_signal = np.zeros((height, width), dtype=np.float32)
    craton_id = np.zeros((height, width), dtype=np.int32)
    for idx, plate in enumerate([p for p in plate_states if p.is_continental]):
        plon = math.radians(plate.lon)
        plat = math.radians(plate.lat)
        sigma = math.radians(max(8.0, plate.radius_deg * 0.75 + (idx % 3) * 1.5))
        signal = np.exp(-(((lon_grid - plon) ** 2) + ((lat_grid - plat) ** 2)) / (2.0 * sigma * sigma))
        continental_signal += signal.astype(np.float32)

        # Persistent craton seeds remain visible and can be reused by later inheritance logic.
        craton_sigma = math.radians(max(2.0, plate.radius_deg * max(0.16, plate.craton_factor * 0.32)))
        craton_signal = np.exp(-(((lon_grid - plon) ** 2) + ((lat_grid - plat) ** 2)) / (2.0 * craton_sigma * craton_sigma))
        craton_id[craton_signal >= 0.64] = int(plate.plate_id)

    crust_type[continental_signal >= 0.32] = CONTINENTAL
    oceanic_age[crust_type == CONTINENTAL] = 0.0

    crust_thickness = np.where(
        crust_type == CONTINENTAL,
        34.0 + continental_signal * 12.0,
        6.5 + oceanic_age * 0.015,
    ).astype(np.float32)

    terrain_height = np.where(
        crust_type == CONTINENTAL,
        0.52 + np.clip(continental_signal * 0.25, 0.0, 0.35),
        0.35 - np.clip(oceanic_age * 0.0012, 0.0, 0.2),
    ).astype(np.float32)
    terrain_height = np.clip(terrain_height, 0.0, 1.0)

    uplift_rate = np.zeros((height, width), dtype=np.float32)
    subsidence_rate = np.zeros((height, width), dtype=np.float32)
    volcanic_flux = np.zeros((height, width), dtype=np.float32)
    erosion_capacity = np.full((height, width), 0.04, dtype=np.float32)
    orogenic_root = np.zeros((height, width), dtype=np.float32)

    tectonic_potential = np.zeros((height, width), dtype=np.float32)

    return OceanicGridState(
        crust_type=crust_type,
        oceanic_age_myr=oceanic_age,
        crust_thickness_km=crust_thickness,
        tectonic_potential=tectonic_potential,
        craton_id=craton_id,
        terrain_height=terrain_height,
        uplift_rate=uplift_rate,
        subsidence_rate=subsidence_rate,
        volcanic_flux=volcanic_flux,
        erosion_capacity=erosion_capacity,
        orogenic_root=orogenic_root,
    )


def _mean_oceanic_age_around_boundary(grid: OceanicGridState, lon: float, lat: float) -> float:
    height, width = grid.shape
    cx, cy = _center_from_lon_lat(lon, lat, width, height)
    kernel = _gaussian_kernel(5)
    radius = kernel.shape[0] // 2

    total_weight = 0.0
    total_age = 0.0
    for ky in range(kernel.shape[0]):
        y = _clamp_index_y(cy + (ky - radius), height)
        for kx in range(kernel.shape[1]):
            x = _wrap_index_x(cx + (kx - radius), width)
            if grid.crust_type[y, x] != OCEANIC:
                continue
            weight = float(kernel[ky, kx])
            total_weight += weight
            total_age += float(grid.oceanic_age_myr[y, x]) * weight

    if total_weight <= 1e-6:
        return 0.0
    return total_age / total_weight


def boundary_average_oceanic_age(grid: OceanicGridState, coords: list[list[float]]) -> float:
    if len(coords) < 2:
        return 0.0
    lon = (float(coords[0][0]) + float(coords[1][0])) * 0.5
    lat = (float(coords[0][1]) + float(coords[1][1])) * 0.5
    return round(_mean_oceanic_age_around_boundary(grid, lon, lat), 3)


def _decay_rates(grid: OceanicGridState) -> None:
    grid.uplift_rate *= 0.86
    grid.subsidence_rate *= 0.86
    grid.volcanic_flux *= 0.9
    grid.orogenic_root *= 0.985


def update_oceanic_grid(
    *,
    grid: OceanicGridState,
    boundary_states: list[BoundaryStateV2],
    boundary_midpoints: dict[str, tuple[float, float]],
    step_myr: int,
    seed: int,
) -> LifecycleUpdate:
    _decay_rates(grid)

    oceanic_mask = grid.crust_type == OCEANIC
    grid.oceanic_age_myr[oceanic_mask] = np.clip(grid.oceanic_age_myr[oceanic_mask] + step_myr, 0.0, 500.0)
    grid.oceanic_age_myr[~oceanic_mask] = 0.0

    subduction_flux_total = 0.0

    for boundary in sorted(boundary_states, key=lambda item: item.segment_id):
        midpoint = boundary_midpoints.get(boundary.segment_id)
        if midpoint is None:
            continue
        lon, lat = midpoint
        height, width = grid.shape
        cx, cy = _center_from_lon_lat(lon, lat, width, height)

        kernel = _gaussian_kernel(7)
        minor_kernel = _gaussian_kernel(4)

        if boundary.state_class.value in {"ridge", "rift"}:
            _apply_local(grid.oceanic_age_myr, center_x=cx, center_y=cy, kernel=kernel, multiplier=0.95, op="blend_to_zero")
            _apply_local(grid.subsidence_rate, center_x=cx, center_y=cy, kernel=kernel, multiplier=0.05, op="add")
            _apply_local(grid.volcanic_flux, center_x=cx, center_y=cy, kernel=minor_kernel, multiplier=0.04, op="add")
            if boundary.state_class.value == "ridge":
                _apply_local_crust_type(grid.crust_type, center_x=cx, center_y=cy, kernel=kernel, threshold=0.35, value=OCEANIC)

        if boundary.state_class.value == "subduction":
            flux = max(0.0, abs(boundary.last_normal_velocity_cm_yr) * 0.8 + boundary.average_oceanic_age_myr * 0.03)
            boundary.subduction_flux = round(flux, 4)
            subduction_flux_total += flux
            _apply_local(grid.uplift_rate, center_x=cx, center_y=cy, kernel=kernel, multiplier=0.085, op="add")
            _apply_local(grid.orogenic_root, center_x=cx, center_y=cy, kernel=minor_kernel, multiplier=0.03, op="add")
            _apply_local(grid.volcanic_flux, center_x=cx, center_y=cy, kernel=minor_kernel, multiplier=0.06, op="add")
            _apply_local(grid.oceanic_age_myr, center_x=cx, center_y=cy, kernel=kernel, multiplier=0.82, op="blend_to_zero")

            terrane_selector = ((seed + int(abs(lon) * 13.0) + int(abs(lat) * 7.0)) % 5) == 0
            if terrane_selector:
                _apply_local_crust_type(grid.crust_type, center_x=cx, center_y=cy, kernel=minor_kernel, threshold=0.68, value=CONTINENTAL)

        if boundary.state_class.value in {"collision", "suture"}:
            _apply_local(grid.uplift_rate, center_x=cx, center_y=cy, kernel=kernel, multiplier=0.11, op="add")
            _apply_local(grid.orogenic_root, center_x=cx, center_y=cy, kernel=kernel, multiplier=0.07, op="add")
            _apply_local_crust_type(grid.crust_type, center_x=cx, center_y=cy, kernel=kernel, threshold=0.52, value=CONTINENTAL)

        if boundary.state_class.value == "transform":
            _apply_local(grid.erosion_capacity, center_x=cx, center_y=cy, kernel=minor_kernel, multiplier=0.0025, op="add")

        if boundary.state_class.value == "passive_margin":
            _apply_local(grid.subsidence_rate, center_x=cx, center_y=cy, kernel=minor_kernel, multiplier=0.025, op="add")

    continental_mask = grid.crust_type == CONTINENTAL
    oceanic_mask = ~continental_mask

    grid.crust_thickness_km[oceanic_mask] = np.clip(6.5 + grid.oceanic_age_myr[oceanic_mask] * 0.015, 6.0, 12.5)
    grid.crust_thickness_km[continental_mask] = np.clip(
        33.0 + grid.orogenic_root[continental_mask] * 35.0,
        30.0,
        75.0,
    )

    slope_y, slope_x = np.gradient(grid.terrain_height)
    slope_mag = np.sqrt((slope_x * slope_x) + (slope_y * slope_y)).astype(np.float32)

    erosion = np.clip(grid.erosion_capacity * slope_mag * 0.22, 0.0, 0.05)
    uplift = np.clip(grid.uplift_rate + grid.orogenic_root * 0.12 + grid.volcanic_flux * 0.2, 0.0, 0.16)
    subsidence = np.clip(grid.subsidence_rate * 1.12, 0.0, 0.14)

    grid.terrain_height = np.clip(grid.terrain_height + uplift - subsidence - erosion, 0.0, 1.0)

    tectonic_raw = uplift - subsidence + grid.volcanic_flux * 0.25
    tmin = float(np.min(tectonic_raw))
    tmax = float(np.max(tectonic_raw))
    if abs(tmax - tmin) < 1e-9:
        grid.tectonic_potential = np.zeros_like(tectonic_raw)
    else:
        grid.tectonic_potential = ((tectonic_raw - tmin) / (tmax - tmin)).astype(np.float32)

    continental_fraction = float(np.mean(continental_mask.astype(np.float32)))
    oceanic_fraction = 1.0 - continental_fraction
    area_error = abs((continental_fraction + oceanic_fraction) - 1.0)

    oceanic_ages = grid.oceanic_age_myr[oceanic_mask]
    oceanic_age_p99 = float(np.percentile(oceanic_ages, 99)) if oceanic_ages.size > 0 else 0.0

    return LifecycleUpdate(
        unexplained_plate_births=0,
        unexplained_plate_deaths=0,
        net_area_balance_error=round(area_error, 6),
        continental_area_fraction=round(continental_fraction, 4),
        oceanic_area_fraction=round(oceanic_fraction, 4),
        oceanic_age_p99_myr=round(oceanic_age_p99, 4),
        subduction_flux_total=round(subduction_flux_total, 4),
    )
