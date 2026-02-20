from __future__ import annotations

import math
from typing import Iterable

from .state import BoundaryStateV2, PlateStateV2


def _wrap_angle(value: float) -> float:
    return value % 360.0


def _direction_to_target(from_lon: float, from_lat: float, to_lon: float, to_lat: float) -> float:
    dlon = to_lon - from_lon
    dlat = to_lat - from_lat
    angle = math.degrees(math.atan2(dlat, dlon))
    return _wrap_angle(angle)


def _angle_delta(current_deg: float, target_deg: float) -> float:
    raw = ((target_deg - current_deg + 180.0) % 360.0) - 180.0
    return raw


def solve_plate_kinematics(
    *,
    plate: PlateStateV2,
    boundaries: Iterable[BoundaryStateV2],
    neighbor_mean_distance_deg: float,
    max_velocity_cm_yr: float,
    supercontinent_bias_strength: float,
    supercontinent_phase: str,
    supercontinent_center: tuple[float, float] | None,
    seed: int,
    time_ma: int,
) -> tuple[float, float]:
    boundaries = list(boundaries)
    subduction_flux = sum(boundary.subduction_flux for boundary in boundaries if boundary.state_class.value == "subduction")
    ridge_count = sum(1 for boundary in boundaries if boundary.state_class.value in {"ridge", "rift"})
    collision_count = sum(1 for boundary in boundaries if boundary.state_class.value in {"collision", "suture"})
    transform_count = sum(1 for boundary in boundaries if boundary.state_class.value == "transform")

    slab_pull = min(1.8, subduction_flux * 0.01)
    ridge_push = min(1.1, ridge_count * 0.18)
    collisional_drag = min(2.2, collision_count * 0.24)
    transform_drag = min(0.8, transform_count * 0.08)

    spacing_bias = 0.0
    if neighbor_mean_distance_deg < 24.0:
        spacing_bias = -0.55
    elif neighbor_mean_distance_deg > 50.0:
        spacing_bias = 0.35

    basal_phase = (seed % 1543) * 0.013 + plate.plate_id * 0.29 + time_ma * 0.008
    basal_traction = 0.3 * math.sin(basal_phase) + 0.16 * math.cos(basal_phase * 1.7)

    speed_delta = slab_pull + ridge_push + spacing_bias + basal_traction - collisional_drag - transform_drag

    steering_delta = 0.0
    if supercontinent_center is not None:
        center_lon, center_lat = supercontinent_center
        target_azimuth = _direction_to_target(plate.lon, plate.lat, center_lon, center_lat)
        toward_center = _angle_delta(plate.azimuth_deg, target_azimuth)

        if supercontinent_phase in {"assembly", "stable"}:
            steering_delta += toward_center * 0.06 * supercontinent_bias_strength
        elif supercontinent_phase in {"dispersal", "assembled"}:
            steering_delta -= toward_center * 0.06 * supercontinent_bias_strength

    steering_delta += basal_traction * 9.0

    new_speed = plate.velocity_cm_yr + speed_delta
    new_speed = max(0.25, min(max_velocity_cm_yr, new_speed))

    new_azimuth = _wrap_angle(plate.azimuth_deg + steering_delta)
    return new_speed, new_azimuth
