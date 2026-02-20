from __future__ import annotations

from ...models import BoundaryStateClass
from .state import BoundaryStateV2


def _infer_candidate_class(
    *,
    boundary: BoundaryStateV2,
    normal_velocity_cm_yr: float,
    tangential_velocity_cm_yr: float,
    average_oceanic_age_myr: float,
    left_plate_is_continental: bool,
    right_plate_is_continental: bool,
) -> BoundaryStateClass:
    shear_ratio = abs(tangential_velocity_cm_yr) / max(0.01, abs(normal_velocity_cm_yr))

    if normal_velocity_cm_yr < -0.7 and left_plate_is_continental and right_plate_is_continental:
        if boundary.collision_streak_myr >= 25:
            return BoundaryStateClass.suture
        return BoundaryStateClass.collision

    if (
        normal_velocity_cm_yr < -1.0
        and average_oceanic_age_myr >= 30.0
        and boundary.convergence_streak_myr >= 5
    ):
        return BoundaryStateClass.subduction

    if normal_velocity_cm_yr > 0.5:
        if boundary.divergence_streak_myr >= 8:
            return BoundaryStateClass.ridge
        return BoundaryStateClass.rift

    if abs(normal_velocity_cm_yr) < 0.5 and shear_ratio > 1.2 and boundary.transform_streak_myr >= 3:
        return BoundaryStateClass.transform

    return BoundaryStateClass.passive_margin


def _required_persistence(
    *,
    current: BoundaryStateClass,
    candidate: BoundaryStateClass,
    boundary: BoundaryStateV2,
) -> bool:
    if current == candidate:
        return True

    if candidate == BoundaryStateClass.ridge:
        return boundary.divergence_streak_myr >= 8

    if candidate == BoundaryStateClass.subduction:
        return boundary.convergence_streak_myr >= 5 and boundary.average_oceanic_age_myr >= 30.0

    if candidate == BoundaryStateClass.transform:
        return boundary.transform_streak_myr >= 3

    if candidate in {BoundaryStateClass.collision, BoundaryStateClass.suture}:
        return boundary.collision_streak_myr >= 2

    return True


def _motion_matches_semantics(state_class: BoundaryStateClass, normal_velocity_cm_yr: float, tangential_velocity_cm_yr: float) -> bool:
    if state_class in {BoundaryStateClass.ridge, BoundaryStateClass.rift}:
        return normal_velocity_cm_yr > 0
    if state_class in {BoundaryStateClass.subduction, BoundaryStateClass.collision, BoundaryStateClass.suture}:
        return normal_velocity_cm_yr < 0
    if state_class == BoundaryStateClass.transform:
        return abs(tangential_velocity_cm_yr) >= abs(normal_velocity_cm_yr)
    return True


def update_boundary_state(
    *,
    boundary: BoundaryStateV2,
    normal_velocity_cm_yr: float,
    tangential_velocity_cm_yr: float,
    relative_velocity_cm_yr: float,
    average_oceanic_age_myr: float,
    left_plate_is_continental: bool,
    right_plate_is_continental: bool,
    step_myr: int,
    time_ma: int,
) -> tuple[BoundaryStateV2, bool]:
    boundary.average_oceanic_age_myr = average_oceanic_age_myr
    boundary.last_normal_velocity_cm_yr = normal_velocity_cm_yr
    boundary.last_tangential_velocity_cm_yr = tangential_velocity_cm_yr
    boundary.last_relative_velocity_cm_yr = relative_velocity_cm_yr

    if normal_velocity_cm_yr > 0.5:
        boundary.divergence_streak_myr += step_myr
    else:
        boundary.divergence_streak_myr = 0

    if normal_velocity_cm_yr < -1.0:
        boundary.convergence_streak_myr += step_myr
    else:
        boundary.convergence_streak_myr = 0

    if abs(normal_velocity_cm_yr) < 0.5 and abs(tangential_velocity_cm_yr) > abs(normal_velocity_cm_yr) * 1.2:
        boundary.transform_streak_myr += step_myr
    else:
        boundary.transform_streak_myr = 0

    if normal_velocity_cm_yr < -0.7 and left_plate_is_continental and right_plate_is_continental:
        boundary.collision_streak_myr += step_myr
    else:
        boundary.collision_streak_myr = 0

    candidate = _infer_candidate_class(
        boundary=boundary,
        normal_velocity_cm_yr=normal_velocity_cm_yr,
        tangential_velocity_cm_yr=tangential_velocity_cm_yr,
        average_oceanic_age_myr=average_oceanic_age_myr,
        left_plate_is_continental=left_plate_is_continental,
        right_plate_is_continental=right_plate_is_continental,
    )

    if _required_persistence(current=boundary.state_class, candidate=candidate, boundary=boundary):
        if candidate != boundary.state_class:
            previous_subducting_side = boundary.subducting_side
            boundary.state_class = candidate
            boundary.last_transition_ma = time_ma
            boundary.type_persistence_myr = step_myr
            boundary.transition_count += 1

            if candidate == BoundaryStateClass.subduction:
                boundary.subducting_side = "left" if tangential_velocity_cm_yr >= 0 else "right"
            elif candidate in {BoundaryStateClass.collision, BoundaryStateClass.suture}:
                boundary.subducting_side = "none"
            elif candidate in {BoundaryStateClass.ridge, BoundaryStateClass.rift, BoundaryStateClass.passive_margin, BoundaryStateClass.transform}:
                boundary.subducting_side = "none"

            if previous_subducting_side != boundary.subducting_side and previous_subducting_side != "none":
                boundary.polarity_flip_count += 1
        else:
            boundary.type_persistence_myr += step_myr
    else:
        boundary.type_persistence_myr += step_myr

    if boundary.state_class == BoundaryStateClass.subduction:
        next_subducting_side = "left" if tangential_velocity_cm_yr >= 0 else "right"
        if boundary.subducting_side not in {"left", "right"}:
            boundary.subducting_side = next_subducting_side
        elif next_subducting_side != boundary.subducting_side:
            boundary.polarity_flip_count += 1
            boundary.subducting_side = next_subducting_side
    else:
        boundary.subducting_side = "none"

    mismatch = not _motion_matches_semantics(boundary.state_class, normal_velocity_cm_yr, tangential_velocity_cm_yr)
    if mismatch:
        boundary.motion_mismatch_count += 1

    return boundary, mismatch
