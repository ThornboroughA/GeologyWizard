from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...models import BoundaryStateClass, ProjectConfig


@dataclass
class PlateStateV2:
    plate_id: int
    name: str
    lat: float
    lon: float
    radius_deg: float
    velocity_cm_yr: float
    azimuth_deg: float
    craton_factor: float
    is_continental: bool


@dataclass
class BoundaryStateV2:
    segment_id: str
    left_plate_id: int
    right_plate_id: int
    state_class: BoundaryStateClass = BoundaryStateClass.passive_margin
    subducting_side: str = "none"
    last_transition_ma: int = 0
    type_persistence_myr: int = 0
    polarity_flip_count: int = 0
    transition_count: int = 0
    divergence_streak_myr: int = 0
    convergence_streak_myr: int = 0
    transform_streak_myr: int = 0
    collision_streak_myr: int = 0
    motion_mismatch_count: int = 0
    subduction_flux: float = 0.0
    average_oceanic_age_myr: float = 0.0
    last_normal_velocity_cm_yr: float = 0.0
    last_tangential_velocity_cm_yr: float = 0.0
    last_relative_velocity_cm_yr: float = 0.0


@dataclass
class OceanicGridState:
    crust_type: np.ndarray
    oceanic_age_myr: np.ndarray
    crust_thickness_km: np.ndarray
    tectonic_potential: np.ndarray
    craton_id: np.ndarray
    terrain_height: np.ndarray
    uplift_rate: np.ndarray
    subsidence_rate: np.ndarray
    volcanic_flux: np.ndarray
    erosion_capacity: np.ndarray
    orogenic_root: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.oceanic_age_myr.shape


@dataclass
class EventState:
    event_key: str
    event_type: str
    segment_id: str
    time_start_ma: int
    last_active_time_ma: int
    active_duration_myr: int
    min_persistence_myr: int
    decay_remaining_myr: int = 0
    intensity: float = 0.0
    confidence: float = 0.0
    phase: str = "initiation"
    source_boundary_ids: list[str] = field(default_factory=list)
    region_geometry: dict[str, Any] = field(default_factory=dict)


@dataclass
class SupercontinentState:
    phase: str = "stable"
    cycle_count: int = 0
    largest_cluster_fraction: float = 0.0
    centroid: tuple[float, float] | None = None
    history: list[tuple[int, float]] = field(default_factory=list)
    last_cycle_transition_ma: int | None = None


@dataclass
class RunStateV2:
    config: ProjectConfig
    plate_states: list[PlateStateV2]
    previous_plate_states: dict[int, PlateStateV2] = field(default_factory=dict)
    boundaries: dict[str, BoundaryStateV2] = field(default_factory=dict)
    oceanic: OceanicGridState | None = None
    events: dict[str, EventState] = field(default_factory=dict)
    supercontinent: SupercontinentState = field(default_factory=SupercontinentState)
    seeds: Any | None = None
    pygplates: Any | None = None
    previous_frame_hash: str | None = None
    last_step_snapshots: list[dict[str, Any]] = field(default_factory=list)
    last_replay_hash: str = ""
