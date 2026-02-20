from __future__ import annotations

from dataclasses import dataclass

from ...models import GeoEvent, GeoEventType
from .state import BoundaryStateV2, EventState


_MIN_PERSISTENCE = {
    GeoEventType.rift.value: 8,
    GeoEventType.arc.value: 8,
    GeoEventType.subduction.value: 8,
    GeoEventType.orogeny.value: 15,
    GeoEventType.terrane.value: 8,
}


@dataclass
class EventLedgerResult:
    events: list[GeoEvent]
    short_lived_orogeny_count: int
    uncoupled_volcanic_belts: int


def _event_candidates(boundary: BoundaryStateV2) -> list[str]:
    state_class = boundary.state_class.value
    if state_class in {"ridge", "rift"}:
        return [GeoEventType.rift.value]
    if state_class == "subduction":
        return [GeoEventType.subduction.value, GeoEventType.arc.value]
    if state_class in {"collision", "suture"}:
        return [GeoEventType.orogeny.value]
    if state_class == "transform" and boundary.type_persistence_myr >= 7:
        return [GeoEventType.terrane.value]
    return []


def _event_intensity(event_type: str, boundary: BoundaryStateV2) -> float:
    rel_v = abs(boundary.last_relative_velocity_cm_yr)
    normal = abs(boundary.last_normal_velocity_cm_yr)
    tangential = abs(boundary.last_tangential_velocity_cm_yr)

    if event_type == GeoEventType.rift.value:
        return min(1.0, 0.25 + normal * 0.08)
    if event_type == GeoEventType.subduction.value:
        return min(1.0, 0.3 + normal * 0.09 + boundary.subduction_flux * 0.01)
    if event_type == GeoEventType.arc.value:
        return min(1.0, 0.2 + normal * 0.05 + boundary.subduction_flux * 0.008)
    if event_type == GeoEventType.orogeny.value:
        return min(1.0, 0.35 + normal * 0.07)
    if event_type == GeoEventType.terrane.value:
        return min(1.0, 0.2 + tangential * 0.08 + rel_v * 0.03)
    return min(1.0, rel_v * 0.08)


def _event_confidence(duration_myr: int, min_persistence_myr: int, state_class: str) -> float:
    base = 0.4 + min(0.5, duration_myr * 0.01)
    if state_class in {"subduction", "collision", "suture"}:
        base += 0.08
    if duration_myr < min_persistence_myr:
        base *= 0.8
    return min(1.0, base)


def _build_geo_event(event: EventState, now_ma: int, boundary: BoundaryStateV2) -> GeoEvent:
    event_type = GeoEventType(event.event_type)
    return GeoEvent(
        eventId=f"evt_{event.event_type}_{event.segment_id}_{now_ma}",
        eventType=event_type,
        timeStartMa=float(max(0, event.time_start_ma)),
        timeEndMa=float(now_ma),
        intensity=round(event.intensity, 4),
        confidence=round(event.confidence, 4),
        drivingMetrics={
            "durationMyr": float(event.active_duration_myr),
            "normalVelocityCmYr": abs(boundary.last_normal_velocity_cm_yr),
            "tangentialVelocityCmYr": abs(boundary.last_tangential_velocity_cm_yr),
            "subductionFlux": boundary.subduction_flux,
            "phaseCode": 0.0 if event.phase == "initiation" else (1.0 if event.phase == "active" else 2.0),
        },
        persistenceClass=(
            "long_lived"
            if event.active_duration_myr >= 25
            else ("sustained" if event.active_duration_myr >= 8 else "transient")
        ),
        sourceBoundaryIds=event.source_boundary_ids,
        regionGeometry=event.region_geometry,
    )


def update_event_ledger(
    *,
    ledger: dict[str, EventState],
    boundaries: list[BoundaryStateV2],
    boundary_geometries: dict[str, dict],
    time_ma: int,
    step_myr: int,
) -> EventLedgerResult:
    active_keys: set[str] = set()
    short_lived_orogeny_count = 0

    boundaries_by_id = {boundary.segment_id: boundary for boundary in boundaries}

    for boundary in sorted(boundaries, key=lambda item: item.segment_id):
        candidates = _event_candidates(boundary)
        for event_type in candidates:
            key = f"{event_type}:{boundary.segment_id}"
            min_persistence = _MIN_PERSISTENCE[event_type]
            state = ledger.get(key)
            if state is None:
                state = EventState(
                    event_key=key,
                    event_type=event_type,
                    segment_id=boundary.segment_id,
                    time_start_ma=time_ma,
                    last_active_time_ma=time_ma,
                    active_duration_myr=step_myr,
                    min_persistence_myr=min_persistence,
                )
                ledger[key] = state
            else:
                state.last_active_time_ma = time_ma
                state.active_duration_myr += step_myr

            state.intensity = _event_intensity(event_type, boundary)
            state.confidence = _event_confidence(state.active_duration_myr, state.min_persistence_myr, boundary.state_class.value)
            state.source_boundary_ids = [boundary.segment_id]
            state.region_geometry = boundary_geometries.get(
                boundary.segment_id,
                {
                    "type": "LineString",
                    "coordinates": [[0.0, 0.0], [0.0, 0.0]],
                },
            )
            state.phase = "active" if state.active_duration_myr >= state.min_persistence_myr else "initiation"
            active_keys.add(key)

    for key in list(ledger.keys()):
        if key in active_keys:
            continue

        state = ledger[key]
        if state.event_type == GeoEventType.orogeny.value and state.decay_remaining_myr < 30:
            state.decay_remaining_myr += step_myr
            state.phase = "decay"
            state.confidence *= 0.98
            state.intensity *= 0.985
            active_keys.add(key)
            continue

        if state.event_type == GeoEventType.orogeny.value and state.active_duration_myr < 10:
            short_lived_orogeny_count += 1
        del ledger[key]

    output_events: list[GeoEvent] = []
    uncoupled_volcanic_belts = 0

    for key in sorted(active_keys):
        state = ledger.get(key)
        if state is None:
            continue

        boundary = boundaries_by_id.get(state.segment_id)
        if boundary is None:
            continue

        if state.phase != "decay" and state.active_duration_myr < state.min_persistence_myr:
            continue

        event = _build_geo_event(state, time_ma, boundary)
        output_events.append(event)

        if state.event_type == GeoEventType.arc.value and boundary.state_class.value != "subduction":
            uncoupled_volcanic_belts += 1

    return EventLedgerResult(
        events=output_events,
        short_lived_orogeny_count=short_lived_orogeny_count,
        uncoupled_volcanic_belts=uncoupled_volcanic_belts,
    )
