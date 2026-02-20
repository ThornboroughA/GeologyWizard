from __future__ import annotations

import math

from .state import PlateStateV2, SupercontinentState


def _distance_deg(a: PlateStateV2, b: PlateStateV2) -> float:
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)
    alat = math.radians(a.lat)
    blat = math.radians(b.lat)
    hav = math.sin(dlat / 2.0) ** 2 + math.cos(alat) * math.cos(blat) * math.sin(dlon / 2.0) ** 2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(hav))))


def _build_clusters(plates: list[PlateStateV2], threshold_deg: float = 35.0) -> list[list[PlateStateV2]]:
    clusters: list[list[PlateStateV2]] = []
    visited: set[int] = set()

    lookup = {plate.plate_id: plate for plate in plates}
    adjacency: dict[int, set[int]] = {plate.plate_id: set() for plate in plates}

    for i, left in enumerate(plates):
        for right in plates[i + 1 :]:
            if _distance_deg(left, right) <= threshold_deg:
                adjacency[left.plate_id].add(right.plate_id)
                adjacency[right.plate_id].add(left.plate_id)

    for plate in plates:
        if plate.plate_id in visited:
            continue

        stack = [plate.plate_id]
        cluster_ids: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster_ids.append(current)
            stack.extend(adjacency[current] - visited)

        clusters.append([lookup[plate_id] for plate_id in cluster_ids])

    return clusters


def _cluster_centroid(cluster: list[PlateStateV2]) -> tuple[float, float]:
    if not cluster:
        return (0.0, 0.0)
    lon = sum(plate.lon for plate in cluster) / len(cluster)
    lat = sum(plate.lat for plate in cluster) / len(cluster)
    return (round(lon, 4), round(lat, 4))


def update_supercontinent_state(
    *,
    state: SupercontinentState,
    continental_plates: list[PlateStateV2],
    time_ma: int,
    step_myr: int,
) -> SupercontinentState:
    if not continental_plates:
        state.phase = "stable"
        state.largest_cluster_fraction = 0.0
        state.centroid = None
        state.history.append((time_ma, 0.0))
        return state

    clusters = _build_clusters(continental_plates)
    largest = max(clusters, key=len)
    largest_fraction = len(largest) / max(1, len(continental_plates))

    previous_fraction = state.largest_cluster_fraction
    trend = largest_fraction - previous_fraction

    if largest_fraction >= 0.72:
        phase = "assembled"
    elif trend > 0.015:
        phase = "assembly"
    elif trend < -0.015:
        phase = "dispersal"
    else:
        phase = "stable"

    if (
        previous_fraction >= 0.72
        and largest_fraction <= 0.45
        and (state.last_cycle_transition_ma is None or abs(state.last_cycle_transition_ma - time_ma) >= 120)
    ):
        state.cycle_count += 1
        state.last_cycle_transition_ma = time_ma

    state.phase = phase
    state.largest_cluster_fraction = round(largest_fraction, 4)
    state.centroid = _cluster_centroid(largest)
    state.history.append((time_ma, round(largest_fraction, 4)))

    # Keep memory bounded for long runs.
    if len(state.history) > 3000:
        state.history = state.history[-3000:]

    return state
