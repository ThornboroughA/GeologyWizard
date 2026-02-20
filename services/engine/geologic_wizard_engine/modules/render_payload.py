from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..models import FrameRender, GeoJsonFeature, GeoJsonFeatureCollection, TimelineFrame


def _normalize_lon(value: float) -> float:
    wrapped = ((value + 180.0) % 360.0) - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _round_point(point: list[float]) -> list[float]:
    return [round(float(point[0]), 5), round(float(point[1]), 5)]


def _dedupe_consecutive(points: list[list[float]]) -> list[list[float]]:
    deduped: list[list[float]] = []
    for point in points:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    return deduped


def _close_ring(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return points
    closed = _dedupe_consecutive(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    return closed


def _ring_area(points: list[list[float]]) -> float:
    if len(points) < 4:
        return 0.0
    total = 0.0
    for idx in range(len(points) - 1):
        x1, y1 = points[idx]
        x2, y2 = points[idx + 1]
        total += (x1 * y2) - (x2 * y1)
    return abs(total) * 0.5


def split_line_antimeridian(line: list[list[float]]) -> list[list[list[float]]]:
    if len(line) < 2:
        return []

    segments: list[list[list[float]]] = [[_round_point([_normalize_lon(line[0][0]), line[0][1]])]]

    for raw_next in line[1:]:
        previous = segments[-1][-1]
        prev_lon = float(previous[0])
        prev_lat = float(previous[1])
        next_lon_norm = _normalize_lon(float(raw_next[0]))
        next_lat = float(raw_next[1])

        next_lon_unwrapped = next_lon_norm
        delta = next_lon_unwrapped - prev_lon
        if delta > 180.0:
            next_lon_unwrapped -= 360.0
        elif delta < -180.0:
            next_lon_unwrapped += 360.0

        crosses = next_lon_unwrapped > 180.0 or next_lon_unwrapped < -180.0
        if not crosses:
            segments[-1].append(_round_point([next_lon_unwrapped, next_lat]))
            continue

        boundary_lon = 180.0 if next_lon_unwrapped > 180.0 else -180.0
        denom = next_lon_unwrapped - prev_lon
        t = 0.0 if abs(denom) < 1e-9 else (boundary_lon - prev_lon) / denom
        lat_cross = prev_lat + t * (next_lat - prev_lat)

        segments[-1].append(_round_point([boundary_lon, lat_cross]))

        opposite_lon = -180.0 if boundary_lon == 180.0 else 180.0
        continued = [_round_point([opposite_lon, lat_cross]), _round_point([_normalize_lon(next_lon_unwrapped), next_lat])]
        segments.append(_dedupe_consecutive(continued))

    clean_segments = [_dedupe_consecutive(segment) for segment in segments]
    return [segment for segment in clean_segments if len(segment) >= 2]


def split_polygon_ring_antimeridian(ring: list[list[float]]) -> list[list[list[float]]]:
    if len(ring) < 4:
        return []

    points = ring[:-1] if ring[0] == ring[-1] else ring[:]
    if len(points) < 3:
        return []

    first = _round_point([_normalize_lon(points[0][0]), points[0][1]])
    current: list[list[float]] = [first]
    rings: list[list[list[float]]] = []

    for idx in range(1, len(points) + 1):
        raw_next = points[idx % len(points)]
        prev_lon = float(current[-1][0])
        prev_lat = float(current[-1][1])
        next_lon_norm = _normalize_lon(float(raw_next[0]))
        next_lat = float(raw_next[1])

        next_lon_unwrapped = next_lon_norm
        delta = next_lon_unwrapped - prev_lon
        if delta > 180.0:
            next_lon_unwrapped -= 360.0
        elif delta < -180.0:
            next_lon_unwrapped += 360.0

        crosses = next_lon_unwrapped > 180.0 or next_lon_unwrapped < -180.0
        if not crosses:
            current.append(_round_point([next_lon_unwrapped, next_lat]))
            continue

        boundary_lon = 180.0 if next_lon_unwrapped > 180.0 else -180.0
        denom = next_lon_unwrapped - prev_lon
        t = 0.0 if abs(denom) < 1e-9 else (boundary_lon - prev_lon) / denom
        lat_cross = prev_lat + t * (next_lat - prev_lat)

        current.append(_round_point([boundary_lon, lat_cross]))
        closed = _close_ring(current)
        if len(closed) >= 4 and _ring_area(closed) > 1e-6:
            rings.append(closed)

        opposite_lon = -180.0 if boundary_lon == 180.0 else 180.0
        current = [_round_point([opposite_lon, lat_cross]), _round_point([_normalize_lon(next_lon_unwrapped), next_lat])]

    closed = _close_ring(current)
    if len(closed) >= 4 and _ring_area(closed) > 1e-6:
        rings.append(closed)

    return rings


def wrap_polygon_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    geom_type = geometry.get("type")
    if geom_type == "MultiPolygon":
        polygons: list[list[list[list[float]]]] = []
        for polygon in geometry.get("coordinates", []):
            if not polygon:
                continue
            shell = polygon[0]
            for split_ring in split_polygon_ring_antimeridian(shell):
                polygons.append([split_ring])
        return {"type": "MultiPolygon", "coordinates": polygons}

    if geom_type == "Polygon":
        shell = geometry.get("coordinates", [[]])[0]
        polygons = [[split_ring] for split_ring in split_polygon_ring_antimeridian(shell)]
        return {"type": "MultiPolygon", "coordinates": polygons}

    return geometry


def wrap_line_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    geom_type = geometry.get("type")
    if geom_type == "MultiLineString":
        lines: list[list[list[float]]] = []
        for line in geometry.get("coordinates", []):
            lines.extend(split_line_antimeridian(line))
        return {"type": "MultiLineString", "coordinates": lines}

    if geom_type == "LineString":
        return {"type": "MultiLineString", "coordinates": split_line_antimeridian(geometry.get("coordinates", []))}

    return geometry


def wrap_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    geom_type = geometry.get("type")
    if geom_type in {"Polygon", "MultiPolygon"}:
        return wrap_polygon_geometry(geometry)
    if geom_type in {"LineString", "MultiLineString"}:
        return wrap_line_geometry(geometry)
    return geometry


def _cell_bounds(col: int, row: int, width: int, height: int) -> tuple[float, float, float, float]:
    lon_min = -180.0 + (float(col) / float(width)) * 360.0
    lon_max = -180.0 + (float(col + 1) / float(width)) * 360.0
    lat_max = 90.0 - (float(row) / float(height)) * 180.0
    lat_min = 90.0 - (float(row + 1) / float(height)) * 180.0
    return lon_min, lon_max, lat_min, lat_max


def _safe_load_array(ref: str | None) -> np.ndarray | None:
    if not ref:
        return None
    path = Path(ref)
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        if arr.ndim == 2:
            return arr
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            return squeezed
    except Exception:
        return None
    return None


def _build_surface_bundle(frame: TimelineFrame) -> dict[str, Any] | None:
    preview = _safe_load_array(frame.previewHeightFieldRef)
    if preview is None:
        return None

    crust_type = _safe_load_array(frame.crustTypeFieldRef)
    if crust_type is not None and crust_type.shape == preview.shape:
        land = np.logical_or(crust_type > 0, preview >= 0.57)
    else:
        land = preview >= 0.53

    stride = max(1, int(max(preview.shape[0], preview.shape[1]) / 240))

    fields = {
        "preview": preview,
        "oceanic_age": _safe_load_array(frame.oceanicAgeFieldRef),
        "crust_type": crust_type,
        "crust_thickness": _safe_load_array(frame.crustThicknessFieldRef),
        "tectonic_potential": _safe_load_array(frame.tectonicPotentialFieldRef),
        "uplift_rate": _safe_load_array(frame.upliftRateFieldRef),
        "subsidence_rate": _safe_load_array(frame.subsidenceRateFieldRef),
        "volcanic_flux": _safe_load_array(frame.volcanicFluxFieldRef),
        "erosion_capacity": _safe_load_array(frame.erosionCapacityFieldRef),
        "orogenic_root": _safe_load_array(frame.orogenicRootFieldRef),
        "craton_id": _safe_load_array(frame.cratonIdFieldRef),
    }

    downsampled: dict[str, np.ndarray | None] = {}
    for name, arr in fields.items():
        if arr is not None and arr.shape == preview.shape:
            downsampled[name] = arr[::stride, ::stride]
        else:
            downsampled[name] = None

    return {
        "mask": land[::stride, ::stride],
        "stride": stride,
        "fields": downsampled,
    }


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=np.uint8)
    components: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or visited[row, col]:
                continue
            stack = [(col, row)]
            visited[row, col] = 1
            cells: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if visited[ny, nx] or not bool(mask[ny, nx]):
                        continue
                    visited[ny, nx] = 1
                    stack.append((nx, ny))
            components.append(cells)

    return components


def _grid_to_lon_lat(point: tuple[int, int], width: int, height: int) -> list[float]:
    x, y = point
    lon = -180.0 + (float(x) / float(width)) * 360.0
    lat = 90.0 - (float(y) / float(height)) * 180.0
    return [round(lon, 5), round(lat, 5)]


def _simplify_grid_ring(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(points) < 4:
        return points
    simplified: list[tuple[int, int]] = [points[0]]
    for idx in range(1, len(points) - 1):
        px, py = simplified[-1]
        cx, cy = points[idx]
        nx, ny = points[idx + 1]
        if (px == cx == nx) or (py == cy == ny):
            continue
        simplified.append((cx, cy))
    simplified.append(points[-1])
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return simplified


def _component_rings(cells: list[tuple[int, int]], width: int, height: int) -> list[list[list[float]]]:
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for x, y in cells:
        p0 = (x, y)
        p1 = (x + 1, y)
        p2 = (x + 1, y + 1)
        p3 = (x, y + 1)
        for edge in ((p0, p1), (p1, p2), (p2, p3), (p3, p0)):
            rev = (edge[1], edge[0])
            if rev in edges:
                edges.remove(rev)
            else:
                edges.add(edge)

    rings: list[list[list[float]]] = []
    while edges:
        start_edge = next(iter(edges))
        edges.remove(start_edge)
        ring: list[tuple[int, int]] = [start_edge[0], start_edge[1]]
        current = start_edge[1]
        start = start_edge[0]

        while current != start:
            next_edge: tuple[tuple[int, int], tuple[int, int]] | None = None
            for candidate in edges:
                if candidate[0] == current:
                    next_edge = candidate
                    break
            if next_edge is None:
                break
            edges.remove(next_edge)
            ring.append(next_edge[1])
            current = next_edge[1]

        if len(ring) < 4 or ring[0] != ring[-1]:
            continue

        simplified = _simplify_grid_ring(ring)
        lonlat_ring = [_grid_to_lon_lat(point, width, height) for point in simplified]
        lonlat_ring = _close_ring(_dedupe_consecutive(lonlat_ring))
        if len(lonlat_ring) >= 4 and _ring_area(lonlat_ring) > 1e-6:
            rings.append(lonlat_ring)

    return rings


def _ring_is_meaningful(ring: list[list[float]]) -> bool:
    if len(ring) < 4:
        return False
    unique_lats = {round(point[1], 5) for point in ring[:-1]}
    unique_lons = {round(point[0], 5) for point in ring[:-1]}
    if len(unique_lats) <= 2 or len(unique_lons) <= 2:
        return False
    return _ring_area(ring) > 1e-4


def _component_feature_properties(
    cells: list[tuple[int, int]],
    fields: dict[str, np.ndarray | None],
    frame: TimelineFrame,
    component_id: int,
) -> dict[str, Any]:
    xs = np.array([cell[0] for cell in cells], dtype=np.int32)
    ys = np.array([cell[1] for cell in cells], dtype=np.int32)

    props: dict[str, Any] = {
        "derived": "surface_component",
        "componentId": component_id,
        "cellCount": len(cells),
        "supercontinentPhase": frame.plateLifecycleState.supercontinentPhase if frame.plateLifecycleState else None,
    }

    preview = fields.get("preview")
    if preview is not None:
        vals = preview[ys, xs]
        props["relief"] = round(float(np.mean(vals)), 4)

    oceanic_age = fields.get("oceanic_age")
    if oceanic_age is not None:
        props["oceanicAgeMeanMyr"] = round(float(np.mean(oceanic_age[ys, xs])), 4)

    crust_thickness = fields.get("crust_thickness")
    if crust_thickness is not None:
        props["crustThicknessMeanKm"] = round(float(np.mean(crust_thickness[ys, xs])), 4)

    tectonic_potential = fields.get("tectonic_potential")
    if tectonic_potential is not None:
        props["tectonicPotentialMean"] = round(float(np.mean(tectonic_potential[ys, xs])), 4)

    uplift_rate = fields.get("uplift_rate")
    if uplift_rate is not None:
        props["upliftMean"] = round(float(np.mean(uplift_rate[ys, xs])), 4)

    subsidence_rate = fields.get("subsidence_rate")
    if subsidence_rate is not None:
        props["subsidenceMean"] = round(float(np.mean(subsidence_rate[ys, xs])), 4)

    volcanic_flux = fields.get("volcanic_flux")
    if volcanic_flux is not None:
        props["volcanicFluxMean"] = round(float(np.mean(volcanic_flux[ys, xs])), 4)

    craton_id = fields.get("craton_id")
    if craton_id is not None:
        props["cratonFraction"] = round(float(np.mean((craton_id[ys, xs] > 0).astype(np.float32))), 4)

    return props


def _surface_continent_features(
    mask: np.ndarray,
    fields: dict[str, np.ndarray | None],
    frame: TimelineFrame,
) -> list[GeoJsonFeature]:
    height, width = mask.shape
    features: list[GeoJsonFeature] = []
    components = sorted(_connected_components(mask), key=len, reverse=True)

    for component_id, cells in enumerate(components, start=1):
        if len(cells) < 4:
            continue
        rings = _component_rings(cells, width, height)
        polygons = [[ring] for ring in rings if _ring_is_meaningful(ring)]
        if not polygons:
            continue

        geometry: dict[str, Any]
        if len(polygons) == 1:
            geometry = {"type": "Polygon", "coordinates": polygons[0]}
        else:
            geometry = {"type": "MultiPolygon", "coordinates": polygons}

        features.append(
            GeoJsonFeature(
                geometry=wrap_polygon_geometry(geometry),
                properties=_component_feature_properties(cells, fields, frame, component_id),
            )
        )

    return features


def _surface_coastline_features(mask: np.ndarray) -> list[GeoJsonFeature]:
    height, width = mask.shape
    features: list[GeoJsonFeature] = []

    for row in range(height):
        for col in range(width):
            is_land = bool(mask[row, col])
            if not is_land:
                continue

            right = bool(mask[row, (col + 1) % width]) if width > 1 else is_land
            down = bool(mask[min(row + 1, height - 1), col]) if height > 1 else is_land

            if not right:
                _, lon_max, lat_min, lat_max = _cell_bounds(col, row, width, height)
                features.append(
                    GeoJsonFeature(
                        geometry={
                            "type": "LineString",
                            "coordinates": [
                                [round(lon_max, 5), round(lat_min, 5)],
                                [round(lon_max, 5), round(lat_max, 5)],
                            ],
                        },
                        properties={"derived": "coastline"},
                    )
                )

            if not down:
                lon_min, lon_max, lat_min, _ = _cell_bounds(col, row, width, height)
                features.append(
                    GeoJsonFeature(
                        geometry={
                            "type": "LineString",
                            "coordinates": [
                                [round(lon_min, 5), round(lat_min, 5)],
                                [round(lon_max, 5), round(lat_min, 5)],
                            ],
                        },
                        properties={"derived": "coastline"},
                    )
                )

    return features


def _surface_craton_features(craton_field: np.ndarray | None) -> list[GeoJsonFeature]:
    if craton_field is None:
        return []
    craton_int = craton_field.astype(np.int32)
    if not np.any(craton_int > 0):
        return []

    height, width = craton_int.shape
    features: list[GeoJsonFeature] = []
    for craton_id in sorted(int(value) for value in np.unique(craton_int) if int(value) > 0):
        mask = craton_int == craton_id
        for cells in _connected_components(mask):
            if len(cells) < 4:
                continue
            rings = _component_rings(cells, width, height)
            polygons = [[ring] for ring in rings if _ring_is_meaningful(ring)]
            if not polygons:
                continue
            geometry: dict[str, Any]
            if len(polygons) == 1:
                geometry = {"type": "Polygon", "coordinates": polygons[0]}
            else:
                geometry = {"type": "MultiPolygon", "coordinates": polygons}
            features.append(
                GeoJsonFeature(
                    geometry=wrap_polygon_geometry(geometry),
                    properties={"derived": "craton_core", "cratonId": craton_id},
                )
            )

    return features


def _field_stats(fields: dict[str, np.ndarray | None]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name, arr in fields.items():
        if arr is None:
            continue
        data = arr.astype(np.float32)
        stats[name] = {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "p01": float(np.percentile(data, 1)),
            "p99": float(np.percentile(data, 99)),
            "mean": float(np.mean(data)),
            "var": float(np.var(data)),
        }
    return stats


def build_frame_render_payload(frame: TimelineFrame, *, source: str, nearest_time_ma: int) -> FrameRender:
    plate_kinematics = {item.plateId: item for item in frame.plateKinematics}
    boundary_kinematics = {item.segmentId: item for item in frame.boundaryKinematics}
    boundary_states = {item.segmentId: item for item in frame.boundaryStates}

    landmass_features: list[GeoJsonFeature] = []
    for plate in frame.plateGeometries:
        kin = plate_kinematics.get(plate.plateId)
        properties = {
            "plateId": plate.plateId,
            "name": plate.name,
            "reconstructionPlateId": plate.reconstructionPlateId,
            "velocityCmYr": kin.velocityCmYr if kin else None,
            "azimuthDeg": kin.azimuthDeg if kin else None,
            "convergenceCmYr": kin.convergenceCmYr if kin else None,
            "divergenceCmYr": kin.divergenceCmYr if kin else None,
            "continuityScore": kin.continuityScore if kin else None,
            "supercontinentPhase": frame.plateLifecycleState.supercontinentPhase if frame.plateLifecycleState else None,
            "supercontinentClusterFraction": frame.plateLifecycleState.supercontinentLargestClusterFraction if frame.plateLifecycleState else None,
            "oceanicAgeP99Myr": frame.plateLifecycleState.oceanicAgeP99Myr if frame.plateLifecycleState else None,
        }
        landmass_features.append(
            GeoJsonFeature(
                geometry=wrap_polygon_geometry(plate.geometry),
                properties=properties,
            )
        )

    boundary_features: list[GeoJsonFeature] = []
    for boundary in frame.boundaryGeometries:
        kin = boundary_kinematics.get(boundary.segmentId)
        state = boundary_states.get(boundary.segmentId)
        properties = {
            "segmentId": boundary.segmentId,
            "leftPlateId": boundary.leftPlateId,
            "rightPlateId": boundary.rightPlateId,
            "boundaryType": boundary.boundaryType.value,
            "digitizationDirection": boundary.digitizationDirection,
            "subductingSide": boundary.subductingSide,
            "isActive": boundary.isActive,
            "relativeVelocityCmYr": kin.relativeVelocityCmYr if kin else None,
            "normalVelocityCmYr": kin.normalVelocityCmYr if kin else None,
            "tangentialVelocityCmYr": kin.tangentialVelocityCmYr if kin else None,
            "strainRate": kin.strainRate if kin else None,
            "recommendedBoundaryType": kin.recommendedBoundaryType.value if kin else None,
            "stateClass": state.stateClass.value if state else None,
            "transitionCount": state.transitionCount if state else 0,
            "subductionFlux": state.subductionFlux if state else 0.0,
            "averageOceanicAgeMyr": state.averageOceanicAgeMyr if state else 0.0,
            "motionMismatch": state.motionMismatch if state else False,
        }
        boundary_features.append(
            GeoJsonFeature(
                geometry=wrap_line_geometry(boundary.geometry),
                properties=properties,
            )
        )

    overlay_features: list[GeoJsonFeature] = []
    for event in frame.eventOverlays:
        phase_code = float(event.drivingMetrics.get("phaseCode", 1.0)) if event.drivingMetrics else 1.0
        phase = "initiation" if phase_code <= 0.5 else ("active" if phase_code <= 1.5 else "decay")
        properties = {
            "eventId": event.eventId,
            "eventType": event.eventType.value,
            "timeStartMa": event.timeStartMa,
            "timeEndMa": event.timeEndMa,
            "intensity": event.intensity,
            "confidence": event.confidence,
            "persistenceClass": event.persistenceClass,
            "sourceBoundaryIds": event.sourceBoundaryIds,
            "drivingMetrics": event.drivingMetrics,
            "phase": phase,
        }
        overlay_features.append(
            GeoJsonFeature(
                geometry=wrap_geometry(event.regionGeometry),
                properties=properties,
            )
        )

    surface_bundle = _build_surface_bundle(frame)
    continent_features: list[GeoJsonFeature] = []
    craton_features: list[GeoJsonFeature] = []
    field_stats: dict[str, dict[str, float]] = {}
    if surface_bundle is not None:
        surface_mask = surface_bundle["mask"]
        fields = surface_bundle["fields"]
        candidate_continents = _surface_continent_features(surface_mask, fields, frame)
        if candidate_continents:
            landmass_features = candidate_continents
            continent_features = candidate_continents
        else:
            continent_features = landmass_features
        coastline_features = _surface_coastline_features(surface_mask)
        craton_features = _surface_craton_features(fields.get("craton_id"))
        field_stats = _field_stats(fields)
    else:
        coastline_features = []
        continent_features = landmass_features

    active_belts_features = [
        feature
        for feature in boundary_features
        if str(feature.properties.get("stateClass")) in {"subduction", "collision", "suture", "rift", "ridge"}
        or str(feature.properties.get("boundaryType")) in {"convergent", "divergent"}
    ]

    return FrameRender(
        timeMa=frame.timeMa,
        landmassGeoJson=GeoJsonFeatureCollection(features=landmass_features),
        continentGeoJson=GeoJsonFeatureCollection(features=continent_features),
        cratonGeoJson=GeoJsonFeatureCollection(features=craton_features),
        boundaryGeoJson=GeoJsonFeatureCollection(features=boundary_features),
        overlayGeoJson=GeoJsonFeatureCollection(features=overlay_features),
        coastlineGeoJson=GeoJsonFeatureCollection(features=coastline_features),
        activeBeltsGeoJson=GeoJsonFeatureCollection(features=active_belts_features),
        fieldStats=field_stats,
        reliefFieldRef=frame.previewHeightFieldRef or None,
        source="cache" if source == "cache" else "generated",
        nearestTimeMa=nearest_time_ma,
    )
