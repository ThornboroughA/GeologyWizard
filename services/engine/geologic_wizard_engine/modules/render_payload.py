from __future__ import annotations

from typing import Any

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


def build_frame_render_payload(frame: TimelineFrame, *, source: str, nearest_time_ma: int) -> FrameRender:
    plate_kinematics = {item.plateId: item for item in frame.plateKinematics}
    boundary_kinematics = {item.segmentId: item for item in frame.boundaryKinematics}

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
        }
        boundary_features.append(
            GeoJsonFeature(
                geometry=wrap_line_geometry(boundary.geometry),
                properties=properties,
            )
        )

    overlay_features: list[GeoJsonFeature] = []
    for event in frame.eventOverlays:
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
        }
        overlay_features.append(
            GeoJsonFeature(
                geometry=wrap_geometry(event.regionGeometry),
                properties=properties,
            )
        )

    return FrameRender(
        timeMa=frame.timeMa,
        landmassGeoJson=GeoJsonFeatureCollection(features=landmass_features),
        boundaryGeoJson=GeoJsonFeatureCollection(features=boundary_features),
        overlayGeoJson=GeoJsonFeatureCollection(features=overlay_features),
        source="cache" if source == "cache" else "generated",
        nearestTimeMa=nearest_time_ma,
    )
