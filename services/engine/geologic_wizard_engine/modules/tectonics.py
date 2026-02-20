from __future__ import annotations

import math
import random
from typing import Any

from ..models import BoundarySegment, BoundaryType, GeoEvent, GeoEventType, PlateFeature, ProjectConfig


def _wrap_lon(lon: float) -> float:
    value = ((lon + 180.0) % 360.0) - 180.0
    if value == -180.0:
        return 180.0
    return value


def _polygon_ring(lat: float, lon: float, radius_deg: float, points: int = 10) -> list[list[float]]:
    ring: list[list[float]] = []
    for idx in range(points):
        theta = 2.0 * math.pi * (idx / points)
        nlat = max(-89.0, min(89.0, lat + radius_deg * math.sin(theta)))
        nlon = _wrap_lon(lon + radius_deg * math.cos(theta))
        ring.append([round(nlon, 5), round(nlat, 5)])
    ring.append(ring[0])
    return ring


def build_plate_features(config: ProjectConfig, time_ma: int) -> list[PlateFeature]:
    features: list[PlateFeature] = []
    base_rng = random.Random(config.seed + (time_ma * 131))

    for plate_idx in range(config.plateCount):
        phase = (2 * math.pi * plate_idx) / config.plateCount
        lat = 62.0 * math.sin(phase + (time_ma * 0.0045) + (config.seed % 17) * 0.03)
        lon = _wrap_lon((plate_idx * (360.0 / config.plateCount)) + time_ma * 0.22 + config.seed * 0.02)
        radius = 9.0 + (plate_idx % 5) * 1.9 + base_rng.random() * 2.2
        geometry = {
            "type": "Polygon",
            "coordinates": [_polygon_ring(lat, lon, radius)],
        }
        features.append(
            PlateFeature(
                plateId=plate_idx + 1,
                name=f"Plate {plate_idx + 1}",
                geometry=geometry,
                validTime=(float(config.startTimeMa), float(config.endTimeMa)),
                reconstructionPlateId=plate_idx + 1,
            )
        )
    return features


def build_boundary_segments(plates: list[PlateFeature], time_ma: int) -> list[BoundarySegment]:
    boundaries: list[BoundarySegment] = []
    types = [BoundaryType.convergent, BoundaryType.divergent, BoundaryType.transform]

    for idx in range(len(plates)):
        left = plates[idx]
        right = plates[(idx + 1) % len(plates)]
        boundary_type = types[(idx + (time_ma // 25)) % len(types)]
        left_ctr = left.geometry["coordinates"][0][0]
        right_ctr = right.geometry["coordinates"][0][0]
        geometry = {
            "type": "LineString",
            "coordinates": [left_ctr, right_ctr],
        }
        boundaries.append(
            BoundarySegment(
                segmentId=f"seg_{left.plateId}_{right.plateId}",
                leftPlateId=left.plateId,
                rightPlateId=right.plateId,
                boundaryType=boundary_type,
                digitizationDirection="forward",
                subductingSide="left" if boundary_type == BoundaryType.convergent else "none",
                isActive=True,
                geometry=geometry,
            )
        )
    return boundaries


def build_geo_events(
    boundaries: list[BoundarySegment],
    time_ma: int,
    edits: list[dict[str, Any]] | None = None,
) -> list[GeoEvent]:
    events: list[GeoEvent] = []
    edit_boost = 0.0

    if edits:
        for edit in edits:
            if edit["editType"] == "event_boost" and abs(edit["timeMa"] - time_ma) <= 30:
                edit_boost += float(edit.get("payload", {}).get("boost", 0.1))

    for segment in boundaries:
        if segment.boundaryType == BoundaryType.convergent:
            event_type = GeoEventType.orogeny
            base_intensity = 0.6
        elif segment.boundaryType == BoundaryType.divergent:
            event_type = GeoEventType.rift
            base_intensity = 0.45
        else:
            event_type = GeoEventType.terrane
            base_intensity = 0.25

        intensity = max(0.0, min(1.0, base_intensity + edit_boost))
        events.append(
            GeoEvent(
                eventId=f"evt_{segment.segmentId}_{time_ma}",
                eventType=event_type,
                timeStartMa=max(0.0, float(time_ma) - 5.0),
                timeEndMa=float(time_ma),
                intensity=intensity,
                sourceBoundaryIds=[segment.segmentId],
                regionGeometry=segment.geometry,
            )
        )

    return events
