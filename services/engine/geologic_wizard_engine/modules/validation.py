from __future__ import annotations

from ..models import TimelineFrame, ValidationIssue


def validate_frame(frame: TimelineFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for plate in frame.plateGeometries:
        if not plate.geometry:
            issues.append(
                ValidationIssue(
                    code="missing_geometry",
                    severity="error",
                    message=f"Plate {plate.plateId} has no geometry",
                )
            )
        if plate.reconstructionPlateId <= 0:
            issues.append(
                ValidationIssue(
                    code="invalid_reconstruction_plate",
                    severity="error",
                    message=f"Plate {plate.plateId} has invalid reconstructionPlateId",
                )
            )

    for boundary in frame.boundaryGeometries:
        if boundary.leftPlateId == boundary.rightPlateId:
            issues.append(
                ValidationIssue(
                    code="boundary_same_plate",
                    severity="warning",
                    message=f"Boundary {boundary.segmentId} references the same plate on both sides",
                )
            )

    if not frame.eventOverlays:
        issues.append(
            ValidationIssue(
                code="no_events",
                severity="warning",
                message="No geologic events were synthesized for this frame",
            )
        )

    return issues
