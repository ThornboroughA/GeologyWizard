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
        if boundary.boundaryType.value == "convergent" and boundary.subductingSide == "none":
            issues.append(
                ValidationIssue(
                    code="convergent_missing_subducting_side",
                    severity="warning",
                    message=f"Convergent boundary {boundary.segmentId} does not declare subducting side",
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

    if frame.uncertaintySummary.coverage > 0.2:
        issues.append(
            ValidationIssue(
                code="topology_coverage_gap",
                severity="warning",
                message="Topology coverage is low for this frame; fallback behavior may dominate",
                details={"coverageUncertainty": frame.uncertaintySummary.coverage},
            )
        )

    return issues


def validate_frame_pair(previous: TimelineFrame | None, current: TimelineFrame) -> list[ValidationIssue]:
    if previous is None:
        return []

    issues: list[ValidationIssue] = []
    prev_kin = {item.plateId: item for item in previous.plateKinematics}
    curr_kin = {item.plateId: item for item in current.plateKinematics}

    for plate_id, kin in curr_kin.items():
        old = prev_kin.get(plate_id)
        if old is None:
            continue

        speed_jump = abs(kin.velocityCmYr - old.velocityCmYr)
        if speed_jump > 4.0:
            issues.append(
                ValidationIssue(
                    code="continuity_speed_jump",
                    severity="warning",
                    message=f"Plate {plate_id} has abrupt velocity change ({speed_jump:.2f} cm/yr)",
                    details={"plateId": plate_id, "speedJumpCmYr": round(speed_jump, 4)},
                )
            )

        if kin.continuityScore < 0.2:
            issues.append(
                ValidationIssue(
                    code="continuity_score_low",
                    severity="warning",
                    message=f"Plate {plate_id} continuity score is low ({kin.continuityScore:.2f})",
                    details={"plateId": plate_id, "continuityScore": kin.continuityScore},
                )
            )

    return issues
