from __future__ import annotations

from ..models import PlausibilityCheck, TimelineFrame, ValidationIssue


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

    if frame.plateLifecycleState is not None:
        if frame.plateLifecycleState.netAreaBalanceError > 0.01:
            issues.append(
                ValidationIssue(
                    code="lifecycle_area_balance_error",
                    severity="error",
                    message="Lifecycle area balance exceeded tolerance (1%)",
                    details={"netAreaBalanceError": frame.plateLifecycleState.netAreaBalanceError},
                )
            )
        if frame.plateLifecycleState.oceanicAgeP99Myr > 280:
            issues.append(
                ValidationIssue(
                    code="oceanic_age_outlier",
                    severity="warning",
                    message="Oceanic age p99 exceeds Earth-like heuristic range",
                    details={"oceanicAgeP99Myr": frame.plateLifecycleState.oceanicAgeP99Myr},
                )
            )

    for boundary_state in frame.boundaryStates:
        if boundary_state.motionMismatch:
            issues.append(
                ValidationIssue(
                    code="boundary_motion_mismatch",
                    severity="error",
                    message=f"Boundary {boundary_state.segmentId} state class mismatches relative motion",
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
        if speed_jump > 5.0:
            issues.append(
                ValidationIssue(
                    code="continuity_speed_jump_error",
                    severity="error",
                    message=f"Plate {plate_id} exceeds max velocity jump ({speed_jump:.2f} cm/yr)",
                    details={"plateId": plate_id, "speedJumpCmYr": round(speed_jump, 4)},
                )
            )
        elif speed_jump > 2.0:
            issues.append(
                ValidationIssue(
                    code="continuity_speed_jump_warning",
                    severity="warning",
                    message=f"Plate {plate_id} has elevated velocity jump ({speed_jump:.2f} cm/yr)",
                    details={"plateId": plate_id, "speedJumpCmYr": round(speed_jump, 4)},
                )
            )

        azimuth_jump = abs(((kin.azimuthDeg - old.azimuthDeg + 180.0) % 360.0) - 180.0)
        if azimuth_jump > 55.0:
            issues.append(
                ValidationIssue(
                    code="continuity_rotation_outlier",
                    severity="warning",
                    message=f"Plate {plate_id} has abrupt azimuth rotation ({azimuth_jump:.1f}°)",
                    details={"plateId": plate_id, "azimuthJumpDeg": round(azimuth_jump, 4)},
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


def build_plausibility_checks_from_frame(frame: TimelineFrame) -> list[PlausibilityCheck]:
    checks: list[PlausibilityCheck] = []

    max_velocity_jump = max([abs(item.convergenceCmYr - item.divergenceCmYr) for item in frame.plateKinematics] + [0.0])
    checks.append(
        PlausibilityCheck(
            checkId="continuity.max_velocity_jump_cm_per_yr",
            severity="warning" if max_velocity_jump > 2.0 else "info",
            timeRangeMa=(frame.timeMa, frame.timeMa),
            regionOrPlateIds=[f"plate:{item.plateId}" for item in frame.plateKinematics[:8]],
            observedValue=round(max_velocity_jump, 4),
            expectedRangeOrRule="warn > 2.0, error > 5.0",
            explanation="Frame-local proxy for abrupt kinematic change.",
            suggestedFix="Smooth force transitions or raise persistence hysteresis.",
        )
    )

    mismatch_count = sum(1 for item in frame.boundaryStates if item.motionMismatch)
    checks.append(
        PlausibilityCheck(
            checkId="boundary.motion_mismatch_count",
            severity="error" if mismatch_count > 0 else "info",
            timeRangeMa=(frame.timeMa, frame.timeMa),
            regionOrPlateIds=[f"boundary:{item.segmentId}" for item in frame.boundaryStates if item.motionMismatch][:8],
            observedValue=mismatch_count,
            expectedRangeOrRule="must equal 0",
            explanation="Boundary class should match motion style.",
            suggestedFix="Fix semantic transition thresholds and polarity constraints.",
        )
    )

    if frame.plateLifecycleState is not None:
        checks.append(
            PlausibilityCheck(
                checkId="lifecycle.net_area_balance_error",
                severity="error" if frame.plateLifecycleState.netAreaBalanceError > 0.01 else "info",
                timeRangeMa=(frame.timeMa, frame.timeMa),
                observedValue=frame.plateLifecycleState.netAreaBalanceError,
                expectedRangeOrRule="error > 0.01",
                explanation="Net area closure error for lifecycle raster update.",
                suggestedFix="Normalize crust-type transitions after lifecycle update.",
            )
        )
        checks.append(
            PlausibilityCheck(
                checkId="events.short_lived_orogeny_count",
                severity="warning" if frame.plateLifecycleState.shortLivedOrogenyCount > 0 else "info",
                timeRangeMa=(frame.timeMa, frame.timeMa),
                observedValue=frame.plateLifecycleState.shortLivedOrogenyCount,
                expectedRangeOrRule="warn when > 0 for duration < 10 Myr",
                explanation="Orogenies should persist over geologic timescales.",
                suggestedFix="Increase collision persistence and decay windows.",
            )
        )

    return checks


def summarize_check_severity(checks: list[PlausibilityCheck]) -> dict[str, int]:
    summary = {"error": 0, "warning": 0, "info": 0}
    for check in checks:
        summary[check.severity] = summary.get(check.severity, 0) + 1
    return summary
