# Tectonic Plausibility Checks

Use this checklist to design validation logic and diagnostics output.

## 1. Kinematic Continuity

- Check plate rotation continuity through time.
- Check velocity-vector smoothness for each plate centroid.
- Check for abrupt plate-ID reassignments without lifecycle event markers.

Recommended diagnostics:
- `continuity.max_velocity_jump_cm_per_yr`
- `continuity.unexplained_id_reassignments`
- `continuity.rotation_outlier_count`

## 2. Boundary Semantics Consistency

- Check that boundary class matches relative motion sign:
  - Divergent: net extension.
  - Convergent: net shortening.
  - Transform: dominant strike-slip/shear.
- Check that trench polarity and overriding/subducting assignment remain coherent along connected segments.
- Check that boundary-type transitions occur gradually or via explicit events.

Recommended diagnostics:
- `boundary.motion_mismatch_count`
- `boundary.polarity_flip_count`
- `boundary.type_flip_rate_per_100myr`

## 3. Plate Lifecycle Realism

- Check that oceanic plate creation is linked to active spreading systems.
- Check that oceanic consumption is linked to subduction, not arbitrary deletion.
- Check that plate birth/death events maintain topological closure and area accounting.

Recommended diagnostics:
- `lifecycle.unexplained_plate_births`
- `lifecycle.unexplained_plate_deaths`
- `lifecycle.net_area_balance_error`

## 4. Event Persistence and Coupling

- Check that orogeny, arc volcanism, and subsidence events persist for geologically reasonable durations.
- Check terrain potential fields for causal linkage:
  - uplift near convergence/collision,
  - subsidence near foreland/passive margins,
  - volcanic potential near arcs/ridges/intraplate anomaly paths.
- Check that event state transitions are deterministic for fixed seed/config.

Recommended diagnostics:
- `events.short_lived_orogeny_count`
- `events.uncoupled_volcanic_belts`
- `events.non_deterministic_transition_count`

## 5. Coverage and Uncertainty

- Check spatial coverage of boundary model, topology solution, and reconstruction data.
- Check temporal coverage gaps in requested timeline.
- Check where fallback logic replaced missing data.

Recommended diagnostics:
- `coverage.spatial_gap_fraction`
- `coverage.temporal_gap_myr`
- `coverage.fallback_region_count`

## 6. Severity Policy

- Emit `error` for deterministic violations and geometric impossibilities.
- Emit `warning` for low-likelihood but plausible outcomes.
- Emit `info` for heuristic inferences or weakly constrained mechanisms.

## 7. Minimum Report Schema

Return at least:
- `check_id`
- `severity`
- `time_range_ma`
- `region_or_plate_ids`
- `observed_value`
- `expected_range_or_rule`
- `explanation`
- `suggested_fix`
