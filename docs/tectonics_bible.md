# Geologic Wizard Tectonics Bible v1

## Purpose
This is the canonical, long-lived design record for the tectonic simulation stack. It defines the modular pipeline, deterministic contracts, phase gates, visualization requirements, diagnostics, and iteration governance.

## Locked Assumptions
- Earth-like gravity, heat budget, and rheology proxies.
- Climate simulation remains out of scope.
- User-facing run controls are `quick` and `full` only.
- Physics-first phase gating.
- Early calibration horizon: `300 Ma`; milestone closeout includes `1 Ga` acceptance.
- Cratons are visible by default as subtle outlines plus dedicated overlay access.
- Determinism is mandatory for fixed seed/config/mode.

## Pipeline Contract (Per Frame)
1. `kinematics_step`
- Inputs: plate states, prior boundary force proxies, supercontinent tendency.
- Outputs: plate velocity/azimuth evolution and continuity proxies.

2. `boundary_semantics_step`
- Inputs: kinematics + boundary history.
- Outputs: state classes, polarity, persistence, transition reasons.

3. `lithosphere_step`
- Inputs: boundary states + crust fields.
- Outputs: crust type, oceanic age, crust thickness, subduction flux.

4. `lifecycle_step`
- Inputs: lithosphere fields + closure accounting.
- Outputs: births/deaths, area closure error, inheritance masks.

5. `events_step`
- Inputs: boundary/lifecycle state.
- Outputs: stateful events with phase, persistence, and confidence.

6. `surface_response_step`
- Inputs: event forcing + lithosphere rates.
- Outputs: uplift, subsidence, volcanic flux, erosion capacity, orogenic root, elevation.

7. `render_geometry_step`
- Inputs: raster fields + boundaries/events.
- Outputs: continent polygons, coastlines, active belts, craton outlines.

8. `diagnostics_step`
- Inputs: all upstream outputs.
- Outputs: checks, severities, and suggested fixes.

## Persisted Module Snapshot Schema
Every frame stores module state snapshots in run storage:
- `stepId`
- `inputDigest`
- `outputDigest`
- `keyMetrics`
- `transitionReasons`

Frame-level deterministic chain:
- `replayHash = hash(ordered step output digests)`

## Phase Roadmap
### Phase 0
- Bible, module contracts, instrumentation, deterministic step hashes.

### Phase 1
- Basic macro world readability:
- persistent cratons,
- continent polygons from connected components + polygonized boundaries,
- causal coastlines.

### Phase 2
- Wilson-cycle foundations:
- staged rift->spreading,
- subduction preconditions,
- oceanic age creation/consumption,
- inherited suture weakness.

### Phase 3
- Orogeny/accretion/arc persistence and strict subduction coupling.

### Phase 4
- Surface memory integration and coastline stability.

### Phase 5
- Quick/Full contract hardening (full may refine, not rewrite macro history).

### Phase 6
- High-detail window hooks (20-40 Myr regional pass APIs/configs).

## Diagnostics Gates
### Hard errors
- Non-determinism for identical seed/config.
- `boundary.motion_mismatch_count > 0`.
- `lifecycle.net_area_balance_error > 0.01`.

### Warnings
- `continuity.max_velocity_jump_cm_per_yr`: warn `>2`, error `>5`.
- `boundary.type_flip_rate_per_100myr`: warn `>6`, error `>10`.
- `oceanicAgeP99Myr`: warn `>280`.
- short-lived orogeny events `<10 Myr`.

### Visualization gates
- `render.one_row_polygon_fraction <= 0.05`.
- `overlay.field_spatial_variance > 0` for field overlays.
- coastline temporal flicker index below threshold (tracked per release).

## Current Baseline (v1)
- `modularity.structure`: WARN
- `visual.continent_geometry`: FAIL (historically row-band artifacts)
- `visual.overlay_fidelity`: FAIL (historically frame-global scalar coloring)
- `runtime.quick_vs_full_separation`: PASS in 300 Ma probe
- `continuity.max_velocity_jump_cm_per_yr`: FAIL baseline
- `boundary.motion_mismatch_count`: PASS baseline probe
- `lifecycle.net_area_balance_error`: PASS baseline probe

## API Surface Additions (v2)
- `GET /v2/projects/{projectId}/frames/{timeMa}/fields/{fieldName}`
- `GET /v2/projects/{projectId}/frames/{timeMa}/module-states`
- `GET /v2/projects/{projectId}/runs/{runId}/metrics`

## Iteration Governance
Each day update this file with:
- What changed.
- What passed/failed.
- Next tuning knobs.

Every PR that changes simulation behavior must update:
- phase status,
- diagnostics baseline table,
- at least one acceptance test.

## Sources
Process framing: Turcotte & Schubert, Kearey et al., Stern (2002), Wilson (1966).
Guardrails: Müller et al. (2008), USGS plate motion rates, NOAA seafloor age context.
Supercontinent uncertainty: Nance et al. (2014).
