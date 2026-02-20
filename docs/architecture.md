# Geologic Wizard Architecture

## Engine modules

1. `SimulationService` (`services/engine/geologic_wizard_engine/simulation_service.py`)
- Orchestrates timeline generation, replay, refinement, exports, diagnostics, and coverage reporting.
- Persists frame caches and run manifests.

2. `Tectonic backends` (`services/engine/geologic_wizard_engine/modules/tectonic_backends.py`)
- Pluggable modes: `fast_plausible`, `hybrid_rigor`.
- Coherent plate state evolution with continuity tracking.
- Boundary kinematics + event synthesis with persistence windows.

3. `PyGPlates adapter` (`services/engine/geologic_wizard_engine/modules/pygplates_adapter.py`)
- Optional runtime binding.
- Caches Rotation/Reconstruct/Topological model handles when available.
- Feeds coverage hints into uncertainty/diagnostics.

4. `Terrain synthesis` (`services/engine/geologic_wizard_engine/modules/terrain_synthesis.py`)
- Builds uplift/subsidence/volcanic/crust-age potentials.
- Produces preview heightfield and strain-aware regional refinement.

5. `Validation` (`services/engine/geologic_wizard_engine/modules/validation.py`)
- Frame invariant checks.
- Cross-frame continuity checks.

6. `Persistence` (`services/engine/geologic_wizard_engine/project_store.py`, `metadata_store.py`)
- SQLite metadata.
- Per-project caches: preview, strain, refined.
- Run manifests and per-time diagnostics.

## Key API additions

- `GET /v1/projects/{projectId}/frames/{timeMa}/diagnostics`
- `GET /v1/projects/{projectId}/coverage`
- Generation overrides via `POST /v1/projects/{projectId}/generate`:
  - `simulationModeOverride`
  - `rigorProfileOverride`
  - `targetRuntimeMinutesOverride`

## Data flow

1. Create project -> persist config and initialize cache layout.
2. Generate run -> backend evolves plate states through time and caches keyframes.
3. Frame request -> read keyframe cache or deterministically replay to target time.
4. Bookmark refinement -> strain-aware regional upres with global tectonic context frozen at bookmark time.
5. Export -> 8K-ready heightmap + metadata provenance (solver/profile/kinematics/uncertainty digests).
