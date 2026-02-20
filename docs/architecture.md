# Geologic Wizard Architecture

## Core modules

1. Desktop Shell Module (`apps/desktop/src-tauri`)
- Tauri process and app lifecycle.
- Command bridge for local secure operations.

2. UI Experience Module (`apps/desktop/src`)
- Guided setup, timeline scrubber, bookmark workflow.
- Focused expert panel for constrained tectonic edits.

3. Map/Globe Module (`apps/desktop/src/components/MapScene.tsx`)
- Shared frame model feeding 2D and 3D views.
- Geometry rendering of plate and boundary overlays.

4. Timeline & Bookmark Module (`apps/desktop/src/components/TimelineScrubber.tsx` + engine bookmark endpoints)
- Geologic time scrubbing from 1000 Ma to 0 Ma.
- Bookmark creation and refinement actions.

5. Simulation Orchestrator Module (`services/engine/geologic_wizard_engine/simulation_service.py`)
- Full-run generation loops.
- Keyframe cache writing and frame retrieval.

6. PyGPlates Core Integration Layer (scaffold)
- Current: deterministic synthetic tectonic generator (`modules/tectonics.py`).
- Intended replacement: PyGPlates reconstruction/topology models and plate partitioning.

7. Tectonic Event Synthesis Module (`modules/tectonics.py`)
- Derives events from boundary types and expert edits.

8. Terrain Synthesis Module (`modules/terrain_synthesis.py`)
- Generates preview terrain and refined regional terrain arrays.

9. Export Module (`simulation_service.py`)
- Heightmap artifact output and provenance metadata generation.

10. Persistence & Cache Module (`metadata_store.py`, `project_store.py`)
- SQLite metadata and deterministic project folder layout.
- Preview and refined raster cache files.

11. Validation Module (`modules/validation.py`)
- Reconstruction field checks and boundary consistency checks.

12. Diagnostics Module (`job_manager.py` + job endpoints)
- Job queue states, progress, and streaming updates.

## Data flow

1. Create project -> persist config and initialize storage.
2. Generate timeline job -> keyframes written every 5 Myr.
3. UI scrubs timeline -> frame loaded from cache or generated on demand.
4. Bookmark -> immutable frame hash references selected time.
5. Refine bookmark -> regional upres cache output.
6. Export -> heightmap + metadata JSON with provenance hashes.

## Determinism guarantees

- Config + seed determine synthetic tectonic features and terrain generation.
- Export metadata records parameter and event hashes.
- Re-running with identical config and seed yields identical frame payloads.
