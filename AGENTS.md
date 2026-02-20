# Geologic Wizard Agent Reference

This file is the canonical context sheet for any Codex agent working in this repository.

## Mission

Build a user-friendly, scientifically grounded world simulation tool for fictional worldbuilders.

The product must let non-experts:
- generate Earth-like tectonic histories across deep time (`1000 Ma` to present by default),
- scrub and inspect a timeline of geologic evolution,
- place bookmarks and refine regions at higher detail,
- export deterministic, high-quality terrain outputs (heightmaps + provenance metadata).

## Product Vision (Non-Negotiable)

1. Usability first for non-experts.
2. Geological plausibility second, but still strict enough to avoid obvious nonsense.
3. Deterministic reproducibility with seeds and provenance.
4. Timeline-centric workflow (time scrubbing is a core interaction, not a side feature).

## Scientific Baseline

- Assume Earth-like planetary conditions unless explicitly configured otherwise.
- Climate is out of scope.
- Model tectonic behavior through coherent plate motion, boundary kinematics, and persistent geologic events.
- Favor physically informed approximations over purely aesthetic procedural generation.

### Terminology

- `kinematic`: plate motions and boundary interactions are driven by prescribed or derived movement/rotation rules.
- `simulationist`: landform and event outcomes emerge from process-like models over time (uplift/subsidence/rifting/subduction behavior), not one-off decoration.

## Simulation Modes

- `fast_plausible`: iteration speed with constrained approximations.
- `hybrid_rigor`: slower, more physically informed mode with stronger continuity/kinematic constraints.

Both modes must remain deterministic for a fixed config + seed.

## Current Architecture Snapshot

- Desktop shell: `Tauri + React + TypeScript`
- Local engine: `FastAPI + Python`
- Core engine modules live under:
  - `services/engine/geologic_wizard_engine/modules/tectonic_backends.py`
  - `services/engine/geologic_wizard_engine/modules/terrain_synthesis.py`
  - `services/engine/geologic_wizard_engine/modules/validation.py`
  - `services/engine/geologic_wizard_engine/modules/pygplates_adapter.py`
- Simulation orchestration:
  - `services/engine/geologic_wizard_engine/simulation_service.py`
- API surface:
  - `services/engine/geologic_wizard_engine/api.py`

## PyGPlates Expectation

PyGPlates integration is optional at runtime today and must degrade cleanly when unavailable.

If `pygplates` is present, agents should keep integrating toward:
- cached `RotationModel`,
- cached `ReconstructModel`,
- cached `TopologicalModel`,
- improved coverage and diagnostics from topology availability.

When unavailable, behavior must remain deterministic and explicit in diagnostics/coverage output.

## UX Standards

1. Keep controls understandable for non-geologists.
2. Hide advanced geological parameters behind focused expert controls.
3. Surface uncertainty, coverage gaps, and runtime expectations clearly.
4. Keep map + timeline interaction responsive.

## Export Standards

All exports must include:
- deterministic artifact output,
- provenance metadata (solver/mode/profile/digests),
- enough information to reproduce the run.

## Priority Order For Future Work

1. Plate lifecycle continuity and boundary semantics.
2. Event synthesis realism (orogeny/rift/subduction/arc/terrane) with persistence behavior.
3. Terrain coupling from tectonic potential fields (uplift/subsidence/volcanic/strain proxies).
4. Diagnostics quality (continuity violations, boundary consistency, coverage gaps).
5. Performance optimization without compromising reproducibility.

## Guardrails For Agents

- Do not sacrifice deterministic behavior for visual noise.
- Do not introduce climate simulation into this codebase.
- Do not silently swallow coverage/uncertainty issues; report them in diagnostics.
- Do not break one-command startup (`make dev` / `make dev-web`).
- Preserve API compatibility where possible; if changing contracts, update:
  - engine models,
  - API handlers,
  - frontend types/client calls,
  - tests and docs in the same change.

## Definition Of A Good Change

A change is good when it:
1. improves geologic plausibility or usability,
2. preserves deterministic reproducibility,
3. includes diagnostics/validation coverage,
4. updates tests and docs to match behavior.

## Fast Start For Agents

- Full desktop run: `make dev`
- Web-only run: `make dev-web`
- Engine tests: `cd services/engine && . .venv/bin/activate && pytest -q`
- Frontend build: `cd apps/desktop && npm run build`
- Tauri check: `cd apps/desktop/src-tauri && cargo check`
