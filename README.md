# Geologic Wizard v1

Geologic Wizard is a desktop-first tectonic worldbuilding tool for non-experts. It generates Earth-like geologic histories from `1000 Ma -> 0 Ma`, supports timeline scrubbing, bookmarking, regional refinement, and high-quality heightmap exports.

## Implemented in this scaffold

- Desktop shell: `Tauri + React + TypeScript`
- Local engine: `FastAPI + Python` with deterministic seeded generation
- API contracts matching plan endpoints:
  - `POST /v1/projects`
  - `POST /v1/projects/{projectId}/generate`
  - `GET /v1/projects/{projectId}/frames/{timeMa}`
  - `POST /v1/projects/{projectId}/bookmarks`
  - `POST /v1/projects/{projectId}/bookmarks/{bookmarkId}/refine`
  - `POST /v1/projects/{projectId}/edits`
  - `POST /v1/projects/{projectId}/exports`
  - `GET /v1/jobs/{jobId}`
  - `GET /v1/jobs/{jobId}/events`
  - `GET /v1/projects/{projectId}/validation`
- Persistence and cache layout under `~/.geologic_wizard` (or `GW_DATA_ROOT`)
- Deterministic preview and export terrain synthesis
- Bookmark refinement with regional emphasis
- Heightmap + metadata provenance export artifacts

## Repository layout

- `/Users/xander/Documents/Fantasy/Geologic_Wizard/apps/desktop` React + Tauri desktop client
- `/Users/xander/Documents/Fantasy/Geologic_Wizard/services/engine` FastAPI simulation/export engine
- `/Users/xander/Documents/Fantasy/Geologic_Wizard/docs/architecture.md` module-level architecture and contracts

## Quick start

### 1. Engine

```bash
cd /Users/xander/Documents/Fantasy/Geologic_Wizard
make engine-install
make engine-dev
```

Engine URL: `http://127.0.0.1:8765`

### 2. Desktop UI (web dev mode)

```bash
cd /Users/xander/Documents/Fantasy/Geologic_Wizard
make desktop-install
make desktop-dev
```

UI URL: `http://127.0.0.1:5173`

### 3. Tauri desktop app

```bash
cd /Users/xander/Documents/Fantasy/Geologic_Wizard/apps/desktop
npm run tauri:dev
```

## Current behavior notes

- Simulation uses deterministic kinematic + rules synthesis stubs designed for extension with PyGPlates-based reconstruction and topology models.
- Internal tectonic files are scaffolded as placeholders (`model.gpml`, `model.rot`) in each project folder.
- Export format currently supports 16-bit PNG (`png16`) and float TIFF (`tiff32`) with metadata sidecar JSON.

## Next implementation steps

1. Replace synthetic plate kinematics with real PyGPlates `RotationModel`, `ReconstructModel`, and `TopologicalModel` pipelines.
2. Add deterministic plate partitioning and explicit left/right boundary editing semantics in the UI map toolset.
3. Add progressive keyframe cache loading and streaming updates for large histories.
4. Add GPlates interoperability tests around generated GPML/ROT.
