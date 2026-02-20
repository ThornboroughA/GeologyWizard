# Geologic Wizard v1

Geologic Wizard is a desktop-first tectonic worldbuilding tool for non-experts. It generates Earth-like geologic histories from `1000 Ma -> 0 Ma`, supports timeline scrubbing, bookmarking, regional refinement, and high-quality heightmap exports.

## Current implementation state

- Desktop shell: `Tauri + React + TypeScript`
- Local engine: `FastAPI + Python`
- Simulation backends:
  - `fast_plausible`
  - `hybrid_rigor`
- Deterministic subsystem seeds (`plates`, `boundaries`, `events`, `terrain`)
- Frame model now includes:
  - plate/boundary kinematics
  - uncertainty summary
  - strain-field cache references
- New diagnostics endpoints:
  - `GET /v1/projects/{projectId}/frames/{timeMa}/diagnostics`
  - `GET /v1/projects/{projectId}/coverage`

## Quick start (one command)

### Recommended: full desktop mode

```bash
cd /Users/xander/Documents/Fantasy/Geologic_Wizard
make dev
```

What this does:
- ensures Python venv/deps for engine
- ensures Node deps for desktop app
- starts FastAPI engine on `http://127.0.0.1:8765`
- starts Tauri desktop app (with Vite frontend)
- shuts everything down cleanly when you exit

### Browser-only dev mode

```bash
cd /Users/xander/Documents/Fantasy/Geologic_Wizard
make dev-web
```

## Component run matrix

| Mode | Engine | Vite UI | Tauri shell | Command |
|---|---:|---:|---:|---|
| Full desktop | yes | yes | yes | `make dev` |
| Web dev | yes | yes | no | `make dev-web` |
| Manual split | optional | optional | optional | `make engine-dev` / `make desktop-dev` / `npm run tauri:dev` |

## Manual commands (troubleshooting)

```bash
# Engine
make engine-install
make engine-dev

# UI in browser
make desktop-install
make desktop-dev

# Tauri app
cd /Users/xander/Documents/Fantasy/Geologic_Wizard/apps/desktop
npm run tauri:dev
```

## API overview

- `POST /v1/projects`
- `POST /v1/projects/{projectId}/generate`
- `GET /v1/projects/{projectId}/frames/{timeMa}`
- `GET /v1/projects/{projectId}/frames/{timeMa}/diagnostics`
- `GET /v1/projects/{projectId}/coverage`
- `POST /v1/projects/{projectId}/bookmarks`
- `POST /v1/projects/{projectId}/bookmarks/{bookmarkId}/refine`
- `POST /v1/projects/{projectId}/edits`
- `POST /v1/projects/{projectId}/exports`
- `GET /v1/jobs/{jobId}`
- `GET /v1/jobs/{jobId}/events`
- `GET /v1/projects/{projectId}/validation`

## Troubleshooting

1. `make dev` fails with Tauri icon errors
- Ensure `/Users/xander/Documents/Fantasy/Geologic_Wizard/apps/desktop/src-tauri/icons/icon.png` exists.

2. Engine not reachable on startup
- Check if port `8765` is in use.
- Run with custom port: `ENGINE_PORT=8877 make dev`.

3. PyGPlates status shows unavailable
- This is expected unless `pygplates` is installed locally and valid tectonic files are present.
- The app will run with deterministic fallback coverage behavior.

4. Slow first startup
- First run may install Python/npm dependencies.
- Later runs reuse `.venv` and `node_modules`.

## Repo layout

- `/Users/xander/Documents/Fantasy/Geologic_Wizard/apps/desktop` React + Tauri client
- `/Users/xander/Documents/Fantasy/Geologic_Wizard/services/engine` FastAPI simulation/export engine
- `/Users/xander/Documents/Fantasy/Geologic_Wizard/docs/architecture.md` module-level architecture
- `/Users/xander/Documents/Fantasy/Geologic_Wizard/scripts/dev.sh` unified startup script
