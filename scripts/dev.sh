#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-desktop}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$ROOT_DIR/services/engine"
DESKTOP_DIR="$ROOT_DIR/apps/desktop"
ENGINE_PORT="${ENGINE_PORT:-8765}"

if [[ "$MODE" != "desktop" && "$MODE" != "web" ]]; then
  echo "Usage: scripts/dev.sh [desktop|web]"
  exit 1
fi

ensure_engine() {
  cd "$ENGINE_DIR"
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate
  if ! python - <<'PY'
import fastapi
import uvicorn
import numpy
from PIL import Image
print("engine deps ok")
PY
  then
    pip install -e .[dev]
  fi
  deactivate
}

ensure_desktop() {
  cd "$DESKTOP_DIR"
  if [[ ! -d node_modules ]]; then
    npm install
  fi
}

wait_for_engine() {
  local retries=60
  while [[ $retries -gt 0 ]]; do
    if curl -fsS "http://127.0.0.1:${ENGINE_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    retries=$((retries - 1))
    sleep 1
  done
  return 1
}

cleanup() {
  if [[ -n "${ENGINE_PID:-}" ]]; then
    kill "$ENGINE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "[geologic-wizard] ensuring dependencies"
ensure_engine
ensure_desktop

echo "[geologic-wizard] starting engine on :${ENGINE_PORT}"
(
  cd "$ENGINE_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uvicorn geologic_wizard_engine.main:app --reload --port "$ENGINE_PORT"
) &
ENGINE_PID=$!

if ! wait_for_engine; then
  echo "[geologic-wizard] engine failed to become healthy"
  exit 1
fi

if [[ "$MODE" == "desktop" ]]; then
  echo "[geologic-wizard] starting desktop app (Tauri + Vite)"
  cd "$DESKTOP_DIR"
  npm run tauri:dev
else
  echo "[geologic-wizard] starting web UI (Vite)"
  cd "$DESKTOP_DIR"
  npm run dev
fi
