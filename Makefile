ENGINE_PORT ?= 8765

.PHONY: engine-install engine-dev engine-test desktop-install desktop-dev dev dev-web

engine-install:
	cd services/engine && python3 -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

engine-dev:
	cd services/engine && . .venv/bin/activate && uvicorn geologic_wizard_engine.main:app --reload --port $(ENGINE_PORT)

engine-test:
	cd services/engine && . .venv/bin/activate && pytest

desktop-install:
	cd apps/desktop && npm install

desktop-dev:
	cd apps/desktop && npm run dev

dev:
	ENGINE_PORT=$(ENGINE_PORT) ./scripts/dev.sh desktop

dev-web:
	ENGINE_PORT=$(ENGINE_PORT) ./scripts/dev.sh web
