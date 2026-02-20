ENGINE_PORT ?= 8765

.PHONY: engine-install engine-dev engine-test desktop-install desktop-dev

engine-install:
	cd services/engine && python -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

engine-dev:
	cd services/engine && . .venv/bin/activate && uvicorn geologic_wizard_engine.main:app --reload --port $(ENGINE_PORT)

engine-test:
	cd services/engine && . .venv/bin/activate && pytest

desktop-install:
	cd apps/desktop && npm install

desktop-dev:
	cd apps/desktop && npm run dev
