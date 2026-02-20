# Geologic Wizard Engine

FastAPI-based local simulation engine for Geologic Wizard.

## Run

```bash
cd /Users/xander/Documents/Fantasy/Geologic_Wizard/services/engine
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn geologic_wizard_engine.main:app --reload --port 8765
```

## API

Interactive docs: `http://127.0.0.1:8765/docs`
