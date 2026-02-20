from __future__ import annotations

from geologic_wizard_engine.metadata_store import MetadataStore
from geologic_wizard_engine.models import ProjectConfig
from geologic_wizard_engine.project_store import ProjectStore
from geologic_wizard_engine.settings import Settings
from geologic_wizard_engine.simulation_service import SimulationService
from geologic_wizard_engine.utils import stable_hash


def test_frame_generation_is_deterministic(tmp_path):
    settings = Settings(data_root=tmp_path)
    metadata = MetadataStore(tmp_path / "metadata.sqlite3")
    project_store = ProjectStore(tmp_path)
    service = SimulationService(settings, metadata, project_store)

    config = ProjectConfig(seed=1337, plateCount=12)
    project = metadata.create_project(
        project_id="p1",
        name="Determinism",
        config=config,
        project_hash=stable_hash(config.model_dump(mode="json")),
        created_at="2026-01-01T00:00:00+00:00",
    )
    project_store.initialize_project("p1", config.model_dump(mode="json"))

    frame_a = service.compute_frame(project, 250)
    frame_b = service.compute_frame(project, 250)

    assert frame_a.model_dump(mode="json") == frame_b.model_dump(mode="json")
