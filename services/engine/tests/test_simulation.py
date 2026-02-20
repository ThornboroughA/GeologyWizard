from __future__ import annotations

from geologic_wizard_engine.metadata_store import MetadataStore
from geologic_wizard_engine.models import ProjectConfig
from geologic_wizard_engine.project_store import ProjectStore
from geologic_wizard_engine.settings import Settings
from geologic_wizard_engine.simulation_service import SimulationService
from geologic_wizard_engine.utils import stable_hash


def _build_service(tmp_path):
    settings = Settings(data_root=tmp_path)
    metadata = MetadataStore(tmp_path / "metadata.sqlite3")
    project_store = ProjectStore(tmp_path)
    service = SimulationService(settings, metadata, project_store)
    return service, metadata, project_store


def test_frame_generation_is_deterministic(tmp_path):
    service, metadata, project_store = _build_service(tmp_path)

    config = ProjectConfig(seed=1337, plateCount=12, startTimeMa=300, endTimeMa=0, timeIncrementMyr=1, stepMyr=1)
    metadata.create_project(
        project_id="p1",
        name="Determinism",
        config=config,
        project_hash=stable_hash(config.model_dump(mode="json")),
        created_at="2026-01-01T00:00:00+00:00",
    )
    project_store.initialize_project("p1", config.model_dump(mode="json"))

    service.generate_project(
        "p1",
        simulation_mode_override=None,
        rigor_profile_override=None,
        runtime_override=None,
        job_callback=lambda *_args: None,
        is_canceled=lambda: False,
    )

    project = metadata.get_project("p1")
    assert project is not None

    frame_a = service.get_or_create_frame(project, 250)
    frame_b = service.get_or_create_frame(project, 250)

    assert frame_a.frameHash == frame_b.frameHash
    assert frame_a.frame.model_dump(mode="json") == frame_b.frame.model_dump(mode="json")


def test_convergent_boundary_has_subducting_side(tmp_path):
    service, metadata, project_store = _build_service(tmp_path)

    config = ProjectConfig(seed=99, plateCount=10, startTimeMa=200, endTimeMa=0, timeIncrementMyr=1, stepMyr=1)
    metadata.create_project(
        project_id="p2",
        name="Boundaries",
        config=config,
        project_hash=stable_hash(config.model_dump(mode="json")),
        created_at="2026-01-01T00:00:00+00:00",
    )
    project_store.initialize_project("p2", config.model_dump(mode="json"))

    service.generate_project(
        "p2",
        simulation_mode_override="hybrid_rigor",
        rigor_profile_override="research",
        runtime_override=120,
        job_callback=lambda *_args: None,
        is_canceled=lambda: False,
    )

    project = metadata.get_project("p2")
    assert project is not None

    frame = service.get_or_create_frame(project, 150).frame
    convergent = [boundary for boundary in frame.boundaryGeometries if boundary.boundaryType.value == "convergent"]

    for boundary in convergent:
        assert boundary.subductingSide in {"left", "right"}


def test_coverage_report_contains_timeline(tmp_path):
    service, metadata, project_store = _build_service(tmp_path)

    config = ProjectConfig(seed=701, plateCount=9, startTimeMa=120, endTimeMa=0, timeIncrementMyr=1, stepMyr=1)
    metadata.create_project(
        project_id="p3",
        name="Coverage",
        config=config,
        project_hash=stable_hash(config.model_dump(mode="json")),
        created_at="2026-01-01T00:00:00+00:00",
    )
    project_store.initialize_project("p3", config.model_dump(mode="json"))

    service.generate_project(
        "p3",
        simulation_mode_override="fast_plausible",
        rigor_profile_override="balanced",
        runtime_override=45,
        job_callback=lambda *_args: None,
        is_canceled=lambda: False,
    )

    report = service.get_coverage_report("p3")
    assert report.projectId == "p3"
    assert len(report.coverageRatioByTime) > 0
    assert 0 <= report.globalCoverageRatio <= 1
