from __future__ import annotations

import time

from fastapi.testclient import TestClient

from geologic_wizard_engine.api import create_app
from geologic_wizard_engine.metadata_store import MetadataStore
from geologic_wizard_engine.models import (
    BoundaryStateClass,
    ProjectConfig,
    QualityMode,
    SolverVersion,
)
from geologic_wizard_engine.modules.tectonics_v2.boundary_machine import update_boundary_state
from geologic_wizard_engine.modules.tectonics_v2.state import BoundaryStateV2
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


def _wait_for_job(client: TestClient, job_id: str, timeout_s: float = 45.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"completed", "failed", "canceled"}:
            return payload
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish in time")


def test_v2_generation_includes_lifecycle_and_boundary_state(tmp_path):
    service, metadata, project_store = _build_service(tmp_path)

    config = ProjectConfig(
        seed=4242,
        plateCount=10,
        startTimeMa=120,
        endTimeMa=0,
        timeIncrementMyr=2,
        stepMyr=2,
        simulationMode="hybrid_rigor",
        rigorProfile="research",
        solverVersion=SolverVersion.tectonic_state_v2,
    )
    metadata.create_project(
        project_id="v2p",
        name="V2 Determinism",
        config=config,
        project_hash=stable_hash(config.model_dump(mode="json")),
        created_at="2026-01-01T00:00:00+00:00",
    )
    project_store.initialize_project("v2p", config.model_dump(mode="json"))

    service.generate_project(
        "v2p",
        simulation_mode_override=None,
        rigor_profile_override=None,
        runtime_override=None,
        job_callback=lambda *_args: None,
        is_canceled=lambda: False,
    )

    project = metadata.get_project("v2p")
    assert project is not None

    frame_a = service.get_or_create_frame(project, 60)
    frame_b = service.get_or_create_frame(project, 60)

    assert frame_a.frameHash == frame_b.frameHash
    assert frame_a.frame.boundaryStates
    assert frame_a.frame.plateLifecycleState is not None
    assert frame_a.frame.plateLifecycleState.netAreaBalanceError <= 0.01
    assert frame_a.frame.oceanicAgeFieldRef
    assert frame_a.frame.crustTypeFieldRef
    assert frame_a.frame.crustThicknessFieldRef
    assert frame_a.frame.tectonicPotentialFieldRef

    report = service.get_plausibility_report("v2p")
    check_ids = {check.checkId for check in report.checks}
    assert "boundary.motion_mismatch_count" in check_ids
    assert "lifecycle.net_area_balance_error" in check_ids


def test_boundary_state_machine_thresholds():
    boundary = BoundaryStateV2(segment_id="seg_1_2", left_plate_id=1, right_plate_id=2)

    for time_ma in range(100, 92, -1):
        boundary, _ = update_boundary_state(
            boundary=boundary,
            normal_velocity_cm_yr=0.8,
            tangential_velocity_cm_yr=0.3,
            relative_velocity_cm_yr=1.2,
            average_oceanic_age_myr=5.0,
            left_plate_is_continental=True,
            right_plate_is_continental=False,
            step_myr=1,
            time_ma=time_ma,
        )
    assert boundary.state_class in {BoundaryStateClass.rift, BoundaryStateClass.ridge}

    for time_ma in range(92, 87, -1):
        boundary, _ = update_boundary_state(
            boundary=boundary,
            normal_velocity_cm_yr=-1.4,
            tangential_velocity_cm_yr=0.2,
            relative_velocity_cm_yr=1.7,
            average_oceanic_age_myr=45.0,
            left_plate_is_continental=False,
            right_plate_is_continental=True,
            step_myr=1,
            time_ma=time_ma,
        )
    assert boundary.state_class == BoundaryStateClass.subduction

    for time_ma in range(87, 83, -1):
        boundary, _ = update_boundary_state(
            boundary=boundary,
            normal_velocity_cm_yr=-0.9,
            tangential_velocity_cm_yr=0.3,
            relative_velocity_cm_yr=1.2,
            average_oceanic_age_myr=20.0,
            left_plate_is_continental=True,
            right_plate_is_continental=True,
            step_myr=1,
            time_ma=time_ma,
        )
    assert boundary.state_class in {BoundaryStateClass.collision, BoundaryStateClass.suture}


def test_v2_api_endpoints(tmp_path):
    app = create_app(Settings(data_root=tmp_path))
    client = TestClient(app)

    create = client.post(
        "/v2/projects",
        json={
            "name": "V2 Flow",
            "config": {
                "seed": 123,
                "startTimeMa": 80,
                "endTimeMa": 0,
                "stepMyr": 1,
                "timeIncrementMyr": 1,
                "plateCount": 8,
                "simulationMode": "fast_plausible",
                "rigorProfile": "balanced"
            },
        },
    )
    create.raise_for_status()
    project = create.json()
    assert project["config"]["solverVersion"] == "tectonic_state_v2"

    generate = client.post(
        f"/v2/projects/{project['projectId']}/generate",
        json={"runLabel": "v2"},
    )
    generate.raise_for_status()
    job = _wait_for_job(client, generate.json()["jobId"])
    assert job["status"] == "completed"

    timeline = client.get(f"/v2/projects/{project['projectId']}/timeline-index")
    timeline.raise_for_status()
    assert timeline.json()["projectId"] == project["projectId"]

    frame = client.get(f"/v2/projects/{project['projectId']}/frames/40")
    frame.raise_for_status()
    frame_payload = frame.json()["frame"]
    assert frame_payload["boundaryStates"]
    assert frame_payload["plateLifecycleState"] is not None

    diagnostics = client.get(f"/v2/projects/{project['projectId']}/frames/40/diagnostics")
    diagnostics.raise_for_status()
    assert "checkIds" in diagnostics.json()

    plausibility = client.get(f"/v2/projects/{project['projectId']}/plausibility")
    plausibility.raise_for_status()
    assert plausibility.json()["projectId"] == project["projectId"]
    assert len(plausibility.json()["checks"]) > 0


def test_full_run_reuses_quick_macro_history(tmp_path):
    service, metadata, project_store = _build_service(tmp_path)

    config = ProjectConfig(
        seed=2048,
        plateCount=10,
        startTimeMa=120,
        endTimeMa=0,
        timeIncrementMyr=2,
        stepMyr=2,
        simulationMode="hybrid_rigor",
        rigorProfile="research",
        solverVersion=SolverVersion.tectonic_state_v2,
    )
    metadata.create_project(
        project_id="macro",
        name="Macro Coupling",
        config=config,
        project_hash=stable_hash(config.model_dump(mode="json")),
        created_at="2026-01-01T00:00:00+00:00",
    )
    project_store.initialize_project("macro", config.model_dump(mode="json"))

    service.generate_project(
        "macro",
        simulation_mode_override=None,
        rigor_profile_override=None,
        runtime_override=None,
        quality_mode=QualityMode.quick,
        source_quick_run_id=None,
        job_callback=lambda *_args: None,
        is_canceled=lambda: False,
    )
    project_after_quick = metadata.get_project("macro")
    assert project_after_quick is not None
    assert project_after_quick.currentRunId is not None
    quick_run_id = project_after_quick.currentRunId
    assert project_after_quick.latestQuickRunId == quick_run_id

    service.generate_project(
        "macro",
        simulation_mode_override=None,
        rigor_profile_override=None,
        runtime_override=None,
        quality_mode=QualityMode.full,
        source_quick_run_id=quick_run_id,
        job_callback=lambda *_args: None,
        is_canceled=lambda: False,
    )
    project_after_full = metadata.get_project("macro")
    assert project_after_full is not None
    assert project_after_full.currentRunId is not None
    full_run_id = project_after_full.currentRunId
    assert full_run_id != quick_run_id
    assert project_after_full.latestQuickRunId == quick_run_id
    assert project_after_full.latestFullRunId == full_run_id

    quick_manifest = project_store.read_json(project_store.run_manifest_path("macro", quick_run_id))
    full_manifest = project_store.read_json(project_store.run_manifest_path("macro", full_run_id))
    assert quick_manifest["qualityMode"] == "quick"
    assert full_manifest["qualityMode"] == "full"
    assert full_manifest["sourceQuickRunId"] == quick_run_id
    assert full_manifest["macroDigest"] == quick_manifest["macroDigest"]

    sample_time = 60
    quick_frame = project_store.read_frame("macro", quick_run_id, sample_time)
    full_frame = project_store.read_frame("macro", full_run_id, sample_time)
    assert quick_frame is not None and full_frame is not None
    assert quick_frame["plateGeometries"] == full_frame["plateGeometries"]
    assert quick_frame["boundaryGeometries"] == full_frame["boundaryGeometries"]
    assert quick_frame["eventOverlays"] == full_frame["eventOverlays"]
    assert full_frame["previewHeightFieldRef"]
    assert quick_run_id in quick_frame["previewHeightFieldRef"]
    assert full_run_id in full_frame["previewHeightFieldRef"]


def test_render_payload_includes_coastline_and_active_belts(tmp_path):
    app = create_app(Settings(data_root=tmp_path))
    client = TestClient(app)

    create = client.post(
        "/v2/projects",
        json={
            "name": "Render Surface",
            "config": {
                "seed": 91,
                "startTimeMa": 60,
                "endTimeMa": 0,
                "stepMyr": 1,
                "timeIncrementMyr": 1,
                "plateCount": 8,
                "simulationMode": "fast_plausible",
                "rigorProfile": "balanced"
            },
        },
    )
    create.raise_for_status()
    project = create.json()

    generate = client.post(
        f"/v2/projects/{project['projectId']}/generate",
        json={"runLabel": "quick", "qualityMode": "quick"},
    )
    generate.raise_for_status()
    job = _wait_for_job(client, generate.json()["jobId"])
    assert job["status"] == "completed"

    render_frame = client.get(
        f"/v2/projects/{project['projectId']}/frames",
        params={
            "time_from": 40,
            "time_to": 40,
            "step": 1,
            "detail": "render",
            "exact": True,
        },
    )
    render_frame.raise_for_status()
    payload = render_frame.json()["renderFrames"][0]
    assert "coastlineGeoJson" in payload
    assert "activeBeltsGeoJson" in payload
