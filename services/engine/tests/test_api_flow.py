from __future__ import annotations

import time

from fastapi.testclient import TestClient

from geologic_wizard_engine.api import create_app
from geologic_wizard_engine.settings import Settings


def _wait_for_job(client: TestClient, job_id: str, timeout_s: float = 30.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"completed", "failed", "canceled"}:
            return payload
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish in time")


def test_end_to_end_project_flow(tmp_path):
    app = create_app(Settings(data_root=tmp_path))
    client = TestClient(app)

    create = client.post(
        "/v1/projects",
        json={
            "name": "Flow",
            "config": {
                "seed": 99,
                "startTimeMa": 100,
                "endTimeMa": 0,
                "stepMyr": 1,
                "plateCount": 8,
            },
        },
    )
    create.raise_for_status()
    project = create.json()

    gen = client.post(f"/v1/projects/{project['projectId']}/generate", json={"runLabel": "test"})
    gen.raise_for_status()
    gen_job = _wait_for_job(client, gen.json()["jobId"])
    assert gen_job["status"] == "completed"

    frame = client.get(f"/v1/projects/{project['projectId']}/frames/50")
    frame.raise_for_status()
    assert frame.json()["frame"]["timeMa"] == 50

    bookmark = client.post(
        f"/v1/projects/{project['projectId']}/bookmarks",
        json={"timeMa": 50, "label": "Midpoint"},
    )
    bookmark.raise_for_status()
    bookmark_id = bookmark.json()["bookmarkId"]

    refine = client.post(
        f"/v1/projects/{project['projectId']}/bookmarks/{bookmark_id}/refine",
        json={"resolution": "2k", "refinementLevel": 1},
    )
    refine.raise_for_status()
    refine_job = _wait_for_job(client, refine.json()["jobId"])
    assert refine_job["status"] == "completed"

    export = client.post(
        f"/v1/projects/{project['projectId']}/exports",
        json={
            "bookmarkId": bookmark_id,
            "format": "png16",
            "width": 1024,
            "height": 512,
            "bitDepth": 16,
        },
    )
    export.raise_for_status()
    export_job = _wait_for_job(client, export.json()["jobId"])
    assert export_job["status"] == "completed"
    assert len(export_job["artifacts"]) >= 1

    validation = client.get(f"/v1/projects/{project['projectId']}/validation")
    validation.raise_for_status()
    assert validation.json()["projectId"] == project["projectId"]
