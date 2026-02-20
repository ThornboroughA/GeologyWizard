from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .job_manager import JobManager
from .metadata_store import MetadataStore
from .models import (
    Bookmark,
    BookmarkCreateRequest,
    BookmarkRefineRequest,
    ExpertEditRequest,
    ExportRequest,
    FrameSummary,
    GenerateRequest,
    JobStatus,
    JobSummary,
    ProjectCreateRequest,
    ProjectSummary,
    ValidationReport,
)
from .project_store import ProjectStore
from .settings import Settings, load_settings
from .simulation_service import SimulationService
from .utils import stable_hash


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    metadata = MetadataStore(settings.data_root / "metadata.sqlite3")
    project_store = ProjectStore(settings.data_root)
    simulation = SimulationService(settings, metadata, project_store)
    jobs = JobManager(metadata)

    app = FastAPI(title="Geologic Wizard Engine", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.metadata = metadata
    app.state.project_store = project_store
    app.state.simulation = simulation
    app.state.jobs = jobs

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/projects", response_model=ProjectSummary)
    def create_project(request: ProjectCreateRequest) -> ProjectSummary:
        project_id = str(uuid.uuid4())
        project_store.initialize_project(project_id, request.config.model_dump(mode="json"))
        project_hash = stable_hash(request.config.model_dump(mode="json"))
        return metadata.create_project(
            project_id=project_id,
            name=request.name,
            config=request.config,
            project_hash=project_hash,
            created_at=_utc_now_iso(),
        )

    @app.get("/v1/projects/{project_id}", response_model=ProjectSummary)
    def get_project(project_id: str) -> ProjectSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return project

    @app.post("/v1/projects/{project_id}/generate", response_model=JobSummary)
    def generate_project(project_id: str, request: GenerateRequest) -> JobSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")

        job = jobs.create_job(project_id, "generate", message=f"queued run {request.runLabel}")

        def run(job_id: str) -> None:
            run_id = simulation.generate_project(
                project_id,
                job_callback=lambda progress, message: jobs.set_state(job_id, progress=progress, message=message),
                is_canceled=lambda: jobs.is_canceled(job_id),
            )
            if jobs.is_canceled(job_id):
                jobs.set_state(job_id, status=JobStatus.canceled, message="generation canceled")
                return
            jobs.set_state(job_id, message=f"run {run_id} cached")

        jobs.submit(job.jobId, run)
        latest = jobs.get_job(job.jobId)
        if latest is None:
            raise HTTPException(status_code=500, detail="job not available")
        return latest

    @app.get("/v1/projects/{project_id}/frames/{time_ma}", response_model=FrameSummary)
    def get_frame(project_id: str, time_ma: int) -> FrameSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        if time_ma < project.config.endTimeMa or time_ma > project.config.startTimeMa:
            raise HTTPException(status_code=400, detail="time out of project range")
        return simulation.get_or_create_frame(project, time_ma)

    @app.post("/v1/projects/{project_id}/bookmarks", response_model=Bookmark)
    def create_bookmark(project_id: str, request: BookmarkCreateRequest) -> Bookmark:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return simulation.create_bookmark(project, request.timeMa, request.label, request.region)

    @app.get("/v1/projects/{project_id}/bookmarks", response_model=list[Bookmark])
    def list_bookmarks(project_id: str) -> list[Bookmark]:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return [Bookmark.model_validate(row) for row in metadata.list_bookmarks(project_id)]

    @app.post("/v1/projects/{project_id}/bookmarks/{bookmark_id}/refine", response_model=JobSummary)
    def refine_bookmark(project_id: str, bookmark_id: str, request: BookmarkRefineRequest) -> JobSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")

        job = jobs.create_job(project_id, "refine", message=f"queued bookmark {bookmark_id}")

        def run(job_id: str) -> None:
            simulation.refine_bookmark(
                project_id,
                bookmark_id,
                request,
                job_callback=lambda progress, message: jobs.set_state(job_id, progress=progress, message=message),
                is_canceled=lambda: jobs.is_canceled(job_id),
            )

        jobs.submit(job.jobId, run)
        latest = jobs.get_job(job.jobId)
        if latest is None:
            raise HTTPException(status_code=500, detail="job not available")
        return latest

    @app.post("/v1/projects/{project_id}/edits")
    def apply_edits(project_id: str, request: ExpertEditRequest) -> dict:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return simulation.add_expert_edits(
            project_id,
            [edit.model_dump(mode="json") for edit in request.edits],
        )

    @app.post("/v1/projects/{project_id}/exports", response_model=JobSummary)
    def export_heightmap(project_id: str, request: ExportRequest) -> JobSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")

        job = jobs.create_job(project_id, "export", message="queued export")

        def run(job_id: str) -> None:
            result = simulation.export_heightmap(
                project_id,
                request,
                job_callback=lambda progress, message: jobs.set_state(job_id, progress=progress, message=message),
                is_canceled=lambda: jobs.is_canceled(job_id),
            )
            jobs.set_state(job_id, artifacts=[artifact.model_dump(mode="json") for artifact in result.artifacts])
            for artifact in result.artifacts:
                metadata.add_export(
                    export_id=str(uuid.uuid4()),
                    project_id=project_id,
                    job_id=job_id,
                    artifact=artifact.model_dump(mode="json"),
                    created_at=_utc_now_iso(),
                )

        jobs.submit(job.jobId, run)
        latest = jobs.get_job(job.jobId)
        if latest is None:
            raise HTTPException(status_code=500, detail="job not available")
        return latest

    @app.get("/v1/projects/{project_id}/validation", response_model=ValidationReport)
    def validate_project(project_id: str) -> ValidationReport:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return simulation.validate_project(project_id)

    @app.get("/v1/jobs/{job_id}", response_model=JobSummary)
    def get_job(job_id: str) -> JobSummary:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobSummary)
    def cancel_job(job_id: str) -> JobSummary:
        ok = jobs.cancel(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail="job not found")
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/v1/jobs/{job_id}/events")
    async def stream_job(job_id: str) -> StreamingResponse:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def event_gen() -> AsyncGenerator[str, None]:
            last_version = -1
            while True:
                current = jobs.get_job(job_id)
                if current is None:
                    yield "event: error\ndata: {\"message\":\"job not found\"}\n\n"
                    return
                version = jobs.job_version(job_id)
                if version != last_version:
                    last_version = version
                    payload = current.model_dump(mode="json")
                    yield f"event: update\ndata: {json.dumps(payload)}\n\n"
                if current.status.value in ("completed", "failed", "canceled"):
                    return
                await asyncio.sleep(0.4)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app
