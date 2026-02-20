from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .job_manager import JobManager
from .metadata_store import MetadataStore
from .models import (
    Bookmark,
    BookmarkCreateRequest,
    BookmarkRefineRequest,
    CoverageReport,
    ExpertEditRequest,
    ExportRequest,
    FieldSampleResponse,
    FrameDiagnostics,
    FrameRangeResponse,
    FrameSummary,
    GenerateRequest,
    JobStatus,
    JobSummary,
    ModuleStateResponse,
    PlausibilityReport,
    ProjectConfig,
    ProjectCreateRequest,
    ProjectSummary,
    RunMetricsResponse,
    SolverVersion,
    TimelineIndex,
    ValidationReport,
)
from .project_store import ProjectStore
from .settings import Settings, load_settings
from .simulation_service import SimulationService
from .utils import stable_hash


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    metadata = MetadataStore(settings.data_root / "metadata.sqlite3")
    project_store = ProjectStore(settings.data_root)
    simulation = SimulationService(settings, metadata, project_store)
    jobs = JobManager(metadata)

    app = FastAPI(title="Geologic Wizard Engine", version="0.2.0")
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
        return metadata.create_project(
            project_id=project_id,
            name=request.name,
            config=request.config,
            project_hash=stable_hash(request.config.model_dump(mode="json")),
            created_at=simulation.utc_now_iso(),
        )

    @app.post("/v2/projects", response_model=ProjectSummary)
    def create_project_v2(request: ProjectCreateRequest) -> ProjectSummary:
        payload = request.config.model_dump(mode="json")
        payload["solverVersion"] = SolverVersion.tectonic_state_v2.value
        if payload.get("coreGridWidth") is None:
            payload["coreGridWidth"] = 720 if payload.get("simulationMode") == "hybrid_rigor" else 512
        if payload.get("coreGridHeight") is None:
            payload["coreGridHeight"] = 360 if payload.get("simulationMode") == "hybrid_rigor" else 256
        config = ProjectConfig.model_validate(payload)

        project_id = str(uuid.uuid4())
        project_store.initialize_project(project_id, config.model_dump(mode="json"))
        return metadata.create_project(
            project_id=project_id,
            name=request.name,
            config=config,
            project_hash=stable_hash(config.model_dump(mode="json")),
            created_at=simulation.utc_now_iso(),
        )

    @app.get("/v1/projects/{project_id}", response_model=ProjectSummary)
    def get_project(project_id: str) -> ProjectSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return project

    @app.get("/v2/projects/{project_id}", response_model=ProjectSummary)
    def get_project_v2(project_id: str) -> ProjectSummary:
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
                simulation_mode_override=request.simulationModeOverride.value if request.simulationModeOverride else None,
                rigor_profile_override=request.rigorProfileOverride.value if request.rigorProfileOverride else None,
                runtime_override=request.targetRuntimeMinutesOverride,
                quality_mode=request.qualityMode,
                source_quick_run_id=request.sourceQuickRunId,
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

    @app.post("/v2/projects/{project_id}/generate", response_model=JobSummary)
    def generate_project_v2(project_id: str, request: GenerateRequest) -> JobSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")

        job = jobs.create_job(project_id, "generate", message=f"queued v2 run {request.runLabel}")

        def run(job_id: str) -> None:
            run_id = simulation.generate_project(
                project_id,
                simulation_mode_override=request.simulationModeOverride.value if request.simulationModeOverride else None,
                rigor_profile_override=request.rigorProfileOverride.value if request.rigorProfileOverride else None,
                runtime_override=request.targetRuntimeMinutesOverride,
                quality_mode=request.qualityMode,
                source_quick_run_id=request.sourceQuickRunId,
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

    @app.get("/v1/projects/{project_id}/timeline-index", response_model=TimelineIndex)
    def get_timeline_index(project_id: str) -> TimelineIndex:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return simulation.get_timeline_index(project)

    @app.get("/v2/projects/{project_id}/timeline-index", response_model=TimelineIndex)
    def get_timeline_index_v2(project_id: str) -> TimelineIndex:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return simulation.get_timeline_index(project)

    @app.get("/v1/projects/{project_id}/frames", response_model=FrameRangeResponse)
    def get_frames_range(
        project_id: str,
        time_from: int = Query(...),
        time_to: int = Query(...),
        step: int = Query(1, ge=1),
        detail: str = Query("render"),
        exact: bool = Query(False),
    ) -> FrameRangeResponse:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        start = project.config.startTimeMa
        end = project.config.endTimeMa
        for value in (time_from, time_to):
            if value < end or value > start:
                raise HTTPException(status_code=400, detail="time out of project range")
        try:
            return simulation.get_frame_range(
                project,
                time_from=time_from,
                time_to=time_to,
                step=step,
                detail=detail,
                exact=exact,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v2/projects/{project_id}/frames", response_model=FrameRangeResponse)
    def get_frames_range_v2(
        project_id: str,
        time_from: int = Query(...),
        time_to: int = Query(...),
        step: int = Query(1, ge=1),
        detail: str = Query("render"),
        exact: bool = Query(False),
    ) -> FrameRangeResponse:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        start = project.config.startTimeMa
        end = project.config.endTimeMa
        for value in (time_from, time_to):
            if value < end or value > start:
                raise HTTPException(status_code=400, detail="time out of project range")
        try:
            return simulation.get_frame_range(
                project,
                time_from=time_from,
                time_to=time_to,
                step=step,
                detail=detail,
                exact=exact,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/projects/{project_id}/frames/{time_ma}", response_model=FrameSummary)
    def get_frame(project_id: str, time_ma: int) -> FrameSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        if time_ma < project.config.endTimeMa or time_ma > project.config.startTimeMa:
            raise HTTPException(status_code=400, detail="time out of project range")
        return simulation.get_or_create_frame(project, time_ma)

    @app.get("/v2/projects/{project_id}/frames/{time_ma}", response_model=FrameSummary)
    def get_frame_v2(project_id: str, time_ma: int) -> FrameSummary:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        if time_ma < project.config.endTimeMa or time_ma > project.config.startTimeMa:
            raise HTTPException(status_code=400, detail="time out of project range")
        return simulation.get_or_create_frame(project, time_ma)

    @app.get("/v1/projects/{project_id}/frames/{time_ma}/diagnostics", response_model=FrameDiagnostics)
    def get_frame_diagnostics(project_id: str, time_ma: int) -> FrameDiagnostics:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        try:
            return simulation.get_frame_diagnostics(project_id, time_ma)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v2/projects/{project_id}/frames/{time_ma}/diagnostics", response_model=FrameDiagnostics)
    def get_frame_diagnostics_v2(project_id: str, time_ma: int) -> FrameDiagnostics:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        try:
            return simulation.get_frame_diagnostics(project_id, time_ma)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v2/projects/{project_id}/frames/{time_ma}/fields/{field_name}", response_model=FieldSampleResponse)
    def get_frame_field_v2(
        project_id: str,
        time_ma: int,
        field_name: str,
        max_dim: int = Query(256, ge=32, le=1024),
    ) -> FieldSampleResponse:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        try:
            return simulation.get_field_sample(project_id, time_ma, field_name, max_dim=max_dim)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v2/projects/{project_id}/frames/{time_ma}/module-states", response_model=ModuleStateResponse)
    def get_module_states_v2(project_id: str, time_ma: int) -> ModuleStateResponse:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        try:
            return simulation.get_module_state(project_id, time_ma)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v2/projects/{project_id}/runs/{run_id}/metrics", response_model=RunMetricsResponse)
    def get_run_metrics_v2(project_id: str, run_id: str) -> RunMetricsResponse:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        try:
            return simulation.get_run_metrics(project_id, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v2/projects/{project_id}/plausibility", response_model=PlausibilityReport)
    def get_plausibility_v2(project_id: str) -> PlausibilityReport:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return simulation.get_plausibility_report(project_id)

    @app.get("/v1/projects/{project_id}/coverage", response_model=CoverageReport)
    def get_coverage(project_id: str) -> CoverageReport:
        project = metadata.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return simulation.get_coverage_report(project_id)

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
                    created_at=simulation.utc_now_iso(),
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
