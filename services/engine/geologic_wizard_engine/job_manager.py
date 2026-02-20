from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .metadata_store import MetadataStore
from .models import ExportArtifact, JobStatus, JobSummary


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRuntimeState:
    summary: JobSummary
    version: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobManager:
    def __init__(self, metadata_store: MetadataStore, max_workers: int = 4):
        self._metadata_store = metadata_store
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gw-job")
        self._states: dict[str, JobRuntimeState] = {}
        self._lock = threading.Lock()

    def create_job(self, project_id: str, kind: str, message: str = "queued") -> JobSummary:
        job_id = str(uuid.uuid4())
        job = JobSummary(
            jobId=job_id,
            projectId=project_id,
            kind=kind,
            status=JobStatus.queued,
            progress=0.0,
            message=message,
            artifacts=[],
            error=None,
        )
        with self._lock:
            self._states[job_id] = JobRuntimeState(summary=job)
        self._metadata_store.create_job(job, _utc_now_iso())
        return job

    def submit(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.set_state(job_id, status=JobStatus.running, message="running", progress=0.0)

        def _runner() -> None:
            try:
                fn(job_id, *args, **kwargs)
                current = self.get_job(job_id)
                if current and current.status != JobStatus.canceled and current.status != JobStatus.failed:
                    self.set_state(job_id, status=JobStatus.completed, progress=1.0, message="completed")
            except Exception as exc:  # pragma: no cover - defensive path
                self.set_state(
                    job_id,
                    status=JobStatus.failed,
                    progress=1.0,
                    message="failed",
                    error=str(exc),
                )

        self._executor.submit(_runner)

    def get_job(self, job_id: str) -> JobSummary | None:
        with self._lock:
            state = self._states.get(job_id)
            if state is not None:
                return state.summary
        return self._metadata_store.get_job(job_id)

    def is_canceled(self, job_id: str) -> bool:
        with self._lock:
            state = self._states.get(job_id)
            return state.cancel_event.is_set() if state else False

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            state = self._states.get(job_id)
            if state is None:
                return False
            state.cancel_event.set()
        self.set_state(job_id, status=JobStatus.canceled, message="canceled")
        return True

    def set_state(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        artifacts: list[dict[str, Any] | ExportArtifact] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            state = self._states.get(job_id)
            if state is None:
                return
            summary = state.summary
            if status is not None:
                summary.status = status
            if progress is not None:
                summary.progress = max(0.0, min(1.0, progress))
            if message is not None:
                summary.message = message
            if artifacts is not None:
                normalized: list[ExportArtifact] = []
                for artifact in artifacts:
                    if isinstance(artifact, ExportArtifact):
                        normalized.append(artifact)
                    else:
                        normalized.append(ExportArtifact.model_validate(artifact))
                summary.artifacts = normalized
            if error is not None:
                summary.error = error
            state.version += 1

            serialized_artifacts = [artifact.model_dump(mode="json") for artifact in summary.artifacts]

        self._metadata_store.update_job(
            job_id,
            status=summary.status,
            progress=summary.progress,
            message=summary.message,
            artifacts=serialized_artifacts,
            error=summary.error,
            updated_at=_utc_now_iso(),
        )

    def job_version(self, job_id: str) -> int:
        with self._lock:
            state = self._states.get(job_id)
            return state.version if state else 0
