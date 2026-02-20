from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import JobStatus, JobSummary, ProjectConfig, ProjectSummary
from .utils import ensure_dir


class MetadataStore:
    def __init__(self, db_path: Path):
        ensure_dir(db_path.parent)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    project_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    current_run_id TEXT
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bookmarks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    time_ma INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    region_json TEXT,
                    parent_frame_hash TEXT NOT NULL,
                    refinement_state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS edits (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    time_ma INTEGER NOT NULL,
                    edit_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS exports (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    artifact_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_project(
        self,
        project_id: str,
        name: str,
        config: ProjectConfig,
        project_hash: str,
        created_at: str,
    ) -> ProjectSummary:
        payload = (project_id, name, config.model_dump_json(), project_hash, created_at, created_at, None)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects(id, name, config_json, project_hash, created_at, updated_at, current_run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return ProjectSummary(
            projectId=project_id,
            name=name,
            config=config,
            createdAt=created_at,
            updatedAt=created_at,
            projectHash=project_hash,
            currentRunId=None,
        )

    def get_project(self, project_id: str) -> ProjectSummary | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            return None
        return self._project_from_row(row)

    def _project_from_row(self, row: sqlite3.Row) -> ProjectSummary:
        config = ProjectConfig.model_validate_json(row["config_json"])
        return ProjectSummary(
            projectId=row["id"],
            name=row["name"],
            config=config,
            createdAt=row["created_at"],
            updatedAt=row["updated_at"],
            projectHash=row["project_hash"],
            currentRunId=row["current_run_id"],
        )

    def set_project_run(self, project_id: str, run_id: str, updated_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE projects SET current_run_id = ?, updated_at = ? WHERE id = ?",
                (run_id, updated_at, project_id),
            )

    def create_job(self, job: JobSummary, created_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(id, project_id, kind, status, progress, message, artifacts_json, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.jobId,
                    job.projectId,
                    job.kind,
                    job.status.value,
                    job.progress,
                    job.message,
                    json.dumps([artifact.model_dump() for artifact in job.artifacts]),
                    job.error,
                    created_at,
                    created_at,
                ),
            )

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus,
        progress: float,
        message: str,
        artifacts: list[dict[str, Any]],
        error: str | None,
        updated_at: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, progress = ?, message = ?, artifacts_json = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, progress, message, json.dumps(artifacts), error, updated_at, job_id),
            )

    def get_job(self, job_id: str) -> JobSummary | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobSummary.model_validate(
            {
                "jobId": row["id"],
                "projectId": row["project_id"],
                "kind": row["kind"],
                "status": row["status"],
                "progress": row["progress"],
                "message": row["message"],
                "artifacts": json.loads(row["artifacts_json"]),
                "error": row["error"],
            }
        )

    def create_bookmark(
        self,
        *,
        bookmark_id: str,
        project_id: str,
        time_ma: int,
        label: str,
        region: dict[str, Any] | None,
        parent_frame_hash: str,
        refinement_state: str,
        created_at: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bookmarks(id, project_id, time_ma, label, region_json, parent_frame_hash, refinement_state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bookmark_id,
                    project_id,
                    time_ma,
                    label,
                    json.dumps(region) if region is not None else None,
                    parent_frame_hash,
                    refinement_state,
                    created_at,
                ),
            )

    def list_bookmarks(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM bookmarks WHERE project_id = ? ORDER BY time_ma DESC", (project_id,)
            ).fetchall()
        bookmarks: list[dict[str, Any]] = []
        for row in rows:
            bookmarks.append(
                {
                    "bookmarkId": row["id"],
                    "timeMa": row["time_ma"],
                    "label": row["label"],
                    "region": json.loads(row["region_json"]) if row["region_json"] else None,
                    "parentFrameHash": row["parent_frame_hash"],
                    "refinementState": row["refinement_state"],
                }
            )
        return bookmarks

    def get_bookmark(self, bookmark_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM bookmarks WHERE id = ?", (bookmark_id,)).fetchone()
        if row is None:
            return None
        return {
            "bookmarkId": row["id"],
            "projectId": row["project_id"],
            "timeMa": row["time_ma"],
            "label": row["label"],
            "region": json.loads(row["region_json"]) if row["region_json"] else None,
            "parentFrameHash": row["parent_frame_hash"],
            "refinementState": row["refinement_state"],
        }

    def update_bookmark_refinement(self, bookmark_id: str, state: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE bookmarks SET refinement_state = ? WHERE id = ?",
                (state, bookmark_id),
            )

    def add_edit(
        self,
        *,
        edit_id: str,
        project_id: str,
        time_ma: int,
        edit_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO edits(id, project_id, time_ma, edit_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (edit_id, project_id, time_ma, edit_type, json.dumps(payload), created_at),
            )

    def list_edits(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edits WHERE project_id = ? ORDER BY created_at ASC", (project_id,)
            ).fetchall()
        edits: list[dict[str, Any]] = []
        for row in rows:
            edits.append(
                {
                    "editId": row["id"],
                    "timeMa": row["time_ma"],
                    "editType": row["edit_type"],
                    "payload": json.loads(row["payload_json"]),
                }
            )
        return edits

    def add_export(self, *, export_id: str, project_id: str, job_id: str, artifact: dict[str, Any], created_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO exports(id, project_id, job_id, artifact_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (export_id, project_id, job_id, json.dumps(artifact), created_at),
            )
