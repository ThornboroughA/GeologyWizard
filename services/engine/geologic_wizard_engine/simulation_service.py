from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .metadata_store import MetadataStore
from .models import (
    Bookmark,
    BookmarkRefineRequest,
    ExportArtifact,
    ExportRequest,
    ExportResult,
    FrameSummary,
    ProjectSummary,
    ProvenanceRecord,
    RefineResult,
    TimelineFrame,
    ValidationIssue,
    ValidationReport,
)
from .modules.tectonics import build_boundary_segments, build_geo_events, build_plate_features
from .modules.terrain_synthesis import synthesize_preview_height, synthesize_refined_region
from .modules.validation import validate_frame
from .project_store import ProjectStore
from .settings import Settings
from .utils import sha256_bytes, stable_hash

ENGINE_VERSION = "0.1.0"
MODEL_VERSION = "tectonic-kinematic-rules-v1"


class SimulationService:
    def __init__(self, settings: Settings, metadata: MetadataStore, project_store: ProjectStore):
        self.settings = settings
        self.metadata = metadata
        self.project_store = project_store

    def utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def compute_project_hash(self, project: ProjectSummary) -> str:
        return stable_hash(project.config.model_dump())

    def compute_frame(self, project: ProjectSummary, time_ma: int) -> TimelineFrame:
        edits = self.metadata.list_edits(project.projectId)
        plates = build_plate_features(project.config, time_ma)
        boundaries = build_boundary_segments(plates, time_ma)
        events = build_geo_events(boundaries, time_ma, edits=edits)
        preview_ref = str(self.project_store.preview_array_path(project.projectId, time_ma))
        return TimelineFrame(
            timeMa=time_ma,
            plateGeometries=plates,
            boundaryGeometries=boundaries,
            eventOverlays=events,
            previewHeightFieldRef=preview_ref,
        )

    def _time_steps(self, start_time_ma: int, end_time_ma: int, step_myr: int) -> list[int]:
        values: list[int] = []
        current = start_time_ma
        while current >= end_time_ma:
            values.append(current)
            current -= step_myr
        return values

    def generate_project(self, project_id: str, *, job_callback: Any, is_canceled: Any) -> str:
        project = self.metadata.get_project(project_id)
        if project is None:
            raise ValueError(f"project {project_id} not found")

        run_id = str(uuid.uuid4())
        self.metadata.set_project_run(project_id, run_id, self.utc_now_iso())

        steps = self._time_steps(project.config.startTimeMa, project.config.endTimeMa, project.config.stepMyr)
        total = len(steps)

        manifest = {
            "runId": run_id,
            "generatedAt": self.utc_now_iso(),
            "keyframeIntervalMyr": self.settings.keyframe_interval_myr,
            "frames": [],
        }

        for idx, time_ma in enumerate(steps, start=1):
            if is_canceled():
                return run_id

            should_cache_keyframe = (
                time_ma % self.settings.keyframe_interval_myr == 0
                or time_ma == project.config.startTimeMa
                or time_ma == project.config.endTimeMa
            )
            if should_cache_keyframe:
                frame = self.compute_frame(project, time_ma)
                frame_payload = frame.model_dump(mode="json")
                self.project_store.write_frame(project.projectId, run_id, time_ma, frame_payload)

                preview = synthesize_preview_height(
                    time_ma,
                    project.config.seed,
                    frame.plateGeometries,
                    frame.eventOverlays,
                    self.settings.default_preview_width,
                    self.settings.default_preview_height,
                )
                self.project_store.write_array(self.project_store.preview_array_path(project.projectId, time_ma), preview)
                manifest["frames"].append({"timeMa": time_ma, "framePath": str(self.project_store.frame_path(project.projectId, run_id, time_ma))})

            progress = idx / total
            job_callback(progress, f"simulating {time_ma} Ma")

        manifest_path = self.project_store.run_dir(project.projectId, run_id) / "manifest.json"
        self.project_store.write_json(manifest_path, manifest)
        return run_id

    def get_or_create_frame(self, project: ProjectSummary, time_ma: int) -> FrameSummary:
        run_id = project.currentRunId
        if run_id:
            cached = self.project_store.read_frame(project.projectId, run_id, time_ma)
            if cached is not None:
                frame_hash = stable_hash(cached)
                return FrameSummary(frame=TimelineFrame.model_validate(cached), frameHash=frame_hash, source="cache")

        frame = self.compute_frame(project, time_ma)
        frame_payload = frame.model_dump(mode="json")

        if run_id:
            self.project_store.write_frame(project.projectId, run_id, time_ma, frame_payload)
        frame_hash = stable_hash(frame_payload)

        preview_path = self.project_store.preview_array_path(project.projectId, time_ma)
        if not preview_path.exists():
            preview = synthesize_preview_height(
                time_ma,
                project.config.seed,
                frame.plateGeometries,
                frame.eventOverlays,
                self.settings.default_preview_width,
                self.settings.default_preview_height,
            )
            self.project_store.write_array(preview_path, preview)

        return FrameSummary(frame=frame, frameHash=frame_hash, source="generated")

    def create_bookmark(self, project: ProjectSummary, time_ma: int, label: str, region: dict[str, Any] | None) -> Bookmark:
        frame_summary = self.get_or_create_frame(project, time_ma)
        bookmark_id = str(uuid.uuid4())
        self.metadata.create_bookmark(
            bookmark_id=bookmark_id,
            project_id=project.projectId,
            time_ma=time_ma,
            label=label,
            region=region,
            parent_frame_hash=frame_summary.frameHash,
            refinement_state="pending",
            created_at=self.utc_now_iso(),
        )
        return Bookmark(
            bookmarkId=bookmark_id,
            timeMa=time_ma,
            label=label,
            region=region,
            refinementState="pending",
            parentFrameHash=frame_summary.frameHash,
        )

    def refine_bookmark(
        self,
        project_id: str,
        bookmark_id: str,
        request: BookmarkRefineRequest,
        *,
        job_callback: Any,
        is_canceled: Any,
    ) -> RefineResult:
        project = self.metadata.get_project(project_id)
        if project is None:
            raise ValueError(f"project {project_id} not found")

        bookmark_data = self.metadata.get_bookmark(bookmark_id)
        if bookmark_data is None:
            raise ValueError(f"bookmark {bookmark_id} not found")

        frame_summary = self.get_or_create_frame(project, bookmark_data["timeMa"])
        preview_path = Path(frame_summary.frame.previewHeightFieldRef)
        preview = self.project_store.read_array(preview_path)

        width, height = self._resolution_dims(request.resolution)
        if is_canceled():
            raise RuntimeError("job canceled")

        job_callback(0.2, "building refined regional terrain")
        refined = synthesize_refined_region(
            preview,
            bookmark_data["region"],
            width,
            height,
            request.refinementLevel,
            project.config.seed,
        )

        if is_canceled():
            raise RuntimeError("job canceled")

        cache_path = self.project_store.refined_array_path(project_id, bookmark_id, request.refinementLevel)
        self.project_store.write_array(cache_path, refined)
        self.metadata.update_bookmark_refinement(bookmark_id, "ready")
        job_callback(1.0, "bookmark refinement completed")

        bookmark = Bookmark.model_validate(
            {
                **bookmark_data,
                "refinementState": "ready",
            }
        )
        return RefineResult(bookmark=bookmark, cachePath=str(cache_path))

    def add_expert_edits(self, project_id: str, edits: list[dict[str, Any]]) -> dict[str, Any]:
        now = self.utc_now_iso()
        impacted_times: set[int] = set()
        for edit in edits:
            edit_id = str(uuid.uuid4())
            time_ma = int(edit["timeMa"])
            self.metadata.add_edit(
                edit_id=edit_id,
                project_id=project_id,
                time_ma=time_ma,
                edit_type=edit["editType"],
                payload=edit.get("payload", {}),
                created_at=now,
            )
            # Scoped invalidation window around edit time.
            for offset in range(-20, 21, self.settings.keyframe_interval_myr):
                candidate = time_ma + offset
                if candidate >= 0:
                    impacted_times.add(candidate)

        return {
            "projectId": project_id,
            "impactedTimesMa": sorted(impacted_times, reverse=True),
            "editCount": len(edits),
        }

    def export_heightmap(
        self,
        project_id: str,
        request: ExportRequest,
        *,
        job_callback: Any,
        is_canceled: Any,
    ) -> ExportResult:
        project = self.metadata.get_project(project_id)
        if project is None:
            raise ValueError(f"project {project_id} not found")

        if request.bookmarkId:
            bookmark_data = self.metadata.get_bookmark(request.bookmarkId)
            if bookmark_data is None:
                raise ValueError(f"bookmark {request.bookmarkId} not found")
            time_ma = int(bookmark_data["timeMa"])
            source_region = bookmark_data["region"]
            refined_path = self.project_store.refined_array_path(project_id, request.bookmarkId, 1)
            if refined_path.exists():
                source = self.project_store.read_array(refined_path)
            else:
                frame_summary = self.get_or_create_frame(project, time_ma)
                source = self.project_store.read_array(Path(frame_summary.frame.previewHeightFieldRef))
                source = synthesize_refined_region(source, source_region, 2048, 1024, 1, project.config.seed)
        else:
            if request.timeMa is None:
                raise ValueError("timeMa is required when bookmarkId is not provided")
            time_ma = int(request.timeMa)
            frame_summary = self.get_or_create_frame(project, time_ma)
            source = self.project_store.read_array(Path(frame_summary.frame.previewHeightFieldRef))

        if is_canceled():
            raise RuntimeError("job canceled")

        job_callback(0.35, "upsampling terrain")
        terrain = self._upsample(source, request.width, request.height)

        if request.region is not None:
            terrain = synthesize_refined_region(terrain, request.region, request.width, request.height, 1, project.config.seed)

        if is_canceled():
            raise RuntimeError("job canceled")

        artifact_id = str(uuid.uuid4())
        suffix = ".png" if request.format == "png16" else ".tiff"
        filename = f"{time_ma}Ma_{artifact_id}{suffix}"
        output_path = self.project_store.export_path(project_id, filename)

        self._write_heightmap(output_path, terrain, request)
        checksum = sha256_bytes(output_path.read_bytes())

        heightmap_artifact = ExportArtifact(
            artifactId=artifact_id,
            type="heightmap",
            format=request.format,
            width=request.width,
            height=request.height,
            bitDepth=request.bitDepth,
            path=str(output_path),
            checksum=checksum,
        )

        provenance = ProvenanceRecord(
            projectHash=project.projectHash,
            seed=project.config.seed,
            engineVersion=ENGINE_VERSION,
            modelVersion=MODEL_VERSION,
            parameterHash=stable_hash(request.model_dump(mode="json")),
            eventHash=stable_hash({"timeMa": time_ma, "bookmarkId": request.bookmarkId}),
        )

        metadata_payload = {
            "provenance": provenance.model_dump(mode="json"),
            "projectId": project.projectId,
            "timeMa": time_ma,
            "bookmarkId": request.bookmarkId,
            "format": request.format,
            "bitDepth": request.bitDepth,
            "dimensions": [request.width, request.height],
        }
        metadata_path = self.project_store.metadata_path(project_id, artifact_id)
        self.project_store.write_json(metadata_path, metadata_payload)
        metadata_checksum = sha256_bytes(metadata_path.read_bytes())

        metadata_artifact = ExportArtifact(
            artifactId=artifact_id,
            type="metadata",
            format="json",
            width=request.width,
            height=request.height,
            bitDepth=request.bitDepth,
            path=str(metadata_path),
            checksum=metadata_checksum,
        )

        job_callback(1.0, "export completed")
        return ExportResult(artifacts=[heightmap_artifact, metadata_artifact], provenance=provenance)

    def validate_project(self, project_id: str) -> ValidationReport:
        project = self.metadata.get_project(project_id)
        if project is None:
            raise ValueError(f"project {project_id} not found")

        issues: list[ValidationIssue] = []

        if project.config.startTimeMa <= project.config.endTimeMa:
            issues.append(
                ValidationIssue(
                    code="invalid_time_range",
                    severity="error",
                    message="startTimeMa must be greater than endTimeMa",
                )
            )

        run_id = project.currentRunId
        if run_id is None:
            issues.append(
                ValidationIssue(
                    code="missing_run",
                    severity="warning",
                    message="No generated run exists for this project yet",
                )
            )
            return ValidationReport(projectId=project_id, issues=issues)

        steps = self._time_steps(project.config.startTimeMa, project.config.endTimeMa, self.settings.keyframe_interval_myr)
        sample_times = steps[:5]
        for time_ma in sample_times:
            frame_payload = self.project_store.read_frame(project_id, run_id, time_ma)
            if frame_payload is None:
                issues.append(
                    ValidationIssue(
                        code="missing_keyframe",
                        severity="warning",
                        message=f"Keyframe {time_ma} Ma is missing from cache",
                        details={"timeMa": time_ma},
                    )
                )
                continue
            frame = TimelineFrame.model_validate(frame_payload)
            issues.extend(validate_frame(frame))

        return ValidationReport(projectId=project_id, issues=issues)

    def _resolution_dims(self, resolution: str) -> tuple[int, int]:
        if resolution == "2k":
            return 2048, 1024
        if resolution == "4k":
            return 4096, 2048
        return 8192, 4096

    def _upsample(self, array: np.ndarray, width: int, height: int) -> np.ndarray:
        image = Image.fromarray((array * 65535.0).astype(np.uint16))
        resized = image.resize((width, height), resample=Image.Resampling.BICUBIC)
        result = np.asarray(resized).astype(np.float32) / 65535.0
        return np.clip(result, 0.0, 1.0)

    def _write_heightmap(self, path: Path, terrain: np.ndarray, request: ExportRequest) -> None:
        if request.format == "png16":
            arr = np.clip(terrain * 65535.0, 0, 65535).astype(np.uint16)
            image = Image.fromarray(arr)
            image.save(path, format="PNG")
            return

        arr_f = terrain.astype(np.float32)
        image = Image.fromarray(arr_f)
        image.save(path, format="TIFF")
