from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .metadata_store import MetadataStore
from .models import (
    Bookmark,
    BookmarkRefineRequest,
    CoverageReport,
    ExportArtifact,
    ExportRequest,
    ExportResult,
    FrameDiagnostics,
    FrameSummary,
    ProjectConfig,
    ProjectSummary,
    ProvenanceRecord,
    RefineResult,
    TimelineFrame,
    ValidationIssue,
    ValidationReport,
)
from .modules.pygplates_adapter import PygplatesAdapter
from .modules.tectonic_backends import (
    aggregate_fallback_times,
    build_backend,
    collect_boundary_semantic_issues,
    collect_continuity_violations,
    derive_seed_bundle,
    frame_coverage_ratio,
)
from .modules.terrain_synthesis import synthesize_preview_height, synthesize_refined_region
from .modules.validation import validate_frame, validate_frame_pair
from .project_store import ProjectStore
from .settings import Settings
from .utils import sha256_bytes, stable_hash

ENGINE_VERSION = "0.2.0"
MODEL_VERSION = "tectonic-hybrid-backends-v1"


@dataclass
class RunContext:
    run_id: str
    project_id: str
    config: ProjectConfig
    seed_bundle: Any
    pygplates_status: str
    pygplates_available: bool
    coverage_by_time: dict[int, float] = field(default_factory=dict)
    diagnostics_by_time: dict[int, FrameDiagnostics] = field(default_factory=dict)
    kinematic_digest_by_time: dict[int, str] = field(default_factory=dict)
    uncertainty_digest_by_time: dict[int, str] = field(default_factory=dict)

    @property
    def global_coverage(self) -> float:
        if not self.coverage_by_time:
            return 0.0
        return round(sum(self.coverage_by_time.values()) / len(self.coverage_by_time), 4)


class SimulationService:
    def __init__(self, settings: Settings, metadata: MetadataStore, project_store: ProjectStore):
        self.settings = settings
        self.metadata = metadata
        self.project_store = project_store
        self.pygplates_adapter = PygplatesAdapter()
        self._run_contexts: dict[str, RunContext] = {}

    def utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _effective_config(
        self,
        project: ProjectSummary,
        simulation_mode_override: str | None = None,
        rigor_profile_override: str | None = None,
        runtime_override: int | None = None,
    ) -> ProjectConfig:
        payload = project.config.model_dump(mode="json")
        if simulation_mode_override is not None:
            payload["simulationMode"] = simulation_mode_override
        if rigor_profile_override is not None:
            payload["rigorProfile"] = rigor_profile_override
        if runtime_override is not None:
            payload["targetRuntimeMinutes"] = runtime_override
        return ProjectConfig.model_validate(payload)

    def _time_steps(self, start_time_ma: int, end_time_ma: int, step_myr: int) -> list[int]:
        values: list[int] = []
        current = start_time_ma
        while current >= end_time_ma:
            values.append(current)
            current -= step_myr
        return values

    def _run_context_from_manifest(self, project_id: str, run_id: str) -> RunContext | None:
        manifest_path = self.project_store.run_manifest_path(project_id, run_id)
        if not manifest_path.exists():
            return None
        manifest = self.project_store.read_json(manifest_path)
        config = ProjectConfig.model_validate(manifest.get("runConfig", {}))
        seed_bundle = manifest.get("seedBundle", derive_seed_bundle(config.seed).model_dump(mode="json"))

        context = RunContext(
            run_id=run_id,
            project_id=project_id,
            config=config,
            seed_bundle=seed_bundle,
            pygplates_status=manifest.get("pygplatesStatus", "unknown"),
            pygplates_available=bool(manifest.get("pygplatesAvailable", False)),
            coverage_by_time={
                int(item["timeMa"]): float(item["coverageRatio"]) for item in manifest.get("coverageByTime", [])
            },
            diagnostics_by_time={},
            kinematic_digest_by_time={
                int(item["timeMa"]): str(item["kinematicDigest"]) for item in manifest.get("coverageByTime", [])
            },
            uncertainty_digest_by_time={
                int(item["timeMa"]): str(item["uncertaintyDigest"]) for item in manifest.get("coverageByTime", [])
            },
        )
        return context

    def _resolve_context(self, project_id: str, run_id: str) -> RunContext | None:
        context = self._run_contexts.get(run_id)
        if context is not None:
            return context

        context = self._run_context_from_manifest(project_id, run_id)
        if context is None:
            return None
        self._run_contexts[run_id] = context
        return context

    def generate_project(
        self,
        project_id: str,
        *,
        simulation_mode_override: str | None,
        rigor_profile_override: str | None,
        runtime_override: int | None,
        job_callback: Any,
        is_canceled: Any,
    ) -> str:
        project = self.metadata.get_project(project_id)
        if project is None:
            raise ValueError(f"project {project_id} not found")

        run_config = self._effective_config(
            project,
            simulation_mode_override=simulation_mode_override,
            rigor_profile_override=rigor_profile_override,
            runtime_override=runtime_override,
        )

        run_id = str(uuid.uuid4())
        self.metadata.set_project_run(project_id, run_id, self.utc_now_iso())

        steps = self._time_steps(run_config.startTimeMa, run_config.endTimeMa, run_config.timeIncrementMyr)
        total = len(steps)
        edits = self.metadata.list_edits(project_id)

        seed_bundle = derive_seed_bundle(run_config.seed)
        backend = build_backend(run_config.simulationMode)
        pygplates_cache = self.pygplates_adapter.build_cache(self.project_store.project_dir(project_id) / "tectonics")
        state = backend.initialize(run_config, seed_bundle, pygplates_cache)

        context = RunContext(
            run_id=run_id,
            project_id=project_id,
            config=run_config,
            seed_bundle=seed_bundle,
            pygplates_status=pygplates_cache.status,
            pygplates_available=pygplates_cache.available,
        )

        manifest = {
            "runId": run_id,
            "generatedAt": self.utc_now_iso(),
            "keyframeIntervalMyr": self.settings.keyframe_interval_myr,
            "runConfig": run_config.model_dump(mode="json"),
            "seedBundle": seed_bundle.model_dump(mode="json"),
            "pygplatesStatus": pygplates_cache.status,
            "pygplatesAvailable": pygplates_cache.available,
            "frames": [],
            "coverageByTime": [],
        }

        for idx, time_ma in enumerate(steps, start=1):
            if is_canceled():
                break

            backend_result = backend.build_frame(
                project_id=project_id,
                config=run_config,
                state=state,
                time_ma=time_ma,
                edits=edits,
            )

            preview, _fields = synthesize_preview_height(
                time_ma=time_ma,
                seed=run_config.seed,
                plates=backend_result.frame.plateGeometries,
                events=backend_result.frame.eventOverlays,
                boundary_kinematics=backend_result.frame.boundaryKinematics,
                uncertainty=backend_result.frame.uncertaintySummary,
                width=self.settings.default_preview_width,
                height=self.settings.default_preview_height,
            )

            preview_path = self.project_store.preview_array_path(project_id, time_ma)
            strain_path = self.project_store.strain_array_path(project_id, time_ma)
            self.project_store.write_array(preview_path, preview)
            self.project_store.write_array(strain_path, backend_result.strain_field)

            backend_result.frame.previewHeightFieldRef = str(preview_path)
            backend_result.frame.strainFieldRef = str(strain_path)

            should_cache_keyframe = (
                time_ma % self.settings.keyframe_interval_myr == 0
                or time_ma == run_config.startTimeMa
                or time_ma == run_config.endTimeMa
            )

            if should_cache_keyframe:
                frame_payload = backend_result.frame.model_dump(mode="json")
                frame_path = self.project_store.write_frame(project_id, run_id, time_ma, frame_payload)
                diag_payload = backend_result.diagnostics.model_dump(mode="json")
                diag_path = self.project_store.write_run_diagnostics(project_id, run_id, time_ma, diag_payload)
                manifest["frames"].append(
                    {
                        "timeMa": time_ma,
                        "framePath": str(frame_path),
                        "diagnosticsPath": str(diag_path),
                    }
                )

            coverage_ratio = round((backend_result.coverage_ratio + frame_coverage_ratio(backend_result.frame)) * 0.5, 4)
            context.coverage_by_time[time_ma] = coverage_ratio
            context.diagnostics_by_time[time_ma] = backend_result.diagnostics
            context.kinematic_digest_by_time[time_ma] = backend_result.kinematic_digest
            context.uncertainty_digest_by_time[time_ma] = backend_result.uncertainty_digest

            manifest["coverageByTime"].append(
                {
                    "timeMa": time_ma,
                    "coverageRatio": coverage_ratio,
                    "kinematicDigest": backend_result.kinematic_digest,
                    "uncertaintyDigest": backend_result.uncertainty_digest,
                }
            )

            job_callback(idx / total, f"simulating {time_ma} Ma ({run_config.simulationMode.value})")

            if idx < total:
                backend.advance(state, run_config, time_ma, run_config.timeIncrementMyr, edits)

        fallback_times = aggregate_fallback_times(steps, context.coverage_by_time)
        manifest["fallbackTimesMa"] = fallback_times
        manifest["globalCoverageRatio"] = context.global_coverage

        manifest_path = self.project_store.run_manifest_path(project_id, run_id)
        self.project_store.write_json(manifest_path, manifest)

        self._run_contexts[run_id] = context
        return run_id

    def _replay_frame(self, project: ProjectSummary, run_config: ProjectConfig, run_id: str, time_ma: int) -> TimelineFrame:
        edits = self.metadata.list_edits(project.projectId)
        seed_bundle = derive_seed_bundle(run_config.seed)
        backend = build_backend(run_config.simulationMode)
        pygplates_cache = self.pygplates_adapter.build_cache(self.project_store.project_dir(project.projectId) / "tectonics")
        state = backend.initialize(run_config, seed_bundle, pygplates_cache)

        for current_time in self._time_steps(run_config.startTimeMa, run_config.endTimeMa, run_config.timeIncrementMyr):
            backend_result = backend.build_frame(
                project_id=project.projectId,
                config=run_config,
                state=state,
                time_ma=current_time,
                edits=edits,
            )
            preview, _fields = synthesize_preview_height(
                time_ma=current_time,
                seed=run_config.seed,
                plates=backend_result.frame.plateGeometries,
                events=backend_result.frame.eventOverlays,
                boundary_kinematics=backend_result.frame.boundaryKinematics,
                uncertainty=backend_result.frame.uncertaintySummary,
                width=self.settings.default_preview_width,
                height=self.settings.default_preview_height,
            )
            preview_path = self.project_store.preview_array_path(project.projectId, current_time)
            strain_path = self.project_store.strain_array_path(project.projectId, current_time)
            self.project_store.write_array(preview_path, preview)
            self.project_store.write_array(strain_path, backend_result.strain_field)
            backend_result.frame.previewHeightFieldRef = str(preview_path)
            backend_result.frame.strainFieldRef = str(strain_path)

            if current_time == time_ma:
                self.project_store.write_frame(project.projectId, run_id, current_time, backend_result.frame.model_dump(mode="json"))
                self.project_store.write_run_diagnostics(
                    project.projectId,
                    run_id,
                    current_time,
                    backend_result.diagnostics.model_dump(mode="json"),
                )
                return backend_result.frame

            backend.advance(state, run_config, current_time, run_config.timeIncrementMyr, edits)

        raise ValueError(f"could not replay frame for time {time_ma}")

    def get_or_create_frame(self, project: ProjectSummary, time_ma: int) -> FrameSummary:
        run_id = project.currentRunId
        if run_id:
            cached = self.project_store.read_frame(project.projectId, run_id, time_ma)
            if cached is not None:
                frame_hash = stable_hash(cached)
                return FrameSummary(frame=TimelineFrame.model_validate(cached), frameHash=frame_hash, source="cache")

            context = self._resolve_context(project.projectId, run_id)
            run_config = context.config if context is not None else project.config
            frame = self._replay_frame(project, run_config, run_id, time_ma)
            payload = frame.model_dump(mode="json")
            return FrameSummary(frame=frame, frameHash=stable_hash(payload), source="generated")

        # No run exists yet: deterministic ephemeral frame from project defaults.
        fake_run_id = "ephemeral"
        frame = self._replay_frame(project, project.config, fake_run_id, time_ma)
        payload = frame.model_dump(mode="json")
        return FrameSummary(frame=frame, frameHash=stable_hash(payload), source="generated")

    def get_frame_diagnostics(self, project_id: str, time_ma: int) -> FrameDiagnostics:
        project = self.metadata.get_project(project_id)
        if project is None:
            raise ValueError(f"project {project_id} not found")
        if project.currentRunId is None:
            raise ValueError("project has no generated run")

        run_id = project.currentRunId
        context = self._resolve_context(project_id, run_id)
        if context and time_ma in context.diagnostics_by_time:
            return context.diagnostics_by_time[time_ma]

        cached = self.project_store.read_run_diagnostics(project_id, run_id, time_ma)
        if cached is not None:
            diagnostics = FrameDiagnostics.model_validate(cached)
            if context:
                context.diagnostics_by_time[time_ma] = diagnostics
            return diagnostics

        frame_summary = self.get_or_create_frame(project, time_ma)
        continuity = collect_continuity_violations(frame_summary.frame)
        semantic = collect_boundary_semantic_issues(frame_summary.frame)
        diagnostics = FrameDiagnostics(
            projectId=project_id,
            timeMa=time_ma,
            continuityViolations=continuity,
            boundaryConsistencyIssues=semantic,
            coverageGapRatio=frame_summary.frame.uncertaintySummary.coverage,
            warnings=[],
            pygplatesStatus=context.pygplates_status if context else "unknown",
        )
        return diagnostics

    def get_coverage_report(self, project_id: str) -> CoverageReport:
        project = self.metadata.get_project(project_id)
        if project is None:
            raise ValueError(f"project {project_id} not found")

        if project.currentRunId is None:
            return CoverageReport(projectId=project_id, globalCoverageRatio=0.0)

        context = self._resolve_context(project_id, project.currentRunId)
        if context is None:
            return CoverageReport(projectId=project_id, globalCoverageRatio=0.0)

        coverage_items = [
            {"timeMa": float(time), "coverageRatio": ratio}
            for time, ratio in sorted(context.coverage_by_time.items(), reverse=True)
        ]
        fallback_times = aggregate_fallback_times(context.coverage_by_time.keys(), context.coverage_by_time)

        return CoverageReport(
            projectId=project_id,
            globalCoverageRatio=context.global_coverage,
            coverageRatioByTime=coverage_items,
            fallbackTimesMa=fallback_times,
            pygplatesAvailable=context.pygplates_available,
        )

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

        strain = None
        if frame_summary.frame.strainFieldRef:
            strain = self.project_store.read_array(Path(frame_summary.frame.strainFieldRef))

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
            strain_field=strain,
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
            for offset in range(-40, 41, self.settings.keyframe_interval_myr):
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

        frame: TimelineFrame
        frame_time_ma: int

        if request.bookmarkId:
            bookmark_data = self.metadata.get_bookmark(request.bookmarkId)
            if bookmark_data is None:
                raise ValueError(f"bookmark {request.bookmarkId} not found")
            frame_time_ma = int(bookmark_data["timeMa"])
            frame = self.get_or_create_frame(project, frame_time_ma).frame
            refined_path = self.project_store.refined_array_path(project_id, request.bookmarkId, 1)
            if refined_path.exists():
                source = self.project_store.read_array(refined_path)
            else:
                source = self.project_store.read_array(Path(frame.previewHeightFieldRef))
                strain = (
                    self.project_store.read_array(Path(frame.strainFieldRef))
                    if frame.strainFieldRef
                    else None
                )
                source = synthesize_refined_region(
                    source,
                    bookmark_data["region"],
                    2048,
                    1024,
                    1,
                    project.config.seed,
                    strain_field=strain,
                )
        else:
            if request.timeMa is None:
                raise ValueError("timeMa is required when bookmarkId is not provided")
            frame_time_ma = int(request.timeMa)
            frame = self.get_or_create_frame(project, frame_time_ma).frame
            source = self.project_store.read_array(Path(frame.previewHeightFieldRef))

        if is_canceled():
            raise RuntimeError("job canceled")

        job_callback(0.35, "upsampling terrain")
        terrain = self._upsample(source, request.width, request.height)

        if request.region is not None:
            strain = self.project_store.read_array(Path(frame.strainFieldRef)) if frame.strainFieldRef else None
            terrain = synthesize_refined_region(
                terrain,
                request.region,
                request.width,
                request.height,
                1,
                project.config.seed,
                strain_field=strain,
            )

        if is_canceled():
            raise RuntimeError("job canceled")

        artifact_id = str(uuid.uuid4())
        suffix = ".png" if request.format == "png16" else ".tiff"
        filename = f"{frame_time_ma}Ma_{artifact_id}{suffix}"
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

        run_context = self._resolve_context(project_id, project.currentRunId) if project.currentRunId else None
        solver_mode = run_context.config.simulationMode if run_context else project.config.simulationMode
        rigor_profile = run_context.config.rigorProfile if run_context else project.config.rigorProfile
        coverage_ratio = run_context.global_coverage if run_context else frame_coverage_ratio(frame)

        kinematic_digest = stable_hash([kin.model_dump(mode="json") for kin in frame.plateKinematics])
        uncertainty_digest = stable_hash(frame.uncertaintySummary.model_dump(mode="json"))

        provenance = ProvenanceRecord(
            projectHash=project.projectHash,
            seed=project.config.seed,
            engineVersion=ENGINE_VERSION,
            modelVersion=MODEL_VERSION,
            solverMode=solver_mode,
            rigorProfile=rigor_profile,
            parameterHash=stable_hash(request.model_dump(mode="json")),
            eventHash=stable_hash([event.model_dump(mode="json") for event in frame.eventOverlays]),
            kinematicDigest=kinematic_digest,
            uncertaintyDigest=uncertainty_digest,
            modelCoverage=coverage_ratio,
        )

        metadata_payload = {
            "provenance": provenance.model_dump(mode="json"),
            "projectId": project.projectId,
            "timeMa": frame_time_ma,
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

        context = self._resolve_context(project_id, run_id)
        run_config = context.config if context else project.config

        steps = self._time_steps(run_config.startTimeMa, run_config.endTimeMa, self.settings.keyframe_interval_myr)
        sample_times = steps[:6]
        previous_frame: TimelineFrame | None = None

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
            issues.extend(validate_frame_pair(previous_frame, frame))
            previous_frame = frame

        coverage = self.get_coverage_report(project_id)
        if coverage.fallbackTimesMa:
            issues.append(
                ValidationIssue(
                    code="coverage_fallback_times",
                    severity="warning",
                    message="Some frames rely heavily on topology fallback behavior",
                    details={"fallbackTimesMa": coverage.fallbackTimesMa[:20]},
                )
            )

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
