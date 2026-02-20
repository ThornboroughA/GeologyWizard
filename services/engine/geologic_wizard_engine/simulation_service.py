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
    FrameRangeResponse,
    FrameRender,
    FrameSummary,
    PlausibilityCheck,
    PlausibilityReport,
    ProjectConfig,
    ProjectSummary,
    ProvenanceRecord,
    QualityMode,
    RefineResult,
    SolverVersion,
    TimelineIndex,
    TimelineIndexHashEntry,
    TimelineFrame,
    ValidationIssue,
    ValidationReport,
)
from .modules.render_payload import build_frame_render_payload
from .modules.pygplates_adapter import PygplatesAdapter
from .modules.tectonic_backends import (
    aggregate_fallback_times,
    build_backend,
    collect_boundary_semantic_issues,
    collect_continuity_violations,
    derive_seed_bundle,
    frame_coverage_ratio,
)
from .modules.tectonics_v2 import build_backend_v2
from .modules.terrain_synthesis import synthesize_preview_height, synthesize_refined_region
from .modules.validation import (
    build_plausibility_checks_from_frame,
    summarize_check_severity,
    validate_frame,
    validate_frame_pair,
)
from .project_store import ProjectStore
from .settings import Settings
from .utils import sha256_bytes, stable_hash

ENGINE_VERSION = "0.2.0"
MODEL_VERSION = "tectonic-hybrid-backends-v1"
MODEL_VERSION_V2 = "tectonic-state-v2"


@dataclass
class RunContext:
    run_id: str
    project_id: str
    config: ProjectConfig
    seed_bundle: Any
    pygplates_status: str
    pygplates_available: bool
    quality_mode: QualityMode = QualityMode.quick
    source_quick_run_id: str | None = None
    macro_digest: str = ""
    coverage_by_time: dict[int, float] = field(default_factory=dict)
    diagnostics_by_time: dict[int, FrameDiagnostics] = field(default_factory=dict)
    plausibility_by_time: dict[int, list[PlausibilityCheck]] = field(default_factory=dict)
    kinematic_digest_by_time: dict[int, str] = field(default_factory=dict)
    uncertainty_digest_by_time: dict[int, str] = field(default_factory=dict)
    timeline_index: TimelineIndex | None = None

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

        quality_value = manifest.get("qualityMode", "quick")
        try:
            quality_mode = QualityMode(str(quality_value))
        except Exception:
            quality_mode = QualityMode.quick

        context = RunContext(
            run_id=run_id,
            project_id=project_id,
            config=config,
            seed_bundle=seed_bundle,
            pygplates_status=manifest.get("pygplatesStatus", "unknown"),
            pygplates_available=bool(manifest.get("pygplatesAvailable", False)),
            quality_mode=quality_mode,
            source_quick_run_id=manifest.get("sourceQuickRunId"),
            macro_digest=str(manifest.get("macroDigest", "")),
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
        context.timeline_index = self._read_timeline_index(project_id, run_id)
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

    def _nearest_time(self, times: list[int], target_time_ma: int) -> int | None:
        if not times:
            return None
        return min(times, key=lambda candidate: abs(candidate - target_time_ma))

    def _empty_timeline_index(self, project_id: str, run_id: str, config: ProjectConfig) -> TimelineIndex:
        return TimelineIndex(
            projectId=project_id,
            runId=run_id,
            startTimeMa=config.startTimeMa,
            endTimeMa=config.endTimeMa,
            stepMyr=config.timeIncrementMyr,
            generatedOrder="descending_ma",
            times=[],
            hashes={},
            availableDetails=["render", "full"],
        )

    def _read_timeline_index(self, project_id: str, run_id: str) -> TimelineIndex | None:
        index_path = self.project_store.timeline_index_path(project_id, run_id)
        if not index_path.exists():
            return None
        try:
            payload = self.project_store.read_json(index_path)
            return TimelineIndex.model_validate(payload)
        except Exception:
            return None

    def _write_timeline_index(self, project_id: str, run_id: str, index: TimelineIndex) -> None:
        self.project_store.write_json(
            self.project_store.timeline_index_path(project_id, run_id),
            index.model_dump(mode="json"),
        )

    def _persist_frame_bundle(
        self,
        *,
        project_id: str,
        run_id: str,
        frame: TimelineFrame,
        diagnostics: FrameDiagnostics,
        source: str,
    ) -> tuple[str, str]:
        full_payload = frame.model_dump(mode="json")
        self.project_store.write_frame(project_id, run_id, frame.timeMa, full_payload)

        render_frame = build_frame_render_payload(
            frame,
            source=source,
            nearest_time_ma=frame.timeMa,
        )
        render_payload = render_frame.model_dump(mode="json")
        self.project_store.write_render_frame(project_id, run_id, frame.timeMa, render_payload)
        self.project_store.write_run_diagnostics(project_id, run_id, frame.timeMa, diagnostics.model_dump(mode="json"))
        return stable_hash(full_payload), stable_hash(render_payload)

    def _build_timeline_index_from_cached_frames(
        self,
        project_id: str,
        run_id: str,
        run_config: ProjectConfig,
    ) -> TimelineIndex:
        index = self._empty_timeline_index(project_id, run_id, run_config)
        frame_times = self.project_store.list_frame_times(project_id, run_id)
        hashes: dict[str, TimelineIndexHashEntry] = {}

        for time_ma in frame_times:
            frame_payload = self.project_store.read_frame(project_id, run_id, time_ma)
            if frame_payload is None:
                continue
            frame = TimelineFrame.model_validate(frame_payload)
            full_hash = stable_hash(frame_payload)

            render_payload = self.project_store.read_render_frame(project_id, run_id, time_ma)
            if render_payload is None:
                render_frame = build_frame_render_payload(frame, source="generated", nearest_time_ma=time_ma)
                render_payload = render_frame.model_dump(mode="json")
                self.project_store.write_render_frame(project_id, run_id, time_ma, render_payload)

            render_hash = stable_hash(render_payload)
            hashes[str(time_ma)] = TimelineIndexHashEntry(full=full_hash, render=render_hash)

        index.times = sorted([int(key) for key in hashes.keys()], reverse=True)
        index.hashes = {str(time): hashes[str(time)] for time in index.times}
        self._write_timeline_index(project_id, run_id, index)
        return index

    def _load_or_build_timeline_index(
        self,
        project_id: str,
        run_id: str,
        run_config: ProjectConfig,
        *,
        context: RunContext | None = None,
    ) -> TimelineIndex:
        if context and context.timeline_index is not None:
            return context.timeline_index

        timeline_index = self._read_timeline_index(project_id, run_id)
        if timeline_index is None:
            timeline_index = self._build_timeline_index_from_cached_frames(project_id, run_id, run_config)

        if context:
            context.timeline_index = timeline_index
        return timeline_index

    def _record_timeline_hash(
        self,
        index: TimelineIndex,
        *,
        time_ma: int,
        full_hash: str,
        render_hash: str,
    ) -> None:
        index.hashes[str(time_ma)] = TimelineIndexHashEntry(full=full_hash, render=render_hash)
        if time_ma not in index.times:
            index.times.append(time_ma)
            index.times.sort(reverse=True)

    def _build_backend_for_config(self, run_config: ProjectConfig) -> Any:
        if run_config.solverVersion == SolverVersion.tectonic_state_v2:
            return build_backend_v2()
        return build_backend(run_config.simulationMode)

    def _solver_model_version(self, run_config: ProjectConfig) -> str:
        if run_config.solverVersion == SolverVersion.tectonic_state_v2:
            return MODEL_VERSION_V2
        return MODEL_VERSION

    def _coverage_ratio_for_result(self, result: Any) -> float:
        if result.frame.plateLifecycleState is not None:
            lifecycle_gap = float(result.frame.plateLifecycleState.netAreaBalanceError)
            lifecycle_coverage = max(0.0, min(1.0, 1.0 - lifecycle_gap))
            return round((result.coverage_ratio + lifecycle_coverage) * 0.5, 4)
        return round((result.coverage_ratio + frame_coverage_ratio(result.frame)) * 0.5, 4)

    def _persist_auxiliary_fields(self, project_id: str, run_id: str, time_ma: int, backend_result: Any) -> None:
        if not hasattr(backend_result, "oceanic_age_field"):
            return

        oceanic_age_path = self.project_store.oceanic_age_array_path(project_id, run_id, time_ma)
        crust_type_path = self.project_store.crust_type_array_path(project_id, run_id, time_ma)
        crust_thickness_path = self.project_store.crust_thickness_array_path(project_id, run_id, time_ma)
        tectonic_potential_path = self.project_store.tectonic_potential_array_path(project_id, run_id, time_ma)

        self.project_store.write_array(oceanic_age_path, backend_result.oceanic_age_field)
        self.project_store.write_array(crust_type_path, backend_result.crust_type_field)
        self.project_store.write_array(crust_thickness_path, backend_result.crust_thickness_field)
        self.project_store.write_array(tectonic_potential_path, backend_result.tectonic_potential_field)

        backend_result.frame.oceanicAgeFieldRef = str(oceanic_age_path)
        backend_result.frame.crustTypeFieldRef = str(crust_type_path)
        backend_result.frame.crustThicknessFieldRef = str(crust_thickness_path)
        backend_result.frame.tectonicPotentialFieldRef = str(tectonic_potential_path)

    def _extract_macro_snapshot(self, frame: TimelineFrame) -> dict[str, Any]:
        payload = frame.model_dump(mode="json")
        payload["previewHeightFieldRef"] = ""
        payload["strainFieldRef"] = None
        payload["oceanicAgeFieldRef"] = None
        payload["crustTypeFieldRef"] = None
        payload["crustThicknessFieldRef"] = None
        payload["tectonicPotentialFieldRef"] = None
        return payload

    def _write_macro_history(self, project_id: str, run_id: str, snapshots: list[dict[str, Any]]) -> str:
        digest = stable_hash(
            {
                "times": [int(item["timeMa"]) for item in snapshots],
                "frames": snapshots,
            }
        )
        payload = {
            "projectId": project_id,
            "runId": run_id,
            "times": [int(item["timeMa"]) for item in snapshots],
            "frames": snapshots,
            "macroDigest": digest,
        }
        self.project_store.write_json(self.project_store.macro_history_path(project_id, run_id), payload)
        return digest

    def _read_macro_history(self, project_id: str, run_id: str) -> tuple[str, list[dict[str, Any]]] | None:
        path = self.project_store.macro_history_path(project_id, run_id)
        if not path.exists():
            return None
        data = self.project_store.read_json(path)
        frames = data.get("frames")
        if not isinstance(frames, list):
            return None
        digest = str(
            data.get("macroDigest")
            or stable_hash(
                {
                    "times": [int(item.get("timeMa", 0)) for item in frames if isinstance(item, dict)],
                    "frames": frames,
                }
            )
        )
        return digest, frames

    def _list_project_run_manifests(self, project_id: str) -> list[dict[str, Any]]:
        runs_root = self.project_store.project_dir(project_id) / "runs"
        manifests: list[dict[str, Any]] = []
        if not runs_root.exists():
            return manifests
        for run_dir in runs_root.iterdir():
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = self.project_store.read_json(manifest_path)
                manifest["runId"] = str(manifest.get("runId") or run_dir.name)
                manifests.append(manifest)
            except Exception:
                continue
        manifests.sort(key=lambda item: str(item.get("generatedAt", "")), reverse=True)
        return manifests

    def _resolve_source_quick_run_id(self, project_id: str, explicit_run_id: str | None) -> str | None:
        if explicit_run_id:
            return explicit_run_id
        for manifest in self._list_project_run_manifests(project_id):
            if str(manifest.get("qualityMode", "quick")) == QualityMode.quick.value:
                return str(manifest["runId"])
        return None

    def _synthesize_surface_from_macro_frame(
        self,
        frame: TimelineFrame,
        *,
        run_config: ProjectConfig,
        seed: int,
        quality_mode: QualityMode,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        base_width = run_config.coreGridWidth or (
            720 if run_config.simulationMode.value == "hybrid_rigor" else self.settings.default_preview_width
        )
        base_height = run_config.coreGridHeight or (
            360 if run_config.simulationMode.value == "hybrid_rigor" else self.settings.default_preview_height
        )
        preview_width = base_width if quality_mode == QualityMode.quick else min(2048, base_width * 2)
        preview_height = base_height if quality_mode == QualityMode.quick else min(1024, base_height * 2)
        preview, fields = synthesize_preview_height(
            time_ma=frame.timeMa,
            seed=seed,
            plates=frame.plateGeometries,
            events=frame.eventOverlays,
            boundary_kinematics=frame.boundaryKinematics,
            uncertainty=frame.uncertaintySummary,
            width=preview_width,
            height=preview_height,
        )

        # Full mode adds additional deterministic erosion/smoothing passes for richer terrain realization.
        if quality_mode == QualityMode.full:
            full_passes = 18 if run_config.simulationMode.value == "fast_plausible" else 28
            if run_config.rigorProfile.value == "research":
                full_passes += 8
            for _ in range(full_passes):
                neighborhood = (
                    preview
                    + np.roll(preview, 1, axis=0)
                    + np.roll(preview, -1, axis=0)
                    + np.roll(preview, 1, axis=1)
                    + np.roll(preview, -1, axis=1)
                ) / 5.0
                relief = np.clip(preview - neighborhood, -0.15, 0.15)
                preview = np.clip(neighborhood + relief * 0.55, 0.0, 1.0)

                if run_config.rigorProfile.value == "research":
                    grad_y, grad_x = np.gradient(preview)
                    slope = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
                    erosion = np.clip(slope * 0.11, 0.0, 0.05)
                    uplift_bias = np.clip(fields["uplift"] * 0.08, 0.0, 0.06)
                    preview = np.clip(preview + uplift_bias - erosion, 0.0, 1.0)

            basin = np.clip(fields["subsidence"] * 0.08, 0.0, 0.08)
            preview = np.clip(preview - basin + np.clip(fields["volcanic"] * 0.05, 0.0, 0.04), 0.0, 1.0)

        strain_height = max(128, min(512, preview_height // 2))
        strain_width = max(256, min(1024, preview_width // 2))
        strain = np.zeros((strain_height, strain_width), dtype=np.float32)
        for boundary in frame.boundaryGeometries:
            coords = boundary.geometry.get("coordinates", [])
            if len(coords) < 2:
                continue
            cx = (float(coords[0][0]) + float(coords[1][0])) * 0.5
            cy = (float(coords[0][1]) + float(coords[1][1])) * 0.5
            x = int(((cx + 180.0) / 360.0) * (strain.shape[1] - 1))
            y = int(((cy + 90.0) / 180.0) * (strain.shape[0] - 1))
            x = max(0, min(strain.shape[1] - 1, x))
            y = max(0, min(strain.shape[0] - 1, y))
            strain[y, x] = min(1.0, strain[y, x] + 0.35)

        for _ in range(4):
            strain = (
                strain
                + np.roll(strain, 1, axis=0)
                + np.roll(strain, -1, axis=0)
                + np.roll(strain, 1, axis=1)
                + np.roll(strain, -1, axis=1)
            ) / 5.0

        return preview.astype(np.float32), strain.astype(np.float32), fields

    def _materialize_full_run_from_macro(
        self,
        *,
        project_id: str,
        run_id: str,
        run_config: ProjectConfig,
        quality_mode: QualityMode,
        macro_snapshots: list[dict[str, Any]],
        context: RunContext,
        timeline_index: TimelineIndex,
        manifest: dict[str, Any],
        job_callback: Any,
        is_canceled: Any,
    ) -> None:
        total = max(1, len(macro_snapshots))
        for idx, snapshot in enumerate(macro_snapshots, start=1):
            if is_canceled():
                break

            frame = TimelineFrame.model_validate(snapshot)
            preview, strain, fields = self._synthesize_surface_from_macro_frame(
                frame,
                run_config=run_config,
                seed=run_config.seed,
                quality_mode=quality_mode,
            )

            preview_path = self.project_store.preview_array_path(project_id, run_id, frame.timeMa)
            strain_path = self.project_store.strain_array_path(project_id, run_id, frame.timeMa)
            self.project_store.write_array(preview_path, preview)
            self.project_store.write_array(strain_path, strain)
            frame.previewHeightFieldRef = str(preview_path)
            frame.strainFieldRef = str(strain_path)

            # Surface-only full pass derives deterministic fields from tectonic potential components.
            oceanic_age = np.clip((1.0 - fields["crust_age"]) * 320.0, 0.0, 500.0).astype(np.float32)
            crust_type = (preview >= 0.53).astype(np.uint8)
            crust_thickness = np.where(crust_type > 0, 34.0 + fields["uplift"] * 18.0, 6.5 + oceanic_age * 0.012).astype(np.float32)
            tectonic_potential = np.clip(fields["uplift"] + fields["volcanic"] - fields["subsidence"], 0.0, 1.0).astype(np.float32)

            oceanic_age_path = self.project_store.oceanic_age_array_path(project_id, run_id, frame.timeMa)
            crust_type_path = self.project_store.crust_type_array_path(project_id, run_id, frame.timeMa)
            crust_thickness_path = self.project_store.crust_thickness_array_path(project_id, run_id, frame.timeMa)
            tectonic_potential_path = self.project_store.tectonic_potential_array_path(project_id, run_id, frame.timeMa)
            self.project_store.write_array(oceanic_age_path, oceanic_age)
            self.project_store.write_array(crust_type_path, crust_type)
            self.project_store.write_array(crust_thickness_path, crust_thickness)
            self.project_store.write_array(tectonic_potential_path, tectonic_potential)

            frame.oceanicAgeFieldRef = str(oceanic_age_path)
            frame.crustTypeFieldRef = str(crust_type_path)
            frame.crustThicknessFieldRef = str(crust_thickness_path)
            frame.tectonicPotentialFieldRef = str(tectonic_potential_path)

            diagnostics = FrameDiagnostics(
                projectId=project_id,
                timeMa=frame.timeMa,
                continuityViolations=[
                    f"plate_{item.plateId}_continuity_low"
                    for item in frame.plateKinematics
                    if item.continuityScore < 0.2
                ],
                boundaryConsistencyIssues=[
                    f"{item.segmentId}_motion_mismatch"
                    for item in frame.boundaryStates
                    if item.motionMismatch
                ],
                coverageGapRatio=frame.uncertaintySummary.coverage,
                warnings=[],
                pygplatesStatus=context.pygplates_status,
            )

            full_hash, render_hash = self._persist_frame_bundle(
                project_id=project_id,
                run_id=run_id,
                frame=frame,
                diagnostics=diagnostics,
                source="generated",
            )
            self._record_timeline_hash(
                timeline_index,
                time_ma=frame.timeMa,
                full_hash=full_hash,
                render_hash=render_hash,
            )

            coverage_ratio = round(max(0.0, min(1.0, 1.0 - frame.uncertaintySummary.coverage)), 4)
            context.coverage_by_time[frame.timeMa] = coverage_ratio
            context.diagnostics_by_time[frame.timeMa] = diagnostics
            context.kinematic_digest_by_time[frame.timeMa] = stable_hash([item.model_dump(mode="json") for item in frame.plateKinematics])
            context.uncertainty_digest_by_time[frame.timeMa] = stable_hash(frame.uncertaintySummary.model_dump(mode="json"))
            context.plausibility_by_time[frame.timeMa] = build_plausibility_checks_from_frame(frame)

            manifest["frames"].append(
                {
                    "timeMa": frame.timeMa,
                    "framePath": str(self.project_store.frame_path(project_id, run_id, frame.timeMa)),
                    "renderFramePath": str(self.project_store.render_frame_path(project_id, run_id, frame.timeMa)),
                    "diagnosticsPath": str(self.project_store.run_diagnostics_path(project_id, run_id, frame.timeMa)),
                    "fullHash": full_hash,
                    "renderHash": render_hash,
                }
            )
            manifest["coverageByTime"].append(
                {
                    "timeMa": frame.timeMa,
                    "coverageRatio": coverage_ratio,
                    "kinematicDigest": context.kinematic_digest_by_time[frame.timeMa],
                    "uncertaintyDigest": context.uncertainty_digest_by_time[frame.timeMa],
                    "checkCount": len(context.plausibility_by_time[frame.timeMa]),
                }
            )

            job_callback(idx / total, f"realizing {quality_mode.value} surface {frame.timeMa} Ma")

    def generate_project(
        self,
        project_id: str,
        *,
        simulation_mode_override: str | None,
        rigor_profile_override: str | None,
        runtime_override: int | None,
        quality_mode: QualityMode = QualityMode.quick,
        source_quick_run_id: str | None = None,
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
        self.metadata.set_project_run(project_id, run_id, self.utc_now_iso(), quality_mode.value)

        steps = self._time_steps(run_config.startTimeMa, run_config.endTimeMa, run_config.timeIncrementMyr)
        total = len(steps)
        edits = self.metadata.list_edits(project_id)

        seed_bundle = derive_seed_bundle(run_config.seed)
        backend = self._build_backend_for_config(run_config)
        pygplates_cache = self.pygplates_adapter.build_cache(self.project_store.project_dir(project_id) / "tectonics")
        state = backend.initialize(run_config, seed_bundle, pygplates_cache)

        context = RunContext(
            run_id=run_id,
            project_id=project_id,
            config=run_config,
            seed_bundle=seed_bundle,
            pygplates_status=pygplates_cache.status,
            pygplates_available=pygplates_cache.available,
            quality_mode=quality_mode,
            source_quick_run_id=source_quick_run_id,
        )

        manifest = {
            "runId": run_id,
            "generatedAt": self.utc_now_iso(),
            "keyframeIntervalMyr": self.settings.keyframe_interval_myr,
            "runConfig": run_config.model_dump(mode="json"),
            "solverVersion": run_config.solverVersion.value,
            "seedBundle": seed_bundle.model_dump(mode="json"),
            "pygplatesStatus": pygplates_cache.status,
            "pygplatesAvailable": pygplates_cache.available,
            "qualityMode": quality_mode.value,
            "sourceQuickRunId": source_quick_run_id,
            "frames": [],
            "coverageByTime": [],
        }
        timeline_index = self._empty_timeline_index(project_id, run_id, run_config)
        macro_snapshots: list[dict[str, Any]] = []

        if quality_mode == QualityMode.full and run_config.solverVersion == SolverVersion.tectonic_state_v2:
            resolved_quick_run_id = self._resolve_source_quick_run_id(project_id, source_quick_run_id)
            if resolved_quick_run_id is None:
                raise ValueError("full run requires an existing quick run for macro-history reuse")
            macro_data = self._read_macro_history(project_id, resolved_quick_run_id)
            if macro_data is None:
                raise ValueError(f"macro history missing for quick run {resolved_quick_run_id}")
            macro_digest, macro_snapshots = macro_data
            context.source_quick_run_id = resolved_quick_run_id
            context.macro_digest = macro_digest
            manifest["sourceQuickRunId"] = resolved_quick_run_id
            manifest["macroDigest"] = macro_digest

            self._materialize_full_run_from_macro(
                project_id=project_id,
                run_id=run_id,
                run_config=run_config,
                quality_mode=quality_mode,
                macro_snapshots=macro_snapshots,
                context=context,
                timeline_index=timeline_index,
                manifest=manifest,
                job_callback=job_callback,
                is_canceled=is_canceled,
            )
            self._write_macro_history(project_id, run_id, macro_snapshots)
        else:
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

                if hasattr(backend_result, "preview_height_field"):
                    preview = backend_result.preview_height_field
                else:
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

                preview_path = self.project_store.preview_array_path(project_id, run_id, time_ma)
                strain_path = self.project_store.strain_array_path(project_id, run_id, time_ma)
                self.project_store.write_array(preview_path, preview)
                self.project_store.write_array(strain_path, backend_result.strain_field)

                backend_result.frame.previewHeightFieldRef = str(preview_path)
                backend_result.frame.strainFieldRef = str(strain_path)
                self._persist_auxiliary_fields(project_id, run_id, time_ma, backend_result)
                full_hash, render_hash = self._persist_frame_bundle(
                    project_id=project_id,
                    run_id=run_id,
                    frame=backend_result.frame,
                    diagnostics=backend_result.diagnostics,
                    source="generated",
                )
                self._record_timeline_hash(
                    timeline_index,
                    time_ma=time_ma,
                    full_hash=full_hash,
                    render_hash=render_hash,
                )
                manifest["frames"].append(
                    {
                        "timeMa": time_ma,
                        "framePath": str(self.project_store.frame_path(project_id, run_id, time_ma)),
                        "renderFramePath": str(self.project_store.render_frame_path(project_id, run_id, time_ma)),
                        "diagnosticsPath": str(self.project_store.run_diagnostics_path(project_id, run_id, time_ma)),
                        "fullHash": full_hash,
                        "renderHash": render_hash,
                    }
                )

                coverage_ratio = self._coverage_ratio_for_result(backend_result)
                context.coverage_by_time[time_ma] = coverage_ratio
                context.diagnostics_by_time[time_ma] = backend_result.diagnostics
                context.plausibility_by_time[time_ma] = list(getattr(backend_result, "plausibility_checks", []))
                context.kinematic_digest_by_time[time_ma] = backend_result.kinematic_digest
                context.uncertainty_digest_by_time[time_ma] = backend_result.uncertainty_digest

                manifest["coverageByTime"].append(
                    {
                        "timeMa": time_ma,
                        "coverageRatio": coverage_ratio,
                        "kinematicDigest": backend_result.kinematic_digest,
                        "uncertaintyDigest": backend_result.uncertainty_digest,
                        "checkCount": len(getattr(backend_result, "plausibility_checks", [])),
                    }
                )

                macro_snapshots.append(self._extract_macro_snapshot(backend_result.frame))

                job_callback(idx / total, f"simulating {time_ma} Ma ({run_config.simulationMode.value})")

                if idx < total:
                    backend.advance(state, run_config, time_ma, run_config.timeIncrementMyr, edits)

            macro_digest = self._write_macro_history(project_id, run_id, macro_snapshots)
            manifest["macroDigest"] = macro_digest
            context.macro_digest = macro_digest

        fallback_times = aggregate_fallback_times(steps, context.coverage_by_time)
        manifest["fallbackTimesMa"] = fallback_times
        manifest["globalCoverageRatio"] = context.global_coverage
        manifest["generatedOrder"] = "descending_ma"
        manifest["timelineIndexPath"] = str(self.project_store.timeline_index_path(project_id, run_id))

        manifest_path = self.project_store.run_manifest_path(project_id, run_id)
        self.project_store.write_json(manifest_path, manifest)
        self._write_timeline_index(project_id, run_id, timeline_index)

        context.timeline_index = timeline_index
        self._run_contexts[run_id] = context
        return run_id

    def _replay_frame(self, project: ProjectSummary, run_config: ProjectConfig, run_id: str, time_ma: int) -> Any:
        edits = self.metadata.list_edits(project.projectId)
        seed_bundle = derive_seed_bundle(run_config.seed)
        backend = self._build_backend_for_config(run_config)
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
            if hasattr(backend_result, "preview_height_field"):
                preview = backend_result.preview_height_field
            else:
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
            preview_path = self.project_store.preview_array_path(project.projectId, run_id, current_time)
            strain_path = self.project_store.strain_array_path(project.projectId, run_id, current_time)
            self.project_store.write_array(preview_path, preview)
            self.project_store.write_array(strain_path, backend_result.strain_field)
            backend_result.frame.previewHeightFieldRef = str(preview_path)
            backend_result.frame.strainFieldRef = str(strain_path)
            self._persist_auxiliary_fields(project.projectId, run_id, current_time, backend_result)

            if current_time == time_ma:
                self._persist_frame_bundle(
                    project_id=project.projectId,
                    run_id=run_id,
                    frame=backend_result.frame,
                    diagnostics=backend_result.diagnostics,
                    source="generated",
                )
                return backend_result

            backend.advance(state, run_config, current_time, run_config.timeIncrementMyr, edits)

        raise ValueError(f"could not replay frame for time {time_ma}")

    def _read_or_build_render_frame(self, project_id: str, run_id: str, time_ma: int) -> FrameRender | None:
        render_payload = self.project_store.read_render_frame(project_id, run_id, time_ma)
        if render_payload is not None:
            return FrameRender.model_validate(render_payload)

        full_payload = self.project_store.read_frame(project_id, run_id, time_ma)
        if full_payload is None:
            return None

        frame = TimelineFrame.model_validate(full_payload)
        render_frame = build_frame_render_payload(frame, source="generated", nearest_time_ma=time_ma)
        self.project_store.write_render_frame(project_id, run_id, time_ma, render_frame.model_dump(mode="json"))
        return render_frame

    def _backfill_time_index_entry(
        self,
        *,
        index: TimelineIndex,
        project_id: str,
        run_id: str,
        time_ma: int,
        frame_payload: dict[str, Any],
    ) -> None:
        render_payload = self.project_store.read_render_frame(project_id, run_id, time_ma)
        if render_payload is None:
            frame = TimelineFrame.model_validate(frame_payload)
            render_payload = build_frame_render_payload(
                frame,
                source="generated",
                nearest_time_ma=time_ma,
            ).model_dump(mode="json")
            self.project_store.write_render_frame(project_id, run_id, time_ma, render_payload)

        self._record_timeline_hash(
            index,
            time_ma=time_ma,
            full_hash=stable_hash(frame_payload),
            render_hash=stable_hash(render_payload),
        )

    def get_or_create_frame(self, project: ProjectSummary, time_ma: int) -> FrameSummary:
        run_id = project.currentRunId
        if run_id:
            context = self._resolve_context(project.projectId, run_id)
            run_config = context.config if context is not None else project.config
            index = self._load_or_build_timeline_index(project.projectId, run_id, run_config, context=context)
            nearest = self._nearest_time(index.times, time_ma)

            cached = self.project_store.read_frame(project.projectId, run_id, time_ma)
            if cached is not None:
                if str(time_ma) not in index.hashes:
                    self._backfill_time_index_entry(
                        index=index,
                        project_id=project.projectId,
                        run_id=run_id,
                        time_ma=time_ma,
                        frame_payload=cached,
                    )
                    self._write_timeline_index(project.projectId, run_id, index)
                frame_hash = index.hashes[str(time_ma)].full
                return FrameSummary(
                    frame=TimelineFrame.model_validate(cached),
                    frameHash=frame_hash,
                    source="cache",
                    nearestAvailableTimeMa=nearest,
                    servedDetail="full",
                )

            replay = self._replay_frame(project, run_config, run_id, time_ma)
            payload = replay.frame.model_dump(mode="json")
            full_hash = stable_hash(payload)
            render_hash = stable_hash(
                self.project_store.read_render_frame(project.projectId, run_id, time_ma)
                or build_frame_render_payload(replay.frame, source="generated", nearest_time_ma=time_ma).model_dump(mode="json")
            )
            self._record_timeline_hash(index, time_ma=time_ma, full_hash=full_hash, render_hash=render_hash)
            self._write_timeline_index(project.projectId, run_id, index)

            if context is not None:
                context.diagnostics_by_time[time_ma] = replay.diagnostics
                coverage_ratio = self._coverage_ratio_for_result(replay)
                context.coverage_by_time[time_ma] = coverage_ratio
                context.plausibility_by_time[time_ma] = list(getattr(replay, "plausibility_checks", []))
                context.kinematic_digest_by_time[time_ma] = replay.kinematic_digest
                context.uncertainty_digest_by_time[time_ma] = replay.uncertainty_digest

            return FrameSummary(
                frame=replay.frame,
                frameHash=full_hash,
                source="generated",
                nearestAvailableTimeMa=nearest,
                servedDetail="full",
            )

        # No run exists yet: deterministic ephemeral frame from project defaults.
        fake_run_id = "ephemeral"
        replay = self._replay_frame(project, project.config, fake_run_id, time_ma)
        payload = replay.frame.model_dump(mode="json")
        return FrameSummary(
            frame=replay.frame,
            frameHash=stable_hash(payload),
            source="generated",
            nearestAvailableTimeMa=time_ma,
            servedDetail="full",
        )

    def get_frame_render(
        self,
        project: ProjectSummary,
        time_ma: int,
        *,
        exact: bool,
    ) -> FrameRender:
        run_id = project.currentRunId
        if run_id is None:
            replay = self._replay_frame(project, project.config, "ephemeral", time_ma)
            return build_frame_render_payload(
                replay.frame,
                source="generated",
                nearest_time_ma=time_ma,
            )

        context = self._resolve_context(project.projectId, run_id)
        run_config = context.config if context is not None else project.config
        index = self._load_or_build_timeline_index(project.projectId, run_id, run_config, context=context)

        nearest = self._nearest_time(index.times, time_ma)
        candidate_time = time_ma if (exact or time_ma in index.times) else nearest

        if candidate_time is not None:
            render = self._read_or_build_render_frame(project.projectId, run_id, candidate_time)
            if render is not None:
                source = "cache" if candidate_time in index.times else "generated"
                return render.model_copy(update={"source": source, "nearestTimeMa": nearest or candidate_time})

        replay = self._replay_frame(project, run_config, run_id, time_ma)
        frame_payload = replay.frame.model_dump(mode="json")
        self._backfill_time_index_entry(
            index=index,
            project_id=project.projectId,
            run_id=run_id,
            time_ma=time_ma,
            frame_payload=frame_payload,
        )
        self._write_timeline_index(project.projectId, run_id, index)

        if context is not None:
            context.diagnostics_by_time[time_ma] = replay.diagnostics

        return build_frame_render_payload(
            replay.frame,
            source="generated",
            nearest_time_ma=nearest or time_ma,
        )

    def get_timeline_index(self, project: ProjectSummary) -> TimelineIndex:
        if project.currentRunId is None:
            return TimelineIndex(
                projectId=project.projectId,
                runId="uninitialized",
                startTimeMa=project.config.startTimeMa,
                endTimeMa=project.config.endTimeMa,
                stepMyr=project.config.timeIncrementMyr,
                generatedOrder="descending_ma",
                times=self._time_steps(
                    project.config.startTimeMa,
                    project.config.endTimeMa,
                    project.config.timeIncrementMyr,
                ),
                hashes={},
                availableDetails=[],
            )

        run_id = project.currentRunId
        context = self._resolve_context(project.projectId, run_id)
        run_config = context.config if context else project.config
        return self._load_or_build_timeline_index(project.projectId, run_id, run_config, context=context)

    def get_frame_range(
        self,
        project: ProjectSummary,
        *,
        time_from: int,
        time_to: int,
        step: int,
        detail: str,
        exact: bool,
    ) -> FrameRangeResponse:
        if step <= 0:
            raise ValueError("step must be positive")
        if detail not in {"render", "full"}:
            raise ValueError("detail must be one of: render, full")

        direction = -1 if time_from >= time_to else 1
        values: list[int] = []
        current = time_from
        while True:
            values.append(current)
            if current == time_to:
                break
            next_value = current + (direction * step)
            if direction < 0 and next_value < time_to:
                next_value = time_to
            if direction > 0 and next_value > time_to:
                next_value = time_to
            current = next_value

        response = FrameRangeResponse(
            projectId=project.projectId,
            detail="render" if detail == "render" else "full",
            timeFrom=time_from,
            timeTo=time_to,
            step=step,
            generatedOrder="descending_ma",
        )

        if detail == "full":
            response.fullFrames = [self.get_or_create_frame(project, value) for value in values]
            return response

        response.renderFrames = [self.get_frame_render(project, value, exact=exact) for value in values]
        return response

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
        if frame_summary.frame.boundaryStates:
            continuity = [
                f"plate_{item.plateId}_continuity_low"
                for item in frame_summary.frame.plateKinematics
                if item.continuityScore < 0.2
            ]
            semantic = [
                f"{item.segmentId}_motion_mismatch"
                for item in frame_summary.frame.boundaryStates
                if item.motionMismatch
            ]
            checks = build_plausibility_checks_from_frame(frame_summary.frame)
            metrics = {
                "boundary.motion_mismatch_count": float(len(semantic)),
                "lifecycle.net_area_balance_error": float(
                    frame_summary.frame.plateLifecycleState.netAreaBalanceError
                    if frame_summary.frame.plateLifecycleState
                    else 0.0
                ),
            }
        else:
            continuity = collect_continuity_violations(frame_summary.frame)
            semantic = collect_boundary_semantic_issues(frame_summary.frame)
            checks = []
            metrics = {}
        diagnostics = FrameDiagnostics(
            projectId=project_id,
            timeMa=time_ma,
            continuityViolations=continuity,
            boundaryConsistencyIssues=semantic,
            coverageGapRatio=frame_summary.frame.uncertaintySummary.coverage,
            warnings=[],
            pygplatesStatus=context.pygplates_status if context else "unknown",
            metrics=metrics,
            checkIds=[item.checkId for item in checks],
        )
        self.project_store.write_run_diagnostics(project_id, run_id, time_ma, diagnostics.model_dump(mode="json"))
        if context:
            context.diagnostics_by_time[time_ma] = diagnostics
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

    def get_plausibility_report(self, project_id: str) -> PlausibilityReport:
        project = self.metadata.get_project(project_id)
        if project is None:
            raise ValueError(f"project {project_id} not found")

        if project.currentRunId is None:
            return PlausibilityReport(projectId=project_id, runId=None, checks=[], summary={"error": 0, "warning": 0, "info": 0})

        run_id = project.currentRunId
        context = self._resolve_context(project_id, run_id)
        checks: list[PlausibilityCheck] = []

        if context and context.plausibility_by_time:
            for time_ma in sorted(context.plausibility_by_time.keys(), reverse=True):
                checks.extend(context.plausibility_by_time[time_ma])
        else:
            run_config = context.config if context else project.config
            steps = self._time_steps(run_config.startTimeMa, run_config.endTimeMa, run_config.timeIncrementMyr)
            sample_times = steps[:: max(1, len(steps) // 12)]
            for time_ma in sample_times:
                frame_payload = self.project_store.read_frame(project_id, run_id, time_ma)
                if frame_payload is None:
                    continue
                frame = TimelineFrame.model_validate(frame_payload)
                checks.extend(build_plausibility_checks_from_frame(frame))

        summary = summarize_check_severity(checks)
        return PlausibilityReport(projectId=project_id, runId=run_id, checks=checks, summary=summary)

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
        solver_version = run_context.config.solverVersion if run_context else project.config.solverVersion
        model_version = self._solver_model_version(run_context.config if run_context else project.config)
        quality_mode = run_context.quality_mode if run_context else QualityMode.quick
        source_quick_run_id = run_context.source_quick_run_id if run_context else None
        macro_digest = run_context.macro_digest if run_context else ""

        kinematic_digest = stable_hash([kin.model_dump(mode="json") for kin in frame.plateKinematics])
        uncertainty_digest = stable_hash(frame.uncertaintySummary.model_dump(mode="json"))
        transition_rules_digest = stable_hash([item.model_dump(mode="json") for item in frame.boundaryStates])
        coefficients_digest = stable_hash(
            {
                "supercontinentBiasStrength": (run_context.config.supercontinentBiasStrength if run_context else project.config.supercontinentBiasStrength),
                "maxPlateVelocityCmYr": (run_context.config.maxPlateVelocityCmYr if run_context else project.config.maxPlateVelocityCmYr),
                "simulationMode": solver_mode.value,
            }
        )
        diagnostic_profile_digest = stable_hash(frame.plateLifecycleState.model_dump(mode="json") if frame.plateLifecycleState else {})

        provenance = ProvenanceRecord(
            projectHash=project.projectHash,
            seed=project.config.seed,
            engineVersion=ENGINE_VERSION,
            modelVersion=model_version,
            solverMode=solver_mode,
            rigorProfile=rigor_profile,
            parameterHash=stable_hash(request.model_dump(mode="json")),
            eventHash=stable_hash([event.model_dump(mode="json") for event in frame.eventOverlays]),
            kinematicDigest=kinematic_digest,
            uncertaintyDigest=uncertainty_digest,
            modelCoverage=coverage_ratio,
            solverVersion=solver_version,
            coefficientsDigest=coefficients_digest,
            transitionRulesDigest=transition_rules_digest,
            diagnosticProfileDigest=diagnostic_profile_digest,
            macroDigest=macro_digest,
            qualityMode=quality_mode,
            sourceQuickRunId=source_quick_run_id,
            surfaceProfileDigest=stable_hash({"qualityMode": quality_mode.value, "width": request.width, "height": request.height}),
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

        steps = self._time_steps(run_config.startTimeMa, run_config.endTimeMa, run_config.timeIncrementMyr)
        sample_times = steps[:6]
        previous_frame: TimelineFrame | None = None

        for time_ma in sample_times:
            frame_payload = self.project_store.read_frame(project_id, run_id, time_ma)
            if frame_payload is None:
                issues.append(
                    ValidationIssue(
                        code="missing_frame_cache",
                        severity="warning",
                        message=f"Frame {time_ma} Ma is missing from cache",
                        details={"timeMa": time_ma},
                    )
                )
                continue
            frame = TimelineFrame.model_validate(frame_payload)
            issues.extend(validate_frame(frame))
            issues.extend(validate_frame_pair(previous_frame, frame))
            if frame.boundaryStates:
                for check in build_plausibility_checks_from_frame(frame):
                    if check.severity not in {"error", "warning"}:
                        continue
                    issues.append(
                        ValidationIssue(
                            code=check.checkId,
                            severity=check.severity,
                            message=check.explanation,
                            details={
                                "observedValue": check.observedValue,
                                "expected": check.expectedRangeOrRule,
                                "timeRangeMa": list(check.timeRangeMa),
                            },
                        )
                    )
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
