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
    ProjectConfig,
    ProjectSummary,
    ProvenanceRecord,
    RefineResult,
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
        timeline_index = self._empty_timeline_index(project_id, run_id, run_config)

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
                coverage_ratio = round((replay.coverage_ratio + frame_coverage_ratio(replay.frame)) * 0.5, 4)
                context.coverage_by_time[time_ma] = coverage_ratio
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
