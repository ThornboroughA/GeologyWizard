import { useEffect, useMemo, useRef, useState } from "react";

import {
  applyExpertEdit,
  createBookmark,
  createProjectV2,
  exportBookmark,
  generateProjectV2,
  getCoverage,
  getFrameV2,
  getFrameDiagnosticsV2,
  getFramesRangeV2,
  getProjectV2,
  getPlausibilityV2,
  getJob,
  getRenderFrameV2,
  getTimelineIndexV2,
  getValidation,
  listBookmarks,
  refineBookmark
} from "./lib/api";
import { JobList } from "./components/JobList";
import { MapScene } from "./components/MapScene";
import { TimelineScrubber } from "./components/TimelineScrubber";
import type {
  Bookmark,
  CoverageReport,
  FrameDiagnostics,
  JobSummary,
  PlausibilityReport,
  ProjectConfig,
  ProjectSummary,
  QualityMode,
  TimelineFrame,
  TimelineFrameRender,
  TimelineIndex,
  ValidationReport
} from "./types";

const DEFAULT_CONFIG: ProjectConfig = {
  seed: 42,
  startTimeMa: 1000,
  endTimeMa: 0,
  stepMyr: 1,
  timeIncrementMyr: 1,
  planetRadiusKm: 6371,
  plateCount: 14,
  fidelityPreset: "kinematic_rules",
  simulationMode: "fast_plausible",
  rigorProfile: "balanced",
  targetRuntimeMinutes: 60,
  maxPlateVelocityCmYr: 14,
  anchorPlateId: null,
  solverVersion: "tectonic_state_v2",
  coreGridWidth: 512,
  coreGridHeight: 256,
  supercontinentBiasStrength: 0.5,
  enableLifecycleChecks: true,
  highDetailWindowMyr: 30
};

const SETTLE_DELAY_MS = 220;
const PREFETCH_WINDOW_MA = 40;
const PREFETCH_STEP_MA = 5;

function mergeJobs(existing: JobSummary[], updates: JobSummary[]): JobSummary[] {
  const map = new Map(existing.map((job) => [job.jobId, job]));
  for (const update of updates) {
    map.set(update.jobId, update);
  }
  return Array.from(map.values()).sort((a, b) => b.jobId.localeCompare(a.jobId));
}

export default function App() {
  const [projectName, setProjectName] = useState("Asterion");
  const [config, setConfig] = useState<ProjectConfig>(DEFAULT_CONFIG);
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [qualityMode, setQualityMode] = useState<QualityMode>("quick");
  const [showExpertControls, setShowExpertControls] = useState(false);

  const [timeMa, setTimeMa] = useState(1000);
  const [scrubState, setScrubState] = useState<"idle" | "dragging" | "settling">("idle");
  const [frameSource, setFrameSource] = useState<"cached_nearest" | "exact">("exact");
  const [displayedTimeMa, setDisplayedTimeMa] = useState<number | null>(null);

  const [frame, setFrame] = useState<TimelineFrame | null>(null);
  const [renderFrame, setRenderFrame] = useState<TimelineFrameRender | null>(null);
  const [timelineIndex, setTimelineIndex] = useState<TimelineIndex | null>(null);

  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [diagnostics, setDiagnostics] = useState<FrameDiagnostics | null>(null);
  const [coverage, setCoverage] = useState<CoverageReport | null>(null);
  const [plausibility, setPlausibility] = useState<PlausibilityReport | null>(null);

  const [bookmarkLabel, setBookmarkLabel] = useState("Key tectonic phase");
  const [selectedBookmarkId, setSelectedBookmarkId] = useState<string>("");
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");
  const [overlay, setOverlay] = useState<
    "none" | "velocity" | "boundary_class" | "event_confidence" | "uplift" | "subsidence" | "boundary_state" | "crust_age" | "craton" | "orogeny_phase" | "subduction_flux"
  >("none");
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);

  const settleTimerRef = useRef<number | null>(null);
  const requestIdRef = useRef(0);
  const activeControllerRef = useRef<AbortController | null>(null);
  const renderCacheRef = useRef<Map<number, TimelineFrameRender>>(new Map());

  const runningJobIds = useMemo(
    () => jobs.filter((job) => ["queued", "running"].includes(job.status)).map((job) => job.jobId),
    [jobs]
  );

  function clearSettleTimer() {
    if (settleTimerRef.current !== null) {
      window.clearTimeout(settleTimerRef.current);
      settleTimerRef.current = null;
    }
  }

  function cacheRenderFrame(payload: TimelineFrameRender) {
    renderCacheRef.current.set(payload.timeMa, payload);
  }

  function nearestCachedTime(target: number): number | null {
    const times = Array.from(renderCacheRef.current.keys());
    if (times.length === 0) {
      return null;
    }
    return times.reduce((best, current) => (Math.abs(current - target) < Math.abs(best - target) ? current : best), times[0]);
  }

  function showNearestCachedFrame(target: number) {
    const nearest = nearestCachedTime(target);
    if (nearest === null) {
      return;
    }
    const cached = renderCacheRef.current.get(nearest);
    if (!cached) {
      return;
    }
    setRenderFrame(cached);
    setDisplayedTimeMa(cached.timeMa);
    setFrameSource("cached_nearest");
  }

  async function prefetchRenderWindow(projectId: string, centerTimeMa: number) {
    const startTimeMa = timelineIndex?.startTimeMa ?? project?.config.startTimeMa;
    const endTimeMa = timelineIndex?.endTimeMa ?? project?.config.endTimeMa;
    if (startTimeMa == null || endTimeMa == null) {
      return;
    }

    const upper = Math.min(startTimeMa, centerTimeMa + PREFETCH_WINDOW_MA);
    const lower = Math.max(endTimeMa, centerTimeMa - PREFETCH_WINDOW_MA);
    try {
      const response = await getFramesRangeV2(projectId, {
        timeFrom: upper,
        timeTo: lower,
        step: PREFETCH_STEP_MA,
        detail: "render",
        exact: false
      });
      for (const framePayload of response.renderFrames) {
        cacheRenderFrame(framePayload);
      }
    } catch {
      // Best-effort prefetch only.
    }
  }

  async function fetchExactFrame(projectId: string, targetTimeMa: number) {
    const requestId = ++requestIdRef.current;

    activeControllerRef.current?.abort();
    const controller = new AbortController();
    activeControllerRef.current = controller;

    setScrubState("settling");
    try {
      const [exactRender, fullFrame, frameDiagnostics] = await Promise.all([
        getRenderFrameV2(projectId, targetTimeMa, { exact: true, signal: controller.signal }),
        getFrameV2(projectId, targetTimeMa, controller.signal),
        getFrameDiagnosticsV2(projectId, targetTimeMa, controller.signal).catch(() => null)
      ]);

      if (controller.signal.aborted || requestId !== requestIdRef.current) {
        return;
      }

      cacheRenderFrame(exactRender);
      setRenderFrame(exactRender);
      setDisplayedTimeMa(exactRender.timeMa);
      setFrameSource("exact");

      setFrame(fullFrame.frame);
      setDiagnostics(frameDiagnostics);
      setScrubState("idle");
      setStatus(`Showing ${targetTimeMa} Ma`);

      void prefetchRenderWindow(projectId, targetTimeMa);
    } catch (err) {
      if (controller.signal.aborted || requestId !== requestIdRef.current) {
        return;
      }
      setScrubState("idle");
      setError((err as Error).message);
    }
  }

  function scheduleSettle(targetTimeMa: number) {
    clearSettleTimer();
    if (!project) {
      return;
    }
    settleTimerRef.current = window.setTimeout(() => {
      settleTimerRef.current = null;
      void fetchExactFrame(project.projectId, targetTimeMa);
    }, SETTLE_DELAY_MS);
  }

  function handleTimelineChange(nextTimeMa: number) {
    setTimeMa(nextTimeMa);
    setScrubState("dragging");
    showNearestCachedFrame(nextTimeMa);
    scheduleSettle(nextTimeMa);
  }

  function handleScrubStart() {
    setScrubState("dragging");
  }

  function handleScrubEnd() {
    clearSettleTimer();
    if (!project) {
      return;
    }
    void fetchExactFrame(project.projectId, timeMa);
  }

  useEffect(() => {
    return () => {
      clearSettleTimer();
      activeControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (runningJobIds.length === 0) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const updates = await Promise.all(runningJobIds.map((jobId) => getJob(jobId)));
        setJobs((current) => mergeJobs(current, updates));

        const hasRefreshEvent = updates.some((job) => ["generate", "refine", "export"].includes(job.kind) && job.status === "completed");
        const hasGenerateCompletion = updates.some((job) => job.kind === "generate" && job.status === "completed");
        if (project && hasRefreshEvent) {
          if (hasGenerateCompletion) {
            renderCacheRef.current.clear();
          }
          await refreshProjectPanels(project.projectId, timeMa);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [project, runningJobIds, timeMa]);

  async function refreshProjectPanels(projectId: string, frameTime: number) {
    const [projectSummary, bookmarkList, validationReport, coverageReport, nextTimelineIndex, plausibilityReport] = await Promise.all([
      getProjectV2(projectId),
      listBookmarks(projectId),
      getValidation(projectId),
      getCoverage(projectId),
      getTimelineIndexV2(projectId),
      getPlausibilityV2(projectId)
    ]);

    setProject(projectSummary);
    setBookmarks(bookmarkList);
    setValidation(validationReport);
    setCoverage(coverageReport);
    setTimelineIndex(nextTimelineIndex);
    setPlausibility(plausibilityReport);

    await fetchExactFrame(projectId, frameTime);
  }

  async function handleCreateProject() {
    setError(null);
    setStatus("Creating project");
    try {
      const created = await createProjectV2(projectName, {
        ...config,
        solverVersion: "tectonic_state_v2"
      });
      setProject(created);
      setTimeMa(created.config.startTimeMa);
      setJobs([]);
      setBookmarks([]);
      setSelectedBookmarkId("");
      setValidation(null);
      setCoverage(null);
      setPlausibility(null);
      setDiagnostics(null);
      setFrame(null);
      setRenderFrame(null);
      setDisplayedTimeMa(null);
      setFrameSource("exact");
      renderCacheRef.current.clear();

      const index = await getTimelineIndexV2(created.projectId);
      setTimelineIndex(index);

      await fetchExactFrame(created.projectId, created.config.startTimeMa);
      setStatus(`Project ${created.name} ready`);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleGenerate() {
    if (!project) {
      return;
    }
    setError(null);
    setStatus(`Queued ${qualityMode} generation`);
    try {
      const sourceQuickRunId =
        qualityMode === "full"
          ? project.latestQuickRunId ?? undefined
          : undefined;
      const job = await generateProjectV2(project.projectId, {
        simulationModeOverride: config.simulationMode,
        rigorProfileOverride: config.rigorProfile,
        targetRuntimeMinutesOverride: config.targetRuntimeMinutes,
        qualityMode,
        sourceQuickRunId
      });
      setJobs((current) => mergeJobs(current, [job]));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleAddBookmark() {
    if (!project) {
      return;
    }
    setError(null);
    try {
      const bookmark = await createBookmark(project.projectId, timeMa, bookmarkLabel);
      const nextBookmarks = [bookmark, ...bookmarks];
      setBookmarks(nextBookmarks);
      setSelectedBookmarkId(bookmark.bookmarkId);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleRefineBookmark() {
    if (!project || !selectedBookmarkId) {
      return;
    }
    setError(null);
    try {
      const job = await refineBookmark(project.projectId, selectedBookmarkId, "8k");
      setJobs((current) => mergeJobs(current, [job]));
      setStatus("Bookmark refinement started");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleExportBookmark() {
    if (!project || !selectedBookmarkId) {
      return;
    }
    setError(null);
    try {
      const job = await exportBookmark(project.projectId, selectedBookmarkId);
      setJobs((current) => mergeJobs(current, [job]));
      setStatus("Export started");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleExpertEdit(editType: "rift_initiation" | "boundary_override" | "subducting_side_override" | "event_gain") {
    if (!project || !frame) {
      return;
    }

    setError(null);
    try {
      if (editType === "rift_initiation") {
        await applyExpertEdit(project.projectId, {
          timeMa,
          editType,
          payload: {
            plateId: frame.plateGeometries[0]?.plateId,
            azimuthDelta: 22,
            speedGain: 0.9,
            durationMyr: 24
          }
        });
      } else if (editType === "boundary_override") {
        const boundary = frame.boundaryGeometries[0];
        await applyExpertEdit(project.projectId, {
          timeMa,
          editType,
          payload: {
            segmentId: boundary?.segmentId,
            boundaryType: "transform",
            durationMyr: 18
          }
        });
      } else if (editType === "subducting_side_override") {
        const convergent = frame.boundaryGeometries.find((item) => item.boundaryType === "convergent");
        await applyExpertEdit(project.projectId, {
          timeMa,
          editType,
          payload: {
            segmentId: convergent?.segmentId,
            subductingSide: "left",
            durationMyr: 22
          }
        });
      } else {
        await applyExpertEdit(project.projectId, {
          timeMa,
          editType,
          payload: {
            gain: 0.14,
            durationMyr: 35
          }
        });
      }

      await refreshProjectPanels(project.projectId, timeMa);
      setStatus(`Applied ${editType} at ${timeMa} Ma`);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const runtimeMessage =
    qualityMode === "quick"
      ? "Quick mode: fast preview run for iteration."
      : "Full mode: reuses quick macro history with higher-fidelity surface realization.";

  const mapSummary = renderFrame
    ? `${renderFrame.continentGeoJson?.features.length ?? renderFrame.landmassGeoJson.features.length} continents, ${renderFrame.coastlineGeoJson?.features.length ?? 0} coastline segments, ${renderFrame.cratonGeoJson?.features.length ?? 0} craton cores, ${renderFrame.activeBeltsGeoJson?.features.length ?? renderFrame.boundaryGeoJson.features.length} active belts`
    : "No frame loaded";

  return (
    <div className="app-shell">
      <header className="hero">
        <h1>Geologic Wizard</h1>
        <p>Earth-like tectonic worldbuilding with tiered simulation rigor and deterministic history replay.</p>
      </header>

      <section className="workspace-grid">
        <aside className="card left-panel">
          <h2>Guided Setup</h2>
          <label>
            World Name
            <input value={projectName} onChange={(event) => setProjectName(event.target.value)} />
          </label>
          <label>
            Seed
            <input
              type="number"
              value={config.seed}
              onChange={(event) => setConfig((current) => ({ ...current, seed: Number(event.target.value) }))}
            />
          </label>
          <label>
            Plate Count
            <input
              type="number"
              value={config.plateCount}
              min={4}
              max={64}
              onChange={(event) => setConfig((current) => ({ ...current, plateCount: Number(event.target.value) }))}
            />
          </label>
          <label>
            Run Mode
            <select value={qualityMode} onChange={(event) => setQualityMode(event.target.value as QualityMode)}>
              <option value="quick">quick</option>
              <option value="full">full</option>
            </select>
          </label>
          <details open={showExpertControls} onToggle={(event) => setShowExpertControls((event.target as HTMLDetailsElement).open)}>
            <summary>Expert Controls</summary>
            <label>
              Simulation Mode
              <select
                value={config.simulationMode}
                onChange={(event) =>
                  setConfig((current) => ({
                    ...current,
                    simulationMode: event.target.value as ProjectConfig["simulationMode"],
                    coreGridWidth: event.target.value === "hybrid_rigor" ? 720 : 512,
                    coreGridHeight: event.target.value === "hybrid_rigor" ? 360 : 256
                  }))
                }
              >
                <option value="fast_plausible">fast_plausible</option>
                <option value="hybrid_rigor">hybrid_rigor</option>
              </select>
            </label>
            <label>
              Rigor Profile
              <select
                value={config.rigorProfile}
                onChange={(event) =>
                  setConfig((current) => ({
                    ...current,
                    rigorProfile: event.target.value as ProjectConfig["rigorProfile"]
                  }))
                }
              >
                <option value="balanced">balanced</option>
                <option value="research">research</option>
              </select>
            </label>
            <label>
              Runtime Target (minutes)
              <input
                type="number"
                min={5}
                max={720}
                value={config.targetRuntimeMinutes}
                onChange={(event) =>
                  setConfig((current) => ({ ...current, targetRuntimeMinutes: Number(event.target.value) }))
                }
              />
            </label>
            <label>
              Max Plate Velocity (cm/yr)
              <input
                type="number"
                min={3}
                max={20}
                step={0.5}
                value={config.maxPlateVelocityCmYr}
                onChange={(event) =>
                  setConfig((current) => ({ ...current, maxPlateVelocityCmYr: Number(event.target.value) }))
                }
              />
            </label>
            <label>
              Core Grid Width
              <input
                type="number"
                min={256}
                max={2048}
                step={16}
                value={config.coreGridWidth ?? 512}
                onChange={(event) =>
                  setConfig((current) => ({ ...current, coreGridWidth: Number(event.target.value) }))
                }
              />
            </label>
            <label>
              Core Grid Height
              <input
                type="number"
                min={128}
                max={1024}
                step={16}
                value={config.coreGridHeight ?? 256}
                onChange={(event) =>
                  setConfig((current) => ({ ...current, coreGridHeight: Number(event.target.value) }))
                }
              />
            </label>
            <label>
              Supercontinent Bias
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={config.supercontinentBiasStrength ?? 0.5}
                onChange={(event) =>
                  setConfig((current) => ({ ...current, supercontinentBiasStrength: Number(event.target.value) }))
                }
              />
            </label>
          </details>
          <p className="muted">{runtimeMessage}</p>
          <div className="button-row">
            <button onClick={handleCreateProject}>Create Project</button>
            <button onClick={handleGenerate} disabled={!project}>
              Run {qualityMode}
            </button>
          </div>

          <h3>Focused Expert Panel</h3>
          <p className="muted">Constrained tectonic interventions with deterministic recompute windows.</p>
          <div className="expert-grid">
            <button onClick={() => void handleExpertEdit("rift_initiation")} disabled={!project || !frame}>
              Initiate Rift
            </button>
            <button onClick={() => void handleExpertEdit("boundary_override")} disabled={!project || !frame}>
              Override Boundary
            </button>
            <button onClick={() => void handleExpertEdit("subducting_side_override")} disabled={!project || !frame}>
              Set Subducting Side
            </button>
            <button onClick={() => void handleExpertEdit("event_gain")} disabled={!project || !frame}>
              Event Gain
            </button>
          </div>

          <h3>Bookmarks</h3>
          <label>
            Label
            <input value={bookmarkLabel} onChange={(event) => setBookmarkLabel(event.target.value)} />
          </label>
          <button onClick={handleAddBookmark} disabled={!project}>
            Add Bookmark at {timeMa} Ma
          </button>
          <select value={selectedBookmarkId} onChange={(event) => setSelectedBookmarkId(event.target.value)}>
            <option value="">Select Bookmark</option>
            {bookmarks.map((bookmark) => (
              <option key={bookmark.bookmarkId} value={bookmark.bookmarkId}>
                {bookmark.label} ({bookmark.timeMa} Ma)
              </option>
            ))}
          </select>
          <div className="button-row">
            <button onClick={handleRefineBookmark} disabled={!selectedBookmarkId}>
              Refine 8K Region
            </button>
            <button onClick={handleExportBookmark} disabled={!selectedBookmarkId}>
              Export Heightmap
            </button>
          </div>
        </aside>

        <main className="card center-panel">
          <TimelineScrubber
            value={timeMa}
            min={project?.config.endTimeMa ?? 0}
            max={project?.config.startTimeMa ?? 1000}
            step={project?.config.timeIncrementMyr ?? 1}
            requestedTime={timeMa}
            displayedTime={displayedTimeMa}
            frameSource={frameSource}
            scrubState={scrubState}
            onChange={handleTimelineChange}
            onScrubStart={handleScrubStart}
            onScrubEnd={handleScrubEnd}
          />

          <div className="map-toolbar">
            <div className="segmented">
              <button className={viewMode === "2d" ? "active" : ""} onClick={() => setViewMode("2d")}>
                2D Projection
              </button>
              <button className={viewMode === "3d" ? "active" : ""} onClick={() => setViewMode("3d")}>
                3D Globe
              </button>
            </div>
            <label>
              Overlay
              <select value={overlay} onChange={(event) => setOverlay(event.target.value as typeof overlay)}>
                <option value="none">none</option>
                <option value="velocity">velocity</option>
                <option value="boundary_class">boundary_class</option>
                <option value="event_confidence">event_confidence</option>
                <option value="uplift">uplift</option>
                <option value="subsidence">subsidence</option>
                <option value="boundary_state">boundary_state</option>
                <option value="crust_age">crust_age</option>
                <option value="craton">craton</option>
                <option value="orogeny_phase">orogeny_phase</option>
                <option value="subduction_flux">subduction_flux</option>
              </select>
            </label>
            <p className="muted">{mapSummary}</p>
          </div>

          <MapScene frame={renderFrame} mode={viewMode} overlay={overlay} />

          {frame ? (
            <div className="metrics-grid">
              <div>
                <strong>Uncertainty</strong>
                <p>Kinematic: {frame.uncertaintySummary.kinematic.toFixed(2)}</p>
                <p>Event: {frame.uncertaintySummary.event.toFixed(2)}</p>
                <p>Terrain: {frame.uncertaintySummary.terrain.toFixed(2)}</p>
                <p>Coverage: {frame.uncertaintySummary.coverage.toFixed(2)}</p>
              </div>
              <div>
                <strong>Kinematics</strong>
                <p>
                  Avg velocity: {(
                    frame.plateKinematics.reduce((sum, item) => sum + item.velocityCmYr, 0) /
                    Math.max(1, frame.plateKinematics.length)
                  ).toFixed(2)}{" "}
                  cm/yr
                </p>
                <p>
                  Avg continuity: {(
                    frame.plateKinematics.reduce((sum, item) => sum + item.continuityScore, 0) /
                    Math.max(1, frame.plateKinematics.length)
                  ).toFixed(2)}
                </p>
              </div>
              <div>
                <strong>Diagnostics</strong>
                <p>PyGPlates: {diagnostics?.pygplatesStatus ?? "n/a"}</p>
                <p>Coverage gap ratio: {diagnostics?.coverageGapRatio.toFixed(2) ?? "n/a"}</p>
                <p>Continuity alerts: {diagnostics?.continuityViolations.length ?? 0}</p>
              </div>
              {frame.plateLifecycleState ? (
                <div>
                  <strong>Lifecycle</strong>
                  <p>Oceanic age p99: {frame.plateLifecycleState.oceanicAgeP99Myr.toFixed(1)} Myr</p>
                  <p>Supercontinent: {frame.plateLifecycleState.supercontinentPhase}</p>
                  <p>Cycle count: {frame.plateLifecycleState.supercontinentCycleCount}</p>
                </div>
              ) : null}
            </div>
          ) : null}
        </main>

        <aside className="card right-panel">
          <h2>Simulation Jobs</h2>
          <JobList jobs={jobs} />

          <h2>Coverage</h2>
          {coverage ? (
            <div className="project-details">
              <p>Global coverage: {(coverage.globalCoverageRatio * 100).toFixed(1)}%</p>
              <p>PyGPlates available: {coverage.pygplatesAvailable ? "yes" : "no"}</p>
              <p>Fallback times: {coverage.fallbackTimesMa.length}</p>
            </div>
          ) : (
            <p className="muted">Coverage appears after generation.</p>
          )}

          <h2>Validation</h2>
          {validation ? (
            validation.issues.length > 0 ? (
              <ul className="issues-list">
                {validation.issues.map((issue, index) => (
                  <li key={`${issue.code}-${index}`} className={issue.severity}>
                    <strong>{issue.code}</strong>
                    <span>{issue.message}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">No validation findings on sampled frames.</p>
            )
          ) : (
            <p className="muted">Validation report appears after first generation or edit.</p>
          )}

          <h2>Plausibility</h2>
          {plausibility ? (
            <div className="project-details">
              <p>Errors: {plausibility.summary.error ?? 0}</p>
              <p>Warnings: {plausibility.summary.warning ?? 0}</p>
              <p>Info: {plausibility.summary.info ?? 0}</p>
            </div>
          ) : (
            <p className="muted">Plausibility report appears for v2 projects after generation.</p>
          )}

          <h2>Timeline Cache</h2>
          {timelineIndex ? (
            <div className="project-details">
              <p>Cached frames: {timelineIndex.times.length}</p>
              <p>Order: {timelineIndex.generatedOrder}</p>
              <p>
                Step: {timelineIndex.stepMyr} Myr ({timelineIndex.startTimeMa} {"->"} {timelineIndex.endTimeMa} Ma)
              </p>
            </div>
          ) : (
            <p className="muted">Timeline index loads after project creation.</p>
          )}

          <h2>Project</h2>
          {project ? (
            <div className="project-details">
              <p>
                <strong>{project.name}</strong>
              </p>
              <p>Seed: {project.config.seed}</p>
              <p>Workflow: quick/full</p>
              <p>
                Range: {project.config.startTimeMa} Ma to {project.config.endTimeMa} Ma
              </p>
              <p>Requested Time: {timeMa} Ma</p>
              <p>Displayed Time: {displayedTimeMa ?? "n/a"} Ma</p>
            </div>
          ) : (
            <p className="muted">Create a project to begin.</p>
          )}
        </aside>
      </section>

      <footer className="status-bar">
        <span>{status}</span>
        {error ? <span className="error">{error}</span> : null}
      </footer>
    </div>
  );
}
