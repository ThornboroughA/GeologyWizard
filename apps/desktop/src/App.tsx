import { useEffect, useMemo, useState } from "react";

import {
  applyExpertEdit,
  createBookmark,
  createProject,
  exportBookmark,
  generateProject,
  getCoverage,
  getFrame,
  getFrameDiagnostics,
  getJob,
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
  ProjectConfig,
  ProjectSummary,
  TimelineFrame,
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
  anchorPlateId: null
};

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
  const [timeMa, setTimeMa] = useState(1000);
  const [frame, setFrame] = useState<TimelineFrame | null>(null);
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [diagnostics, setDiagnostics] = useState<FrameDiagnostics | null>(null);
  const [coverage, setCoverage] = useState<CoverageReport | null>(null);
  const [bookmarkLabel, setBookmarkLabel] = useState("Key tectonic phase");
  const [selectedBookmarkId, setSelectedBookmarkId] = useState<string>("");
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");
  const [overlay, setOverlay] = useState<"none" | "velocity" | "boundary_class" | "event_confidence" | "uplift" | "subsidence">("none");
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);

  const runningJobIds = useMemo(
    () => jobs.filter((job) => ["queued", "running"].includes(job.status)).map((job) => job.jobId),
    [jobs]
  );

  useEffect(() => {
    if (!project) {
      return;
    }
    const timer = window.setTimeout(async () => {
      try {
        const summary = await getFrame(project.projectId, timeMa);
        setFrame(summary.frame);
        try {
          const frameDiagnostics = await getFrameDiagnostics(project.projectId, timeMa);
          setDiagnostics(frameDiagnostics);
        } catch {
          setDiagnostics(null);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    }, 180);

    return () => window.clearTimeout(timer);
  }, [project, timeMa]);

  useEffect(() => {
    if (runningJobIds.length === 0) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const updates = await Promise.all(runningJobIds.map((jobId) => getJob(jobId)));
        setJobs((current) => mergeJobs(current, updates));

        const hasRefreshEvent = updates.some((job) => ["generate", "refine", "export"].includes(job.kind) && job.status === "completed");
        if (project && hasRefreshEvent) {
          const [frameSummary, bookmarkList, validationReport, coverageReport, frameDiagnostics] = await Promise.all([
            getFrame(project.projectId, timeMa),
            listBookmarks(project.projectId),
            getValidation(project.projectId),
            getCoverage(project.projectId),
            getFrameDiagnostics(project.projectId, timeMa)
          ]);
          setFrame(frameSummary.frame);
          setBookmarks(bookmarkList);
          setValidation(validationReport);
          setCoverage(coverageReport);
          setDiagnostics(frameDiagnostics);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [project, runningJobIds, timeMa]);

  async function refreshProjectPanels(projectId: string, frameTime: number) {
    const [frameSummary, bookmarkList, validationReport, coverageReport] = await Promise.all([
      getFrame(projectId, frameTime),
      listBookmarks(projectId),
      getValidation(projectId),
      getCoverage(projectId)
    ]);
    setFrame(frameSummary.frame);
    setBookmarks(bookmarkList);
    setValidation(validationReport);
    setCoverage(coverageReport);
    try {
      const frameDiagnostics = await getFrameDiagnostics(projectId, frameTime);
      setDiagnostics(frameDiagnostics);
    } catch {
      setDiagnostics(null);
    }
  }

  async function handleCreateProject() {
    setError(null);
    setStatus("Creating project");
    try {
      const created = await createProject(projectName, config);
      setProject(created);
      setTimeMa(created.config.startTimeMa);
      setJobs([]);
      setBookmarks([]);
      setSelectedBookmarkId("");
      setValidation(null);
      setCoverage(null);
      setDiagnostics(null);
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
    setStatus(`Queued ${config.simulationMode} generation`);
    try {
      const job = await generateProject(project.projectId, {
        simulationModeOverride: config.simulationMode,
        rigorProfileOverride: config.rigorProfile,
        targetRuntimeMinutesOverride: config.targetRuntimeMinutes
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
    config.simulationMode === "hybrid_rigor"
      ? "Hybrid rigor mode: better physical plausibility, potentially longer runtime than fast mode."
      : "Fast plausible mode: optimized for iteration speed with controlled geologic approximations.";

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
            Simulation Mode
            <select
              value={config.simulationMode}
              onChange={(event) =>
                setConfig((current) => ({
                  ...current,
                  simulationMode: event.target.value as ProjectConfig["simulationMode"]
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
          <p className="muted">{runtimeMessage}</p>
          <div className="button-row">
            <button onClick={handleCreateProject}>Create Project</button>
            <button onClick={handleGenerate} disabled={!project}>
              Generate Timeline
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
            onChange={setTimeMa}
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
              </select>
            </label>
            <p className="muted">
              {frame
                ? `${frame.plateGeometries.length} plates, ${frame.boundaryGeometries.length} boundaries, ${frame.eventOverlays.length} active events`
                : "No frame loaded"}
            </p>
          </div>

          <MapScene frame={frame} mode={viewMode} overlay={overlay} />

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
              <p className="muted">No validation findings on sampled keyframes.</p>
            )
          ) : (
            <p className="muted">Validation report appears after first generation or edit.</p>
          )}

          <h2>Project</h2>
          {project ? (
            <div className="project-details">
              <p>
                <strong>{project.name}</strong>
              </p>
              <p>Seed: {project.config.seed}</p>
              <p>
                Range: {project.config.startTimeMa} Ma to {project.config.endTimeMa} Ma
              </p>
              <p>Current Time: {timeMa} Ma</p>
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
