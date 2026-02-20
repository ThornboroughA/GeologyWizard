import { useEffect, useMemo, useState } from "react";

import {
  applyExpertEdit,
  createBookmark,
  createProject,
  exportBookmark,
  generateProject,
  getFrame,
  getJob,
  getValidation,
  listBookmarks,
  refineBookmark
} from "./lib/api";
import { JobList } from "./components/JobList";
import { MapScene } from "./components/MapScene";
import { TimelineScrubber } from "./components/TimelineScrubber";
import type { Bookmark, JobSummary, ProjectConfig, ProjectSummary, TimelineFrame, ValidationReport } from "./types";

const DEFAULT_CONFIG: ProjectConfig = {
  seed: 42,
  startTimeMa: 1000,
  endTimeMa: 0,
  stepMyr: 1,
  planetRadiusKm: 6371,
  plateCount: 14,
  fidelityPreset: "kinematic_rules",
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
  const [bookmarkLabel, setBookmarkLabel] = useState("Key tectonic phase");
  const [selectedBookmarkId, setSelectedBookmarkId] = useState<string>("");
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");
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

        const hasGenerateComplete = updates.some((job) => job.kind === "generate" && job.status === "completed");
        const hasRefineComplete = updates.some((job) => job.kind === "refine" && job.status === "completed");

        if (project && (hasGenerateComplete || hasRefineComplete)) {
          const [frameSummary, bookmarkList, validationReport] = await Promise.all([
            getFrame(project.projectId, timeMa),
            listBookmarks(project.projectId),
            getValidation(project.projectId)
          ]);
          setFrame(frameSummary.frame);
          setBookmarks(bookmarkList);
          setValidation(validationReport);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [project, runningJobIds, timeMa]);

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
    setStatus("Queued world generation");
    try {
      const job = await generateProject(project.projectId);
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

  async function handleExpertNudge() {
    if (!project) {
      return;
    }
    setError(null);
    try {
      await applyExpertEdit(project.projectId, {
        timeMa,
        editType: "event_boost",
        payload: {
          boost: 0.12,
          note: "UI expert nudge"
        }
      });
      const [frameSummary, validationReport] = await Promise.all([
        getFrame(project.projectId, timeMa),
        getValidation(project.projectId)
      ]);
      setFrame(frameSummary.frame);
      setValidation(validationReport);
      setStatus(`Expert nudge applied at ${timeMa} Ma`);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <h1>Geologic Wizard</h1>
        <p>Earth-like tectonic worldbuilding for non-experts, from 1000 Ma to present.</p>
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
          <div className="button-row">
            <button onClick={handleCreateProject}>Create Project</button>
            <button onClick={handleGenerate} disabled={!project}>
              Generate Timeline
            </button>
          </div>

          <h3>Focused Expert Panel</h3>
          <p className="muted">Apply constrained tectonic nudges without exposing full low-level controls.</p>
          <button onClick={handleExpertNudge} disabled={!project}>
            Boost Local Event Intensity
          </button>

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
            step={project?.config.stepMyr ?? 1}
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
            <p className="muted">
              {frame
                ? `${frame.plateGeometries.length} plates, ${frame.boundaryGeometries.length} boundaries, ${frame.eventOverlays.length} active events`
                : "No frame loaded"}
            </p>
          </div>

          <MapScene frame={frame} mode={viewMode} />
        </main>

        <aside className="card right-panel">
          <h2>Simulation Jobs</h2>
          <JobList jobs={jobs} />

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
