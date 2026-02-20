import type {
  Bookmark,
  CoverageReport,
  FrameDiagnostics,
  FrameRangeResponse,
  FrameSummary,
  JobSummary,
  PlausibilityReport,
  ProjectConfig,
  ProjectSummary,
  QualityMode,
  TimelineFrameRender,
  TimelineIndex,
  RigorProfile,
  SimulationMode,
  ValidationReport
} from "../types";

const ENGINE_URL = import.meta.env.VITE_ENGINE_BASE_URL ?? "http://127.0.0.1:8765";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${ENGINE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${message}`);
  }
  return (await response.json()) as T;
}

export function healthCheck(): Promise<{ status: string }> {
  return request<{ status: string }>("/health");
}

export function createProject(name: string, config: ProjectConfig): Promise<ProjectSummary> {
  return request<ProjectSummary>("/v1/projects", {
    method: "POST",
    body: JSON.stringify({ name, config })
  });
}

export function createProjectV2(name: string, config: ProjectConfig): Promise<ProjectSummary> {
  return request<ProjectSummary>("/v2/projects", {
    method: "POST",
    body: JSON.stringify({ name, config })
  });
}

export function getProject(projectId: string): Promise<ProjectSummary> {
  return request<ProjectSummary>(`/v1/projects/${projectId}`);
}

export function getProjectV2(projectId: string): Promise<ProjectSummary> {
  return request<ProjectSummary>(`/v2/projects/${projectId}`);
}

export function generateProject(
  projectId: string,
  options?: {
    simulationModeOverride?: SimulationMode;
    rigorProfileOverride?: RigorProfile;
    targetRuntimeMinutesOverride?: number;
    qualityMode?: QualityMode;
    sourceQuickRunId?: string;
  }
): Promise<JobSummary> {
  return request<JobSummary>(`/v1/projects/${projectId}/generate`, {
    method: "POST",
    body: JSON.stringify({
      runLabel: "ui",
      ...(options ?? {})
    })
  });
}

export function generateProjectV2(
  projectId: string,
  options?: {
    simulationModeOverride?: SimulationMode;
    rigorProfileOverride?: RigorProfile;
    targetRuntimeMinutesOverride?: number;
    qualityMode?: QualityMode;
    sourceQuickRunId?: string;
  }
): Promise<JobSummary> {
  return request<JobSummary>(`/v2/projects/${projectId}/generate`, {
    method: "POST",
    body: JSON.stringify({
      runLabel: "ui",
      ...(options ?? {})
    })
  });
}

export function getFrame(projectId: string, timeMa: number, signal?: AbortSignal): Promise<FrameSummary> {
  return request<FrameSummary>(`/v1/projects/${projectId}/frames/${timeMa}`, {
    signal
  });
}

export function getFrameV2(projectId: string, timeMa: number, signal?: AbortSignal): Promise<FrameSummary> {
  return request<FrameSummary>(`/v2/projects/${projectId}/frames/${timeMa}`, {
    signal
  });
}

export function getFrameDiagnostics(projectId: string, timeMa: number, signal?: AbortSignal): Promise<FrameDiagnostics> {
  return request<FrameDiagnostics>(`/v1/projects/${projectId}/frames/${timeMa}/diagnostics`, {
    signal
  });
}

export function getFrameDiagnosticsV2(projectId: string, timeMa: number, signal?: AbortSignal): Promise<FrameDiagnostics> {
  return request<FrameDiagnostics>(`/v2/projects/${projectId}/frames/${timeMa}/diagnostics`, {
    signal
  });
}

export function getTimelineIndex(projectId: string, signal?: AbortSignal): Promise<TimelineIndex> {
  return request<TimelineIndex>(`/v1/projects/${projectId}/timeline-index`, {
    signal
  });
}

export function getTimelineIndexV2(projectId: string, signal?: AbortSignal): Promise<TimelineIndex> {
  return request<TimelineIndex>(`/v2/projects/${projectId}/timeline-index`, {
    signal
  });
}

export function getFramesRange(
  projectId: string,
  options: {
    timeFrom: number;
    timeTo: number;
    step?: number;
    detail?: "render" | "full";
    exact?: boolean;
  },
  signal?: AbortSignal
): Promise<FrameRangeResponse> {
  const params = new URLSearchParams({
    time_from: String(options.timeFrom),
    time_to: String(options.timeTo),
    step: String(options.step ?? 1),
    detail: options.detail ?? "render",
    exact: String(Boolean(options.exact))
  });
  return request<FrameRangeResponse>(`/v1/projects/${projectId}/frames?${params.toString()}`, {
    signal
  });
}

export function getFramesRangeV2(
  projectId: string,
  options: {
    timeFrom: number;
    timeTo: number;
    step?: number;
    detail?: "render" | "full";
    exact?: boolean;
  },
  signal?: AbortSignal
): Promise<FrameRangeResponse> {
  const params = new URLSearchParams({
    time_from: String(options.timeFrom),
    time_to: String(options.timeTo),
    step: String(options.step ?? 1),
    detail: options.detail ?? "render",
    exact: String(Boolean(options.exact))
  });
  return request<FrameRangeResponse>(`/v2/projects/${projectId}/frames?${params.toString()}`, {
    signal
  });
}

export async function getRenderFrame(
  projectId: string,
  timeMa: number,
  options?: { exact?: boolean; signal?: AbortSignal }
): Promise<TimelineFrameRender> {
  const response = await getFramesRange(
    projectId,
    {
      timeFrom: timeMa,
      timeTo: timeMa,
      step: 1,
      detail: "render",
      exact: options?.exact ?? false
    },
    options?.signal
  );
  return response.renderFrames[0];
}

export async function getRenderFrameV2(
  projectId: string,
  timeMa: number,
  options?: { exact?: boolean; signal?: AbortSignal }
): Promise<TimelineFrameRender> {
  const response = await getFramesRangeV2(
    projectId,
    {
      timeFrom: timeMa,
      timeTo: timeMa,
      step: 1,
      detail: "render",
      exact: options?.exact ?? false
    },
    options?.signal
  );
  return response.renderFrames[0];
}

export function getCoverage(projectId: string): Promise<CoverageReport> {
  return request<CoverageReport>(`/v1/projects/${projectId}/coverage`);
}

export function createBookmark(projectId: string, timeMa: number, label: string): Promise<Bookmark> {
  return request<Bookmark>(`/v1/projects/${projectId}/bookmarks`, {
    method: "POST",
    body: JSON.stringify({ timeMa, label })
  });
}

export function listBookmarks(projectId: string): Promise<Bookmark[]> {
  return request<Bookmark[]>(`/v1/projects/${projectId}/bookmarks`);
}

export function refineBookmark(projectId: string, bookmarkId: string, resolution: "2k" | "4k" | "8k"): Promise<JobSummary> {
  return request<JobSummary>(`/v1/projects/${projectId}/bookmarks/${bookmarkId}/refine`, {
    method: "POST",
    body: JSON.stringify({ resolution, refinementLevel: 1 })
  });
}

export function exportBookmark(projectId: string, bookmarkId: string, width = 8192, height = 4096): Promise<JobSummary> {
  return request<JobSummary>(`/v1/projects/${projectId}/exports`, {
    method: "POST",
    body: JSON.stringify({
      bookmarkId,
      format: "png16",
      width,
      height,
      bitDepth: 16
    })
  });
}

export function applyExpertEdit(projectId: string, payload: object): Promise<{ impactedTimesMa: number[]; editCount: number }> {
  return request<{ impactedTimesMa: number[]; editCount: number }>(`/v1/projects/${projectId}/edits`, {
    method: "POST",
    body: JSON.stringify({ edits: [payload] })
  });
}

export function getJob(jobId: string): Promise<JobSummary> {
  return request<JobSummary>(`/v1/jobs/${jobId}`);
}

export function getValidation(projectId: string): Promise<ValidationReport> {
  return request<ValidationReport>(`/v1/projects/${projectId}/validation`);
}

export function getPlausibilityV2(projectId: string): Promise<PlausibilityReport> {
  return request<PlausibilityReport>(`/v2/projects/${projectId}/plausibility`);
}
