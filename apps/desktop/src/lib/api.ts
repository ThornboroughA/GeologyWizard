import type {
  Bookmark,
  CoverageReport,
  FrameDiagnostics,
  FrameSummary,
  JobSummary,
  ProjectConfig,
  ProjectSummary,
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

export function generateProject(
  projectId: string,
  options?: {
    simulationModeOverride?: SimulationMode;
    rigorProfileOverride?: RigorProfile;
    targetRuntimeMinutesOverride?: number;
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

export function getFrame(projectId: string, timeMa: number): Promise<FrameSummary> {
  return request<FrameSummary>(`/v1/projects/${projectId}/frames/${timeMa}`);
}

export function getFrameDiagnostics(projectId: string, timeMa: number): Promise<FrameDiagnostics> {
  return request<FrameDiagnostics>(`/v1/projects/${projectId}/frames/${timeMa}/diagnostics`);
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
