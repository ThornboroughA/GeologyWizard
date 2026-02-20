export type FidelityPreset = "kinematic_rules" | "high_physics" | "procedural_light";

export interface PolygonGeometry {
  type: "Polygon";
  coordinates: number[][][];
}

export interface LineStringGeometry {
  type: "LineString";
  coordinates: number[][];
}

export interface ProjectConfig {
  seed: number;
  startTimeMa: number;
  endTimeMa: number;
  stepMyr: number;
  planetRadiusKm: number;
  plateCount: number;
  fidelityPreset: FidelityPreset;
  anchorPlateId?: number | null;
}

export interface ProjectSummary {
  projectId: string;
  name: string;
  config: ProjectConfig;
  createdAt: string;
  updatedAt: string;
  projectHash: string;
  currentRunId?: string | null;
}

export interface PlateFeature {
  plateId: number;
  name: string;
  geometry: PolygonGeometry;
  validTime: [number, number];
  reconstructionPlateId: number;
}

export interface BoundarySegment {
  segmentId: string;
  leftPlateId: number;
  rightPlateId: number;
  boundaryType: "convergent" | "divergent" | "transform";
  digitizationDirection: "forward" | "reverse";
  subductingSide: "left" | "right" | "none";
  isActive: boolean;
  geometry: LineStringGeometry;
}

export interface GeoEvent {
  eventId: string;
  eventType: "orogeny" | "rift" | "subduction" | "arc" | "terrane";
  timeStartMa: number;
  timeEndMa: number;
  intensity: number;
  sourceBoundaryIds: string[];
  regionGeometry: LineStringGeometry | PolygonGeometry;
}

export interface TimelineFrame {
  timeMa: number;
  plateGeometries: PlateFeature[];
  boundaryGeometries: BoundarySegment[];
  eventOverlays: GeoEvent[];
  previewHeightFieldRef: string;
}

export interface FrameSummary {
  frame: TimelineFrame;
  frameHash: string;
  source: "cache" | "generated";
}

export interface Bookmark {
  bookmarkId: string;
  timeMa: number;
  label: string;
  region?: PolygonGeometry;
  refinementState: "pending" | "ready" | "failed";
  parentFrameHash: string;
}

export interface ExportArtifact {
  artifactId: string;
  type: "heightmap" | "metadata";
  format: "png16" | "tiff32" | "json";
  width: number;
  height: number;
  bitDepth: number;
  path: string;
  checksum: string;
}

export interface JobSummary {
  jobId: string;
  projectId: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed" | "canceled";
  progress: number;
  message: string;
  artifacts: ExportArtifact[];
  error?: string | null;
}

export interface ValidationIssue {
  code: string;
  severity: "error" | "warning";
  message: string;
}

export interface ValidationReport {
  projectId: string;
  checkedAt: string;
  issues: ValidationIssue[];
}
