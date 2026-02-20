export type FidelityPreset = "kinematic_rules" | "high_physics" | "procedural_light";
export type SimulationMode = "fast_plausible" | "hybrid_rigor";
export type RigorProfile = "balanced" | "research";

export interface PolygonGeometry {
  type: "Polygon";
  coordinates: number[][][];
}

export interface MultiPolygonGeometry {
  type: "MultiPolygon";
  coordinates: number[][][][];
}

export interface LineStringGeometry {
  type: "LineString";
  coordinates: number[][];
}

export interface MultiLineStringGeometry {
  type: "MultiLineString";
  coordinates: number[][][];
}

export interface ProjectConfig {
  seed: number;
  startTimeMa: number;
  endTimeMa: number;
  stepMyr: number;
  timeIncrementMyr: number;
  planetRadiusKm: number;
  plateCount: number;
  fidelityPreset: FidelityPreset;
  simulationMode: SimulationMode;
  rigorProfile: RigorProfile;
  targetRuntimeMinutes: number;
  maxPlateVelocityCmYr: number;
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

export interface PlateKinematics {
  plateId: number;
  velocityCmYr: number;
  azimuthDeg: number;
  convergenceCmYr: number;
  divergenceCmYr: number;
  continuityScore: number;
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

export interface BoundaryKinematics {
  segmentId: string;
  relativeVelocityCmYr: number;
  normalVelocityCmYr: number;
  tangentialVelocityCmYr: number;
  strainRate: number;
  recommendedBoundaryType: "convergent" | "divergent" | "transform";
}

export interface GeoEvent {
  eventId: string;
  eventType: "orogeny" | "rift" | "subduction" | "arc" | "terrane";
  timeStartMa: number;
  timeEndMa: number;
  intensity: number;
  confidence: number;
  drivingMetrics: Record<string, number>;
  persistenceClass: "transient" | "sustained" | "long_lived";
  sourceBoundaryIds: string[];
  regionGeometry: LineStringGeometry | PolygonGeometry;
}

export interface UncertaintySummary {
  kinematic: number;
  event: number;
  terrain: number;
  coverage: number;
}

export interface TimelineFrame {
  timeMa: number;
  plateGeometries: PlateFeature[];
  boundaryGeometries: BoundarySegment[];
  eventOverlays: GeoEvent[];
  plateKinematics: PlateKinematics[];
  boundaryKinematics: BoundaryKinematics[];
  strainFieldRef?: string | null;
  uncertaintySummary: UncertaintySummary;
  previewHeightFieldRef: string;
}

export interface FrameSummary {
  frame: TimelineFrame;
  frameHash: string;
  source: "cache" | "generated";
  nearestAvailableTimeMa?: number | null;
  servedDetail: "full" | "render";
}

export interface GeoJsonFeature {
  type: "Feature";
  geometry: PolygonGeometry | MultiPolygonGeometry | LineStringGeometry | MultiLineStringGeometry;
  properties: Record<string, unknown>;
}

export interface GeoJsonFeatureCollection {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
}

export interface TimelineFrameRender {
  timeMa: number;
  landmassGeoJson: GeoJsonFeatureCollection;
  boundaryGeoJson: GeoJsonFeatureCollection;
  overlayGeoJson: GeoJsonFeatureCollection;
  source: "cache" | "generated";
  nearestTimeMa: number;
}

export interface TimelineIndexHashEntry {
  full: string;
  render: string;
}

export interface TimelineIndex {
  projectId: string;
  runId: string;
  startTimeMa: number;
  endTimeMa: number;
  stepMyr: number;
  generatedOrder: "descending_ma";
  times: number[];
  hashes: Record<string, TimelineIndexHashEntry>;
  availableDetails: Array<"render" | "full">;
}

export interface FrameRangeResponse {
  projectId: string;
  detail: "render" | "full";
  timeFrom: number;
  timeTo: number;
  step: number;
  generatedOrder: "descending_ma";
  fullFrames: FrameSummary[];
  renderFrames: TimelineFrameRender[];
}

export interface FrameDiagnostics {
  projectId: string;
  timeMa: number;
  continuityViolations: string[];
  boundaryConsistencyIssues: string[];
  coverageGapRatio: number;
  warnings: string[];
  pygplatesStatus: string;
}

export interface CoverageReport {
  projectId: string;
  globalCoverageRatio: number;
  coverageRatioByTime: Array<{ timeMa: number; coverageRatio: number }>;
  fallbackTimesMa: number[];
  pygplatesAvailable: boolean;
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
