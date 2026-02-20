export type FidelityPreset = "kinematic_rules" | "high_physics" | "procedural_light";
export type SimulationMode = "fast_plausible" | "hybrid_rigor";
export type RigorProfile = "balanced" | "research";
export type SolverVersion = "tectonic_hybrid_backends_v1" | "tectonic_state_v2";
export type QualityMode = "quick" | "full";

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
  solverVersion?: SolverVersion;
  coreGridWidth?: number | null;
  coreGridHeight?: number | null;
  supercontinentBiasStrength?: number;
  processProfiles?: Record<string, unknown>;
  enableLifecycleChecks?: boolean;
  highDetailWindowMyr?: number;
}

export interface ProjectSummary {
  projectId: string;
  name: string;
  config: ProjectConfig;
  createdAt: string;
  updatedAt: string;
  projectHash: string;
  currentRunId?: string | null;
  latestQuickRunId?: string | null;
  latestFullRunId?: string | null;
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

export type BoundaryStateClass = "ridge" | "rift" | "transform" | "subduction" | "collision" | "passive_margin" | "suture";

export interface BoundaryStateRecord {
  segmentId: string;
  stateClass: BoundaryStateClass;
  lastTransitionMa: number;
  typePersistenceMyr: number;
  polarityFlipCount: number;
  transitionCount: number;
  subductionFlux: number;
  averageOceanicAgeMyr: number;
  motionMismatch: boolean;
}

export interface PlateLifecycleState {
  unexplainedPlateBirths: number;
  unexplainedPlateDeaths: number;
  netAreaBalanceError: number;
  continentalAreaFraction: number;
  oceanicAreaFraction: number;
  oceanicAgeP99Myr: number;
  supercontinentPhase: "assembly" | "dispersal" | "stable" | "assembled";
  supercontinentLargestClusterFraction: number;
  supercontinentCycleCount: number;
  shortLivedOrogenyCount: number;
  uncoupledVolcanicBelts: number;
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
  boundaryStates?: BoundaryStateRecord[];
  plateLifecycleState?: PlateLifecycleState | null;
  strainFieldRef?: string | null;
  oceanicAgeFieldRef?: string | null;
  crustTypeFieldRef?: string | null;
  crustThicknessFieldRef?: string | null;
  tectonicPotentialFieldRef?: string | null;
  upliftRateFieldRef?: string | null;
  subsidenceRateFieldRef?: string | null;
  volcanicFluxFieldRef?: string | null;
  erosionCapacityFieldRef?: string | null;
  orogenicRootFieldRef?: string | null;
  cratonIdFieldRef?: string | null;
  moduleStateRef?: string | null;
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
  continentGeoJson?: GeoJsonFeatureCollection;
  cratonGeoJson?: GeoJsonFeatureCollection;
  boundaryGeoJson: GeoJsonFeatureCollection;
  overlayGeoJson: GeoJsonFeatureCollection;
  coastlineGeoJson?: GeoJsonFeatureCollection;
  activeBeltsGeoJson?: GeoJsonFeatureCollection;
  fieldStats?: Record<string, Record<string, number>>;
  reliefFieldRef?: string | null;
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
  metrics?: Record<string, number>;
  checkIds?: string[];
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
  severity: "error" | "warning" | "info";
  message: string;
  details?: Record<string, unknown>;
}

export interface ValidationReport {
  projectId: string;
  checkedAt: string;
  issues: ValidationIssue[];
}

export interface PlausibilityCheck {
  checkId: string;
  severity: "error" | "warning" | "info";
  timeRangeMa: [number, number];
  regionOrPlateIds: string[];
  observedValue: number | string;
  expectedRangeOrRule: string;
  explanation: string;
  suggestedFix: string;
}

export interface PlausibilityReport {
  projectId: string;
  runId?: string | null;
  checkedAt: string;
  checks: PlausibilityCheck[];
  summary: Record<string, number>;
}

export interface FieldSampleResponse {
  projectId: string;
  runId: string;
  timeMa: number;
  fieldName: string;
  width: number;
  height: number;
  sourceRef: string;
  stats: Record<string, number>;
  data: number[][];
}

export interface ModuleStepSnapshot {
  stepId: string;
  inputDigest: string;
  outputDigest: string;
  keyMetrics: Record<string, number>;
  transitionReasons: string[];
}

export interface ModuleStateResponse {
  projectId: string;
  runId: string;
  timeMa: number;
  replayHash: string;
  steps: ModuleStepSnapshot[];
}

export interface RunMetricsResponse {
  projectId: string;
  runId: string;
  frameCount: number;
  coverage: Record<string, number>;
  diagnostics: Record<string, number>;
  plausibility: Record<string, number>;
}
