from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SimulationMode(str, Enum):
    fast_plausible = "fast_plausible"
    hybrid_rigor = "hybrid_rigor"


class RigorProfile(str, Enum):
    balanced = "balanced"
    research = "research"


class SolverVersion(str, Enum):
    tectonic_hybrid_backends_v1 = "tectonic_hybrid_backends_v1"
    tectonic_state_v2 = "tectonic_state_v2"


class QualityMode(str, Enum):
    quick = "quick"
    full = "full"


class ProjectConfig(BaseModel):
    seed: int = 42
    startTimeMa: int = 1000
    endTimeMa: int = 0
    stepMyr: int = 1
    timeIncrementMyr: int = 1
    planetRadiusKm: float = 6371.0
    plateCount: int = 14
    fidelityPreset: Literal["kinematic_rules", "high_physics", "procedural_light"] = "kinematic_rules"
    simulationMode: SimulationMode = SimulationMode.fast_plausible
    rigorProfile: RigorProfile = RigorProfile.balanced
    targetRuntimeMinutes: int = 60
    maxPlateVelocityCmYr: float = 14.0
    anchorPlateId: int | None = None
    solverVersion: SolverVersion = SolverVersion.tectonic_hybrid_backends_v1
    coreGridWidth: int | None = None
    coreGridHeight: int | None = None
    supercontinentBiasStrength: float = 0.5
    processProfiles: dict[str, Any] = Field(default_factory=dict)
    enableLifecycleChecks: bool = True
    highDetailWindowMyr: int = 30

    @model_validator(mode="after")
    def validate_range(self) -> "ProjectConfig":
        if self.startTimeMa <= self.endTimeMa:
            raise ValueError("startTimeMa must be greater than endTimeMa")
        if self.stepMyr <= 0:
            raise ValueError("stepMyr must be positive")
        if self.timeIncrementMyr <= 0:
            raise ValueError("timeIncrementMyr must be positive")
        if self.stepMyr != self.timeIncrementMyr:
            # Keep backward compatibility with existing UI payloads using stepMyr.
            self.stepMyr = self.timeIncrementMyr
        if self.plateCount < 4 or self.plateCount > 64:
            raise ValueError("plateCount must be between 4 and 64")
        if self.targetRuntimeMinutes < 5 or self.targetRuntimeMinutes > 720:
            raise ValueError("targetRuntimeMinutes must be between 5 and 720")
        if self.maxPlateVelocityCmYr <= 0:
            raise ValueError("maxPlateVelocityCmYr must be positive")
        if self.supercontinentBiasStrength < 0 or self.supercontinentBiasStrength > 1:
            raise ValueError("supercontinentBiasStrength must be between 0 and 1")
        if self.highDetailWindowMyr < 10 or self.highDetailWindowMyr > 100:
            raise ValueError("highDetailWindowMyr must be between 10 and 100")
        if self.solverVersion == SolverVersion.tectonic_state_v2:
            if self.coreGridWidth is None:
                self.coreGridWidth = 720 if self.simulationMode == SimulationMode.hybrid_rigor else 512
            if self.coreGridHeight is None:
                self.coreGridHeight = 360 if self.simulationMode == SimulationMode.hybrid_rigor else 256
        if self.coreGridWidth is not None and self.coreGridWidth < 128:
            raise ValueError("coreGridWidth must be at least 128")
        if self.coreGridHeight is not None and self.coreGridHeight < 64:
            raise ValueError("coreGridHeight must be at least 64")
        return self


class SeedBundle(BaseModel):
    plates: int
    boundaries: int
    events: int
    terrain: int


class PlateFeature(BaseModel):
    plateId: int
    name: str
    geometry: dict[str, Any]
    validTime: tuple[float, float]
    reconstructionPlateId: int


class PlateKinematics(BaseModel):
    plateId: int
    velocityCmYr: float
    azimuthDeg: float
    convergenceCmYr: float
    divergenceCmYr: float
    continuityScore: float


class BoundaryType(str, Enum):
    convergent = "convergent"
    divergent = "divergent"
    transform = "transform"


class BoundaryStateClass(str, Enum):
    ridge = "ridge"
    rift = "rift"
    transform = "transform"
    subduction = "subduction"
    collision = "collision"
    passive_margin = "passive_margin"
    suture = "suture"


class BoundarySegment(BaseModel):
    segmentId: str
    leftPlateId: int
    rightPlateId: int
    boundaryType: BoundaryType
    digitizationDirection: Literal["forward", "reverse"] = "forward"
    subductingSide: Literal["left", "right", "none"] = "none"
    isActive: bool = True
    geometry: dict[str, Any]


class BoundaryKinematics(BaseModel):
    segmentId: str
    relativeVelocityCmYr: float
    normalVelocityCmYr: float
    tangentialVelocityCmYr: float
    strainRate: float
    recommendedBoundaryType: BoundaryType


class GeoEventType(str, Enum):
    orogeny = "orogeny"
    rift = "rift"
    subduction = "subduction"
    arc = "arc"
    terrane = "terrane"


class GeoEvent(BaseModel):
    eventId: str
    eventType: GeoEventType
    timeStartMa: float
    timeEndMa: float
    intensity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    drivingMetrics: dict[str, float] = Field(default_factory=dict)
    persistenceClass: Literal["transient", "sustained", "long_lived"] = "transient"
    sourceBoundaryIds: list[str] = Field(default_factory=list)
    regionGeometry: dict[str, Any]


class BoundaryStateRecord(BaseModel):
    segmentId: str
    stateClass: BoundaryStateClass
    lastTransitionMa: int
    typePersistenceMyr: int
    polarityFlipCount: int = 0
    transitionCount: int = 0
    subductionFlux: float = 0.0
    averageOceanicAgeMyr: float = 0.0
    motionMismatch: bool = False


class PlateLifecycleState(BaseModel):
    unexplainedPlateBirths: int = 0
    unexplainedPlateDeaths: int = 0
    netAreaBalanceError: float = 0.0
    continentalAreaFraction: float = 0.0
    oceanicAreaFraction: float = 0.0
    oceanicAgeP99Myr: float = 0.0
    supercontinentPhase: Literal["assembly", "dispersal", "stable", "assembled"] = "stable"
    supercontinentLargestClusterFraction: float = 0.0
    supercontinentCycleCount: int = 0
    shortLivedOrogenyCount: int = 0
    uncoupledVolcanicBelts: int = 0


class UncertaintySummary(BaseModel):
    kinematic: float = Field(ge=0.0, le=1.0)
    event: float = Field(ge=0.0, le=1.0)
    terrain: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)


class TimelineFrame(BaseModel):
    timeMa: int
    plateGeometries: list[PlateFeature]
    boundaryGeometries: list[BoundarySegment]
    eventOverlays: list[GeoEvent]
    plateKinematics: list[PlateKinematics] = Field(default_factory=list)
    boundaryKinematics: list[BoundaryKinematics] = Field(default_factory=list)
    boundaryStates: list[BoundaryStateRecord] = Field(default_factory=list)
    plateLifecycleState: PlateLifecycleState | None = None
    strainFieldRef: str | None = None
    oceanicAgeFieldRef: str | None = None
    crustTypeFieldRef: str | None = None
    crustThicknessFieldRef: str | None = None
    tectonicPotentialFieldRef: str | None = None
    upliftRateFieldRef: str | None = None
    subsidenceRateFieldRef: str | None = None
    volcanicFluxFieldRef: str | None = None
    erosionCapacityFieldRef: str | None = None
    orogenicRootFieldRef: str | None = None
    cratonIdFieldRef: str | None = None
    moduleStateRef: str | None = None
    uncertaintySummary: UncertaintySummary
    previewHeightFieldRef: str


class GeoJsonFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict[str, Any]
    properties: dict[str, Any] = Field(default_factory=dict)


class GeoJsonFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[GeoJsonFeature] = Field(default_factory=list)


class Bookmark(BaseModel):
    bookmarkId: str
    timeMa: int
    label: str
    region: dict[str, Any] | None = None
    refinementState: Literal["pending", "ready", "failed"] = "pending"
    parentFrameHash: str


class ExportArtifact(BaseModel):
    artifactId: str
    type: Literal["heightmap", "metadata"]
    format: Literal["png16", "tiff32", "json"]
    width: int
    height: int
    bitDepth: int
    path: str
    checksum: str


class ProvenanceRecord(BaseModel):
    projectHash: str
    seed: int
    engineVersion: str
    modelVersion: str
    solverMode: SimulationMode
    rigorProfile: RigorProfile
    parameterHash: str
    eventHash: str
    kinematicDigest: str
    uncertaintyDigest: str
    modelCoverage: float
    solverVersion: SolverVersion = SolverVersion.tectonic_hybrid_backends_v1
    coefficientsDigest: str = ""
    transitionRulesDigest: str = ""
    diagnosticProfileDigest: str = ""
    macroDigest: str = ""
    qualityMode: QualityMode = QualityMode.quick
    sourceQuickRunId: str | None = None
    surfaceProfileDigest: str = ""


class ProjectSummary(BaseModel):
    projectId: str
    name: str
    config: ProjectConfig
    createdAt: str
    updatedAt: str
    projectHash: str
    currentRunId: str | None = None
    latestQuickRunId: str | None = None
    latestFullRunId: str | None = None


class ProjectCreateRequest(BaseModel):
    name: str = Field(default="Untitled World")
    config: ProjectConfig = Field(default_factory=ProjectConfig)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


class JobSummary(BaseModel):
    jobId: str
    projectId: str
    kind: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    artifacts: list[ExportArtifact] = Field(default_factory=list)
    error: str | None = None


class GenerateRequest(BaseModel):
    runLabel: str = "default"
    simulationModeOverride: SimulationMode | None = None
    rigorProfileOverride: RigorProfile | None = None
    targetRuntimeMinutesOverride: int | None = None
    qualityMode: QualityMode = QualityMode.quick
    sourceQuickRunId: str | None = None


class BookmarkCreateRequest(BaseModel):
    timeMa: int
    label: str = "Bookmark"
    region: dict[str, Any] | None = None


class BookmarkRefineRequest(BaseModel):
    resolution: Literal["2k", "4k", "8k"] = "8k"
    refinementLevel: int = 1


class ExpertEdit(BaseModel):
    timeMa: int
    editType: Literal[
        "rift_initiation",
        "boundary_override",
        "subducting_side_override",
        "event_gain",
        "rift_start",
        "event_boost",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)


class ExpertEditRequest(BaseModel):
    edits: list[ExpertEdit]


class ExportRequest(BaseModel):
    timeMa: int | None = None
    bookmarkId: str | None = None
    format: Literal["png16", "tiff32"] = "png16"
    width: int = 8192
    height: int = 4096
    bitDepth: Literal[16, 32] = 16
    region: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_time_or_bookmark(self) -> "ExportRequest":
        if self.timeMa is None and self.bookmarkId is None:
            raise ValueError("either timeMa or bookmarkId is required")
        return self


class ValidationIssue(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    projectId: str
    checkedAt: str = Field(default_factory=utc_now_iso)
    issues: list[ValidationIssue] = Field(default_factory=list)


class TimelineIndexHashEntry(BaseModel):
    full: str
    render: str


class TimelineIndex(BaseModel):
    projectId: str
    runId: str
    startTimeMa: int
    endTimeMa: int
    stepMyr: int
    generatedOrder: Literal["descending_ma"]
    times: list[int] = Field(default_factory=list)
    hashes: dict[str, TimelineIndexHashEntry] = Field(default_factory=dict)
    availableDetails: list[Literal["render", "full"]] = Field(default_factory=lambda: ["render", "full"])


class FrameRender(BaseModel):
    timeMa: int
    landmassGeoJson: GeoJsonFeatureCollection
    continentGeoJson: GeoJsonFeatureCollection = Field(default_factory=GeoJsonFeatureCollection)
    cratonGeoJson: GeoJsonFeatureCollection = Field(default_factory=GeoJsonFeatureCollection)
    boundaryGeoJson: GeoJsonFeatureCollection
    overlayGeoJson: GeoJsonFeatureCollection
    coastlineGeoJson: GeoJsonFeatureCollection = Field(default_factory=GeoJsonFeatureCollection)
    activeBeltsGeoJson: GeoJsonFeatureCollection = Field(default_factory=GeoJsonFeatureCollection)
    fieldStats: dict[str, dict[str, float]] = Field(default_factory=dict)
    reliefFieldRef: str | None = None
    source: Literal["cache", "generated"]
    nearestTimeMa: int


class FrameSummary(BaseModel):
    frame: TimelineFrame
    frameHash: str
    source: Literal["cache", "generated"]
    nearestAvailableTimeMa: int | None = None
    servedDetail: Literal["full", "render"] = "full"


class FrameRangeResponse(BaseModel):
    projectId: str
    detail: Literal["render", "full"]
    timeFrom: int
    timeTo: int
    step: int
    generatedOrder: Literal["descending_ma"]
    fullFrames: list[FrameSummary] = Field(default_factory=list)
    renderFrames: list[FrameRender] = Field(default_factory=list)


class FrameDiagnostics(BaseModel):
    projectId: str
    timeMa: int
    continuityViolations: list[str] = Field(default_factory=list)
    boundaryConsistencyIssues: list[str] = Field(default_factory=list)
    coverageGapRatio: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    pygplatesStatus: str = "unavailable"
    metrics: dict[str, float] = Field(default_factory=dict)
    checkIds: list[str] = Field(default_factory=list)


class PlausibilityCheck(BaseModel):
    checkId: str
    severity: Literal["error", "warning", "info"]
    timeRangeMa: tuple[int, int]
    regionOrPlateIds: list[str] = Field(default_factory=list)
    observedValue: float | int | str
    expectedRangeOrRule: str
    explanation: str
    suggestedFix: str


class PlausibilityReport(BaseModel):
    projectId: str
    runId: str | None = None
    checkedAt: str = Field(default_factory=utc_now_iso)
    checks: list[PlausibilityCheck] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class CoverageReport(BaseModel):
    projectId: str
    globalCoverageRatio: float
    coverageRatioByTime: list[dict[str, float]] = Field(default_factory=list)
    fallbackTimesMa: list[int] = Field(default_factory=list)
    pygplatesAvailable: bool = False


class RefineResult(BaseModel):
    bookmark: Bookmark
    cachePath: str


class ExportResult(BaseModel):
    artifacts: list[ExportArtifact]
    provenance: ProvenanceRecord


class FieldSampleResponse(BaseModel):
    projectId: str
    runId: str
    timeMa: int
    fieldName: str
    width: int
    height: int
    sourceRef: str
    stats: dict[str, float]
    data: list[list[float]]


class ModuleStepSnapshot(BaseModel):
    stepId: str
    inputDigest: str
    outputDigest: str
    keyMetrics: dict[str, float] = Field(default_factory=dict)
    transitionReasons: list[str] = Field(default_factory=list)


class ModuleStateResponse(BaseModel):
    projectId: str
    runId: str
    timeMa: int
    replayHash: str
    steps: list[ModuleStepSnapshot] = Field(default_factory=list)


class RunMetricsResponse(BaseModel):
    projectId: str
    runId: str
    frameCount: int
    coverage: dict[str, float]
    diagnostics: dict[str, float]
    plausibility: dict[str, int]
