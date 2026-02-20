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
    strainFieldRef: str | None = None
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


class ProjectSummary(BaseModel):
    projectId: str
    name: str
    config: ProjectConfig
    createdAt: str
    updatedAt: str
    projectHash: str
    currentRunId: str | None = None


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
    severity: Literal["error", "warning"]
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
    boundaryGeoJson: GeoJsonFeatureCollection
    overlayGeoJson: GeoJsonFeatureCollection
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
