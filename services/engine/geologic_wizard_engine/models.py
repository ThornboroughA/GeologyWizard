from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectConfig(BaseModel):
    seed: int = 42
    startTimeMa: int = 1000
    endTimeMa: int = 0
    stepMyr: int = 1
    planetRadiusKm: float = 6371.0
    plateCount: int = 14
    fidelityPreset: Literal["kinematic_rules", "high_physics", "procedural_light"] = "kinematic_rules"
    anchorPlateId: int | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "ProjectConfig":
        if self.startTimeMa <= self.endTimeMa:
            raise ValueError("startTimeMa must be greater than endTimeMa")
        if self.stepMyr <= 0:
            raise ValueError("stepMyr must be positive")
        if self.plateCount < 4 or self.plateCount > 64:
            raise ValueError("plateCount must be between 4 and 64")
        return self


class PlateFeature(BaseModel):
    plateId: int
    name: str
    geometry: dict[str, Any]
    validTime: tuple[float, float]
    reconstructionPlateId: int


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
    sourceBoundaryIds: list[str] = Field(default_factory=list)
    regionGeometry: dict[str, Any]


class TimelineFrame(BaseModel):
    timeMa: int
    plateGeometries: list[PlateFeature]
    boundaryGeometries: list[BoundarySegment]
    eventOverlays: list[GeoEvent]
    previewHeightFieldRef: str


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
    parameterHash: str
    eventHash: str


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


class BookmarkCreateRequest(BaseModel):
    timeMa: int
    label: str = "Bookmark"
    region: dict[str, Any] | None = None


class BookmarkRefineRequest(BaseModel):
    resolution: Literal["2k", "4k", "8k"] = "8k"
    refinementLevel: int = 1


class ExpertEdit(BaseModel):
    timeMa: int
    editType: Literal["rift_start", "boundary_override", "event_boost"]
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


class FrameSummary(BaseModel):
    frame: TimelineFrame
    frameHash: str
    source: Literal["cache", "generated"]


class RefineResult(BaseModel):
    bookmark: Bookmark
    cachePath: str


class ExportResult(BaseModel):
    artifacts: list[ExportArtifact]
    provenance: ProvenanceRecord
