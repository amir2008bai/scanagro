from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.db.models import ProcessingStatus
from app.schemas.vision import NormalizedBBox


class BBoxOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PlateOut(BaseModel):
    """Plate geometry and reading, reported separately from the vehicle."""

    bbox: BBoxOut | None
    quad: list[list[float]] | None
    detection_confidence: float | None
    ocr_confidence: float | None
    text: str | None
    text_raw: str | None
    readable: bool
    format: str | None
    region: str | None
    ocr_engine: str | None
    ocr_variant: str | None
    rectified: bool | None
    perspective_skew: float | None
    status: str | None
    crop_filename: str | None
    rectified_crop_filename: str | None


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    image_id: UUID
    job_id: UUID | None
    vehicle_type: str
    manufacturer: str | None
    model: str | None
    license_plate: str | None
    plate_readable: bool
    confidence: float | None
    bbox: BBoxOut
    extra: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    # Added alongside the local pipeline; null for providers that do not produce them.
    vehicle_confidence: float | None = None
    manufacturer_confidence: float | None = None
    source_label: str | None = None
    #: reference_gallery | badge_text | detector — read this before trusting the type.
    vehicle_type_source: str | None = None
    vehicle_type_confidence: float | None = None
    plate: PlateOut | None = None


class DetectionPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    vehicle_type: str | None = Field(default=None, min_length=1, max_length=120)
    manufacturer: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    license_plate: str | None = Field(default=None, max_length=64)
    plate_readable: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: NormalizedBBox | None = None
    #: Correct where the plate actually is. Pass null to drop a wrong plate box.
    plate_bbox: NormalizedBBox | None = None

    @model_validator(mode="after")
    def reject_null_required_fields(self):
        for field in ("vehicle_type", "plate_readable", "bbox"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    mime_type: str
    file_size: int
    sha256: str
    width: int
    height: int
    annotated_filename: str | None
    created_at: datetime
    updated_at: datetime


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    image_id: UUID
    status: ProcessingStatus
    provider: str
    model: str
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    attempts: int
    created_at: datetime
    updated_at: datetime


class UploadItem(BaseModel):
    image: ImageOut
    job: JobOut | None


class BatchUploadResponse(BaseModel):
    items: list[UploadItem]
    count: int


class PaginatedDetections(BaseModel):
    items: list[DetectionOut]
    total: int
    limit: int
    offset: int
