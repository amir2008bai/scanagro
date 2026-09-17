from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NormalizedBBox(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have x2>x1 and y2>y1")
        return self


class PlateObservation(BaseModel):
    """One licence plate, with its own geometry and its own confidences.

    ``bbox`` is the plate, not the vehicle: the two are deliberately separate, so a
    consumer can draw the plate box, crop it, or check that the plate really sits on the
    machine it was attributed to. ``detection_confidence`` comes from the plate detector
    and ``ocr_confidence`` is measured from the recognition model's per-character
    probabilities — they answer different questions and are never merged.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    bbox: NormalizedBBox
    #: Four normalised corners of the plate, in image coordinates, when geometry succeeded.
    quad: list[list[float]] | None = None
    detection_confidence: float = Field(ge=0.0, le=1.0)
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Accepted text in display form. ``None`` whenever the plate was not accepted.
    text: str | None = Field(default=None, max_length=64)
    #: Best raw OCR string, kept even when rejected, so a human can review the call.
    text_raw: str | None = Field(default=None, max_length=64)
    readable: bool = False
    plate_format: str | None = Field(default=None, max_length=40)
    region: str | None = Field(default=None, max_length=60)
    ocr_engine: str | None = Field(default=None, max_length=40)
    ocr_variant: str | None = Field(default=None, max_length=40)
    #: True when the accepted read came from the perspective-corrected crop.
    rectified: bool = False
    #: Departure from a parallelogram, in plate heights. ~0 means rotation would do.
    perspective_skew: float | None = Field(default=None, ge=0.0, le=10.0)
    status: str = Field(default="unknown", max_length=60)
    crop_filename: str | None = Field(default=None, max_length=255)
    rectified_crop_filename: str | None = Field(default=None, max_length=255)
    alternatives: list[dict[str, Any]] = Field(default_factory=list, max_length=12)
    disagreement_positions: list[int] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def unreadable_has_no_text(self):
        if not self.readable:
            self.text = None
        return self


class VisionDetection(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    vehicle_type: str = Field(min_length=1, max_length=120)
    manufacturer: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    license_plate: str | None = Field(default=None, max_length=64)
    plate_readable: bool = False
    #: Retained for API compatibility; mirrors ``vehicle_confidence``.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: NormalizedBBox
    notes: str | None = Field(default=None, max_length=1000)

    # --- added by the local pipeline; all optional so older payloads still validate ---
    vehicle_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    manufacturer_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    plate: PlateObservation | None = None
    #: Further plates found on the same machine, e.g. a towed trailer's own plate.
    extra_plates: list[PlateObservation] = Field(default_factory=list, max_length=4)
    #: Raw detector class before mapping, kept as evidence for the reported type.
    source_label: str | None = Field(default=None, max_length=60)
    #: Which stage settled the type: reference_gallery | badge_text | detector.
    #: Worth reading before trusting `vehicle_type`: `detector` means COCO's coarse
    #: guess, which has no class for agricultural machinery.
    vehicle_type_source: str | None = Field(default=None, max_length=40)
    #: Confidence of whichever stage decided the type — a cosine similarity for the
    #: gallery, an OCR score for a badge, a detector score otherwise.
    vehicle_type_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=12)

    @field_validator("license_plate")
    @classmethod
    def normalize_plate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.strip().upper().split())
        return value or None

    @model_validator(mode="after")
    def readable_plate(self):
        if not self.plate_readable:
            self.license_plate = None
        elif self.license_plate is None:
            raise ValueError("Readable plate requires license_plate text")
        if self.confidence is None and self.vehicle_confidence is not None:
            self.confidence = self.vehicle_confidence
        if self.vehicle_confidence is None and self.confidence is not None:
            self.vehicle_confidence = self.confidence
        return self


class VisionImageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detections: list[VisionDetection] = Field(max_length=100)
