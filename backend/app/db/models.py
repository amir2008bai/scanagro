import enum
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ProcessingStatus(enum.StrEnum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ImageAsset(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "image_assets"

    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    annotated_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )
    detections: Mapped[list["Detection"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class ProcessingJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        Index(
            "uq_processing_jobs_active_image",
            "image_id",
            unique=True,
            postgresql_where=text("status IN ('pending', 'processing')"),
        ),
        Index("ix_processing_jobs_lease_expires_at", "lease_expires_at"),
    )

    image_id: Mapped[UUID] = mapped_column(
        ForeignKey("image_assets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, name="processing_status"),
        default=ProcessingStatus.pending,
        index=True,
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    lease_token: Mapped[UUID | None] = mapped_column(nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    image: Mapped[ImageAsset] = relationship(back_populates="jobs")


class Detection(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "detections"
    __table_args__ = (
        CheckConstraint(
            "bbox_x1 >= 0 AND bbox_y1 >= 0 AND bbox_x2 <= 1 AND bbox_y2 <= 1 "
            "AND bbox_x2 > bbox_x1 AND bbox_y2 > bbox_y1",
            name="ck_detection_bbox",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_detection_confidence",
        ),
        CheckConstraint(
            "(plate_readable AND license_plate IS NOT NULL) OR "
            "(NOT plate_readable AND license_plate IS NULL)",
            name="ck_detection_plate",
        ),
        # The plate box is optional, but when present it must be a well-formed box.
        CheckConstraint(
            "(plate_bbox_x1 IS NULL AND plate_bbox_y1 IS NULL "
            " AND plate_bbox_x2 IS NULL AND plate_bbox_y2 IS NULL) OR "
            "(plate_bbox_x1 >= 0 AND plate_bbox_y1 >= 0 "
            " AND plate_bbox_x2 <= 1 AND plate_bbox_y2 <= 1 "
            " AND plate_bbox_x2 > plate_bbox_x1 AND plate_bbox_y2 > plate_bbox_y1)",
            name="ck_detection_plate_bbox",
        ),
        CheckConstraint(
            "(plate_detection_confidence IS NULL OR "
            " (plate_detection_confidence >= 0 AND plate_detection_confidence <= 1)) AND "
            "(plate_ocr_confidence IS NULL OR "
            " (plate_ocr_confidence >= 0 AND plate_ocr_confidence <= 1)) AND "
            "(vehicle_confidence IS NULL OR "
            " (vehicle_confidence >= 0 AND vehicle_confidence <= 1))",
            name="ck_detection_stage_confidences",
        ),
        CheckConstraint(
            "vehicle_type_confidence IS NULL OR "
            "(vehicle_type_confidence >= 0 AND vehicle_type_confidence <= 1)",
            name="ck_detection_vehicle_type_confidence",
        ),
    )

    image_id: Mapped[UUID] = mapped_column(
        ForeignKey("image_assets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True, nullable=True
    )
    vehicle_type: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    manufacturer: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)
    license_plate: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    plate_readable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # --- per-stage results, added in 0003 ---------------------------------------
    # `confidence` above stays the vehicle score for API compatibility; the three
    # columns below keep the stages apart so a confident box around an illegible
    # plate is distinguishable from a weak box around a crisp one.
    vehicle_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    manufacturer_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Raw detector class before mapping onto our vocabulary, kept as evidence.
    source_label: Mapped[str | None] = mapped_column(String(60), nullable=True)
    #: Which stage settled `vehicle_type`, and how confident that stage was.
    vehicle_type_source: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    vehicle_type_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    #: Plate box in normalised image coordinates — deliberately distinct from the
    #: vehicle box above, so both can be drawn and audited independently.
    plate_bbox_x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_bbox_y1: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_bbox_x2: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_bbox_y2: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Four normalised corners, when perspective estimation succeeded.
    plate_quad: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    plate_detection_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Best raw OCR string, retained even when the read was rejected as unreadable.
    plate_text_raw: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    plate_format: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    plate_region: Mapped[str | None] = mapped_column(String(60), nullable=True)
    plate_ocr_engine: Mapped[str | None] = mapped_column(String(40), nullable=True)
    plate_ocr_variant: Mapped[str | None] = mapped_column(String(40), nullable=True)
    plate_rectified: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    plate_perspective_skew: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_status: Mapped[str | None] = mapped_column(String(60), nullable=True)
    plate_crop_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    plate_rectified_crop_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    image: Mapped[ImageAsset] = relationship(back_populates="detections")

    @property
    def has_plate_box(self) -> bool:
        return self.plate_bbox_x1 is not None and self.plate_bbox_x2 is not None
