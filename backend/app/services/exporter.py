import csv
import io
import json
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.db.models import ImageAsset, ProcessingJob, ProcessingStatus


def detection_payload(det) -> dict[str, Any]:
    """One detection as exported.

    The original field names and shape are unchanged, so existing consumers keep working.
    Everything the local pipeline adds lives under ``plate`` and the three ``*_confidence``
    keys, which keep the vehicle, plate-detection and OCR stages separately auditable.
    """
    payload = {
        "detection_id": str(det.id),
        "vehicle_type": det.vehicle_type,
        "manufacturer": det.manufacturer,
        "model": det.model,
        "license_plate": det.license_plate,
        "plate_readable": det.plate_readable,
        "confidence": det.confidence,
        "bbox": {c: getattr(det, f"bbox_{c}") for c in ("x1", "y1", "x2", "y2")},
        "extra": det.extra,
        "vehicle_confidence": det.vehicle_confidence,
        "manufacturer_confidence": det.manufacturer_confidence,
        "source_label": det.source_label,
        "vehicle_type_source": det.vehicle_type_source,
        "vehicle_type_confidence": det.vehicle_type_confidence,
        "plate": None,
    }
    if det.has_plate_box or det.plate_text_raw is not None:
        payload["plate"] = {
            "bbox": (
                {c: getattr(det, f"plate_bbox_{c}") for c in ("x1", "y1", "x2", "y2")}
                if det.has_plate_box
                else None
            ),
            "quad": det.plate_quad,
            "detection_confidence": det.plate_detection_confidence,
            "ocr_confidence": det.plate_ocr_confidence,
            "text": det.license_plate if det.plate_readable else None,
            "text_raw": det.plate_text_raw,
            "readable": det.plate_readable,
            "format": det.plate_format,
            "region": det.plate_region,
            "ocr_engine": det.plate_ocr_engine,
            "ocr_variant": det.plate_ocr_variant,
            "rectified": det.plate_rectified,
            "perspective_skew": det.plate_perspective_skew,
            "status": det.plate_status,
            "crop_filename": det.plate_crop_filename,
            "rectified_crop_filename": det.plate_rectified_crop_filename,
        }
    return payload


async def collect_results(
    session: AsyncSession,
    image_ids: list[UUID] | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    maximum = get_settings().export_max_images
    effective_limit = min(limit or maximum, maximum)
    stmt = (
        select(ImageAsset)
        .options(selectinload(ImageAsset.detections))
        .order_by(ImageAsset.created_at, ImageAsset.id)
    )
    if image_ids:
        stmt = stmt.where(ImageAsset.id.in_(image_ids))
    images = (await session.scalars(stmt.offset(offset).limit(effective_limit + 1))).all()
    if limit is None and len(images) > maximum:
        raise HTTPException(413, "Export too large; use image_id filters or limit/offset")
    images = images[:effective_limit]
    ids = [image.id for image in images]
    latest = {}
    completed = {}
    # DISTINCT ON returns only one row per image, even after many reprocessing runs.
    for successful_only, destination in ((False, latest), (True, completed)):
        jobs = select(ProcessingJob).where(ProcessingJob.image_id.in_(ids))
        if successful_only:
            jobs = jobs.where(ProcessingJob.status == ProcessingStatus.completed)
        jobs = jobs.distinct(ProcessingJob.image_id).order_by(
            ProcessingJob.image_id, ProcessingJob.created_at.desc(), ProcessingJob.id.desc()
        )
        for job in (await session.scalars(jobs)).all():
            destination[job.image_id] = job
    payload = []
    for image in images:
        last = latest.get(image.id)
        result_job = completed.get(image.id)
        payload.append(
            {
                "image_id": str(image.id),
                "filename": image.original_filename,
                "width": image.width,
                "height": image.height,
                "annotated_available": bool(image.annotated_filename),
                "status": last.status.value if last else "not_processed",
                "latest_job_id": str(last.id) if last else None,
                "error_message": last.error_message if last else None,
                "result_job_id": str(result_job.id) if result_job else None,
                "provider": result_job.provider if result_job else None,
                "recognition_model": result_job.model if result_job else None,
                "is_mock": bool(result_job and result_job.provider == "mock"),
                "detections": [
                    detection_payload(det)
                    for det in sorted(image.detections, key=lambda d: str(d.id))
                ],
            }
        )
    return payload


async def export_json(session: AsyncSession, **kwargs) -> bytes:
    return json.dumps(
        await collect_results(session, **kwargs), ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")


def csv_safe(value):
    # Spreadsheet software can execute formula-looking filenames/model/OCR text.
    if isinstance(value, str) and (
        value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n"))
    ):
        return "'" + value
    return value


async def export_csv(session: AsyncSession, **kwargs) -> bytes:
    rows = await collect_results(session, **kwargs)
    buf = io.StringIO(newline="")
    image_fields = [
        "image_id",
        "filename",
        "status",
        "latest_job_id",
        "result_job_id",
        "provider",
        "recognition_model",
        "is_mock",
        "error_message",
    ]
    detection_fields = [
        "detection_id",
        "vehicle_type",
        "manufacturer",
        "model",
        "license_plate",
        "plate_readable",
        "confidence",
        "vehicle_confidence",
        "manufacturer_confidence",
        "source_label",
        "vehicle_type_source",
        "vehicle_type_confidence",
    ]
    bbox_fields = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    # Flattened plate columns: a spreadsheet user needs the plate box and the OCR
    # evidence next to the text, not buried in a nested object.
    plate_fields = [
        "plate_bbox_x1",
        "plate_bbox_y1",
        "plate_bbox_x2",
        "plate_bbox_y2",
        "plate_detection_confidence",
        "plate_ocr_confidence",
        "plate_text_raw",
        "plate_format",
        "plate_region",
        "plate_ocr_engine",
        "plate_ocr_variant",
        "plate_rectified",
        "plate_perspective_skew",
        "plate_status",
        "plate_crop_filename",
        "plate_rectified_crop_filename",
    ]
    writer = csv.DictWriter(
        buf, fieldnames=image_fields + detection_fields + bbox_fields + plate_fields
    )
    writer.writeheader()
    for image in rows:
        base = {field: image[field] for field in image_fields}
        for det in image["detections"] or [None]:
            row = dict(base)
            if det:
                row.update({field: det[field] for field in detection_fields})
                row.update({f"bbox_{c}": det["bbox"][c] for c in ("x1", "y1", "x2", "y2")})
                plate = det.get("plate")
                if plate:
                    box = plate.get("bbox") or {}
                    for corner in ("x1", "y1", "x2", "y2"):
                        row[f"plate_bbox_{corner}"] = box.get(corner)
                    for key in (
                        "detection_confidence",
                        "ocr_confidence",
                        "text_raw",
                        "format",
                        "region",
                        "ocr_engine",
                        "ocr_variant",
                        "rectified",
                        "perspective_skew",
                        "status",
                        "crop_filename",
                        "rectified_crop_filename",
                    ):
                        row[f"plate_{key}"] = plate.get(key)
            writer.writerow({key: csv_safe(value) for key, value in row.items()})
    return buf.getvalue().encode("utf-8-sig")
