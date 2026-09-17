from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Detection
from app.db.session import get_session
from app.schemas.api import BBoxOut, DetectionOut, DetectionPatch, PaginatedDetections, PlateOut
from app.schemas.vision import VisionDetection
from app.services.processing import ensure_idle, lock_image, regenerate_annotation
from app.services.storage import LocalStorage

router = APIRouter(prefix="/detections", tags=["detections"])
storage = LocalStorage()

#: Columns a correction is allowed to write. Keeping this explicit stops new schema
#: fields (plate observations, evidence lists) from being blind-copied onto the row.
PATCHABLE_COLUMNS = (
    "vehicle_type",
    "manufacturer",
    "model",
    "license_plate",
    "plate_readable",
    "confidence",
    "vehicle_confidence",
)


def plate_out(det: Detection) -> PlateOut | None:
    """Only build a plate payload when the row actually carries plate evidence."""
    if not det.has_plate_box and det.plate_text_raw is None:
        return None
    box = None
    if det.has_plate_box:
        box = BBoxOut(
            x1=det.plate_bbox_x1, y1=det.plate_bbox_y1, x2=det.plate_bbox_x2, y2=det.plate_bbox_y2
        )
    return PlateOut(
        bbox=box,
        quad=det.plate_quad,
        detection_confidence=det.plate_detection_confidence,
        ocr_confidence=det.plate_ocr_confidence,
        text=det.license_plate if det.plate_readable else None,
        text_raw=det.plate_text_raw,
        readable=det.plate_readable,
        format=det.plate_format,
        region=det.plate_region,
        ocr_engine=det.plate_ocr_engine,
        ocr_variant=det.plate_ocr_variant,
        rectified=det.plate_rectified,
        perspective_skew=det.plate_perspective_skew,
        status=det.plate_status,
        crop_filename=det.plate_crop_filename,
        rectified_crop_filename=det.plate_rectified_crop_filename,
    )


def to_out(det: Detection) -> DetectionOut:
    return DetectionOut(
        id=det.id,
        image_id=det.image_id,
        job_id=det.job_id,
        vehicle_type=det.vehicle_type,
        manufacturer=det.manufacturer,
        model=det.model,
        license_plate=det.license_plate,
        plate_readable=det.plate_readable,
        confidence=det.confidence,
        bbox=BBoxOut(x1=det.bbox_x1, y1=det.bbox_y1, x2=det.bbox_x2, y2=det.bbox_y2),
        extra=det.extra,
        created_at=det.created_at,
        updated_at=det.updated_at,
        vehicle_confidence=det.vehicle_confidence,
        manufacturer_confidence=det.manufacturer_confidence,
        source_label=det.source_label,
        vehicle_type_source=det.vehicle_type_source,
        vehicle_type_confidence=det.vehicle_type_confidence,
        plate=plate_out(det),
    )


@router.get("", response_model=PaginatedDetections)
async def list_detections(
    vehicle_type: str | None = None,
    manufacturer: str | None = None,
    license_plate: str | None = None,
    image_id: UUID | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> PaginatedDetections:
    filters = []
    if vehicle_type:
        filters.append(Detection.vehicle_type.icontains(vehicle_type, autoescape=True))
    if manufacturer:
        filters.append(Detection.manufacturer.icontains(manufacturer, autoescape=True))
    if license_plate:
        filters.append(Detection.license_plate.icontains(license_plate, autoescape=True))
    if image_id:
        filters.append(Detection.image_id == image_id)

    stmt = select(Detection)
    count_stmt = select(func.count()).select_from(Detection)
    if filters:
        stmt = stmt.where(*filters)
        count_stmt = count_stmt.where(*filters)
    total = (await session.execute(count_stmt)).scalar_one()
    result = await session.execute(
        stmt.order_by(Detection.created_at.desc(), Detection.id.desc()).limit(limit).offset(offset)
    )
    return PaginatedDetections(
        items=[to_out(item) for item in result.scalars().all()],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{detection_id}/plate-crop", response_class=FileResponse)
async def get_plate_crop(
    detection_id: UUID,
    variant: Literal["before", "after"] = Query(
        "after",
        description="'before' is the detector crop as found; "
        "'after' is the same crop with the perspective removed.",
    ),
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """Serve the plate crop kept as review evidence.

    Both variants exist so a reviewer can judge the geometry stage, not just its output:
    'after' is missing when the corners could not be recovered and no warp was applied.
    """
    det = await session.get(Detection, detection_id)
    if not det:
        raise HTTPException(404, "Detection not found")
    name = det.plate_crop_filename if variant == "before" else det.plate_rectified_crop_filename
    if not name:
        raise HTTPException(
            404,
            "Plate crop was not stored for this detection"
            if variant == "before"
            else "No rectified crop: the plate corners were not recovered",
        )
    try:
        path = storage.plate_path(name)
    except ValueError:
        raise HTTPException(404, "Stored plate crop name is not usable") from None
    if not path.is_file():
        raise HTTPException(404, "Plate crop file is missing from storage")
    return FileResponse(path, media_type="image/jpeg", filename=f"{det.id}-plate-{variant}.jpg")


@router.get("/{detection_id}", response_model=DetectionOut)
async def get_detection(
    detection_id: UUID, session: AsyncSession = Depends(get_session)
) -> DetectionOut:
    det = await session.get(Detection, detection_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found")
    return to_out(det)


async def editable_detection(session: AsyncSession, detection_id: UUID):
    image_id = await session.scalar(select(Detection.image_id).where(Detection.id == detection_id))
    if image_id is None:
        raise HTTPException(404, "Detection not found")
    image = await lock_image(session, image_id)
    await ensure_idle(session, image_id)
    det = await session.get(Detection, detection_id, populate_existing=True)
    if det is None:
        raise HTTPException(404, "Detection no longer exists")
    return image, det


async def commit_correction(session: AsyncSession, image):
    storage = LocalStorage()
    new = None
    commit_started = False
    try:
        await session.flush()
        old, new = await regenerate_annotation(session, image)
        commit_started = True
        await session.commit()
    except BaseException:
        await session.rollback()
        if not commit_started:
            storage.remove_annotation(new)
        raise
    storage.remove_annotation(old)


@router.patch("/{detection_id}", response_model=DetectionOut)
async def patch_detection(
    detection_id: UUID,
    patch: DetectionPatch,
    session: AsyncSession = Depends(get_session),
) -> DetectionOut:
    image, det = await editable_detection(session, detection_id)
    changes = patch.model_dump(exclude_unset=True)
    if not changes:
        return to_out(det)
    plate_bbox_change = "plate_bbox" in changes
    plate_bbox = changes.pop("plate_bbox", None)
    merged = {name: getattr(det, name) for name in PATCHABLE_COLUMNS}
    merged["bbox"] = {c: getattr(det, f"bbox_{c}") for c in ("x1", "y1", "x2", "y2")}
    merged.update(changes)
    if "license_plate" in changes and "plate_readable" not in changes:
        merged["plate_readable"] = bool((changes["license_plate"] or "").strip())
    if changes.get("plate_readable") is False:
        merged["license_plate"] = None
    try:
        value = VisionDetection.model_validate(merged)
    except ValidationError as exc:
        raise HTTPException(
            422, "Correction has inconsistent plate text/readability or invalid values"
        ) from exc
    for name in PATCHABLE_COLUMNS:
        setattr(det, name, getattr(value, name))
    for name, item in value.bbox.model_dump().items():
        setattr(det, f"bbox_{name}", item)
    if plate_bbox_change:
        # A corrected plate box invalidates the machine-estimated corners and the stored
        # crops: the operator is telling us the model looked at the wrong place.
        for corner in ("x1", "y1", "x2", "y2"):
            setattr(det, f"plate_bbox_{corner}", plate_bbox[corner] if plate_bbox else None)
        det.plate_quad = None
        det.plate_rectified = None
        det.plate_perspective_skew = None
        det.plate_detection_confidence = None
        det.plate_status = "manually_set" if plate_bbox else "manually_cleared"
    if "vehicle_type" in changes:
        # The type came from a human, so the stage that guessed it is no longer the
        # provenance of this row.
        det.vehicle_type_source = "manual"
        det.vehicle_type_confidence = None
    if "license_plate" in changes or "plate_readable" in changes:
        # The text came from a human, so the machine's OCR score no longer describes it.
        det.plate_ocr_confidence = None
        det.plate_ocr_engine = None
        det.plate_ocr_variant = None
        det.plate_status = "manually_corrected"
    det.extra = {**(det.extra or {}), "manually_corrected": True}
    await commit_correction(session, image)
    await session.refresh(det)
    return to_out(det)


@router.delete("/{detection_id}", status_code=204)
async def delete_detection(
    detection_id: UUID, session: AsyncSession = Depends(get_session)
) -> None:
    image, det = await editable_detection(session, detection_id)
    await session.delete(det)
    await commit_correction(session, image)
