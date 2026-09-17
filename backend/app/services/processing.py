import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from uuid import UUID, uuid4

import structlog
from fastapi import HTTPException
from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Detection, ImageAsset, ProcessingJob, ProcessingStatus
from app.db.session import SessionLocal
from app.services.annotation import annotate_image
from app.services.storage import LocalStorage
from app.services.vision.factory import get_vision_provider

logger = structlog.get_logger(__name__)
storage = LocalStorage()
ACTIVE = (ProcessingStatus.pending, ProcessingStatus.processing)


async def collect_plate_crops(session: AsyncSession, image_id: UUID) -> list[str]:
    """Plate crop filenames currently referenced by an image's detections.

    Collected before the rows are replaced so the files can be unlinked *after* the
    transaction commits; deleting them earlier would lose evidence if the commit rolls
    back. A crash between commit and unlink leaves orphans for the sweep in
    `scripts/cleanup_storage.py`, which is the same contract as annotated images.
    """
    rows = await session.execute(
        select(Detection.plate_crop_filename, Detection.plate_rectified_crop_filename).where(
            Detection.image_id == image_id
        )
    )
    names: list[str] = []
    for before, after in rows.all():
        names.extend(name for name in (before, after) if name)
    return names


def build_detection(image_id: UUID, job_id: UUID, item) -> Detection:
    """Map one `VisionDetection` onto a row, keeping the stages' results separate.

    Providers that do not produce plate geometry (mock, and any OpenAI-compatible model
    server) simply leave the new columns null, so this is safe for every provider.
    """
    plate = getattr(item, "plate", None)
    extra: dict = {}
    if item.notes:
        extra["notes"] = item.notes
    if getattr(item, "evidence", None):
        extra["attribute_evidence"] = item.evidence
    if plate is not None:
        if plate.alternatives:
            extra["plate_ocr_candidates"] = plate.alternatives
        if plate.disagreement_positions:
            extra["plate_ocr_disagreement_positions"] = plate.disagreement_positions
    if getattr(item, "extra_plates", None):
        extra["additional_plates"] = [
            p.model_dump(mode="json", exclude={"alternatives"}) for p in item.extra_plates
        ]

    return Detection(
        image_id=image_id,
        job_id=job_id,
        vehicle_type=item.vehicle_type,
        manufacturer=item.manufacturer,
        model=item.model,
        license_plate=item.license_plate,
        plate_readable=item.plate_readable,
        confidence=item.confidence,
        bbox_x1=item.bbox.x1,
        bbox_y1=item.bbox.y1,
        bbox_x2=item.bbox.x2,
        bbox_y2=item.bbox.y2,
        vehicle_confidence=getattr(item, "vehicle_confidence", None),
        manufacturer_confidence=getattr(item, "manufacturer_confidence", None),
        source_label=getattr(item, "source_label", None),
        vehicle_type_source=getattr(item, "vehicle_type_source", None),
        vehicle_type_confidence=getattr(item, "vehicle_type_confidence", None),
        plate_bbox_x1=plate.bbox.x1 if plate else None,
        plate_bbox_y1=plate.bbox.y1 if plate else None,
        plate_bbox_x2=plate.bbox.x2 if plate else None,
        plate_bbox_y2=plate.bbox.y2 if plate else None,
        plate_quad=plate.quad if plate else None,
        plate_detection_confidence=plate.detection_confidence if plate else None,
        plate_ocr_confidence=plate.ocr_confidence if plate else None,
        plate_text_raw=plate.text_raw if plate else None,
        plate_format=plate.plate_format if plate else None,
        plate_region=plate.region if plate else None,
        plate_ocr_engine=plate.ocr_engine if plate else None,
        plate_ocr_variant=plate.ocr_variant if plate else None,
        plate_rectified=plate.rectified if plate else None,
        plate_perspective_skew=plate.perspective_skew if plate else None,
        plate_status=plate.status if plate else None,
        plate_crop_filename=plate.crop_filename if plate else None,
        plate_rectified_crop_filename=plate.rectified_crop_filename if plate else None,
        extra=extra or None,
    )


async def lock_image(session: AsyncSession, image_id: UUID) -> ImageAsset:
    image = await session.scalar(
        select(ImageAsset)
        .where(ImageAsset.id == image_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if image is None:
        raise HTTPException(404, "Image not found")
    return image


async def ensure_idle(session: AsyncSession, image_id: UUID):
    active = await session.scalar(
        select(ProcessingJob.id)
        .where(ProcessingJob.image_id == image_id, ProcessingJob.status.in_(ACTIVE))
        .limit(1)
    )
    if active:
        raise HTTPException(409, "Image has an active job; wait until it finishes")


async def create_job(session: AsyncSession, image: ImageAsset) -> ProcessingJob:
    # Caller owns the transaction and locks existing images before calling.
    await ensure_idle(session, image.id)
    provider = get_vision_provider()
    job = ProcessingJob(
        image_id=image.id,
        status=ProcessingStatus.pending,
        provider=provider.name,
        model=provider.model,
    )
    session.add(job)
    await session.flush()
    return job


async def claim_job(job_id: UUID | None = None) -> ProcessingJob | None:
    settings = get_settings()
    async with SessionLocal() as session, session.begin():
        now = await session.scalar(select(func.now()))
        stmt = select(ProcessingJob).where(
            or_(
                ProcessingJob.status == ProcessingStatus.pending,
                and_(
                    ProcessingJob.status == ProcessingStatus.processing,
                    ProcessingJob.lease_expires_at < now,
                ),
            )
        )
        if job_id is not None:
            stmt = stmt.where(ProcessingJob.id == job_id)
        job = await session.scalar(
            stmt.order_by(ProcessingJob.created_at, ProcessingJob.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if job is None:
            return None
        if job.attempts >= settings.job_max_attempts:
            job.status = ProcessingStatus.failed
            job.error_message = "Worker recovery limit exceeded; reprocess the image to retry"
            job.finished_at = now
            job.lease_token = None
            job.lease_expires_at = None
            return None
        job.status = ProcessingStatus.processing
        job.started_at = now
        job.finished_at = None
        job.error_message = None
        job.attempts += 1
        job.lease_token = uuid4()
        job.lease_expires_at = now + timedelta(seconds=settings.job_lease_seconds)
        return job


async def _finish_failure(job: ProcessingJob, message: str):
    async with SessionLocal() as session, session.begin():
        current = await session.scalar(
            select(ProcessingJob)
            .where(
                ProcessingJob.id == job.id,
                ProcessingJob.lease_token == job.lease_token,
                ProcessingJob.status == ProcessingStatus.processing,
            )
            .with_for_update()
        )
        if current:
            current.status = ProcessingStatus.failed
            current.finished_at = await session.scalar(select(func.now()))
            current.error_message = message
            current.lease_token = None
            current.lease_expires_at = None


async def process_job(job_id: UUID | None = None) -> bool:
    job = await claim_job(job_id)
    if job is None:
        return False
    settings = get_settings()
    annotation_name = f"{job.image_id}-{job.lease_token}.jpg"
    committed = False
    publication_started = False
    try:
        async with asyncio.timeout(settings.job_timeout_seconds):
            async with SessionLocal() as session:
                image = await session.get(ImageAsset, job.image_id)
            if image is None:
                return True
            # Honor the provider and model recorded when the job was submitted.
            provider = get_vision_provider(job.provider, job.model)
            result, raw = await provider.analyze(
                storage.original_path(image.stored_filename), image.mime_type
            )
            detections = [build_detection(image.id, job.id, item) for item in result.detections]
            await asyncio.to_thread(
                annotate_image,
                storage.original_path(image.stored_filename),
                storage.annotated_path(annotation_name),
                detections,
            )
            publication_started = True
            async with SessionLocal() as session, session.begin():
                current_image = await lock_image(session, image.id)
                current = await session.scalar(
                    select(ProcessingJob)
                    .where(
                        ProcessingJob.id == job.id,
                        ProcessingJob.lease_token == job.lease_token,
                        ProcessingJob.status == ProcessingStatus.processing,
                        ProcessingJob.lease_expires_at > func.now(),
                    )
                    .with_for_update()
                )
                if current is None:
                    publication_started = False
                    return True  # Expired/reclaimed worker must never overwrite newer results.
                old_annotation = current_image.annotated_filename
                old_crops = await collect_plate_crops(session, image.id)
                await session.execute(delete(Detection).where(Detection.image_id == image.id))
                session.add_all(detections)
                current_image.annotated_filename = annotation_name
                current.raw_response = raw if settings.vision_store_raw_response else None
                current.status = ProcessingStatus.completed
                current.finished_at = await session.scalar(select(func.now()))
                current.lease_token = None
                current.lease_expires_at = None
            committed = True
            storage.remove_annotation(old_annotation)
            for name in old_crops:
                storage.remove_plate_crop(name)
            await logger.ainfo("job_completed", job_id=str(job.id), detections=len(detections))
    except asyncio.CancelledError:
        # Leave the lease: another worker will recover it after a bounded delay.
        raise
    except Exception as exc:
        message = _failure_message(exc)
        await _finish_failure(job, message)
        await logger.aerror("job_failed", job_id=str(job.id), error_type=type(exc).__name__)
    finally:
        if not committed and not publication_started:
            storage.remove_annotation(annotation_name)
        # Preserve files after an uncertain COMMIT; the orphan sweep reclaims them.
    return True


def _failure_message(exc: BaseException) -> str:
    """A safe, actionable reason for the client.

    Provider errors are already written for this purpose; everything else is reduced to a
    job ID so that model responses, paths and secrets never reach an API consumer.
    """
    from app.services.vision.openai_compatible import VisionProviderError

    try:
        from app.services.vision.local_provider import VisionModelUnavailable
    except Exception:  # the local pipeline's dependencies may not be installed
        VisionModelUnavailable = ()  # type: ignore[assignment]

    if VisionModelUnavailable and isinstance(exc, VisionModelUnavailable):
        return "Recognition models are missing; run scripts/fetch_models.py"
    if isinstance(exc, VisionProviderError):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return "Processing deadline exceeded"
    return "Processing failed; check worker logs using this job ID"


async def worker_loop(stop: asyncio.Event):
    while not stop.is_set():
        try:
            worked = await process_job()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await logger.aerror("worker_poll_failed", error_type=type(exc).__name__)
            worked = False
        if not worked:
            try:
                await asyncio.wait_for(stop.wait(), timeout=get_settings().worker_poll_seconds)
            except TimeoutError:
                pass


@asynccontextmanager
async def running_workers():
    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(worker_loop(stop), name=f"vision-worker-{n}")
        for n in range(get_settings().worker_concurrency)
    ]
    try:
        yield
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def regenerate_annotation(session: AsyncSession, image: ImageAsset) -> tuple[str | None, str]:
    detections = (
        await session.scalars(
            select(Detection).where(Detection.image_id == image.id).order_by(Detection.id)
        )
    ).all()
    old = image.annotated_filename
    new = f"{image.id}-{uuid4().hex}.jpg"
    await asyncio.to_thread(
        annotate_image,
        storage.original_path(image.stored_filename),
        storage.annotated_path(new),
        detections,
    )
    image.annotated_filename = new
    return old, new
