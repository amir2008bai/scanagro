import asyncio
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import UploadFile

from app.core.config import get_settings
from app.db.models import ImageAsset
from app.db.session import get_session
from app.schemas.api import BatchUploadResponse, ImageOut, JobOut, UploadItem
from app.services.processing import create_job, ensure_idle, lock_image
from app.services.storage import LocalStorage

router = APIRouter(prefix="/images", tags=["images"])
storage = LocalStorage()


@router.post(
    "",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            }
                        },
                    }
                }
            },
        }
    },
)
async def upload_images(
    request: Request,
    process: bool = Query(True),
    session: AsyncSession = Depends(get_session),
) -> BatchUploadResponse:
    saved_files = []
    items = []
    committed = False
    commit_started = False
    try:
        async with request.form(
            max_files=get_settings().max_upload_files, max_fields=0, max_part_size=1024
        ) as form:
            files = form.getlist("files")
            if not files or any(not isinstance(f, UploadFile) for f in files):
                raise HTTPException(400, "At least one image in the files field is required")
            if any(key != "files" for key in form):
                raise HTTPException(400, "Only the files form field is supported")
            for upload in files:
                saved = await storage.save_upload(upload)
                saved_files.append(saved)
            for saved in saved_files:
                image = ImageAsset(**asdict(saved))
                session.add(image)
                await session.flush()
                job = await create_job(session, image) if process else None
                items.append(
                    UploadItem(
                        image=ImageOut.model_validate(image),
                        job=JobOut.model_validate(job) if job else None,
                    )
                )
            commit_started = True
            await session.commit()  # Images and queued jobs become visible together.
            committed = True
        return BatchUploadResponse(items=items, count=len(items))
    finally:
        if not committed:
            await session.rollback()
            if not commit_started:
                for saved in saved_files:
                    storage.delete_image_files(saved.stored_filename, None)
            # If COMMIT lost its connection, its outcome can be unknown. Keep files
            # until the maintenance orphan sweep can establish DB references.


@router.get("", response_model=list[ImageOut])
async def list_images(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[ImageOut]:
    result = await session.execute(
        select(ImageAsset)
        .order_by(desc(ImageAsset.created_at), desc(ImageAsset.id))
        .limit(limit)
        .offset(offset)
    )
    return [ImageOut.model_validate(item) for item in result.scalars().all()]


@router.get("/{image_id}", response_model=ImageOut)
async def get_image(image_id: UUID, session: AsyncSession = Depends(get_session)) -> ImageOut:
    image = await session.get(ImageAsset, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageOut.model_validate(image)


@router.get("/{image_id}/file", response_class=FileResponse)
async def get_original(
    image_id: UUID, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    image = await session.get(ImageAsset, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    path = storage.original_path(image.stored_filename)
    if not path.is_file():
        raise HTTPException(404, "Image file is missing from storage")
    filename = Path(image.original_filename).stem + path.suffix
    return FileResponse(path, media_type=image.mime_type, filename=filename)


@router.get("/{image_id}/thumbnail", response_class=FileResponse)
async def get_thumbnail(
    image_id: UUID, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    """Small cached copy of the normalised image, for grids and lists."""
    image = await session.get(ImageAsset, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    if not storage.original_path(image.stored_filename).is_file():
        raise HTTPException(404, "Image file is missing from storage")
    try:
        path = await asyncio.to_thread(storage.build_thumbnail, image.stored_filename)
    except OSError:
        raise HTTPException(404, "Image could not be read") from None
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=f"{image.id}-thumb.jpg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/{image_id}/annotated", response_class=FileResponse)
async def get_annotated(
    image_id: UUID, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    image = await session.get(ImageAsset, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    if not image.annotated_filename:
        raise HTTPException(status_code=404, detail="Annotated image not available yet")
    path = storage.annotated_path(image.annotated_filename)
    if not path.is_file():
        raise HTTPException(404, "Annotated file is missing from storage")
    return FileResponse(path, media_type="image/jpeg", filename=f"{image.id}-annotated.jpg")


@router.post("/{image_id}/process", response_model=JobOut, status_code=202)
async def reprocess_image(image_id: UUID, session: AsyncSession = Depends(get_session)) -> JobOut:
    image = await lock_image(session, image_id)
    job = await create_job(session, image)
    await session.commit()
    return JobOut.model_validate(job)


@router.delete("/{image_id}", status_code=204)
async def delete_image(image_id: UUID, session: AsyncSession = Depends(get_session)) -> None:
    image = await lock_image(session, image_id)
    await ensure_idle(session, image_id)
    stored, annotated = image.stored_filename, image.annotated_filename
    await session.delete(image)
    await session.commit()
    await asyncio.to_thread(storage.delete_image_files, stored, annotated)
