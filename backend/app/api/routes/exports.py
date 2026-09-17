from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_export_session
from app.services.exporter import export_csv, export_json

router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("/results.json")
async def results_json(
    image_id: list[UUID] | None = Query(None, max_length=100),
    limit: int | None = Query(None, ge=1, le=get_settings().export_max_images),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_export_session),
) -> Response:
    content = await export_json(session, image_ids=image_id, limit=limit, offset=offset)
    return Response(
        content=content,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="results.json"'},
    )


@router.get("/results.csv")
async def results_csv(
    image_id: list[UUID] | None = Query(None, max_length=100),
    limit: int | None = Query(None, ge=1, le=get_settings().export_max_images),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_export_session),
) -> Response:
    content = await export_csv(session, image_ids=image_id, limit=limit, offset=offset)
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="results.csv"'},
    )
