from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ProcessingJob, ProcessingStatus
from app.db.session import get_session
from app.schemas.api import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobOut])
async def list_jobs(
    status: ProcessingStatus | None = None,
    image_id: UUID | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[JobOut]:
    stmt = select(ProcessingJob).order_by(desc(ProcessingJob.created_at), desc(ProcessingJob.id))
    if image_id:
        stmt = stmt.where(ProcessingJob.image_id == image_id)
    if status:
        stmt = stmt.where(ProcessingJob.status == status)
    result = await session.execute(stmt.limit(limit).offset(offset))
    return [JobOut.model_validate(item) for item in result.scalars().all()]


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: UUID, session: AsyncSession = Depends(get_session)) -> JobOut:
    job = await session.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobOut.model_validate(job)
