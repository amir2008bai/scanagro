import asyncio
import shutil
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.storage import LocalStorage

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health")
async def health():
    try:
        async with asyncio.timeout(3):
            async with SessionLocal() as session:
                # Also verifies that migrations have been applied.
                await session.execute(text("SELECT lease_token FROM processing_jobs LIMIT 1"))
            await asyncio.to_thread(check_storage)
    except Exception:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return {"status": "ok"}


def check_storage():
    storage = LocalStorage()
    storage.ensure_directories()
    test_file = storage.root / (".health-" + uuid4().hex)
    try:
        test_file.write_bytes(b"ok")
        if shutil.disk_usage(storage.root).free < get_settings().min_free_disk_mb * 1024 * 1024:
            raise OSError("Storage reserve exhausted")
    finally:
        test_file.unlink(missing_ok=True)
