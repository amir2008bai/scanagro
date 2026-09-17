import subprocess
import sys
from uuid import uuid4

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


async def test_upgrade_preserves_legacy_data_and_resolves_duplicate_jobs(db):
    # Runs only against the explicitly provided disposable TEST_DATABASE_URL.
    subprocess.run([sys.executable, "-m", "alembic", "downgrade", "0001_initial"], check=True)
    image_id = uuid4()
    try:
        await db.execute(
            text(
                """INSERT INTO image_assets
            (id, original_filename, stored_filename, mime_type, file_size, sha256, width, height, created_at, updated_at)
            VALUES (:id, 'legacy.jpg', 'legacy.jpg', 'image/jpeg', 123, :sha, 100, 100, now(), now())"""
            ),
            {"id": image_id, "sha": "a" * 64},
        )
        for state in ("pending", "processing"):
            await db.execute(
                text(
                    """INSERT INTO processing_jobs
                (id, image_id, status, provider, model, created_at, updated_at)
                VALUES (:id, :image_id, :state, 'mock', 'mock-empty', now(), now())"""
                ),
                {"id": uuid4(), "image_id": image_id, "state": state},
            )
        await db.execute(
            text(
                """INSERT INTO detections
            (id, image_id, vehicle_type, plate_readable, bbox_x1, bbox_y1, bbox_x2, bbox_y2, created_at, updated_at)
            VALUES (:id, :image_id, 'truck', true, 0, 0, 1, 1, now(), now())"""
            ),
            {"id": uuid4(), "image_id": image_id},
        )
        await db.commit()
    finally:
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
    states = (
        (await db.execute(text("SELECT status FROM processing_jobs ORDER BY status")))
        .scalars()
        .all()
    )
    assert sorted(states) == ["failed", "pending"]
    assert (await db.execute(text("SELECT plate_readable FROM detections"))).scalar_one() is False
    assert (
        await db.execute(text("SELECT original_filename FROM image_assets"))
    ).scalar_one() == "legacy.jpg"
