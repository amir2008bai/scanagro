import io
import os
import subprocess
import sys
import tempfile

import httpx
import pytest
from PIL import Image

# Tests never use the application's .env database or real recognition service.
TEST_URL = os.environ.get("TEST_DATABASE_URL", "")
TEST_STORAGE = tempfile.TemporaryDirectory(prefix="fable-test-storage-")
os.environ.update(
    APP_ENV="test",
    PROCESSING_MODE="worker",
    VISION_PROVIDER="mock",
    API_KEY="test-service-key-32-characters-long",
    STORAGE_DIR=TEST_STORAGE.name,
    ALLOWED_HOSTS="testserver,localhost,127.0.0.1",
    MIN_FREE_DISK_MB="0",
)
os.environ["DATABASE_URL"] = TEST_URL or "postgresql+asyncpg://test:test@127.0.0.1:1/unused_test"


@pytest.fixture
def image_bytes():
    out = io.BytesIO()
    Image.new("RGB", (128, 96), "white").save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture(scope="session")
def migrated_database():
    if not TEST_URL:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated database ending in _test")
    from sqlalchemy.engine import make_url

    if not (make_url(TEST_URL).database or "").endswith("_test"):
        pytest.fail("TEST_DATABASE_URL must name a disposable database ending in _test")
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)


@pytest.fixture
async def db(migrated_database):
    from sqlalchemy import text

    from app.db.session import SessionLocal, engine
    from app.services.storage import LocalStorage

    storage = LocalStorage()
    storage.ensure_directories()
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE detections, processing_jobs, image_assets CASCADE"))
    async with SessionLocal() as session:
        yield session
    await engine.dispose()
    for folder in (storage.upload_dir, storage.annotated_dir):
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()


@pytest.fixture
async def client(db):
    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers={"X-API-Key": "test-service-key-32-characters-long"},
    ) as client:
        yield client


@pytest.fixture
def one_detection_provider(monkeypatch):
    from app.schemas.vision import VisionImageResult
    from app.services import processing

    class Provider:
        name = "mock"
        model = "mock-empty"

        async def analyze(self, *args):
            return VisionImageResult.model_validate(
                {
                    "detections": [
                        {
                            "vehicle_type": "truck",
                            "manufacturer": "KamAZ",
                            "license_plate": "123 abc 01",
                            "plate_readable": True,
                            "confidence": 0.9,
                            "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.9},
                        }
                    ]
                }
            ), {"test_fixture": True}

    monkeypatch.setattr(processing, "get_vision_provider", lambda *args: Provider())
    return Provider
