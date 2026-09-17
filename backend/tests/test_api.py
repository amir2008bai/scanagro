import asyncio
import csv
import io
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import ImageAsset, ProcessingJob
from app.db.session import SessionLocal
from app.services.processing import claim_job, process_job
from app.services.storage import LocalStorage
from app.services.vision.openai_compatible import VisionProviderError

pytestmark = pytest.mark.integration
BASE = "/api/v1"


async def uploaded(client, data, process=True, filename="test.jpg"):
    response = await client.post(
        BASE + "/images",
        params={"process": str(process).lower()},
        files=[("files", (filename, data, "image/jpeg"))],
    )
    assert response.status_code == 201, response.text
    return response.json()["items"][0]


async def test_auth_health_and_openapi(client):
    denied = await client.get(BASE + "/images", headers={"X-API-Key": "wrong"})
    assert denied.status_code == 401
    assert (await client.get(BASE + "/health", headers={"X-API-Key": ""})).status_code == 200
    schema = (await client.get("/openapi.json")).json()
    assert "APIKeyHeader" in schema["components"]["securitySchemes"]
    assert "requestBody" in schema["paths"][BASE + "/images"]["post"]
    assert (await client.get(BASE + "/images", headers={"Host": "evil.test"})).status_code == 400


async def test_end_to_end_mock_and_delete(client, db, image_bytes):
    item = await uploaded(client, image_bytes)
    image_id, job_id = item["image"]["id"], item["job"]["id"]
    assert (await client.get(BASE + f"/images/{image_id}/annotated")).status_code == 404
    assert await process_job()
    job = (await client.get(BASE + f"/jobs/{job_id}")).json()
    assert job["status"] == "completed" and job["attempts"] == 1
    annotated = await client.get(BASE + f"/images/{image_id}/annotated")
    assert annotated.status_code == 200 and annotated.content.startswith(b"\xff\xd8")
    results = (await client.get(BASE + "/exports/results.json")).json()
    assert results[0]["is_mock"] is True and results[0]["detections"] == []
    response = await client.delete(BASE + f"/images/{image_id}")
    assert response.status_code == 204 and not response.content
    assert await db.scalar(select(func.count()).select_from(ProcessingJob)) == 0
    assert not list(LocalStorage().upload_dir.iterdir())
    assert not list(LocalStorage().annotated_dir.iterdir())


async def test_upload_batch_rollback(client, db, image_bytes):
    response = await client.post(
        BASE + "/images",
        files=[
            ("files", ("good.jpg", image_bytes, "image/jpeg")),
            ("files", ("bad.jpg", b"garbage", "image/jpeg")),
        ],
    )
    assert response.status_code == 400
    assert await db.scalar(select(func.count()).select_from(ImageAsset)) == 0
    assert not list(LocalStorage().upload_dir.iterdir())


async def test_active_job_conflicts(client, image_bytes):
    item = await uploaded(client, image_bytes)
    path = BASE + f"/images/{item['image']['id']}"
    assert (await client.post(path + "/process")).status_code == 409
    assert (await client.delete(path)).status_code == 409
    await process_job()
    responses = await asyncio.gather(client.post(path + "/process"), client.post(path + "/process"))
    assert sorted(r.status_code for r in responses) == [202, 409]


async def test_two_workers_claim_once(client, image_bytes, one_detection_provider):
    await uploaded(client, image_bytes)
    results = await asyncio.gather(process_job(), process_job())
    assert sorted(results) == [False, True]
    assert (await client.get(BASE + "/detections")).json()["total"] == 1


async def test_failed_reprocess_preserves_previous_result(
    client, image_bytes, one_detection_provider, monkeypatch
):
    from app.services import processing

    item = await uploaded(client, image_bytes)
    await process_job()
    before = (await client.get(BASE + "/detections")).json()["items"]

    class Failing(one_detection_provider):
        async def analyze(self, *args):
            raise VisionProviderError("Vision endpoint returned HTTP 401")

    monkeypatch.setattr(processing, "get_vision_provider", lambda *args: Failing())
    response = await client.post(BASE + f"/images/{item['image']['id']}/process")
    await process_job()
    job = (await client.get(BASE + f"/jobs/{response.json()['id']}")).json()
    assert job["status"] == "failed"
    assert (await client.get(BASE + "/detections")).json()["items"] == before
    export = (await client.get(BASE + "/exports/results.json")).json()[0]
    assert export["status"] == "failed" and export["result_job_id"] == item["job"]["id"]


async def test_expired_lease_recovered(client, image_bytes):
    await uploaded(client, image_bytes)
    first = await claim_job()
    async with SessionLocal() as session, session.begin():
        job = await session.get(ProcessingJob, first.id)
        job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await process_job()
    job = (await client.get(BASE + f"/jobs/{first.id}")).json()
    assert job["status"] == "completed" and job["attempts"] == 2


async def test_recovery_limit(client, image_bytes):
    await uploaded(client, image_bytes)
    first = await claim_job()
    async with SessionLocal() as session, session.begin():
        job = await session.get(ProcessingJob, first.id)
        job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        job.attempts = get_settings().job_max_attempts
    assert await process_job() is False
    assert (await client.get(BASE + f"/jobs/{first.id}")).json()["status"] == "failed"


async def test_corrections_update_annotations(client, image_bytes, one_detection_provider):
    item = await uploaded(client, image_bytes)
    await process_job()
    image_path = BASE + f"/images/{item['image']['id']}"
    before = (await client.get(image_path)).json()["annotated_filename"]
    det = (await client.get(BASE + "/detections")).json()["items"][0]
    det_path = BASE + f"/detections/{det['id']}"
    response = await client.patch(
        det_path,
        json={"license_plate": " 456 xyz 02 ", "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["license_plate"] == "456 XYZ 02"
    after = (await client.get(image_path)).json()["annotated_filename"]
    assert before != after
    assert not LocalStorage().annotated_path(before).exists()
    for patch in (
        {"vehicle_type": None},
        {"plate_readable": None},
        {"bbox": None},
        {"vehicle_type": " "},
        {"confidence": 2},
        {"unsupported": 1},
    ):
        assert (await client.patch(det_path, json=patch)).status_code == 422
    response = await client.patch(det_path, json={"license_plate": None})
    assert response.json()["plate_readable"] is False
    assert (await client.patch(det_path, json={"plate_readable": True})).status_code == 422
    assert (await client.delete(det_path)).status_code == 204
    assert (await client.get(BASE + "/detections")).json()["total"] == 0
    assert (await client.get(image_path + "/annotated")).status_code == 200


async def test_export_scope_and_csv_injection(client, image_bytes, one_detection_provider):
    first = await uploaded(client, image_bytes, filename="=cmd.jpg")
    await uploaded(client, image_bytes, process=False, filename="other.jpg")
    await process_job()
    params = {"image_id": first["image"]["id"]}
    results = (await client.get(BASE + "/exports/results.json", params=params)).json()
    assert len(results) == 1
    csv_text = (await client.get(BASE + "/exports/results.csv", params=params)).content.decode(
        "utf-8-sig"
    )
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert rows[0]["filename"].startswith("'=") and rows[0]["license_plate"] == "123 ABC 01"
    filtered = await client.get(BASE + "/detections", params={"license_plate": "%"})
    assert filtered.json()["total"] == 0


async def test_missing_storage_file_and_request_limit(client, image_bytes, monkeypatch):
    item = await uploaded(client, image_bytes, process=False)
    stored = next(LocalStorage().upload_dir.iterdir())
    stored.unlink()
    assert (await client.get(BASE + f"/images/{item['image']['id']}/file")).status_code == 404
    response = await client.post(
        BASE + "/images", content=b"", headers={"Content-Length": str(101 * 1024 * 1024)}
    )
    assert response.status_code == 413
    # Authentication runs before body parsing, even for oversized requests.
    response = await client.post(
        BASE + "/images",
        content=b"",
        headers={"Content-Length": str(101 * 1024 * 1024), "X-API-Key": "bad"},
    )
    assert response.status_code == 401


async def test_upload_file_count_limit(client, image_bytes, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_files", 1)
    response = await client.post(
        BASE + "/images",
        files=[
            ("files", ("1.jpg", image_bytes, "image/jpeg")),
            ("files", ("2.jpg", image_bytes, "image/jpeg")),
        ],
    )
    assert response.status_code == 400


async def test_stale_worker_cannot_overwrite_new_result(
    client, image_bytes, one_detection_provider, monkeypatch
):
    from app.services import processing

    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class Provider(one_detection_provider):
        async def analyze(self, *args):
            nonlocal calls
            calls += 1
            number = calls
            result, raw = await super().analyze(*args)
            result.detections[0].manufacturer = f"result-{number}"
            if number == 1:
                entered.set()
                await release.wait()
            return result, raw

    monkeypatch.setattr(processing, "get_vision_provider", lambda *args: Provider())
    item = await uploaded(client, image_bytes)
    first = asyncio.create_task(process_job())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        async with SessionLocal() as session, session.begin():
            from uuid import UUID

            job = await session.get(ProcessingJob, UUID(item["job"]["id"]))
            job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert await process_job()
    finally:
        release.set()
        await first
    result = (await client.get(BASE + "/detections")).json()
    assert result["total"] == 1
    assert result["items"][0]["manufacturer"] == "result-2"
    assert len(list(LocalStorage().annotated_dir.iterdir())) == 1


async def test_deadline_marks_failed(client, image_bytes, one_detection_provider, monkeypatch):
    from app.services import processing

    class Slow(one_detection_provider):
        async def analyze(self, *args):
            await asyncio.sleep(10)

    monkeypatch.setattr(processing, "get_vision_provider", lambda *args: Slow())
    monkeypatch.setattr(get_settings(), "job_timeout_seconds", 0.05)
    item = await uploaded(client, image_bytes)
    await process_job()
    job = (await client.get(BASE + f"/jobs/{item['job']['id']}")).json()
    assert job["status"] == "failed" and job["error_message"] == "Processing deadline exceeded"


async def test_running_worker_processes_persisted_queue(client, image_bytes):
    from app.services.processing import running_workers

    item = await uploaded(client, image_bytes)
    async with running_workers():
        async with asyncio.timeout(5):
            while True:
                result = await client.get(BASE + f"/jobs/{item['job']['id']}")
                if result.json()["status"] == "completed":
                    break
                await asyncio.sleep(0.05)


async def test_commit_failure_keeps_previous_correction(
    client, image_bytes, one_detection_provider, monkeypatch
):
    from sqlalchemy.ext.asyncio import AsyncSession

    item = await uploaded(client, image_bytes)
    await process_job()
    det = (await client.get(BASE + "/detections")).json()["items"][0]
    path = BASE + f"/detections/{det['id']}"

    async def fail_commit(self):
        raise OSError("simulated lost database connection")

    with monkeypatch.context() as patcher:
        patcher.setattr(AsyncSession, "commit", fail_commit)
        assert (await client.patch(path, json={"manufacturer": "bad-change"})).status_code == 500
    assert (await client.get(path)).json()["manufacturer"] == "KamAZ"
    assert (await client.get(BASE + f"/images/{item['image']['id']}/annotated")).status_code == 200


async def test_pagination_not_processed_and_export_limit(client, image_bytes, monkeypatch):
    first = await uploaded(client, image_bytes, process=False)
    await uploaded(client, image_bytes, process=False)
    images = (await client.get(BASE + "/images", params={"limit": 1})).json()
    assert len(images) == 1
    result = (
        await client.get(BASE + "/exports/results.json", params={"image_id": first["image"]["id"]})
    ).json()[0]
    assert result["status"] == "not_processed" and result["provider"] is None
    monkeypatch.setattr(get_settings(), "export_max_images", 1)
    assert (await client.get(BASE + "/exports/results.json")).status_code == 413
    assert len((await client.get(BASE + "/exports/results.json", params={"limit": 1})).json()) == 1


async def test_http_provider_through_worker(client, image_bytes, monkeypatch):
    import json

    import httpx

    from app.services import processing
    from app.services.vision.openai_compatible import OpenAICompatibleVisionProvider

    def handler(request):
        assert json.loads(request.content)["model"] == "contract-test-model"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "detections": [
                                        {
                                            "vehicle_type": "tractor",
                                            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.8, "y2": 0.8},
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        processing,
        "get_vision_provider",
        lambda *args: OpenAICompatibleVisionProvider(
            model="contract-test-model", transport=httpx.MockTransport(handler)
        ),
    )
    await uploaded(client, image_bytes)
    await process_job()
    result = (await client.get(BASE + "/detections")).json()["items"]
    assert len(result) == 1 and result[0]["vehicle_type"] == "tractor"
