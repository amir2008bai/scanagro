"""The review UI and the thumbnail endpoint it depends on.

The UI is static files served by the API. Two things must hold for it to work at all: the
page has to load *before* a key is typed, and everything behind /api/v1 must still demand
that key. Both are asserted here, because getting the first one wrong makes the page
unusable and getting the second one wrong opens the API.
"""

import io

import pytest
from PIL import Image

BASE = "/api/v1"


@pytest.fixture
def big_image_bytes():
    """Wide enough that a thumbnail is genuinely smaller, unlike the 128x96 fixture."""
    out = io.BytesIO()
    Image.new("RGB", (2688, 1520), "white").save(out, format="JPEG", quality=95)
    return out.getvalue()


async def upload(client, data, name="frame.jpg"):
    response = await client.post(
        BASE + "/images?process=false",
        files=[("files", (name, data, "image/jpeg"))],
    )
    assert response.status_code == 201, response.text
    return response.json()["items"][0]["image"]["id"]


async def test_thumbnail_is_served_and_much_smaller(client, db, big_image_bytes):
    image_id = await upload(client, big_image_bytes)

    full = await client.get(BASE + f"/images/{image_id}/file")
    thumb = await client.get(BASE + f"/images/{image_id}/thumbnail")

    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert thumb.content.startswith(b"\xff\xd8"), "thumbnail must be a real JPEG"
    # The point of the endpoint: a grid of these instead of multi-megabyte originals.
    assert len(thumb.content) < len(full.content) / 5

    with Image.open(io.BytesIO(thumb.content)) as decoded:
        assert max(decoded.size) <= 480
        assert decoded.size[0] / decoded.size[1] == pytest.approx(2688 / 1520, rel=0.02)


async def test_thumbnail_is_cached_and_byte_identical(client, db, big_image_bytes):
    from app.services.storage import LocalStorage

    image_id = await upload(client, big_image_bytes)
    first = await client.get(BASE + f"/images/{image_id}/thumbnail")
    cached_files = list(LocalStorage().thumb_dir.iterdir())
    second = await client.get(BASE + f"/images/{image_id}/thumbnail")

    assert len(cached_files) == 1, "the first request should leave exactly one cache file"
    assert first.content == second.content
    assert not any(p.name.endswith(".tmp") for p in LocalStorage().thumb_dir.iterdir())


async def test_deleting_an_image_removes_its_thumbnail(client, db, big_image_bytes):
    from app.services.storage import LocalStorage

    image_id = await upload(client, big_image_bytes)
    await client.get(BASE + f"/images/{image_id}/thumbnail")
    assert list(LocalStorage().thumb_dir.iterdir())

    assert (await client.delete(BASE + f"/images/{image_id}")).status_code == 204
    assert not list(LocalStorage().thumb_dir.iterdir()), "thumbnail outlived its image"


async def test_thumbnail_of_a_missing_image_is_404(client, db):
    response = await client.get(BASE + "/images/00000000-0000-0000-0000-000000000000/thumbnail")
    assert response.status_code == 404


async def test_thumbnail_still_needs_the_api_key(db, big_image_bytes):
    import httpx

    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as anonymous:
        response = await anonymous.get(
            BASE + "/images/00000000-0000-0000-0000-000000000000/thumbnail"
        )
    assert response.status_code == 401


async def test_ui_loads_without_a_key_but_the_api_does_not(db):
    """The page carries no secrets and must load first; the API behind it stays shut."""
    import httpx

    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as anonymous:
        page = await anonymous.get("/ui/")
        script = await anonymous.get("/ui/app.js")
        api = await anonymous.get(BASE + "/images")

    assert page.status_code == 200 and b"<html" in page.content.lower()
    assert script.status_code == 200
    assert api.status_code == 401, "the UI mount must not have opened the API"


async def test_ui_mount_cannot_be_used_to_escape_the_directory(db):
    import httpx

    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as anonymous:
        for attempt in ("/ui/../.env", "/ui/%2e%2e/.env", "/ui/../app/main.py"):
            response = await anonymous.get(attempt)
            assert response.status_code in (400, 401, 403, 404), attempt
            assert b"API_KEY" not in response.content


async def test_root_advertises_the_ui(client, db):
    body = (await client.get("/")).json()
    assert body["ui"] == "/ui/"
