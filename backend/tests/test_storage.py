import io

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

from app.core.config import get_settings
from app.services.storage import LocalStorage


def upload(data, mime="image/jpeg", filename="input.jpg"):
    return UploadFile(io.BytesIO(data), filename=filename, headers=Headers({"content-type": mime}))


async def test_canonical_image_and_safe_filename(tmp_path, image_bytes):
    storage = LocalStorage(tmp_path)
    result = await storage.save_upload(upload(image_bytes, filename="C:\\folder\\test.jpg"))
    assert result.width == 128 and result.height == 96
    assert result.original_filename == "test.jpg"
    assert storage.original_path(result.stored_filename).is_file()
    assert result.file_size > 0 and len(result.sha256) == 64


@pytest.mark.parametrize(
    "data,mime,code",
    [(b"", "image/jpeg", 400), (b"garbage", "image/jpeg", 400), (b"data", "text/html", 415)],
)
async def test_invalid_uploads(tmp_path, data, mime, code):
    with pytest.raises(HTTPException) as error:
        await LocalStorage(tmp_path).save_upload(upload(data, mime))
    assert error.value.status_code == code


async def test_mime_spoofing(tmp_path, image_bytes):
    with pytest.raises(HTTPException) as error:
        await LocalStorage(tmp_path).save_upload(upload(image_bytes, "image/png"))
    assert error.value.status_code == 415


async def test_truncated_jpeg(tmp_path, image_bytes):
    with pytest.raises(HTTPException) as error:
        await LocalStorage(tmp_path).save_upload(upload(image_bytes[:-40]))
    assert error.value.status_code == 400


async def test_pixel_and_byte_limits(tmp_path, image_bytes, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_image_pixels", 100)
    with pytest.raises(HTTPException) as error:
        await LocalStorage(tmp_path).save_upload(upload(image_bytes))
    assert error.value.status_code == 413
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    with pytest.raises(HTTPException) as error:
        await LocalStorage(tmp_path).save_upload(upload(b"x" * (1024 * 1024 + 1)))
    assert error.value.status_code == 413


async def test_exif_orientation_and_metadata_removal(tmp_path):
    out = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    exif[315] = "private photographer"
    Image.new("RGB", (80, 40), "red").save(out, "JPEG", exif=exif)
    storage = LocalStorage(tmp_path)
    result = await storage.save_upload(upload(out.getvalue()))
    assert (result.width, result.height) == (40, 80)
    with Image.open(storage.original_path(result.stored_filename)) as normalized:
        assert not normalized.getexif()


async def test_animated_webp_rejected(tmp_path):
    out = io.BytesIO()
    Image.new("RGB", (40, 40), "red").save(
        out, "WEBP", save_all=True, append_images=[Image.new("RGB", (40, 40), "blue")], duration=100
    )
    with pytest.raises(HTTPException) as error:
        await LocalStorage(tmp_path).save_upload(upload(out.getvalue(), "image/webp"))
    assert error.value.status_code == 415


@pytest.mark.parametrize("filename", ["../outside", "..\\outside", "/etc/passwd", "C:\\test", ""])
def test_path_traversal_rejected(tmp_path, filename):
    with pytest.raises(ValueError):
        LocalStorage(tmp_path).original_path(filename)
