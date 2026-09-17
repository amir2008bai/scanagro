import asyncio
import hashlib
import io
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)
_decode_slots = asyncio.Semaphore(settings.worker_concurrency)
FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}


@dataclass(slots=True)
class StoredImage:
    stored_filename: str
    original_filename: str
    mime_type: str
    file_size: int
    sha256: str
    width: int
    height: int


class LocalStorage:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.storage_dir).resolve()
        self.upload_dir = self.root / "uploads"
        self.annotated_dir = self.root / "annotated"
        # Plate crops before and after perspective correction, kept as review evidence.
        self.plate_dir = self.root / "plates"

    def ensure_directories(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.plate_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile) -> StoredImage:
        content_type = (upload.content_type or "").split(";")[0].lower()
        if content_type not in settings.allowed_content_types:
            raise HTTPException(415, "Supported image types: JPEG, PNG, WEBP")
        async with _decode_slots:
            data = await upload.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, "Image exceeds MAX_UPLOAD_MB")
            if not data:
                raise HTTPException(400, "Empty image")
            return await asyncio.to_thread(self._save, data, content_type, upload.filename)

    def _save(self, data: bytes, content_type: str, filename: str | None) -> StoredImage:
        self.ensure_directories()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    fmt = source.format
                    if fmt not in FORMATS or FORMATS[fmt][0] != content_type:
                        raise HTTPException(415, "Declared MIME type does not match image contents")
                    if source.width * source.height > settings.max_image_pixels:
                        raise HTTPException(413, "Image exceeds MAX_IMAGE_PIXELS")
                    if getattr(source, "n_frames", 1) != 1:
                        raise HTTPException(415, "Animated or multi-frame images are not supported")
                    source.verify()
                with Image.open(io.BytesIO(data)) as source:
                    source.load()  # verify() alone does not reject all truncated JPEGs
                    with ImageOps.exif_transpose(source) as oriented:
                        rgba = oriented.convert("RGBA")
                        image = Image.new("RGB", oriented.size, "white")
                        image.paste(rgba, mask=rgba.getchannel("A"))
                        rgba.close()
                        image.info.clear()  # Strip EXIF, GPS, XMP and untrusted metadata
                        out = io.BytesIO()
                        image.save(out, format=fmt, **({"quality": 95} if fmt != "PNG" else {}))
                        width, height = image.size
                        image.close()
                        canonical = out.getvalue()
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise HTTPException(413, "Image dimensions are too large") from exc
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise HTTPException(400, "Invalid or truncated image") from exc
        if len(canonical) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "Normalized image exceeds MAX_UPLOAD_MB")
        if (
            shutil.disk_usage(self.root).free
            < len(canonical) + settings.min_free_disk_mb * 1024 * 1024
        ):
            raise HTTPException(507, "Insufficient image storage")
        stored_filename = uuid4().hex + FORMATS[fmt][1]
        target = self.original_path(stored_filename)
        temp = target.with_suffix(target.suffix + ".tmp")
        try:
            with temp.open("xb") as stream:
                stream.write(canonical)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        original = (filename or stored_filename).replace("\\", "/").split("/")[-1]
        original = re.sub(r"[\x00-\x1f\x7f]", "", original).strip()[:255] or stored_filename
        return StoredImage(
            stored_filename,
            original,
            content_type,
            len(canonical),
            hashlib.sha256(canonical).hexdigest(),
            width,
            height,
        )

    @staticmethod
    def _safe_path(directory: Path, filename: str) -> Path:
        if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("Unsafe storage filename")
        target = (directory / filename).resolve()
        if target.parent != directory.resolve():
            raise ValueError("Storage path escapes root")
        return target

    def original_path(self, stored_filename: str) -> Path:
        return self._safe_path(self.upload_dir, stored_filename)

    def annotated_path(self, annotated_filename: str) -> Path:
        return self._safe_path(self.annotated_dir, annotated_filename)

    def plate_path(self, plate_filename: str) -> Path:
        return self._safe_path(self.plate_dir, plate_filename)

    def remove_annotation(self, filename: str | None):
        if filename:
            self._unlink(self.annotated_path(filename))

    def remove_plate_crop(self, filename: str | None):
        if filename:
            try:
                self._unlink(self.plate_path(filename))
            except ValueError:
                logger.warning("unsafe_plate_filename")

    @staticmethod
    def _unlink(path: Path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("file_cleanup_failed", filename=path.name)

    def delete_image_files(self, stored_filename: str, annotated_filename: str | None) -> None:
        self._unlink(self.original_path(stored_filename))
        self.remove_annotation(annotated_filename)
