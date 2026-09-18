"""Standalone example: uvicorn examples.alpr_api:app --host 127.0.0.1 --port 8081.

POST raw JPEG/PNG bytes to /recognize with Content-Type: image/jpeg or image/png.
Requires ALPR_OCR_MODEL_DIR; optional ALPR_SR_MODEL_PATH and ALPR_PATTERNS (CSV).
"""
import logging
import os
import warnings
from contextlib import asynccontextmanager
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.services.alpr_processor import ALPRConfig, ALPRError, ALPRProcessor, ALPRResult

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    patterns = tuple(p.strip() for p in os.environ.get("ALPR_PATTERNS", "").split(",") if p.strip())
    app.state.alpr = ALPRProcessor(
        ocr_model_dir=os.environ["ALPR_OCR_MODEL_DIR"],
        sr_model_path=os.environ.get("ALPR_SR_MODEL_PATH") or None,
        config=ALPRConfig(patterns=patterns, max_pixels=MAX_PIXELS),
    )
    yield
    del app.state.alpr


app = FastAPI(title="Local ALPR", lifespan=lifespan)


def decode_image(payload: bytes) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as header:
                if header.format not in {"JPEG", "PNG"}:
                    raise HTTPException(415, "Only JPEG and PNG are supported")
                if min(header.size) < 8 or header.width * header.height > MAX_PIXELS:
                    raise HTTPException(413, "Image dimensions outside limits")
                header.verify()
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(413, "Image exceeds pixel limit") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, cv2.error) as exc:
        raise HTTPException(400, "Invalid image") from exc
    if image is None:
        raise HTTPException(400, "Invalid image")
    return image


@app.post("/recognize", response_model=ALPRResult)
async def recognize(request: Request) -> ALPRResult:
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "Maximum image size is 10 MiB")
        data.extend(chunk)
    image = await run_in_threadpool(decode_image, bytes(data))
    try:
        return await run_in_threadpool(request.app.state.alpr.process_image, image)
    except ValueError as exc:
        raise HTTPException(422, "Image outside processing limits") from exc
    except ALPRError as exc:
        logger.error("ALPR inference failed")
        raise HTTPException(503, "Recognition unavailable") from exc
