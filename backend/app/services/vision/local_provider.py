"""`VisionProvider` adapter for the local pipeline.

The rest of the service is unchanged: it still asks a provider to analyse a path and gets
back a `VisionImageResult`. What is new is that this provider does the recognition in
this process instead of calling out to a model server, so it also owns the concurrency
limit that keeps several workers from each holding a full set of intermediate tensors.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from app.core.config import Settings, get_settings
from app.schemas.vision import VisionImageResult
from app.services.vision.base import VisionProvider
from app.services.vision.local import runtime
from app.services.vision.local.pipeline import LocalRecognitionPipeline, PipelineConfig

logger = structlog.get_logger(__name__)


def build_config(settings: Settings) -> PipelineConfig:
    return PipelineConfig(
        model_dir=Path(settings.vision_model_dir),
        device=settings.vision_device,
        onnx_threads=settings.vision_onnx_threads,
        max_parallel_images=settings.vision_max_parallel_images,
        vehicle_input_size=settings.vision_vehicle_input_size,
        vehicle_threshold=settings.vision_vehicle_threshold,
        plate_model=settings.vision_plate_model,
        plate_threshold=settings.vision_plate_threshold,
        plate_tile=settings.vision_plate_tile,
        plate_tile_overlap=settings.vision_plate_tile_overlap,
        plate_frame_sweep=settings.vision_plate_frame_sweep,
        ocr_model=settings.vision_ocr_model,
        use_ppocr=settings.vision_use_ppocr,
        alpr_enabled=settings.vision_alpr_enabled,
        alpr_ocr_dir=settings.vision_alpr_ocr_dir,
        alpr_sr_path=settings.vision_alpr_sr_path,
        ocr_accept_confidence=settings.vision_ocr_accept_confidence,
        ocr_accept_confidence_unformatted=settings.vision_ocr_accept_confidence_unformatted,
        read_attributes=settings.vision_read_attributes,
        cyrillic_attributes=settings.vision_cyrillic_attributes,
        reference_gallery=settings.vision_reference_gallery,
        reference_dir=Path(settings.vision_reference_dir),
        reference_min_similarity=settings.vision_reference_min_similarity,
        reference_min_margin=settings.vision_reference_min_margin,
        reference_top_k=settings.vision_reference_top_k,
        overlay_band_top=settings.vision_overlay_band_top,
        save_plate_crops=settings.vision_save_plate_crops,
    )


def get_pipeline(settings: Settings | None = None) -> LocalRecognitionPipeline:
    """One pipeline per process, built on first use and reused afterwards."""
    settings = settings or get_settings()
    config = build_config(settings)
    return runtime.get_or_create(
        f"pipeline::{config!r}",
        lambda: LocalRecognitionPipeline(config),
    )


class LocalVisionProvider(VisionProvider):
    """Runs the open-source recognition stack in-process on CPU or CUDA."""

    name = "local"

    def __init__(self, model: str | None = None, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self.model = model or self._settings.vision_local_model_id

    async def analyze(
        self, image_path: Path, mime_type: str
    ) -> tuple[VisionImageResult, dict[str, Any]]:
        settings = self._settings
        pipeline = get_pipeline(settings)
        slot = runtime.concurrency_slot(settings.vision_max_parallel_images)

        def run() -> tuple[VisionImageResult, dict[str, Any]]:
            # Bound how many images are in flight: the sessions are shared, the
            # intermediate tensors are not, and those are what drive peak memory.
            with slot:
                return pipeline.analyze_path(image_path, crop_sink=_crop_sink(settings))

        try:
            result, raw = await asyncio.to_thread(run)
        except FileNotFoundError as exc:
            raise VisionModelUnavailable(str(exc)) from exc
        raw["model"] = self.model
        await logger.ainfo(
            "local_recognition_done",
            detections=len(result.detections),
            seconds=raw.get("total_seconds"),
        )
        return result, raw


class VisionModelUnavailable(RuntimeError):
    """Raised when the configured weights are missing, so the job fails with a clear cause."""


def _crop_sink(settings: Settings):
    """Persist the plate crop before and after rectification, for human review."""
    if not settings.vision_save_plate_crops:
        return None

    from app.services.storage import LocalStorage

    storage = LocalStorage()

    def sink(before, after):
        import cv2

        storage.ensure_directories()
        stem = uuid4().hex
        before_name = f"{stem}-before.jpg"
        cv2.imwrite(str(storage.plate_path(before_name)), _display(before))
        after_name = None
        if after is not None and after.size:
            after_name = f"{stem}-after.jpg"
            cv2.imwrite(str(storage.plate_path(after_name)), _display(after))
        return before_name, after_name

    return sink


def _display(image, min_height: int = 120):
    """Upscale a thumbnail-sized crop so a reviewer can actually see it."""
    import cv2

    height = image.shape[0]
    if height >= min_height:
        return image
    scale = min_height / max(1, height)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
