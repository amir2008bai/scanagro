"""Process-wide lifecycle for the recognition models.

Loading ONNX weights costs seconds and hundreds of megabytes, so every model is built
once per process, guarded by a lock, and reused for every image. A semaphore bounds how
many images may be inside the pipeline at the same time, which is what actually caps
peak RSS: the sessions themselves are shared, the per-image tensors are not.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {}
_SLOTS: threading.BoundedSemaphore | None = None
_SLOT_LOCK = threading.Lock()


def _providers(device: str) -> list[str]:
    """Resolve the ONNX Runtime execution providers for the requested device."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "VISION_DEVICE=cuda but onnxruntime has no CUDAExecutionProvider. "
                "Install onnxruntime-gpu with a matching CUDA runtime, or use VISION_DEVICE=cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "auto" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def session_options(threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return opts


def concurrency_slot(limit: int) -> threading.BoundedSemaphore:
    """One shared semaphore per process; the first caller fixes the limit."""
    global _SLOTS
    with _SLOT_LOCK:
        if _SLOTS is None:
            _SLOTS = threading.BoundedSemaphore(max(1, limit))
        return _SLOTS


def get_or_create(key: str, factory) -> Any:
    """Build `key` exactly once per process, even under concurrent first use."""
    existing = _STATE.get(key)
    if existing is not None:
        return existing
    with _LOCK:
        existing = _STATE.get(key)
        if existing is None:
            logger.info("loading recognition model: %s", key)
            existing = factory()
            _STATE[key] = existing
    return existing


def loaded_keys() -> list[str]:
    with _LOCK:
        return sorted(_STATE)


def release_all() -> None:
    """Drop every cached model. Used by tests and by graceful shutdown."""
    with _LOCK:
        _STATE.clear()


def model_cache_dir(configured: Path | None = None) -> Path:
    """Directory the third-party libraries should use for their own weight caches."""
    if configured is not None:
        path = Path(configured)
    else:
        path = Path(os.environ.get("VISION_MODEL_DIR", "./data/models"))
    path.mkdir(parents=True, exist_ok=True)
    return path
