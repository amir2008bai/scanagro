"""Stage 1 — find every piece of machinery in the frame.

RT-DETRv2 is trained on COCO, which has no class for agricultural machinery: a Kirovets
or an MTZ is reported as `truck`, sometimes `train` or `bus`. We therefore keep the raw
COCO label as evidence, map it onto a coarse in-domain vocabulary, and let the attribute
stage refine the type when the image actually shows a name plate or badge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services.vision.local import runtime
from app.services.vision.local.model_registry import RTDETR_V2_R18, RTDETR_V2_R18_CONFIG, verify

#: COCO classes that can plausibly be a vehicle in a weighbridge yard.
VEHICLE_CLASSES: dict[str, str] = {
    "truck": "truck",
    "bus": "bus",
    "car": "car",
    "train": "truck",  # elongated grain trailers are routinely called `train` by COCO
    "boat": "unknown_vehicle",
    "motorcycle": "motorcycle",
}

#: Classes whose COCO meaning is too weak to stand on its own; kept only when large.
WEAK_CLASSES = {"train", "boat"}


@dataclass(slots=True)
class VehicleBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    coco_label: str
    vehicle_type: str
    evidence: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


class VehicleDetector:
    """RT-DETRv2 ONNX wrapper. One session per process, reused across images."""

    def __init__(
        self, model_dir: Path, device: str = "cpu", threads: int = 0, input_size: int = 960
    ):
        self._model_dir = Path(model_dir)
        self._device = device
        self._threads = threads
        self.input_size = int(input_size)

    def _load(self):
        import onnxruntime as ort

        weights = self._model_dir / RTDETR_V2_R18.filename
        config = self._model_dir / RTDETR_V2_R18_CONFIG.filename
        verify(RTDETR_V2_R18, weights)
        if not config.is_file():
            raise FileNotFoundError(f"Missing {config}. Run scripts/fetch_models.py.")
        raw = json.loads(config.read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in raw["id2label"].items()}
        session = ort.InferenceSession(
            str(weights),
            sess_options=runtime.session_options(self._threads),
            providers=runtime._providers(self._device),
        )
        return session, id2label

    @property
    def _session(self):
        return runtime.get_or_create(f"rtdetr::{self._model_dir}::{self._device}", self._load)

    def detect(self, image_bgr: np.ndarray, threshold: float = 0.30) -> list[VehicleBox]:
        import cv2

        session, id2label = self._session
        height, width = image_bgr.shape[:2]
        size = self.input_size
        resized = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), (size, size))
        tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        logits, boxes = session.run(None, {"pixel_values": tensor})
        scores = 1.0 / (1.0 + np.exp(-logits[0]))
        best = scores.max(axis=1)
        labels = scores.argmax(axis=1)

        found: list[VehicleBox] = []
        for index in np.flatnonzero(best > threshold):
            coco = id2label.get(int(labels[index]), "unknown")
            if coco not in VEHICLE_CLASSES:
                continue
            cx, cy, bw, bh = boxes[0][index]
            x1 = int(round((cx - bw / 2) * width))
            y1 = int(round((cy - bh / 2) * height))
            x2 = int(round((cx + bw / 2) * width))
            y2 = int(round((cy + bh / 2) * height))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            box = VehicleBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(best[index]),
                coco_label=coco,
                vehicle_type=VEHICLE_CLASSES[coco],
                evidence={"detector": "rtdetr_v2_r18vd", "coco_label": coco},
            )
            if coco in WEAK_CLASSES and box.area < 0.01 * width * height:
                continue
            found.append(box)
        return _suppress(found)


def _iou(a: VehicleBox, b: VehicleBox) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _contained(inner: VehicleBox, outer: VehicleBox) -> float:
    ix = max(0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    iy = max(0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    return (ix * iy) / inner.area if inner.area else 0.0


def _suppress(
    boxes: list[VehicleBox], iou_threshold: float = 0.55, containment: float = 0.88
) -> list[VehicleBox]:
    """Drop duplicate and nested boxes.

    RT-DETR emits several queries per object; a tractor pulling a trailer also yields a
    box for the pair. Keeping the highest-scoring box and removing anything that is
    almost entirely inside it gives one detection per physical machine.
    """
    kept: list[VehicleBox] = []
    for box in sorted(boxes, key=lambda b: -b.confidence):
        duplicate = False
        for existing in kept:
            if _iou(box, existing) > iou_threshold or _contained(box, existing) > containment:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept
