"""Stage 2 — locate licence plates.

The frames are 2688x1520 and a plate is typically 60-140 px wide, so feeding the whole
frame to a 608x608 detector shrinks the plate to ~20 px and it disappears. The detector
is therefore run at native resolution over overlapping tiles of each vehicle crop, plus
one sweep of the full frame to catch plates on machinery the vehicle stage missed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.vision.local import runtime
from app.services.vision.local.vehicle import VehicleBox


@dataclass(slots=True)
class PlateBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    source: str  # "vehicle" or "frame"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class PlateDetector:
    """YOLOv9 plate detector from `open-image-models` (MIT), driven tile by tile."""

    def __init__(
        self,
        model_name: str,
        model_dir: Path,
        conf_threshold: float = 0.25,
        tile: int = 608,
        overlap: float = 0.35,
        max_upscale: float = 3.0,
        device: str = "cpu",
        threads: int = 0,
    ):
        self.model_name = model_name
        self._model_dir = Path(model_dir)
        self._device = device
        self._threads = threads
        self.conf_threshold = float(conf_threshold)
        self.tile = int(tile)
        self.overlap = float(overlap)
        self.max_upscale = float(max_upscale)

    def _load(self):
        # Loaded by explicit path from our own model directory. The library's default is
        # to fetch into ~/.cache, which a container with a read-only root and no home
        # directory cannot write, and which no checksum guards.
        from open_image_models.detection.factory import create_detector

        from app.services.vision.local.model_registry import YOLO_V9_PLATE, verify

        weights = self._model_dir / YOLO_V9_PLATE.filename
        verify(YOLO_V9_PLATE, weights)
        return create_detector(
            str(weights),
            backend="yolo_v9",
            class_labels=("License Plate",),
            conf_thresh=self.conf_threshold,
            providers=runtime._providers(self._device),
            sess_options=runtime.session_options(self._threads),
        )

    @property
    def _detector(self):
        return runtime.get_or_create(
            f"plate_detector::{self.model_name}::{self._device}", self._load
        )

    def _predict(self, patch: np.ndarray, ox: int, oy: int, source: str) -> list[PlateBox]:
        out = []
        for det in self._detector.predict(patch):
            box = det.bounding_box
            out.append(
                PlateBox(
                    ox + int(box.x1),
                    oy + int(box.y1),
                    ox + int(box.x2),
                    oy + int(box.y2),
                    float(det.confidence),
                    source,
                )
            )
        return out

    def _scan(self, patch: np.ndarray, ox: int, oy: int, source: str) -> list[PlateBox]:
        import cv2

        height, width = patch.shape[:2]
        tile = self.tile
        if height <= tile and width <= tile:
            # Small crop: upscale so the plate covers more of the detector's input grid.
            scale = min(self.max_upscale, tile / max(1, max(height, width)))
            if scale > 1.05:
                big = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                found = self._predict(big, 0, 0, source)
                return [
                    PlateBox(
                        ox + int(b.x1 / scale),
                        oy + int(b.y1 / scale),
                        ox + int(b.x2 / scale),
                        oy + int(b.y2 / scale),
                        b.confidence,
                        source,
                    )
                    for b in found
                ]
            return self._predict(patch, ox, oy, source)

        step = max(1, int(tile * (1.0 - self.overlap)))
        ys = list(range(0, max(height - tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(width - tile, 0) + 1, step)) or [0]
        if ys[-1] + tile < height:
            ys.append(height - tile)
        if xs[-1] + tile < width:
            xs.append(width - tile)
        out: list[PlateBox] = []
        for y in ys:
            for x in xs:
                out += self._predict(patch[y : y + tile, x : x + tile], ox + x, oy + y, source)
        return out

    def detect(
        self,
        image_bgr: np.ndarray,
        vehicles: list[VehicleBox],
        pad: int = 24,
        frame_sweep: bool = True,
    ) -> list[PlateBox]:
        height, width = image_bgr.shape[:2]
        found: list[PlateBox] = []
        for vehicle in vehicles:
            x1, y1 = max(0, vehicle.x1 - pad), max(0, vehicle.y1 - pad)
            x2, y2 = min(width, vehicle.x2 + pad), min(height, vehicle.y2 + pad)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            found += self._scan(image_bgr[y1:y2, x1:x2], x1, y1, "vehicle")
        # Sweeping the whole frame costs another full tiling pass, so it is a fallback:
        # it runs when the vehicle crops yielded nothing, or when the vehicle stage found
        # nothing to crop in the first place.
        if frame_sweep and (not vehicles or not found):
            found += self._scan(image_bgr, 0, 0, "frame")
        return suppress(found)


def suppress(boxes: list[PlateBox], overlap_threshold: float = 0.3) -> list[PlateBox]:
    """Tiles overlap by design, so the same plate is found several times."""
    kept: list[PlateBox] = []
    for box in sorted(boxes, key=lambda b: -b.confidence):
        duplicate = False
        for existing in kept:
            ix = max(0, min(box.x2, existing.x2) - max(box.x1, existing.x1))
            iy = max(0, min(box.y2, existing.y2) - max(box.y1, existing.y1))
            inter = ix * iy
            smaller = max(1, min(box.area, existing.area))
            if inter / smaller > overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept
