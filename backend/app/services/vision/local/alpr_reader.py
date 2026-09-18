"""Adapt single-line ALPR to the existing vehicle/plate/review schema."""
from __future__ import annotations

import base64
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.services.alpr_processor import ALPRConfig, ALPRProcessor, PaddleRecognizer
from app.services.vision.local import geometry, plate_format
from app.services.vision.local.plate_ocr import OcrRead, PlateReader, PlateReading


class PlateCropCorners:
    """Refine real plate boundaries inside the existing YOLO plate crop."""

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        estimate = geometry.estimate_quad(image)
        if estimate is None or estimate.aspect < 2.2:
            return []
        quad = estimate.quad.copy()
        # The fitter can put a boundary just outside the crop; clip subpixel edges.
        quad[:, 0] = np.clip(quad[:, 0], 0, image.shape[1] - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, image.shape[0] - 1)
        return [quad.astype(np.float32)]


class ALPRPlateReader:
    def __init__(self, fallback: PlateReader, model_dir: Path, sr_path: Path | None):
        self.fallback = fallback
        self.processor = ALPRProcessor(
            ocr_model_dir=model_dir,
            sr_model_path=sr_path,
            detector=PlateCropCorners(),
            config=ALPRConfig(min_confidence=max(0.65, fallback.accept_confidence)),
        )

    def _read_region(self, warp: np.ndarray, estimate) -> PlateReading | None:
        """Read small region separately; consensus alone never increases confidence.

        Four bounded hypotheses cover the typical right-hand region layout. A
        stable LDDDLL main text and 2/3 actual CTC digits must agree across crops.
        This creates a review candidate below the acceptance floor, not a fact.
        """
        recognizer = self.processor._recognizer
        if not isinstance(recognizer, PaddleRecognizer):
            return None
        bodies: list[tuple[str, float]] = []
        height, width = warp.shape[:2]
        splits = (0.72, 0.74, 0.76, 0.78)
        for split in splits:
            main = cv2.copyMakeBorder(
                warp[:, :int(width * split)], 8, 8, 8, 8, cv2.BORDER_REPLICATE
            )
            raw, score = recognizer.recognize(main)
            text = raw.upper().replace(" ", "")
            if score >= 0.80 and re.fullmatch(r"[A-Z][0-9]{3}[A-Z]{2}", text):
                bodies.append((text, score))
        counts = Counter(text for text, _ in bodies)
        if not counts:
            return None
        body, count = counts.most_common(1)[0]
        if count < 2 or len(counts) > 1:
            return None
        regions: list[tuple[str, float]] = []
        for split in splits:
            region = cv2.copyMakeBorder(
                warp[:int(height * 0.75), int(width * split):],
                8, 8, 8, 8, cv2.BORDER_REPLICATE,
            )
            text, score = recognizer.recognize_allowed(region, "0123456789")
            if score >= 0.20 and re.fullmatch(r"[0-9]{2,3}", text) and int(text) != 0:
                regions.append((text, score))
        counts = Counter(text for text, _ in regions)
        if not counts or len(counts) > 1:
            return None
        region, count = counts.most_common(1)[0]
        if count < 2:
            return None
        text = body + region
        score = min(
            float(np.median([s for t, s in bodies if t == body])),
            float(np.median([s for t, s in regions if t == region])),
        )
        accepted = score >= max(0.75, self.fallback.accept_confidence)
        return PlateReading(
            text=plate_format.format_for_display(text) if accepted else None,
            raw_text=text, readable=accepted, ocr_confidence=round(score, 4),
            engine="paddleocr_alpr", variant="separate_region_ctc",
            plate_format="letter_digits_letters_region", region=region,
            rectified=True, skew=estimate.skew if estimate else None,
            reason="accepted_region" if accepted else "region_needs_review",
            candidates=[OcrRead("paddleocr_alpr", "separate_region_ctc", text, score, 0.0)],
            quad=estimate, rectified_image=warp,
        )

    def read(self, crop_bgr: np.ndarray) -> PlateReading:
        if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 8:
            return self.fallback.read(crop_bgr)
        result = self.processor.process_image(crop_bgr)
        text = result["plate_text"]
        estimate = geometry.estimate_quad(crop_bgr)
        aspect = estimate.aspect if estimate is not None and estimate.trusted_aspect else None
        fmt = plate_format.classify(text, aspect)
        rectified = None
        if result["warp_image_base64"]:
            rectified = cv2.imdecode(
                np.frombuffer(base64.b64decode(result["warp_image_base64"]), np.uint8), cv2.IMREAD_COLOR
            )
        if rectified is not None and (fmt is None or fmt.name in {"kz_private_pre2012", "kz_other_pre2012"}):
            regional = self._read_region(rectified, estimate)
            if regional is not None:
                return regional
        if not text:
            return self.fallback.read(crop_bgr)
        if fmt is None or (
            fmt.provisional and result["confidence"] < self.fallback.accept_confidence_unformatted
        ):
            return self.fallback.read(crop_bgr)
        # ALPR exports mean confidence; do not invent a per-character minimum.
        candidate = OcrRead("paddleocr_alpr", "three_variants", text, result["confidence"], 0.0)
        return PlateReading(
            text=plate_format.format_for_display(text) if fmt else text,
            raw_text=text,
            readable=True,
            ocr_confidence=result["confidence"],
            engine="paddleocr_alpr",
            variant="three_variants",
            plate_format=fmt.name if fmt else None,
            region=plate_format.region_name(text) if fmt else None,
            rectified=True,
            skew=estimate.skew if estimate else None,
            reason="accepted_alpr",
            candidates=[candidate],
            quad=estimate,
            rectified_image=rectified,
        )
