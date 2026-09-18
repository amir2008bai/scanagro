"""Local single-line ALPR. BGR uint8 input; no model download or external API.

PaddleOCR 3 TextRecognition does not support EasyOCR's ``allowlist``.
We validate its output strictly; the model's trained dictionary is never replaced.
See docs/ALPR_PROCESSOR.md for deployment, limits and the detector extension point.
"""
from __future__ import annotations

import base64
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict

Image = NDArray[np.uint8]
Quad = NDArray[np.float32]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LOGGER = logging.getLogger(__name__)


class ALPRError(RuntimeError):
    """Model initialization or inference failed (distinct from no plate found)."""


class ALPRResult(TypedDict):
    plate_text: str
    confidence: float
    warp_image_base64: str


class Recognizer(Protocol):
    def recognize(self, image: Image) -> tuple[str, float]: ...


class CornerDetector(Protocol):
    def detect(self, image: Image) -> Sequence[Quad]:
        """Return candidate quadrilaterals in ORIGINAL image coordinates."""
        ...


@dataclass(frozen=True)
class ALPRConfig:
    # D = digit; L = Latin letter; X = either. Empty means no positional guesses.
    patterns: tuple[str, ...] = ()
    min_confidence: float = 0.65
    min_length: int = 4
    max_length: int = 12
    max_pixels: int = 24_000_000
    max_candidates: int = 5
    detection_max_side: int = 1600
    sr_blur_threshold: float = 75.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 1 <= self.min_length <= self.max_length <= 32:
            raise ValueError("Invalid plate length limits")
        if self.max_pixels < 64 or not 1 <= self.max_candidates <= 32:
            raise ValueError("Invalid processing limits")
        if not 128 <= self.detection_max_side <= 4096:
            raise ValueError("detection_max_side must be in [128, 4096]")
        if not math.isfinite(self.sr_blur_threshold) or self.sr_blur_threshold < 0:
            raise ValueError("Invalid SR blur threshold")
        if any(not self.min_length <= len(p) <= self.max_length
               or set(p) - set("DLX") for p in self.patterns):
            raise ValueError("Patterns must contain D/L/X and match length limits")


class PaddleRecognizer:
    """One CPU PaddleOCR 3 recognition model, loaded once from local artifacts."""

    def __init__(self, model_dir: str | Path) -> None:
        self._lock = Lock()
        directory = Path(model_dir).resolve(strict=True)
        required = ("inference.yml", "inference.pdiparams")
        if not directory.is_dir() or any(not (directory / f).is_file() for f in required):
            raise ValueError("Expected a local Paddle inference model with YAML and weights")
        if not any((directory / f).is_file() for f in ("inference.json", "inference.pdmodel")):
            raise ValueError("Local Paddle model graph is missing")
        # Disable PaddleX's model-source connectivity probe before importing it.
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"  # Older PaddleX 3 releases.
        try:
            import yaml
            from paddleocr import TextRecognition

            with (directory / "inference.yml").open(encoding="utf-8") as stream:
                metadata = yaml.safe_load(stream)
            model_name = metadata["Global"]["model_name"]
            if not isinstance(model_name, str) or not model_name:
                raise ValueError("Missing Global.model_name in inference.yml")
            self._model = TextRecognition(model_name=model_name, model_dir=str(directory), device="cpu")
        except Exception as exc:
            raise ALPRError("Cannot load local PaddleOCR recognition model") from exc

    def recognize(self, image: Image) -> tuple[str, float]:
        with self._lock:
            return self._predict(image)

    def recognize_allowed(self, image: Image, alphabet: str) -> tuple[str, float]:
        """Constrain CTC classes without renormalizing their original probabilities.

        Requires the pinned PaddleX 3.3 CTC predictor. Fail explicitly on another
        decoder/API instead of silently falling back to unconstrained recognition.
        """
        with self._lock:
            predictor = self._model.paddlex_predictor
            original = predictor.post_op
            predictor.post_op = RestrictedCTC(original, alphabet)
            try:
                return self._predict(image)
            finally:
                predictor.post_op = original

    def _predict(self, image: Image) -> tuple[str, float]:
        results = list(self._model.predict(input=image, batch_size=1))
        if len(results) != 1:
            raise ALPRError("Expected one PaddleOCR text-line result")
        result = results[0]
        return str(result["rec_text"]), float(result["rec_score"])


class RestrictedCTC:
    """Mask impossible classes, retaining blank and UNCHANGED class probabilities."""

    def __init__(self, decoder, alphabet: str):
        if type(decoder).__name__ != "CTCLabelDecode" or not alphabet:
            raise ALPRError("Restricted alphabet requires the PaddleX CTC decoder")
        self.decoder = decoder
        characters = decoder.character
        self.keep = np.array([
            i in decoder.get_ignored_tokens() or character in alphabet
            for i, character in enumerate(characters)
        ])

    def __call__(self, pred, **kwargs):
        probabilities = np.array(pred[0], copy=True)
        if probabilities.ndim != 3 or probabilities.shape[-1] != len(self.keep):
            raise ALPRError("Unexpected PaddleX CTC output shape")
        probabilities[..., ~self.keep] = 0.0
        return self.decoder([probabilities], **kwargs)


def order_quad(points: Quad) -> Quad:
    """Order convex unordered corners TL, TR, BR, BL; reject degenerate input."""
    q = np.asarray(points, dtype=np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("Corners must be four finite XY pairs")
    hull = cv2.convexHull(q).reshape(-1, 2)
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 16:
        raise ValueError("Degenerate plate quadrilateral")
    # Rotate the cyclic convex hull, preserving distinct corners at steep angles.
    hull = np.roll(hull, -int(np.argmin(hull.sum(axis=1))), axis=0)
    a, b = hull[1] - hull[0], hull[2] - hull[1]
    if a[0] * b[1] - a[1] * b[0] < 0:
        hull = hull[[0, 3, 2, 1]]
    # Long side is horizontal for the supported single-line plate geometry.
    if np.linalg.norm(hull[1] - hull[0]) < np.linalg.norm(hull[3] - hull[0]):
        hull = hull[[3, 0, 1, 2]]
    return np.ascontiguousarray(hull, dtype=np.float32)


class OpenCVCornerDetector:
    """Bounded contour detector. Replace with a plate-trained detector in busy scenes.

    This is a geometric candidate generator, not a semantic vehicle detector.
    It never presents the corners of an axis-aligned bbox as measured plate corners.
    """

    def __init__(self, config: ALPRConfig) -> None:
        self.config = config

    def detect(self, image: Image) -> list[Quad]:
        h, w = image.shape[:2]
        scale = min(1.0, self.config.detection_max_side / max(h, w))
        small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 40, 40)
        edges = cv2.Canny(gray, 40, 140)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, Quad]] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:200]:
            area = cv2.contourArea(contour)
            if area < 120 or area > gray.size * 0.95:
                continue
            approx = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            q = order_quad(approx.reshape(4, 2))
            width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
            if height < 7 or not 1.8 <= width / height <= 7:
                continue
            # Suppress inner/outer edges of the same plate before limiting candidates.
            if any(np.linalg.norm(q.mean(axis=0) - other.mean(axis=0)) < height * 0.5
                   and abs(cv2.contourArea(other) - area) / area < 0.5
                   for _, other in candidates):
                continue
            candidates.append((area, q))
            if len(candidates) >= self.config.max_candidates:
                break
        xy_scale = np.float32([w / small.shape[1], h / small.shape[0]])
        return [q * xy_scale for _, q in candidates]


class ALPRProcessor:
    """Reusable synchronous processor; one instance per worker, calls serialized.

    ``process_image`` returns the highest-confidence valid read across candidates
    and three preprocessing branches. No detection returns empty fields. A detected
    but unreadable plate returns empty text/zero score with the rectified PNG.
    Invalid input raises ValueError; operational failures raise ALPRError.
    """

    def __init__(
        self,
        ocr_model_dir: str | Path | None = None,
        *,
        config: ALPRConfig | None = None,
        sr_model_path: str | Path | None = None,
        sr_algorithm: str = "fsrcnn",
        sr_scale: int = 2,
        recognizer: Recognizer | None = None,
        detector: CornerDetector | None = None,
    ) -> None:
        self.config = config or ALPRConfig()
        if recognizer is None:
            if ocr_model_dir is None:
                raise ValueError("ocr_model_dir is required; automatic downloads are disabled")
            recognizer = PaddleRecognizer(ocr_model_dir)
        self._recognizer = recognizer
        self._detector = detector if detector is not None else OpenCVCornerDetector(self.config)
        self._lock = Lock()
        self._sr = None
        if sr_model_path is not None:
            path = Path(sr_model_path).resolve(strict=True)
            if not path.is_file():
                raise ValueError("SR model must be a local .pb file")
            if sr_algorithm not in {"fsrcnn", "espcn", "edsr"} or sr_scale not in {2, 3, 4}:
                raise ValueError("Supported SR models: fsrcnn/espcn/edsr, scale 2/3/4")
            try:
                self._sr = cv2.dnn_superres.DnnSuperResImpl_create()
                self._sr.readModel(str(path))
                self._sr.setModel(sr_algorithm, sr_scale)
            except (AttributeError, cv2.error) as exc:
                raise ALPRError("SR requires opencv-contrib and matching local weights") from exc

    @staticmethod
    def _warp(image: Image, quad: Quad) -> Image:
        destination = np.float32([[0, 0], [469, 0], [469, 109], [0, 109]])
        matrix = cv2.getPerspectiveTransform(quad, destination)
        return cv2.warpPerspective(image, matrix, (470, 110), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

    def _variants(self, warp: Image) -> tuple[Image, Image, Image]:
        source = warp
        blur = cv2.Laplacian(cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if self._sr is not None and blur < self.config.sr_blur_threshold:
            source = self._sr.upsample(warp)
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        value = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)[:, :, 2]
        binary = cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 9)
        smooth = cv2.bilateralFilter(source, 5, 45, 45)
        sharpened = cv2.addWeighted(smooth, 1.7, cv2.GaussianBlur(smooth, (0, 0), 1.2), -0.7, 0)
        # Independent branches, recognized serially on a shared inference engine.
        return clahe, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), sharpened

    def _correct(self, raw: str) -> str:
        # Do not silently delete unknown glyphs or transliterate Cyrillic lookalikes.
        if not raw.isascii():
            return ""
        text = "".join(raw.upper().split()).replace("-", "")
        if not self.config.min_length <= len(text) <= self.config.max_length:
            return ""
        if any(c not in ALPHABET for c in text):
            return ""
        if not self.config.patterns:
            return text
        to_digit = dict(zip("OIBSZ", "01852", strict=True))
        to_letter = {v: k for k, v in to_digit.items()}
        alternatives: list[tuple[int, str]] = []
        for pattern in self.config.patterns:
            if len(pattern) != len(text):
                continue
            chars = list(text)
            for i, kind in enumerate(pattern):
                if kind == "D":
                    chars[i] = to_digit.get(chars[i], chars[i])
                elif kind == "L":
                    chars[i] = to_letter.get(chars[i], chars[i])
            if all(kind == "X" or (kind == "D" and c.isdigit())
                   or (kind == "L" and "A" <= c <= "Z") for kind, c in zip(pattern, chars, strict=True)):
                corrected = "".join(chars)
                alternatives.append((sum(a != b for a, b in zip(text, corrected, strict=True)), corrected))
        if not alternatives:
            return ""
        fewest = min(count for count, _ in alternatives)
        best = {value for count, value in alternatives if count == fewest}
        return best.pop() if len(best) == 1 else ""  # Ambiguous formats: abstain.

    def process_image(self, np_array: Image) -> ALPRResult:
        if (not isinstance(np_array, np.ndarray) or np_array.dtype != np.uint8
                or np_array.ndim != 3 or np_array.shape[2] != 3):
            raise ValueError("Expected a uint8 HxWx3 BGR numpy array")
        h, w = np_array.shape[:2]
        if min(h, w) < 8 or h * w > self.config.max_pixels:
            raise ValueError("Image dimensions outside configured limits")
        with self._lock:
            try:
                return self._process(np.ascontiguousarray(np_array))
            except ALPRError:
                raise
            except Exception as exc:
                raise ALPRError("Local ALPR processing failed") from exc

    def _process(self, image: Image) -> ALPRResult:
        best_text, best_score = "", 0.0
        best_warp: Image | None = None
        h, w = image.shape[:2]
        candidates = self._detector.detect(image)
        failures = successes = 0
        last_error: Exception | None = None
        for corners in candidates[:self.config.max_candidates]:
            quad = order_quad(corners)
            if (quad < 0).any() or (quad[:, 0] > w - 1).any() or (quad[:, 1] > h - 1).any():
                raise ALPRError("Detector returned corners outside the image")
            warp = self._warp(image, quad)
            if best_warp is None:
                best_warp = warp
            for variant in self._variants(warp):
                try:
                    raw, score = self._recognizer.recognize(variant)
                    score = float(score)
                    if not isinstance(raw, str) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ALPRError("Invalid OCR result")
                    text = self._correct(raw)
                    successes += 1
                except Exception as exc:
                    failures += 1
                    last_error = exc
                    continue
                if text and score >= self.config.min_confidence and score > best_score:
                    best_text, best_score, best_warp = text, score, warp
        if failures:
            # Do not put recognized plates or potentially sensitive exception text in logs.
            LOGGER.warning("ALPR OCR branch failures: %d", failures)
            if not successes:
                raise ALPRError("All OCR branches failed") from last_error
        encoded = ""
        if best_warp is not None:
            ok, png = cv2.imencode(".png", best_warp)
            if not ok:
                raise ALPRError("Cannot encode rectified plate")
            encoded = base64.b64encode(png.tobytes()).decode("ascii")
        return {"plate_text": best_text, "confidence": best_score, "warp_image_base64": encoded}
