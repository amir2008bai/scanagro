"""Stage 4 — read the plate.

Two independent open-source engines are run over several renderings of the same crop:

* ``fast-plate-ocr`` (MIT) — a CCT model trained specifically on plates. It is very fast
  (single-digit milliseconds) and returns a probability per emitted character, which is a
  real OCR confidence rather than a detector score.
* ``rapidocr-onnxruntime`` (Apache-2.0) — PP-OCRv4 detection plus recognition. Slower
  (hundreds of milliseconds) and general-purpose, but markedly more accurate on this
  dataset, and it doubles as the reader for badges and body lettering.

Each rendering (raw crop, rotation-only, perspective-rectified, contrast-enhanced) is a
*candidate*. Every candidate is scored by measured OCR confidence combined with how well
the string fits the Kazakh plate grammar, and the best-scoring one wins. Nothing is
repaired: if no candidate clears the threshold the plate is reported unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services.vision.local import geometry, plate_format, runtime


@dataclass(slots=True)
class OcrRead:
    engine: str
    variant: str
    text: str
    #: Mean probability over the emitted characters.
    mean_confidence: float
    #: Lowest probability over the emitted characters — the weakest link in the read.
    min_confidence: float

    @property
    def compact(self) -> str:
        return plate_format.normalise(self.text)


@dataclass(slots=True)
class PlateReading:
    text: str | None  # display form, only when accepted as readable
    raw_text: str | None  # best compact string, kept even when rejected
    readable: bool
    ocr_confidence: float | None
    engine: str | None
    variant: str | None
    plate_format: str | None
    region: str | None
    rectified: bool
    skew: float | None
    reason: str
    candidates: list[OcrRead] = field(default_factory=list)
    disagreement_positions: list[int] = field(default_factory=list)
    #: Geometry computed while reading, handed back so the caller need not recompute it.
    quad: geometry.QuadEstimate | None = None
    rectified_image: np.ndarray | None = None


class PlateReader:
    def __init__(
        self,
        model_dir: Path,
        ocr_model: str = "cct-s-v2-global-model",
        use_ppocr: bool = True,
        device: str = "cpu",
        accept_confidence: float = 0.55,
        accept_confidence_unformatted: float = 0.88,
    ):
        self._model_dir = Path(model_dir)
        self._ocr_model = ocr_model
        self._use_ppocr = use_ppocr
        self._device = device
        self.accept_confidence = float(accept_confidence)
        self.accept_confidence_unformatted = float(accept_confidence_unformatted)

    # ----- engines ---------------------------------------------------------------

    def _load_cct(self):
        # Explicit paths for the same reason as the plate detector: the library's default
        # cache is ~/.cache, which is neither writable in the container nor checksummed.
        from fast_plate_ocr import LicensePlateRecognizer

        from app.services.vision.local.model_registry import (
            CCT_S_V2_GLOBAL,
            CCT_S_V2_GLOBAL_CONFIG,
            verify,
        )

        weights = self._model_dir / CCT_S_V2_GLOBAL.filename
        config = self._model_dir / CCT_S_V2_GLOBAL_CONFIG.filename
        verify(CCT_S_V2_GLOBAL, weights)
        verify(CCT_S_V2_GLOBAL_CONFIG, config)
        return LicensePlateRecognizer(
            onnx_model_path=str(weights),
            plate_config_path=str(config),
            device="cuda" if self._device == "cuda" else "cpu",
        )

    @property
    def _cct(self):
        return runtime.get_or_create(f"plate_ocr_cct::{self._ocr_model}", self._load_cct)

    def _load_ppocr(self):
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()

    @property
    def _ppocr(self):
        return runtime.get_or_create("text_ocr_ppocr", self._load_ppocr)

    # ----- single reads ----------------------------------------------------------

    def _read_cct(self, image_bgr: np.ndarray, variant: str) -> OcrRead | None:
        import cv2

        prediction = self._cct.run(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), return_confidence=True
        )[0]
        text = plate_format.normalise(prediction.plate)
        if not text:
            return None
        probs = prediction.char_probs
        if probs is None:
            return OcrRead("fast_plate_ocr", variant, text, 0.0, 0.0)
        values = [float(p) for p in list(probs)[: len(text)]]
        if not values:
            return None
        return OcrRead("fast_plate_ocr", variant, text, float(np.mean(values)), float(min(values)))

    def _read_ppocr(self, image_bgr: np.ndarray, variant: str) -> OcrRead | None:
        if not self._use_ppocr:
            return None
        image_bgr = geometry.upscale(image_bgr, min_height=56)
        result, _ = self._ppocr(image_bgr, use_det=True, use_cls=False, use_rec=True)
        if not result:
            return None

        # Left-to-right, then top-to-bottom: two-row plates must read row by row.
        def key(entry):
            box = entry[0]
            ys = [point[1] for point in box]
            xs = [point[0] for point in box]
            return (round(sum(ys) / len(ys) / 24), sum(xs) / len(xs))

        ordered = sorted(result, key=key)
        text = plate_format.normalise("".join(entry[1] for entry in ordered))
        if not text:
            return None
        scores = [float(entry[2]) for entry in ordered]
        return OcrRead("ppocr", variant, text, float(np.mean(scores)), float(min(scores)))

    # ----- candidate construction ------------------------------------------------

    @staticmethod
    def build_variants(
        crop_bgr: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], geometry.QuadEstimate | None]:
        """Renderings to try, plus the quad estimate they were derived from."""
        variants: dict[str, np.ndarray] = {"raw": geometry.upscale(crop_bgr, 48)}
        rotated = geometry.rotate_only(crop_bgr)
        if rotated is not None:
            variants["rotated"] = geometry.upscale(rotated, 48)
        estimate = geometry.estimate_quad(crop_bgr)
        if estimate is not None:
            rectified = geometry.rectify(crop_bgr, estimate.quad)
            if rectified is not None:
                variants["rectified"] = rectified
                variants["rectified_clahe"] = geometry.enhance_contrast(rectified)
        return variants, estimate

    def _score(self, read: OcrRead) -> float:
        """Rank candidates by measured confidence weighted by structural plausibility."""
        structure = plate_format.plausibility(read.text)
        # min_confidence guards against a single hallucinated character carried by an
        # otherwise confident read; the structural term breaks ties between engines.
        confidence = 0.65 * read.min_confidence + 0.35 * read.mean_confidence
        return confidence * (0.45 + 0.55 * structure)

    def read(self, crop_bgr: np.ndarray) -> PlateReading:
        """Read one plate crop and decide whether the result may be asserted."""
        if crop_bgr is None or crop_bgr.size == 0:
            return PlateReading(
                None, None, False, None, None, None, None, None, False, None, "empty_crop"
            )
        variants, estimate = self.build_variants(crop_bgr)
        candidates: list[OcrRead] = []
        completed_reads = 0
        failures: list[Exception] = []

        # The CCT model costs single-digit milliseconds, so every rendering gets it.
        for name, image in variants.items():
            try:
                read = self._read_cct(image, name)
                completed_reads += 1
            except Exception as exc:  # preserve the other engine's fallback
                failures.append(exc)
                read = None
            if read is not None:
                candidates.append(read)

        # PP-OCR costs ~0.5-1 s per call, so it only sees the renderings worth the money:
        # the untouched crop and the best geometric correction available.
        if self._use_ppocr:
            wanted = ["raw"]
            for name in ("rectified", "rotated"):
                if name in variants:
                    wanted.append(name)
                    break
            if "rectified" in variants and self._low_contrast(variants["rectified"]):
                wanted.append("rectified_clahe")
            for name in wanted:
                try:
                    read = self._read_ppocr(variants[name], name)
                    completed_reads += 1
                except Exception as exc:
                    failures.append(exc)
                    read = None
                if read is not None:
                    candidates.append(read)

        if not completed_reads and failures:
            raise RuntimeError("All configured plate OCR engines failed") from failures[-1]

        if not candidates:
            return PlateReading(
                None,
                None,
                False,
                None,
                None,
                None,
                None,
                None,
                estimate is not None,
                estimate.skew if estimate else None,
                "no_text_recognised",
                [],
                quad=estimate,
                rectified_image=variants.get("rectified"),
            )

        # Only the top-ranked candidate is validated, deliberately. Scanning further down
        # for one that happens to match the grammar would let a weak reading win purely
        # because it has the expected shape — which is how a system starts inventing
        # plates. The structural term already inside `_score` is the principled way to
        # favour well-formed reads, and it applies to every candidate equally.
        ranked = sorted(candidates, key=self._score, reverse=True)
        best = ranked[0]
        score = self._score(best)
        # Gate the grammar on the plate's *own* proportions, but only when the glyph-row
        # fit measured them. The detector's axis-aligned box is not a usable proxy — a
        # plate seen at 40 degrees has a box near 1.5:1 though the plate is 4.7:1 — and
        # neither is the bright-field fallback. Without a trustworthy measurement we pass
        # None and let the confidence thresholds do the filtering on their own.
        shape_aspect = estimate.aspect if estimate is not None and estimate.trusted_aspect else None
        fmt = plate_format.classify(best.text, shape_aspect)

        # Cross-engine comparison on the winning string, recorded for audit.
        other_engine = next(
            (c for c in ranked if c.engine != best.engine and len(c.compact) == len(best.compact)),
            None,
        )
        disagree = plate_format.disagreements(best.text, other_engine.text) if other_engine else []

        # Acceptance requires a *structurally possible* plate, not just a confident model.
        # A read outside the currently supported layouts is reported as `text_raw`.
        # This grammar is not an exhaustive list of legal Kazakh plates: two-letter
        # corporate formats exist and need separately labelled validation data.
        # Unsupported text is retained with the reason, and
        # left unreadable. That keeps a plausible-looking but unconfirmable string out of
        # `license_plate`, where a downstream consumer would treat it as fact.
        if fmt is None:
            readable = False
            reason = "no_matching_plate_format"
        else:
            threshold = (
                self.accept_confidence_unformatted if fmt.provisional else self.accept_confidence
            )
            readable = score >= threshold
            reason = "accepted" if readable else "below_confidence_threshold"

        return PlateReading(
            text=plate_format.format_for_display(best.text) if readable else None,
            raw_text=best.compact or None,
            readable=readable,
            ocr_confidence=round(float(score), 4),
            engine=best.engine,
            variant=best.variant,
            plate_format=fmt.name if fmt else None,
            region=plate_format.region_name(best.text) if fmt else None,
            rectified=best.variant.startswith("rectified"),
            skew=round(estimate.skew, 4) if estimate else None,
            reason=reason,
            candidates=ranked[:8],
            disagreement_positions=disagree,
            quad=estimate,
            rectified_image=variants.get("rectified"),
        )

    @staticmethod
    def _low_contrast(image: np.ndarray, threshold: float = 42.0) -> bool:
        """Only spend a CLAHE pass on crops that are actually flat."""
        import cv2

        return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std()) < threshold
