"""Stage 5 — manufacturer, model and a refined machine type.

The assignment asks for the manufacturer and model *only when the image confirms them*.
There is no open dataset of KAMAZ/MAZ/MTZ badges to classify against, and a zero-shot
guess from silhouette alone would be exactly the kind of confident invention the brief
rules out. So this stage reads the lettering that manufacturers put on their own grilles,
cabs and tailgates, and reports a brand only when the text is actually there.

Everything it returns carries its evidence: the string that was read, its OCR confidence
and where on the vehicle it was found. When nothing is legible the answer is ``None``.

Latin recognition is supplemented with pinned Cyrillic PP-OCRv3 weights. Raw body
text is evidence, not proof that every word names the vehicle's manufacturer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services.vision.local import runtime
from app.services.vision.local.model_registry import CYRILLIC_DICT, CYRILLIC_OCR, verify


@dataclass(frozen=True, slots=True)
class BrandRule:
    manufacturer: str
    #: Latin tokens that appear on the machine itself.
    tokens: tuple[str, ...]
    #: Machine type implied by the brand, when the brand only makes one kind.
    implies_type: str | None = None
    #: Brands this one's tokens are a substring of. A KAMAZ grille whose first letters
    #: fall outside the crop reads as "MAZ", so a bare "MAZ" cannot be asserted: OCR
    #: gives us no word boundary. When only the ambiguous form is present we report no
    #: manufacturer and keep the raw token as evidence instead of guessing.
    ambiguous_with: tuple[str, ...] = ()


BRANDS: tuple[BrandRule, ...] = (
    BrandRule("KAMAZ", ("KAMAZ", "KAMA3", "KAMAS", "КАМАЗ"), "truck"),
    BrandRule("MAZ", ("MAZ", "MA3", "МАЗ"), "truck", ambiguous_with=("KAMAZ",)),
    BrandRule("MAN", ("MAN",), "truck"),
    BrandRule("GAZ", ("GAZ", "GA3", "ГАЗ"), "truck"),
    BrandRule("ZIL", ("ZIL", "3IL"), "truck"),
    BrandRule("KRAZ", ("KRAZ", "KRA3", "КРАЗ"), "truck"),
    BrandRule("URAL", ("URAL", "УРАЛ"), "truck"),
    BrandRule("VOLVO", ("VOLVO",), "truck"),
    BrandRule("SCANIA", ("SCANIA",), "truck"),
    BrandRule("DAF", ("DAF",), "truck"),
    BrandRule("IVECO", ("IVECO",), "truck"),
    BrandRule("MERCEDES-BENZ", ("MERCEDES", "BENZ"), None),
    BrandRule("STEYR", ("STEYR",), "truck"),
    BrandRule("HOWO", ("HOWO", "SINOTRUK"), "truck"),
    BrandRule("SHACMAN", ("SHACMAN", "SHAANXI"), "truck"),
    BrandRule("FAW", ("FAW",), "truck"),
    BrandRule("BELARUS", ("BELARUS", "MTZ", "БЕЛАРУС", "МТЗ"), "tractor"),
    BrandRule("KIROVETS", ("KIROVETS", "KIROVEC", "КИРОВЕЦ"), "tractor"),
    BrandRule("JOHN DEERE", ("DEERE", "JOHNDEERE"), "tractor"),
    BrandRule("CLAAS", ("CLAAS",), "tractor"),
    BrandRule("NEW HOLLAND", ("HOLLAND",), "tractor"),
    BrandRule("CASE", ("CASEIH",), "tractor"),
)

#: Model designations that are unambiguous when they appear next to a known brand.
MODEL_HINTS: dict[str, tuple[str, ...]] = {
    "KIROVETS": ("K700", "K701", "K744"),
    "BELARUS": ("MTZ80", "MTZ82", "MTZ1221", "820", "892", "1221"),
    "KAMAZ": ("5320", "55111", "53212", "54115", "65115", "45143"),
    "MAZ": ("5516", "5551", "6501", "555102"),
}

#: Tokens that must never be read as a brand: CCTV overlay text and common yard signage.
NOISE_TOKENS = {
    "CAMERA",
    "CAMERAO1",
    "WED",
    "MON",
    "TUE",
    "THU",
    "FRI",
    "SAT",
    "SUN",
    "СТАЙЕР",
    "СТАИЕР",
    "STAYER",  # decorative visor text, not the STEYR brand
}


@dataclass(slots=True)
class VehicleAttributes:
    manufacturer: str | None = None
    model: str | None = None
    vehicle_type: str | None = None
    manufacturer_confidence: float | None = None
    evidence: list[dict] = field(default_factory=list)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _matches(token: str, target: str) -> bool:
    """Exact match, or one edit for tokens long enough that one edit is not a coin flip."""
    if token == target:
        return True
    if len(target) >= 5 and abs(len(token) - len(target)) <= 1:
        return _levenshtein(token, target) <= 1
    return False


class AttributeReader:
    """Reads badge and body lettering with PP-OCR and maps it onto known brands."""

    def __init__(
        self,
        model_dir: Path,
        enabled: bool = True,
        max_side: int = 1024,
        min_confidence: float = 0.55,
        cyrillic: bool = True,
    ):
        self._model_dir = Path(model_dir)
        self.enabled = enabled
        self.max_side = int(max_side)
        self.min_confidence = float(min_confidence)
        self.cyrillic = cyrillic

    @property
    def _cyrillic_ocr(self):
        def load():
            from rapidocr_onnxruntime import RapidOCR

            for spec in (CYRILLIC_OCR, CYRILLIC_DICT):
                verify(spec, self._model_dir / spec.filename)
            return RapidOCR(
                rec_model_path=str(self._model_dir / CYRILLIC_OCR.filename),
                rec_keys_path=str(self._model_dir / CYRILLIC_DICT.filename),
                intra_op_num_threads=2,
                inter_op_num_threads=1,
            )

        return runtime.get_or_create(f"badge_ocr_cyrillic::{self._model_dir.resolve()}", load)

    @property
    def _ppocr(self):
        def load():
            from rapidocr_onnxruntime import RapidOCR

            return RapidOCR()

        return runtime.get_or_create("text_ocr_ppocr", load)

    def _body_text(self, crop_bgr: np.ndarray) -> list[tuple[str, float, tuple]]:
        import cv2

        height, width = crop_bgr.shape[:2]
        longest = max(height, width)
        image = crop_bgr
        if longest > self.max_side:
            scale = self.max_side / longest
            image = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result, _ = self._ppocr(image, use_det=True, use_cls=False, use_rec=True)
        result = list(result or [])
        if self.cyrillic:
            extra, _ = self._cyrillic_ocr(image, use_det=True, use_cls=False, use_rec=True)
            # Keep the proven Latin engine's decisions. The extra engine contributes
            # only text actually containing Cyrillic characters.
            result.extend(
                r for r in (extra or []) if any("А" <= c <= "Я" or c == "Ё" for c in r[1].upper())
            )
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
        out = []
        for box, text, score in result:
            token = "".join(ch for ch in str(text).upper() if ch in allowed)
            if len(token) >= 2 and float(score) >= self.min_confidence:
                out.append((token, float(score), tuple(map(tuple, np.asarray(box).tolist()))))
        return out

    def read(self, crop_bgr: np.ndarray, coco_label: str) -> VehicleAttributes:
        attributes = VehicleAttributes()
        if not self.enabled or crop_bgr is None or crop_bgr.size == 0:
            return attributes
        # A missing/corrupt configured model is an operational failure, not an
        # image on which there happened to be no readable badge.
        tokens = self._body_text(crop_bgr)

        # Rank matches by specificity first: "MAZ" is a substring of "KAMAZ", so a frame
        # containing both must resolve to KAMAZ regardless of which token scored higher.
        best: tuple[int, float, BrandRule, str] | None = None
        for token, score, _box in tokens:
            if token in NOISE_TOKENS:
                continue
            for rule in BRANDS:
                for candidate in rule.tokens:
                    if not _matches(token, candidate):
                        continue
                    ranked = (len(candidate), score, rule, token)
                    if best is None or ranked[:2] > best[:2]:
                        best = ranked
        if best is not None:
            matched_length, score, rule, token = best
            if rule.ambiguous_with and matched_length <= 4:
                # Only the ambiguous short form was read. Unless a longer brand token also
                # appeared (handled above by the specificity ordering), we cannot tell
                # "MAZ" from the tail of "KAMAZ", so we assert nothing.
                attributes.evidence = [
                    {
                        "source": "body_text_ocr",
                        "text": token,
                        "confidence": round(score, 4),
                        "rejected": "ambiguous_brand_token",
                        "could_be": [rule.manufacturer, *rule.ambiguous_with],
                    }
                ]
                return attributes
            best = (rule, token, score)
        if best is None:
            attributes.evidence = [{"text": t, "confidence": round(s, 3)} for t, s, _ in tokens[:8]]
            return attributes

        rule, token, score = best
        attributes.manufacturer = rule.manufacturer
        attributes.manufacturer_confidence = round(score, 4)
        attributes.evidence = [
            {
                "source": "body_text_ocr",
                "text": token,
                "confidence": round(score, 4),
                "matched_brand": rule.manufacturer,
            }
        ]
        if rule.implies_type:
            attributes.vehicle_type = rule.implies_type

        # A model is only reported when its designation is itself legible on the machine.
        for hint in MODEL_HINTS.get(rule.manufacturer, ()):  # noqa: SIM118
            for token2, score2, _box in tokens:
                if token2 == hint:
                    attributes.model = f"{rule.manufacturer} {hint}"
                    attributes.evidence.append(
                        {
                            "source": "body_text_ocr",
                            "text": token2,
                            "confidence": round(score2, 4),
                            "matched_model": hint,
                        }
                    )
                    break
            if attributes.model:
                break
        return attributes
