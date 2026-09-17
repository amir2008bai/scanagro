"""Declarative registry of the model weights the local pipeline needs.

Every entry pins an exact artefact, its download URL, its SHA-256 and the licence of
both the code that produced it and the weights themselves. `scripts/fetch_models.py`
reads this registry, so adding a model here is enough to make it reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    source: str
    notes: str = ""


#: Vehicle detector. RT-DETRv2 R18 exported to ONNX by the onnx-community organisation
#: from PekingU/rtdetr_v2_r18vd. COCO-80 classes, NMS-free, dynamic input resolution.
RTDETR_V2_R18 = ModelSpec(
    key="vehicle_detector",
    filename="rtdetr_v2_r18vd.onnx",
    url="https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX/resolve/main/onnx/model.onnx",
    sha256="583a236ac21c95a7fd94f284fc21485e42355bfef82c27011ba78fbc09ee87e2",
    size_bytes=81057510,
    license="Apache-2.0",
    source="https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX",
    notes="Base model PekingU/rtdetr_v2_r18vd (Apache-2.0), COCO-80.",
)

RTDETR_V2_R18_CONFIG = ModelSpec(
    key="vehicle_detector_config",
    filename="rtdetr_v2_r18vd.config.json",
    url="https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX/resolve/main/config.json",
    sha256="",  # small JSON, validated by parsing id2label instead of by digest
    size_bytes=5731,
    license="Apache-2.0",
    source="https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX",
    notes="Supplies the COCO id2label mapping used to name detections.",
)

#: Plate detector. Published by `open-image-models`, which would otherwise fetch it into
#: `~/.cache` on first use — no good in a container with a read-only root and no home
#: directory, and not reproducible either. We pin it here and load it by explicit path.
YOLO_V9_PLATE = ModelSpec(
    key="plate_detector",
    filename="yolo-v9-s-608-license-plates-end2end.onnx",
    url=(
        "https://github.com/ankandrew/open-image-models/releases/download/assets/"
        "yolo-v9-s-608-license-plates-end2end.onnx"
    ),
    sha256="2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a",
    size_bytes=28612350,
    license="MIT",
    source="https://github.com/ankandrew/open-image-models",
    notes="YOLOv9-s, 608x608, end-to-end (NMS baked in). Single class: License Plate.",
)

#: Plate OCR. Same reasoning as above; the YAML carries the alphabet and input geometry,
#: so the two must be fetched and version-matched together.
CCT_S_V2_GLOBAL = ModelSpec(
    key="plate_ocr_cct",
    filename="cct_s_v2_global.onnx",
    url=(
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global.onnx"
    ),
    sha256="384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b",
    size_bytes=5262230,
    license="MIT",
    source="https://github.com/ankandrew/fast-plate-ocr",
    notes="CCT-S global plate OCR. Alphabet is Latin + digits, which covers Kazakh plates.",
)

CCT_S_V2_GLOBAL_CONFIG = ModelSpec(
    key="plate_ocr_cct_config",
    filename="cct_s_v2_global_plate_config.yaml",
    url=(
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/"
        "cct_s_v2_global_plate_config.yaml"
    ),
    sha256="0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6",
    size_bytes=1725,
    license="MIT",
    source="https://github.com/ankandrew/fast-plate-ocr",
    notes="Alphabet, slot count, input size and colour mode for the OCR model above.",
)

#: Image encoder for the reference gallery. DINOv2 is self-supervised, which is exactly
#: what a nearest-neighbour gallery wants: its features were never fitted to a fixed label
#: set, so adding a new machine type needs reference photos, not retraining.
DINOV2_SMALL = ModelSpec(
    key="reference_encoder",
    filename="dinov2_small.onnx",
    url="https://huggingface.co/onnx-community/dinov2-small-ONNX/resolve/main/onnx/model.onnx",
    sha256="6266c3cd72db6953cecdcbfeab9422a9f783d96f1a4e296ba70ffbac43b54a18",
    size_bytes=88513997,
    license="Apache-2.0",
    source="https://huggingface.co/onnx-community/dinov2-small-ONNX",
    notes="ViT-S/14, base model facebook/dinov2-small (Apache-2.0). 384-d CLS + mean patch.",
)

CYRILLIC_OCR = ModelSpec(
    key="badge_ocr_cyrillic",
    filename="cyrillic_PP-OCRv3_rec_mobile.onnx",
    url="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/cyrillic_PP-OCRv3_rec_mobile.onnx",
    sha256="1efb65bdc460af1c0e8733d005b20952b17ca5aac10ddb56c968333791c5eaa3",
    size_bytes=8972413,
    license="Apache-2.0",
    source="https://github.com/RapidAI/RapidOCR",
    notes="PaddleOCR Cyrillic PP-OCRv3; upstream RapidOCR registry pins the same digest.",
)

CYRILLIC_DICT = ModelSpec(
    key="badge_ocr_cyrillic_dictionary",
    filename="cyrillic_dict.txt",
    url="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/dict/cyrillic_dict.txt",
    sha256="369a82c6c8c479784a5d726448b83b1eafb5fef0a4129a5eaa3929625ddcd132",
    size_bytes=410,
    license="Apache-2.0",
    source="https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7",
    notes="Cyrillic character dictionary paired with PP-OCRv3.",
)

DOWNLOADABLE: tuple[ModelSpec, ...] = (
    RTDETR_V2_R18,
    RTDETR_V2_R18_CONFIG,
    YOLO_V9_PLATE,
    CCT_S_V2_GLOBAL,
    CCT_S_V2_GLOBAL_CONFIG,
    DINOV2_SMALL,
    CYRILLIC_OCR,
    CYRILLIC_DICT,
)

#: Weights that ship inside a wheel and so need no fetching or checksum of ours.
VENDORED_NOTES = {
    "text_ocr_ppocr": (
        "rapidocr-onnxruntime (Apache-2.0) — PP-OCRv4 detection/recognition ONNX weights "
        "(PaddleOCR, Apache-2.0) shipped inside the wheel; no network access needed."
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify(spec: ModelSpec, path: Path) -> None:
    """Raise if the artefact on disk is not the one this registry pins."""
    if not path.is_file():
        raise FileNotFoundError(f"Model file missing: {path}. Run scripts/fetch_models.py.")
    if not spec.sha256:
        return
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {spec.sha256}, got {actual}. "
            "Delete the file and re-run scripts/fetch_models.py."
        )
