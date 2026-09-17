"""Stage 1b — name the machine type by comparing it to reference photos.

COCO, which the vehicle detector is trained on, has no class for agricultural machinery.
A Kirovets and an MTZ both come back as `truck`, and a grain trailer often as `train`.
Retraining a detector needs a labelled dataset nobody has for this yard; what a depot
*does* have is photographs of its own machines.

So this stage keeps the detector for *where* the machine is and answers *what it is* by
retrieval: every reference image is embedded once with DINOv2, and a detected crop is
labelled by its nearest references. DINOv2 is self-supervised, so its features were never
fitted to a fixed label set — adding a new machine type means dropping photos into a
folder and rebuilding the index, not retraining a model.

Two properties matter more than raw accuracy here:

* **It abstains.** Below the similarity floor, or when the top two classes are too close
  to call, it returns nothing and the detector's own coarse label stands.
* **It shows its work.** Every answer carries the references it matched, their similarity
  and where those references came from, so a wrong answer can be traced to the photo that
  caused it and that photo can be removed.

Measured limit, on this dataset: retrieval separates *types* (tractor, truck, light truck,
trailer) cleanly, but **not brands** — a MAZ cab and a KAMAZ cab sit 0.91 apart from each
other and 0.91 from their own kind. Brand stays the badge-OCR stage's job. See
docs/REFERENCE_GALLERY.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services.vision.local import runtime
from app.services.vision.local.model_registry import DINOV2_SMALL, verify

#: ImageNet statistics, as DINOv2's own preprocessor config specifies.
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

#: Index filename inside the gallery directory.
INDEX_FILENAME = "index.npz"
MANIFEST_FILENAME = "manifest.json"
CROP_POLICY_FILENAME = "crop_policy.json"


def vehicle_crop(image: np.ndarray, box, margin_x: float = 0.0, margin_y: float = 0.0):
    """Include context consistently for references and queries; clip at frame edges."""
    if not (0 <= margin_x <= 1 and 0 <= margin_y <= 1):
        raise ValueError("Reference crop margins must be between 0 and 1")
    height, width = image.shape[:2]
    dx, dy = int(margin_x * (box.x2 - box.x1)), int(margin_y * (box.y2 - box.y1))
    return image[
        max(0, box.y1 - dy) : min(height, box.y2 + dy),
        max(0, box.x1 - dx) : min(width, box.x2 + dx),
    ]


@dataclass(slots=True)
class ReferenceMatch:
    vehicle_type: str
    similarity: float
    #: Gap to the best-scoring *other* class. A small margin means "could be either".
    margin: float
    #: The individual references that voted, best first.
    neighbours: list[dict] = field(default_factory=list)
    #: Why a match was refused, when `vehicle_type` is None-ish.
    reason: str = "accepted"


class ReferenceEncoder:
    """DINOv2 image embeddings, L2-normalised. One session per process."""

    def __init__(self, model_dir: Path, device: str = "cpu", threads: int = 0, size: int = 224):
        self._model_dir = Path(model_dir)
        self._device = device
        self._threads = threads
        # DINOv2 is patch-14, so the input side must be a multiple of 14.
        self.size = int(size) // 14 * 14

    def _load(self):
        import onnxruntime as ort

        weights = self._model_dir / DINOV2_SMALL.filename
        verify(DINOV2_SMALL, weights)
        return ort.InferenceSession(
            str(weights),
            sess_options=runtime.session_options(self._threads),
            providers=runtime._providers(self._device),
        )

    @property
    def _session(self):
        return runtime.get_or_create(f"reference_encoder::{self._device}", self._load)

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        tensor = ((rgb.astype(np.float32) / 255.0) - _MEAN) / _STD
        hidden = self._session.run(None, {"pixel_values": tensor.transpose(2, 0, 1)[None]})[0][0]
        # CLS token plus the mean of the patch tokens: the pairing DINOv2's own retrieval
        # evaluations use, and measurably better here than CLS alone.
        vector = np.concatenate([hidden[0], hidden[1:].mean(axis=0)])
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else vector


class ReferenceGallery:
    """Nearest-neighbour type classifier over a directory of reference photos."""

    def __init__(
        self,
        gallery_dir: Path,
        model_dir: Path,
        device: str = "cpu",
        threads: int = 0,
        min_similarity: float = 0.62,
        min_margin: float = 0.04,
        top_k: int = 5,
    ):
        self.gallery_dir = Path(gallery_dir)
        self.encoder = ReferenceEncoder(model_dir, device, threads)
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)
        self.top_k = int(top_k)

    # ----- index ------------------------------------------------------------------

    def _load_index(self):
        index_path = self.gallery_dir / INDEX_FILENAME
        if not index_path.is_file():
            return None
        data = np.load(index_path, allow_pickle=False)
        manifest_path = self.gallery_dir / MANIFEST_FILENAME
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {"references": []}
        )
        return {
            "vectors": data["vectors"].astype(np.float32),
            "labels": [str(x) for x in data["labels"]],
            "names": [str(x) for x in data["names"]],
            "sources": [str(x) for x in data["sources"]],
            "manifest": manifest,
        }

    @property
    def index(self):
        return runtime.get_or_create(f"reference_index::{self.gallery_dir}", self._load_index)

    @property
    def available(self) -> bool:
        return self.index is not None and len(self.index["labels"]) > 0

    def vehicle_crop(self, image: np.ndarray, box) -> np.ndarray:
        # Old indexes retain their original tight-query behaviour until rebuilt.
        policy = self.index["manifest"].get("crop_policy", {}) if self.index else {}
        return vehicle_crop(image, box, policy.get("margin_x", 0.0), policy.get("margin_y", 0.0))

    def describe(self) -> dict:
        """Summary for logs and for the docs: what the gallery actually contains."""
        index = self.index
        if index is None:
            return {"available": False, "references": 0, "classes": {}}
        classes: dict[str, int] = {}
        sources: dict[str, int] = {}
        for label, source in zip(index["labels"], index["sources"], strict=True):
            classes[label] = classes.get(label, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        return {
            "available": True,
            "references": len(index["labels"]),
            "classes": dict(sorted(classes.items())),
            "sources": dict(sorted(sources.items())),
        }

    # ----- classification ---------------------------------------------------------

    def classify(
        self,
        crop_bgr: np.ndarray,
        exclude_sources: set[str] | None = None,
        exclude_source_dirs: set[str] | None = None,
    ) -> ReferenceMatch | None:
        """Label a vehicle crop, or return None when the gallery declines to answer.

        `exclude_sources` drops individual references by filename: the evaluation uses it
        to remove the crop taken from the very frame being scored, without which the index
        would be measured against itself. `exclude_source_dirs` drops a whole source at
        once, which is how the contribution of web photos versus site footage is measured.
        Neither is used in normal operation.
        """
        index = self.index
        if index is None or not len(index["labels"]):
            return None
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        vectors, labels = index["vectors"], index["labels"]
        names, sources = index["names"], index["sources"]
        keep = np.ones(len(labels), dtype=bool)
        if exclude_sources:
            for position, name in enumerate(names):
                if name in exclude_sources:
                    keep[position] = False
        if exclude_source_dirs:
            for position, source in enumerate(sources):
                if source in exclude_source_dirs:
                    keep[position] = False
        if not keep.any():
            return None

        query = self.encoder.embed(crop_bgr)
        similarity = vectors[keep] @ query
        kept_labels = [label for label, ok in zip(labels, keep, strict=True) if ok]
        kept_names = [name for name, ok in zip(names, keep, strict=True) if ok]
        kept_sources = [source for source, ok in zip(sources, keep, strict=True) if ok]

        order = np.argsort(-similarity)[: self.top_k]
        neighbours = [
            {
                "reference": kept_names[i],
                "vehicle_type": kept_labels[i],
                "source": kept_sources[i],
                "similarity": round(float(similarity[i]), 4),
            }
            for i in order
        ]

        # Similarity-weighted vote over the top-k, so one very close reference outweighs
        # several mediocre ones but a single outlier cannot carry the decision alone.
        votes: dict[str, float] = {}
        for i in order:
            weight = max(0.0, float(similarity[i]))
            votes[kept_labels[i]] = votes.get(kept_labels[i], 0.0) + weight**3
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        best_label = ranked[0][0]
        best_similarity = max(float(similarity[i]) for i in order if kept_labels[i] == best_label)
        # The gap is to the closest reference of *any* other class, not merely to the
        # runner-up by vote. A third-ranked class with one very close reference is still a
        # reason to doubt the answer, and that is exactly the rear-view truck/trailer case.
        others = [float(similarity[i]) for i in order if kept_labels[i] != best_label]
        margin = best_similarity - (max(others) if others else 0.0)

        if best_similarity < self.min_similarity:
            return ReferenceMatch(
                best_label,
                round(best_similarity, 4),
                round(margin, 4),
                neighbours,
                "below_similarity_floor",
            )
        if others and margin < self.min_margin:
            return ReferenceMatch(
                best_label,
                round(best_similarity, 4),
                round(margin, 4),
                neighbours,
                "classes_too_close",
            )
        return ReferenceMatch(
            best_label, round(best_similarity, 4), round(margin, 4), neighbours, "accepted"
        )
