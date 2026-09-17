"""Ablation only: compare tight and contextual retrieval, without training or label tuning."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.vision.local.reference_gallery import ReferenceGallery
from app.services.vision.local.vehicle import VehicleDetector
from eval.evaluate_reference import frame_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--baseline-gallery", required=True, help="Index built from original Opus tight crops"
)
args = parser.parse_args()
gallery = ReferenceGallery(Path(args.baseline_gallery), Path("data/models"), threads=2)
if not gallery.available or gallery.index["manifest"].get("crop_policy"):
    parser.error("A baseline tight-crop index is required, not the upgraded contextual index")
detector = VehicleDetector(Path("data/models"), threads=2)
raw = json.loads(Path("eval/reference_labels.json").read_text())
labels = raw["frames"]
index = gallery.index
original = index["vectors"].copy()
queries = {}
context_refs = original.copy()
for stem, entry in labels.items():
    image = cv2.imread(str(frame_path(Path("../photos"), stem)))
    box = max(detector.detect(image, 0.35), key=lambda b: b.confidence * b.area**0.25)
    h, w = image.shape[:2]
    dx, dy = int(0.3 * (box.x2 - box.x1)), int(0.1 * (box.y2 - box.y1))
    tight = image[box.y1 : box.y2, box.x1 : box.x2]
    context = image[
        max(0, box.y1 - dy) : min(h, box.y2 + dy), max(0, box.x1 - dx) : min(w, box.x2 + dx)
    ]
    queries[stem] = (gallery.encoder.embed(tight), gallery.encoder.embed(context))
    name = f"frame{stem}_{entry['view']}.jpg"
    context_refs[index["names"].index(name)] = queries[stem][1]
    print("embedded", stem, flush=True)

rows = []
for mode in ("tight", "context_query", "context_both", "dual"):
    index["vectors"] = context_refs if mode == "context_both" else original
    if mode == "dual":
        matrix = np.concatenate([original, context_refs], axis=1)
        index["vectors"] = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    correct = wrong = abstained = 0
    for stem, entry in labels.items():
        excluded = {stem}
        for group in raw["same_machine_groups"]:
            if stem in group:
                excluded.update(group)
        excluded = {f"frame{s}_{labels[s]['view']}.jpg" for s in excluded}
        tight, context = queries[stem]
        query = tight if mode == "tight" else context
        if mode == "dual":
            query = np.concatenate([tight, context])
            query /= np.linalg.norm(query)
        gallery.encoder.embed = lambda image, q=query: q
        match = gallery.classify(np.zeros((1, 1, 3), np.uint8), exclude_sources=excluded)
        predicted = match.vehicle_type if match.reason == "accepted" else None
        correct += predicted == entry["vehicle_type"]
        wrong += predicted is not None and predicted != entry["vehicle_type"]
        abstained += predicted is None
        rows.append(
            dict(
                mode=mode,
                frame=stem,
                expected=entry["vehicle_type"],
                predicted=predicted,
                reason=match.reason,
                similarity=match.similarity,
                margin=match.margin,
            )
        )
    print(mode, dict(correct=correct, wrong=wrong, abstained=abstained), flush=True)
Path("eval/context_ablation.json").write_text(json.dumps(rows, indent=2))
