#!/usr/bin/env python
"""Cut reference crops out of labelled frames into the gallery.

    python scripts/seed_reference_from_frames.py --photos ../photos \
        --labels eval/reference_labels.json --source site_frames

This is how a depot turns its own footage into references: point it at frames and a label
file, and it writes one crop per frame under `data/reference/<type>/<source>/`. The
`--source` tag is carried into the index, which is what lets the evaluation exclude
references cut from the frame it is scoring.

Crops written by this script from the *same* 25 frames the system is measured on are
useful for demonstrating the mechanism, not for claiming accuracy on those frames. Keep
archive photos under a different `--source` so the two can be told apart later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def frame_path(photos: Path, stem: str) -> Path | None:
    for candidate in (f"{stem}.jpg", f"{stem} (1).jpg", f"{stem}.jpeg", f"{stem}.png"):
        path = photos / candidate
        if path.is_file():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--labels", default="eval/reference_labels.json")
    parser.add_argument("--dir", default=None, help="gallery directory")
    parser.add_argument("--models", default=None)
    parser.add_argument("--source", default="site_frames", help="source tag for these crops")
    parser.add_argument("--margin-x", type=float, default=0.30, help="horizontal padding per side")
    parser.add_argument("--margin-y", type=float, default=0.10, help="vertical padding per side")
    args = parser.parse_args()

    os.chdir(ROOT)
    import cv2

    from app.services.vision.local.reference_gallery import CROP_POLICY_FILENAME, vehicle_crop
    from app.services.vision.local.vehicle import VehicleDetector

    gallery = Path(args.dir or os.environ.get("VISION_REFERENCE_DIR", "./data/reference"))
    model_dir = Path(args.models or os.environ.get("VISION_MODEL_DIR", "./data/models"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))["frames"]
    photos = Path(args.photos)
    policy = {"margin_x": args.margin_x, "margin_y": args.margin_y}
    if not all(0 <= value <= 1 for value in policy.values()):
        parser.error("crop margins must be between 0 and 1")
    gallery.mkdir(parents=True, exist_ok=True)
    policy_path = gallery / CROP_POLICY_FILENAME
    if not policy_path.exists() and any(gallery.glob(f"*/{args.source}/*.jpg")):
        parser.error("Existing source has no crop policy; seed into a new directory")
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        parser.error("Existing crop policy differs; use a separate gallery directory")

    detector = VehicleDetector(model_dir)
    written = 0
    for stem, entry in labels.items():
        path = frame_path(photos, stem)
        if path is None:
            print(f"  missing frame {stem}")
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        boxes = detector.detect(image, 0.35)
        if not boxes:
            print(f"  no vehicle in frame {stem}")
            continue
        # The machine the label refers to: the confident box that dominates the frame.
        box = max(boxes, key=lambda b: b.confidence * b.area**0.25)
        crop = vehicle_crop(image, box, args.margin_x, args.margin_y)
        if crop.size == 0:
            continue
        target_dir = gallery / entry["vehicle_type"] / args.source
        target_dir.mkdir(parents=True, exist_ok=True)
        name = f"frame{stem}_{entry.get('view', 'unknown')}.jpg"
        cv2.imwrite(str(target_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1
        print(f"  {entry['vehicle_type']:14s} {name}")

    print(f"\nwrote {written} reference crops under {gallery.resolve()}")
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print("next: python scripts/build_reference_gallery.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
