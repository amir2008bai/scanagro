#!/usr/bin/env python
"""Score the reference gallery against `eval/reference_labels.json`.

    python eval/evaluate_reference.py --photos ../photos --out eval/reference_results

Two figures are reported, and they are not the same thing:

* **leave-one-frame-out** — when scoring frame N, the reference cut from frame N is
  removed from the index. This is the minimum honesty bar; without it the index is being
  measured against itself.
* **leave-one-machine-out** — additionally removes references cut from *other* frames
  showing the same physical machine, listed in `same_machine_groups`. Two photos of the
  same lorry taken minutes apart are nearly the same image, so the first figure is
  optimistic and this one is the number to quote.

Both still measure a gallery seeded from the same 25 frames the system is tested on. They
say the retrieval mechanism works on this footage. They do not say what accuracy to expect
on a machine the gallery has never seen — only a depot archive can answer that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
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
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="drop a whole source directory, e.g. --exclude-source web, to measure what it adds",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    import cv2

    from app.services.vision.local.reference_gallery import ReferenceGallery
    from app.services.vision.local.vehicle import VehicleDetector

    gallery_dir = Path(args.dir or os.environ.get("VISION_REFERENCE_DIR", "./data/reference"))
    model_dir = Path(args.models or os.environ.get("VISION_MODEL_DIR", "./data/models"))
    raw = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    labels, groups = raw["frames"], raw.get("same_machine_groups", [])

    #: frame -> every frame showing the same physical machine, itself included
    machine_of: dict[str, set[str]] = {stem: {stem} for stem in labels}
    for group in groups:
        for stem in group:
            if stem in machine_of:
                machine_of[stem] |= set(group)

    gallery = ReferenceGallery(gallery_dir, model_dir, device=args.device)
    if not gallery.available:
        print(f"No index at {gallery_dir.resolve()}. Run scripts/build_reference_gallery.py.")
        return 1
    print("gallery:", json.dumps(gallery.describe(), ensure_ascii=False))

    detector = VehicleDetector(model_dir)
    photos = Path(args.photos)
    rows = []
    for stem, entry in labels.items():
        path = frame_path(photos, stem)
        if path is None:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        boxes = detector.detect(image, 0.35)
        if not boxes:
            print(f"{stem:>3s} no vehicle box")
            continue
        box = max(boxes, key=lambda b: b.confidence * b.area**0.25)
        crop = gallery.vehicle_crop(image, box)

        def references_from(frames: set[str]) -> set[str]:
            return {f"frame{f}_{labels[f].get('view', 'unknown')}.jpg" for f in frames}

        dropped_dirs = set(args.exclude_source)
        frame_out = gallery.classify(
            crop,
            exclude_sources=references_from({stem}),
            exclude_source_dirs=dropped_dirs,
        )
        machine_out = gallery.classify(
            crop,
            exclude_sources=references_from(machine_of[stem]),
            exclude_source_dirs=dropped_dirs,
        )

        def verdict(match):
            if match is None or match.reason != "accepted":
                return None, (match.reason if match else "no_match")
            return match.vehicle_type, match.reason

        frame_type, frame_reason = verdict(frame_out)
        machine_type, machine_reason = verdict(machine_out)
        expected = entry["vehicle_type"]
        rows.append(
            {
                "frame": stem,
                "expected": expected,
                "coco_label": box.coco_label,
                "detector_type": box.vehicle_type,
                "leave_frame_out": frame_type,
                "leave_frame_out_reason": frame_reason,
                "leave_frame_out_similarity": frame_out.similarity if frame_out else None,
                "leave_machine_out": machine_type,
                "leave_machine_out_reason": machine_reason,
                "leave_machine_out_similarity": machine_out.similarity if machine_out else None,
                "neighbours": (machine_out.neighbours[:3] if machine_out else []),
                "view": entry.get("view"),
            }
        )
        mark = "OK " if machine_type == expected else ("-- " if machine_type is None else "XX ")
        print(
            f"{stem:>3s} {mark} want={expected:12s} "
            f"frame_out={str(frame_type):12s} machine_out={str(machine_type):12s} "
            f"sim={machine_out.similarity if machine_out else 0:.2f}  "
            f"(coco said '{box.coco_label}')",
            flush=True,
        )

    def tally(key: str) -> dict:
        correct = sum(1 for r in rows if r[key] == r["expected"])
        wrong = sum(1 for r in rows if r[key] is not None and r[key] != r["expected"])
        abstained = sum(1 for r in rows if r[key] is None)
        return {
            "correct": f"{correct}/{len(rows)}",
            "wrong": wrong,
            "abstained": abstained,
            "accuracy_when_answering": (
                round(correct / (correct + wrong), 3) if correct + wrong else None
            ),
        }

    coco_correct = sum(1 for r in rows if r["detector_type"] == r["expected"])
    summary = {
        "frames_scored": len(rows),
        "excluded_sources": sorted(args.exclude_source),
        "classes": dict(sorted(Counter(r["expected"] for r in rows).items())),
        "detector_alone_correct": f"{coco_correct}/{len(rows)}",
        "leave_one_frame_out": tally("leave_frame_out"),
        "leave_one_machine_out": tally("leave_machine_out"),
    }
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "per_frame.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {out / 'per_frame.json'} and {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
