#!/usr/bin/env python
"""Embed the reference photos into a searchable index.

Layout the script expects — one directory per machine type, one subdirectory per source:

    data/reference/
      tractor/
        site_archive/kirovets_2026-09-16_a.jpg
        web/john_deere_8r.jpg
      truck/
        site_archive/kamaz_dump_01.jpg
      trailer/
        ...
      SOURCES.md            provenance and licence for anything not filmed on site

Run after adding or removing photos:

    python scripts/build_reference_gallery.py
    python scripts/build_reference_gallery.py --dir data/reference --report

The index records, for every reference, its class, its filename and its source directory.
The source is what lets the evaluation exclude references cut from the frame it is
scoring, and what lets you see whether web photos are helping or hurting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.vision.local.reference_gallery import (  # noqa: E402
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    ReferenceEncoder,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
#: Reserved directory names that are not machine classes.
NON_CLASS_DIRS = {"__pycache__"}


def collect(gallery_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (class, source, path) for every reference image, sorted for reproducibility."""
    found: list[tuple[str, str, Path]] = []
    for class_dir in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
        if class_dir.name in NON_CLASS_DIRS:
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(class_dir)
            source = relative.parts[0] if len(relative.parts) > 1 else "unsorted"
            found.append((class_dir.name, source, path))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default=None, help="gallery directory (default: VISION_REFERENCE_DIR)"
    )
    parser.add_argument(
        "--models", default=None, help="model directory (default: VISION_MODEL_DIR)"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--report", action="store_true", help="print the contents and exit")
    parser.add_argument(
        "--seed-from",
        default=None,
        help="copy this directory into an EMPTY gallery before indexing; never overwrites",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    import cv2

    gallery_dir = Path(args.dir or os.environ.get("VISION_REFERENCE_DIR", "./data/reference"))
    model_dir = Path(args.models or os.environ.get("VISION_MODEL_DIR", "./data/models"))
    gallery_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_from:
        seed = Path(args.seed_from)
        if not seed.is_dir():
            print(f"seed directory {seed.resolve()} does not exist; nothing to copy")
        elif collect(gallery_dir):
            # Someone's own photos are already here. Leaving them untouched matters more
            # than shipping our examples, so this is a no-op rather than a merge.
            print(f"gallery already populated; leaving {gallery_dir.resolve()} as it is")
        else:
            copied = 0
            for source_path in sorted(seed.rglob("*")):
                if not source_path.is_file():
                    continue
                target = gallery_dir / source_path.relative_to(seed)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source_path.read_bytes())
                copied += 1
            print(f"seeded {copied} files from {seed} into {gallery_dir.resolve()}")

    items = collect(gallery_dir)
    if args.report:
        by_class = Counter(c for c, _s, _p in items)
        by_source = Counter(s for _c, s, _p in items)
        print(f"gallery: {gallery_dir.resolve()}  ({len(items)} references)")
        for name, count in sorted(by_class.items()):
            sources = Counter(s for c, s, _p in items if c == name)
            detail = ", ".join(f"{k}={v}" for k, v in sorted(sources.items()))
            print(f"  {name:22s} {count:4d}   ({detail})")
        print("by source:", dict(sorted(by_source.items())))
        return 0

    if not items:
        # An empty gallery is a valid state: the pipeline simply falls back to the
        # detector's coarse label. Say so and succeed, so container startup is not
        # blocked on having reference photos.
        print(f"No reference images under {gallery_dir.resolve()}; type stage will be disabled.")
        return 0

    encoder = ReferenceEncoder(model_dir, args.device)
    vectors, labels, names, sources, manifest = [], [], [], [], []
    for vehicle_class, source, path in items:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  skip (undecodable) {path}")
            continue
        vectors.append(encoder.embed(image))
        labels.append(vehicle_class)
        names.append(path.name)
        sources.append(source)
        manifest.append(
            {
                "reference": path.name,
                "vehicle_type": vehicle_class,
                "source": source,
                "path": str(path.relative_to(gallery_dir)).replace("\\", "/"),
                "pixels": [int(image.shape[1]), int(image.shape[0])],
            }
        )
        print(f"  {vehicle_class:22s} {source:14s} {path.name}")

    if not vectors:
        print("Nothing could be embedded.")
        return 1

    from app.services.vision.local.reference_gallery import CROP_POLICY_FILENAME

    policy_path = gallery_dir / CROP_POLICY_FILENAME
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    if not all(0 <= policy.get(key, 0) <= 1 for key in ("margin_x", "margin_y")):
        raise ValueError("Invalid reference crop policy")
    matrix = np.stack(vectors).astype(np.float32)
    np.savez_compressed(
        gallery_dir / INDEX_FILENAME,
        vectors=matrix,
        labels=np.array(labels),
        names=np.array(names),
        sources=np.array(sources),
    )
    counts = Counter(labels)
    (gallery_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "encoder": "dinov2-small (Apache-2.0), CLS + mean patch, L2-normalised",
                "dimensions": int(matrix.shape[1]),
                "crop_policy": policy,
                "references": manifest,
                "counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nindexed {len(labels)} references, {matrix.shape[1]}-d")
    for name, count in sorted(counts.items()):
        flag = "" if count >= 4 else "   <- thin; add more photos of this type"
        print(f"  {name:22s} {count:4d}{flag}")
    print(f"\nwrote {gallery_dir / INDEX_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
