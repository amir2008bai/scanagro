#!/usr/bin/env python
"""Measure what perspective correction actually buys, and save the evidence.

For every plate the detector finds, this reads the same crop four ways — untouched,
rotation-only, perspective-rectified, and rectified plus CLAHE — with both OCR engines,
and writes:

* ``crops/<frame>_<n>_{before,after}.jpg`` — the crop before and after rectification;
* ``comparison.json`` / ``comparison.csv`` — every reading with its measured confidence;
* a per-variant tally of how often each rendering produced the winning read.

    python eval/perspective_report.py --photos ../photos --out eval/perspective

The point of keeping the rotation-only variant is that it is the null hypothesis: if
rotating were enough, it would win as often as the full homography. It does not.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--out", default="eval/perspective")
    parser.add_argument("--models", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()

    os.chdir(ROOT)
    import cv2

    from app.services.vision.local.pipeline import LocalRecognitionPipeline, PipelineConfig

    out = Path(args.out)
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.models or os.environ.get("VISION_MODEL_DIR", "./data/models"))
    pipeline = LocalRecognitionPipeline(
        PipelineConfig(model_dir=model_dir, device=args.device, save_plate_crops=False)
    )
    pipeline.warm_up()
    reader = pipeline.plate_reader

    photos = sorted(
        Path(args.photos).glob("*.jpg"),
        key=lambda p: int(p.stem.split(" ")[0]) if p.stem.split(" ")[0].isdigit() else 0,
    )
    rows: list[dict] = []
    winners: Counter = Counter()

    for photo in photos:
        stem = photo.stem.replace(" (1)", "")
        image = cv2.imread(str(photo))
        if image is None:
            continue
        vehicles = pipeline.vehicle_detector.detect(image, pipeline.config.vehicle_threshold)
        plates = pipeline.plate_detector.detect(image, vehicles)
        plates = [p for p in plates if not pipeline._in_overlay_band(p, image.shape[0])]
        for index, plate in enumerate(sorted(plates, key=lambda p: -p.confidence)[:3]):
            crop, _, _ = pipeline._crop(image, plate)
            if crop.size == 0:
                continue
            variants, estimate = reader.build_variants(crop)
            aspect = estimate.aspect if estimate is not None and estimate.trusted_aspect else None

            # Upscaled so a reviewer can see the strokes, and JPEG-encoded at high
            # quality: these are for judging the geometry by eye, not for re-running OCR.
            enc = [cv2.IMWRITE_JPEG_QUALITY, 94]
            cv2.imwrite(
                str(crops / f"{stem}_{index}_before.jpg"),
                cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                enc,
            )
            if "rectified" in variants:
                cv2.imwrite(
                    str(crops / f"{stem}_{index}_after.jpg"),
                    cv2.resize(
                        variants["rectified"], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
                    ),
                    enc,
                )

            reads = []
            for name, rendering in variants.items():
                for engine, fn in (
                    ("fast_plate_ocr", reader._read_cct),
                    ("ppocr", reader._read_ppocr),
                ):
                    try:
                        read = fn(rendering, name)
                    except Exception:
                        read = None
                    if read is None:
                        continue
                    from app.services.vision.local import plate_format

                    fmt = plate_format.classify(read.text, aspect)
                    reads.append(
                        {
                            "frame": stem,
                            "plate_index": index,
                            "detection_confidence": round(plate.confidence, 4),
                            "rectified_aspect": round(aspect, 2) if aspect else None,
                            "quad_method": estimate.method if estimate else None,
                            "perspective_skew": round(estimate.skew, 3) if estimate else None,
                            "variant": name,
                            "engine": engine,
                            "text": read.compact,
                            "mean_confidence": round(read.mean_confidence, 4),
                            "min_confidence": round(read.min_confidence, 4),
                            "score": round(reader._score(read), 4),
                            "matches_format": fmt.name if fmt else None,
                        }
                    )
            if reads:
                top = max(reads, key=lambda r: r["score"])
                winners[f"{top['variant']}/{top['engine']}"] += 1
                for row in reads:
                    row["is_winner"] = row is top
                rows += reads

    (out / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        with (out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nwinning rendering/engine per plate:")
    for key, count in winners.most_common():
        print(f"  {key:28s} {count}")
    print(f"\n{len(rows)} readings over {len(winners)} plates -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
