#!/usr/bin/env python
"""Measure the local pipeline against `eval/ground_truth.json`.

    python eval/evaluate.py --photos ../photos --out eval/results

Reports, per frame and in aggregate:

* plate localisation — was a plate box produced where the labeller saw a plate;
* exact match — accepted text identical to the label, over the frames a human could
  read in full. That set is the honest denominator: scoring against frames whose plates
  nobody can read would flatter the number;
* abstention — on frames the labeller marked partial or illegible, did the system
  correctly decline to assert a plate, or did it emit one anyway (a false accept);
* character errors on the readable set;
* the effect of perspective correction, by re-reading each crop with and without it.

Twenty-five frames from one camera cannot establish general accuracy. They establish
behaviour on this camera, and they catch regressions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalise(text: str | None) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def char_errors(expected: str, actual: str) -> int:
    """Levenshtein distance between two compact plate strings."""
    if expected == actual:
        return 0
    previous = list(range(len(actual) + 1))
    for i, ce in enumerate(expected, 1):
        current = [i]
        for j, ca in enumerate(actual, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ce != ca)))
        previous = current
    return previous[-1]


def frame_path(photos: Path, stem: str) -> Path | None:
    for candidate in (f"{stem}.jpg", f"{stem} (1).jpg", f"{stem}.jpeg", f"{stem}.png"):
        path = photos / candidate
        if path.is_file():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", required=True, help="directory holding the labelled frames")
    parser.add_argument("--truth", default=str(Path(__file__).with_name("ground_truth.json")))
    parser.add_argument(
        "--models", default=None, help="model directory (default: VISION_MODEL_DIR)"
    )
    parser.add_argument("--out", default=None, help="write per-frame JSON here")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument(
        "--alpr-ocr-dir", default=None, help="enable the paddle_alpr reader with this model dir"
    )
    parser.add_argument("--alpr-sr", default=None, help="FSRCNN .pb for the paddle_alpr reader")
    args = parser.parse_args()

    os.chdir(ROOT)
    from app.services.vision.local.pipeline import LocalRecognitionPipeline, PipelineConfig

    model_dir = Path(args.models or os.environ.get("VISION_MODEL_DIR", "./data/models"))
    pipeline = LocalRecognitionPipeline(
        PipelineConfig(
            model_dir=model_dir,
            device=args.device,
            save_plate_crops=False,
            alpr_enabled=args.alpr_ocr_dir is not None,
            alpr_ocr_dir=Path(args.alpr_ocr_dir) if args.alpr_ocr_dir else None,
            alpr_sr_path=Path(args.alpr_sr) if args.alpr_sr else None,
        )
    )
    pipeline.warm_up()

    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))["frames"]
    photos = Path(args.photos)
    rows, total_seconds = [], 0.0

    for stem, label in truth.items():
        path = frame_path(photos, stem)
        if path is None:
            print(f"!! missing frame {stem}")
            continue
        started = time.perf_counter()
        result, raw = pipeline.analyze_path(path)
        elapsed = time.perf_counter() - started
        total_seconds += elapsed

        plates = [d.plate for d in result.detections if d.plate is not None]
        accepted = [d for d in result.detections if d.plate_readable and d.license_plate]
        best = accepted[0] if accepted else None
        expected = normalise(label["plate"]) if label["plate"] else ""
        actual = normalise(best.license_plate) if best else ""

        rows.append(
            {
                "frame": stem,
                "category": label["category"],
                "expected_plate": label["plate"],
                "expected_manufacturer": label["manufacturer"],
                "reported_plate": best.license_plate if best else None,
                "reported_manufacturer": (
                    best.manufacturer
                    if best
                    else next((d.manufacturer for d in result.detections if d.manufacturer), None)
                ),
                "plate_box_found": bool(plates),
                "best_raw_text": next((p.text_raw for p in plates if p.text_raw), None),
                "ocr_confidence": best.plate.ocr_confidence if best and best.plate else None,
                "ocr_engine": best.plate.ocr_engine if best and best.plate else None,
                "ocr_variant": best.plate.ocr_variant if best and best.plate else None,
                "rectified": best.plate.rectified if best and best.plate else None,
                "status": plates[0].status if plates else "no_plate_box",
                "vehicles": len(result.detections),
                "seconds": round(elapsed, 2),
                "exact_match": bool(expected) and expected == actual,
                "char_errors": char_errors(expected, actual) if expected else None,
            }
        )
        print(
            f"{stem:>3s} [{label['category']:<12s}] want={label['plate'] or '-':<12s} "
            f"got={rows[-1]['reported_plate'] or '-':<12s} "
            f"raw={rows[-1]['best_raw_text'] or '-':<12s} {elapsed:5.1f}s",
            flush=True,
        )

    readable = [r for r in rows if r["category"] == "readable"]
    partial = [r for r in rows if r["category"] == "partial"]
    illegible = [r for r in rows if r["category"] == "none_legible"]
    with_plate = readable + partial

    summary = {
        "frames": len(rows),
        "readable_frames": len(readable),
        "partial_frames": len(partial),
        "none_legible_frames": len(illegible),
        "plate_localised_on_frames_with_a_plate": f"{sum(r['plate_box_found'] for r in with_plate)}/{len(with_plate)}",
        "exact_match_on_readable": f"{sum(r['exact_match'] for r in readable)}/{len(readable)}",
        "character_errors_on_readable": sum(r["char_errors"] or 0 for r in readable),
        "abstained_on_partial": f"{sum(not r['reported_plate'] for r in partial)}/{len(partial)}",
        "false_accepts_on_none_legible": f"{sum(bool(r['reported_plate']) for r in illegible)}/{len(illegible)}",
        "manufacturer_correct": f"{sum(1 for r in rows if r['expected_manufacturer'] and r['reported_manufacturer'] == r['expected_manufacturer'])}"
        f"/{sum(1 for r in rows if r['expected_manufacturer'])}",
        "manufacturer_wrong": sum(
            1
            for r in rows
            if r["reported_manufacturer"]
            and r["expected_manufacturer"]
            and r["reported_manufacturer"] != r["expected_manufacturer"]
        ),
        "accepted_via_rectified_crop": sum(1 for r in rows if r["rectified"]),
        "seconds_per_frame_mean": round(total_seconds / max(1, len(rows)), 2),
    }

    print("\n=== summary ===")
    for key, value in summary.items():
        print(f"{key:48s} {value}")

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
