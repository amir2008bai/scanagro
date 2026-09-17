#!/usr/bin/env python
"""Download and verify every model weight the local pipeline needs.

Run once before starting the service (the Docker image runs it at build time). Files are
written atomically and checked against the SHA-256 pinned in
``app/services/vision/local/model_registry.py``, so a truncated or substituted download
fails loudly instead of silently degrading recognition.

    python scripts/fetch_models.py [--dir DATA_DIR] [--skip-warm] [--check-only]
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.vision.local.model_registry import (  # noqa: E402
    DOWNLOADABLE,
    VENDORED_NOTES,
    ModelSpec,
    sha256_file,
)

USER_AGENT = "fable-backend-model-fetch/1.0"


def download(spec: ModelSpec, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{uuid4().hex}.part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            total = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                stream.write(chunk)
                total += len(chunk)
                print(f"\r  {spec.filename}: {total / 1e6:6.1f} MB", end="", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        print()
        if spec.sha256:
            actual = sha256_file(temporary)
            if actual != spec.sha256:
                raise RuntimeError(
                    f"{spec.filename}: checksum mismatch\n"
                    f"  expected {spec.sha256}\n  actual   {actual}"
                )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def warm_library_caches(model_dir: Path) -> None:
    """Build every session once, proving the downloaded artefacts actually load.

    All ONNX weights are fetched by this script and loaded by explicit path, so nothing
    reaches for ``~/.cache`` at runtime. The only weights not handled here are PP-OCR's,
    which ship inside the ``rapidocr-onnxruntime`` wheel.
    """
    from app.services.vision.local.pipeline import LocalRecognitionPipeline, PipelineConfig

    print("loading the pipeline to verify every artefact...")
    pipeline = LocalRecognitionPipeline(PipelineConfig(model_dir=model_dir, save_plate_crops=False))
    for key in pipeline.warm_up():
        print(f"  ready    {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=None, help="model directory (default: VISION_MODEL_DIR)")
    parser.add_argument(
        "--skip-warm", action="store_true", help="do not load the models after downloading"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify what is on disk and exit non-zero if anything is wrong",
    )
    args = parser.parse_args()

    model_dir = (
        Path(args.dir) if args.dir else Path(os.environ.get("VISION_MODEL_DIR", "./data/models"))
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"model directory: {model_dir.resolve()}")

    problems = 0
    for spec in DOWNLOADABLE:
        target = model_dir / spec.filename
        if target.is_file():
            if not spec.sha256 or sha256_file(target) == spec.sha256:
                print(f"  ok       {spec.filename}  [{spec.license}]")
                continue
            print(f"  MISMATCH {spec.filename} — re-downloading")
            # check-only must never delete operational data. A replacement download
            # is verified before os.replace atomically publishes it.
        if args.check_only:
            print(f"  MISSING  {spec.filename}")
            problems += 1
            continue
        print(f"  fetching {spec.filename} from {spec.source}")
        download(spec, target)
        print(f"  ok       {spec.filename}  [{spec.license}]")

    if args.check_only:
        return 1 if problems else 0

    if not args.skip_warm:
        warm_library_caches(model_dir)

    print("\nWeights provided inside a wheel:")
    for key, note in VENDORED_NOTES.items():
        print(f"  - {key}: {note}")
    print("\nAll model artefacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
