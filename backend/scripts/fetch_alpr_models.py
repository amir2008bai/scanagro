"""Explicit installation only; runtime inference never downloads weights."""
from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

ARTIFACTS = (
    (
        "en_PP-OCRv4_mobile_rec.tar",
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
        "paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar",
        "42067abe43473ce9d68e3b4bf050765038062acfd810f7e1738af078c3198c8a",
        7_833_600,
    ),
    (
        "FSRCNN_x2.pb",
        "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb",
        "366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c",
        38_973,
    ),
)


def fetch(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, url, digest, size in ARTIFACTS:
        target = directory / name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Downloaded artifact failed verification: {name}")
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(target)
        if target.stat().st_size != size or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing artifact failed verification: {name}")
        print(f"Verified {name}", flush=True)
    model = directory / "en_PP-OCRv4_mobile_rec"
    model.mkdir(exist_ok=True)
    allowed = {"inference.json", "inference.pdiparams", "inference.yml"}
    found = set()
    with tarfile.open(directory / ARTIFACTS[0][0]) as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name not in allowed:
                continue
            if not member.isfile() or name in found or not 0 < member.size < 16_000_000:
                raise ValueError("Unexpected model archive member")
            found.add(name)
            with archive.extractfile(member) as stream:
                data = stream.read()
            target = model / name  # Only the fixed allowlisted basename; no archive paths/links.
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f"Extracted model failed verification: {name}")
            else:
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(target)
    if found != allowed:
        raise ValueError("Incomplete OCR archive")
    print("ALPR models ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    fetch(parser.parse_args().dir)
