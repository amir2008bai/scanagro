#!/usr/bin/env python3
"""Upload only this run's images, poll durable jobs, export and download annotations."""

import argparse
import asyncio
import csv
import io
import json
import mimetypes
import os
import time
from pathlib import Path

import httpx


async def run(args) -> int:
    directory = args.directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"Not an image directory: {directory}")
    candidates = directory.rglob("*") if args.recursive else directory.iterdir()
    files = sorted(
        p
        for p in candidates
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not files:
        raise ValueError(f"No JPEG/PNG/WEBP images found in {directory}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    api = args.api_url.rstrip("/") + args.api_prefix
    manifest = {"directory": str(directory), "api": api, "images": []}
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["directory"] != str(directory) or manifest["api"] != api:
            raise ValueError("Resume manifest belongs to a different directory or API")
    elif manifest_path.exists():
        raise ValueError("Output already has a manifest; use --resume or a new --out directory")

    def save_manifest():
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(manifest_path)

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient(timeout=180, headers=headers, follow_redirects=False) as client:
        uploaded = {item["source"] for item in manifest["images"]}
        for path in files:
            relative = str(path.relative_to(directory))
            if relative in uploaded:
                continue
            with path.open("rb") as stream:
                response = await client.post(
                    api + "/images",
                    files=[
                        (
                            "files",
                            (
                                path.name,
                                stream,
                                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                            ),
                        )
                    ],
                )
            response.raise_for_status()
            item = response.json()["items"][0]
            manifest["images"].append(
                {"source": relative, "image_id": item["image"]["id"], "job_id": item["job"]["id"]}
            )
            save_manifest()
        deadline = time.monotonic() + args.timeout_seconds
        pending = {item["job_id"] for item in manifest["images"]}
        while pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("Polling deadline reached; use --resume to continue this run")
            for job_id in list(pending):
                response = await client.get(api + f"/jobs/{job_id}")
                response.raise_for_status()
                job = response.json()
                if job["status"] in {"completed", "failed"}:
                    pending.remove(job_id)
                    print(job_id, job["status"], job.get("error_message") or "")
            if pending:
                await asyncio.sleep(args.poll_seconds)
        results = []
        csv_parts = []
        image_ids = [item["image_id"] for item in manifest["images"]]
        for start in range(0, len(image_ids), 50):
            params = [("image_id", value) for value in image_ids[start : start + 50]]
            response = await client.get(api + "/exports/results.json", params=params)
            response.raise_for_status()
            results.extend(response.json())
            response = await client.get(api + "/exports/results.csv", params=params)
            response.raise_for_status()
            csv_parts.append(response.content.decode("utf-8-sig"))
        (args.out / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        combined = io.StringIO(newline="")
        writer = csv.writer(combined)
        for index, part in enumerate(csv_parts):
            rows = list(csv.reader(io.StringIO(part)))
            writer.writerows(rows if index == 0 else rows[1:])
        (args.out / "results.csv").write_text(combined.getvalue(), encoding="utf-8-sig", newline="")
        annotated_dir = args.out / "annotated"
        annotated_dir.mkdir(exist_ok=True)
        for result in results:
            if not result["annotated_available"]:
                continue
            path = annotated_dir / f"{result['image_id']}.jpg"
            temp = path.with_suffix(".tmp")
            try:
                async with client.stream(
                    "GET", api + f"/images/{result['image_id']}/annotated"
                ) as response:
                    response.raise_for_status()
                    with temp.open("wb") as stream:
                        async for chunk in response.aiter_bytes():
                            stream.write(chunk)
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        # Plate crops before and after perspective correction: the evidence a reviewer
        # needs to judge the geometry stage rather than only its output.
        plate_dir = args.out / "plates"
        saved_crops = 0
        for result in results:
            for det in result["detections"]:
                plate = det.get("plate")
                if not plate:
                    continue
                for variant, key in (
                    ("before", "crop_filename"),
                    ("after", "rectified_crop_filename"),
                ):
                    if not plate.get(key):
                        continue
                    plate_dir.mkdir(exist_ok=True)
                    target = plate_dir / f"{det['detection_id']}-{variant}.jpg"
                    response = await client.get(
                        api + f"/detections/{det['detection_id']}/plate-crop",
                        params={"variant": variant},
                    )
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    target.write_bytes(response.content)
                    saved_crops += 1

        readable = sum(1 for r in results for d in r["detections"] if d.get("plate_readable"))
        plates_seen = sum(1 for r in results for d in r["detections"] if d.get("plate"))
        summary = {
            "images": len(results),
            "plates_detected": plates_seen,
            "plates_read": readable,
            "plate_crops_saved": saved_crops,
            "completed": sum(r["status"] == "completed" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
            "mock_images": sum(r["is_mock"] for r in results),
            "detections": sum(len(r["detections"]) for r in results),
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), "Outputs:", args.out.resolve())
        if summary["mock_images"]:
            print(
                "MOCK MODE: these outputs verify the pipeline; no real recognition was performed."
            )
        return 1 if summary["failed"] else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-prefix", default="/api/v1")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", ""),
        help="Prefer the API_KEY environment variable",
    )
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("poll-seconds and timeout-seconds must be positive")
    return args


def main():
    try:
        return asyncio.run(run(parse_args()))
    except httpx.HTTPStatusError as exc:
        print(
            f"API returned HTTP {exc.response.status_code}; check configuration and saved manifest"
        )
    except (httpx.HTTPError, OSError, ValueError, TimeoutError) as exc:
        print(f"Batch processing failed: {type(exc).__name__}: {exc}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
