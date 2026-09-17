#!/usr/bin/env python
"""Fetch openly licensed reference photos from Wikimedia Commons.

    python scripts/fetch_web_references.py --dir reference_seed --limit 4

Only files under a licence that permits redistribution are kept, and every one that is
kept is recorded in `SOURCES.md` with its author, licence and page URL, because CC BY and
CC BY-SA both require attribution and a ZIP handed to someone else has to carry it.

Web photos are a fallback, not a substitute for site footage. They are catalogue-style
shots taken from the side in good light; the camera this service watches is a fisheye
looking down at an angle. That gap is real and it is why these land in their own `web`
source directory — `eval/evaluate_reference.py` can then measure whether they help.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "fable-backend-reference/1.0 (open-source ALPR evaluation)"

#: Licences that allow redistribution inside a delivered archive. Anything else is
#: skipped rather than argued about.
ALLOWED_LICENCES = re.compile(
    r"^(cc0|public domain|cc by [0-9.]+|cc by-sa [0-9.]+|cc-by|cc-by-sa)", re.IGNORECASE
)

#: What to search for, per machine class in our vocabulary.
QUERIES: dict[str, list[str]] = {
    "tractor": [
        "Kirovets K-700A tractor",
        "MTZ Belarus 82 tractor",
        "John Deere tractor field",
    ],
    "truck": [
        "KamAZ dump truck",
        "MAZ truck",
    ],
    "truck_light": [
        "GAZ-53 truck",
        "ZIL-130 truck",
    ],
    "trailer": [
        "grain trailer agricultural",
        "tipping trailer tractor",
    ],
    "combine_harvester": [
        "combine harvester field",
        "Claas Lexion combine harvester",
    ],
    "wheel_loader": [
        "wheel loader construction",
    ],
}


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def search(term: str, limit: int) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": term,
        "gsrlimit": str(limit * 3),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": "900",
    }
    request = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return list(payload.get("query", {}).get("pages", {}).values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="reference_seed", help="gallery to add a 'web' source to")
    parser.add_argument("--limit", type=int, default=4, help="images to keep per search term")
    parser.add_argument("--source", default="web", help="source subdirectory name")
    args = parser.parse_args()

    gallery = ROOT / args.dir
    attributions: list[dict] = []
    kept_total = 0

    for vehicle_class, terms in QUERIES.items():
        target = gallery / vehicle_class / args.source
        for term in terms:
            try:
                pages = search(term, args.limit)
            except Exception as exc:
                print(f"  search failed for {term!r}: {exc}")
                continue
            kept = 0
            for page in pages:
                if kept >= args.limit:
                    break
                info = (page.get("imageinfo") or [{}])[0]
                meta = info.get("extmetadata", {})
                licence = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
                if not ALLOWED_LICENCES.match(licence):
                    continue
                if info.get("mime") not in {"image/jpeg", "image/png"}:
                    continue
                url = info.get("thumburl") or info.get("url")
                if not url:
                    continue
                name = re.sub(r"[^A-Za-z0-9._-]", "_", page["title"].removeprefix("File:"))
                name = Path(name).with_suffix(".jpg").name
                target.mkdir(parents=True, exist_ok=True)
                destination = target / name
                if destination.exists():
                    kept += 1
                    continue
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        destination.write_bytes(response.read())
                except Exception as exc:
                    print(f"  download failed {name}: {exc}")
                    continue
                attributions.append(
                    {
                        "file": f"{vehicle_class}/{args.source}/{name}",
                        "title": page["title"],
                        "author": strip_html(meta.get("Artist", {}).get("value", "")) or "unknown",
                        "licence": licence,
                        "page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page['title'])}",
                    }
                )
                kept += 1
                kept_total += 1
                print(f"  {vehicle_class:18s} {licence:14s} {name}")

    if not attributions:
        print("nothing downloaded")
        return 0

    lines = [
        "# Reference photo sources",
        "",
        "Photos in a `web/` directory come from Wikimedia Commons under the licence named",
        "below. CC BY and CC BY-SA require attribution, so this file travels with them.",
        "Anything in a `site_frames/` or `site_archive/` directory was cut from the",
        "operator's own footage and is not covered here.",
        "",
        "Regenerate with `python scripts/fetch_web_references.py`.",
        "",
        "| File | Author | Licence | Source |",
        "| --- | --- | --- | --- |",
    ]
    existing = gallery / "SOURCES.md"
    for item in sorted(attributions, key=lambda a: a["file"]):
        author = item["author"].replace("|", "/")[:70]
        lines.append(
            f"| `{item['file']}` | {author} | {item['licence']} | [{item['title']}]({item['page']}) |"
        )
    existing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\ndownloaded {kept_total} images; attribution written to {existing}")
    print("next: python scripts/build_reference_gallery.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
