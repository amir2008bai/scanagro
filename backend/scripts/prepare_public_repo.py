#!/usr/bin/env python
"""Audit what a public release would expose, and optionally build a clean tree.

    python scripts/prepare_public_repo.py                 # audit only, changes nothing
    python scripts/prepare_public_repo.py --out ../public  # build a publishable copy

This repository mixes source code with operational CCTV material: original frames, crops
of vehicles, and licence-plate numbers stored as text. Plate numbers identify vehicles and,
through them, people; in most jurisdictions that makes them personal data. Publishing them
is a decision for whoever owns the footage, so this script reports rather than assumes.

`--out` copies the code, documentation and openly licensed reference photos into a fresh
directory and leaves the site material behind. Note what that costs: without the
`site_frames` references the machine-type gallery falls from 20/24 to 6/24 until the
operator adds their own photos, which is the documented design — see
docs/REFERENCE_GALLERY.md.

This script does not touch git history. If frames were already committed, removing them
from the working tree is not enough: the objects remain reachable. Publish a fresh
repository built with `--out`, or rewrite history deliberately before making the existing
one public.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (glob relative to the repo root, why it is sensitive)
SITE_MATERIAL: list[tuple[str, str]] = [
    ("photos/**/*", "original CCTV frames: vehicles, plates, and people in the yard"),
    ("assignment/**/*", "internal assignment document, not part of the software"),
    ("backend/reference_seed/*/site_frames/*", "vehicle crops taken from site footage"),
    ("backend/demo_results/annotated/*", "full frames with plates drawn on them"),
    ("backend/demo_results/plates/*", "close-up crops of licence plates"),
    ("backend/demo_results/results.json", "licence-plate numbers as text"),
    ("backend/demo_results/results.csv", "licence-plate numbers as text"),
    ("backend/demo_results/manifest.json", "maps result ids to original filenames"),
    ("backend/eval/perspective/crops/*", "close-up crops of licence plates"),
    ("backend/eval/perspective/comparison.*", "licence-plate readings as text"),
    ("backend/eval/ground_truth.json", "hand-written licence-plate numbers"),
    ("backend/eval/**/per_frame.json", "per-frame licence-plate readings"),
    ("backend/eval/astra6_*/**/*", "per-frame licence-plate readings"),
]

#: Files that must never be published regardless of the operator's decision.
SECRETS = ["**/.env", "**/.env.local", "**/*.pem", "**/*.key", "**/id_rsa*"]


def repo_root() -> Path:
    """The git top level if there is one, otherwise the backend directory's parent."""
    candidate = ROOT.parent
    return candidate if (candidate / ".git").exists() else ROOT


def collect(base: Path, patterns: list[str]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for pattern in patterns:
        hits = [p for p in base.glob(pattern) if p.is_file()]
        if hits:
            found[pattern] = sorted(hits)
    return found


#: Literal \\?\ — the Windows extended-length path prefix.
_WINDOWS_LONG_PREFIX = "\\\\?\\"


def long_path(path: Path) -> str:
    """Windows refuses paths over 260 characters unless given the extended-length prefix.

    Reference photos from Wikimedia keep their original long filenames, so a deep output
    directory tips over that limit and the copy fails halfway. Everywhere else this is a
    no-op.
    """
    text = str(path)
    if os.name == "nt" and not text.startswith(_WINDOWS_LONG_PREFIX):
        return _WINDOWS_LONG_PREFIX + os.path.abspath(text)
    return text


def plate_quoting_docs(base: Path) -> list[tuple[Path, int]]:
    """Markdown that quotes plate numbers in prose.

    Stripping images is not enough: the reports quote readings like `160 ALV 10` as the
    evidence behind every accuracy figure. Removing them would gut the documentation, so
    these are reported for a human to judge rather than deleted.
    """
    pattern = re.compile(r"\b\d{3}\s?[A-Z]{3}\s?\d{2}\b")
    found = []
    for path in sorted(base.rglob("*.md")):
        if set(path.relative_to(base).parts) & {".git", "node_modules"}:
            continue
        try:
            hits = pattern.findall(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        if hits:
            found.append((path, len(hits)))
    return found


def human(size: int) -> str:
    return f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} KB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=None, help="build a publishable copy here")
    parser.add_argument(
        "--include-site-data",
        action="store_true",
        help="deliberately publish the CCTV material as well",
    )
    args = parser.parse_args()

    base = repo_root()
    print(f"repository root: {base}\n")

    secrets = collect(base, SECRETS)
    if secrets:
        print("SECRETS PRESENT — these must never be committed:")
        for _pattern, paths in secrets.items():
            for path in paths:
                print(f"  !! {path.relative_to(base)}")
        print("  (.gitignore already excludes .env; check `git ls-files` to be sure)\n")

    site = collect(base, [pattern for pattern, _ in SITE_MATERIAL])
    reasons = dict(SITE_MATERIAL)
    total_files = sum(len(paths) for paths in site.values())
    total_bytes = sum(p.stat().st_size for paths in site.values() for p in paths)

    if not site:
        print("No site material found. Nothing here identifies a vehicle or a person.")
    else:
        print(
            f"Site material that a public repository would expose "
            f"({total_files} files, {human(total_bytes)}):\n"
        )
        for pattern, paths in site.items():
            size = sum(p.stat().st_size for p in paths)
            print(f"  {len(paths):>4d} files  {human(size):>9s}  {pattern}")
            print(f"{'':>17}{reasons[pattern]}")
        print(
            "\nLicence-plate numbers identify vehicles and, through them, people.\n"
            "Decide before making the repository public: afterwards the git history is\n"
            "out of your hands, and deleting the files in a later commit does not remove\n"
            "them from it."
        )

    quoting = plate_quoting_docs(base)
    if quoting:
        print(
            "\nDocumentation that quotes licence-plate numbers as evidence — these are NOT\n"
            "removed, because the measurements are the point of the documents. Read them and\n"
            "decide whether to redact:\n"
        )
        for path, count in quoting:
            print(f"  {count:>3d} mentions  {path.relative_to(base)}")

    if not args.out:
        print("\nAudit only. Re-run with --out DIR to build a publishable copy.")
        return 0

    target = Path(args.out).resolve()
    if target.exists() and any(target.iterdir()):
        print(f"\n{target} exists and is not empty; refusing to overwrite.")
        return 1

    excluded = (
        set() if args.include_site_data else {p.resolve() for paths in site.values() for p in paths}
    )
    excluded |= {p.resolve() for paths in secrets.values() for p in paths}
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "data"}

    copied = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file() or set(path.relative_to(base).parts) & skip_dirs:
            continue
        if path.resolve() in excluded:
            continue
        destination = target / path.relative_to(base)
        os.makedirs(long_path(destination.parent), exist_ok=True)
        shutil.copy2(long_path(path), long_path(destination))
        copied += 1

    # A gallery index that names removed references would mislead; make it rebuildable.
    stale = target / "backend" / "data" / "reference"
    for name in ("index.npz", "manifest.json"):
        (stale / name).unlink(missing_ok=True)

    print(f"\nwrote {copied} files to {target}")
    if excluded:
        print(f"left behind {len(excluded)} files of site material and secrets")
        print(
            "\nThe published gallery now holds only the openly licensed web references.\n"
            "Machine-type accuracy from those alone was measured at 6/24 — see\n"
            "backend/docs/REFERENCE_GALLERY.md. Operators are expected to add photos\n"
            "from their own camera; that is what takes it to 20/24."
        )
    print("\nNext: cd into it, `git init`, review `git status`, then push to a new repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
