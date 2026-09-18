#!/usr/bin/env python3
"""Find orphan files left after crashes. Defaults to a dry run; use --apply to delete."""

import argparse
import asyncio
import time

from sqlalchemy import select

from app.db.models import Detection, ImageAsset
from app.db.session import SessionLocal, engine
from app.services.storage import LocalStorage


async def cleanup(apply=False, minimum_age_hours=24):
    storage = LocalStorage()
    storage.ensure_directories()
    async with SessionLocal() as session:
        rows = (
            await session.execute(select(ImageAsset.stored_filename, ImageAsset.annotated_filename))
        ).all()
        # Plate crops are referenced by detections, not by the image row.
        crops = (
            await session.execute(
                select(Detection.plate_crop_filename, Detection.plate_rectified_crop_filename)
            )
        ).all()
    referenced = {storage.original_path(row[0]) for row in rows}
    referenced.update(storage.thumb_path(row[0]) for row in rows)
    referenced.update(storage.annotated_path(row[1]) for row in rows if row[1])
    for before, after in crops:
        for name in (before, after):
            if name:
                try:
                    referenced.add(storage.plate_path(name))
                except ValueError:
                    continue  # a name that cannot be resolved cannot be protected
    cutoff = time.time() - minimum_age_hours * 3600
    count = 0
    for folder in (storage.upload_dir, storage.annotated_dir, storage.plate_dir, storage.thumb_dir):
        for path in folder.iterdir():
            if (
                path.is_file()
                and path.resolve() not in referenced
                and path.stat().st_mtime < cutoff
            ):
                print("remove" if apply else "orphan", path.name)
                if apply:
                    path.unlink(missing_ok=True)
                count += 1
    print(f"Orphan files: {count}")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(cleanup(args.apply))
