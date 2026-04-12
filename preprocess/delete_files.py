"""
Delete orphaned files from the 'interpolated' folder.

After preprocessing, both 'images' and 'interpolated' subfolders are created
with matching filenames (e.g. elevation_0001.png / elevation_0001.txt).
When abnormal samples are manually removed from 'images', this script deletes
the corresponding files from 'interpolated' so the two folders stay in sync.

Usage:
    python preprocess/delete_files.py --dir /path/to/design_folder

    # Dry run (preview without deleting)
    python preprocess/delete_files.py --dir /path/to/design_folder --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def sync_interpolated(target_dir: Path, dry_run: bool = False) -> None:
    images_dir = target_dir / "images"
    interpolated_dir = target_dir / "interpolated"

    if not images_dir.is_dir():
        logger.error("'images' folder not found: %s", images_dir)
        sys.exit(1)
    if not interpolated_dir.is_dir():
        logger.error("'interpolated' folder not found: %s", interpolated_dir)
        sys.exit(1)

    # Collect stems present in images (e.g. "elevation_0001" from "elevation_0001.png")
    image_stems = {p.stem for p in images_dir.iterdir() if p.is_file()}

    orphans = [
        p for p in interpolated_dir.iterdir()
        if p.is_file() and p.stem not in image_stems
    ]

    if not orphans:
        logger.info("No orphaned files found in '%s'. Nothing to delete.", interpolated_dir)
        return

    logger.info(
        "%s orphaned file(s) found in '%s'%s:",
        len(orphans),
        interpolated_dir,
        " (dry run)" if dry_run else "",
    )
    for p in sorted(orphans):
        if dry_run:
            logger.info("  [dry run] would delete: %s", p.name)
        else:
            p.unlink()
            logger.info("  deleted: %s", p.name)

    if not dry_run:
        logger.info("Done. %s file(s) deleted.", len(orphans))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete files from 'interpolated' that are missing from 'images'."
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Directory containing 'images' and 'interpolated' subfolders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which files would be deleted without actually deleting them.",
    )
    args = parser.parse_args()

    target_dir = args.dir.resolve()
    if not target_dir.is_dir():
        logger.error("Directory not found: %s", target_dir)
        sys.exit(1)

    sync_interpolated(target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
