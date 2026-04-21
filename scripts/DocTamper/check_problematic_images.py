"""
Check if JPEG images are readable using PIL.

This script verifies whether image files can be opened and validated by PIL.
It is useful to detect corrupted or problematic JPEG files in a dataset.

Usage:
  python scripts/check_problematic_images.py \
    --dir /path/to/images \
    --ext .jpg .jpeg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image



def main():
    parser = argparse.ArgumentParser(description="Find problematic JPEG images.")
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing images to check.",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".jpg", ".jpeg", ".JPG", ".JPEG"],
        help="Extensions to include.",
    )
    args = parser.parse_args()

    base = args.dir.expanduser().resolve()
    if not base.exists():
        print(f"[ERROR] Directory does not exist: {base}")
        raise SystemExit(1)

    exts = set(args.ext)
    paths = [p for p in sorted(base.iterdir()) if p.is_file() and p.suffix in exts]

    if not paths:
        print(f"[INFO] No images found in {base} with extensions: {sorted(exts)}")
        raise SystemExit(0)

    bad = 0
    for img_path in paths:
        try:
            im = Image.open(img_path)
            im.verify()  # basic JPEG validation
        except Exception as e:
            bad += 1
            print(f"[BAD] {img_path}: {e}")

    print(f"\n[INFO] Checked {len(paths)} files. Problematic: {bad}.")
    raise SystemExit(0 if bad == 0 else 2)


if __name__ == "__main__":
    main()