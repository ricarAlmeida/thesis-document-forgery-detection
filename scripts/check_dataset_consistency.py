"""
Check dataset consistency between images and masks.

- Compares filenames by stem (without extension)
- Reports images without masks and masks without images
- Reports zero-byte files

Usage:
  python scripts/check_dataset_consistency.py \
    --images /path/to/.../test/tampered /path/to/.../test/tampered \
    --masks  /path/to/.../test/mask     /path/to/.../test/mask
"""

from __future__ import annotations

import argparse
from pathlib import Path



IMAGE_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
MASK_EXTS_DEFAULT = [".png", ".PNG"]


def collect_files(roots: list[Path], exts: set[str]):
    """Return (stem->Path map, all_paths list)."""
    stem_to_path: dict[str, Path] = {}
    all_paths: list[Path] = []

    for root in roots:
        if not root.exists():
            print(f"[WARN] Folder does not exist: {root}")
            continue

        for p in root.iterdir():
            if p.is_file() and p.suffix in exts:
                all_paths.append(p)
                stem_to_path.setdefault(p.stem, p)  # keep first occurrence
    return stem_to_path, all_paths


def main():
    parser = argparse.ArgumentParser(description="Check image/mask consistency by filename stem.")
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        required=True,
        help="One or more folders containing images.",
    )
    parser.add_argument(
        "--masks",
        nargs="+",
        type=Path,
        required=True,
        help="One or more folders containing masks.",
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=IMAGE_EXTS_DEFAULT,
        help="Image extensions to consider.",
    )
    parser.add_argument(
        "--mask-exts",
        nargs="+",
        default=MASK_EXTS_DEFAULT,
        help="Mask extensions to consider.",
    )
    args = parser.parse_args()

    images_roots = [p.expanduser().resolve() for p in args.images]
    masks_roots = [p.expanduser().resolve() for p in args.masks]

    img_map, all_imgs = collect_files(images_roots, set(args.image_exts))
    mask_map, all_masks = collect_files(masks_roots, set(args.mask_exts))

    img_stems = set(img_map.keys())
    mask_stems = set(mask_map.keys())

    imgs_without_mask = sorted(img_stems - mask_stems)
    masks_without_img = sorted(mask_stems - img_stems)

    zero_size_imgs = [p for p in all_imgs if p.stat().st_size == 0]
    zero_size_masks = [p for p in all_masks if p.stat().st_size == 0]

    print("=== DATASET CONSISTENCY SUMMARY ===\n")
    print(f"Total images found: {len(all_imgs)}")
    print(f"Total masks found : {len(all_masks)}\n")

    print("Images without corresponding mask (by stem):")
    if imgs_without_mask:
        for s in imgs_without_mask:
            print("  -", img_map[s])
    else:
        print("  ✔ All images have masks.\n")

    print("\nMasks without corresponding image (by stem):")
    if masks_without_img:
        for s in masks_without_img:
            print("  -", mask_map[s])
    else:
        print("  ✔ All masks have images.\n")

    print("\nZero-byte files (IMAGES):")
    if zero_size_imgs:
        for p in zero_size_imgs:
            print("  -", p)
    else:
        print("  ✔ None.\n")

    print("\nZero-byte files (MASKS):")
    if zero_size_masks:
        for p in zero_size_masks:
            print("  -", p)
    else:
        print("  ✔ None.\n")

    # Exit code: 0 OK, 2 if issues found
    issues = bool(imgs_without_mask or masks_without_img or zero_size_imgs or zero_size_masks)
    raise SystemExit(2 if issues else 0)


if __name__ == "__main__":
    main()