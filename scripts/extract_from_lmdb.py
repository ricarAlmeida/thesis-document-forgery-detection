"""
Extract DocTamper LMDB samples into existing target folders.

This script:
- opens an LMDB folder that contains data.mdb / lock.mdb
- reads num-samples
- maps LMDB sample index -> existing (tampered, mask, test/tampered, test/mask) file paths
- writes image and mask bytes to those paths, preserving order

Usage:
  python scripts/extract_from_lmdb.py \
    --base-dir /path/to/datasets/doc-tamper \
    --datasets DocTamperV1-SCD DocTamperV1-TestingSet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lmdb
import six
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm



def open_lmdb(root: Path):
    env = lmdb.open(
        str(root),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    with env.begin(write=False) as txn:
        raw = txn.get(b"num-samples")
        if raw is None:
            raise RuntimeError(f"'num-samples' key not found in LMDB at: {root}")
        n_samples = int(raw)
    print(f"[INFO] LMDB: {root} | num-samples = {n_samples}")
    return env, n_samples


def load_target_pairs(base_dir: Path):
    """
    Reads existing files in:
      - tampered/ + mask/
      - test/tampered/ + test/mask/
    Returns list of pairs (img_path, mask_path), ordered by filename.
    """
    pairs = []

    # --- train/val: tampered + mask ---
    tampered_dir = base_dir / "tampered"
    mask_dir = base_dir / "mask"

    if not tampered_dir.exists() or not mask_dir.exists():
        print(
            f"[WARN] Missing expected folders in {base_dir}.\n"
            f"       Expected: {tampered_dir} and {mask_dir}\n"
            f"       Tip: create/prepare these folders before extracting."
        )
    train_imgs = sorted(tampered_dir.glob("*.png")) if tampered_dir.exists() else []
    train_masks = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []

    if len(train_imgs) != len(train_masks):
        print(
            f"[WARN] In {base_dir}: tampered ({len(train_imgs)}) "
            f"!= mask ({len(train_masks)})"
        )
    train_n = min(len(train_imgs), len(train_masks))
    train_pairs = list(zip(train_imgs[:train_n], train_masks[:train_n]))
    print(f"[INFO] {base_dir.name}: {train_n} pairs (tampered/mask)")

    pairs.extend(train_pairs)

    # --- test: test/tampered + test/mask ---
    test_tampered_dir = base_dir / "test" / "tampered"
    test_mask_dir = base_dir / "test" / "mask"

    if test_tampered_dir.exists() and test_mask_dir.exists():
        test_imgs = sorted(test_tampered_dir.glob("*.png"))
        test_masks = sorted(test_mask_dir.glob("*.png"))

        if len(test_imgs) != len(test_masks):
            print(
                f"[WARN] In {base_dir}/test: tampered ({len(test_imgs)}) "
                f"!= mask ({len(test_masks)})"
            )

        test_n = min(len(test_imgs), len(test_masks))
        test_pairs = list(zip(test_imgs[:test_n], test_masks[:test_n]))
        print(f"[INFO] {base_dir.name}: {test_n} pairs (test/tampered, test/mask)")

        pairs.extend(test_pairs)
    else:
        print(
            f"[INFO] {base_dir.name}: no test/tampered or test/mask folders found — skipping test."
        )

    print(f"[INFO] Total target pairs in {base_dir.name}: {len(pairs)}")
    return pairs


def extract_for_dataset(root_dir: Path):
    """
    For one dataset folder (e.g., DocTamperV1-SCD):
      - opens the LMDB
      - loads target pairs from existing folders
      - for each idx i: reads image-i / label-i and writes into target paths
    """
    print("\n" + "=" * 80)
    print(f"[INFO] Processing dataset: {root_dir}")
    print("=" * 80)

    if not root_dir.exists():
        print(f"[ERROR] Dataset folder does not exist: {root_dir}")
        return 1

    # 1) LMDB open
    try:
        env, n_samples = open_lmdb(root_dir)
    except Exception as e:
        print(f"[ERROR] Failed to open LMDB at {root_dir}: {e}")
        return 1

    # 2) target pairs
    target_pairs = load_target_pairs(root_dir)
    n_targets = len(target_pairs)

    if n_targets == 0:
        print(f"[WARN] No target pairs found under {root_dir}. Nothing to do.")
        return 0

    N = min(n_samples, n_targets)
    print(f"[INFO] Extracting {N} samples from LMDB into: {root_dir.name}")

    # 3) extraction loop
    with env.begin(write=False) as txn:
        for idx in tqdm(range(1, N + 1), desc=f"Extracting {root_dir.name}"):
            img_key = f"image-{idx:09d}".encode("utf-8")
            lbl_key = f"label-{idx:09d}".encode("utf-8")

            imgbuf = txn.get(img_key)
            lblbuf = txn.get(lbl_key)

            if imgbuf is None or lblbuf is None:
                print(f"[WARN] Sample {idx}: image or label not found — skipping.")
                continue

            out_img_path, out_mask_path = target_pairs[idx - 1]

            # --- image ---
            try:
                buf = six.BytesIO(imgbuf)
                im = Image.open(buf).convert("RGB")
                out_img_path.parent.mkdir(parents=True, exist_ok=True)
                im.save(str(out_img_path))
            except Exception as e:
                print(f"[ERROR] Writing image {out_img_path}: {e}")

            # --- mask ---
            try:
                mask_np = cv2.imdecode(np.frombuffer(lblbuf, np.uint8), cv2.IMREAD_UNCHANGED)
                if mask_np is None:
                    raise RuntimeError("cv2.imdecode returned None")
                out_mask_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(out_mask_path), mask_np)
                if not ok:
                    raise RuntimeError("cv2.imwrite returned False")
            except Exception as e:
                print(f"[ERROR] Writing mask {out_mask_path}: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Extract DocTamper LMDB into existing target folders.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base folder that contains DocTamperV1-SCD / DocTamperV1-TestingSet etc.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DocTamperV1-SCD", "DocTamperV1-TestingSet"],
        help="Dataset folder names under --base-dir to process.",
    )
    args = parser.parse_args()

    base = args.base_dir.expanduser().resolve()

    exit_code = 0
    for name in args.datasets:
        d = base / name
        code = extract_for_dataset(d, overwrite=overwrite)
        exit_code = max(exit_code, code)

    print("\n[OK] Extraction finished.")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()