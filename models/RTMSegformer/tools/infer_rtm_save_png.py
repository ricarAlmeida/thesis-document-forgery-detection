# ASCFormer/tools/infer_rtm_save_png.py

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model
from mmseg.utils import register_all_modules
from mmseg.registry import DATASETS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="test.txt")  # test.txt por defeito
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    register_all_modules()  # garante que datasets/models estão registados
    cfg = Config.fromfile(args.config)

    # Inicializa modelo
    model = init_model(cfg, args.checkpoint, device="cuda:0")

    # Resolve paths do dataset
    data_root = cfg.get("data_root", "RTM/")
    img_dir = cfg.get("img_dir", "images")
    ann_dir = cfg.get("ann_dir", "masks")

    # Split file
    split_path = os.path.join(data_root, args.split)
    with open(split_path, "r") as f:
        names = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Inferência imagem a imagem
    for name in tqdm(names, desc="Infer RTM"):
        img_path = os.path.join(data_root, img_dir, name)

        # mmseg inference
        result = model.predict(img_path)
        # result[0].pred_sem_seg.data -> tensor [H,W] com classes
        pred = result[0].pred_sem_seg.data.cpu().numpy().astype(np.uint8)

        # Para RTM queremos máscara binária 0/255:
        # classe 1 = tamper
        pred_bin = (pred == 1).astype(np.uint8) * 255

        out_path = os.path.join(args.out_dir, os.path.splitext(name)[0] + ".png")
        cv2.imwrite(out_path, pred_bin)

    print("Done. Saved predictions to:", args.out_dir)


if __name__ == "__main__":
    main()
