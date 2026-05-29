""" Test script (MIML + RGB only) """

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collections import OrderedDict

from models.torchtools.metrics import (
    BinaryPrecisionMetric, 
    BinaryRecallMetric, 
    BinaryF1ScoreMetric, 
    Metrics,
)
from models.torchtools.test import evaluate

from models.doc_forgery_dataset_rgb import DocForgeryDatasetRGB, Feature
from models.miml.model_miml_rgb import MIMLRGBRunner


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

height = 512
width = 512
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

dataset = DocForgeryDatasetRGB(
    images_repo=[
        Path("/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-SCD/test/tampered"),
        Path("/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-TestingSet/test/tampered"),
    ],
    masks_repo=[
        Path("/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-SCD/test/mask"),
        Path("/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-TestingSet/test/mask"),
    ],
    crop_size=(height, width),
    grid_crop=True,
    features=[Feature.RGB],
    original_probability=0.0,
    seed=3,
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
# MIMLRGBRunner is a wrapper, so the checkpoint is loaded manually into
# the internal PyTorch model stored in model.model_. This also removes the
# "module." prefix saved when the model was trained with DataParallel.

ckpt_path = "/media/generalstorage5/weights_miml_rgb/miml-rgb-batchnorm/2026-05-26 15:07:59.980318-checkpoint15.pth"

model = MIMLRGBRunner(
    load_path=None,
    use_data_parallel=False,
)

ckpt = torch.load(ckpt_path, map_location=device)

if isinstance(ckpt, dict):
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
    new_state_dict[new_k] = v

missing, unexpected = model.model_.load_state_dict(new_state_dict, strict=False)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model.model_ = model.model_.to(device)
model.model_.eval()


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

metrics = Metrics(
    {
        "precision": BinaryPrecisionMetric(threshold=0.5, ignore_index=-1),
        "recall": BinaryRecallMetric(threshold=0.5, ignore_index=-1),
        "f1-score": BinaryF1ScoreMetric(threshold=0.5, ignore_index=-1),
    }
)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

evaluate(
    model=model,
    loader=loader,
    metrics=metrics,
    device=device,
    save_path="./test_doctamper_rgb_outputs/test_miml_rgb_all_epochs.json",
)
