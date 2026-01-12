""" Test script (MIML + PEP only) """

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from torchtools.metrics import (
    BinaryPrecisionMetric, 
    BinaryRecallMetric, 
    BinaryF1ScoreMetric, 
    Metrics,
)
from torchtools.test import evaluate

from ..doc_forgery_dataset_pep import DocForgeryDatasetPEP, Feature
from model_segformer_b2_pep import SegFormerB2PEPRunner
from model_miml_pep import MIMLPEPRunner


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

dataset = DocForgeryDataset(
    images_repo=[
        Path("..."),
        Path("..."),
    ],
    masks_repo=[
        Path("..."),
        Path("..."),
    ],
    crop_size=(height, width),
    grid_crop=True,
    features=[Feature.PEP],
    min_quality_factor=97,
    max_quality_factor=100,
    quality_factor=None,
    original_probability=0.0,
    T=30,
    seed=3,
)

# QF used for PEP recompression (must match up with training)
dataset.QF = 90

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

model = MIMLPEPRunner(
    load_path="./weights_miml_100/...",
    use_data_parallel=True,
)


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
    dataloader=loader,
    metrics=metrics,
    device=device,
    save_path="test_miml_pep_q2_97_100-q3_90_100.json",
)