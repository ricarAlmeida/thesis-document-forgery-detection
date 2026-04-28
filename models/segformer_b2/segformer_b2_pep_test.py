""" Test script (SegFormer-B2 + PEP only) """

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.torchtools.metrics import (
    BinaryPrecisionMetric,
    BinaryRecallMetric,
    BinaryF1ScoreMetric,
    Metrics,
)
from models.torchtools.test import evaluate

from models.doc_forgery_dataset_pep import DocForgeryDatasetPEP, Feature
from models.segformer_b2.model_segformer_b2_pep import SegFormerB2PEPRunner


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

dataset = DocForgeryDatasetPEP(
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
    features=[Feature.PEP],
    min_quality_factor=97,
    max_quality_factor=100,
    quality_factor=None,
    original_probability=0.0,
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

model = SegFormerB2PEPRunner(
    load_path="./weights_segformer_b2_20/",
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
    loader=loader,
    metrics=metrics,
    device=device,
    save_path="test_doctamper_pep_outputs/test_segformer_b2_pep_100epochs_q2_97_100-q3_90_100.json",
)