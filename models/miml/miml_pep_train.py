""" Train script (MIML + PEP only) """

import argparse
import time
import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter

from models.torchtools.metrics import BinaryPrecisionMetric, BinaryRecallMetric, BinaryF1ScoreMetric, Metrics
from models.torchtools.schedulers import CosineScheduler, PowerDecayScheduler
from models.torchtools.model import ModelRunner
from models.torchtools.losses import DiceLoss, FocalLoss
from models.torchtools.train import TrainParameters, train_fn

from models.doc_forgery_dataset_pep import DocForgeryDatasetPEP, Feature

from models.hrnet.model_hrnet_pep import HRNetRunnerForPEPSegmentation
from models.segformer_b2.model_segformer_b2_pep import SegFormerB2PEPRunner
from models.miml.model_miml_pep import MIMLPEPRunner


parser = argparse.ArgumentParser(description='Train Args (MIML + PEP only)')

parser.add_argument(
    '--minQF_2', 
    type=int,  
    required=True, 
    help='Tampered document minimum compression QF',
)
parser.add_argument(
    '--maxQF_2', 
    type=int, 
    required=True, 
    help='Tampered document maximum compression QF',
)
parser.add_argument(
    '--QF_3', 
    type=int, 
    required=True,  
    help='Additional compression QF',
)
parser.add_argument(
    "--backbone",
    type=str,
    default="hrnet",
    choices=["hrnet", "segformer-b2", "miml"],
    help="Segmentation network backbone"
)
parser.add_argument(
    "--images_repo",
    nargs="+",
    type=str,
    default=[
        "/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-FCD/tampered",
        "/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-SCD/tampered",
        "/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-TestingSet/tampered",
    ],
    help="Images repositories",
)
parser.add_argument(
    "--masks_repo",
    nargs="+",
    type=str,
    default=[
        "/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-FCD/mask",
        "/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-SCD/mask",
        "/media/general_storage6/rmastorage/datasets/doc-tamper/DocTamperV1-TestingSet/mask",
    ],
    help="Masks repositories",
)

parser.add_argument("-N", "--epochs", type=int, default=100)
parser.add_argument("-B", "--batch_size", type=int, default=8)
parser.add_argument(
    "--accum_batch_size",
    type=int,
    default=64,
    help="Batch size for gradient calculation",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Model checkpoint path",
)
parser.add_argument(
    "--pre_trained_weights",
    type=str,
    default=None,
    help="Model pre-trained weights path",
)

parser.add_argument(
    "--save_root",
    type=str,
    default="./weights_miml_100",
    help="Model checkpoints directory path",
)
parser.add_argument(
    "--logger_path",
    type=str,
    default="./train_miml_100.log",
    help="Training logs path",
)
parser.add_argument(
    "--tensorboard_path",
    type=str,
    default="./runs_miml_100",
    help="Tensorboard files path",
)

args = parser.parse_args()


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

epochs = args.epochs
batch_size = args.batch_size
accum_batch_size = args.accum_batch_size

height = 512
width = 512
lr_0 = 0.005

minQF2 = args.minQF_2
maxQF2 = args.maxQF_2
QF3 = args.QF_3

pre_trained_weights = args.pre_trained_weights
checkpoint = args.checkpoint


# MIML + PEP only
feature = Feature.PEP


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

if args.backbone == "hrnet":
    model = HRNetRunnerForPEPSegmentation(
        load_path=pre_trained_weights,
        use_data_parallel=True,
    )
elif args.backbone == "segformer-b2":
    model = SegFormerB2PEPRunner(
        load_path=pre_trained_weights,
        use_data_parallel=True,
    )
elif args.backbone == "miml":
    model = MIMLPEPRunner(
        load_path=pre_trained_weights,
        use_data_parallel=True,
    )
else:
    raise ValueError("Unknown feature")


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

dataset = DocForgeryDatasetPEP(
    images_repo=[Path(p) for p in args.images_repo],
    masks_repo=[Path(p) for p in args.masks_repo],
    crop_size=(height, width),
    grid_crop=True,
    features=[feature],
    min_quality_factor=minQF2,
    max_quality_factor=maxQF2,
    quality_factor=None,
    original_probability=0.0,
    seed=3,
)

dataset.QF = QF3

train_parameters = TrainParameters(
    model_name = f"miml-pep-batchnorm-q2_{maxQF2}_{minQF2}-q3_{QF3}",
    epochs=epochs,
    batch_size=batch_size,
    accum_batch_size=accum_batch_size,
    save_root=Path(args.save_root),
    load_path=checkpoint,
    logger_path=args.logger_path,
)

optimizer = torch.optim.SGD(
    [
        {
            'params': filter(lambda p: p.requires_grad, model.model.parameters()),
            'lr': lr_0,
        }
    ],
    lr=lr_0,
    momentum= 0.9,
    weight_decay=0.0005,
    nesterov=False,
)

scheduler = PowerDecayScheduler(
    lr_0=lr_0,
    max_iters=None,
    power=0.9,
)


class CustomLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.dice_loss_fn = DiceLoss(mode="binary", ignore_index=-1)
        self.focal_loss_fn = FocalLoss(mode="binary", alpha=0.25, gamma=2.0, ignore_index=-1)

    def forward(self, y_pred, y_true):
        return (3 * self.dice_loss_fn(y_pred, y_true)) + self.focal_loss_fn(y_pred, y_true)

        
metrics = Metrics(
    {
        "precision": BinaryPrecisionMetric(threshold=0.5, ignore_index=-1),
        "recall": BinaryRecallMetric(threshold=0.5, ignore_index=-1),
        "f1-score": BinaryF1ScoreMetric(threshold=0.5, ignore_index=-1),
    }
)

writer = SummaryWriter(
     Path(args.tensorboard_path) 
     / train_parameters.model_name 
     / f"h:{height}_w:{width}_epochs:{epochs}_batch:{batch_size}_lr_base:{lr_0}"
 )

if __name__ == "__main__":
    train_fn(
        parameters=train_parameters,
        dataset=dataset,
        validation_ratio=0.1,
        model=model,
        optimizer=optimizer,
        loss_fn=CustomLoss(),
        metrics=metrics,
        validation_metric="f1-score",
        scheduler=scheduler,
        use_cpu=False,
        writer=writer,
        save_checkpoint=True,
        save_model=True,
    )