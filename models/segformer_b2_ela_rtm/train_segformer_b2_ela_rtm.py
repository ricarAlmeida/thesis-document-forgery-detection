"""Training script for ELA-based document forgery localization."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter

from models.torchtools.losses import DiceLoss, FocalLoss
from models.torchtools.metrics import (
    BinaryF1ScoreMetric,
    BinaryPrecisionMetric,
    BinaryRecallMetric,
    Metrics,
)
from torch.optim import lr_scheduler
from models.torchtools.torch_train_segformer_b2_ela_rtm import TrainParameters, train_fn

from models.doc_forgery_dataset_ela_rtm import DocForgeryELADataset
from models.segformer_b2_ela_rtm.model_segformer_b2_ela_rtm import SegFormerELARunner


parser = argparse.ArgumentParser(description="Train ELA segmentation model")

parser.add_argument(
    "--QF_3",
    type=int,
    required=True,
    help="JPEG quality factor used for ELA extraction",
)
parser.add_argument(
    "--images_repo",
    nargs="+",
    type=str,
    default=[
        "/media/general_storage6/rmastorage/datasets/RealTextManipulation/train_v2/JPEGImages",
        "/media/general_storage6/rmastorage/datasets/RealTextManipulation/val_v2/JPEGImages",
    ],
    help="Image repositories: [train_images, val_images]",
)
parser.add_argument(
    "--masks_repo",
    nargs="+",
    type=str,
    default=[
        "/media/general_storage6/rmastorage/datasets/RealTextManipulation/train_v2/SegmentationClass",
        "/media/general_storage6/rmastorage/datasets/RealTextManipulation/val_v2/SegmentationClass",
    ],
    help="Mask repositories: [train_masks, val_masks]",
)
parser.add_argument(
    "-N",
    "--epochs",
    type=int,
    default=100,
)
parser.add_argument(
    "-B",
    "--batch_size",
    type=int,
    default=8,
)
parser.add_argument(
    "--accum_batch_size",
    type=int,
    default=64,
    help="Effective batch size for gradient accumulation",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint path to resume training from",
)
parser.add_argument(
    "--pre_trained_weights",
    type=str,
    default=None,
    help="Optional pretrained weights path",
)
parser.add_argument(
    "--save_root",
    type=str,
    default="./rtm_weights_segformer_b2_ela",
    help="Directory where checkpoints and models will be saved",
)
parser.add_argument(
    "--logger_path",
    type=str,
    default="./rtm_train_segformer_b2_ela.log",
    help="Training log file path",
)
parser.add_argument(
    "--tensorboard_path",
    type=str,
    default="./rtm_runs_segformer_b2_ela_rtm",
    help="TensorBoard output directory",
)

args = parser.parse_args()

# ---------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------
def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def is_main_process():
    return get_rank() == 0


Path(args.save_root).mkdir(parents=True, exist_ok=True)
Path(args.tensorboard_path).mkdir(parents=True, exist_ok=True)
Path(args.logger_path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------
epochs = args.epochs
batch_size = args.batch_size
accum_batch_size = args.accum_batch_size

height = 512
width = 512
lr_0 = 0.005

QF3 = args.QF_3

pre_trained_weights = args.pre_trained_weights
checkpoint = args.checkpoint


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
model = SegFormerELARunner(
    load_path=pre_trained_weights,
    #use_data_parallel=True,
)

m = model.model
if hasattr(m, "module"):
    m = m.module

for name, p in m.named_parameters():
    print(name, p.shape, p.requires_grad)
    

# ---------------------------------------------------------------------
# Dataset repositories
# ---------------------------------------------------------------------
train_images_repo = [Path(args.images_repo[0])]
train_masks_repo  = [Path(args.masks_repo[0])]

val_images_repo   = [Path(args.images_repo[1])]
val_masks_repo    = [Path(args.masks_repo[1])]


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
dataset_train = DocForgeryELADataset(
    images_repo=train_images_repo,
    masks_repo=train_masks_repo,
    crop_size=(height, width),
    grid_crop=True,
    seed=3,
    balance_crops=True,
)

dataset_train.use_augs = False


dataset_val = DocForgeryELADataset(
    images_repo=val_images_repo,
    masks_repo=val_masks_repo,
    crop_size=(height, width),
    grid_crop=True,
    seed=3,
    balance_crops=False,      # <-- "natural" validation
)

dataset_val.use_augs = False

# Quality factor used for ELA extraction
dataset_train.QF = QF3
dataset_val.QF = QF3


# ---------------------------------------------------------------------
# Train parameters
# ---------------------------------------------------------------------
train_parameters = TrainParameters(
    model_name = f"segformer_b2-ela-q3_{QF3}",
    epochs=epochs,
    batch_size=batch_size,
    accum_batch_size=accum_batch_size,
    save_root=Path(args.save_root),
    load_path=checkpoint,
    logger_path=args.logger_path,
)


# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    threshold=1e-4,
    min_lr=1e-6,
    verbose=True,
)


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
class CustomLoss(_Loss):
    """
    Dice + Focal loss for binary segmentation with ignore_index = -1.
    """
    
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(mode="binary", ignore_index=-1)
        self.focal = FocalLoss(mode="binary", alpha=0.25, gamma=2.0, ignore_index=-1)

    def forward(self, y_pred, y_true):
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        return (2.0 * self.dice(y_pred, y_true)) + self.focal(y_pred, y_true)
     

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
metrics = Metrics(
    {
        "precision": BinaryPrecisionMetric(threshold=0.5, ignore_index=-1),
        "recall": BinaryRecallMetric(threshold=0.5, ignore_index=-1),
        "f1-score": BinaryF1ScoreMetric(threshold=0.5, ignore_index=-1),
    }
)


# ---------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------
writer = None

if is_main_process():
    writer = SummaryWriter(
        Path(args.tensorboard_path)
        / train_parameters.model_name
        / f"h:{height}_w:{width}_epochs:{epochs}_batch:{batch_size}_lr_base:{lr_0}"
    )


if __name__ == "__main__":
    train_fn(
        parameters=train_parameters,
        dataset=dataset_train,
        model=model,
        optimizer=optimizer,
        loss_fn=CustomLoss(),
        metrics=metrics,
        validation_metric="f1-score",
        validation_dataset=dataset_val,
        scheduler=scheduler,
        use_cpu=False,
        writer=writer,
        save_checkpoint=True,
        save_model=True,
    )
