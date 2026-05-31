"""Training script for dual-branch MMSEG SegFormer-B2 RGB-based with convolution layers document forgery localization and classification / image_only branch."""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter

from models.torchtools.losses import DiceLoss, FocalLoss
from models.torchtools.metrics import (
    BinaryPrecisionMetric,
    BinaryRecallMetric,
    BinaryF1ScoreMetric,
    Metrics,
)

from torch.optim import lr_scheduler
from models.torchtools.torch_train_dual_branch_segformer_b2_rgb_rtm import TrainParameters, train_fn

from models.doc_forgery_dataset_rgb_rtm import DocForgeryDataset
from models.dual_branch_with_conv_mmseg_segformer_b2_rgb_rtm.model_dual_branch_with_conv_segformer_b2_mmseg_rgb_rtm import SegFormerImageOnlyRunner  #SegFormerRGBRunner


parser = argparse.ArgumentParser(description='Train dual-branch RGB MMSEG segmentation model with convolution layers / image_only branch')

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
    #default="./dual_branch_weights_mmseg_segformer_b2_rgb_with_convs_rtm_corretoo/dual_branch-mmseg-segformer_b2-rgb/checkpoint.pth",
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
    default="./dual_image_only_branch_weights_mmseg_segformer_b2_rgb_with_convs_rtm_corretoo",   # dual_branch_weights_mmseg_segformer_b2_rgb_with_convs_rtm_corretoo
    help="Directory where checkpoints and models will be saved",
)
parser.add_argument(
    "--logger_path",
    type=str,
    default="./dual_image_only_branch_train_mmseg_segformer_b2_rgb_with_convs_rtm_corretoo.log",  # dual_branch_train_mmseg_segformer_b2_rgb_with_convs_rtm_corretoo
    help="Training log file path",
)
parser.add_argument(
    "--tensorboard_path",
    type=str,
    default="./dual_image_only_branch_runs_mmseg_segformer_b2_rgb_with_convs_rtm_corretoo",    # dual_branch_runs_mmseg_segformer_b2_rgb_with_convs_rtm_corretoo
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

pre_trained_weights = args.pre_trained_weights
checkpoint = args.checkpoint


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
model = SegFormerImageOnlyRunner(  # or SegFormerRGBRunner   
    load_path=pre_trained_weights,
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
dataset_train = DocForgeryDataset(
    images_repo=train_images_repo,
    masks_repo=train_masks_repo,
    crop_size=(height, width),
    grid_crop=True,
    seed=3,
    balance_crops=True,
)

dataset_train.use_augs = False


dataset_val = DocForgeryDataset(
    images_repo=val_images_repo,
    masks_repo=val_masks_repo,
    crop_size=(height, width),
    grid_crop=True,
    seed=3,
    balance_crops=False,
)

dataset_val.use_augs = False


# ---------------------------------------------------------------------
# Train parameters
# ---------------------------------------------------------------------
train_parameters = TrainParameters(
    model_name = f"dual-image-only-branch-mmseg-segformer_b2-rgb-with_conv",
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
            'lr': 0.005,
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
# Losses
# ---------------------------------------------------------------------
class CustomLoss(_Loss):
    """
    Dice + Focal loss for binary segmentation with ignore_index = -1.
    """
    
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(mode="binary", ignore_index=-1)
        self.focal = FocalLoss(mode="binary", alpha=0.9, gamma=2.0, ignore_index=-1)

    def forward(self, y_pred, y_true):
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        return (2.0 * self.dice(y_pred, y_true)) + self.focal(y_pred, y_true)


class MultiTaskLoss(_Loss):
    """Joint loss for the two-branch model: segmentation loss + image-level BCE loss."""

    def __init__(
        self,
        seg_loss: _Loss,
        cls_weight: float = 0.2,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()

        self.seg_loss = seg_loss
        self.cls_weight = float(cls_weight)

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([float(pos_weight)], dtype=torch.float32),
            )

        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, outputs, y_true):

        seg_logits, cls_logit = outputs

        loss_seg = self.seg_loss(seg_logits, y_true)

        mask = y_true.squeeze(1) if y_true.dim() == 4 else y_true

        valid = mask != -1
        tampered = (mask == 1) & valid
        y_img = tampered.flatten(1).any(dim=1).float()

        loss_cls = self.bce(cls_logit.view(-1), y_img)

        return loss_seg + self.cls_weight * loss_cls


class ImageOnlyLoss(_Loss):
    """Image-level BCE loss for the image-only classification model."""

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([float(pos_weight)], dtype=torch.float32),
            )

        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, cls_logit, y_true_mask):

        mask = y_true_mask.squeeze(1) if y_true_mask.dim() == 4 else y_true_mask

        valid = mask != -1
        tampered = (mask == 1) & valid
        y_img = tampered.flatten(1).any(dim=1).float()

        return self.bce(cls_logit.view(-1), y_img)


#seg_loss = CustomLoss()

#loss_fn = MultiTaskLoss(
#    seg_loss=seg_loss,
#    cls_weight=0.2,
#    pos_weight=None,
#) # or loss_fn = ImageOnlyLoss()

loss_fn = ImageOnlyLoss()


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
        loss_fn=loss_fn,
        metrics=metrics,
        validation_metric="f1-score",
        validation_dataset=dataset_val,
        scheduler=scheduler,
        use_cpu=False,
        writer=writer,
        save_checkpoint=True,
        save_model=True,
    )