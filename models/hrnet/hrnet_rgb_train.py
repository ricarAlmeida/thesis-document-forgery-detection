""" Train script (HRNet-based + RGB only) """

import argparse
from pathlib import Path

import torch
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter

from models.torchtools.metrics import (
    BinaryPrecisionMetric,
    BinaryRecallMetric,
    BinaryF1ScoreMetric,
    Metrics,
)
from models.torchtools.schedulers import PowerDecayScheduler
from models.torchtools.losses import DiceLoss, FocalLoss
from models.torchtools.train import TrainParameters, train_fn

from models.doc_forgery_dataset_rgb import DocForgeryDatasetRGB, Feature

from models.hrnet.model_hrnet_rgb import HRNetRunnerForRGBSegmentation


parser = argparse.ArgumentParser(description="Train Args (HRNet-based + RGB only)")

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
    default="./doctamper_weights_hrnet_100_rgb",
    help="Model checkpoints directory path",
)
parser.add_argument(
    "--logger_path",
    type=str,
    default="./doctamper_train_hrnet_100_rgb.log",
    help="Training logs path",
)
parser.add_argument(
    "--tensorboard_path",
    type=str,
    default="./doctamper_runs_hrnet_100_rgb",
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

pre_trained_weights = args.pre_trained_weights
checkpoint = args.checkpoint


# HRNet-based + RGB only
feature = Feature.RGB


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

model = HRNetRunnerForRGBSegmentation(
    load_path=pre_trained_weights,
    use_data_parallel=False,
)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

dataset = DocForgeryDatasetRGB(
    images_repo=[Path(p) for p in args.images_repo],
    masks_repo=[Path(p) for p in args.masks_repo],
    crop_size=(height, width),
    grid_crop=True,
    features=[feature],
    original_probability=0.0,
    seed=3,
)

train_parameters = TrainParameters(
    model_name=f"hrnet-rgb-batchnorm",
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
            "params": filter(lambda p: p.requires_grad, model.model.parameters()),
            "lr": lr_0,
        }
    ],
    lr=lr_0,
    momentum=0.9,
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
        scheduler=None,
        early_stopping_patience=15,
        min_delta=1e-4,
        use_cpu=False,
        writer=writer,
        save_checkpoint=True,
        save_model=True,
    )
