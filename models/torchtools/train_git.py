""" Train workflow implementation """

import os
import time
import logging
import torch
import subprocess as sp
from typing import Optional, Union
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.nn.modules.loss import _Loss

from peft import get_peft_model, PeftConfig, PeftModel

from .model import ModelRunner   # esta bem
from .metrics import Metrics   # esta bem
from .schedulers import SchedulerReference, Scheduler   # esta bem
from .utils import train_validation_split, AverageMeter   # esta bem


def get_logger(path: Path) -> logging.Logger:
    """
    Create and configure a logger that writes to 'path' and stdout.
    """
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(funcName)s:%(levelname)s:%(message)s")
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    return logger


@dataclass
class TrainParameters:s
    """
    Training configuration and checkpoint paths.

    Attributes:
        model_name: Experiment identifier.
        epochs: Total number of epochs.
        batch_size: Batch size used by the DataLoader.
        accum_batch_size: Effective batch size for gradient accumulation.
        save_root: Root directory where checkpoints will be stored.
        load_path: Optional checkpoint to resume from.
        logger_path: Optional file path for logging.
    """
    
    model_name: str
    epochs: int
    batch_size: int
    accum_batch_size: int = 64
    save_root: Path = Path("./weights")
    load_path: Optional[Path] = None
    logger_path: Optional[Path] = Path("./train.log")

    def __post_init__(self):

        if self.batch_size > self.accum_batch_size:
            raise Exception("Batch size must be less or equal then the accumulative batch size")

        if self.accum_batch_size % self.batch_size != 0:
            raise Exception("Accumulative batch size must be divisible by the batch size")
        
        self.accum_grads_steps = int(self.accum_batch_size / self.batch_size)
        
        self.save_dir = self.save_root / self.model_name

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=False)

        self.save_model_path = self.save_dir / f"{self.model_name}.pt"

    def save_path(self, epoch: Optional[int] = None) -> Path:
        """
        Build checkpoint path. If epoch is provided, include a timestamp and epoch id.
        """
        
        if epoch is not None:
            
            assert epoch >= 0
            
            save_path = self.save_dir / f"{str(datetime.now())}-checkpoint{epoch}.pth"
        else:
            save_path = self.save_dir / "checkpoint.pth"
            
        return save_path
        

def train_fn(
    parameters: TrainParameters,
    dataset: Dataset,
    validation_ratio: float,
    model: ModelRunner,
    optimizer: torch.optim.Optimizer,
    loss_fn: _Loss,
    metrics: Metrics,
    validation_metric: Optional[str],
    scheduler: Optional[Union[lr_scheduler.LRScheduler, Scheduler]] = None,
    use_cpu: bool = False,
    writer: Optional[SummaryWriter] = None,
    save_checkpoint: bool = False,
    save_model: bool = False,
    save_best_model: bool = True,
    update_scheduler_per_batch: bool = False,
    peft_config: Optional[PeftConfig] = None,
    enable_gradient_checkpoint: bool = False,
):
    """
    Train a segmentation model with optional validation split, checkpointing and TensorBoard logging.

    Args:
        parameters: Training hyperparameters and paths.
        dataset: Full dataset used for train/validation split.
        validation_ratio: Fraction of the dataset used for validation (0 < ratio < 1).
        model: A ModelRunner that implements `logits(batch, device, loss_fn)`.
        optimizer: Torch optimizer.
        loss_fn: Loss function used for training.
        metrics: Metrics accumulator used for logging/selection of best model.
        validation_metric: Metric name (key inside metrics) used to track the best model.
        scheduler: Optional LR scheduler (torch or torchtools scheduler).
        use_cpu: Train on CPU if True.
        writer: Optional TensorBoard SummaryWriter.
        save_checkpoint: Save "last checkpoint" every epoch if True.
        save_model: Save full model object if True.
        save_best_model: Save checkpoint when validation_metric improves if True.
        update_scheduler_per_batch: If True, update scheduler per batch.
        peft_config: Optional PEFT configuration (LoRA, etc.).
        enable_gradient_checkpoint: Enable gradient checkpointing (reduces memory, slower).
    """

    assert 0 < validation_ratio < 1

    assert validation_metric in metrics.metrics

    if enable_gradient_checkpoint:
        model.gradient_checkpointing_enable()

    if save_model:
        torch.save(model.model, parameters.save_model_path)

    if parameters.logger_path is not None:
        logger = get_logger(parameters.logger_path)
    else:
        logger = None

    if enable_gradient_checkpoint:
        model.gradient_checkpointing_enable()
    
    if not use_cpu:
    
        assert torch.cuda.is_available()
    
        device = torch.device("cuda")

        if logger is not None:
            logger.info("--- Using GPU ---")
        
    else:
        device = torch.device("cpu")

    train_ds, validation_ds = train_validation_split(
        dataset,
        validation_ratio=validation_ratio,
    )

    if logger is not None:
        logger.info("Training Size: %s", len(train_ds))
        logger.info("Validation Size: %s", len(validation_ds))
    
    train_loader = DataLoader(train_ds, batch_size=parameters.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=parameters.batch_size, shuffle=False)

    best_val_metric = 0.0
    epoch_start = 0

    if parameters.load_path is not None:
        print("Loading ...")
        ckpt = torch.load(parameters.load_path)

        epoch_start = ckpt.get("epoch", -1) + 1

        best_val_metric = ckpt.get("best_val_metric", float("-inf"))

        if "adapter_weights" in ckpt:
            model.model = PeftModel.from_pretrained(model.model, ckpt["adapter_weights"])
            model.model.print_trainable_parameters()
        else:
            model.model.load_state_dict(ckpt["state_dict"])

        model.model.to(device)
        metrics.to(device)
        optimizer.load_state_dict(ckpt["optimizer"])
        print("Done.")
    else:
        if peft_config is not None:
            model.model = get_peft_model(model.model, peft_config)
            model.model.print_trainable_parameters()

            
        model.model.to(device)
        metrics.to(device)

    trainable_params = sum(
        p.numel() for p in model.model.parameters() if p.requires_grad
    )

    epoch_iters = len(train_loader)    
    epoch_end = parameters.epochs           
    
    if logger is not None:
        logger.info("Number of trainable parameters: %s", trainable_params)
        logger.info("Number of Iterations per Epoch: %s", epoch_iters)
    
    assert epoch_end > epoch_start
    
    total_epochs = parameters.epochs
    max_iters = total_epochs * epoch_iters

    num_updates = 0
    checkpoint_id = 0
    for epoch in range(epoch_start, epoch_end):

        if save_checkpoint:
            if peft_config is None:
                state = {'epoch': epoch - 1, 'state_dict': model.model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, parameters.save_path())
            else:
                adapter_weights_path = parameters.save_dir / f"peft-adapters-weights / checkpoint-{checkpoint_id}"
                model.model.save_pretrained(adapter_weights_path)
                state = {'epoch': epoch - 1, 'adapter_weights': adapter_weights_path, 'optimizer': optimizer.state_dict()}

                checkpoint_id += 1
           
        t_epoch_start = time.time()

        model.model.train()

        # Initialize epoch training metrics
        train_loss = AverageMeter()
        running_train_loss = AverageMeter()

        if logger is not None:
            logger.info("Epoch %s", epoch)

        for batch_idx, batch in tqdm(enumerate(train_loader)):

            if scheduler is not None:
                
                if scheduler.__module__ == "torchtools.schedulers":
                    # Using custom defined scheduler
                    if scheduler.reference == SchedulerReference.EPOCH:
                        step = epoch
                        if batch_idx == 0:
                            update_lr = True 
                        else:
                            update_lr = False
                            
                    elif scheduler.reference == SchedulerReference.ITERATION:
                        update_lr = True
                        cur_iters = epoch * epoch_iters
                        step = (batch_idx + cur_iters) 
                        
                    else:
                        raise Exception("Unhandled Scheduler Reference")

                    if update_lr:
                        lr = scheduler(step, max_iters=max_iters)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                    if writer:
                        writer.add_scalar(
                            'Learning Rate',
                            lr,
                            step,
                        )
                else:
                    if update_scheduler_per_batch:
                        if num_updates == 0:
                            pass
                        else:   
                            scheduler.step_update(num_updates=num_updates)
                            
                        num_updates += 1
                        
                    elif batch_idx == 0:
                        # Using pytorch scheduler
                        scheduler.step()


                    if writer:
                        writer.add_scalar(
                            'Learning Rate',
                            optimizer.param_groups[0]["lr"],
                            num_updates - 1,
                        )
            
            output = model.logits(batch, device, loss_fn)

            logits = output.logits
            labels = output.labels
            loss = output.loss

            running_train_loss.update(output.loss.item())

            loss = loss / parameters.accum_grads_steps

            loss.backward()

            if (
                (batch_idx + 1) % parameters.accum_grads_steps == 0 
                or (batch_idx + 1) == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad()

                # scheduler.step(epoch + batch_idx / train_loader_size)

                with torch.no_grad():
    
                    train_loss.update(loss.detach().item() * parameters.accum_grads_steps)
                    metrics.add_batch(logits, labels)

                    if logger is not None:
                        logger.info(
                            '[train] epoch:{} batch:{} training loss:{:.6f}'.format(
                                epoch,
                                batch_idx,
                                running_train_loss.avg
                            )
                        )

                    if writer:
                        writer.add_scalar(
                            'Running Training Loss',
                            running_train_loss.avg,
                            epoch * len(train_loader) + batch_idx,
                        )

                    running_train_loss.reset()

        # Calculate epoch training metrics
        avg_train_loss = train_loss.avg
        train_loss.reset()

        metrics_evaluations = metrics.evaluate()

        t_epoch = time.time() - t_epoch_start

        if writer:
            writer.add_scalar(
                'Training Loss',
                avg_train_loss,
                epoch,
            )

            for metric, value in metrics_evaluations.items():
                writer.add_scalar(
                    f"Training {metric}",
                    value,
                    epoch,
                )
            
            writer.add_scalar(
                'Epoch Duration',
                t_epoch,
                epoch,
            )

        if logger is not None:
            logger_train_str = "[train] epoch:{} loss:{:.6f}".format(epoch, avg_train_loss)
    
            for metric, value in metrics_evaluations.items():
                logger_train_str += " {}:{:.6f}".format(metric, value)
    
            logger_train_str += " time:{} min".format(t_epoch / 60)
    
            logger.info(logger_train_str)

        # Validation
    
        # Clean up memory before validation
        torch.cuda.empty_cache()

        model.model.eval()
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(validation_loader):

                output = model.logits(batch, device, loss_fn)

                logits = output.logits
                labels = output.labels
                loss = output.loss

                val_loss.update(loss.item())
                metrics.add_batch(logits, labels)

            avg_val_loss = val_loss.avg
            val_loss.reset()
            
            metrics_evaluations = metrics.evaluate()

            if writer:
                writer.add_scalar(
                    'Validation Loss',
                    avg_val_loss,
                    epoch,
                )

                for metric, value in metrics_evaluations.items():
                    writer.add_scalar(
                        f"Validation {metric}",
                        value,
                        epoch,
                    )

            if logger is not None:
                logger_val_str = "[val] epoch:{} loss:{:.6f}".format(epoch, avg_val_loss)
    
                for metric, value in metrics_evaluations.items():
                    logger_val_str += " {}:{:.6f}".format(metric, value)
    
                logger.info(logger_val_str)

        if save_best_model:
            if metrics_evaluations:
                val_metric_reference = metrics_evaluations[validation_metric]
            else:
                val_metric_reference = avg_val_loss
                
            if val_metric_reference > best_val_metric:  
                if peft_config is None:
                    state = {
                        'epoch': epoch, 
                        'state_dict': model.model.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'best_val_metric': best_val_metric,
                    }
                else:
                    adapter_weights_path = parameters.save_dir / f"peft-adapters-weights / best-checkpoint"
                    model.model.save_pretrained(adapter_weights_path)
                    state = {'epoch': epoch - 1, 'adapter_weights': adapter_weights_path, 'optimizer': optimizer.state_dict()}
                    
                torch.save(state, parameters.save_path(epoch))
                
                best_val_metric = val_metric_reference
    
                if logger is not None:
                    logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))

    if writer:
        writer.close()