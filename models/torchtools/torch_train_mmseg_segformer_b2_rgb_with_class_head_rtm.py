"""Training workflow implementation for RGB-based document forgery localization."""

import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from peft import PeftConfig, PeftModel, get_peft_model
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .torch_collate_mmseg_segformer_b2_rgb_with_class_head_rtm import pad_collate_classification
from .model import ModelRunner
from .utils import AverageMeter, train_validation_split
from .schedulers import SchedulerReference, Scheduler

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

    
def fix_sgd_param_groups(optimizer: torch.optim.Optimizer) -> None:
    if not isinstance(optimizer, torch.optim.SGD):
        return

    for group in optimizer.param_groups:
        group.setdefault("momentum", optimizer.defaults.get("momentum", 0.0))
        group.setdefault("dampening", optimizer.defaults.get("dampening", 0.0))
        group.setdefault("weight_decay", optimizer.defaults.get("weight_decay", 0.0))
        group.setdefault("nesterov", optimizer.defaults.get("nesterov", False))
        group.setdefault("maximize", optimizer.defaults.get("maximize", False))
        group.setdefault("foreach", optimizer.defaults.get("foreach", None))
        group.setdefault("differentiable", optimizer.defaults.get("differentiable", False))
        

def optimizer_to(optim: torch.optim.Optimizer, device: torch.device) -> None:
    """
    Move optimizer state tensors to the target device.
    Useful when resuming training from checkpoints saved on a different device.
    """
    
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def get_logger(path: Path) -> logging.Logger:
    """
    Create a logger for training messages.
    """
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(funcName)s:%(levelname)s:%(message)s")
    
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    return logger


@torch.no_grad()
def validate_one_pass_classification(
    model: "ModelRunner",
    dataloader,
    device,
    loss_fn,
    best_thr_running: float,
    thresholds=None,
    eps: float = 1e-8,
):
    """
    Validate a binary classification model in one pass and compute metrics at the
    running threshold and at the best validation threshold.
    """
    
    if thresholds is None:
        thresholds = torch.linspace(0.01, 0.99, 99, device=device, dtype=torch.float32)
    else:
        thresholds = torch.as_tensor(thresholds, device=device, dtype=torch.float32)

    thr_use = torch.tensor([best_thr_running], device=device, dtype=torch.float32)
    thresholds_all = torch.unique(torch.cat([thresholds, thr_use])).sort().values
    T = thresholds_all.numel()

    tp = torch.zeros(T, device=device)
    fp = torch.zeros(T, device=device)
    fn = torch.zeros(T, device=device)
    tn = torch.zeros(T, device=device)

    val_loss = AverageMeter()
    model.model.eval()

    for batch in dataloader:
        out = model.logits(batch, device, loss_fn)
        logits = out.logits.view(-1)          # [B]
        labels = out.labels.view(-1).long()   # [B]
        loss = out.loss

        val_loss.update(float(loss.item()))

        probs = torch.sigmoid(logits)         # [B]
        pred = probs.unsqueeze(0) >= thresholds_all.unsqueeze(1)   # [T,B]
        gt = labels.unsqueeze(0).expand_as(pred).bool()

        tp += (pred & gt).sum(dim=1).float()
        fp += (pred & ~gt).sum(dim=1).float()
        fn += (~pred & gt).sum(dim=1).float()
        tn += (~pred & ~gt).sum(dim=1).float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    best_idx = torch.argmax(f1)
    best_f1 = float(f1[best_idx].item())
    best_thr = float(thresholds_all[best_idx].item())

    use_idx = torch.argmin(torch.abs(thresholds_all - thr_use[0]))
    p_use = float(precision[use_idx].item())
    r_use = float(recall[use_idx].item())
    f1_use = float(f1[use_idx].item())
    acc_use = float(acc[use_idx].item())

    return {
        "avg_val_loss": float(val_loss.avg),
        "best_f1": best_f1,
        "best_thr": best_thr,
        "p_use": p_use,
        "r_use": r_use,
        "f1_use": f1_use,
        "acc_use": acc_use,
    }


@torch.no_grad()
def clf_init_counters(device: torch.device):
    """Initialize binary classification counters on the target device."""
    
    return {
        "tp": torch.tensor(0.0, device=device),
        "fp": torch.tensor(0.0, device=device),
        "fn": torch.tensor(0.0, device=device),
        "tn": torch.tensor(0.0, device=device),
    }


@torch.no_grad()
def clf_update_counters(
    counters: dict,
    logits: torch.Tensor,
    labels: torch.Tensor,
    thr: float,
):
    """Update binary classification counters for one batch at a fixed threshold."""
    
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    probs = torch.sigmoid(logits.view(-1))      # [B]
    gt = labels.view(-1).long()                 # [B]
    pred = (probs >= thr).long()                # [B]

    counters["tp"] += ((pred == 1) & (gt == 1)).sum().float()
    counters["fp"] += ((pred == 1) & (gt == 0)).sum().float()
    counters["fn"] += ((pred == 0) & (gt == 1)).sum().float()
    counters["tn"] += ((pred == 0) & (gt == 0)).sum().float()


@torch.no_grad()
def clf_finalize(counters: dict, eps: float = 1e-8):
    """Convert binary classification counters into precision, recall, F1, and accuracy."""
    
    tp = float(counters["tp"].item())
    fp = float(counters["fp"].item())
    fn = float(counters["fn"].item())
    tn = float(counters["tn"].item())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


@dataclass
class TrainParameters:
    """
    Parameters for training

    Attributes: 
        model_name: model identifier string
        epochs: maximum number of epochs
        batch_size: number of instances per batch
        accum_batch_size: number of instances for gradient calculation
        save_root: root for saving the model trained weights
        load_path: path with the model weights to be loaded
        logger_path: path to write training logs

    Note: if batch_size == accum_batch_size then there is no gradient accumulation
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
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_model_path = self.save_dir / f"{self.model_name}.pt"

    def save_path(self, epoch: Optional[int] = None) -> Path:
        """
        Return checkpoint save path.
        """
        
        if epoch is not None:
            
            assert epoch >= 0
            
            save_path = self.save_dir / f"{str(datetime.now())}-checkpoint{epoch}.pth"
        else:
            save_path = self.save_dir / "checkpoint.pth"
            
        return save_path
    

def _to_1d_cpu_tensor(x):
    """
    Convert list/np.ndarray/tensor/scalar into a 1D CPU tensor.
    """
    
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().flatten().cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).flatten().cpu()
    if isinstance(x, (list, tuple)):
        return torch.tensor(x).flatten().cpu()
    return torch.tensor([x]).flatten().cpu()


def unwrap_dp(model):
    """
    Return underlying model if wrapped in DataParallel or DDP.
    """
    
    return model.module if hasattr(model, "module") else model


def find_segformer_encoder(model):
    """
    Locate the MMSEG-style SegFormer encoder/backbone.
    Returns:
        encoder_module, set_of_encoder_parameter_ids
    """
    
    m = unwrap_dp(model)

    if not hasattr(m, "backbone"):
        return None, set()

    encoder = m.backbone
    enc_ids = {id(p) for p in encoder.parameters()}

    return encoder, enc_ids


def freeze_by_prefix(model, prefixes, freeze=True, logger=None, is_main=True):
    """
    Freeze or unfreeze parameters whose names start with one of the given prefixes.
    """
    
    m = unwrap_dp(model)
    prefixes = tuple(prefixes)

    hit = 0
    total = 0
    for name, p in m.named_parameters():
        total += 1
        if name.startswith(prefixes):
            p.requires_grad = (not freeze)
            hit += 1

    if logger and is_main:
        logger.info(f"[freeze_by_prefix] freeze={freeze} matched_params={hit}/{total} prefixes={prefixes}")


def train_fn(
    parameters: TrainParameters,
    dataset: Dataset,
    model: ModelRunner,
    optimizer: torch.optim.Optimizer,
    loss_fn: _Loss,
    validation_metric: Optional[str],
    validation_dataset: Dataset,
    scheduler: Optional[Union[lr_scheduler.LRScheduler, Scheduler]] = None,
    early_stopping_patience: int = 15,
    min_delta: float = 1e-4,
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
    Training procedure

    Parameters:
        parameters: parameters for training
        dataset: dataset to be used for train and validation
        validation_ratio: validation set ratio
        model: model to be trained
        optimizer: optimizer to use for gradient updates
        loss_fn: loss function to minimize
        scheduler: learning rate scheduler. Can be torch built-in or user defined
        validation_metric: validation metric to use in order to quantify performance
        use_cpu: True to train on CPU
        writer: tensorboard writer
        save_checkpoint: True to always save the model for each epoch
        save_model: True to save the model architecture
        save_best_model: True to save the model with the best validation matric
        update_scheduler_per_batch: True to update scheduler learning rate per batch
        peft_config: PEFT (Parameter Efficient Fine Tuning) configuration
        enable_gradient_checkpoint: True to enable gradient checkpointing. Increases memory 
                                    efficiency but penalizes backward pass speed.
    """

    ENABLE_OPTIMIZER_SWAPS = True
    ENABLE_FREEZE_SCHEDULE = True
    ENABLE_LR_SCALING_ON_BALANCE_OFF = True

    FREEZE_START = 20
    FREEZE_END = 35  # freeze on epochs [20..34]

    EARLY_STOP_START = 50

    local_rank, rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    parameters.accum_grads_steps = max(1, parameters.accum_batch_size // (parameters.batch_size * world_size))

    if enable_gradient_checkpoint:
        model.model.gradient_checkpointing_enable()

    if save_model:
        if is_main:
            torch.save(model.model, parameters.save_model_path)

    if parameters.logger_path is not None and is_main:
        logger = get_logger(parameters.logger_path)
    else:
        logger = None
    
    if not use_cpu:
        assert torch.cuda.is_available()
        device = torch.device(f"cuda:{local_rank}")
        if logger and is_main:
            logger.info("--- Using GPU ---")
    else:
        device = torch.device("cpu")

    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    # -----------------------------
    # Train and Validation Datasets
    # -----------------------------
    train_ds, validation_ds = dataset, validation_dataset

    if logger and is_main:
        logger.info("Training Size: %s", len(train_ds))
        logger.info("Validation Size: %s", len(validation_ds))

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=parameters.batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=pad_collate_classification,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=parameters.batch_size,
        shuffle=False,
        collate_fn=pad_collate_classification,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ------------------------------------------------------------------
    # Optional balanced validation loader for diagnostics
    # ------------------------------------------------------------------
    validation_ds_bal = None
    validation_loader_bal = None
    
    base_val = validation_ds.dataset if isinstance(validation_ds, Subset) else validation_ds
    if hasattr(base_val, "balance_crops"):
        validation_ds_bal = copy.deepcopy(validation_ds)
    
        base_val_bal = (
            validation_ds_bal.dataset
            if isinstance(validation_ds_bal, Subset)
            else validation_ds_bal
        )
        base_val_bal.balance_crops = True
    
        validation_loader_bal = DataLoader(
            validation_ds_bal,
            batch_size=parameters.batch_size,
            shuffle=False,
            collate_fn=pad_collate_classification,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    
        if logger and is_main:
            logger.info("[val-bal] created balanced validation loader")

    best_val_metric = float("-inf")
    epoch_start = 0

    if parameters.load_path is not None:
        if is_main:
            print("Loading ...")
    
        ckpt = torch.load(parameters.load_path, map_location="cpu")
    
        epoch_start = ckpt.get("epoch", -1) + 1
        best_val_metric = ckpt.get("best_val_metric", float("-inf"))
        best_thr_running = ckpt.get("best_thr", 0.5)

        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        best_epoch = ckpt.get("best_epoch", None)
    
        state_dict = ckpt.get("state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
        model.model.load_state_dict(state_dict, strict=False)
    
    else:
        best_thr_running = 0.5
    
    # common in both cases
    model.model = model.model.to(device)
    model.model = DDP(model.model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    
    loss_fn.to(device)
    
    m = unwrap_dp(model.model)
    
    # create optimizer AFTER DDP
    if epoch_start < 20:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, m.parameters()),
            lr=5e-3,
            momentum=0.9,
            weight_decay=1e-4,
        )
    
    elif epoch_start < 35:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, m.parameters()),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
    
    else:
        encoder, enc_ids = find_segformer_encoder(model.model)
    
        enc_params, head_params = [], []
        for p in m.parameters():
            if not p.requires_grad:
                continue
            (enc_params if id(p) in enc_ids else head_params).append(p)
    
        optimizer = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": 3e-5},
                {"params": head_params, "lr": 3e-4},
            ],
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
    
    optimizer_to(optimizer, device)
    scheduler = None
    
    if parameters.load_path is not None and is_main:
        logger.info(f"Loaded checkpoint best_thr={best_thr_running:.4f}")
        logger.info("Done loading.")

    trainable_params = sum(
        p.numel() for p in model.model.parameters() if p.requires_grad
    )

    epoch_iters = len(train_loader)
    epoch_end = parameters.epochs
    
    if logger and is_main:
        logger.info("Number of trainable parameters: %s", trainable_params)
        logger.info("Number of Iterations per Epoch: %s", epoch_iters)
    
    assert epoch_end > epoch_start
    
    total_epochs = parameters.epochs
    max_iters = total_epochs * epoch_iters

    num_updates = 0
    checkpoint_id = 0
    epochs_no_improve = 0
    best_epoch = None
    best_state = None

    for epoch in range(epoch_start, epoch_end):

        fixed_epochs = {20, 50, 100}
        current_epoch_1based = epoch + 1
        train_sampler.set_epoch(epoch)
        # ------------------------------------------------------------------
        # Curriculum on balanced crops
        # ------------------------------------------------------------------
        base = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        if hasattr(base, "balance_crops"):
            prev = getattr(base, "balance_crops", None)
            base.balance_crops = (epoch < 20)
        
            if logger and is_main and epoch in [19, 20]:
                logger.info(f"[data] balance_crops={base.balance_crops} at epoch {epoch}")

            # if ENABLE_LR_SCALING_ON_BALANCE_OFF and (prev is True) and (base.balance_crops is False):
            if ENABLE_LR_SCALING_ON_BALANCE_OFF and epoch == 20 and (prev is True) and (base.balance_crops is False):
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.1

                if logger and is_main:
                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    logger.info(f"[data] balance_crops OFF -> scaled lrs={lrs}")

        # ------------------------------------------------------------------
        # Freeze/unfreeze schedule
        # ------------------------------------------------------------------
        if ENABLE_FREEZE_SCHEDULE:
            should_freeze = (FREEZE_START <= epoch < FREEZE_END)
            freeze_by_prefix(
                model.model,
                prefixes=["backbone"],
                freeze=should_freeze,
                logger=logger,
                is_main=is_main,
            )
            if logger and is_main:
                logger.info(f"[freeze] encoder/backbone {'frozen' if should_freeze else 'UNfrozen'} (epoch {epoch})")
                m = unwrap_dp(model.model)
                n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in m.parameters())
                logger.info(f"[debug] trainable: {n_trainable}/{n_total}")

        # ------------------------------------------------------------------
        # Optimizer swaps
        # ------------------------------------------------------------------
        if ENABLE_OPTIMIZER_SWAPS and epoch == 20:
            m = unwrap_dp(model.model)
            head_params = [p for p in m.parameters() if p.requires_grad]
        
            if logger and is_main:
                logger.info(
                    f"[opt] switching to AdamW (trainable-only) at epoch {epoch}. "
                    f"trainable_params={sum(p.numel() for p in head_params)}"
                )
        
            if len(head_params) == 0:
                if logger and is_main:
                    logger.info("[opt][ERROR] trainable_params=0 after freeze. Not switching optimizer.")
            else:
                optimizer = torch.optim.AdamW(
                    head_params,
                    lr=1e-4,
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                )
                optimizer_to(optimizer, device)
                scheduler = None

        if ENABLE_OPTIMIZER_SWAPS and epoch == 35:
            m = unwrap_dp(model.model)
            encoder, enc_ids = find_segformer_encoder(model.model)
        
            if encoder is None or len(enc_ids) == 0:
                all_params = [p for p in m.parameters() if p.requires_grad]
                if logger and is_main:
                    logger.info(
                        f"[opt][WARN] Could not identify encoder at epoch {epoch}. "
                        f"Using single-group AdamW on all trainable params={sum(p.numel() for p in all_params)}"
                    )
        
                optimizer = torch.optim.AdamW(
                    all_params,
                    lr=3e-4,
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                )
                optimizer_to(optimizer, device)
                scheduler = None
        
            else:
                enc_params = []
                head_params = []
                for p in m.parameters():
                    if not p.requires_grad:
                        continue
                    if id(p) in enc_ids:
                        enc_params.append(p)
                    else:
                        head_params.append(p)
        
                if logger and is_main:
                    logger.info(
                        f"[opt] AdamW unfreeze at epoch {epoch}. "
                        f"enc_params={sum(p.numel() for p in enc_params)} "
                        f"head_params={sum(p.numel() for p in head_params)}"
                    )
        
                optimizer = torch.optim.AdamW(
                    [
                        {"params": enc_params, "lr": 3e-5},
                        {"params": head_params, "lr": 3e-4},
                    ],
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                )
                optimizer_to(optimizer, device)
                scheduler = None

            # reset patience after the final training-stage transition
            epochs_no_improve = 0
        
            if logger and is_main:
                logger.info("[early-stop] patience counter reset after epoch-35 unfreeze")

        # -----------------------------
        # Save checkpoint
        # -----------------------------
        if save_checkpoint:
            if peft_config is None:
                state = {
                    "epoch": epoch - 1,
                    "state_dict": unwrap_dp(model.model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_metric": best_val_metric,
                    "best_thr": float(best_thr_running),
                    "epochs_no_improve": epochs_no_improve,
                    "best_epoch": best_epoch,
                }
                if is_main:
                    torch.save(state, parameters.save_path())
            else:
                adapter_weights_path = parameters.save_dir / f"peft-adapters-weights / checkpoint-{checkpoint_id}"
                model.model.save_pretrained(adapter_weights_path)
                state = {
                    "epoch": epoch - 1,
                    "adapter_weights": str(adapter_weights_path),
                    "optimizer": optimizer.state_dict(),
                    "best_val_metric": best_val_metric,
                    "best_thr": float(best_thr_running),
                    "epochs_no_improve": epochs_no_improve,
                    "best_epoch": best_epoch,
                }
                if is_main:
                    torch.save(state, parameters.save_path())
                checkpoint_id += 1
           
        t_epoch_start = time.time()
        model.model.train()

        train_loss = AverageMeter()
        running_train_loss = AverageMeter()

        if logger and is_main:
            logger.info("Epoch %s", epoch)
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            logger.info(f"[lr] epoch={epoch} lrs={','.join(f'{x:.8e}' for x in lrs)}")

        print_every = 15
        s_total = 0
        s_want_pos = 0
        s_actual_pos = 0
        pos_ratios = []
        neg_ratios = []

        train_use_counters = clf_init_counters(device)

        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), disable=not is_main):
            labels_for_stats = batch["label"]
            pos_frac = labels_for_stats.float().mean().item()
        
            if logger and is_main and (batch_idx + 1) % 15 == 0:
                logger.info(
                    f"[BATCH-STATS] epoch={epoch} batch={batch_idx+1} "
                    f"pos_frac={pos_frac:.3f}"
                )
            
            if scheduler is not None:
                if scheduler.__module__ == "torchtools.schedulers":
                    if scheduler.reference == SchedulerReference.EPOCH:
                        step = epoch
                        update_lr = (batch_idx == 0)
                    elif scheduler.reference == SchedulerReference.ITERATION:
                        update_lr = True
                        cur_iters = epoch * epoch_iters
                        step = (batch_idx + cur_iters)
                    else:
                        raise Exception("Unhandled Scheduler Reference")
            
                    if update_lr:
                        lr = scheduler(step, max_iters=max_iters)
            
                        # applies only before swaps (because afterwards scheduler=None)
                        for pg in optimizer.param_groups:
                            pg["lr"] = lr
            
                    if writer and is_main and update_lr:
                        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], step)

                else:
                    if update_scheduler_per_batch:
                        if num_updates == 0:
                            pass
                        else:   
                            scheduler.step_update(num_updates=num_updates)
                            
                        num_updates += 1

                    if writer and is_main:
                        writer.add_scalar(
                            'Learning Rate',
                            optimizer.param_groups[0]["lr"],
                            num_updates - 1,
                        )

            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model.logits(batch, device, loss_fn)
                logits = output.logits[0] if isinstance(output.logits, (tuple, list)) else output.logits
                labels = output.labels
                loss = output.loss / parameters.accum_grads_steps
            
            clf_update_counters(
                counters=train_use_counters,
                logits=logits,
                labels=labels,
                thr=float(best_thr_running),
            )
            
            running_train_loss.update(float(output.loss.item()))

            finite_tensor = torch.tensor(
                [1 if torch.isfinite(loss).item() else 0],
                device=device,
                dtype=torch.int,
            )
            
            dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
            
            if finite_tensor.item() == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "rank": rank,
                        "local_rank": local_rank,
                        "loss": str(loss.detach().item()),
                        "output_loss": str(output.loss.detach().item()),
                        "logits_min": float(logits.detach().min().item()),
                        "logits_max": float(logits.detach().max().item()),
                        "logits_mean": float(logits.detach().mean().item()),
                        "labels_unique": torch.unique(labels.detach().cpu()),
                        "batch": {
                            k: v.detach().cpu() if torch.is_tensor(v) else v
                            for k, v in batch.items()
                        },
                    },
                    f"bad_batch_rank{rank}_e{epoch}_b{batch_idx}.pt",
                )
            
                if logger and is_main:
                    logger.error(
                        f"[ERROR] NaN/Inf loss detected at epoch={epoch} batch={batch_idx}. Aborting all ranks."
                    )
            
                raise RuntimeError(
                    f"NaN/Inf loss detected at epoch={epoch} batch={batch_idx}; aborting DDP safely."
                )
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (
                (batch_idx + 1) % parameters.accum_grads_steps == 0
                or (batch_idx + 1) == len(train_loader)
            ):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    train_loss.update(loss.detach().item() * parameters.accum_grads_steps)

                    if logger and is_main:
                        logger.info(
                            "[train] epoch:{} batch:{} training loss:{:.6f}".format(
                                epoch, batch_idx, running_train_loss.avg
                            )
                        )

                    if writer and is_main:
                        writer.add_scalar(
                            "Running Training Loss",
                            running_train_loss.avg,
                            epoch * len(train_loader) + batch_idx,
                        )

                    running_train_loss.reset()

        avg_train_loss = train_loss.avg
        train_loss.reset()

        train_stats = clf_finalize(train_use_counters)

        p_use_tr = train_stats["precision"]
        r_use_tr = train_stats["recall"]
        f1_use_tr = train_stats["f1"]
        acc_use_tr = train_stats["accuracy"]

        if writer and is_main:
            writer.add_scalar("Training UseThr", best_thr_running, epoch)
            writer.add_scalar("Training Precision@UseThr", p_use_tr, epoch)
            writer.add_scalar("Training Recall@UseThr", r_use_tr, epoch)
            writer.add_scalar("Training F1@UseThr", f1_use_tr, epoch)
            writer.add_scalar("Training Accuracy@UseThr", acc_use_tr, epoch)
        
        if logger and is_main:
            logger.info(
                f"[train-use] epoch:{epoch} useThr:{best_thr_running:.4f} "
                f"p/r/f1:{p_use_tr:.4f}/{r_use_tr:.4f}/{f1_use_tr:.4f}"
            )

        t_epoch = time.time() - t_epoch_start

        if writer and is_main:
            writer.add_scalar("Training Loss", avg_train_loss, epoch)
            writer.add_scalar("Training Precision@UseThr", p_use_tr, epoch)
            writer.add_scalar("Training Recall@UseThr", r_use_tr, epoch)
            writer.add_scalar("Training F1@UseThr", f1_use_tr, epoch)
            writer.add_scalar("Training Accuracy@UseThr", acc_use_tr, epoch)
            writer.add_scalar("Epoch Duration", t_epoch, epoch)
        
        if logger and is_main:
            logger.info(
                f"[train] epoch:{epoch} "
                f"loss:{avg_train_loss:.6f} "
                f"p/r/f1/acc:{p_use_tr:.4f}/{r_use_tr:.4f}/{f1_use_tr:.4f}/{acc_use_tr:.4f} "
                f"time:{t_epoch/60:.2f} min"
            )

        torch.cuda.empty_cache()

        stop_tensor = torch.tensor([0], device=device, dtype=torch.int)

        dist.barrier()
        
        if is_main:
            ddp_model = model.model
            model.model = unwrap_dp(model.model)
        
            try:
                val = validate_one_pass_classification(
                    model=model,
                    dataloader=validation_loader,
                    device=device,
                    loss_fn=loss_fn,
                    best_thr_running=best_thr_running,
                    thresholds=None,
                )
        
                val_bal = None
                if validation_loader_bal is not None:
                    val_bal = validate_one_pass_classification(
                        model=model,
                        dataloader=validation_loader_bal,
                        device=device,
                        loss_fn=loss_fn,
                        best_thr_running=best_thr_running,
                        thresholds=None,
                    )
            finally:
                model.model = ddp_model
        
            avg_val_loss = val["avg_val_loss"]
            best_f1 = val["best_f1"]
            best_thr = val["best_thr"]
            
            p_use = val["p_use"]
            r_use = val["r_use"]
            f1_use = val["f1_use"]
            acc_use = val["acc_use"]
        
            if writer:
                writer.add_scalar("Validation Loss", avg_val_loss, epoch)

                writer.add_scalar("Validation UseThr", best_thr_running, epoch)
                writer.add_scalar("Validation Precision@UseThr", p_use, epoch)
                writer.add_scalar("Validation Recall@UseThr", r_use, epoch)
                writer.add_scalar("Validation F1@UseThr", f1_use, epoch)
                writer.add_scalar("Validation Accuracy@UseThr", acc_use, epoch)
                
                writer.add_scalar("Validation BestF1", best_f1, epoch)
                writer.add_scalar("Validation BestThreshold", best_thr, epoch)
        
            if logger:
                logger.info(
                    f"[val] epoch:{epoch} loss:{avg_val_loss:.6f} "
                    f"useThr:{best_thr_running:.4f} "
                    f"p/r/f1/acc:{p_use:.4f}/{r_use:.4f}/{f1_use:.4f}/{acc_use:.4f} "
                    f"bestF1:{best_f1:.4f} bestThr:{best_thr:.4f}"
                )
        
            if val_bal is not None:
                if writer:
                    writer.add_scalar("Validation_BAL Loss", val_bal["avg_val_loss"], epoch)
                    writer.add_scalar("Validation_BAL Precision@UseThr", val_bal["p_use"], epoch)
                    writer.add_scalar("Validation_BAL Recall@UseThr", val_bal["r_use"], epoch)
                    writer.add_scalar("Validation_BAL F1@UseThr", val_bal["f1_use"], epoch)
                    writer.add_scalar("Validation_BAL Accuracy@UseThr", val_bal["acc_use"], epoch)
                    writer.add_scalar("Validation_BAL BestF1", val_bal["best_f1"], epoch)
                    writer.add_scalar("Validation_BAL BestThreshold", val_bal["best_thr"], epoch)
        
                if logger:
                    logger.info(
                        f"[val-BAL] epoch:{epoch} loss:{val_bal['avg_val_loss']:.6f} "
                        f"useThr:{best_thr_running:.4f} "
                        f"p/r/f1/acc:{val_bal['p_use']:.4f}/{val_bal['r_use']:.4f}/{val_bal['f1_use']:.4f}/{val_bal['acc_use']:.4f} "
                        f"bestF1:{val_bal['best_f1']:.4f} bestThr:{val_bal['best_thr']:.4f}"
                    )
        
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(best_f1)
        
            if save_best_model:
                val_metric_reference = float(best_f1)
        
                if val_metric_reference > best_val_metric + min_delta:
                    best_val_metric = val_metric_reference
                    best_thr_running = float(best_thr)
        
                    best_state = {
                        "model": {k: v.detach().cpu() for k, v in unwrap_dp(model.model).state_dict().items()},
                        "best_thr": best_thr_running,
                        "epoch": epoch,
                        "best_val": best_val_metric,
                        "epochs_no_improve": epochs_no_improve,
                        "best_epoch": best_epoch,
                    }
        
                    state = {
                        "epoch": epoch,
                        "state_dict": unwrap_dp(model.model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_metric": best_val_metric,
                        "best_thr": best_thr_running,
                        "epochs_no_improve": epochs_no_improve,
                        "best_epoch": best_epoch,
                    }
        
                    torch.save(state, parameters.save_path(epoch))
        
                    if logger:
                        logger.info(
                            f"[save] Best Model saved at epoch:{epoch} "
                            f"bestF1:{best_val_metric:.4f} bestThr:{best_thr_running:.4f}"
                        )
        
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
                if current_epoch_1based >= EARLY_STOP_START and best_epoch is not None and (epoch - best_epoch) >= early_stopping_patience:
                    stop_tensor[0] = 1
        
            if current_epoch_1based in fixed_epochs:
                fixed_state = {
                    "epoch": epoch,
                    "state_dict": unwrap_dp(model.model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_metric": best_val_metric,
                    "best_thr": float(best_thr_running),
                    "epochs_no_improve": epochs_no_improve,
                    "best_epoch": best_epoch,
                }
        
                fixed_path = parameters.save_dir / f"checkpoint_epoch_{current_epoch_1based}.pth"
                torch.save(fixed_state, fixed_path)
        
                if logger:
                    logger.info(f"[save-fixed] Saved checkpoint at epoch {current_epoch_1based}: {fixed_path}")

        # ------------------------------------------------------------------
        # Sync early stopping and validation threshold
        # ------------------------------------------------------------------
        # all ranks receive the decision to stop
        dist.broadcast(stop_tensor, src=0)

        best_thr_tensor = torch.tensor([best_thr_running], device=device, dtype=torch.float32)
        dist.broadcast(best_thr_tensor, src=0)
        best_thr_running = float(best_thr_tensor.item())
        
        if stop_tensor.item() == 1:
            break

    if writer and is_main:
        writer.close()
    
    dist.destroy_process_group()
