""" SegFormer-B2 model implementation for learning PEP features"""

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation, SegformerConfig

from models.torchtools.model import ModelRunner, ModelOutput

from huggingface_hub import snapshot_download


class SegFormerB2PepBackbone(nn.Module):
    """
    SegFormer-B2 backbone for PEP-based forgery segmentation.

    This module consumes a single-channel PEP map and predicts a single-channel
    segmentation logit map.

    Notes:
        - The HuggingFace SegFormer expects 3-channel inputs. We convert the PEP
          map from 1 channel to 3 channels by repetition (PEP, PEP, PEP).
        - The decoder output is typically lower resolution; we upsample back to
          the original input resolution.

    Args:
        num_labels: Number of output labels. For binary segmentation this is 1.

    Input:
        pep: Tensor of shape [B, 1, H, W]

    Output:
        logits: Tensor of shape [B, num_labels, H, W]
    """

    def __init__(self, num_labels: int = 1):
        super().__init__()

        MODEL_ID = "nvidia/mit-b2"
        HF_CACHE = Path.home() / ".cache" / "hf_models"
        HF_CACHE.mkdir(parents=True, exist_ok=True)

        model_dir = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(HF_CACHE),
            local_files_only=False,
        )

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

    def forward(self, pep: torch.Tensor) -> torch.Tensor:
        if pep.dim() != 4 or pep.size(1) != 1:
            raise ValueError("pep must be a 4D tensor with shape [B,1,H,W].")

        # SegFormer expects 3 channels
        x = pep.repeat(1, 3, 1, 1)  # [B,3,H,W]

        outputs = self.segformer(pixel_values=x)
        logits_lowres = outputs.logits  # [B,num_labels,h,w]

        logits = F.interpolate(
            logits_lowres,
            size=pep.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        
        return logits


class SegFormerB2PEPRunner(ModelRunner):
    """
    SegFormer-B2 runner compatible with torchtools' training/testing loops.

    Responsibilities:
        - Build the SegFormer PEP model
        - Optionally load a training checkpoint
        - Optionally wrap with nn.DataParallel
        - Provide a 'logits(...)' method returning a ModelOutput

    Args:
        load_path: Optional path to a checkpoint saved by torchtools' train_fn.
                   Expected format: {"state_dict": ...} or a raw state_dict.
        use_data_parallel: If True and multiple GPUs are available, wraps the model in DataParallel.
    """

    def __init__(self, load_path: Optional[str] = None, use_data_parallel: bool = True):
        model = SegFormerB2PepBackbone(num_labels=1)

        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            print("... Loading pre-trained weights.")
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict)
            print("Done.")

        if use_data_parallel and torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                print(f"[SegFormerB2PEPRunner] Using DataParallel on {n_gpus} GPUs")
                model = nn.DataParallel(model)

        self.model_ = model

    @property
    def model(self) -> nn.Module:
        return self.model_

    def logits(
        self,
        batch: dict,
        device: torch.device,
        loss_fn: Optional[nn.Module] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Computes logits (and loss, if provided) for a given batch.

        Expected batch keys:
            - "pep":  [B,1,H,W] float tensor
            - "mask": [B,1,H,W] or [B,H,W] long/float tensor (binary or ignore_index-aware)

        Returns:
            ModelOutput with fields: logits, labels, loss
        """
        
        if not {"pep", "mask"} <= set(batch.keys()):
            raise KeyError('Batch must contain keys {"pep","mask"}.')

        pep = batch["pep"].to(device)
        labels = batch["mask"].to(device)

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)  # [B,1,H,W]

        logits = self.model(pep).float()

        loss = loss_fn(logits, labels) if loss_fn is not None else None

        return ModelOutput(
            loss=loss,
            logits=logits,
            labels=labels,
        )