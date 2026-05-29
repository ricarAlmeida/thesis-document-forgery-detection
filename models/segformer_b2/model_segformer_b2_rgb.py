""" SegFormer-B2 model implementation for learning RGB features """

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation

from models.torchtools.model import ModelRunner, ModelOutput

from huggingface_hub import snapshot_download


class SegFormerB2RGBBackbone(nn.Module):
    """
    SegFormer-B2 backbone for RGB-based forgery segmentation.

    This module consumes RGB image crops and predicts a single-channel
    segmentation logit map.

    Args:
        num_labels: Number of output labels. For binary segmentation this is 1.

    Input:
        image: Tensor of shape [B, 3, H, W]

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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError("image must be a 4D tensor with shape [B,3,H,W].")

        outputs = self.segformer(pixel_values=image)
        logits_lowres = outputs.logits

        logits = F.interpolate(
            logits_lowres,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        
        return logits


class SegFormerB2RGBRunner(ModelRunner):
    """
    SegFormer-B2 runner compatible with torchtools' training/testing loops.

    Expected batch keys:
        - "image": [B,3,H,W] float tensor
        - "mask":  [B,1,H,W] or [B,H,W] tensor
    """

    def __init__(self, load_path: Optional[str] = None, use_data_parallel: bool = True):
        model = SegFormerB2RGBBackbone(num_labels=1)

        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            print("... Loading pre-trained weights.")
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict)
            print("Done.")

        if use_data_parallel and torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                print(f"[SegFormerB2RGBRunner] Using DataParallel on {n_gpus} GPUs")
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

        if not {"image", "mask"} <= set(batch.keys()):
            raise KeyError('Batch must contain keys {"image","mask"}.')

        image = batch["image"].to(device)
        labels = batch["mask"].to(device)

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        logits = self.model(image).float()

        loss = loss_fn(logits, labels) if loss_fn is not None else None

        return ModelOutput(
            loss=loss,
            logits=logits,
            labels=labels,
        )