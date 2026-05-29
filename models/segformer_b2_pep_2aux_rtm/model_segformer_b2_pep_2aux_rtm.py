from typing import Optional
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import constants as hf_const
from huggingface_hub import snapshot_download
from transformers import SegformerForSemanticSegmentation

from models.torchtools.model import ModelRunner, ModelOutput


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images with ImageNet statistics.
    Expects x with shape [B, 3, H, W] in [0, 1].
    """
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class SegFormerPEPAuxBackbone(nn.Module):
    """
    SegFormer-B2 backbone for PEP-based binary segmentation with two auxiliary channels.

    Inputs:
        pep: [B, 1, H, W] in [0, 1]
        aux: [B, 2, H, W] in [0, 1]

    Output:
        logits: [B, 1, H, W]
    """

    def __init__(self, num_labels: int = 1):
        super().__init__()

        # Project the 3-channel input (PEP + auxiliary channels) to the format expected by SegFormer.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
        )

        print("=== HF DEBUG ===")
        print("cwd:", os.getcwd())
        print("HOME:", os.environ.get("HOME"))
        print("HF_HOME:", os.environ.get("HF_HOME"))
        print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
        print("HF_HUB_CACHE (const):", hf_const.HF_HUB_CACHE)
        print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE"))
        print("TRANSFORMERS_OFFLINE:", os.environ.get("TRANSFORMERS_OFFLINE"))
        print("cache exists?", Path(hf_const.HF_HUB_CACHE).exists())
        print("================")

        model_id = "nvidia/mit-b2"

        # Controlled local cache for Hugging Face model downloads.
        hf_cache = Path.home() / ".cache" / "hf_models"
        hf_cache.mkdir(parents=True, exist_ok=True)

        model_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=str(hf_cache),
            local_files_only=False,
        )

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

        print("[SegFormer] num_labels:", self.segformer.config.num_labels)
        print(
            "[SegFormer] classifier weight:",
            tuple(self.segformer.decode_head.classifier.weight.shape),
        )
        
    def forward(self, pep: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PEP input and auxiliary channels.

        Args:
            pep: [B, 1, H, W]
            aux: [B, 2, H, W]

        Returns:
            Segmentation logits resized to the original spatial resolution.
        """
        assert pep.dim() == 4 and pep.shape[1] == 1, pep.shape
        assert aux.dim() == 4 and aux.shape[1] == 2, aux.shape

        x = torch.cat([pep, aux], dim=1)  # [B, 3, H, W]
        x = torch.clamp(x, 0.0, 1.0)
        x = self.stem(x)
        x = imagenet_normalize(x)

        outputs = self.segformer(pixel_values=x)
        logits_low = outputs.logits

        # Upsample decoder output back to the input crop size.
        logits = F.interpolate(
            logits_low,
            size=pep.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


class SegFormerPEPAuxRunner(ModelRunner):
    """
    Runner wrapper for SegFormer with PEP input and internally derived auxiliary channels.
    """

    def __init__(self, load_path: Optional[str] = None, use_data_parallel: bool = False):
        super().__init__()
        self.model_ = SegFormerPEPAuxBackbone(num_labels=1)

        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            # Remove the "module." prefix if the checkpoint was saved from DataParallel.
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

            print("... Loading weights")
            self.model_.load_state_dict(state_dict, strict=False)
            print("Done.")

        if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model_ = nn.DataParallel(self.model_)

    @property
    def model(self):
        return self.model_

    @model.setter
    def model(self, value: nn.Module):
        self.model_ = value
        
    @staticmethod
    def _compute_hp_edge_from_pep(pep_01: torch.Tensor) -> torch.Tensor:
        """
        Derive two auxiliary channels from PEP:
        - high-pass response
        - Sobel edge magnitude

        Args:
            pep_01: [B, 1, H, W] in [0, 1]

        Returns:
            aux: [B, 2, H, W] in [0, 1]
        """
        assert pep_01.dim() == 4 and pep_01.shape[1] == 1, pep_01.shape

        # High-pass response: pep - local average.
        blur = F.avg_pool2d(pep_01, kernel_size=7, stride=1, padding=3)
        hp = pep_01 - blur
        hp = torch.clamp(hp, -0.5, 0.5)
        hp_01 = hp + 0.5

        # Sobel edge magnitude on the PEP map.
        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=pep_01.device,
            dtype=pep_01.dtype,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            device=pep_01.device,
            dtype=pep_01.dtype,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(pep_01, kx, padding=1)
        gy = F.conv2d(pep_01, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        edge_01 = torch.clamp(mag / 4.0, 0.0, 1.0)

        return torch.cat([hp_01, edge_01], dim=1)

    def logits(self, batch, device, loss_fn=None, **kwargs):
        """
        Compute logits and optional loss from a batch.

        Expected batch keys:
            - "pep": [B, 1, H, W]
            - "mask": [B, H, W] or [B, 1, H, W]
        """
        required = {"pep", "mask"}
        if not required <= set(batch.keys()):
            raise KeyError(f"Batch must contain {required}")

        pep = torch.clamp(batch["pep"].to(device), 0.0, 1.0)

        labels = batch["mask"].to(device)
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        aux = self._compute_hp_edge_from_pep(pep)

        logits = self.model(pep, aux)
        loss = loss_fn(logits, labels) if loss_fn is not None else None

        return ModelOutput(
            logits=logits,
            labels=labels,
            loss=loss,
        )