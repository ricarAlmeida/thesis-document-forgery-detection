from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import snapshot_download
from transformers import SegformerForSemanticSegmentation

from models.torchtools.model import ModelRunner, ModelOutput


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize input tensors with ImageNet statistics, as expected by SegFormer.
    Expects x in [0, 1] with shape [B, 3, H, W].
    """
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class SegFormerRGBELAPEPAuxBackbone(nn.Module):
    """
    SegFormer-B2 backbone for RGB + ELA + PEP binary segmentation with two auxiliary channels.

    Inputs:
        rgb: [B, 3, H, W] in [0, 1]
        ela: [B, 3, H, W] in [0, 1]
        pep: [B, 1, H, W] in [0, 1]
        aux: [B, 2, H, W] in [0, 1]

    Output:
        logits: [B, 1, H, W]
    """

    def __init__(self, num_labels: int = 1):
        super().__init__()

        # Project the 9-channel input (RGB + ELA + PEP + auxiliary channels) to 3 channels for SegFormer.
        self.stem = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1, bias=False),
        )

        model_id = "nvidia/mit-b2"
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

    def forward(
        self,
        rgb: torch.Tensor,
        ela: torch.Tensor,
        pep: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for RGB, ELA, PEP, and auxiliary inputs.
        """
        
        assert rgb.dim() == 4 and rgb.shape[1] == 3, rgb.shape
        assert ela.dim() == 4 and ela.shape[1] == 3, ela.shape
        assert pep.dim() == 4 and pep.shape[1] == 1, pep.shape
        assert aux.dim() == 4 and aux.shape[1] == 2, aux.shape

        x = torch.cat([rgb, ela, pep, aux], dim=1)  # [B, 9, H, W]
        x = self.stem(x)                            # [B, 3, H, W]
        x = imagenet_normalize(torch.clamp(x, 0.0, 1.0))

        outputs = self.segformer(pixel_values=x)
        logits_low = outputs.logits

        logits = F.interpolate(
            logits_low,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


class SegFormerRGBELAPEPAuxRunner(ModelRunner):
    """
    Runner wrapper for SegFormer with RGB + ELA + PEP input and internally derived auxiliary channels.
    """

    def __init__(self, load_path: Optional[str] = None, use_data_parallel: bool = False):
        super().__init__()
        self.model_ = SegFormerRGBELAPEPAuxBackbone(num_labels=1)

        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            # Remove the "module." prefix if the checkpoint was saved with DataParallel.
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

            self.model_.load_state_dict(state_dict, strict=False)

        if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model_ = nn.DataParallel(self.model_)

    @property
    def model(self):
        return self.model_

    @model.setter
    def model(self, value: nn.Module):
        self.model_ = value

    @staticmethod
    def _compute_hp_edge(rgb_01: torch.Tensor) -> torch.Tensor:
        """
        Derive two auxiliary channels from RGB:
        - high-pass grayscale response
        - Sobel edge magnitude

        Args:
            rgb_01: [B, 3, H, W] in [0, 1]

        Returns:
            aux: [B, 2, H, W] in [0, 1]
        """
        
        assert rgb_01.dim() == 4 and rgb_01.shape[1] == 3, rgb_01.shape

        r, g, b = rgb_01[:, 0:1], rgb_01[:, 1:2], rgb_01[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        blur = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        hp = gray - blur
        hp = torch.clamp(hp, -0.5, 0.5)
        hp_01 = hp + 0.5

        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        edge_01 = torch.clamp(mag / 4.0, 0.0, 1.0)

        return torch.cat([hp_01, edge_01], dim=1)

    def logits(self, batch, device, loss_fn=None, **kwargs):
        """
        Compute logits and optional loss from a batch.

        Expected batch keys:
            - "image": [B, 3, H, W]
            - "ela":   [B, 3, H, W]
            - "pep":   [B, 1, H, W]
            - "mask":  [B, H, W] or [B, 1, H, W]
        """
        
        required = {"image", "ela", "pep", "mask"}
        if not required <= set(batch.keys()):
            raise KeyError(f"Batch must contain {required}")

        rgb = torch.clamp(batch["image"].to(device), 0.0, 1.0)
        ela = torch.clamp(batch["ela"].to(device), 0.0, 1.0)
        pep = torch.clamp(batch["pep"].to(device), 0.0, 1.0)

        labels = batch["mask"].to(device)
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        aux = self._compute_hp_edge(rgb)

        logits = self.model(rgb, ela, pep, aux)
        loss = loss_fn(logits, labels) if loss_fn is not None else None

        return ModelOutput(
            logits=logits,
            labels=labels,
            loss=loss,
        )