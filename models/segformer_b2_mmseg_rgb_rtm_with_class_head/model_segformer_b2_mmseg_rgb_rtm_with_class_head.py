from typing import Optional, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torchtools.model import ModelRunner, ModelOutput

from models.RTMSegformer.models.backbone.mix_transformer import MixVisionTransformer
from models.RTMSegformer.models.decode_heads.segformer_head import SegformerHead


def mmseg_rgb_normalize_01(x01: torch.Tensor) -> torch.Tensor:
    """
    MMSEG-style RGB normalization for inputs in [0, 1].

    MMSEG uses ImageNet mean and std in the 0..255 scale.
    Here they are converted to the 0..1 scale.
    """
    
    mean = torch.tensor(
        [123.675, 116.28, 103.53],
        device=x01.device,
        dtype=x01.dtype,
    ) / 255.0

    std = torch.tensor(
        [58.395, 57.12, 57.375],
        device=x01.device,
        dtype=x01.dtype,
    ) / 255.0

    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)

    return (x01 - mean) / std


def load_mit_b2_backbone_weights(backbone: nn.Module, ckpt_path: str) -> None:
    """Load only the MIT-B2 backbone weights from an MMSEG-style checkpoint."""

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", None) or ckpt.get("model", None) or ckpt
    else:
        sd = ckpt

    cleaned = {}

    for k, v in sd.items():
        if k.startswith("backbone."):
            # Remove the "backbone." prefix and ignore all non-backbone weights.
            cleaned[k[len("backbone."):]] = v

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)

    print(
        f"[mit_b2] loaded backbone-only. "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

    if unexpected:
        print("  unexpected examples:", unexpected[:10])

    if missing:
        print("  missing examples:", missing[:10])


class SegFormerBinaryClassifier(nn.Module):
    """MMSEG-style SegFormer-B2 binary image-level classifier."""
    
    def __init__(
        self,
        backbone_ckpt: str = "/home/guests3/rma/tese/codigo_tese/detect_forgery_documents/opcao_5/models/RTMSegformer/configs/segformer/pretrain/mit_b2.pth",
        drop_path_rate: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone = MixVisionTransformer(
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            norm_cfg=dict(type="LN", eps=1e-6),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),          # [B, 512, 1, 1] -> [B, 512]
            nn.Dropout(head_dropout),
            nn.Linear(512, 1),     # binary logit
        )

        if backbone_ckpt is not None:
            load_mit_b2_backbone_weights(self.backbone, backbone_ckpt)

    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        """Run the classifier and return one binary logit per image."""

        x = torch.clamp(rgb01, 0, 1)
        x = mmseg_rgb_normalize_01(x)

        feats = self.backbone(x)       # feature list: [c1, c2, c3, c4]
        last_feat = feats[-1]          # last feature map: [B, 512, H', W']

        logits = self.classifier(last_feat)  # [B, 1]

        return logits.squeeze(1)       # [B]


class SegFormerBinaryClassificationRunner(ModelRunner):
    """ModelRunner wrapper for the SegFormer-B2 binary classification model."""

    def __init__(
        self,
        load_path: Optional[str] = None,
        backbone_ckpt: str = "/home/guests3/rma/tese/codigo_tese/detect_forgery_documents/opcao_5/models/RTMSegformer/configs/segformer/pretrain/mit_b2.pth",
        use_data_parallel: bool = False,
    ):
        super().__init__()

        self.model_ = SegFormerBinaryClassifier(
            backbone_ckpt=backbone_ckpt,
        )

        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("module.", "", 1): v
                    for k, v in state_dict.items()
                }

            print("... Loading trained weights")
            self.model_.load_state_dict(state_dict, strict=False)
            print("Done.")

        if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model_ = nn.DataParallel(self.model_)

    @property
    def model(self):
        """Return the underlying PyTorch model."""

        return self.model_

    @model.setter
    def model(self, value: nn.Module):
        """Set the underlying PyTorch model."""
        
        self.model_ = value

    def logits(self, batch, device, loss_fn=None, **kwargs):
        """Run one batch and return image-level logits, labels, and optionally the loss."""
    
        required = {"image", "label"}
    
        if not required <= set(batch.keys()):
            raise KeyError(
                f"Batch must contain {required} (got {set(batch.keys())})"
            )
    
        rgb = torch.clamp(batch["image"].to(device), 0.0, 1.0)  # [B, 3, H, W]
        logits = self.model(rgb)                                # [B]
    
        labels = batch["label"].to(device).float().view(-1)      # [B]
    
        loss = loss_fn(logits, labels) if loss_fn is not None else None
    
        return ModelOutput(
            logits=logits,
            labels=labels,
            loss=loss,
        )