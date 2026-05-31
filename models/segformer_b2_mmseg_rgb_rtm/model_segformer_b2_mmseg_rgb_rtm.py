from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torchtools.model import ModelRunner, ModelOutput

from models.RTMSegformer.models.backbone.mix_transformer import MixVisionTransformer
from models.RTMSegformer.models.decode_heads.segformer_head import SegformerHead


def mmseg_rgb_normalize_01(x01: torch.Tensor) -> torch.Tensor:
    """
    Normalize RGB inputs using MMSEG SegDataPreProcessor statistics.

    The input is expected to be in [0, 1], while MMSEG mean/std values are
    defined in the 0..255 scale, so they are converted to the 0..1 scale here.
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
    """
    Load MIT-B2 pretrained weights into the MMSEG backbone only.

    The checkpoint may contain a full segmentation model. Only keys prefixed
    with 'backbone.' are loaded into the backbone.
    """
    
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", None) or ckpt.get("model", None) or ckpt
    else:
        state_dict = ckpt

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            cleaned[key[len("backbone."):]] = value

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)

    print(
        f"[mit_b2] loaded backbone-only. "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

    if unexpected:
        print("  unexpected examples:", unexpected[:10])

    if missing:
        print("  missing examples:", missing[:10])


class SegFormerRGB_MMSEG(nn.Module):
    """
    MMSEG-style SegFormer-B2 model for RGB binary segmentation.

    Input:
        rgb01: [B, 3, H, W] in [0, 1]

    Output:
        logits: [B, num_classes, H, W]
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone_ckpt: str = "/home/guests3/rma/tese/codigo_tese/detect_forgery_documents/opcao_5/models/RTMSegformer/configs/segformer/pretrain/mit_b2.pth",
        channels: int = 256,
        drop_path_rate: float = 0.1,
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

        self.decode_head = SegformerHead(
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=channels,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
        )

        if backbone_ckpt is not None:
            load_mit_b2_backbone_weights(self.backbone, backbone_ckpt)

    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RGB input.

        Args:
            rgb01: RGB tensor in [0, 1], shape [B, 3, H, W]

        Returns:
            Segmentation logits resized to the input spatial resolution.
        """
        
        x = torch.clamp(rgb01, 0, 1)
        x = mmseg_rgb_normalize_01(x)

        feats = self.backbone(x)
        logits_low = self.decode_head(feats)

        logits = F.interpolate(
            logits_low,
            size=rgb01.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class SegFormerMMSEG_RGBRunner(ModelRunner):
    """
    Runner wrapper for the MMSEG-style SegFormer RGB model.
    """

    def __init__(
        self,
        load_path: Optional[str] = None,
        backbone_ckpt: str = "/home/guests3/rma/tese/codigo_tese/detect_forgery_documents/opcao_5/models/RTMSegformer/configs/segformer/pretrain/mit_b2.pth",
        num_classes: int = 1,
        use_data_parallel: bool = False,
    ):
        super().__init__()

        self.model_ = SegFormerRGB_MMSEG(
            num_classes=num_classes,
            backbone_ckpt=backbone_ckpt,
        )

        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            # remove the "module." prefix if the checkpoint was saved with DataParallel
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {
                    key.replace("module.", "", 1): value
                    for key, value in state_dict.items()
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
        """
        Compute logits and optional loss from a batch.

        Expected batch keys:
            - image: [B, 3, H, W]
            - mask:  [B, H, W] or [B, 1, H, W]
        """
        
        required = {"image", "mask"}
        if not required <= set(batch.keys()):
            raise KeyError(f"Batch must contain {required} (got {set(batch.keys())})")

        rgb = torch.clamp(batch["image"].to(device), 0.0, 1.0)
        logits = self.model(rgb)

        labels = batch["mask"].to(device)

        if labels.dim() == 4 and labels.shape[1] == 1:
            pass
        elif labels.dim() == 3:
            labels = labels.unsqueeze(1)
        else:
            raise ValueError(f"Expected mask [B,H,W] or [B,1,H,W], got {labels.shape}")

        valid = labels != -1

        labels_bin = labels.clone()
        labels_bin[valid] = (labels_bin[valid] > 0).long()
        labels = labels_bin

        loss = loss_fn(logits, labels) if loss_fn is not None else None

        return ModelOutput(
            logits=logits,
            labels=labels,
            loss=loss,
        )