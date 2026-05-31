from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torchtools.model import ModelRunner, ModelOutput

from models.RTMSegformer.models.backbone.mix_transformer import MixVisionTransformer
from models.RTMSegformer.models.decode_heads.segformer_head import SegformerHead


def mmseg_rgb_normalize_01(x01: torch.Tensor) -> torch.Tensor:
    """Normalize RGB images in [0, 1] using the MMSEG ImageNet mean and std."""

    mean = torch.tensor(
        [123.675, 116.28, 103.53],
        device=x01.device,
        dtype=x01.dtype,
    ).view(1, 3, 1, 1) / 255.0

    std = torch.tensor(
        [58.395, 57.12, 57.375],
        device=x01.device,
        dtype=x01.dtype,
    ).view(1, 3, 1, 1) / 255.0

    return (x01 - mean) / std


def load_mit_b2_backbone_weights(
    backbone: nn.Module,
    ckpt_path: str,
) -> None:
    """Load only the MIT-B2 backbone weights from an MMSEG-style checkpoint."""

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", None) or ckpt.get("model", None) or ckpt
    else:
        state_dict = ckpt

    backbone_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("backbone."):
            clean_key = key[len("backbone."):]
            backbone_state_dict[clean_key] = value

    missing, unexpected = backbone.load_state_dict(
        backbone_state_dict,
        strict=False,
    )

    print(
        f"[mit_b2] Loaded backbone-only weights. "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

    if missing:
        print("  missing examples:", missing[:10])

    if unexpected:
        print("  unexpected examples:", unexpected[:10])


class SegFormerRGB_MMSEG(nn.Module):
    """MMSEG-style SegFormer-B2 with optional segmentation and image-level branches."""

    def __init__(
        self,
        num_classes: int = 1,
        backbone_ckpt: Optional[str] = "/home/guests3/rma/tese/codigo_tese/detect_forgery_documents/opcao_5/models/RTMSegformer/configs/segformer/pretrain/mit_b2.pth",
        image_only: bool = False,
        channels: int = 256,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.image_only = image_only

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
                use_sigmoid=True,
                loss_weight=1.0,
            ),
        )

        self.cls_refine = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        if backbone_ckpt is not None:
            load_mit_b2_backbone_weights(
                backbone=self.backbone,
                ckpt_path=backbone_ckpt,
            )

    def forward(
        self,
        rgb01: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Run the model and return segmentation logits and image-level logits."""

        x = torch.clamp(rgb01, 0.0, 1.0)
        x = mmseg_rgb_normalize_01(x)

        features = self.backbone(x)

        f4 = features[-1]                         # [B, 512, h, w]
        cls_features = self.cls_refine(f4)         # [B, 256, h, w]
        cls_features = self.cls_pool(cls_features).flatten(1)  # [B, 256]
        cls_logit = self.cls_head(cls_features).squeeze(1)     # [B]

        if self.image_only:
            return None, cls_logit

        seg_logits_low = self.decode_head(features)
        seg_logits = F.interpolate(
            seg_logits_low,
            size=rgb01.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return seg_logits, cls_logit


def clean_state_dict_prefixes(state_dict: Dict) -> Dict:
    """Remove the DataParallel or DDP 'module.' prefix when it exists."""

    if any(key.startswith("module.") for key in state_dict.keys()):
        return {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    return state_dict


def load_trained_weights(
    model: nn.Module,
    load_path: str,
) -> None:
    """Load trained weights into a model with non-strict matching."""

    ckpt = torch.load(load_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = clean_state_dict_prefixes(state_dict)

    print("... Loading trained weights")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Done.")

    if missing:
        print("  missing examples:", missing[:10])

    if unexpected:
        print("  unexpected examples:", unexpected[:10])


def prepare_binary_mask(
    mask: torch.Tensor,
) -> torch.Tensor:
    """Convert a mask to binary labels while preserving ignore-index pixels."""

    if mask.dim() == 4 and mask.shape[1] == 1:
        labels = mask
    elif mask.dim() == 3:
        labels = mask.unsqueeze(1)
    else:
        raise ValueError(
            f"Expected mask with shape [B,H,W] or [B,1,H,W], got {mask.shape}"
        )

    valid = labels != -1
    labels_bin = labels.clone()
    labels_bin[valid] = (labels_bin[valid] > 0).long()

    return labels_bin
    

class SegFormerRGBRunner(ModelRunner):
    """Runner for the two-branch SegFormer model."""

    def __init__(
        self,
        load_path: Optional[str] = None,
        backbone_ckpt: Optional[str] = "/home/guests3/rma/tese/codigo_tese/detect_forgery_documents/opcao_5/models/RTMSegformer/configs/segformer/pretrain/mit_b2.pth",
        num_classes: int = 1,
        use_data_parallel: bool = False,
    ):
        super().__init__()

        self.model_ = SegFormerRGB_MMSEG(
            num_classes=num_classes,
            backbone_ckpt=backbone_ckpt,
            image_only=False,
        )

        if load_path is not None:
            load_trained_weights(
                model=self.model_,
                load_path=load_path,
            )

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

    def logits(
        self,
        batch: Dict,
        device: torch.device,
        loss_fn=None,
        **kwargs,
    ) -> ModelOutput:
        """Run one batch and return segmentation logits, labels, loss, and cls logits."""

        required_keys = {"image", "mask"}

        if not required_keys <= set(batch.keys()):
            raise KeyError(
                f"Batch must contain {required_keys}, got {set(batch.keys())}"
            )

        rgb = torch.clamp(batch["image"].to(device), 0.0, 1.0)
        labels = prepare_binary_mask(batch["mask"].to(device))

        seg_logits, cls_logit = self.model(rgb)

        loss = None
        if loss_fn is not None:
            loss = loss_fn((seg_logits, cls_logit), labels)

        return ModelOutput(
            logits=seg_logits,
            labels=labels,
            loss=loss,
            cls_logit=cls_logit,
        )


class SegFormerImageOnlyRunner(ModelRunner):
    """Runner for the image-only SegFormer classification model."""

    def __init__(
        self,
        load_path: Optional[str] = None,
        backbone_ckpt: Optional[str] = "/home/guests3/rma/tese/codigo_tese/detect_forgery_documents/opcao_5/models/RTMSegformer/configs/segformer/pretrain/mit_b2.pth",
        num_classes: int = 1,
        use_data_parallel: bool = False,
    ):
        super().__init__()

        self.model_ = SegFormerRGB_MMSEG(
            num_classes=num_classes,
            backbone_ckpt=backbone_ckpt,
            image_only=True,
        )

        if load_path is not None:
            load_trained_weights(
                model=self.model_,
                load_path=load_path,
            )

        if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model_ = nn.DataParallel(self.model_)

    @property
    def model(self) -> nn.Module:
        """Return the underlying PyTorch model."""

        return self.model_

    @model.setter
    def model(self, value: nn.Module):
        self.model_ = value

    def logits(
        self,
        batch: dict,
        device: torch.device,
        loss_fn=None,
        **kwargs,
    ) -> ModelOutput:
        """Run one batch and return image-level logits, labels, and loss."""

        required_keys = {"image", "mask"}

        if not required_keys <= set(batch.keys()):
            raise KeyError(
                f"Batch must contain {required_keys}, got {set(batch.keys())}"
            )

        rgb = torch.clamp(batch["image"].to(device), 0.0, 1.0)
        labels = prepare_binary_mask(batch["mask"].to(device))

        _, cls_logit = self.model(rgb)

        loss = None
        if loss_fn is not None:
            loss = loss_fn(cls_logit, labels)

        return ModelOutput(
            logits=None,
            labels=labels,
            loss=loss,
            cls_logit=cls_logit,
        )