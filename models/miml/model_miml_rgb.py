""" MIML model implementation using RGB only"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torchtools.model import ModelRunner, ModelOutput

from models.hrnet.model_hrnet_rgb import HRNetRGBEncoder


class MIMLHead(nn.Module):
    """
    MIML head (inspired by Qu et al.) for multi-instance learning.

    This module receives a sequence of instances (patches) and produces
    a tampering score per instance.

    Input:
        x_seq: Tensor of shape [B, N, C]

    Output:
        logits: Tensor of shape [B, N]
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(in_dim, 1)  # tampering score per instance

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() != 3:
            raise ValueError("x_seq must be a 3D tensor with shape [B, N, C].")

        h = self.transformer(x_seq)              # [B, N, C]
        logits = self.classifier(h).squeeze(-1)  # [B, N]
        return logits


class MIMLRGBBackbone(nn.Module):
    """
    MIML backbone for RGB-based forgery segmentation.

    Pipeline:
        RGB image [B,3,H,W]
          -> HRNet RGB encoder [B,C,Hf,Wf]
          -> patchification (mean pooling per patch) [B,N,C]
          -> MIML Transformer producing patch-level scores [B,N]
          -> coarse logit map [B,1,Hp,Wp]
          -> upsample to encoder resolution [B,1,Hf,Wf]
          -> upsample to original resolution [B,1,H,W]

    Args:
        patch_size: Patch size in the encoder feature space.
        feat_dim: Feature dimension C expected from the encoder output.
    """

    def __init__(self, patch_size: int = 4, feat_dim: int = 360):
        super().__init__()
        self.encoder = HRNetRGBEncoder()
        self.miml_head = MIMLHead(in_dim=feat_dim)
        
        self.patch_size = patch_size
        self.feat_dim = feat_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError("image must be a 4D tensor with shape [B,3,H,W].")
    
        feat = self.encoder(image)
        if feat.dim() != 4:
            raise ValueError("Encoder output must be a 4D tensor [B,C,Hf,Wf].")

        B, C, Hf, Wf = feat.shape
        if C != self.feat_dim:
            raise ValueError(
                f"Expected encoder channels C={self.feat_dim}, but got C={C}."
            )

        p = self.patch_size
        if p <= 0:
            raise ValueError("patch_size must be a positive integer.")

        # Ensure Hf and Wf are divisible by patch_size
        H_pad = (p - (Hf % p)) % p
        W_pad = (p - (Wf % p)) % p
        if H_pad > 0 or W_pad > 0:
            feat = F.pad(feat, (0, W_pad, 0, H_pad))
            B, C, Hf, Wf = feat.shape

        # Patchification: [B,C,Hf,Wf] -> [B,N,C]
        feat_unfold = feat.unfold(2, p, p).unfold(3, p, p)  # [B,C,Hp,Wp,p,p]
        Hp, Wp = feat_unfold.shape[2], feat_unfold.shape[3]

        # [B,C,Hp,Wp,p,p] -> [B,C,N,p,p]
        feat_unfold = feat_unfold.contiguous().view(B, C, Hp * Wp, p, p)

        # Spatial mean per patch: [B,C,N]
        feat_patches = feat_unfold.mean(dim=[3, 4])

        # [B,C,N] -> [B,N,C]
        feat_patches = feat_patches.permute(0, 2, 1)

        # Patch-level MIML scores: [B,N]
        patch_logits = self.miml_head(feat_patches)

        # [B,N] -> [B,1,Hp,Wp]
        patch_logits = patch_logits.view(B, 1, Hp, Wp)

        # Upsample coarse map to encoder resolution
        logits_coarse = F.interpolate(
            patch_logits,
            size=(Hf, Wf),
            mode="nearest",
        )

        # Upsample to original input resolution
        logits = F.interpolate(
            logits_coarse,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class MIMLRGBRunner(ModelRunner):
    """
    Runner similar to the others, but with a MIML+HRNet backbone.
    """

    def __init__(self, load_path: Optional[str] = None, use_data_parallel: bool = True):
        model = MIMLRGBBackbone(patch_size=4, feat_dim=360)

        if load_path is not None:
            ckpt = torch.load(load_path, map_location="cpu")
            print("... Loading pre-trained weights.")
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict)
            print("Done.")

        if use_data_parallel and torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                print(f"[MIMLRGBRunner] Using DataParallel on {n_gpus} GPUs")
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
        Computes logits and, optionally, the loss for a given batch.
        
        Expected batch keys:
            - "image": [B,3,H,W] float tensor
            - "mask":  [B,1,H,W] or [B,H,W] tensor with binary labels and optional ignore index
        
        Returns:
            ModelOutput with fields: logits, labels, loss
        """
        
        if not {"image", "mask"} <= set(batch.keys()):
            raise KeyError('Batch must contain keys {"image","mask"}.')
        
        image = batch["image"].to(device)
        labels = batch["mask"].to(device)
        
        logits = self.model(image).float()

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        
        loss = loss_fn(logits, labels) if loss_fn is not None else None

        return ModelOutput(
            loss=loss,
            logits=logits,
            labels=labels,
        )