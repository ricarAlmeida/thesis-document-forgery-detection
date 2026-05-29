import torch
import torch.nn.functional as F


def pad_collate(batch, ignore_index=-1):
    """
    Collate function for variable-size RGB, ELA, and PEP crops.

    Each batch item is expected to be:
        (block, meta)

    where:
        block.image -> tensor [3, H, W]
        block.ela   -> tensor [3, H, W]
        block.pep   -> tensor [1, H, W]
        block.mask  -> tensor [H, W]
        meta        -> dict

    Returns:
        dict with:
            image -> [B, 3, H_max, W_max]
            ela   -> [B, 3, H_max, W_max]
            pep   -> [B, 1, H_max, W_max]
            mask  -> [B, 1, H_max, W_max]
            meta  -> list[dict]
    """
    
    blocks, metas = zip(*batch)

    # Use RGB image size as the padding reference.
    max_h = max(b.image.shape[-2] for b in blocks)
    max_w = max(b.image.shape[-1] for b in blocks)

    def _pad(t: torch.Tensor, pad_value: float) -> torch.Tensor:
        """
        Pad a tensor on the bottom and right up to (max_h, max_w).
        """
        
        h, w = t.shape[-2], t.shape[-1]
        pad = (0, max_w - w, 0, max_h - h)  # (left, right, top, bottom)
        return F.pad(t, pad, value=pad_value)

    return {
        "image": torch.stack([_pad(b.image, 0.0) for b in blocks], dim=0),
        "ela": torch.stack([_pad(b.ela, 0.0) for b in blocks], dim=0),
        "pep": torch.stack([_pad(b.pep, 0.0) for b in blocks], dim=0),
        "mask": torch.stack(
            [_pad(b.mask.unsqueeze(0), ignore_index) for b in blocks],
            dim=0,
        ).long(),
        "meta": list(metas),
    }