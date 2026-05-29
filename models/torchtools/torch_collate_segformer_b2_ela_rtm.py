import torch
import torch.nn.functional as F


def pad_collate(batch, ignore_index=-1):
    """
    Collate function for variable-size ELA crops.

    Each batch item is expected to be:
        (block, meta)

    where:
        block.ela  -> tensor [3, H, W]
        block.mask -> tensor [H, W]
        meta       -> dict

    Returns:
        dict with:
            ela  -> [B, 3, H_max, W_max]
            mask -> [B, 1, H_max, W_max]
            meta -> list[dict]
    """
    
    blocks, metas = zip(*batch)

    # Use ELA as the spatial reference for batch padding.
    max_h = max(b.ela.shape[-2] for b in blocks)
    max_w = max(b.ela.shape[-1] for b in blocks)

    def _pad(t: torch.Tensor, pad_value: float) -> torch.Tensor:
        """
        Pad a tensor on the bottom and right up to (max_h, max_w).
        """
        
        h, w = t.shape[-2], t.shape[-1]
        pad = (0, max_w - w, 0, max_h - h)  # (left, right, top, bottom)
        return F.pad(t, pad, value=pad_value)

    # Stack ELA tensors into [B, 3, H_max, W_max].
    ela_list = []
    for b in blocks:
        ela = b.ela
        if ela.dim() == 2:
            ela = ela.unsqueeze(0)  # fallback for unexpected single-channel input
        ela_list.append(_pad(ela, 0.0))

    # Stack masks into [B, 1, H_max, W_max].
    mask_list = [_pad(b.mask.unsqueeze(0), ignore_index) for b in blocks]

    return {
        "ela": torch.stack(ela_list, dim=0).float(),
        "mask": torch.stack(mask_list, dim=0).long(),
        "meta": list(metas),
    }