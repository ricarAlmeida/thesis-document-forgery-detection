import torch
import torch.nn.functional as F


def pad_collate(batch, ignore_index=-1):
    """
    Collate function for variable-size PEP crops.

    Each batch item is expected to be:
        (block, meta)

    where:
        block.pep  -> tensor [1, H, W] or [H, W]
        block.mask -> tensor [H, W]
        meta       -> dict

    Returns:
        dict with:
            pep  -> [B, 1, H_max, W_max]
            mask -> [B, 1, H_max, W_max]
            meta -> list[dict]
    """
    
    blocks, metas = zip(*batch)

    # Use PEP as the spatial reference for batch padding.
    max_h = max(b.pep.shape[-2] for b in blocks)
    max_w = max(b.pep.shape[-1] for b in blocks)

    def _pad(t: torch.Tensor, pad_value: float) -> torch.Tensor:
        """
        Pad a tensor on the bottom and right up to (max_h, max_w).
        """
        
        h, w = t.shape[-2], t.shape[-1]
        pad = (0, max_w - w, 0, max_h - h)  # (left, right, top, bottom)
        return F.pad(t, pad, value=pad_value)

    # Stack PEP tensors into [B, 1, H_max, W_max].
    pep_list = []
    for b in blocks:
        pep = b.pep
        if pep.dim() == 2:
            pep = pep.unsqueeze(0)  # [1, H, W]
        pep_list.append(_pad(pep, 0.0))

    # Stack masks into [B, 1, H_max, W_max].
    mask_list = [_pad(b.mask.unsqueeze(0), ignore_index) for b in blocks]

    return {
        "pep": torch.stack(pep_list, dim=0).float(),
        "mask": torch.stack(mask_list, dim=0).long(),
        "meta": list(metas),
    }