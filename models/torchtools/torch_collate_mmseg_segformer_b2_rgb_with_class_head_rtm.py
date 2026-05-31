import torch
import torch.nn.functional as F


def pad_collate_classification(batch):
    """
    Collate function for variable-size RGB crops used in binary classification.

    Each batch item is expected to be:
        (block, meta)

    where:
        block.image -> tensor [3, H, W]
        block.label -> scalar binary label {0, 1}
        meta        -> dict

    Returns:
        dict with:
            image -> [B, 3, H_max, W_max]
            label -> [B]
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
        "label": torch.stack([b.label.view(()) for b in blocks], dim=0).float(),
        "meta": list(metas),
    }