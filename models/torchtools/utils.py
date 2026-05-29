""" Utilitary functions """

from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class AverageMeter:

    """
    Tracks the values of a given metric
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def display_image(img: np.array, figsize: Tuple[int, int] = (20, 20), cmap: Optional[str] = None):
    """
    Display array

    Note: cmap="gray" for grayscale images 
    """
    plt.figure(figsize=figsize)
    plt.axis("off")
    if cmap is not None:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    

def display_tensor(tensor: torch.Tensor, figsize: Tuple[int, int] = (20, 20), cmap: Optional[str] = None):
    """
    Display torch Tensor
    """
    if len(tensor.shape) == 3:  # RGB image
        img = tensor.numpy().transpose([1, 2, 0])
    else:
        img = tensor.numpy()

    display_image(img, figsize, cmap)


def train_validation_split(dataset: Dataset, validation_ratio: float = 0.1, seed: int = 12345) -> Tuple[Dataset, Dataset]:
    """
    Train validation split of dataset
    """
    train_length = int(len(dataset) * (1 - validation_ratio))

    train_ds, validation_ds = torch.utils.data.random_split(
        dataset, 
        [train_length, (len(dataset) - train_length)],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_ds, validation_ds


def binary_mask_from_logits(
    logits: torch.Tensor, 
    target_size: Optional[Tuple[int, int]] = None, 
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Get the predicted binary mask from logits

    Parameters:
        logits: model predictions before any final activation
        target_size: target size for the mask (should match the label size)
        threshold: minimum value to consider has positive
    """

    assert 0 <= threshold <= 1
    
    if target_size is not None:
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        upsampled_logits = logits
    
    predicted_probs = torch.sigmoid(upsampled_logits)
    predicted = torch.where(predicted_probs > threshold, 1, 0)

    return predicted
    