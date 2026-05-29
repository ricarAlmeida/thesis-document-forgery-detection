""" 
Implementation of the dataset to train and test document forgery localization models
(HRNet-based + RGB only, SegFormer-B2 + RGB only, MIML + RGB only)
"""

from typing import List, Tuple, Optional, Union, Callable, Iterable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from PIL import Image
from torch.utils.data import Dataset
from scipy.stats import norm

import numpy as np
import random
import torch


class Feature(Enum):
    """
    Image Block/Crop possible feature tipologies

    Attributes:
        RGB: RGB image crop
    """
    
    RGB = auto()
    

@dataclass
class BlockValues:
    """
    Image Block/Crop features values (HRNet-based + RGB only | SegFormer-B2 + RGB only | MIML + RGB only)

    Attributes:
        rgb: RGB image crop (3xHxW)
        mask: image's tampering binary semantic segmentation mask (HxW)
        crop_size: prediction crop size (H,W)
        origin: crop origin (row, col)
        grid_crop: True if the crop respects the JPEG 8x8 patches.
    """

    rgb: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    crop_size: Optional[Tuple[int, int]]
    origin: Optional[Tuple[int, int]] = None
    grid_crop: bool = True
    
    def __post_init__(self):

        if self.mask is not None:
            assert self.crop_size is not None
            assert self.mask.shape == self.crop_size

        if self.rgb is not None:
            assert self.crop_size is not None
            assert self.rgb.shape[0] == 3
            assert self.rgb.shape[-2:] == self.crop_size

        if self.grid_crop and self.crop_size is not None:
            assert self.crop_size[0] % 8 == 0
            assert self.crop_size[1] % 8 == 0


class DocForgeryDatasetRGB(Dataset):
    """
    Dataset for HRNet-based, SegFormer-B2 and MIML training/testing using only RGB + mask.
    """

    def __init__(
        self, 
        images_repo: Iterable[Union[Path, str]],
        masks_repo: Optional[Iterable[Union[Path, str]]],
        crop_size: Optional[Tuple[int, int]], 
        features: List[Feature], 
        grid_crop: Optional[bool] = True, 
        filter_func: Callable[[str], bool] = lambda x: True,
        size: Optional[int] = None,
        original_probability: float = 0.0,
        seed: int = 1234,
    ):
        """
        Implementation of the dataset to train and test document forgery localization models
        (HRNet-based + RGB only | SegFormer-B2 + RGB only)

        Attributes:
            images_repo: tampered images repositories
            masks_repo: tampered images ground truth masks repositories. If masks_repo is None,
                        each image's mask is overwritten as a matrix of zeros
            crop_size: crop height and width. H and W must be multiples of 8 when grid_crop=True
            grid_crop: True to crop within 8x8 grid, False to crop anywhere
            features: list of image's features of interest (here: [Feature.RGB])
            filter_func: function to filter out examples
            size: limit the size of the dataset's examples
            original_probability: retrieve crop without tampered pixels with   prob=original_probability
            seed: seed for random operations
        """

        assert features is not None and len(features) > 0
        assert all(f == Feature.RGB for f in features), "This dataset supports only Feature.RGB"
        assert 0.0 <= original_probability <= 0.2

        self.features = features
        
        self.original_probability = original_probability
        self.x_limit = norm.ppf(1 - self.original_probability)

        if grid_crop and crop_size is not None:
            assert crop_size[0] % 8 == 0
            assert crop_size[1] % 8 == 0
        
        self.images_repo = [
            repo if isinstance(repo, Path) else Path(repo)
            for repo in images_repo
        ]

        images : List[Path]= []
        for im_repo in self.images_repo:
            images += (
                list(
                    p for p in im_repo.glob("*.[jpJP][npNP][egEG]")
                    if filter_func(str(p))
                )
            )

        img_names = [str(img) for img in images]
        assert len(set(img_names)) == len(img_names)  # assert no duplicate image names
        self.images = sorted(images, key=lambda x: str(x))  

        if masks_repo is not None:
            self.masks_repo = [
                repo if isinstance(repo, Path) else Path(repo)
                for repo in masks_repo
            ]

            masks : List[Path] = []
            for m_repo in self.masks_repo:
                masks += list(
                    p for p in m_repo.glob("*.png")
                    if filter_func(str(p))
                )

            mask_names = [str(mask) for mask in masks]
            assert len(set(mask_names)) == len(mask_names)  # assert no duplicate mask names

            self.masks = sorted(masks, key=lambda x: str(x))

            assert len(self.images) == len(self.masks)
            assert [img.stem for img in self.images] == [mask.stem for mask in self.masks]
        else:
            self.masks_repo = None
            self.masks = None

        if size:
            rnd = random.Random(seed)
            rnd.shuffle(self.images)
            self.images = self.images[:size]

            if masks_repo is not None:
                rnd = random.Random(seed)
                rnd.shuffle(self.masks)
                self.masks = self.masks[:size]
                assert [img.stem for img in self.images] == [mask.stem for mask in self.masks]
            
        self._crop_size = crop_size
        self._grid_crop = grid_crop
        
    def __len__(self):
        return len(self.images)

    def read_mask(self, path: Path) -> np.array:
        """
        Read a mask and binarize:
          - 0 -> 0 (background / original)
          - qualquer valor > 0 -> 1 (tampered)
        """
        
        mask = np.array(Image.open(str(path))).astype(np.int16)
        mask = (mask > 0).astype(np.int16)
        return mask

    @staticmethod
    def crop_image_and_mask(
        image_path: str,
        mask: Optional[np.array],
        crop_size: Optional[Tuple[int, int]],
        grid_crop: bool,
        x_limit: float,
        ignore_index: int = -1,
    ) -> dict:
        """
        Crop image and respective mask

        Parameters:
            image_path: path of the .jpeg image file of interest
            mask: ground-truth mask
            crop_size: prediction crop size
            grid_crop: True to respect divisibility by 8x8 JPEG blocks
            x_limit: if a random value from a normal distribution with mean=0 and std=1
                     is higher than x_limit then crop without manipulated pixels
            ignore_index: index for mask padding
        """

        image = np.array(Image.open(image_path))

        try:
            h, w, c = image.shape
        except ValueError: # occurs when image has not 3 channels
                h_p, w_p = image.shape
                image = np.repeat(image[..., np.newaxis], 3, axis=2)  # quasi-RGB
                h, w, c = image.shape
                assert h_p == h 
                assert w_p == w
            
        assert c == 3 # image must have 3 channels

        if mask is None:
            mask = np.zeros((h, w))

        if crop_size is None:
            if grid_crop:
                crop_size = ((h // 8) * 8, (w // 8) * 8)
            else:
                pass  # use entire image. No crop and no pad

        if crop_size is not None:
            
            # Pad if crop_size is larger than image size
            if h < crop_size[0] or w < crop_size[1]:
                # pad img_RGB
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 255.0)
                temp[:image.shape[0], :image.shape[1], :] = image
                image = temp

                # pad mask
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1])), ignore_index)
                temp[:mask.shape[0], :mask.shape[1]] = mask
                mask = temp

            # Determine where to crop
            h_diff =  max(h - crop_size[0], 0) 
            w_diff = max(w - crop_size[1], 0)
            
            # Determine the position of tampered pixels
            mask_locations = np.argwhere(mask == 1)

            position_noise_h = np.random.randint(50, 91)
            position_noise_w = np.random.randint(50, 91)

            if mask_locations.shape[0] <= 0:
                h_center = np.random.randint(0, h_diff + 1)
                w_center = np.random.randint(0, w_diff + 1)
            else:
                if np.random.normal(loc=0.0, scale=1.0) > x_limit:
                    # Try not to include any tampering pixel if possible
                    print("... Trying to crop without tampering pixels")

                    h_center, w_center = sorted(
                        sorted(mask_locations, key=lambda x: x[0]),
                        key=lambda x: x[1],
                    )[-1]

                    h_center = min(h_diff, h_center + position_noise_h)
                    w_center = min(w_diff, w_center + position_noise_w)
                else:
                    h_center, w_center = mask_locations[
                        np.random.randint(0, mask_locations.shape[0])
                    ]

                    h_center = min(h_diff, max(h_center - position_noise_h, 0))
                    w_center = min(w_diff, max(w_center - position_noise_w, 0))
        
            if grid_crop:
                s_r = (h_center // 8) * 8
                s_c = (w_center // 8) * 8
            else:
                s_r = h_center
                s_c = w_center

            # crop img_RGB
            image = image[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1], :]

            # crop mask
            mask = mask[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1]]

        else:
            s_r, s_c = 0, 0

        t_RGB = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float)
        t_mask = torch.tensor(mask, dtype=torch.long)

        return {
            "raw_image": t_RGB,
            "image": (t_RGB - torch.min(t_RGB)) / (torch.max(t_RGB) - torch.min(t_RGB)),
            "mask": t_mask,
            "crop_size": crop_size,
            "origin": (s_r, s_c),
            "grid_crop": grid_crop,
        }

    @staticmethod
    def create_tensor(
        image_path: str, 
        mask: Optional[np.array],
        features: List[Feature], 
        crop_size: Optional[Tuple[int, int]], 
        grid_crop: bool, 
        x_limit: float,
        ignore_index: int = -1,
    ) -> BlockValues:
        """
        Retrieves RGB input and ground-truth mask.
        """
    
        crop_values = DocForgeryDatasetRGB.crop_image_and_mask(
            image_path=image_path,
            mask=mask,
            crop_size=crop_size,
            grid_crop=grid_crop,
            x_limit=x_limit,
            ignore_index=ignore_index,
        )
    
        block_values = BlockValues(
            rgb=crop_values["image"],      # [3,H,W], normalized RGB
            mask=crop_values["mask"],      # [H,W]
            crop_size=crop_values["crop_size"],
            origin=crop_values["origin"],
            grid_crop=crop_values["grid_crop"],
        )
          
        return block_values

    def _create_tensor(
        self, 
        image_path: str,
        mask: Optional[np.array],
    ) -> BlockValues:

        return DocForgeryDatasetRGB.create_tensor(
            image_path=image_path,
            mask=mask,
            features=self.features,
            crop_size = self._crop_size,
            grid_crop=self._grid_crop,
            x_limit=self.x_limit,
        )

    def __getitem__(self, idx: int):
    
        if self.masks is not None:
            assert self.images[idx].stem == self.masks[idx].stem
            mask = self.read_mask(self.masks[idx])
            labels_path = str(self.masks[idx])
        else:
            mask = None
            labels_path = None
        
        image_pil = Image.open(str(self.images[idx]))
        
        if image_pil.format == "PNG":
            values = self._create_tensor(
                image_path=str(self.images[idx]),
                mask=mask,
            )
        else:
            raise Exception("Unhandled image format")

        values_dict = {
            "image_path": str(self.images[idx]),
            "labels_path": labels_path,
            "image": values.rgb,
            "mask": values.mask.unsqueeze(0),      # [1,H,W]
            "crop_size": values.crop_size,
            "origin": values.origin,
            "grid_crop": values.grid_crop,
        }

        return {k: v for k, v in values_dict.items() if v is not None}