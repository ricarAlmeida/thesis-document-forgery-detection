"""Dataset for document forgery localization using PEP features."""

import os
import io
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import utils
import error_potential


def _origins_1d(L: int, win: int, stride: int) -> List[int]:
    """
    Generate 1D sliding-window origins that fully cover an axis.
    Ensures the last window is aligned with the image border.
    """
    
    if L <= win:
        return [0]

    xs = list(range(0, L - win + 1, stride))
    last = L - win
    if xs[-1] != last:
        xs.append(last)
    return xs


def _sliding_origins(H: int, W: int, win: int, stride: int) -> List[Tuple[int, int]]:
    """
    Generate all 2D sliding-window origins for an image of size HxW.
    """
    
    ys = _origins_1d(H, win, stride)
    xs = _origins_1d(W, win, stride)
    return [(y, x) for y in ys for x in xs]


@dataclass
class BlockValues:
    """
    Container for one training/evaluation crop.

    Attributes:
        image: RGB image crop in [0,1], shape [3,H,W]
        pep: PEP tensor in [0,1], shape [1,H,W]
        mask: binary segmentation mask, shape [H,W]
        crop_size: crop height and width
        origin: crop origin in the original image
        grid_crop: whether the crop respects the JPEG 8x8 grid
    """

    image: torch.Tensor
    pep: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    crop_size: Optional[Tuple[int, int]]
    origin: Optional[Tuple[int, int]] = None
    grid_crop: bool = True
    
    def __post_init__(self):
        if self.crop_size is not None and self.image is not None:
            assert self.image.shape[-2:] == self.crop_size
            assert self.image.shape[0] == 3

        if self.pep is not None:
            assert self.pep.shape[-2:] == self.crop_size
            assert self.pep.shape[0] == 1

        if self.mask is not None:
            assert self.mask.shape == self.crop_size

        if self.grid_crop and (self.crop_size is not None):
            assert self.crop_size[0] % 8 == 0
            assert self.crop_size[1] % 8 == 0


def jpeg_recompress_np(img_uint8: np.ndarray, qf: int) -> np.ndarray:
    """
    Recompress an RGB uint8 image with JPEG quality factor qf.
    """
    
    qf = int(np.clip(qf, 1, 100))
    pil = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=qf, subsampling=0, optimize=False)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return np.array(out, dtype=np.uint8)


def jpeg_double_compress_np(img_uint8: np.ndarray, qf1: int, qf2: int) -> np.ndarray:
    """
    Apply global double JPEG compression.
    Useful as augmentation for ELA robustness.
    """
    
    x = jpeg_recompress_np(img_uint8, qf1)
    x = jpeg_recompress_np(x, qf2)
    return x


def build_doc_aug_pipeline(
    h: int = 512,
    w: int = 512,
    p_geom: float = 0.75,
    p_photo: float = 0.60,
    p_scan: float = 0.60,
):
    """
    Build an augmentation pipeline for document crops.

    Includes:
    - geometric transforms
    - photometric transforms
    - scan/camera degradation simulation
    """
    
    geom = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.05,
            rotate_limit=4,
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
            mask_value=0,
            p=0.35,
        ),
        A.Perspective(
            scale=(0.02, 0.05),
            keep_size=True,
            pad_mode=cv2.BORDER_CONSTANT,
            pad_val=(255, 255, 255),
            mask_pad_val=0,
            p=0.25,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.10),
    ], p=1.0)

    photo = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.15, p=0.35),
        A.RandomGamma(gamma_limit=(85, 115), p=0.25),
        A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=0.15),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=10, p=0.10),
    ], p=1.0)

    scan_cam = A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.25),
        A.GaussNoise(var_limit=(5.0, 30.0), mean=0, p=0.25),
        A.Downscale(scale_min=0.65, scale_max=0.90, interpolation=cv2.INTER_LINEAR, p=0.25),
        A.CoarseDropout(
            max_holes=12,
            max_height=int(h * 0.03),
            max_width=int(w * 0.03),
            min_holes=1,
            fill_value=255,
            mask_fill_value=0,
            p=0.15,
        ),
    ], p=1.0)

    return A.Compose(
        [
            A.OneOf([geom, A.NoOp()], p=p_geom),
            A.OneOf([photo, A.NoOp()], p=p_photo),
            A.OneOf([scan_cam, A.NoOp()], p=p_scan),
        ],
        additional_targets={"mask_ignore": "mask"},
    )


class DocForgeryPEPDataset(Dataset):
    """
    Dataset for document forgery localization using PEP as the only auxiliary input.

    Pipeline:
    1. read full image and mask
    2. extract sliding-window crop
    3. optionally apply augmentations
    4. optionally apply JPEG forensic augmentation
    5. compute PEP from the final crop
    """

    QF = 95

    def __init__(
        self,
        images_repo: Iterable[Union[Path, str]],
        masks_repo: Optional[Iterable[Union[Path, str]]],
        crop_size: Tuple[int, int],
        grid_crop: bool = True,
        min_quality_factor: int = 70,
        max_quality_factor: int = 100,
        quality_factor: Optional[int] = None,
        seed: int = 1234,
        indices: Optional[List[int]] = None,
        balance_crops: bool = False,                # True -> training, False -> validation
        p_pos_crop: float = 0.5,                    # 50/50 crops when balance_crops=True
        use_augs: bool = False,
        filter_func: Callable[[str], bool] = lambda x: True,
        size: Optional[int] = None,
        verbose_stats: bool = True,
        stride=None,
    ):
        
        assert 0 < min_quality_factor <= 100
        assert 0 < max_quality_factor <= 100
        assert max_quality_factor >= min_quality_factor
        assert crop_size is not None, "crop_size must be provided, e.g. (512, 512)"
        assert 0.0 <= p_pos_crop <= 1.0

        self.min_quality_factor = int(min_quality_factor)
        self.max_quality_factor = int(max_quality_factor)
        
        self.seed = int(seed)
        self._crop_size = crop_size
        self._grid_crop = grid_crop

        if quality_factor is None:
            self.randomize = True
            self.quality_factor = int(np.random.randint(self.min_quality_factor, self.max_quality_factor + 1))
        else:
            self.randomize = False
            self.quality_factor = int(quality_factor)

        
        win = int(self._crop_size[0])
        self.stride = int(stride) if stride is not None else win

        self.balance_crops = bool(balance_crops)
        self.p_pos_crop = float(p_pos_crop)

        self.use_augs = bool(use_augs)
        self.aug = build_doc_aug_pipeline(h=self._crop_size[0], w=self._crop_size[1])

        self.jpeg_aug_prob = 0.70
        self.jpeg_double_prob = 0.25
        self.jpeg_qf_min = 65
        self.jpeg_qf_max = 100

        self.images_repo = [Path(r) if not isinstance(r, Path) else r for r in images_repo]
        valid_exts: Set[str] = {".jpg", ".jpeg", ".png"}

        images: List[Path] = []
        for im_repo in self.images_repo:
            for p in im_repo.iterdir():
                if p.is_file() and (p.suffix.lower() in valid_exts) and filter_func(str(p)):
                    images.append(p)
        self.images = sorted(images, key=lambda x: str(x))

        if masks_repo is not None:
            self.masks_repo = [Path(r) if not isinstance(r, Path) else r for r in masks_repo]
            masks: List[Path] = []
            for m_repo in self.masks_repo:
                for p in m_repo.iterdir():
                    if p.is_file() and (p.suffix.lower() == ".png") and filter_func(str(p)):
                        masks.append(p)
            self.masks = sorted(masks, key=lambda x: str(x))

            assert len(self.images) == len(self.masks)
            assert [img.stem for img in self.images] == [mask.stem for mask in self.masks]
        else:
            self.masks_repo = None
            self.masks = None

        if indices is not None:
            indices = list(indices)
            assert len(indices) > 0, "indices is empty"
            self.images = [self.images[i] for i in indices]
            if self.masks is not None:
                self.masks = [self.masks[i] for i in indices]

        if size is not None:
            rng = random.Random(self.seed)
            order = list(range(len(self.images)))
            rng.shuffle(order)
            order = order[: int(size)]
            self.images = [self.images[i] for i in order]
            if self.masks is not None:
                self.masks = [self.masks[i] for i in order]

        # Build a sliding-window index over all documents
        self._index = []
        self.pos_windows = []
        self.neg_windows = []

        for doc_i, img_path in enumerate(self.images):
            with Image.open(str(img_path)) as im:
                W, H = im.size

            m = self.read_mask(self.masks[doc_i]) if self.masks is not None else None

            for (y, x) in _sliding_origins(H, W, win=win, stride=self.stride):
                k = len(self._index)
                self._index.append((doc_i, y, x))

                if m is None:
                    self.neg_windows.append(k)
                else:
                    mc = m[y:y+win, x:x+win]
                    is_pos = bool((mc == 1).any())
                    (self.pos_windows if is_pos else self.neg_windows).append(k)

        if verbose_stats:
            print(
                f"[INFO] docs={len(self.images)} windows={len(self._index)} "
                f"pos={len(self.pos_windows)} neg={len(self.neg_windows)} "
                f"balance_crops={self.balance_crops} p_pos_crop={self.p_pos_crop} stride={self.stride}"
            )
            

    def apply_albu_on_crop(self, img_np: np.ndarray, mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply geometric/photometric augmentations to a crop and preserve ignore regions.
        """
        
        if (self.aug is None) or (not self.use_augs):
            return img_np, mask_np
    
        ignore_map = (mask_np == -1).astype(np.uint8)
    
        mask_safe = mask_np.copy()
        mask_safe[mask_safe < 0] = 0
    
        out = self.aug(
            image=img_np,
            mask=mask_safe.astype(np.uint8),
            mask_ignore=ignore_map,
        )
    
        img_aug = out["image"]
        mask_aug = out["mask"].astype(np.int16)
        ignore_aug = out["mask_ignore"].astype(bool)
    
        mask_aug[ignore_aug] = -1
        return img_aug, mask_aug


    def apply_jpeg_forensic_on_crop(self, img_np: np.ndarray) -> np.ndarray:
        """
        Apply JPEG recompression or double JPEG augmentation.
        This helps the model become more robust to compression artifacts.
        """
        
        if not self.use_augs:
            return img_np
    
        if random.random() > self.jpeg_aug_prob:
            return img_np
    
        qf1 = random.randint(self.jpeg_qf_min, self.jpeg_qf_max)
    
        if random.random() < self.jpeg_double_prob:
            qf2 = random.randint(self.jpeg_qf_min, self.jpeg_qf_max)
            if abs(qf2 - qf1) < 5:
                qf2 = int(np.clip(qf2 + 10, 1, 100))
            return jpeg_double_compress_np(img_np, qf1, qf2)
    
        return jpeg_recompress_np(img_np, qf1)


    @staticmethod
    def _save_crop_as_jpeg_tmp(img_np: np.ndarray, qf: int) -> str:
        """
        Save an HWC uint8 crop as a temporary JPEG and return its path.
    
        The caller is responsible for deleting the temporary file.
        """

        tmp_dir = Path("/media/general_storage6/tmp_aux")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            dir=tmp_dir,
            delete=False,
        )
        tmp_path = tmp.name
        tmp.close()
    
        img = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
        img.save(
            tmp_path,
            "JPEG",
            quality=int(qf),
            subsampling=0,
            optimize=False,
        )
    
        return tmp_path
    

    @staticmethod
    def pep_features(
        image_path: Path,
        dct: np.array, 
        qtable: np.array,
        origin: Tuple[int, int],
        crop_size: Optional[Tuple[int, int]],
        qf_pep: int = 95,
    ) -> torch.Tensor:
        """
        Compute probabilistic error potential (PEP) features from a JPEG image by
        dequantizing the input DCT coefficients, recompressing the image with the
        target PEP quality factor, retrieving the corresponding quantization table,
        and building a block-divisibility map over the luminance channel.
    
        The resulting PEP map is optionally padded and cropped to the requested
        region, then normalized and returned as a tensor with shape [1, H, W].
        """

        assert 0 < qf_pep <= 100

        img = Image.open(str(image_path))
        C_hat_grid = utils.dequantization(dct, qtable)
        h, w = C_hat_grid.shape

        tmp_dir = Path("/media/general_storage6/tmp_aux")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir=tmp_dir, delete=True) as tmp:
            img.save(
                tmp.name,
                "JPEG",
                quality=qf_pep,
                subsampling=0,
                optimize=False,
            )
            _, qtables_pep = utils.get_jpeg_info(tmp.name, 1)

        qtable_pep = qtables_pep[0]

        bdiv_grid = np.zeros(C_hat_grid.shape)
        for block_i in range(0, h, 8):
            for block_j in range(0, w, 8):
                bdiv_grid[block_i:block_i+8, block_j:block_j+8] = error_potential.block_divisibility(
                    C_hat_grid[block_i:block_i+8, block_j:block_j+8],
                    qtable_pep,
                )

        if crop_size is not None:
            s_r, s_c = origin
        
            # Pad if crop_size is larger than image size
            if h < crop_size[0] or w < crop_size[1]:
                # Pad divisibility grid 
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1])), 1)
                temp[:bdiv_grid.shape[0], :bdiv_grid.shape[1]] = bdiv_grid
                bdiv_grid = temp
    
            # Crop PEP
            bdiv_grid = bdiv_grid[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1]]
                
        bdiv_grid = np.nan_to_num(bdiv_grid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # robust percentile-based normalization to reduce the effect of outliers
        p = np.percentile(bdiv_grid, 99.5)
        p = max(float(p), 1e-6)
        bdiv_grid = np.clip(bdiv_grid, 0.0, p) / p
        
        t_bdiv_grid = torch.tensor(bdiv_grid, dtype=torch.float32).unsqueeze(0)
        return t_bdiv_grid
        
        
    def pep_features_from_crop_np(
        self,
        img_crop_np: np.ndarray,      # HWC uint8 (final crop)
        mask_crop_np: np.ndarray,     # HW int16 (-1,0,1)
        dct_channels: int,
        qf_pep: int,
    ) -> dict:
        """
        Compute PEP features from the final JPEG crop.
    
        The crop is first saved as a temporary JPEG file, then its DCT coefficients
        and quantization table are extracted in order to compute the PEP map.
        All outputs are defined in the crop reference frame:
            origin = (0, 0)
            crop_size = (H, W)
        """
        H, W, _ = img_crop_np.shape
        crop_size = (H, W)
    
        jpg_path = None
        try:
            # Save final crop as a temporary JPEG file.
            jpg_path = DocForgeryPEPDataset._save_crop_as_jpeg_tmp(
                img_crop_np,
                qf=int(self.quality_factor),
            )
    
            # Extract JPEG information from the saved crop.
            dct_coef, qtables = utils.get_jpeg_info(jpg_path, dct_channels)
            dct_coef = np.array(dct_coef)
    
            # Compute PEP using the luminance channel.
            pep = DocForgeryPEPDataset.pep_features(
                image_path=Path(jpg_path),
                dct=dct_coef[0],
                qtable=qtables[0],
                origin=(0, 0),
                crop_size=crop_size,
                qf_pep=qf_pep,
            )
    
            mask_t = torch.tensor(mask_crop_np, dtype=torch.long)
    
            return {
                "pep": pep,
                "mask": mask_t,
                "crop_size": crop_size,
                "origin": (0, 0),
                "grid_crop": True,
            }
    
        finally:
            if jpg_path is not None:
                try:
                    os.remove(jpg_path)
                except Exception:
                    pass

    
    def __len__(self):
        return len(self._index)
    

    def __getitem__(self, idx: int):
        """
        Return one crop with:
        - RGB image
        - PEP tensor
        - mask
        - metadata
        """

        if getattr(self, "randomize", False):
            self.quality_factor = random.randint(self.min_quality_factor, self.max_quality_factor)

        if self.balance_crops and (self.masks is not None):
            want_pos = (random.random() < self.p_pos_crop)
    
            if want_pos and len(self.pos_windows) > 0:
                idx = random.choice(self.pos_windows)
            elif (not want_pos) and len(self.neg_windows) > 0:
                idx = random.choice(self.neg_windows)
    
        doc_i, y, x = self._index[idx]
    
        img_path = self.images[doc_i]
        mask_np = self.read_mask(self.masks[doc_i]) if self.masks is not None else None
    
        img_crop_np, mask_crop_np, is_pos, tampered_ratio = self.crop_at_origin_with_pad(
            image_path=str(img_path),
            mask=mask_np,
            origin=(y, x),
            crop_size=self._crop_size,
            ignore_index=-1,
        )
    
        if self.use_augs and (self.aug is not None):
            img_crop_np, mask_crop_np = self.apply_albu_on_crop(img_crop_np, mask_crop_np)
            
        if self.use_augs:
            img_crop_np = self.apply_jpeg_forensic_on_crop(img_crop_np)
    
        raw_image = torch.tensor(img_crop_np.transpose(2, 0, 1), dtype=torch.float32)
        image = torch.clamp(raw_image / 255.0, 0.0, 1.0)
        mask_t = torch.tensor(mask_crop_np, dtype=torch.long)

        freq = self.pep_features_from_crop_np(
            img_crop_np=img_crop_np,
            mask_crop_np=mask_crop_np,
            dct_channels=1,
            #qf_pep=self.jpeg_qf_min,
            qf_pep=self.QF,
            
        )
    
        block = BlockValues(
            image=image,
            pep=None if freq is None else freq["pep"],
            mask=mask_t,
            crop_size=self._crop_size,
            origin=(y, x),
            grid_crop=self._grid_crop,
        )
    
        meta = {
            "tampered_ratio": float(tampered_ratio),
            "is_pos": int(is_pos),
            "doc_i": int(doc_i),
            "origin": (int(y), int(x)),
            "stem": img_path.stem,
        }
    
        return block, meta


    def read_mask(self, path: Path) -> np.array:
        """
        Read a binary mask:
        - 0 stays 0
        - any value > 0 becomes 1
        """
        
        mask = np.array(Image.open(str(path))).astype(np.int16)
        mask = (mask > 0).astype(np.int16)
        return mask


    @staticmethod
    def crop_at_origin_with_pad(
        image_path: str,
        mask: Optional[np.ndarray],
        origin: Tuple[int, int],
        crop_size: Tuple[int, int],
        ignore_index: int = -1,
    ):
        """
        Extract a crop from the image and mask at a given origin.
        Pads incomplete border crops with:
        - white (255) for the image
        - ignore_index for the mask
        """
        
        y, x = origin
        crop_h, crop_w = crop_size
    
        img_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        H, W, _ = img_np.shape
    
        if mask is None:
            mask = np.zeros((H, W), dtype=np.int16)
        else:
            mask = mask.astype(np.int16)
    
        img_crop = img_np[y:y+crop_h, x:x+crop_w, :]
        mask_crop = mask[y:y+crop_h, x:x+crop_w]
    
        out_img = np.full((crop_h, crop_w, 3), 255, dtype=np.uint8)
        out_mask = np.full((crop_h, crop_w), ignore_index, dtype=np.int16)
    
        hh, ww = img_crop.shape[0], img_crop.shape[1]
        out_img[:hh, :ww] = img_crop
        out_mask[:hh, :ww] = mask_crop
        
        valid = (out_mask != ignore_index)
        is_pos = bool(((out_mask == 1) & valid).any())
    
        denom = int(valid.sum()) or 1
        num = int(((out_mask == 1) & valid).sum())
        tampered_ratio = num / denom
    
        return out_img, out_mask, is_pos, tampered_ratio