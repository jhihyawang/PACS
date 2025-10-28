# data/dataset_mamvt.py
from typing import Dict, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random

VIEWS = ["L-CC", "R-CC", "L-MLO", "R-MLO"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
GRAY_MEAN = (0.485,)
GRAY_STD  = (0.229,)

class MammoDataset(Dataset):
    """
    MaMVT-style dataset
    - Unify orientation: flip right-side views so nipples face RIGHT.
    - 4-view synchronized augmentations (same random params).
    - Stretch resize to 1536x1536 (no padding).
    - Optional random left–right exam swap (train only).
    - Supports grayscale (1ch) or RGB (3ch) inputs.
    
    CSV columns (exactly 5): "L-CC", "R-CC", "L-MLO", "R-MLO", "label"
    """
    def __init__(self, csv_path: str, root_dir: str = None,
                 img_size: int = 1536, train: bool = True,
                 enable_lr_swap: bool = True, ensure_orientation: bool = True,
                 in_chans: int = 1):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir or os.path.dirname(csv_path)
        self.train = train
        self.img_size = int(img_size)
        self.enable_lr_swap = bool(enable_lr_swap)
        self.ensure_orientation = bool(ensure_orientation)
        self.in_chans = int(in_chans)

        required = set(VIEWS + ["label"])
        assert required.issubset(self.df.columns), f"CSV must contain columns: {required}"

        self.transforms = build_exam_transforms(
            img_size=self.img_size,
            train=self.train,
            in_chans=self.in_chans
        )

    def __len__(self):
        return len(self.df)

    def _read_img_gray(self, rel_path: str) -> np.ndarray:
        full_path = os.path.join(self.root_dir, rel_path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {full_path} (original path: {rel_path})")
        return img

    @staticmethod
    def _to_rgb3(img_gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _hflip(img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 1)

    def _ensure_orientation(self, imgs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Flip right-side images to make nipples face right."""
        if not self.ensure_orientation:
            return imgs
        imgs["R-CC"]  = self._hflip(imgs["R-CC"])
        imgs["R-MLO"] = self._hflip(imgs["R-MLO"])
        return imgs

    def _maybe_left_right_swap(self, imgs_t: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Randomly swap L↔R during training."""
        if self.train and self.enable_lr_swap and random.random() < 0.5:
            imgs_t["L-CC"],  imgs_t["R-CC"]  = imgs_t["R-CC"],  imgs_t["L-CC"]
            imgs_t["L-MLO"], imgs_t["R-MLO"] = imgs_t["R-MLO"], imgs_t["L-MLO"]
        return imgs_t

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # 1. read grayscale (1ch)
        imgs_gray = {v: self._read_img_gray(row[v]) for v in VIEWS}

        # 2. convert to RGB if needed
        imgs_in = (
            {v: self._to_rgb3(imgs_gray[v]) for v in VIEWS}
            if self.in_chans == 3 else imgs_gray
        )

        # 3. unify orientation
        imgs_in = self._ensure_orientation(imgs_in)

        # 4. synchronized 4-view augmentation (Albumentations)
        data_in = {
            "image":  imgs_in["L-CC"],
            "r_cc":   imgs_in["R-CC"],
            "l_mlo":  imgs_in["L-MLO"],
            "r_mlo":  imgs_in["R-MLO"],
        }
        out = self.transforms(**data_in)

        imgs_t = {
            "L-CC":  out["image"],
            "R-CC":  out["r_cc"],
            "L-MLO": out["l_mlo"],
            "R-MLO": out["r_mlo"],
        }

        # 5. optional random L<->R swap
        imgs_t = self._maybe_left_right_swap(imgs_t)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {**imgs_t, "label": label}


def build_exam_transforms(img_size: int = 1536, train: bool = True, in_chans: int = 1) -> A.Compose:
    """
    Build Albumentations transform with channel-dependent normalization.
    """
    aug = [A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC)]
    if train:
        aug += [
            A.Affine(scale=(0.8, 1.2),
                     translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                     rotate=0, shear=0, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.0, contrast_limit=0.2, p=0.8),
            A.GaussNoise(p=0.5),
        ]

    # choose normalize stats based on in_chans
    if in_chans == 1:
        mean, std = GRAY_MEAN, GRAY_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    aug += [
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(
        aug,
        additional_targets={
            "r_cc": "image",
            "l_mlo": "image",
            "r_mlo": "image",
        },
    )
