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

class MammoDataset(Dataset):
    """
    MaMVT-style input:
      - Unify orientation: flip right-side views (R-CC, R-MLO) so nipples face RIGHT.
      - 4-view synchronized augmentations (same random params).
      - Stretch resize to 1536x1536 (no padding) for Swin patching.
      - Random left–right exam swap (train only) — exam-level label unchanged.
      - Grayscale -> 3-channel for ImageNet normalization.
    CSV columns (exactly 5): "L-CC", "R-CC", "L-MLO", "R-MLO", "label"
    """
    def __init__(self, csv_path: str, root_dir: str = None,
                 img_size: int = 1536, train: bool = True,
                 enable_lr_swap: bool = True, ensure_orientation: bool = True):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir or os.path.dirname(csv_path)
        self.train = train
        self.img_size = int(img_size)
        self.enable_lr_swap = bool(enable_lr_swap)
        self.ensure_orientation = bool(ensure_orientation)

        required = set(VIEWS + ["label"])
        assert required.issubset(self.df.columns), f"CSV must contain columns: {required}"

        self.transforms = build_exam_transforms(img_size=self.img_size, train=self.train)

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

    def _ensure_orientation(self, imgs_rgb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.ensure_orientation:
            return imgs_rgb
        imgs_rgb["R-CC"]  = self._hflip(imgs_rgb["R-CC"])
        imgs_rgb["R-MLO"] = self._hflip(imgs_rgb["R-MLO"])
        return imgs_rgb

    def _maybe_left_right_swap(self, imgs_t: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Train only: swap L<->R images with 50% prob. Exam-level label stays the same.
        if self.train and self.enable_lr_swap and random.random() < 0.5:
            imgs_t["L-CC"],  imgs_t["R-CC"]  = imgs_t["R-CC"],  imgs_t["L-CC"]
            imgs_t["L-MLO"], imgs_t["R-MLO"] = imgs_t["R-MLO"], imgs_t["L-MLO"]
        return imgs_t

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # 1) read grayscale -> RGB; 2) unify orientation (flip right-side)
        imgs = {v: self._to_rgb3(self._read_img_gray(row[v])) for v in VIEWS}
        imgs = self._ensure_orientation(imgs)

        # 2) synchronized 4-view augmentation (same random params)
        data_in = {
            "image":  imgs["L-CC"],
            "r_cc":   imgs["R-CC"],
            "l_mlo":  imgs["L-MLO"],
            "r_mlo":  imgs["R-MLO"],
        }
        out = self.transforms(**data_in)

        # 3) tensors
        imgs_t = {
            "L-CC":  out["image"],
            "R-CC":  out["r_cc"],
            "L-MLO": out["l_mlo"],
            "R-MLO": out["r_mlo"],
        }

        # 4) optional random L<->R swap (does NOT change exam-level label)
        imgs_t = self._maybe_left_right_swap(imgs_t)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {**imgs_t, "label": label}


def build_exam_transforms(img_size: int = 1536, train: bool = True) -> A.Compose:
    aug = [
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
    ]
    if train:
        aug += [
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=0,
                shear=0,
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.0,
                contrast_limit=0.2,  # ~0.8~1.2 effective
                p=0.8
            ),
            A.GaussNoise(p=0.5),
        ]
    aug += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(
        aug,
        additional_targets={
            "r_cc": "image",
            "l_mlo": "image",
            "r_mlo": "image",
        },
        is_check_shapes=False,   # ✅ 加這行，禁用 shape 檢查
    )
