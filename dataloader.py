"""Data loading and fixed preprocessing for four-view mammography exams."""
import os
import random
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


cv2.setNumThreads(0)

VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_THRESHOLD = 5
CROP_MARGIN_RATIO = 0.05


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MammoDataset(Dataset):
    """One sample contains L-CC, R-CC, L-MLO, and R-MLO images."""

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        img_size: Tuple[int, int] = (1024, 512),
        train: bool = False,
        seed: int = 42,
    ):
        self.frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.height, self.width = img_size
        self.train = train

        required = set(VIEWS) | {"label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        self.transform = self._build_transform(seed)

    def _build_transform(self, seed):
        transforms = [
            A.Resize(
                height=self.height,
                width=self.width,
                interpolation=cv2.INTER_CUBIC,
            )
        ]
        if self.train:
            transforms.extend(
                [
                    A.Affine(
                        translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                        scale=(0.95, 1.05),
                        rotate=(-5, 5),
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        p=0.7,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.08,
                        contrast_limit=0.08,
                        brightness_by_max=True,
                        p=0.4,
                    ),
                ]
            )
        transforms.extend(
            [
                A.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
        return A.Compose(
            transforms,
            additional_targets={view: "image" for view in VIEWS[1:]},
            is_check_shapes=False,
            seed=seed,
        )

    def _read_image(self, relative_path):
        path = os.path.join(self.root_dir, relative_path)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = self._crop_black_border(image)
        return np.repeat(image[..., None], 3, axis=-1)

    @staticmethod
    def _crop_black_border(image):
        rows, columns = np.where(image > CROP_THRESHOLD)
        if not len(rows):
            raise ValueError("Cannot crop image: no foreground pixels found")

        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(columns.min()), int(columns.max()) + 1
        margin_y = round((y1 - y0) * CROP_MARGIN_RATIO)
        margin_x = round((x1 - x0) * CROP_MARGIN_RATIO)
        y0, y1 = max(0, y0 - margin_y), min(image.shape[0], y1 + margin_y)
        x0, x1 = max(0, x0 - margin_x), min(image.shape[1], x1 + margin_x)
        return image[y0:y1, x0:x1]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        images = {view: self._read_image(row[view]) for view in VIEWS}

        # Normalize chest-wall orientation before synchronized augmentation.
        images["R-CC"] = cv2.flip(images["R-CC"], 1)
        images["R-MLO"] = cv2.flip(images["R-MLO"], 1)
        transformed = self.transform(
            image=images["L-CC"],
            **{view: images[view] for view in VIEWS[1:]},
        )
        tensors = {
            "L-CC": transformed["image"],
            **{view: transformed[view] for view in VIEWS[1:]},
        }
        stacked = torch.stack([tensors[view] for view in VIEWS], dim=0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return stacked, label


def get_dataloaders(
    csv_path_train: str,
    csv_path_val: str,
    csv_path_test: str,
    root_dir: str,
    img_size: Tuple[int, int] = (1024, 512),
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
):
    generator = torch.Generator().manual_seed(seed)

    def make_loader(csv_path, train):
        dataset = MammoDataset(csv_path, root_dir, img_size, train=train, seed=seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=train,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    return (
        make_loader(csv_path_train, train=True),
        make_loader(csv_path_val, train=False),
        make_loader(csv_path_test, train=False),
    )
