"""
Segmentation dataset that returns image, pseudo mask, and confidence map.
All transforms must keep spatial alignment.
"""

import os
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PseudoSegDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        conf_dir: str,
        transform: Callable,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.conf_dir = conf_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.rsplit(".", 1)[0] + ".png")
        conf_path = os.path.join(self.conf_dir, img_name.rsplit(".", 1)[0] + ".npy")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        conf = Image.fromarray((np.load(conf_path).astype(np.float32) * 255).astype(np.uint8))

        img_t, mask_t, conf_t = self.transform(image, mask, conf)
        return img_t, mask_t, conf_t, img_name
