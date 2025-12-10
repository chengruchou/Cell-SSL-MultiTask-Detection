# src/datasets/unlabeled_dataset.py
import os
from typing import Callable, Tuple

from PIL import Image
from torch.utils.data import Dataset


class UnlabeledDetectionDataset(Dataset):
    """
    Unlabeled images for semi-supervised detection.
    Returns weak/strong augmented tensors plus original size (w, h).
    """

    def __init__(
        self,
        root: str,
        weak_transform: Callable,
        strong_transform: Callable,
        extensions=(".jpg", ".jpeg", ".png"),
    ):
        super().__init__()
        self.paths = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(extensions)
        ])
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size

        img_w = self.weak_transform(img)
        img_s = self.strong_transform(img)

        return img_w, img_s, (w, h), path
