"""
Segmentation augmentations that keep image/mask/conf alignment.
"""

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.utils.common import get_normalization


class SegTransform:
    def __call__(self, img: Image.Image, mask: Image.Image, conf: Image.Image):
        raise NotImplementedError


class Resize(SegTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img, mask, conf):
        img = TF.resize(img, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        conf = TF.resize(conf, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        return img, mask, conf


class RandomHorizontalFlip(SegTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, mask, conf):
        if random.random() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            conf = TF.hflip(conf)
        return img, mask, conf


class ToTensor(SegTransform):
    def __call__(self, img, mask, conf):
        img_t = TF.to_tensor(img)
        mask_t = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.float32) / 255.0
        conf_t = torch.as_tensor(TF.pil_to_tensor(conf), dtype=torch.float32) / 255.0
        if mask_t.dim() == 3:
            mask_t = mask_t[0:1]
        if conf_t.dim() == 3:
            conf_t = conf_t[0:1]
        return img_t, mask_t, conf_t


class Normalize(SegTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask, conf):
        img = TF.normalize(img, self.mean, self.std)
        return img, mask, conf


class Compose(SegTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, conf):
        for t in self.transforms:
            img, mask, conf = t(img, mask, conf)
        return img, mask, conf


def build_default_transforms(
    size: Tuple[int, int],
    hflip_p: float = 0.5,
) -> Compose:
    mean, std = get_normalization("segmentation")
    return Compose([
        Resize(size),
        RandomHorizontalFlip(p=hflip_p),
        ToTensor(),
        Normalize(mean, std),
    ])
