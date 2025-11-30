# src/datasets/transforms/ssl_aug.py
import torch
import torch.nn as nn
import kornia.augmentation as K

class DinoAugment(nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.aug = nn.Sequential(
            # 在 GPU 上做 heavy augmentation
            K.RandomResizedCrop((size, size), scale=(0.5, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(0.3, 0.3, 0.3, 0.05),
            K.RandomGrayscale(p=0.1),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, H, W]，已經在 GPU 上的 tensor
        回傳兩個 augmented views: v1, v2
        """
        v1 = self.aug(x)
        v2 = self.aug(x)
        return v1, v2
