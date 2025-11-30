# src/models/cell_classifier.py
import torch.nn as nn
from src.models.ssl_dino import DinoModel

class CellClassifier(nn.Module):
    def __init__(self, ssl_model: DinoModel, hidden_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.backbone = ssl_model.student
        for p in self.backbone.parameters():
            p.requires_grad = False

        in_dim = self.backbone.out_dim
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits
