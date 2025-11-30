import torch.nn as nn
import timm

class SharedEncoder(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(name, pretrained=pretrained, features_only=True)
        self.out_dim = self.encoder.feature_info.channels()[-1]

    def forward(self, x):
        return self.encoder(x)[-1]  # 最後一層 feature
