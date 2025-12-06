"""
MAE → RT-DETR Multiscale Backbone
Author: ChatGPT + chengruchou pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 引入你的 MAE encoder（CellViTBackbone）
from src.models.cell_mae_vit import MAE, CellViTBackbone

# RT-DETR register
from src.rtdetrv2_pytorch.src.core.workspace import register


# ============================================================
#   MAE backbone → Multiscale features for RT-DETR
# ============================================================
@register()
class MAE_RTDETR_Backbone(nn.Module):
    """
    Convert ViT-based MAE backbone into 3-scale features required by RT-DETR.

    RT-DETR expects:
        output = {
            "feat2": Tensor[B,C,H/8,W/8],
            "feat3": Tensor[B,C,H/16,W/16],
            "feat4": Tensor[B,C,H/32,W/32],
        }

    But MAE patch=16 → feature map = H/16, W/16
    So:
        feat3 = original ViT feature map
        feat2 = upsample x2  (H/8)
        feat4 = downsample x2 (H/32)

    This preserves consistency with RT-DETR pipeline.
    """

    __inject__ = ["mae"]     # RT-DETR YAML 可以注入 MAE module
    __share__  = ["num_channels"]

    def __init__(
        self,
        mae: MAE,
        num_channels: int = 384,
        freeze: bool = False,
        mae_ckpt: str = None,
    ):
        super().__init__()
        
        if mae_ckpt is not None:
            print(f"[MAE_RTDETR_Backbone] Loading MAE checkpoint from: {mae_ckpt}")
            ckpt = torch.load(mae_ckpt, map_location="cpu")

            # 依你的 ckpt 結構調整
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt

            missing, unexpected = mae.load_state_dict(state_dict, strict=False)
            print(
                f"[MAE_RTDETR_Backbone] load_state_dict done. "
                f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
        
        # 包住你的 MAE encoder → 回傳 CLS + (B,C,h,w)
        self.backbone = CellViTBackbone(mae, freeze_encoder=freeze)
        self.num_channels = num_channels

        C = num_channels

        # For scale transforms
        self.to_feat2 = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)  # upsample H/16 → H/8
        self.to_feat4 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1) # downsample H/16 → H/32


    def forward(self, x):
        """
        Returns dict for RT-DETR encoder:
        {
            "feat2": B,C,H/8,W/8,
            "feat3": B,C,H/16,W/16,
            "feat4": B,C,H/32,W/32
        }
        """
        cls_token, feat = self.backbone(x)   # feat = B,C,H/16,W/16

        feat3 = feat
        feat2 = self.to_feat2(F.relu(feat3))  # H/8
        feat4 = self.to_feat4(F.relu(feat3))  # H/32

        return [feat2, feat3, feat4]
