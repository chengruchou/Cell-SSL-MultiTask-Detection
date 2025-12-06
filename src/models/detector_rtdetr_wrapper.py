import torch
import torch.nn as nn

# 官方 RT-DETR config loader
from src.rtdetrv2_pytorch.src.core.yaml_config import YAMLConfig

# 你的 MAE backbone
from src.models.rtdetr_mae_backbone import MAE_RTDETR_Backbone
from src.models.cell_mae_vit import MAE



class RTDETRWrapper(nn.Module):
    """
    Wrapper that builds RT-DETR model using:
        - Official RT-DETR YAMLConfig (same as train.py)
        - Your MAE-based multiscale backbone
    """

    def __init__(
        self,
        cfg_path: str,
        mae_ckpt: str = None,
        freeze_backbone: bool = False,
        device="cuda",
    ):
        super().__init__()
        self.device = torch.device(device)

        # -------------------------------
        # Load official RT-DETR config
        # -------------------------------
        self.cfg = YAMLConfig(cfg_path)
        print("[INFO] Loaded YAML config!")

        # -------------------------------
        # Build MAE backbone
        # -------------------------------
        img_size = self.cfg.yaml_cfg.get("input_size", [640, 640])[0]

        mae = MAE(
            img_size=img_size,
            patch_size=16,
            embed_dim=384,
            depth=6,
            num_heads=6,
        )

        if mae_ckpt:
            ckpt = torch.load(mae_ckpt, map_location=self.device)
            ckpt = ckpt["model"] if "model" in ckpt else ckpt
            mae.load_state_dict(ckpt)
            print(f"[INFO] Loaded MAE checkpoint: {mae_ckpt}")

        # Replace official backbone with your MAE backbone
        self.cfg.yaml_cfg["model"]["backbone"] = {
            "type": "MAE_RTDETR_Backbone",
            "freeze": freeze_backbone,
            "num_channels": 384,
            "mae": mae,   # inject your MAE instance
        }

        # -------------------------------
        # Build RT-DETR model from YAML
        # -------------------------------
        self.model = self.cfg.model.to(self.device)
        print("[INFO] RT-DETR model built with MAE backbone!")

    def forward(self, images, targets=None):
        return self.model(images, targets)
