# src/models/detector_rtdetr_wrapper.py

import sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

# ----------------------------------------------------
# 1. 把官方 rtdetrv2_pytorch 加進 sys.path 方便 import
# ----------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # 專案根目錄 Cell-SSL-MultiTask-Detection/
RTDETR_ROOT = ROOT / "src" / "rtdetrv2_pytorch"
sys.path.append(str(RTDETR_ROOT))

# ⚠️ 以下 import 可能要根據官方 repo 結構微調名稱
from src.rtdetrv2_pytorch.src.core import YAMLConfig   # 通常在 rtdetrv2_pytorch/src/core/yaml_config.py

class RTDETRDetector(nn.Module):
    """
    你的專案內部使用的 RT-DETR 偵測封裝。

    - 以官方 RT-DETRv2 model 為實作
    - 未來若要改成 SSL backbone，可以在這一層動手
    """

    def __init__(self,
                 cfg_rel_path: str = "configs/rtdetrv2/rtdetrv2_r18vd_3x_coco.yml",
                 ckpt_rel_path: str = "output/cell_rtdetr/best.pth",
                 device: str = "cuda"):
        super().__init__()

        self.device = torch.device(device)

        cfg_path = RTDETR_ROOT / cfg_rel_path
        ckpt_path = RTDETR_ROOT / ckpt_rel_path

        # 1. 用官方 YAMLConfig 建 model
        yaml_cfg = YAMLConfig(str(cfg_path))
        self.model = yaml_cfg.model
        self.model.to(self.device)

        # 2. 載入官方訓練好的 checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # 3. 建一份與官方 config 對應的前處理（視情況調整）
        #   - 如果官方 tools/infer.py 有 transforms，建議照抄過來
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            # 正規化也建議比照官方設定
        ])

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        主 pipeline (batch tensor 版)，給你的後續 multi-task 用。
        images: (B, 3, H, W), 已經是 tensor & to(device)
        """
        self.model.eval()
        outputs = self.model(images)
        return outputs

    @torch.no_grad()
    def infer_single_pil(self, img: Image.Image) -> Dict:
        """
        方便你在 inference script 直接丟 PIL image 進來做推論。
        回傳格式可以依你自己的 pipeline 設計。
        """
        x = self.transform(img).unsqueeze(0).to(self.device)
        outputs = self.model(x)

        # FIXME: 這裡要依官方 RT-DETR 的輸出格式解析 boxes/scores/labels
        # 通常是 outputs["pred_boxes"], outputs["pred_logits"]
        return outputs
