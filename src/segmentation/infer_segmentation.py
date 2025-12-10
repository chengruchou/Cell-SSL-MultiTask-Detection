"""
Minimal segmentation inference for trained CellSegmenter.
"""

import os
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.cell_mae_vit import MAE, CellViTBackbone, CellSegmenter
from src.segmentation.transforms import build_default_transforms
from src.utils.common import get_normalization


def build_model(ckpt_path: str, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    mean, std = get_normalization("segmentation")
    mae = MAE(
        img_size=640,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        mask_ratio=0.75,
    )
    backbone = CellViTBackbone(mae=mae, freeze_encoder=True, normalize_input=True, mean=mean, std=std)
    model = CellSegmenter(backbone=backbone, num_classes=1, upsample_factor=16)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, device


def parse_args():
    ap = argparse.ArgumentParser("Segmentation inference")
    ap.add_argument("--images", type=str, required=True, help="Input image directory")
    ap.add_argument("--checkpoint", type=str, required=True, help="Trained model checkpoint")
    ap.add_argument("--output", type=str, required=True, help="Output mask directory")
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    model, device = build_model(args.checkpoint, device=args.device)
    tf = build_default_transforms(size=(640, 640), hflip_p=0.0)

    img_paths = [os.path.join(args.images, f) for f in os.listdir(args.images)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    with torch.no_grad():
        for p in tqdm(img_paths, desc="Infer"):
            img = Image.open(p).convert("RGB")
            dummy_mask = Image.new("L", img.size, 0)
            dummy_conf = Image.new("L", img.size, 255)
            img_t, _, _ = tf(img, dummy_mask, dummy_conf)
            img_t = img_t.unsqueeze(0).to(device)

            logits = model(img_t)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask = (prob >= 0.5).astype(np.uint8)
            base = os.path.splitext(os.path.basename(p))[0]
            out_mask = os.path.join(args.output, f"{base}.png")
            cv2.imwrite(out_mask, mask * 255)


if __name__ == "__main__":
    main()
