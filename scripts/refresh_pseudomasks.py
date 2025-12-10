import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.segmentation.transforms import build_default_transforms
from src.segmentation.train_segmentor import build_model


def parse_args():
    ap = argparse.ArgumentParser("Refresh pseudo masks using trained segmentor")
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out-mask", type=str, required=True)
    ap.add_argument("--out-conf", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_mask, exist_ok=True)
    os.makedirs(args.out_conf, exist_ok=True)

    model = build_model(mae_ckpt=None, device=args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tf = build_default_transforms(size=(640, 640), hflip_p=0.0)

    paths = [os.path.join(args.images, f) for f in os.listdir(args.images)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    with torch.no_grad():
        for p in tqdm(paths, desc="Refreshing masks"):
            img = Image.open(p).convert("RGB")
            dummy_mask = Image.new("L", img.size, 0)
            dummy_conf = Image.new("L", img.size, 255)
            img_t, _, _ = tf(img, dummy_mask, dummy_conf)
            img_t = img_t.unsqueeze(0).to(device)
            logits = model(img_t)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            mask = (prob >= 0.5).astype(np.uint8)
            conf = np.abs(prob - 0.5) * 2.0
            base = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(os.path.join(args.out_mask, f"{base}.png"), mask * 255)
            np.save(os.path.join(args.out_conf, f"{base}.npy"), conf.astype(np.float32))


if __name__ == "__main__":
    main()
