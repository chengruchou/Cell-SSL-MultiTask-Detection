import argparse
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser("Visualize masks over images")
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--masks", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    img_paths = [p for p in os.listdir(args.images) if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    for fname in tqdm(img_paths, desc="Visualizing"):
        img_path = os.path.join(args.images, fname)
        mask_path = os.path.join(args.masks, fname.rsplit(".", 1)[0] + ".png")
        if not os.path.exists(mask_path):
            continue
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_bool = (mask > 127).astype(np.uint8)

        overlay = img.copy()
        overlay[mask_bool > 0] = (255, 0, 0)  # red overlay
        blended = cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0)

        out_path = os.path.join(args.output, fname)
        Image.fromarray(blended).save(out_path)


if __name__ == "__main__":
    main()
