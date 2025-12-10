import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    mask_bool = mask > 0
    overlay = image.copy()
    overlay[mask_bool] = (255, 0, 0)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def visualize(image_dir: Path, mask_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
    for img_path in tqdm(images, desc="Visualizing"):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        vis = overlay_mask(img, mask)
        out_path = out_dir / img_path.name
        Image.fromarray(vis).save(out_path)


def parse_args():
    ap = argparse.ArgumentParser("Visualize segmentation results")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--seg_mask_dir", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    visualize(Path(args.image_dir), Path(args.seg_mask_dir), Path(args.out))


if __name__ == "__main__":
    main()
