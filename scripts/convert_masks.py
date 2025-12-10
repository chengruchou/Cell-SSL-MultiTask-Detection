import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_mask_file(src: Path, dst: Path):
    img = Image.open(src).convert("RGB")
    arr = np.array(img)
    # if grayscale, expand
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    fg = (arr > 0).any(axis=-1).astype(np.uint8)  # bg=0, fg=1
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fg, mode="L").save(dst)


def convert_dir(input_dir: Path, output_dir: Path):
    masks = [p for p in input_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
    for p in tqdm(masks, desc="Converting masks"):
        rel = p.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        convert_mask_file(p, out_path)


def parse_args():
    ap = argparse.ArgumentParser("Convert masks to binary label maps (bg=0, fg=1)")
    ap.add_argument("--input_mask_dir", required=True)
    ap.add_argument("--output_mask_dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    convert_dir(Path(args.input_mask_dir), Path(args.output_mask_dir))


if __name__ == "__main__":
    main()
