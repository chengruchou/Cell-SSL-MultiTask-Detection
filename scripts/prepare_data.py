import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def scan_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]


def split_paths(paths: List[Path], seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    random.seed(seed)
    random.shuffle(paths)
    n = len(paths)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]
    return train, val, test


def copy_paths(paths: List[Path], dst_root: Path, label: str):
    for p in paths:
        out_dir = dst_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out_dir / p.name)


def prepare(raw_data_dir: Path, output_dir: Path, default_label: str = "positive", seed: int = 42):
    images = scan_images(raw_data_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {raw_data_dir}")
    train, val, test = split_paths(images, seed=seed)

    for split, paths in [("train", train), ("val", val), ("test", test)]:
        split_root = output_dir / "classification" / split
        copy_paths(paths, split_root, default_label)
    print(f"Prepared dataset at {output_dir / 'classification'}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")


def parse_args():
    ap = argparse.ArgumentParser("Prepare ImageFolder-style classification splits")
    ap.add_argument("--raw_data_dir", required=True, help="Root directory of raw images")
    ap.add_argument("--output_dir", required=True, help="Output root directory")
    ap.add_argument("--label", default="positive", help="Default label name to use")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    prepare(Path(args.raw_data_dir), Path(args.output_dir), default_label=args.label, seed=args.seed)


if __name__ == "__main__":
    main()
