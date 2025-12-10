"""
Mask refinement utilities.

Given an initial pseudo mask and confidence map, apply morphological smoothing,
thresholding, small-component cleanup, and adjust confidence to down-weight
edges and unstable regions. Outputs refined mask and confidence.
"""

import os
import argparse
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


def load_mask_conf(mask_path: str, conf_path: str) -> Tuple[np.ndarray, np.ndarray]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    mask = (mask > 127).astype(np.uint8)
    conf = np.load(conf_path).astype(np.float32)
    if conf.shape != mask.shape:
        conf = cv2.resize(conf, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    return mask, conf


def remove_small_components(mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    sizes = stats[1:, -1]
    nb_components -= 1
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for i in range(nb_components):
        if sizes[i] >= min_size:
            cleaned[output == i + 1] = 1
    return cleaned


def fill_holes(mask: np.ndarray) -> np.ndarray:
    inv = 1 - mask
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    flood_mask = inv.copy().astype(np.uint8)
    cv2.floodFill(flood_mask, flood, (0, 0), 255)
    flood_mask = flood_mask[1:-1, 1:-1]
    holes = (1 - inv) & (flood_mask == 0)
    filled = mask.copy()
    filled[holes.astype(bool)] = 1
    return filled


def adjust_confidence(mask: np.ndarray, conf: np.ndarray, edge_decay: float = 0.5, ignore_thresh: float = 0.1):
    """
    Down-weight edges and mark very low-confidence pixels as ignore (value=0 in confidence).
    """
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    conf_adj = conf.copy()
    conf_adj[edges > 0] *= edge_decay
    conf_adj = np.clip(conf_adj, 0.0, 1.0)
    conf_adj[conf_adj < ignore_thresh] = 0.0
    return conf_adj


def refine_mask_and_conf(
    mask: np.ndarray,
    conf: np.ndarray,
    gaussian_ksize: int = 5,
    gaussian_sigma: float = 1.0,
    min_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply smoothing, thresholding, morphology, small-component removal, hole fill,
    and confidence adjustment.
    """
    mask_sm = cv2.GaussianBlur(mask.astype(np.float32), (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    mask_sm = (mask_sm > 0.5).astype(np.uint8)

    mask_open = cv2.morphologyEx(mask_sm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    mask_clean = remove_small_components(mask_close, min_size=min_size)
    mask_filled = fill_holes(mask_clean)

    conf_adj = adjust_confidence(mask_filled, conf)
    return mask_filled, conf_adj


def refine_dir(
    pseudo_mask_dir: str,
    conf_dir: str,
    out_mask_dir: str,
    out_conf_dir: str,
    gaussian_ksize: int = 5,
    gaussian_sigma: float = 1.0,
    min_size: int = 64,
    overwrite: bool = False,
) -> None:
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_conf_dir, exist_ok=True)

    mask_paths = sorted([p for p in os.listdir(pseudo_mask_dir) if p.lower().endswith(".png")])
    for fname in tqdm(mask_paths, desc="Refining masks"):
        mask_path = os.path.join(pseudo_mask_dir, fname)
        conf_path = os.path.join(conf_dir, fname.replace(".png", ".npy"))
        out_mask = os.path.join(out_mask_dir, fname)
        out_conf = os.path.join(out_conf_dir, fname.replace(".png", ".npy"))

        if (not overwrite) and os.path.exists(out_mask) and os.path.exists(out_conf):
            continue

        mask, conf = load_mask_conf(mask_path, conf_path)
        mask_ref, conf_ref = refine_mask_and_conf(
            mask,
            conf,
            gaussian_ksize=gaussian_ksize,
            gaussian_sigma=gaussian_sigma,
            min_size=min_size,
        )

        cv2.imwrite(out_mask, (mask_ref * 255).astype(np.uint8))
        np.save(out_conf, conf_ref.astype(np.float32))


def parse_args():
    parser = argparse.ArgumentParser("Refine pseudo masks")
    parser.add_argument("--mask-dir", type=str, required=True, help="Input pseudo mask dir (.png)")
    parser.add_argument("--conf-dir", type=str, required=True, help="Input confidence dir (.npy)")
    parser.add_argument("--out-mask-dir", type=str, required=True, help="Output refined masks (.png)")
    parser.add_argument("--out-conf-dir", type=str, required=True, help="Output refined confidence (.npy)")
    parser.add_argument("--gaussian-ksize", type=int, default=5)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    refine_dir(
        pseudo_mask_dir=args.mask_dir,
        conf_dir=args.conf_dir,
        out_mask_dir=args.out_mask_dir,
        out_conf_dir=args.out_conf_dir,
        gaussian_ksize=args.gaussian_ksize,
        gaussian_sigma=args.gaussian_sigma,
        min_size=args.min_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
