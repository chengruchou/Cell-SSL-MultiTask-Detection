"""
Cluster-based pseudo mask generation from patch embeddings.

Steps:
- iterate saved feature maps (.pt / .npz) produced by feature_extractor.py
- flatten patch embeddings to (N, C)
- run 2-class clustering (KMeans or GaussianMixture)
- map clusters back to (H', W') patch grid -> binary mask
- compute per-pixel confidence via cluster-margin
- upsample to full resolution (default 640x640)
- save mask (.png) and confidence (.npy)
"""

import os
import glob
import argparse
from typing import Literal, Tuple

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def _load_feat(path: str) -> Tuple[np.ndarray, str]:
    """
    Returns:
        feat: np.ndarray (C, h, w)
        orig_path: original image path if stored, else basename
    """
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
        feat = data["feat"].numpy() if isinstance(data["feat"], torch.Tensor) else data["feat"]
        orig_path = data.get("path", os.path.basename(path))
    elif path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        feat = data["feat"]
        orig_path = str(data["path"]) if "path" in data else os.path.basename(path)
    else:
        raise ValueError(f"Unsupported feature file: {path}")
    return feat, orig_path


def _cluster_features(
    feat: np.ndarray,
    method: Literal["kmeans", "gmm"] = "kmeans",
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        feat: (C, h, w)
    Returns:
        mask_patch: (h, w) binary (0/1)
        prob_fg: (h, w) float in [0,1]
    """
    c, h, w = feat.shape
    flat = feat.reshape(c, -1).T  # (N, C)

    if method == "kmeans":
        km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
        labels = km.fit_predict(flat)  # (N,)
        centers = km.cluster_centers_  # (2, C)
        # pseudo probability by inverse distance softmax
        dists = np.stack([np.linalg.norm(flat - centers[i], axis=1) for i in range(2)], axis=1)
        dists = np.clip(dists, 1e-6, None)
        logits = -dists
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)  # (N,2)
        # choose fg as higher-norm centroid
        norm_centers = np.linalg.norm(centers, axis=1)
        fg_idx = int(norm_centers.argmax())
        prob_fg = probs[:, fg_idx]
    elif method == "gmm":
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
        gmm.fit(flat)
        probs = gmm.predict_proba(flat)  # (N,2)
        # choose fg as higher-norm mean
        means = gmm.means_  # (2,C)
        fg_idx = int(np.linalg.norm(means, axis=1).argmax())
        prob_fg = probs[:, fg_idx]
        labels = probs.argmax(axis=1)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    mask_patch = (prob_fg.reshape(h, w) >= 0.5).astype(np.uint8)
    prob_fg_map = prob_fg.reshape(h, w)
    return mask_patch, prob_fg_map


def _confidence_from_prob(prob_fg: np.ndarray) -> np.ndarray:
    """
    Confidence from foreground probability margin: |p - 0.5| * 2, clamped [0,1].
    """
    conf = np.abs(prob_fg - 0.5) * 2.0
    return np.clip(conf, 0.0, 1.0)


def generate_pseudo_for_dir(
    feat_dir: str,
    output_mask_dir: str,
    output_conf_dir: str,
    method: Literal["kmeans", "gmm"] = "kmeans",
    target_size: Tuple[int, int] = (640, 640),
    overwrite: bool = False,
    random_state: int = 0,
) -> None:
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_conf_dir, exist_ok=True)

    feat_paths = sorted(glob.glob(os.path.join(feat_dir, "*.pt")) + glob.glob(os.path.join(feat_dir, "*.npz")))
    for fp in tqdm(feat_paths, desc="Clustering pseudo masks"):
        feat, orig_path = _load_feat(fp)  # (C,h,w)
        base = os.path.splitext(os.path.basename(orig_path))[0]
        out_mask_path = os.path.join(output_mask_dir, f"{base}.png")
        out_conf_path = os.path.join(output_conf_dir, f"{base}.npy")

        if (not overwrite) and os.path.exists(out_mask_path) and os.path.exists(out_conf_path):
            continue

        mask_patch, prob_fg = _cluster_features(feat, method=method, random_state=random_state)
        conf_patch = _confidence_from_prob(prob_fg)

        # Upsample to full resolution
        mask_full = cv2.resize(mask_patch.astype(np.uint8), target_size[::-1], interpolation=cv2.INTER_LINEAR)
        conf_full = cv2.resize(conf_patch.astype(np.float32), target_size[::-1], interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(out_mask_path, (mask_full * 255).astype(np.uint8))
        np.save(out_conf_path, conf_full.astype(np.float32))


def parse_args():
    parser = argparse.ArgumentParser("Cluster-based pseudo mask generator")
    parser.add_argument("--feat-dir", type=str, required=True, help="Directory containing feature .pt/.npz files")
    parser.add_argument("--out-mask-dir", type=str, required=True, help="Output directory for binary masks (.png)")
    parser.add_argument("--out-conf-dir", type=str, required=True, help="Output directory for confidence maps (.npy)")
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "gmm"])
    parser.add_argument("--target-size", type=int, nargs=2, default=[640, 640], help="H W of output masks")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for clustering")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_pseudo_for_dir(
        feat_dir=args.feat_dir,
        output_mask_dir=args.out_mask_dir,
        output_conf_dir=args.out_conf_dir,
        method=args.method,
        target_size=(args.target_size[0], args.target_size[1]),
        overwrite=args.overwrite,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
