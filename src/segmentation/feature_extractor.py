"""
Feature extraction for unsupervised segmentation.

This module loads the MAE-based CellViTBackbone and dumps patch-level feature
maps for a given (unlabeled) dataloader. Each sample's feature map is saved to
disk (per-image .pt or .npz) for downstream clustering and pseudo-mask
generation.
"""

import os
from typing import Iterable, Literal, Optional

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from src.models.cell_mae_vit import CellViTBackbone, MAE
from src.utils.common import get_normalization


def _default_path_key(sample) -> Optional[str]:
    """
    Try to extract an image path identifier from a dataloader sample.
    Expected patterns:
      - (img, path)
      - (img, target_dict_with_path)
      - dict with "path" or "img_path"
    """
    if isinstance(sample, (list, tuple)):
        if len(sample) >= 2 and isinstance(sample[1], str):
            return sample[1]
        if len(sample) >= 2 and isinstance(sample[1], dict):
            path = sample[1].get("path") or sample[1].get("img_path")
            if isinstance(path, str):
                return path
    if isinstance(sample, dict):
        path = sample.get("path") or sample.get("img_path")
        if isinstance(path, str):
            return path
    return None


def build_backbone(
    mae_ckpt: Optional[str] = None,
    device: str = "cuda",
    normalize_input: bool = True,
) -> CellViTBackbone:
    """
    Instantiate CellViTBackbone and optionally load a MAE checkpoint.
    """
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
    if mae_ckpt is not None and os.path.isfile(mae_ckpt):
        ckpt = torch.load(mae_ckpt, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        mae.load_state_dict(state_dict, strict=False)

    backbone = CellViTBackbone(
        mae=mae,
        freeze_encoder=True,  # frozen during feature dump
        normalize_input=normalize_input,
        mean=mean,
        std=std,
    )
    backbone.to(device)
    backbone.eval()
    return backbone


def extract_features(
    backbone: CellViTBackbone,
    dataloader: Iterable,
    output_dir: str,
    device: str = "cuda",
    output_format: Literal["pt", "npz"] = "pt",
    overwrite: bool = False,
    path_key_fn=_default_path_key,
) -> None:
    """
    Run inference over dataloader and save patch feature maps.

    Args:
        backbone: CellViTBackbone instance (eval mode recommended).
        dataloader: yields (image_tensor, path | target_with_path, ...) batches.
        output_dir: where per-image features are stored.
        device: compute device.
        output_format: "pt" or "npz".
        overwrite: whether to replace existing files.
        path_key_fn: function to derive an identifier from a batch sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    backbone.eval()

    # Fallback path list (for ImageFolder-like datasets)
    dataset_paths = None
    if hasattr(dataloader, "dataset"):
        ds = dataloader.dataset
        if hasattr(ds, "paths"):
            dataset_paths = list(ds.paths)
        elif hasattr(ds, "imgs"):
            dataset_paths = [p if isinstance(p, str) else p[0] for p in ds.imgs]
        elif hasattr(ds, "samples"):
            dataset_paths = [p if isinstance(p, str) else p[0] for p in ds.samples]

    from torch.utils.data import SequentialSampler
    sampler = getattr(dataloader, "sampler", None)
    if dataset_paths is not None and sampler is not None and not isinstance(sampler, SequentialSampler):
        raise ValueError("To infer paths from dataset metadata, use a sequential sampler (disable shuffle).")

    used_names = set()
    global_ptr = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0]
            elif isinstance(batch, dict) and "image" in batch:
                images = batch["image"]
            else:
                raise ValueError("Dataloader batch format not recognized for images.")

            images = images.to(device, non_blocking=True)
            cls_tokens, feat_maps = backbone(images)  # feat_maps: (B, C, H/ps, W/ps)

            batch_size = images.shape[0]
            for i in range(batch_size):
                sample = batch if batch_size == 1 else (
                    [b[i] if torch.is_tensor(b) else b for b in batch]
                    if isinstance(batch, (list, tuple)) else batch
                )
                path = path_key_fn(sample)
                if path is None:
                    if dataset_paths is None:
                        raise ValueError("Image path not provided; supply paths in dataset or disable shuffle to infer.")
                    if global_ptr >= len(dataset_paths):
                        raise IndexError("Dataset path list exhausted; sampler ordering mismatch.")
                    path = dataset_paths[global_ptr]
                    global_ptr += 1

                base = os.path.splitext(os.path.basename(path))[0]
                if base in used_names:
                    # auto-prefix parent folder to avoid collision
                    parent = os.path.basename(os.path.dirname(path))
                    base = f"{parent}_{base}"
                if base in used_names and (not overwrite):
                    raise ValueError(f"Duplicate basename detected even after prefix: {base}. Ensure unique image filenames.")
                used_names.add(base)

                feat = feat_maps[i].detach().cpu()  # (C, h, w)
                cls = cls_tokens[i].detach().cpu()

                out_path = os.path.join(output_dir, f"{base}.{output_format}")
                if (not overwrite) and os.path.exists(out_path):
                    raise FileExistsError(f"{out_path} exists. Use overwrite=True or ensure unique basenames.")

                if output_format == "pt":
                    torch.save({"path": path, "feat": feat, "cls": cls}, out_path)
                elif output_format == "npz":
                    np.savez_compressed(
                        out_path,
                        path=path,
                        feat=feat.numpy(),
                        cls=cls.numpy(),
                    )
                else:
                    raise ValueError(f"Unsupported output_format: {output_format}")
