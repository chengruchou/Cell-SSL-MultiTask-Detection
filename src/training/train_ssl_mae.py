# src/training/train_ssl_mae.py
import os
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from torch import optim

from src.models.cell_mae_vit import MAE


def parse_args():
    parser = argparse.ArgumentParser(description="MAE Self-Supervised Pretraining")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data_root", type=str, default="data/ssl_pt")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()


def load_cfg(args):
    if args.config is None:
        return vars(args)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    # ensure numeric types
    def _get_num(key, default, cast):
        val = cfg.get(key, default)
        try:
            return cast(val)
        except Exception:
            return cast(default)
    cfg_map = {
        "data_root": cfg.get("dataset", args.data_root),
        "img_size": _get_num("img_size", args.img_size, int),
        "batch_size": _get_num("batch_size", args.batch_size, int),
        "epochs": _get_num("epochs", args.epochs, int),
        "lr": _get_num("lr", args.lr, float),
        "weight_decay": _get_num("weight_decay", args.weight_decay, float),
        "mask_ratio": _get_num("mask_ratio", args.mask_ratio, float),
        "num_workers": _get_num("num_workers", args.num_workers, int),
        "device": cfg.get("device", args.device),
        "save_dir": cfg.get("save_dir", args.save_dir),
    }
    return cfg_map


def build_dataloader(cfg):
    from src.datasets.ssl_dataset_pt import SSLMicroscopyPTDataset
    dataset = SSLMicroscopyPTDataset(cfg["data_root"])
    n_samples = len(dataset)
    print(f"[INFO] SSLMicroscopyPTDataset len = {n_samples}, root = {cfg['data_root']}")
    if n_samples == 0:
        raise ValueError(f"SSLMicroscopyPTDataset root={cfg['data_root']} is empty.")

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    return loader


def main(cfg=None):
    if cfg is None:
        args = parse_args()
        cfg = load_cfg(args)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    mae = MAE(
        img_size=cfg["img_size"],
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        mask_ratio=cfg["mask_ratio"],
    ).to(device)

    optimizer = optim.AdamW(
        mae.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=cfg["lr"] * 0.1
    )

    loader = build_dataloader(cfg)

    best_loss = float("inf")
    best_ckpt = os.path.join(cfg["save_dir"], "ssl_mae_best.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        mae.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device)

            loss, _, _ = mae.forward_loss(imgs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch}/{cfg['epochs']}] MAE Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": mae.state_dict()}, best_ckpt)
            print(f"[INFO] Updated best model: {best_ckpt}")

    print(f"[INFO] Training finished. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(load_cfg(args))
