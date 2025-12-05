# src/training/train_ssl_mae.py

import os
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch import optim

from src.models.cell_mae_vit import MAE


def parse_args():
    parser = argparse.ArgumentParser(description="MAE Self-Supervised Pretraining")

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


def build_dataloader(args):
    from src.datasets.ssl_dataset_pt import SSLMicroscopyPTDataset

    dataset = SSLMicroscopyPTDataset(args.data_root)


    n_samples = len(dataset)
    print(f"[INFO] SSLMicroscopyPTDataset len = {n_samples}, root = {args.data_root}")
    if n_samples == 0:
        raise ValueError(
            f"SSLMicroscopyPTDataset 在 root={args.data_root} 底下沒有找到任何樣本，"
            f"請確認路徑是否正確（例如 data/ssl_pt）"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader



def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. 建 MAE
    mae = MAE(
        img_size=args.img_size,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        mask_ratio=args.mask_ratio,
    ).to(device)

    # 2. Optimizer
    optimizer = optim.AdamW(
        mae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )

    # 3. DataLoader
    loader = build_dataloader(args)

    # 4. Train — only keep best model
    best_loss = float("inf")
    best_ckpt = os.path.join(args.save_dir, "ssl_mae_best.pth")

    for epoch in range(1, args.epochs + 1):
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
        print(f"[Epoch {epoch}/{args.epochs}] MAE Loss: {avg_loss:.4f}")

        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": mae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "args": vars(args),
                    "timestamp": datetime.now().isoformat(),
                },
                best_ckpt,
            )
            print(f"[INFO] Saved BEST model → {best_ckpt} (loss={best_loss:.4f})")

    print(f"[DONE] Best MAE loss = {best_loss:.4f}")
    print(f"[DONE] Best model saved at: {best_ckpt}")


if __name__ == "__main__":
    main()
