# src/training/train_classifier_mae.py
import os
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from torch import optim, nn

from src.models.cell_mae_vit import MAE, CellViTBackbone, CellClassifier
from src.datasets.cls_dataset import build_classification_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Classifier finetune on MAE backbone")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data_root", type=str, default="data/classification")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mae_ckpt", type=str, default="checkpoints/ssl_mae_best.pth")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()


def load_cfg(args):
    if args.config is None:
        return vars(args)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    def _get(key, default, cast):
        val = cfg.get(key, default)
        try:
            return cast(val)
        except Exception:
            return cast(default)

    return {
        "data_root": cfg.get("dataset", args.data_root),
        "img_size": _get("img_size", args.img_size, int),
        "batch_size": _get("batch_size", args.batch_size, int),
        "epochs": _get("epochs", args.epochs, int),
        "lr": _get("lr", args.lr, float),
        "weight_decay": _get("weight_decay", args.weight_decay, float),
        "mae_ckpt": cfg.get("mae_ckpt", args.mae_ckpt),
        "freeze_encoder": cfg.get("freeze_encoder", args.freeze_encoder),
        "num_workers": _get("num_workers", args.num_workers, int),
        "device": cfg.get("device", args.device),
        "save_dir": cfg.get("save_dir", args.save_dir),
    }


def main(cfg=None):
    if cfg is None:
        args = parse_args()
        cfg = load_cfg(args)
    print("[INFO] Configuration:", cfg)

    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading MAE checkpoint:", cfg["mae_ckpt"])
    mae = MAE(img_size=cfg["img_size"])
    mae_ckpt = torch.load(cfg["mae_ckpt"], map_location="cpu")
    if isinstance(mae_ckpt, dict) and "model" in mae_ckpt:
        mae.load_state_dict(mae_ckpt["model"], strict=False)
    else:
        mae.load_state_dict(mae_ckpt, strict=False)

    backbone = CellViTBackbone(
        mae,
        freeze_encoder=cfg["freeze_encoder"],
        normalize_input=True
    )

    train_ds, val_ds = build_classification_datasets(cfg["data_root"], img_size=cfg["img_size"])
    num_classes = len(train_ds.classes)
    model = CellClassifier(backbone, num_classes=num_classes).to(device)
    model.head = nn.Linear(model.head.in_features, num_classes).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"[Epoch {epoch}/{cfg['epochs']}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(cfg["save_dir"], "cell_classifier_mae_best.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "classes": train_ds.classes,
            }, save_path)
            print(f"[INFO] New best model saved at: {save_path}")

    print(f"[INFO] Training finished. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(load_cfg(args))
