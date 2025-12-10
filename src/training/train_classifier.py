# src/training/train_classifier.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_dino import DinoModel
from src.models.cell_classifier import CellClassifier
from src.datasets.cls_dataset import build_classification_datasets


def parse_args():
    ap = argparse.ArgumentParser("Classifier training")
    ap.add_argument("-c", "--config", type=str, default=None)
    ap.add_argument("--data_root", type=str, default="data/classification")
    ap.add_argument("--ssl_ckpt_path", type=str, default="checkpoints/ssl_pretrain.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--output", type=str, default="checkpoints/cell_classifier_best.pth")
    return ap.parse_args()


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
        "ssl_ckpt_path": cfg.get("ssl_ckpt_path", args.ssl_ckpt_path),
        "epochs": _get("epochs", args.epochs, int),
        "batch_size": _get("batch_size", args.batch_size, int),
        "lr": _get("lr", args.lr, float),
        "img_size": _get("img_size", args.img_size, int),
        "num_workers": _get("num_workers", args.num_workers, int),
        "output": cfg.get("output", args.output),
    }


def train_classifier(cfg=None):
    if cfg is None:
        args = parse_args()
        cfg = load_cfg(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_ds, val_ds = build_classification_datasets(cfg["data_root"], img_size=cfg["img_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")

    ssl_model = DinoModel()
    state = torch.load(cfg["ssl_ckpt_path"], map_location="cpu")
    ssl_model.load_state_dict(state)
    ssl_model.to(device)
    ssl_model.eval()

    num_classes = len(train_ds.classes)
    model = CellClassifier(ssl_model, hidden_dim=512, num_classes=num_classes).to(device)

    optimizer = optim.AdamW(model.head.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(cfg["output"]) or ".", exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", ncols=100):
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
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", ncols=100):
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
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg["output"])
            print(f"[INFO] New best model saved at: {cfg['output']}")


if __name__ == "__main__":
    args = parse_args()
    train_classifier(load_cfg(args))
