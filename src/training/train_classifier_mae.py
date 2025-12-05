# src/training/train_classifier_mae.py

import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch import optim, nn

from src.models.cell_mae_vit import MAE, CellViTBackbone, CellClassifier
from src.datasets.cls_dataset import build_classification_datasets


DATA_ROOT = "data/classification"
MAE_CKPT = "checkpoints\ssl_mae_best.pth"      # 你的 MAE SSL 預訓練檔案
IMG_SIZE = 640
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4
FREEZE_ENCODER = False                    # 如果你想只訓練 head → 設 True
SAVE_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloaders():
    train_ds, val_ds = build_classification_datasets(
        DATA_ROOT,
        img_size=IMG_SIZE
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")

    return train_loader, val_loader, train_ds.classes


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Using device:", DEVICE)

    # --------------------------------------------
    # 1. Load pretrained MAE encoder
    # --------------------------------------------
    print("[INFO] Loading MAE checkpoint:", MAE_CKPT)
    ckpt = torch.load(MAE_CKPT, map_location=DEVICE)

    mae = MAE(img_size=IMG_SIZE)
    mae.load_state_dict(ckpt["model"])

    backbone = CellViTBackbone(mae, freeze_encoder=FREEZE_ENCODER)

    # --------------------------------------------
    # 2. Build classifier
    # --------------------------------------------
    train_loader, val_loader, classes = build_dataloaders()
    num_classes = len(classes)

    model = CellClassifier(backbone, num_classes=num_classes).to(DEVICE)

    # --------------------------------------------
    # 3. Optimizer / Loss
    # --------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------
    # 4. Training + Validation
    # --------------------------------------------
    best_val_acc = 0.0
    best_ckpt = os.path.join(SAVE_DIR, "cell_classifier_mae_best.pth")

    for epoch in range(1, EPOCHS + 1):
        # ----- Train -----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ----- Validation -----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "acc": val_acc,
                    "timestamp": datetime.now().isoformat(),
                },
                best_ckpt,
            )
            print(f" → New BEST val acc: {best_val_acc*100:.2f}% (saved)\n")

    print(f"Done. Best Val Acc = {best_val_acc*100:.2f}%")
    print(f"Saved at {best_ckpt}")


if __name__ == "__main__":
    main()
