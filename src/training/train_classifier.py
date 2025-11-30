# src/training/train_classifier.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ssl_dino import DinoModel
from src.models.cell_classifier import CellClassifier
from src.datasets.cls_dataset import build_classification_datasets


def train_classifier(
    data_root: str = "data/classification",
    ssl_ckpt_path: str = "checkpoints/ssl_pretrain.pth",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    img_size: int = 224,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) 讀陽/陰分類資料集
    train_ds, val_ds = build_classification_datasets(data_root, img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")   # e.g. ['negative', 'positive']

    # 2) 載入 SSL checkpoint，取出 student backbone
    ssl_model = DinoModel()
    state = torch.load(ssl_ckpt_path, map_location="cpu")
    ssl_model.load_state_dict(state)
    ssl_model.to(device)
    ssl_model.eval()  # backbone 模型設為 eval（但 head 用不到）

    # 3) 建 classifier（使用 SSL student backbone）
    num_classes = len(train_ds.classes)
    model = CellClassifier(ssl_model, hidden_dim=512, num_classes=num_classes).to(device)

    # 只優化 classifier head 的參數
    optimizer = optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        # ====== Train ======
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

        # ====== Val ======
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", ncols=100):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%"
        )

        # 儲存最佳驗證結果
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/cell_classifier_best.pth")
            print(f"  → New best val acc: {best_val_acc*100:.2f}%, model saved.")

    print(f"\nDone. Best Val Acc = {best_val_acc*100:.2f}% "
          f"(saved to checkpoints/cell_classifier_best.pth)")


if __name__ == "__main__":
    train_classifier()
