# src/datasets/cls_dataset.py
import os
from torchvision import datasets, transforms

def build_classification_datasets(
    root: str = "data/classification",
    img_size: int = 640,
):
    """
    預期資料結構：
    root/train/positive/*.png|jpg
    root/train/negative/*.png|jpg
    root/val/positive/*.png|jpg
    root/val/negative/*.png|jpg
    """
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "valid")

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # 可以加一點簡單 aug（可視需要調整或拿掉）
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    return train_ds, val_ds
