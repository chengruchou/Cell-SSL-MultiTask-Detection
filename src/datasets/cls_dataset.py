# src/datasets/cls_dataset.py
import os
from torchvision import datasets, transforms
from src.utils.common import get_normalization

def build_classification_datasets(
    root: str = "data/classification",
    img_size: int = 640,
):
    """
    構建分類資料集
    root/train/positive/*.png|jpg
    root/train/negative/*.png|jpg
    root/val/positive/*.png|jpg
    root/val/negative/*.png|jpg
    """
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "valid")

    mean, std = get_normalization("classification")

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    return train_ds, val_ds
