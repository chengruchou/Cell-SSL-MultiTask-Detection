# src/training/train_ssl.py
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from timm.utils import AverageMeter
from src.datasets.ssl_dataset_pt import SSLMicroscopyPTDataset
from src.datasets.transforms.ssl_aug import DinoAugment
from src.models.ssl_dino import DinoModel
from src.models.ssl_loss import DinoLoss


def train_ssl():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 限制 CPU threads（避免一直把所有 core 撐滿）
    # torch.set_num_threads(2)
    # torch.set_num_interop_threads(2)

    # 1) Dataset：只做基本處理（在 CPU）
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = SSLMicroscopyPTDataset("data/ssl_pt")

    loader = DataLoader(
        dataset,
        batch_size=128,        # 5090 可以吃很大，先試 128，不夠再加
        shuffle=True,
        drop_last=True,
        num_workers=0,         # augment 都在 GPU，0 就夠
        pin_memory=True,
    )

    # 2) Model & Loss
    model = DinoModel().to(device)
    criterion = DinoLoss().to(device)
    optimizer = optim.AdamW(model.student.parameters(), lr=1e-4)

    # 3) GPU augmentation（Kornia）
    gpu_aug = DinoAugment(size=224).to(device)

    os.makedirs("checkpoints", exist_ok=True)

    print("\n===== SSL Pretraining Start =====\n")

    for epoch in range(1000):
        model.train()
        loss_meter = AverageMeter()
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)

        for batch_idx, imgs in enumerate(progress_bar):
            # imgs: [B, 3, H, W] on CPU → 丟到 GPU
            imgs = imgs.to(device, non_blocking=True)

            # 在 GPU 上產生兩個 views
            with torch.no_grad():
                v1, v2 = gpu_aug(imgs)

            # 第一次 batch 印一下裝置，確認真的都在 cuda 上
            if epoch == 0 and batch_idx == 0:
                print("\n[Debug] imgs device:", imgs.device)
                print("[Debug] v1 device:", v1.device, "\n")

            # DINO forward
            s_out, t_out = model(v1, v2)
            loss = criterion(s_out, t_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), n=imgs.size(0))
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        print(f"Epoch {epoch}: Final Loss = {loss_meter.avg:.4f}")

    torch.save(model.state_dict(), "checkpoints/ssl_pretrain.pth")
    print("\nSaved SSL weights to checkpoints/ssl_pretrain.pth\n")


if __name__ == "__main__":
    train_ssl()
