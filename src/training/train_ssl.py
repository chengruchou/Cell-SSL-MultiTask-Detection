# src/training/train_ssl.py
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from src.utils.common import get_normalization

from timm.utils import AverageMeter
from src.datasets.ssl_dataset_pt import SSLMicroscopyPTDataset
from src.datasets.transforms.ssl_aug import DinoAugment
from src.models.ssl_dino import DinoModel
from src.models.ssl_loss import DinoLoss


def parse_args():
    parser = argparse.ArgumentParser("DINO SSL pretraining")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data_root", type=str, default="data/ssl_pt")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="checkpoints/ssl_pretrain.pth")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    return parser.parse_args()


def load_cfg(args):
    if args.config is None:
        return vars(args)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_map = {
        "data_root": cfg.get("dataset", args.data_root),
        "img_size": cfg.get("img_size", args.img_size),
        "batch_size": cfg.get("batch_size", args.batch_size),
        "epochs": cfg.get("epochs", args.epochs),
        "lr": cfg.get("lr", args.lr),
        "num_workers": cfg.get("num_workers", args.num_workers),
        "output": cfg.get("output", args.output),
    }
    return cfg_map


def train_ssl(cfg=None):
    if cfg is None:
        args = parse_args()
        cfg = load_cfg(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    mean, std = get_normalization("ssl")
    _ = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = SSLMicroscopyPTDataset(cfg["data_root"])

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = DinoModel().to(device)
    criterion = DinoLoss().to(device)
    optimizer = optim.AdamW(model.student.parameters(), lr=cfg["lr"])
    gpu_aug = DinoAugment(size=cfg["img_size"]).to(device)

    os.makedirs(os.path.dirname(cfg["output"]) or ".", exist_ok=True)

    print("\n===== SSL Pretraining Start =====\n")

    for epoch in range(cfg["epochs"]):
        model.train()
        loss_meter = AverageMeter()
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)

        for batch_idx, imgs in enumerate(progress_bar):
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad():
                v1, v2 = gpu_aug(imgs)

            if epoch == 0 and batch_idx == 0:
                print("\n[Debug] imgs device:", imgs.device)
                print("[Debug] v1 device:", v1.device, "\n")

            s_out, t_out = model(v1, v2)
            loss = criterion(s_out, t_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), n=imgs.size(0))
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        print(f"Epoch {epoch}: Final Loss = {loss_meter.avg:.4f}")

    torch.save(model.state_dict(), cfg["output"])
    print(f"\nSaved SSL weights to {cfg['output']}\n")


if __name__ == "__main__":
    args = parse_args()
    train_ssl(load_cfg(args))
