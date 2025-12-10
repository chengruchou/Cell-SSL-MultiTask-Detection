import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import torch

from src.segmentation.feature_extractor import build_backbone, extract_features


class FlatDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.paths = [os.path.join(root, f) for f in os.listdir(root)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), img_path


def parse_args():
    ap = argparse.ArgumentParser("Extract patch features from CellViTBackbone")
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--mae-ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--format", type=str, default="pt", choices=["pt", "npz"])
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    tf = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    if os.path.isdir(os.path.join(args.images, "0")):
        ds = datasets.ImageFolder(args.images, transform=tf)
    else:
        ds = FlatDataset(args.images, tf)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    backbone = build_backbone(mae_ckpt=args.mae_ckpt, device=args.device, normalize_input=True)
    extract_features(
        backbone=backbone,
        dataloader=loader,
        output_dir=args.output,
        device=args.device,
        output_format=args.format,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
