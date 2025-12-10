# Standalone preprocessing: convert images to .pt tensors for SSL pretraining.
import os
import torch
from torchvision import transforms
from PIL import Image
from src.utils.common import get_normalization


def preprocess_folder(input_dir: str, output_dir: str, img_size: int = 224):
    os.makedirs(output_dir, exist_ok=True)
    mean, std = get_normalization("ssl")
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        tensor = tf(img)
        out_path = os.path.join(output_dir, fname + ".pt")
        torch.save(tensor, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()
    preprocess_folder(args.input, args.output, img_size=args.img_size)
