import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

SRC = "data/ssl_images"
DST = "data/ssl_pt"
os.makedirs(DST, exist_ok=True)

# 基本前處理（跟你 train_ssl 裡的 base_transform 保持一致）
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

files = [
    f for f in os.listdir(SRC)
    if f.lower().endswith(("png", "jpg", "jpeg"))
]

print(f"Found {len(files)} images. Converting to .pt ...")

for f in tqdm(files):
    img = Image.open(os.path.join(SRC, f)).convert("RGB")
    t = tf(img)  # [3, 224, 224]
    torch.save(t, os.path.join(DST, f"{os.path.splitext(f)[0]}.pt"))

print("Done! All tensors saved to:", DST)
