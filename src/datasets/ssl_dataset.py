# src/datasets/ssl_dataset.py
from torch.utils.data import Dataset
from PIL import Image
import os

class SSLMicroscopyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)   # 回傳一個 [3,H,W] tensor
        return img
