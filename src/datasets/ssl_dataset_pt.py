# src/datasets/ssl_dataset_pt.py
import os
import torch
from torch.utils.data import Dataset

class SSLMicroscopyPTDataset(Dataset):
    def __init__(self, root):
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx])   # 直接讀 tensor（超快）
