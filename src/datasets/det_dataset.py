import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.images)

    def load_boxes(self, label_path):
        boxes = []
        classes = []

        if not os.path.exists(label_path):
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)

        with open(label_path, "r") as f:
            for line in f.read().strip().splitlines():
                cls, cx, cy, w, h = map(float, line.split())
                boxes.append([cx, cy, w, h])
                classes.append(int(cls))

        return torch.tensor(boxes, dtype=torch.float32), \
               torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))

        img = Image.open(img_path).convert("RGB")

        boxes, classes = self.load_boxes(label_path)

        if self.transform:
            img, boxes = self.transform(img, boxes)

        target = {
            "boxes": boxes,
            "labels": classes
        }

        return img, target
