import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.det_dataset import DetectionDataset
from src.datasets.transforms.det_aug import DetAug
from src.models.detector_rtdetr_wrapper import RTDETRWithSSL

def train_detector():

    device = "cuda"

    # Dataset
    train_set = DetectionDataset(
        "data/det/train/images",
        "data/det/train/labels",
        transform=DetAug(size=640)
    )
    val_set = DetectionDataset(
        "data/det/val/images",
        "data/det/val/labels",
        transform=DetAug(size=640)
    )

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: list(zip(*x)))
    val_loader   = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=lambda x: list(zip(*x)))

    # Model
    model = RTDETRWithSSL(num_classes=1, freeze_ssl=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    scaler = torch.cuda.amp.GradScaler()

    print("Start training...")

    for epoch in range(50):
        model.train()
        pbar = tqdm(train_loader)

        for imgs, targets in pbar:
            imgs = torch.stack([i.to(device) for i in imgs])

            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss_dict = model.detector.compute_loss(outputs, targets)
                loss = sum(loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

        torch.save(model.state_dict(), f"checkpoints/det_epoch{epoch}.pth")
