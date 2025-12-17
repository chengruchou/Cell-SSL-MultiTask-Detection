"""
Training loop for unsupervised segmentation using pseudo masks.
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cell_mae_vit import MAE, CellViTBackbone, CellSegmenter
from src.datasets.seg_dataset import PseudoSegDataset
from src.datasets.transforms.seg_aug import build_default_transforms
from src.utils.masked_losses import masked_bce_dice_loss
from src.utils.metric_seg import iou_score, dice_score, binarize
from src.utils.common import get_normalization


def build_model(mae_ckpt: str = None, device: str = "cuda"):
    mae = MAE(
        img_size=640,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        mask_ratio=0.75,
    )
    if mae_ckpt is not None and os.path.isfile(mae_ckpt):
        ckpt = torch.load(mae_ckpt, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        mae.load_state_dict(state_dict, strict=False)

    mean, std = get_normalization("segmentation")
    backbone = CellViTBackbone(mae=mae, freeze_encoder=True, normalize_input=True, mean=mean, std=std)
    model = CellSegmenter(backbone=backbone, num_classes=1, upsample_factor=16)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    running = 0.0
    for imgs, masks, confs, _ in tqdm(loader, desc="Train", ncols=100):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        confs = confs.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(imgs)
            loss, comps, _ = masked_bce_dice_loss(logits, masks, confs)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item()
    return running / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    count = 0
    for imgs, masks, confs, _ in tqdm(loader, desc="Val", ncols=100, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        confs = confs.to(device, non_blocking=True)

        logits = model(imgs)
        loss, comps, _ = masked_bce_dice_loss(logits, masks, confs)
        prob = torch.sigmoid(logits)
        pred_bin = binarize(prob)
        valid = (confs > 0).float()
        total_iou += iou_score(pred_bin, masks, valid)
        total_dice += dice_score(pred_bin, masks, valid)
        total_loss += loss.item()
        count += 1
    if count == 0:
        return 0.0, 0.0, 0.0
    return total_loss / count, total_iou / count, total_dice / count


def save_ckpt(state, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    return path


def parse_args(input_args=None):
    ap = argparse.ArgumentParser("Train pseudo-label CellSegmenter")
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--masks", type=str, required=True)
    ap.add_argument("--confs", type=str, required=True)
    ap.add_argument("--val-images", type=str, default=None)
    ap.add_argument("--val-masks", type=str, default=None)
    ap.add_argument("--val-confs", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mae-ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, default="./logs/seg")
    ap.add_argument("--use-amp", action="store_true", default=True, help="Use automatic mixed precision")
    return ap.parse_args(input_args)


def main(input_args=None):
    args = parse_args(input_args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_tf = build_default_transforms(size=(640, 640))
    val_tf = build_default_transforms(size=(640, 640), hflip_p=0.0)

    train_ds = PseudoSegDataset(args.images, args.masks, args.confs, transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    if args.val_images and args.val_masks and args.val_confs:
        val_ds = PseudoSegDataset(args.val_images, args.val_masks, args.val_confs, transform=val_tf)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    else:
        val_loader = None

    model = build_model(mae_ckpt=args.mae_ckpt, device=device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') if (args.use_amp and device.type == "cuda") else None

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        if val_loader is not None:
            val_loss, val_iou, val_dice = validate(model, val_loader, device)
        else:
            val_loss, val_iou, val_dice = 0.0, 0.0, 0.0

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "val_iou": val_iou,
        }
        save_ckpt(state, args.output, "latest.pth")
        if val_iou > best_iou:
            best_iou = val_iou
            save_ckpt(state, args.output, "best.pth")
            ckpt_dir = "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state, os.path.join(ckpt_dir, "segmentor_best.pth"))

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f}")


if __name__ == "__main__":
    main()

# alias for external callers
train = main
