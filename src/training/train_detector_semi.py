import argparse
import os
import time
from copy import deepcopy
from itertools import cycle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets.unlabeled_dataset import UnlabeledDetectionDataset
from src.utils.pseudo_label import generate_pseudo_labels
from src.training.train_detector import set_seed, validate, save_checkpoint

from src.rtdetrv2_pytorch.src.core import YAMLConfig
from src.rtdetrv2_pytorch.src.optim import ModelEMA


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

def build_unlabeled_loader(cfg_yaml: dict):
    semi_cfg = cfg_yaml.get("semi_supervised", {})
    data_root = semi_cfg.get("unlabeled_img_folder", "./data/unlabeled")
    batch_size = semi_cfg.get("batch_size", 2)
    num_workers = semi_cfg.get("num_workers", 4)

    # default size follow eval_spatial_size / input_size
    size = cfg_yaml.get("eval_spatial_size", cfg_yaml.get("input_size", [640, 640]))[0]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    weak_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    strong_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = UnlabeledDetectionDataset(
        root=data_root,
        weak_transform=weak_tf,
        strong_transform=strong_tf,
    )

    def collate_fn(batch):
        imgs_w, imgs_s, orig_sizes, paths = zip(*batch)
        orig_sizes = torch.tensor(orig_sizes, dtype=torch.long)
        return (
            torch.stack(imgs_w, dim=0),
            torch.stack(imgs_s, dim=0),
            orig_sizes,
            paths,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader


def ramp_weight(base: float, epoch: int, warmup: int) -> float:
    if warmup <= 0:
        return base
    return base * min(1.0, float(epoch) / float(warmup))


# --------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------

def train_one_epoch_semi(
    model,
    criterion,
    optimizer,
    labeled_loader,
    unlabeled_loader,
    ema: ModelEMA,
    device,
    epoch: int,
    scaler=None,
    lambda_u: float = 1.0,
    pseudo_score_thr: float = 0.5,
    max_pseudo: int = None,
    print_freq: int = 10,
):
    model.train()
    criterion.train()
    ema.module.eval()

    unlabeled_iter = cycle(unlabeled_loader)
    running_sup = 0.0
    running_unsup = 0.0
    num_batches = len(labeled_loader)
    start_time = time.time()

    pbar = tqdm(labeled_loader, desc=f"Epoch {epoch} [Semi-Train]", ncols=120)
    for i, (samples, targets) in enumerate(pbar, start=1):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        imgs_w, imgs_s, orig_sizes, _ = next(unlabeled_iter)
        imgs_w = imgs_w.to(device)
        imgs_s = imgs_s.to(device)

        # --- supervised ---
        with torch.autocast(device_type=str(device), enabled=scaler is not None):
            outputs_sup = model(samples, targets=targets)
            loss_sup_dict = criterion(outputs_sup, targets)
            loss_sup = sum(loss_sup_dict.values())

        # --- pseudo labels with teacher ---
        with torch.no_grad():
            teacher_out = ema.module(imgs_w)
            pseudo_targets = generate_pseudo_labels(
                teacher_out,
                score_thr=pseudo_score_thr,
                max_pseudo=max_pseudo,
            )
            # keep device alignment
            pseudo_targets = [
                {k: v.to(device) for k, v in tgt.items()} for tgt in pseudo_targets
            ]

        # --- unsupervised ---
        with torch.autocast(device_type=str(device), enabled=scaler is not None):
            outputs_unsup = model(imgs_s, targets=pseudo_targets)
            loss_unsup_dict = criterion(outputs_unsup, pseudo_targets)
            loss_unsup = sum(loss_unsup_dict.values())
            loss_total = loss_sup + lambda_u * loss_unsup

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            optimizer.step()

        ema.update(model)

        running_sup += loss_sup.item()
        running_unsup += loss_unsup.item()
        pbar.set_postfix({
            "sup": f"{(running_sup / i):.4f}",
            "unsup": f"{(running_unsup / i):.4f}",
            "λ": f"{lambda_u:.2f}",
        })

    epoch_time = time.time() - start_time
    print(
        f"[Epoch {epoch:3d}] Semi-train done. "
        f"sup={running_sup/num_batches:.4f} "
        f"unsup={running_unsup/num_batches:.4f} "
        f"time={epoch_time:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser("Semi-supervised RT-DETR trainer")
    parser.add_argument("-c", "--config", type=str, default="configs/detector_rtdetr.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = YAMLConfig(args.config)
    if args.output_dir is not None:
        cfg.yaml_cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg.yaml_cfg["epochs"] = args.epochs

    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    model = cfg.model
    criterion = cfg.criterion
    optimizer = cfg.optimizer
    scheduler = cfg.lr_scheduler
    postprocessor = getattr(cfg, "postprocessor", None)
    base_evaluator = getattr(cfg, "evaluator", None)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    ema_decay = cfg.yaml_cfg.get("semi_supervised", {}).get("ema_decay", 0.999)
    ema = ModelEMA(model, decay=ema_decay)
    ema.to(device)

    scaler = torch.amp.GradScaler('cuda') if (args.use_amp and device.type == "cuda") else None
    epochs = cfg.yaml_cfg.get("epochs", cfg.yaml_cfg.get("epoches"))
    if epochs is None:
        raise KeyError("Neither 'epochs' nor 'epoches' found in YAML config.")

    # Semi-supervised hyperparams
    semi_cfg = cfg.yaml_cfg.get("semi_supervised", {})
    pseudo_thr = semi_cfg.get("pseudo_score_thr", 0.5)
    lambda_u_max = semi_cfg.get("lambda_u", 1.0)
    warmup_epochs = semi_cfg.get("unsup_warmup_epochs", 5)
    max_pseudo = semi_cfg.get("max_pseudo", None)

    unlabeled_loader = build_unlabeled_loader(cfg.yaml_cfg)

    # resume (student + ema)
    start_epoch = 1
    best_ap = -1.0
    best_val_loss = float("inf")
    output_dir = cfg.yaml_cfg["output_dir"]
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None and ckpt.get("scaler", None) is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_ap = ckpt.get("best_ap", -1.0)

    print("[INFO] Starting semi-supervised training...")

    for epoch in range(start_epoch, epochs + 1):
        lambda_u = ramp_weight(lambda_u_max, epoch, warmup_epochs)

        train_one_epoch_semi(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            labeled_loader=train_loader,
            unlabeled_loader=unlabeled_loader,
            ema=ema,
            device=device,
            epoch=epoch,
            scaler=scaler,
            lambda_u=lambda_u,
            pseudo_score_thr=pseudo_thr,
            max_pseudo=max_pseudo,
            print_freq=10,
        )

        if scheduler is not None:
            scheduler.step()

        # validation uses EMA weights for stability
        ema_model = ema.module.to(device)
        ema_model.eval()
        ema_state = deepcopy(model.state_dict())
        model.load_state_dict(ema.module.state_dict(), strict=False)

        epoch_evaluator = None
        if base_evaluator is not None:
            try:
                epoch_evaluator = type(base_evaluator)(
                    base_evaluator.coco_gt,
                    iou_types=base_evaluator.iou_types,
                )
            except Exception:
                epoch_evaluator = base_evaluator

        val_loss, ap, ap50, ap75 = validate(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            evaluator=epoch_evaluator,
            postprocessor=postprocessor,
        )

        # restore student weights after validation
        model.load_state_dict(ema_state, strict=False)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema.state_dict(),
            "best_val_loss": best_val_loss,
            "best_ap": best_ap,
        }
        save_checkpoint(state, output_dir, epoch)

        improved = False
        if ap is not None and ap > best_ap:
            best_ap = ap
            improved = True
        if ap is None and val_loss < best_val_loss:
            improved = True

        if improved:
            best_val_loss = min(best_val_loss, val_loss)
            best_path = os.path.join(output_dir, "best_semi.pth")
            torch.save(state, best_path)
            print(f"[INFO] New best model (semi) saved at: {best_path}")

    print("[INFO] Semi-supervised training finished.")


if __name__ == "__main__":
    main()
