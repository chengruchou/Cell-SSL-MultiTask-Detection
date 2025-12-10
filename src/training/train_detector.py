import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # 進度條

# 先 import → 觸發 register()
import src.models.rtdetr_mae_backbone  # noqa: F401
import src.models.cell_mae_vit        # noqa: F401

from src.rtdetrv2_pytorch.src.core import YAMLConfig
from src.rtdetrv2_pytorch.src.solver import TASKS  # 目前沒用到，但保留以後可切回 solver.fit() 用


# ------------------------------
#  Argparse
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser("RT-DETR + MAE backbone trainer")

    # 基本設定
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/detector_rtdetr.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override 'epochs' in YAML config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable torch.amp mixed precision.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir in YAML.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (this script's format).",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run validation once, no training.",
    )

    # 如果你之後想加入像原版一樣的 yaml override，可以再加:
    # parser.add_argument("-u", "--update", nargs='+', help="update yaml config")

    return parser.parse_args()


# ------------------------------
#  Utilities
# ------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 如果有用 numpy / random 再補（例如 numpy / random）


def save_checkpoint(
    state,
    output_dir: str,
    epoch: int,
):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
    torch.save(state, ckpt_path)
    print(f"[INFO] Saved checkpoint: {ckpt_path}")


# ------------------------------
#  Train loop (with tqdm)
# ------------------------------

def train_one_epoch(
    model,
    criterion,
    optimizer,
    train_loader,
    device,
    epoch: int,
    scaler=None,
):
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    start_time = time.time()

    # tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", ncols=100)

    for i, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            # AMP branch
            with torch.amp.autocast('cuda'):
                outputs = model(images, targets)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        avg_loss = running_loss / i
        cur_lr = optimizer.param_groups[0]["lr"]

        # 更新 tqdm 顯示
        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "avg": f"{avg_loss:.4f}",
            "lr": f"{cur_lr:.6f}",
        })

    epoch_time = time.time() - start_time
    avg_epoch_loss = running_loss / num_batches
    cur_lr = optimizer.param_groups[0]["lr"]
    print(
        f"[Epoch {epoch:3d}] "
        f"Train done. avg_loss={avg_epoch_loss:.4f} "
        f"time={epoch_time:.1f}s "
        f"lr={cur_lr:.6f}"
    )

    return avg_epoch_loss


# ------------------------------
#  Validation loop (loss + mAP)
# ------------------------------

@torch.no_grad()
def validate(
    model,
    criterion,
    val_loader,
    device,
    epoch: int,
    evaluator=None,
    postprocessor=None,
):
    """
    如果 cfg 有設定 CocoEvaluator + postprocessor，這裡會同時計算：
      - avg validation loss
      - COCO AP / AP50 / AP75
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", ncols=100, leave=False)

    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # --------- 1) forward for loss（跟 train 一樣）---------
        outputs_loss = model(images, targets)
        loss_dict = criterion(outputs_loss, targets)
        loss = sum(loss_dict.values())
        total_loss += loss.item()

        avg_val = total_loss / max(pbar.n + 1, 1)
        pbar.set_postfix({"val_avg": f"{avg_val:.4f}"})

        # --------- 2) forward for detection（不帶 targets）---------
        if (evaluator is not None) and (postprocessor is not None):
            outputs_det = model(images)  # ★ 推論模式：不帶 targets

            # 取得原圖尺寸（通常在 target["orig_size"] 或 "orig_shape"）
            if "orig_size" in targets[0]:
                orig_sizes = torch.stack([t["orig_size"] for t in targets])
            elif "orig_shape" in targets[0]:
                orig_sizes = torch.stack([t["orig_shape"] for t in targets])
            else:
                # 找不到就跳過這 batch 的 mAP 計算
                continue

            # postprocess：將模型輸出轉成 coco 格式的 boxes/scores/labels
            processed = (
                postprocessor(outputs_det, orig_sizes)
                if callable(postprocessor)
                else postprocessor["bbox"](outputs_det, orig_sizes)
            )

            res = {}
            for t, r in zip(targets, processed):
                image_id = t["image_id"]
                if isinstance(image_id, torch.Tensor):
                    image_id = image_id.item()
                res[image_id] = {
                    "boxes": r["boxes"].cpu(),
                    "scores": r["scores"].cpu(),
                    "labels": r["labels"].cpu(),
                }

            evaluator.update(res)

    avg_val_loss = total_loss / max(num_batches, 1)

    ap, ap50, ap75 = None, None, None
    if evaluator is not None:
        # 這幾個 method 不是每個版本都有，用 hasattr 保護
        if hasattr(evaluator, "synchronize_between_processes"):
            try:
                evaluator.synchronize_between_processes()
            except Exception:
                pass

        if hasattr(evaluator, "accumulate"):
            try:
                evaluator.accumulate()
            except Exception:
                pass

        if hasattr(evaluator, "summarize"):
            try:
                evaluator.summarize()
            except Exception:
                pass

        # 嘗試讀取 coco_eval 統計
        try:
            coco_eval = evaluator.coco_eval["bbox"]
            stats = coco_eval.stats  # [AP, AP50, AP75, ...]
            ap = float(stats[0])
            ap50 = float(stats[1])
            ap75 = float(stats[2])
            print(
                f"[Epoch {epoch:3d}] "
                f"Validation avg_loss={avg_val_loss:.4f} | "
                f"AP={ap:.4f}, AP50={ap50:.4f}, AP75={ap75:.4f}"
            )
        except Exception:
            print(f"[Epoch {epoch:3d}] Validation avg_loss={avg_val_loss:.4f}")
    else:
        print(f"[Epoch {epoch:3d}] Validation avg_loss={avg_val_loss:.4f}")

    return avg_val_loss, ap, ap50, ap75


# ------------------------------
#  Main
# ------------------------------

def main():
    args = parse_args()

    # seed
    set_seed(args.seed)

    # Load YAML config
    cfg = YAMLConfig(args.config)

    # 如有指定 output-dir / epochs 就覆蓋 YAML 裡的設定
    if args.output_dir is not None:
        cfg.yaml_cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg.yaml_cfg["epochs"] = args.epochs

    # Build dataloaders / model / criterion / optimizer / scheduler from YAMLConfig
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    model = cfg.model
    print(f"[INFO] Model built:\n{model}")
    criterion = cfg.criterion
    optimizer = cfg.optimizer
    scheduler = cfg.lr_scheduler

    # 盡量從 cfg 拿 evaluator / postprocessor，如果 YAML 有設定:
    base_evaluator = getattr(cfg, "evaluator", None)
    postprocessor = getattr(cfg, "postprocessor", None)

    if base_evaluator is None:
        print("[WARN] cfg.evaluator not found; mAP will NOT be computed.")
    if postprocessor is None:
        print("[WARN] cfg.postprocessor not found; evaluator may not work.")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] Using device: {device}")

    # AMP
    scaler = torch.amp.GradScaler('cuda') if (args.use_amp and device.type == "cuda") else None
    if scaler is not None:
        print("[INFO] AMP enabled.")
    else:
        print("[INFO] AMP disabled.")

    # Epochs
    epochs = cfg.yaml_cfg.get("epochs")
    if epochs is None:
        epochs = cfg.yaml_cfg.get("epoches")
    if epochs is None:
        raise KeyError("Neither 'epochs' nor 'epoches' found in YAML config.")
    print(f"[INFO] Total epochs: {epochs}")

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    best_ap = -1.0
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
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_ap = ckpt.get("best_ap", -1.0)

    output_dir = cfg.yaml_cfg["output_dir"]

    # 如果只想跑一次 val
    if args.test_only:
        print("[INFO] Test-only mode: running validation once.")
        # 這裡直接用 base_evaluator（不用 clone）
        _ = validate(
            model, criterion, val_loader, device,
            epoch=0, evaluator=base_evaluator, postprocessor=postprocessor
        )
        return

    print("[INFO] Starting training...")

    for epoch in range(start_epoch, epochs + 1):
        # ---------------- Train ----------------
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            scaler=scaler,
        )

        # scheduler step (有些 scheduler 需要 val loss 再 step，就自行調整)
        if scheduler is not None:
            scheduler.step()

        # ---------------- 建立這個 epoch 專用的 evaluator ----------------
        if base_evaluator is not None:
            try:
                # 大部分 CocoEvaluator 實作都有 coco_gt / iou_types
                epoch_evaluator = type(base_evaluator)(
                    base_evaluator.coco_gt,
                    iou_types=base_evaluator.iou_types,
                )
            except Exception as e:
                print(f"[WARN] Failed to re-init evaluator: {e}. Reusing base_evaluator.")
                epoch_evaluator = base_evaluator
        else:
            epoch_evaluator = None

        # ---------------- Validation (含 mAP) ----------------
        val_loss, ap, ap50, ap75 = validate(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            evaluator=epoch_evaluator,
            postprocessor=postprocessor,
        )

        # Save checkpoint (含 optimizer / scheduler / scaler / best_val / best_ap)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "best_ap": best_ap,
        }
        save_checkpoint(state, output_dir, epoch)

        # 以 val_loss 或 AP 當作「最好 model」的判準都可以
        improved = False
        if ap is not None and ap > best_ap:
            best_ap = ap
            improved = True
        # elif ap is None and val_loss < best_val_loss:
        #     improved = True

        if improved:
            best_val_loss = min(best_val_loss, val_loss)
            best_path = os.path.join(output_dir, "best.pth")
            torch.save(state, best_path)
            ckpt_dir = "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            best_ckpt = os.path.join(ckpt_dir, "detector_best.pth")
            torch.save(state, best_ckpt)
            print(f"[INFO] New best model saved at: {best_path} (and copied to {best_ckpt})")

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
