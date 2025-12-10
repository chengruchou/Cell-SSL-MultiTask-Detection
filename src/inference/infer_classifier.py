# infer_classifier.py
import argparse
import os
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.utils.common import get_normalization

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

from src.models.ssl_dino import DinoModel
from src.models.cell_classifier import CellClassifier


# --------------------------------------------------
# Dataset / Transform
# --------------------------------------------------
def build_test_loader(data_root, img_size=224, batch_size=64, num_workers=4):
    """
    Test-time transform：記得要跟訓練時的 resize / normalize 對齊。
    這裡假設你訓練時是：
      - Resize 到固定大小
      - ToTensor
      - ImageNet normalize
    """
    mean, std = get_normalization("classification")
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = datasets.ImageFolder(root=data_root, transform=test_tf)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, dataset


# --------------------------------------------------
# Model building / loading
# --------------------------------------------------
def build_model(num_classes: int, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    依照 train_classifier 的設計：
      - 先建一個 DinoModel 當 ssl backbone
      - 再包成 CellClassifier
      - 最後 load classifier 的 state_dict
    """
    ssl_model = DinoModel()  # 權重會由 classifier ckpt 一起 load 進來
    model = CellClassifier(ssl_model=ssl_model, num_classes=num_classes)

    print(f"[INFO] Loading classifier checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    # train_classifier.py 是用 torch.save(model.state_dict(), path)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


# --------------------------------------------------
# Metrics & Visualization
# --------------------------------------------------
def save_confusion_matrix(cm: np.ndarray, class_names, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # 在格子裡面印數字
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix to: {out_path}")


def save_roc_curve(y_true, y_scores, pos_idx: int, class_names, out_path: str):
    """
    只支援 binary classification。
    y_true: list[int] (class indices)
    y_scores: list[float] (predicted probability of positive class)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    y_true_bin = np.array([1 if y == pos_idx else 0 for y in y_true], dtype=np.int64)
    y_scores = np.array(y_scores, dtype=np.float32)

    fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved ROC curve to: {out_path}")


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    """
    將 normalize 過的 (3, H, W) tensor 轉成 [0,1] 的 HxWx3 numpy array。
    """
    if tensor.ndim != 3:
        raise ValueError("Expected tensor shape (3, H, W)")
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0.0, 1.0)
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    return img


def compute_saliency_heatmap(model: torch.nn.Module,
                             image: torch.Tensor,
                             target_class: int,
                             device: torch.device) -> np.ndarray:
    """
    Gradient-based saliency:
      - 對 target_class 的 logit 做 backward
      - 將 |grad| 在 channel 維度做 max，得到 (H, W) heatmap
    """
    model.eval()
    img = image.unsqueeze(0).to(device)
    img.requires_grad_(True)

    logits = model(img)
    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    grads = img.grad[0]  # shape: (C, H, W)
    saliency = grads.abs().max(dim=0)[0]  # (H, W)

    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-8)
    saliency = saliency.detach().cpu().numpy()
    return saliency


def save_heatmaps_for_samples(model: torch.nn.Module,
                              samples,
                              class_names,
                              device: torch.device,
                              out_dir: str,
                              max_heatmaps: int = 50):
    """
    samples: list of dicts:
      {
        "img": img_tensor (3,H,W),   # 已做 normalize
        "pred": int,                 # predicted class idx
        "true": int,                 # true class idx
        "path": str                  # original file path
      }
    """
    os.makedirs(out_dir, exist_ok=True)
    n = min(len(samples), max_heatmaps)

    print(f"[INFO] Generating saliency heatmaps for first {n} samples...")
    for i in range(n):
        sample = samples[i]
        img_tensor = sample["img"]
        pred_idx = sample["pred"]
        true_idx = sample["true"]
        path = sample["path"]

        heatmap = compute_saliency_heatmap(model, img_tensor, pred_idx, device)
        rgb = tensor_to_rgb_image(img_tensor)

        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        out_path = os.path.join(out_dir, f"{stem}_pred-{class_names[pred_idx]}_true-{class_names[true_idx]}.png")

        # 疊圖
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(rgb)
        ax.imshow(heatmap, cmap="jet", alpha=0.4)
        ax.axis("off")
        ax.set_title(f"pred: {class_names[pred_idx]} / true: {class_names[true_idx]}")

        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"[INFO] Saved heatmaps to dir: {out_dir}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for cell classifier on test set, with metrics & visualizations."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/classification/test",
        help="ImageFolder root for TEST set (must contain subfolders per class).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/cell_classifier_best.pth",
        help="Path to trained classifier checkpoint (state_dict).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for test loader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="num_workers for DataLoader.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Resize images to (img_size, img_size).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device: "cuda" or "cpu".',
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="logs/cls_test_predictions.csv",
        help="CSV to save per-image predictions.",
    )
    parser.add_argument(
        "--cm_path",
        type=str,
        default="logs/confusion_matrix.png",
        help="Where to save confusion matrix figure.",
    )
    parser.add_argument(
        "--roc_path",
        type=str,
        default="logs/roc_curve.png",
        help="Where to save ROC curve figure (binary only).",
    )
    parser.add_argument(
        "--heatmap_dir",
        type=str,
        default="logs/heatmaps",
        help="Directory to save saliency heatmaps.",
    )
    parser.add_argument(
        "--max_heatmaps",
        type=int,
        default=50,
        help="Max number of heatmaps to generate (starting from first samples).",
    )
    parser.add_argument(
        "--no_heatmap",
        action="store_true",
        help="If set, do NOT generate heatmaps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    # 1. Dataset & DataLoader
    test_loader, test_dataset = build_test_loader(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    class_to_idx = test_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"[INFO] Test classes: {class_names}")
    print(f"[INFO] Total test images: {len(test_dataset)}")

    num_classes = len(class_names)

    # 2. Model
    model = build_model(num_classes=num_classes, ckpt_path=args.ckpt, device=device)

    # 3. Inference loop: collect preds, labels, probs
    all_labels = []
    all_preds = []
    all_probs = []  # shape: (N, num_classes)
    all_paths = []
    heatmap_samples = []  # for saliency

    model.eval()
    ptr = 0  # 用來對應 dataset.samples 的 index

    with torch.no_grad():
        for imgs, labels in test_loader:
            batch_size = imgs.size(0)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            # 對應到檔名
            for b in range(batch_size):
                path, _ = test_dataset.samples[ptr + b]
                all_paths.append(path)

                # 存起來之後做 heatmap（先放 CPU，不要佔 GPU）
                heatmap_samples.append({
                    "img": test_dataset[ptr + b][0].cpu(),  # transform 後的 tensor
                    "true": labels[b].item(),
                    "pred": preds[b].item(),
                    "path": path,
                })

            ptr += batch_size

    total = len(all_labels)
    correct = sum(int(p == t) for p, t in zip(all_preds, all_labels))
    acc = correct / total if total > 0 else 0.0
    print(f"\n[RESULT] Test Accuracy: {acc * 100:.2f}% (total {total} samples)\n")

    # 4. Confusion matrix + classification report
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    print("Confusion matrix (rows = true, cols = pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    save_confusion_matrix(cm, class_names, args.cm_path)

    # 5. ROC curve (if binary)
    if num_classes == 2:
        # 找到 positive class index：優先找名字叫 "positive" 的
        if "positive" in class_names:
            pos_idx = class_names.index("positive")
        else:
            pos_idx = 1  # fallback

        # all_probs: list[list[float]] -> prob of pos class
        y_scores = [p[pos_idx] for p in all_probs]
        save_roc_curve(all_labels, y_scores, pos_idx, class_names, args.roc_path)
    else:
        print("[WARN] num_classes != 2, skip ROC curve drawing.")

    # 6. Save CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ["image_path", "true_idx", "true_label", "pred_idx", "pred_label"]
    for i, cname in enumerate(class_names):
        header.append(f"prob_class_{i}_{cname}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for path, t, p, prob_vec in zip(all_paths, all_labels, all_preds, all_probs):
            row = [path, t, class_names[t], p, class_names[p]]
            row.extend(prob_vec)
            writer.writerow(row)

    print(f"[INFO] Saved per-image predictions to: {args.out_csv}")

    # 7. Heatmaps
    if not args.no_heatmap:
        save_heatmaps_for_samples(
            model=model,
            samples=heatmap_samples,
            class_names=class_names,
            device=device,
            out_dir=args.heatmap_dir,
            max_heatmaps=args.max_heatmaps,
        )
    else:
        print("[INFO] Skipped heatmap generation (--no_heatmap set).")


if __name__ == "__main__":
    main()
