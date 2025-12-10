"""
End-to-end EM-style loop for unsupervised segmentation.
"""

import os
import argparse
import torch

from src.segmentation.feature_extractor import build_backbone, extract_features
from src.segmentation.cluster_pseudolabel import generate_pseudo_for_dir
from src.segmentation.refine_mask import refine_dir
from src.training.train_segmentor import train as train_segmentor


def parse_args():
    ap = argparse.ArgumentParser("EM loop for unsupervised segmentation")
    ap.add_argument("--unlabeled-dir", type=str, required=True, help="Path to unlabeled images")
    ap.add_argument("--work-dir", type=str, required=True, help="Working directory for caches and outputs")
    ap.add_argument("--mae-ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--em-iters", type=int, default=1)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-cluster", action="store_true")
    ap.add_argument("--skip-refine", action="store_true")
    ap.add_argument("--train-epochs", type=int, default=10)
    return ap.parse_args()


def run_em(args):
    img_dir = args.unlabeled_dir
    work = args.work_dir
    feat_dir = os.path.join(work, "features")
    pseudo_dir = os.path.join(work, "pseudo_raw")
    conf_dir = os.path.join(work, "pseudo_conf")
    refine_mask_dir = os.path.join(work, "pseudo_refined")
    refine_conf_dir = os.path.join(work, "pseudo_conf_refined")
    seg_logs = os.path.join(work, "seg_logs")

    for em in range(1, args.em_iters + 1):
        print(f"[EM {em}] Starting iteration")
        # 1) feature extraction
        if not args.skip_extract:
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms
            tf = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])
            ds = datasets.ImageFolder(img_dir, transform=tf) if os.path.isdir(os.path.join(img_dir, "0")) else None
            if ds is None:
                # fallback: flat folder
                class FlatDataset(torch.utils.data.Dataset):
                    def __init__(self, root, transform):
                        import os
                        from PIL import Image
                        self.paths = [os.path.join(root, f) for f in os.listdir(root)
                                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                        self.transform = transform
                    def __len__(self): return len(self.paths)
                    def __getitem__(self, idx):
                        from PIL import Image
                        p = self.paths[idx]
                        img = Image.open(p).convert("RGB")
                        return self.transform(img), p
                import torch
                ds = FlatDataset(img_dir, tf)
            loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
            backbone = build_backbone(mae_ckpt=args.mae_ckpt, device=args.device, normalize_input=True)
            extract_features(backbone, loader, feat_dir, device=args.device, output_format="pt", overwrite=True)

        # 2) clustering
        if not args.skip_cluster:
            generate_pseudo_for_dir(
                feat_dir=feat_dir,
                output_mask_dir=pseudo_dir,
                output_conf_dir=conf_dir,
                method="kmeans",
                target_size=(640, 640),
                overwrite=True,
                random_state=0,
            )

        # 3) refinement
        if not args.skip_refine:
            refine_dir(
                pseudo_mask_dir=pseudo_dir,
                conf_dir=conf_dir,
                out_mask_dir=refine_mask_dir,
                out_conf_dir=refine_conf_dir,
                gaussian_ksize=5,
                gaussian_sigma=1.0,
                min_size=64,
                overwrite=True,
            )

        # 4) train
        train_args = [
            "--images", img_dir,
            "--masks", refine_mask_dir,
            "--confs", refine_conf_dir,
            "--epochs", str(args.train_epochs),
            "--device", args.device,
            "--output", seg_logs,
        ]
        if args.mae_ckpt:
            train_args.extend(["--mae-ckpt", args.mae_ckpt])
        train_segmentor(train_args)


def main():
    args = parse_args()
    run_em(args)


if __name__ == "__main__":
    main()
