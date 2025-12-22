"""
Pipeline inference:
- run SSL backbone + classifier
- if positive, run detector and segmentor
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

import src.models.rtdetr_mae_backbone  # noqa: F401
import src.models.cell_mae_vit  # noqa: F401
from src.models.cell_mae_vit import MAE, CellViTBackbone, CellClassifier, CellSegmenter
from src.rtdetrv2_pytorch.src.core import YAMLConfig
from src.segmentation.transforms import build_default_transforms
from src.utils.common import get_normalization


def load_classifier(mae_ckpt, num_classes, device):
    mean, std = get_normalization("classification")
    mae = MAE(img_size=640, patch_size=16, in_chans=3, embed_dim=384, depth=6, num_heads=6,
              decoder_dim=192, decoder_depth=4, mask_ratio=0.75)
    if mae_ckpt:
        ckpt = torch.load(mae_ckpt, map_location="cpu")
        mae.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    backbone = CellViTBackbone(mae=mae, freeze_encoder=True, normalize_input=True, mean=mean, std=std)
    clf = CellClassifier(backbone, num_classes=num_classes)
    clf.to(device).eval()
    return clf, mean, std


def load_segmentor(seg_ckpt, device):
    mean, std = get_normalization("segmentation")
    mae = MAE(img_size=640, patch_size=16, in_chans=3, embed_dim=384, depth=6, num_heads=6,
              decoder_dim=192, decoder_depth=4, mask_ratio=0.75)
    backbone = CellViTBackbone(mae=mae, freeze_encoder=True, normalize_input=True, mean=mean, std=std)
    seg = CellSegmenter(backbone=backbone, num_classes=1, upsample_factor=16)
    ckpt = torch.load(seg_ckpt, map_location="cpu")
    seg.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    seg.to(device).eval()
    return seg, mean, std


def load_detector(cfg_path, ckpt_path, device):
    cfg = YAMLConfig(cfg_path)
    model = cfg.model
    postprocessor = cfg.postprocessor
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    postprocessor.to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    postprocessor.eval()
    size = cfg.yaml_cfg.get("eval_spatial_size", cfg.yaml_cfg.get("input_size", [640, 640]))[0]
    return model, postprocessor, device, size


def parse_args():
    parser = argparse.ArgumentParser("Pipeline inference: clf -> (det + seg)")
    parser.add_argument("--image", default=None, help="Single image path for inference.")
    parser.add_argument("--inference_folder", default=None, help="If set, run inference over all images in this folder (recursive).")
    parser.add_argument("--clf_ckpt", default="checkpoints/cell_classifier_mae_best.pth")
    # parser.add_argument("--seg_ckpt", required=True)
    parser.add_argument("--det_config", default="configs/detector_rtdetr.yaml")
    parser.add_argument("--det_ckpt", default="checkpoints/detector_best.pth")
    parser.add_argument("--clf_thr", type=float, default=0.4)
    parser.add_argument("--det_thr", type=float, default=0.6)
    parser.add_argument("--seg_thr", type=float, default=0.6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision for inference.")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.inference_folder is None and not args.image:
        raise ValueError("Either --image or --inference_folder must be provided.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # classifier
    clf, clf_mean, clf_std = load_classifier(args.clf_ckpt, num_classes=2, device=device)
    clf_tf = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(clf_mean, clf_std),
    ])

    # detector
    det_model, det_postprocessor, det_device, det_size = load_detector(args.det_config, args.det_ckpt, device)
    det_tf = transforms.Compose([
        transforms.Resize((det_size, det_size)),
        transforms.ToTensor(),
        transforms.Normalize(get_normalization("detection")[0], get_normalization("detection")[1]),
    ])

    # segmentor
    # seg_model, _, _ = load_segmentor(args.seg_ckpt, device=device)
    # seg_tf = build_default_transforms(size=(640, 640), hflip_p=0.0)

    out_dir = os.path.join("output", "det")
    os.makedirs(out_dir, exist_ok=True)

    def run_single_image(img_path: str):
        print(f"\n[IMAGE] {img_path}")
        img = Image.open(img_path).convert("RGB")

        # classification
        x_clf = clf_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = clf(x_clf)
            prob = torch.softmax(logits, dim=1)
            positive = prob[0, 1].item() >= args.clf_thr
        print(f"Classification positive prob={prob[0,1].item():.3f}, positive={positive}")
        if not positive:
            return

        # detection
        x_det = det_tf(img).unsqueeze(0).to(det_device)
        orig_w, orig_h = img.size
        with torch.no_grad():
            det_out = det_model(x_det)
            orig_sizes = torch.tensor([[orig_w, orig_h]], device=det_device, dtype=torch.float32)
            det_proc = det_postprocessor(det_out, orig_target_sizes=orig_sizes)[0]
        print("Detections:")
        drawn = img.copy()
        draw = ImageDraw.Draw(drawn)
        boxes = det_proc.get("boxes", torch.empty(0)).cpu().numpy()
        scores = det_proc.get("scores", torch.empty(0)).cpu().numpy()
        labels = det_proc.get("labels", torch.empty(0)).cpu().numpy()
        for b, s, l in zip(boxes, scores, labels):
            if s > args.det_thr:
                print(f"label={int(l)} score={float(s):.3f} box={b.tolist()}")
                xmin, ymin, xmax, ymax = b
                xmin = max(0, min(xmin, orig_w - 1))
                xmax = max(0, min(xmax, orig_w - 1))
                ymin = max(0, min(ymin, orig_h - 1))
                ymax = max(0, min(ymax, orig_h - 1))
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
                draw.text((xmin, ymin), f"{int(l)} {s:.2f}", fill="red")
        name, ext = os.path.splitext(os.path.basename(img_path))
        out_path = os.path.join(out_dir, f"{name}_det{ext}")
        drawn.save(out_path)
        print(f"Detection image saved to {out_path} (original resolution)")

    if args.inference_folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for root, _, files in os.walk(args.inference_folder):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    run_single_image(os.path.join(root, fn))
    else:
        run_single_image(args.image)

    # segmentation
    # dummy_mask = Image.new("L", img.size, 0)
    # dummy_conf = Image.new("L", img.size, 255)
    # x_seg, _, _ = seg_tf(img, dummy_mask, dummy_conf)
    # x_seg = x_seg.unsqueeze(0).to(device)
    # with torch.no_grad():
    #     seg_logits = seg_model(x_seg)
    #     seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    # import cv2
    # import numpy as np
    # mask_bin = (seg_prob >= 0.5).astype(np.uint8) * 255
    # out_mask = os.path.splitext(args.image)[0] + "_seg.png"
    # cv2.imwrite(out_mask, mask_bin)
    # print(f"Segmentation mask saved to {out_mask}")


if __name__ == "__main__":
    main()
