"""
Minimal detector inference for a single image (RT-DETR + MAE backbone).
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

import src.models.rtdetr_mae_backbone  # noqa: F401 (registers backbone)
import src.models.cell_mae_vit  # noqa: F401 (registers MAE)
from src.rtdetrv2_pytorch.src.core import YAMLConfig
from src.rtdetrv2_pytorch.src.solver.det_engine import postprocess


def parse_args():
    ap = argparse.ArgumentParser("RT-DETR inference")
    ap.add_argument("--config", type=str, default="configs/detector_rtdetr.yaml")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def load_model(cfg_path, ckpt_path, device):
    cfg = YAMLConfig(cfg_path)
    model = cfg.model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    return model, device, cfg


def main():
    args = parse_args()
    model, device, cfg = load_model(args.config, args.checkpoint, args.device)

    size = cfg.yaml_cfg.get("eval_spatial_size", cfg.yaml_cfg.get("input_size", [640, 640]))[0]
    mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = img.size
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        processed = postprocess(
            outputs,
            target_sizes=torch.tensor([[orig_h, orig_w]], device=device)
        )[0]

    boxes = processed["boxes"].cpu().numpy() if "boxes" in processed else []
    scores = processed["scores"].cpu().numpy() if "scores" in processed else []
    labels = processed["labels"].cpu().numpy() if "labels" in processed else []
    print("Detections:")
    for b, s, l in zip(boxes, scores, labels):
        print(f"label={int(l)} score={float(s):.3f} box={b.tolist()}")


if __name__ == "__main__":
    main()
