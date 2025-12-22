"""
Minimal detector inference for a single image (RT-DETR + MAE backbone).
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

import src.models.rtdetr_mae_backbone  # noqa: F401 (registers backbone)
import src.models.cell_mae_vit  # noqa: F401 (registers MAE)
from src.rtdetrv2_pytorch.src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser("RT-DETR inference")
    parser.add_argument("--config", type=str, default="configs/detector_rtdetr.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints\detector_best_semi.pth")
    parser.add_argument("--det_thr", type=float, default=0.5)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_model(cfg_path, ckpt_path, device):
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


def main():
    args = parse_args()
    model, postprocessor, device, size = load_model(args.config, args.checkpoint, args.device)

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
        processed = postprocessor(
            outputs,
            orig_target_sizes=torch.tensor([[orig_h, orig_w]], device=device)
        )[0]

    boxes = processed["boxes"].cpu().numpy() if "boxes" in processed else []
    scores = processed["scores"].cpu().numpy() if "scores" in processed else []
    labels = processed["labels"].cpu().numpy() if "labels" in processed else []
    print("Detections:")
    out_dir = os.path.join("output", "det")
    os.makedirs(out_dir, exist_ok=True)
    drawn = img.copy()
    draw = ImageDraw.Draw(drawn)
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
    name, ext = os.path.splitext(os.path.basename(args.image))
    out_path = os.path.join(out_dir, f"{name}_det{ext}")
    drawn.save(out_path)
    print(f"Detection image saved to {out_path} (original resolution)")


if __name__ == "__main__":
    main()
