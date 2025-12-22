import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser("Pipeline runner: clf -> (det + seg)")
    parser.add_argument("--image", default=None, help="Single image path for inference.")
    parser.add_argument("--inference_folder", default=None, help="If set, run inference over all images in this folder (recursive).")
    parser.add_argument("--clf_ckpt", default="checkpoints/cell_classifier_mae_best.pth")
    # parser.add_argument("--seg_ckpt", required=True)
    parser.add_argument("--det_config", default="configs/detector_rtdetr.yaml")
    parser.add_argument("--det_ckpt", default="checkpoints/detector_best_semi.pth")
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

    cmd = [
        sys.executable,
        "-m",
        "src.inference.infer_pipeline",
    ]

    if args.inference_folder:
        cmd += ["--inference_folder", args.inference_folder]
    else:
        cmd += ["--image", args.image]

    cmd += [
        "--clf_ckpt", args.clf_ckpt,
        "--det_config", args.det_config,
        "--det_ckpt", args.det_ckpt,
        "--clf_thr", str(args.clf_thr),
        "--det_thr", str(args.det_thr),
        "--seg_thr", str(args.seg_thr),
        "--device", args.device,
        "--use_amp" if args.use_amp else "",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
