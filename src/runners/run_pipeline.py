import argparse
import subprocess
import sys


def parse_args():
    ap = argparse.ArgumentParser("Pipeline runner: clf -> (det + seg)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--clf-ckpt", required=True)
    ap.add_argument("--seg-ckpt", required=True)
    ap.add_argument("--det-config", default="configs/detector_rtdetr.yaml")
    ap.add_argument("--det-ckpt", required=True)
    ap.add_argument("--clf-thr", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-amp ", action="store_true", help="Use automatic mixed precision if supported")
    return ap.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "src.inference.infer_pipeline",
        "--image",
        args.image,
        "--clf-ckpt",
        args.clf_ckpt,
        "--seg-ckpt",
        args.seg_ckpt,
        "--det-config",
        args.det_config,
        "--det-ckpt",
        args.det_ckpt,
        "--clf-thr",
        str(args.clf_thr),
        "--device",
        args.device,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
