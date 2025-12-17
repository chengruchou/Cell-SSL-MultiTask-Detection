import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser("Pipeline runner: clf -> (det + seg)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--clf-ckpt", required=True)
    parser.add_argument("--seg-ckpt", required=True)
    parser.add_argument("--det-config", default="configs/detector_rtdetr.yaml")
    parser.add_argument("--det-ckpt", required=True)
    parser.add_argument("--clf-thr", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use automatic mixed precision if supported")
    return parser.parse_args()

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
        "--use-amp",
        str(args.use_amp)
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
