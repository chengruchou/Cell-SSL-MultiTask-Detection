import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser("Run segmentation trainer")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional args passed through")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use automatic mixed precision if supported")
    return parser.parse_args()

def main():
    args = parse_args()
    cmd = [sys.executable, "-m", "src.training.train_segmentor"]
    if args.config:
        cmd += ["-c", args.config]
    if args.use_amp:
        cmd += ["--use_amp"]
    if args.extra_args:
        cmd += args.extra_args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
