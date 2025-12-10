import argparse
import subprocess
import sys


def parse_args():
    ap = argparse.ArgumentParser("Run semi-supervised detector trainer")
    ap.add_argument("--config", type=str, default="configs\detector_rtdetr.yaml")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional args passed through")
    ap.add_argument("--use-amp ", action="store_true", help="Use automatic mixed precision if supported")
    return ap.parse_args()


def main():
    args = parse_args()
    cmd = [sys.executable, "-m", "src.training.train_detector_semi", "-c", args.config]
    if args.extra_args:
        cmd += args.extra_args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
