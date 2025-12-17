import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser("Run semi-supervised detector trainer")
    parser.add_argument("--config", type=str, default="configs\detector_rtdetr.yaml")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional args passed through")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use automatic mixed precision if supported")
    return parser.parse_args()

def main():
    args = parse_args()
    cmd = [sys.executable, "-m", "src.training.train_detector_semi", "--config", args.config]
    if args.use_amp:
        cmd += ["--use-amp"]
    if args.extra_args:
        cmd += args.extra_args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
