import argparse
import subprocess
import sys


def parse_args():
    ap = argparse.ArgumentParser("Run classifier trainer")
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--mode", choices=["dino", "mae"], default="mae", help="Which trainer to run")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional args passed through")
    ap.add_argument("--use-amp ", action="store_true", help="Use automatic mixed precision if supported")
    return ap.parse_args()


def main():
    args = parse_args()
    module = "src.training.train_classifier" if args.mode == "dino" else "src.training.train_classifier_mae"
    cmd = [sys.executable, "-m", module]
    if args.config:
        cmd += ["-c", args.config]
    if args.extra_args:
        cmd += args.extra_args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
