import argparse
import subprocess
import sys


def parse_args():
    ap = argparse.ArgumentParser("Run SSL pretraining")
    ap.add_argument("--mode", choices=["dino", "mae"], default="mae", help="Which trainer to run")
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--data-root", type=str, default=None, help="Override data root")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional args passed through")
    return ap.parse_args()


def main():
    args = parse_args()
    module = "src.training.train_ssl" if args.mode == "dino" else "src.training.train_ssl_mae"
    cmd = [sys.executable, "-m", module]
    if args.config:
        cmd += ["--config", args.config]
    if args.data_root:
        cmd += ["--data_root", args.data_root]
    if args.extra_args:
        cmd += args.extra_args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
