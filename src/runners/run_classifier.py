import argparse

from src.training.train_classifier import main as train_classifier_main


def parse_args():
    ap = argparse.ArgumentParser("Run classifier trainer")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--use-amp", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    argv = []
    if args.config:
        argv += ["-c", args.config]
    argv += ["--device", args.device]
    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]
    argv += ["--seed", str(args.seed)]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.resume:
        argv += ["--resume", args.resume]
    if args.use_amp:
        argv += ["--use-amp"]
    train_classifier_main(argv)


if __name__ == "__main__":
    main()
