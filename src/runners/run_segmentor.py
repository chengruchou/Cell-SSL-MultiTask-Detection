import argparse

from src.training.train_segmentor import main as train_segmentor_main


def parse_args():
    ap = argparse.ArgumentParser("Run segmentation trainer")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--confs", required=True)
    ap.add_argument("--val-images", default=None)
    ap.add_argument("--val-masks", default=None)
    ap.add_argument("--val-confs", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mae-ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="./logs/seg")
    ap.add_argument("--use-amp", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    argv = [
        "--images", args.images,
        "--masks", args.masks,
        "--confs", args.confs,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--device", args.device,
        "--output", args.output,
    ]
    if args.val_images and args.val_masks and args.val_confs:
        argv += ["--val-images", args.val_images, "--val-masks", args.val_masks, "--val-confs", args.val_confs]
    if args.mae_ckpt:
        argv += ["--mae-ckpt", args.mae_ckpt]
    if args.use_amp:
        argv += ["--use-amp"]

    train_segmentor_main(argv)


if __name__ == "__main__":
    main()
