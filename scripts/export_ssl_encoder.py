import argparse
import torch


def export_encoder(ssl_checkpoint: str, output_path: str):
    ckpt = torch.load(ssl_checkpoint, map_location="cpu")
    if isinstance(ckpt, dict):
        # try common keys
        for k in ["encoder", "model", "student", "teacher"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        else:
            state = ckpt
    else:
        state = ckpt

    torch.save(state, output_path)
    print(f"Encoder weights saved to: {output_path}")


def parse_args():
    ap = argparse.ArgumentParser("Export SSL encoder weights")
    ap.add_argument("--ssl_checkpoint", required=True, help="Path to SSL checkpoint")
    ap.add_argument("--output_encoder_path", required=True, help="Path to save encoder-only weights")
    return ap.parse_args()


def main():
    args = parse_args()
    export_encoder(args.ssl_checkpoint, args.output_encoder_path)


if __name__ == "__main__":
    main()
