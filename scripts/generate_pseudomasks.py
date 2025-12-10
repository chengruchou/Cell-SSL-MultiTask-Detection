import argparse

from src.segmentation.cluster_pseudolabel import generate_pseudo_for_dir
from src.segmentation.refine_mask import refine_dir


def parse_args():
    ap = argparse.ArgumentParser("Generate pseudo masks (cluster + refine)")
    ap.add_argument("--feat-dir", type=str, required=True)
    ap.add_argument("--out-mask", type=str, required=True, help="Raw pseudo mask dir")
    ap.add_argument("--out-conf", type=str, required=True, help="Raw confidence dir")
    ap.add_argument("--out-mask-ref", type=str, required=True, help="Refined mask dir")
    ap.add_argument("--out-conf-ref", type=str, required=True, help="Refined confidence dir")
    ap.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "gmm"])
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    generate_pseudo_for_dir(
        feat_dir=args.feat_dir,
        output_mask_dir=args.out_mask,
        output_conf_dir=args.out_conf,
        method=args.method,
        target_size=(640, 640),
        overwrite=args.overwrite,
        random_state=0,
    )
    refine_dir(
        pseudo_mask_dir=args.out_mask,
        conf_dir=args.out_conf,
        out_mask_dir=args.out_mask_ref,
        out_conf_dir=args.out_conf_ref,
        overwrite=True,
    )


if __name__ == "__main__":
    main()
