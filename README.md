# Cell-SSL-MultiTask-Detection

Self-supervised ViT backbones for cell microscopy that support three tasks in one codebase: binary classification, object detection, and pixel-wise segmentation. MAE/DINO pretraining is used to warm-start RT-DETR detection and lightweight classifier/segmentor heads, with utilities to create pseudo-masks from unlabeled data.

## Key Features
- MAE and DINO self-supervised pretraining for microscopy images.
- Classification finetuning on MAE or DINO backbones with evaluation utilities (ROC, confusion matrix, saliency).
- RT-DETR v2 detection with an MAE backbone (COCO-style annotations).
- Unsupervised segmentation: feature extraction → clustering → mask refinement → segmentor training, plus an EM-style loop.
- End-to-end pipeline runner that gates detection/segmentation by classifier confidence.
- Export helpers (ONNX/TRT via bundled RT-DETR toolkit) and data preparation scripts.

## Repository Structure
- `configs/` – YAML defaults for SSL, classifier, detector (RT-DETR), segmentation, pipeline runner.
- `scripts/` – Data prep and pseudo-mask utilities (`extract_features.py`, `generate_pseudomasks.py`, `refresh_pseudomasks.py`, `prepare_data.py`, visualization helpers).
- `src/models/` – MAE ViT backbone, DINO model/loss, classifier/segmentor heads, RT-DETR MAE backbone wrapper.
- `src/datasets/` – Dataset builders for classification (`ImageFolder`), detection (COCO-style), SSL pretraining, pseudo-segmentation, and augmentations.
- `src/training/` – Task-specific trainers: `train_ssl.py`, `train_ssl_mae.py`, `train_classifier.py`, `train_classifier_mae.py`, `train_detector.py`, `train_detector_semi.py`, `train_segmentor.py`.
- `src/segmentation/` – Feature extraction, clustering pseudo-labels, mask refinement, EM loop.
- `src/inference/` – Inference scripts for classifier (DINO/MAE), detector, segmentor, and combined pipeline.
- `src/runners/` – Thin CLI wrappers for the trainers and pipeline.
- `checkpoints/` – Sample outputs/checkpoints (MAE SSL, classifier, detector, segmentor).
- `data/` – Example folder layout for classification, detection (COCO jsons), SSL images.

## Installation & Environment
- Python ≥3.10 and a CUDA-enabled GPU are recommended.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- The RT-DETR code lives in `src/rtdetrv2_pytorch` and is imported by the detector trainer/inference.

## Dataset Preparation
- **SSL pretraining**: place unlabeled images under `data/ssl_pt/` (any flat structure; images are read recursively).
- **Classification**: `data/classification/{train,valid,test}/{positive,negative}/*.png|jpg`. Use `scripts/prepare_data.py` to split a flat folder if needed.
- **Detection**: COCO-style jsons in `data/detection/annotations/{train,valid,test}_annotations.coco.json` with images in matching `data/detection/{train,valid,test}/`.
- **Segmentation (pseudo)**: images plus matching pseudo-masks and confidence maps:
  - Images in one folder (e.g., `data/segmentation/images`).
  - Binary masks (`.png`) and confidences (`.npy` in [0,1]) in parallel folders (see `configs/segmentation.yaml`).
  - Pseudo-masks can be generated from unlabeled images via the feature-extract → cluster → refine utilities (below).

## Training Workflows
All commands assume the repo root as the working directory.

### 1) Self-Supervised Pretraining
<!-- - DINO:
  ```bash
  python src/runners/run_ssl.py --mode dino --config configs/ssl.yaml
  ``` -->
- MAE:
  ```bash
  python src/runners/run_ssl.py --mode mae --config configs/ssl.yaml
  ```
  Checkpoints are saved to `checkpoints/` (e.g., `ssl_mae_best.pth`).

### 2) Classifier Finetuning
<!-- - On DINO backbone:
  ```bash
  python src/runners/run_classifier.py --mode dino --config configs/classifier.yaml
  ``` -->
- On MAE backbone:
  ```bash
  python src/runners/run_classifier.py --mode mae --config configs/classifier.yaml
  ```
  Outputs land in `checkpoints/` (e.g., `cell_classifier_mae_best.pth`).

### 3) Detector (RT-DETR + MAE)
```bash
python src/runners/run_detector.py --config configs/detector_rtdetr.yaml
```
- Uses COCO loaders defined in the YAML; checkpoints saved under `logs/detector_rtdetr/` and mirrored to `checkpoints/detector_best.pth`.
- Semi-supervised variant: `python src/runners/run_detector_semi.py --config configs/detector_rtdetr.yaml` (uses `semi_supervised` block in the config).

### 4) Unsupervised Segmentation
Direct training with existing pseudo-labels:
```bash
python src/runners/run_segmentor.py --config configs/segmentation.yaml --extra-args \
  --images data/segmentation/images \
  --masks data/segmentation/masks \
  --confs data/segmentation/confidences \
  --mae-ckpt checkpoints/ssl_mae_best.pth \
  --output logs/seg
```

End-to-end EM loop from unlabeled images:
```bash
python src/segmentation/em_loop.py \
  --unlabeled-dir data/segmentation/images \
  --work-dir workdir/seg_em \
  --mae-ckpt checkpoints/ssl_mae_best.pth \
  --em-iters 1 --train-epochs 10
```
This runs feature extraction → clustering → refinement → segmentor training, caching artifacts in `work-dir`.

Pseudo-mask generation utilities (manual steps):
```bash
# 1) Extract patch features with MAE backbone
python scripts/extract_features.py --images data/segmentation/images --out work/features --mae-ckpt checkpoints/ssl_mae_best.pth

# 2) Cluster into pseudo-masks and refine
python scripts/generate_pseudomasks.py \
  --feat-dir work/features \
  --out-mask work/pseudo_raw \
  --out-conf work/pseudo_conf \
  --out-mask-ref work/pseudo_refined \
  --out-conf-ref work/pseudo_conf_refined

# 3) (Optional) Refresh masks after retraining
python scripts/refresh_pseudomasks.py --work-dir work --mae-ckpt checkpoints/ssl_mae_best.pth
```

## Inference & Evaluation
- **Classifier (MAE)**:
  ```bash
  python -m src.inference.infer_classifier_mae \
    --data_root data/classification/test \
    --ckpt checkpoints/cell_classifier_mae_best.pth \
    --img_size 640 \
    --out_csv logs/mae_cls_test_predictions.csv
  ```
  Produces accuracy, confusion matrix, ROC (binary), and optional saliency heatmaps.

- **Detector**:
  ```bash
  python -m src.inference.infer_detector \
    --config configs/detector_rtdetr.yaml \
    --checkpoint checkpoints/detector_best.pth \
    --image path/to/image.jpg
  ```
  Prints boxes/scores/labels in original image coordinates.

- **Segmentation**:
  ```bash
  python -m src.inference.infer_segmentation \
    --images path/to/image_dir \
    --checkpoint checkpoints/segmentor_best.pth \
    --output out_masks/
  ```
  Saves binary masks (`.png`) alongside inputs.

- **Full pipeline (classify → detect/segment if positive)**:
  ```bash
  python -m src.inference.infer_pipeline \
    --image path/to/image.jpg \
    --clf-ckpt checkpoints/cell_classifier_mae_best.pth \
    --seg-ckpt checkpoints/segmentor_best.pth \
    --det-config configs/detector_rtdetr.yaml \
    --det-ckpt checkpoints/detector_best.pth \
    --clf-thr 0.5
  ```
  Outputs classification probability, detection boxes, and a saved mask (`*_seg.png`).

## Configuration Notes
- Each task has a YAML config in `configs/`; CLI flags can override values.
- Detector config (`detector_rtdetr.yaml`) includes MAE backbone settings and dataset paths; adjust `train_dataloader`/`val_dataloader` and `eval_spatial_size` to match your data.
- Segmentation config (`segmentation.yaml`) records image/mask/conf paths and training hyperparameters.
- Pipeline config (`pipeline.yaml`) collects the three checkpoints and runner path for convenience.

## Expected Inputs/Outputs
- Inputs: RGB microscopy images (default 640×640 resize), optional COCO annotations for detection, pseudo-masks/confidence maps for segmentation.
- Outputs: Classification scores/metrics, detection boxes with scores, binary segmentation masks, and saved checkpoints under `checkpoints/` or `logs/`.

## License & Citation
- License: MIT (see `LICENSE`).
- Citation: please cite this repository and the underlying MAE/DINO/RT-DETR papers if you use this code in academic work.
