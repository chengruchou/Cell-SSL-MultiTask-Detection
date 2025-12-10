# src/utils/pseudo_label.py
from typing import List, Optional

import torch


def generate_pseudo_labels(
    outputs: dict,
    score_thr: float = 0.5,
    max_pseudo: Optional[int] = None,
) -> List[dict]:
    """
    Convert RT-DETR outputs into pseudo targets (cxcywh normalized).
    - outputs: model outputs containing 'pred_logits' and 'pred_boxes'
    - score_thr: minimum class probability to keep a query
    - max_pseudo: optional cap on boxes per image
    """
    logits = outputs["pred_logits"]  # [B, Q, C]
    boxes = outputs["pred_boxes"]    # [B, Q, 4], cxcywh normalized

    probs = logits.sigmoid()
    scores, labels = probs.max(dim=-1)  # [B, Q]

    pseudo_targets = []
    B, Q = scores.shape
    for b in range(B):
        score_b = scores[b]
        label_b = labels[b]
        box_b = boxes[b]

        keep = score_b >= score_thr
        if keep.any():
            score_keep = score_b[keep]
            label_keep = label_b[keep]
            box_keep = box_b[keep]

            if max_pseudo is not None and score_keep.numel() > max_pseudo:
                topk_vals, topk_idx = torch.topk(score_keep, k=max_pseudo)
                label_keep = label_keep[topk_idx]
                box_keep = box_keep[topk_idx]
        else:
            label_keep = torch.zeros((0,), device=logits.device, dtype=torch.long)
            box_keep = torch.zeros((0, 4), device=logits.device, dtype=logits.dtype)

        pseudo_targets.append({
            "labels": label_keep.long(),
            "boxes": box_keep,
        })

    return pseudo_targets
