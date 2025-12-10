"""
Segmentation metrics with optional masking (ignore low-confidence pixels).
"""

import torch


def _flatten(pred, target, valid):
    return pred.view(pred.size(0), -1), target.view(target.size(0), -1), valid.view(valid.size(0), -1)


def iou_score(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    """
    pred, target, valid: (B,1,H,W) float/bool
    """
    pred_f, target_f, valid_f = _flatten(pred, target, valid)
    intersection = (pred_f * target_f * valid_f).sum(dim=1)
    union = ((pred_f + target_f) * valid_f).clamp(max=1.0).sum(dim=1) + eps
    iou = intersection / union
    return iou.mean().item()


def dice_score(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    pred_f, target_f, valid_f = _flatten(pred, target, valid)
    intersection = (pred_f * target_f * valid_f).sum(dim=1)
    denom = (pred_f * valid_f).sum(dim=1) + (target_f * valid_f).sum(dim=1) + eps
    dice = (2 * intersection + eps) / denom
    return dice.mean().item()


def binarize(pred: torch.Tensor, thresh: float = 0.5):
    return (pred >= thresh).float()
