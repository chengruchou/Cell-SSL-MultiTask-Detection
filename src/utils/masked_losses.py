"""
Masked BCE + Dice losses with confidence weighting and ignore_index support.
"""

import torch
import torch.nn.functional as F


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
):
    """
    Args:
        logits: (B,1,H,W) raw logits
        target: (B,1,H,W) binary mask in {0,1}
        confidence: (B,1,H,W) in [0,1]; if using ignore_index, pass conf=1 and set target=ignore where needed
    Returns:
        total_loss, dict of components, valid_ratio
    """
    prob = torch.sigmoid(logits)

    if target.dtype != torch.float32:
        target = target.float()
    if confidence.dtype != torch.float32:
        confidence = confidence.float()

    valid = confidence.clone()
    if ignore_index is not None:
        ignore_mask = (target == ignore_index).float()
        valid = valid * (1.0 - ignore_mask)
        target = target * (1.0 - ignore_mask)

    bce = F.binary_cross_entropy(prob, target, reduction="none")
    bce = (bce * valid).sum() / (valid.sum() + eps)

    prob_flat = prob.view(prob.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    valid_flat = valid.view(valid.size(0), -1)

    intersection = (prob_flat * target_flat * valid_flat).sum(dim=1)
    denom = (prob_flat * valid_flat).sum(dim=1) + (target_flat * valid_flat).sum(dim=1) + eps
    dice = 1.0 - (2.0 * intersection + eps) / denom
    dice = dice.mean()

    total = bce + dice
    valid_ratio = (valid.sum() / valid.numel()).item()
    return total, {"bce": bce.item(), "dice": dice.item()}, valid_ratio
