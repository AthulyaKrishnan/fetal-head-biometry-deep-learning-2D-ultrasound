"""Segmentation metrics. Torch versions (training) and numpy versions (evaluation)."""
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


@torch.no_grad()
def batch_dice_iou(logits, target, thr=0.5, eps=1e-7):
    """Per-sample Dice and IoU tensors of shape (B,)."""
    pred = (torch.sigmoid(logits) > thr).float().flatten(1)
    t = target.flatten(1)
    inter = (pred * t).sum(1)
    dice = (2 * inter + eps) / (pred.sum(1) + t.sum(1) + eps)
    iou = (inter + eps) / (pred.sum(1) + t.sum(1) - inter + eps)
    return dice, iou


def dice_iou_np(pred, gt, eps=1e-7):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps)), float((inter + eps) / (np.logical_or(p, g).sum() + eps))


def hd95(pred, gt, spacing=1.0):
    """95th-percentile symmetric Hausdorff distance (in px, or mm if spacing is mm/px)."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() or not g.any():
        return float("nan")
    ps, gs = p & ~binary_erosion(p), g & ~binary_erosion(g)
    d_to_g, d_to_p = distance_transform_edt(~gs), distance_transform_edt(~ps)
    d = np.concatenate([d_to_g[ps], d_to_p[gs]]) * spacing
    return float(np.percentile(d, 95))
