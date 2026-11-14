import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, w_bce=0.5, w_dice=0.5):
        super().__init__()
        self.bce, self.dice = nn.BCEWithLogitsLoss(), DiceLoss()
        self.w_bce, self.w_dice = w_bce, w_dice

    def forward(self, logits, target):
        return self.w_bce * self.bce(logits, target) + self.w_dice * self.dice(logits, target)


def get_loss(name: str):
    name = name.lower()
    if name == "dice":
        return DiceLoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name in ("bce_dice", "bcedice"):
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss '{name}' (use dice | bce | bce_dice)")


# ----------------------------------------------------------------- Hausdorff-distance loss


def _distance_map(mask, max_dist):
    """Per-pixel distance to the nearest boundary pixel of `mask`, clipped at max_dist and scaled to [0, 1].
    Empty mask -> all ones (maximum distance)."""
    if not mask.any():
        return np.ones(mask.shape, np.float32)
    surface = mask & ~binary_erosion(mask, border_value=0)
    d = distance_transform_edt(~surface)
    return (np.minimum(d, max_dist) / max_dist).astype(np.float32)


class HausdorffDistanceLoss(nn.Module):
    """Distance-weighted boundary loss (Karimi & Salcudean, 2019):

        L = mean over pixels of  (p - g)^2 * (D_g^alpha + D_p^alpha)

    p = predicted probability, g = ground-truth mask, D_g / D_p = normalised distance maps of the
    ground-truth mask and of the thresholded prediction (no gradient through the distance maps).
    Errors far from a boundary (missed interior, spurious blobs) are penalised most, which pushes
    the contour towards the correct Hausdorff-type boundary behaviour.
    """

    def __init__(self, alpha=2.0, max_dist=32.0, thr=0.5):
        super().__init__()
        self.alpha, self.max_dist, self.thr = alpha, max_dist, thr

    def forward(self, logits, target):
        p = torch.sigmoid(logits.float())[:, 0]
        g = target.float()[:, 0]
        with torch.no_grad():
            gn = (g > 0.5).cpu().numpy()
            pn = (p > self.thr).cpu().numpy()
            dg = torch.from_numpy(np.stack([_distance_map(m, self.max_dist) for m in gn])).to(p.device)
            dp = torch.from_numpy(np.stack([_distance_map(m, self.max_dist) for m in pn])).to(p.device)
            w = dg ** self.alpha + dp ** self.alpha
        return ((p - g) ** 2 * w).mean()


def hd_weight_at(epoch, max_weight=0.5, warmup_epochs=30):
    """Linear warm-up of the Hausdorff-loss weight. Epochs are 1-based:
    weight is 0 at epoch 1, rises linearly and reaches max_weight at epoch warmup_epochs + 1, then stays."""
    if warmup_epochs <= 0:
        return float(max_weight)
    return float(max_weight) * min(1.0, (epoch - 1) / warmup_epochs)
