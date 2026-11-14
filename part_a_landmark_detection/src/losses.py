"""Heatmap regression loss."""
import torch.nn as nn
import torch.nn.functional as F


class HeatmapMSELoss(nn.Module):
    """MSE between predicted and Gaussian target heatmaps.

    pos_weight > 0 up-weights the region near landmark peaks, which focuses the loss on the
    few pixels that matter. With pos_weight = 0 this is plain MSE.
    """

    def __init__(self, pos_weight=0.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        weight = 1.0 + self.pos_weight * target
        return (weight * (pred - target) ** 2).mean()


class CoordSmoothL1Loss(nn.Module):
    """Smooth-L1 (Huber) loss on normalised coordinates (fractions of the input size) for the
    direct-regression baseline. beta = 0.02 is about 8 px at a 384 px input: errors below that
    are penalised quadratically, larger ones linearly, so outliers do not dominate the gradient."""

    def __init__(self, beta=0.02):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, beta=self.beta)
