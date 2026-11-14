"""Per-image metrics for the full evaluation CSV.

Conventions (all binary masks are at ORIGINAL image resolution unless stated otherwise):
  * An empty ground truth AND an empty prediction count as a perfect match (Dice = IoU = 1).
  * Ratios with an empty denominator are NaN (e.g. precision when nothing was predicted).
  * Surface distances are NaN when either mask is empty (no surface to compare to).
  * Distances are returned in mm when a pixel spacing is given, otherwise in pixels.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_div(a, b):
    return float(a) / float(b) if b > 0 else float("nan")


def confusion(pred, gt):
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = int(np.logical_and(p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    return tp, tn, fp, fn


def overlap_metrics(pred, gt):
    """Dice, IoU, sensitivity, specificity, precision, F1 from the confusion matrix."""
    tp, tn, fp, fn = confusion(pred, gt)
    if tp + fp + fn == 0:                       # both masks empty: perfect agreement
        dice = iou = 1.0
    else:
        dice = _safe_div(2 * tp, 2 * tp + fp + fn)
        iou = _safe_div(tp, tp + fp + fn)
    sens = _safe_div(tp, tp + fn)               # recall
    spec = _safe_div(tn, tn + fp)
    prec = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * prec * sens, prec + sens) if np.isfinite(prec) and np.isfinite(sens) else float("nan")
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, dice=dice, iou=iou,
                sensitivity=sens, specificity=spec, precision=prec, f1_score=f1)


def ranking_metrics(prob, gt):
    """Pixel-wise ROC-AUC and PR-AUC (average precision). NaN if the GT has only one class."""
    y = np.asarray(gt, bool).ravel()
    s = np.asarray(prob, np.float64).ravel()
    if y.all() or (~y).all():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _surface(mask):
    m = np.asarray(mask, bool)
    return m & ~binary_erosion(m, border_value=0)


def surface_distances(pred, gt, spacing=1.0):
    """Distances (in units of `spacing`) from each predicted surface pixel to the nearest GT
    surface pixel, and vice versa. Returns (d_pred_to_gt, d_gt_to_pred) or None if a mask is empty."""
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    if not p.any() or not g.any():
        return None
    ps, gs = _surface(p), _surface(g)
    d_to_gs = distance_transform_edt(~gs) * spacing     # distance of every pixel to GT surface
    d_to_ps = distance_transform_edt(~ps) * spacing
    return d_to_gs[ps], d_to_ps[gs]


def boundary_metrics(pred, gt, spacing=1.0, tolerance_mm=2.0):
    """Hausdorff distance, HD95, average surface distance (ASD) and normalised surface distance (NSD).

    NSD = share of all surface points (both directions) that lie within `tolerance_mm` of the other
    contour (Nikolov et al. 2018). Tolerance is given in mm; if spacing is 1 it is taken in pixels.
    """
    d = surface_distances(pred, gt, spacing)
    if d is None:
        return dict(hausdorff_distance=float("nan"), hausdorff_95=float("nan"),
                    average_surface_distance=float("nan"), normalised_surface_distance=float("nan"))
    dp, dg = d
    all_d = np.concatenate([dp, dg])
    tol = tolerance_mm
    nsd = float(((dp <= tol).sum() + (dg <= tol).sum()) / (len(dp) + len(dg)))
    return dict(hausdorff_distance=float(all_d.max()),
                hausdorff_95=float(np.percentile(all_d, 95)),
                average_surface_distance=float((dp.mean() + dg.mean()) / 2.0),
                normalised_surface_distance=nsd)
