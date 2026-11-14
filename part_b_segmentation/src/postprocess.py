"""Mask clean-up, ellipse fitting and extraction of the biometry points.

Biometry points = the 4 end-points of the fitted ellipse's two axes:
  * major axis end-points  (~ occipito-frontal diameter, OFD)
  * minor axis end-points  (~ biparietal diameter, BPD)
If your assignment defines the two points differently, change only
`ellipse_geometry` / `point_pair_error`.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def clean_mask(mask, close_ksize=5):
    """Close small gaps, fill holes, keep only the largest connected component."""
    m = (mask > 0).astype(np.uint8)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = binary_fill_holes(m).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return np.zeros_like(m)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == keep).astype(np.uint8)


def fit_ellipse(mask):
    """Least-squares ellipse fit to the outer contour. Returns dict or None."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(c)       # d1, d2 = FULL axis lengths
    return dict(cx=cx, cy=cy, d1=d1, d2=d2, angle=ang)


def ellipse_geometry(ell):
    """Axis end-points (x, y) and lengths. major_pts/minor_pts have shape (2, 2)."""
    t = np.deg2rad(ell["angle"])
    c = np.array([ell["cx"], ell["cy"]])
    u = np.array([np.cos(t), np.sin(t)])               # direction of the d1 axis
    v = np.array([-np.sin(t), np.cos(t)])              # direction of the d2 axis
    p1 = np.stack([c - u * ell["d1"] / 2, c + u * ell["d1"] / 2])
    p2 = np.stack([c - v * ell["d2"] / 2, c + v * ell["d2"] / 2])
    if ell["d1"] >= ell["d2"]:
        return dict(major_pts=p1, minor_pts=p2, major_len=ell["d1"], minor_len=ell["d2"])
    return dict(major_pts=p2, minor_pts=p1, major_len=ell["d2"], minor_len=ell["d1"])


def hc_ramanujan(major_len, minor_len):
    """Ellipse circumference (Ramanujan approximation) from FULL axis lengths."""
    a, b = major_len / 2.0, minor_len / 2.0
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def point_pair_error(pred_pts, gt_pts):
    """End-points have no inherent order, so take the better of the 2 pairings.

    Returns (mean_error, (err_point_a, err_point_b)) in the units of the inputs.
    """
    d_same = [np.linalg.norm(pred_pts[0] - gt_pts[0]), np.linalg.norm(pred_pts[1] - gt_pts[1])]
    d_swap = [np.linalg.norm(pred_pts[0] - gt_pts[1]), np.linalg.norm(pred_pts[1] - gt_pts[0])]
    best = d_same if np.mean(d_same) <= np.mean(d_swap) else d_swap
    return float(np.mean(best)), (float(best[0]), float(best[1]))


def measure_mask(mask):
    """mask -> dict with ellipse, axis points, lengths, HC (all in pixels), or None."""
    ell = fit_ellipse(mask)
    if ell is None:
        return None
    geo = ellipse_geometry(ell)
    geo["ellipse"] = ell
    geo["hc"] = hc_ramanujan(geo["major_len"], geo["minor_len"])
    return geo
