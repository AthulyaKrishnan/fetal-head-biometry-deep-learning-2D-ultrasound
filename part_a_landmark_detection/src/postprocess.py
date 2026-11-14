"""Biometry lengths from landmarks, and drawing of landmarks for visual checks."""
import cv2
import numpy as np

from .constants import BPD_IDX, KEYPOINT_NAMES, OFD_IDX

COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0)]  # BGR per landmark


def biometry_lengths(pts):
    """pts: (N, 4, 2) -> OFD and BPD lengths (N,) in the same units as pts."""
    ofd = np.linalg.norm(pts[:, OFD_IDX[0]] - pts[:, OFD_IDX[1]], axis=-1)
    bpd = np.linalg.norm(pts[:, BPD_IDX[0]] - pts[:, BPD_IDX[1]], axis=-1)
    return ofd, bpd


def _pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_landmarks(gray, pred, gt=None):
    """gray: uint8 (H, W). pred / gt: (4, 2) in original pixels. Returns a BGR image.
    Filled dots = prediction, hollow white circles = ground truth (when given)."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for a, b in (OFD_IDX, BPD_IDX):
        cv2.line(img, _pt(pred[a]), _pt(pred[b]), (255, 255, 255), 1)
    for k, name in enumerate(KEYPOINT_NAMES):
        cv2.circle(img, _pt(pred[k]), 4, COLORS[k], -1)
        x, y = _pt(pred[k])
        cv2.putText(img, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[k], 1, cv2.LINE_AA)
        if gt is not None:
            cv2.circle(img, _pt(gt[k]), 7, (255, 255, 255), 1)
    return img
