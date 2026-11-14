"""Preprocessing shared by dataset check, training, evaluation and prediction.

Key design choice: images are *letterboxed* (resized with the aspect ratio kept,
then zero-padded to a square). A plain resize to NxN would squash the ellipse,
which distorts the axis end-points we later need for biometry. The letterbox
metadata lets us map predictions back to the original image exactly.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


# ----------------------------------------------------------------- GT masks
def outline_to_mask(gt_gray: np.ndarray):
    """Convert a ground-truth image to a filled binary mask (uint8 0/1).

    Handles: closed outline (fill it), already-filled mask (keep), outline with
    small gaps (close the gap then fill). Returns (mask, status) where status is
    one of: 'outline', 'already_filled', 'repaired', 'open_outline', 'empty'.
    """
    h, w = gt_gray.shape
    ring = gt_gray > 127
    n_ring = int(ring.sum())
    if n_ring == 0:
        return np.zeros((h, w), np.uint8), "empty"

    filled = binary_fill_holes(ring)
    if filled.sum() > 1.5 * n_ring:
        return filled.astype(np.uint8), "outline"

    if n_ring > 0.03 * h * w:                       # thick solid region already
        return ring.astype(np.uint8), "already_filled"

    # Outline with a gap: thicken it until the gap closes, fill, then shrink back.
    # (A plain morphological close cannot bridge a gap in a 1-pixel-thick line.)
    for k in (3, 5, 9, 13, 21):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thick = cv2.dilate(ring.astype(np.uint8), kernel)
        filled = binary_fill_holes(thick)
        if filled.sum() > 1.5 * thick.sum():
            return cv2.erode(filled.astype(np.uint8), kernel), "repaired"
    return np.zeros((h, w), np.uint8), "open_outline"


# --------------------------------------------------------------- letterboxing
def letterbox_meta(h: int, w: int, size: int) -> dict:
    scale = size / max(h, w)
    nh = min(size, max(1, int(round(h * scale))))
    nw = min(size, max(1, int(round(w * scale))))
    return dict(orig_h=h, orig_w=w, new_h=nh, new_w=nw, scale=scale,
                top=(size - nh) // 2, left=(size - nw) // 2, size=size)


def _pad(arr, meta, value=0):
    s = meta["size"]
    out = np.full((s, s), value, dtype=arr.dtype)
    out[meta["top"]:meta["top"] + meta["new_h"],
        meta["left"]:meta["left"] + meta["new_w"]] = arr
    return out


def apply_clahe(img_u8, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img_u8)


def normalise(img_u8, mode="zscore"):
    x = img_u8.astype(np.float32)
    if mode == "zscore":
        return (x - x.mean()) / (x.std() + 1e-6)
    if mode == "minmax":
        return x / 255.0
    raise ValueError(f"Unknown normalisation: {mode}")


def preprocess_image(gray_u8, image_size=256, clahe=False, norm="zscore"):
    """uint8 HxW -> (float32 SxS letterboxed image, meta)."""
    h, w = gray_u8.shape
    meta = letterbox_meta(h, w, image_size)
    interp = cv2.INTER_AREA if meta["scale"] < 1 else cv2.INTER_LINEAR
    x = cv2.resize(gray_u8, (meta["new_w"], meta["new_h"]), interpolation=interp)
    if clahe:
        x = apply_clahe(x)
    x = normalise(x, norm)            # stats computed on the real image area only
    return _pad(x, meta, 0.0), meta


def preprocess_mask(mask01, meta):
    """uint8 {0,1} HxW -> uint8 {0,1} SxS letterboxed with the same meta."""
    m = cv2.resize(mask01.astype(np.float32) * 255.0,
                   (meta["new_w"], meta["new_h"]), interpolation=cv2.INTER_LINEAR)
    return _pad((m > 127).astype(np.uint8), meta, 0)


def unletterbox_prob(prob_sq, meta):
    """SxS probability map -> original HxW probability map."""
    crop = prob_sq[meta["top"]:meta["top"] + meta["new_h"],
                   meta["left"]:meta["left"] + meta["new_w"]]
    return cv2.resize(crop, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR)
