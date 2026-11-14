"""Gaussian heatmap targets and heatmap-to-coordinate decoding.

Convention: heatmap cell i corresponds to input pixel i * stride.
"""
import numpy as np


def generate_heatmaps(keypoints, heatmap_size, stride, sigma):
    """keypoints: (K, 2) input-space (x, y). Returns (K, H, W) float32 heatmaps."""
    num_kp = len(keypoints)
    H = W = heatmap_size
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    heatmaps = np.zeros((num_kp, H, W), dtype=np.float32)
    for k, (x, y) in enumerate(keypoints):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cx, cy = x / stride, y / stride
        heatmaps[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return heatmaps


def decode_heatmaps(heatmaps, stride):
    """(N, K, H, W) -> (N, K, 2) input-space coordinates.

    Argmax with a quarter-pixel shift toward the higher neighbour (simple sub-pixel refinement).
    """
    N, K, H, W = heatmaps.shape
    idx = heatmaps.reshape(N, K, -1).argmax(-1)
    pts = np.zeros((N, K, 2), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            y, x = divmod(int(idx[n, k]), W)
            hm = heatmaps[n, k]
            fx, fy = float(x), float(y)
            if 0 < x < W - 1:
                fx += 0.25 * np.sign(hm[y, x + 1] - hm[y, x - 1])
            if 0 < y < H - 1:
                fy += 0.25 * np.sign(hm[y + 1, x] - hm[y - 1, x])
            pts[n, k] = (fx * stride, fy * stride)
    return pts
