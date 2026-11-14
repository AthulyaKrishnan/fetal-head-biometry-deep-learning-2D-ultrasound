#!/usr/bin/env python
"""Show WHAT each landmark heatmap looks like on a chosen image.

    python tools/visualize_heatmaps.py --checkpoint runs/hrnet_w32_base/best.pt --image some.png --out heatmaps.png

One panel per landmark: the network's predicted heatmap (sigmoid-free, shown as a colour map on
the image) and the decoded landmark (red cross). A sharp, single peak at the right structure means
the model is confident; a diffuse or double-peaked map flags an ambiguous landmark.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.constants import KEYPOINT_NAMES
from src.heatmap import decode_heatmaps
from src.inference import get_device, load_model
from src.io_utils import read_gray
from src.preprocessing import preprocess_gray, to_model_input


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="heatmaps.png")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = get_device(a.device)
    model, cfg = load_model(a.checkpoint, device)
    if cfg["model"].get("type", "heatmap") == "regression":
        raise SystemExit("the regression baseline has no heatmaps; visualise a heatmap checkpoint instead")
    S, stride = cfg["data"]["input_size"], cfg["data"]["heatmap_stride"]
    gray = read_gray(a.image)
    H0, W0 = gray.shape
    x = preprocess_gray(cv2.resize(gray, (S, S), interpolation=cv2.INTER_LINEAR), **cfg["preprocess"])
    with torch.no_grad():
        hm = model(torch.from_numpy(to_model_input(x))[None].to(device))[0].cpu().numpy()
    pts = decode_heatmaps(hm[None], stride)[0] * np.array([W0 / S, H0 / S])

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for k, name in enumerate(KEYPOINT_NAMES):
        heat = cv2.resize(hm[k], (W0, H0), interpolation=cv2.INTER_LINEAR)
        ax[k].imshow(gray, cmap="gray")
        ax[k].imshow(heat, cmap="jet", alpha=0.45)
        ax[k].plot(pts[k, 0], pts[k, 1], "r+", ms=14, mew=2)
        ax[k].set_title(f"{name}  (peak {hm[k].max():.2f})")
        ax[k].axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
