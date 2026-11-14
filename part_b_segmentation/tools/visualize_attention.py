#!/usr/bin/env python
"""Show WHERE the attention layers look.  For a chosen query position, draws the attention weights
over the image for every self-attention and cross-attention layer of a trained model.

    python tools/visualize_attention.py --checkpoint runs/attn_run/best.pt --image some.png \
           --query_xy 0.5 0.5 --out attention.png

--query_xy : query position as fractions of the image (x, y) in 0..1; default = image centre.
Self-attention : 'if I am at the red dot, which positions do I gather information from?'
Cross-attention: 'the decoder at the red dot asks the encoder skip features - which positions answer?'
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference import get_device, load_model
from src.io_utils import read_gray
from src.preprocessing import preprocess_image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--query_xy", type=float, nargs=2, default=[0.5, 0.5])
    ap.add_argument("--out", default="attention.png")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = get_device(a.device)
    model, pre = load_model(a.checkpoint, device)
    gray = read_gray(a.image)
    x, meta = preprocess_image(gray, pre["image_size"], pre["clahe"], pre["norm"])
    mods = model.attention_modules()
    if not mods:
        sys.exit("This checkpoint has no attention layers.")
    for _, m in mods:
        m.store_attention = True
    with torch.no_grad():
        model(torch.from_numpy(x)[None, None].to(device))

    S = pre["image_size"]
    qx, qy = a.query_xy
    panels = []
    for name, m in mods:
        A = m.last_attn[0].cpu().numpy()                       # (Nq, Nk)
        Nq, Nk = A.shape
        hq = int(round(Nq ** 0.5)); hk = int(round(Nk ** 0.5))
        qi = min(int(qy * hq), hq - 1) * hq + min(int(qx * hq), hq - 1)
        w = A[qi].reshape(hk, hk)
        panels.append((name, cv2.resize(w, (S, S), interpolation=cv2.INTER_LINEAR), hq))

    cols = min(4, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (name, w, hq) in zip(axs.ravel(), panels):
        ax.imshow(x, cmap="gray")
        ax.imshow(w, cmap="jet", alpha=0.5)
        ax.plot([qx * S], [qy * S], "wo", ms=7, mec="r")
        ax.set_title(f"{name}\n({hq}x{hq} query grid)", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
