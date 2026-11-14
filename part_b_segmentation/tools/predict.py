#!/usr/bin/env python
"""STEP 5 - Predict on new images (no ground truth needed).

    python tools/predict.py --checkpoint runs/unet_base32/best.pt --input some_folder_or_image.png --out_dir predictions
Writes per image: <stem>_mask.png, <stem>_overlay.png and one predictions.json with the
ellipse, the 2 major-axis end-points, the 2 minor-axis end-points, lengths and HC (px, and mm if --pixel_size).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.inference import get_device, load_model, predict_prob
from src.io_utils import IMG_EXTS, list_images, read_gray, save_json
from src.postprocess import clean_mask, measure_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--pixel_size", type=float, default=None, help="mm per pixel (optional)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    src = Path(a.input)
    files = [src] if src.is_file() else list_images(src)
    device = get_device(a.device)
    model, pre_cfg = load_model(a.checkpoint, device)

    results = {}
    for f in files:
        img = read_gray(f)
        mask = clean_mask(predict_prob(model, img, pre_cfg, device, a.tta) > a.threshold)
        m = measure_mask(mask)
        cv2.imwrite(str(out / f"{f.stem}_mask.png"), mask * 255)
        ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if m is None:
            results[f.name] = {"error": "no head found / ellipse fit failed"}
        else:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnts, -1, (0, 0, 255), 1)
            cv2.line(ov, *[tuple(int(round(v)) for v in p) for p in m["major_pts"]], (0, 255, 255), 1)
            cv2.line(ov, *[tuple(int(round(v)) for v in p) for p in m["minor_pts"]], (255, 255, 0), 1)
            for p in np.vstack([m["major_pts"], m["minor_pts"]]):
                cv2.circle(ov, tuple(int(round(v)) for v in p), 4, (0, 255, 0), -1)
            res = dict(major_axis_points_xy=m["major_pts"].round(2).tolist(),
                       minor_axis_points_xy=m["minor_pts"].round(2).tolist(),
                       major_len_px=m["major_len"], minor_len_px=m["minor_len"], hc_px=m["hc"], ellipse=m["ellipse"])
            if a.pixel_size:
                res.update(major_len_mm=m["major_len"] * a.pixel_size, minor_len_mm=m["minor_len"] * a.pixel_size,
                           hc_mm=m["hc"] * a.pixel_size)
            results[f.name] = res
        cv2.imwrite(str(out / f"{f.stem}_overlay.png"), ov)
    save_json(results, out / "predictions.json")
    print(f"{len(files)} image(s) processed -> {out}")


if __name__ == "__main__":
    main()
