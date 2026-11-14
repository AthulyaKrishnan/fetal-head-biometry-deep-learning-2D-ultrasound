#!/usr/bin/env python
"""STEP 5 - Predict landmarks on new images (no ground truth needed).

    python tools/predict.py --checkpoint runs/hrnet_w32_base/best.pt --input some_folder_or_image.png --out_dir predictions
Writes per image: <name>_overlay.png (points and both axes drawn), and one predictions.json with
the 4 landmarks, the OFD and BPD lengths (px, and mm if --pixel_size is given).
"""
import argparse
from pathlib import Path

import cv2

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.dataset import ImageFolderDataset, make_loader
from src.inference import get_device, load_model, run_inference
from src.io_utils import ensure_dir, list_images, read_gray, save_json
from src.constants import KEYPOINT_NAMES
from src.postprocess import biometry_lengths, draw_landmarks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--pixel_size", type=float, default=None, help="mm per pixel (optional)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    out = ensure_dir(a.out_dir)
    src = Path(a.input)
    files = [src] if src.is_file() else list_images(src)
    device = get_device(a.device)
    model, cfg = load_model(a.checkpoint, device)

    ds = ImageFolderDataset(files, cfg["data"]["input_size"], cfg["preprocess"])
    loader = make_loader(ds, a.batch_size, shuffle=False, num_workers=0)
    res = run_inference(model, loader, device, cfg["data"]["heatmap_stride"])
    ofd, bpd = biometry_lengths(res["pred"])

    results = {}
    for f, name, pts, o, b in zip(files, res["names"], res["pred"], ofd, bpd):
        gray = read_gray(f)
        cv2.imwrite(str(out / f"{Path(name).stem}_overlay.png"), draw_landmarks(gray, pts))
        entry = dict(landmarks_xy={k: [round(float(x), 2), round(float(y), 2)]
                                   for k, (x, y) in zip(KEYPOINT_NAMES, pts)},
                     ofd_len_px=round(float(o), 2), bpd_len_px=round(float(b), 2))
        if a.pixel_size:
            entry.update(ofd_len_mm=round(float(o) * a.pixel_size, 2), bpd_len_mm=round(float(b) * a.pixel_size, 2))
        results[name] = entry
    save_json(results, out / "predictions.json")
    print(f"{len(files)} image(s) processed -> {out}")


if __name__ == "__main__":
    main()
