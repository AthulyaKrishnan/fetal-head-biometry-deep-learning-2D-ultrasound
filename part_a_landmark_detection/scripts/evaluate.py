#!/usr/bin/env python
"""STEP 4 - Evaluate on a split: landmark errors and biometry (OFD / BPD) errors.

Example:
    python scripts/evaluate.py --checkpoint runs/hrnet_w32_base/best.pt --split test
Outputs in runs/<run>/eval_<split>/ : per_image.csv, summary.json, overlays_worst/*.png
Errors are in original image pixels; mm are added when a pixel spacing is known (from the
checkpoint configuration or --pixel_spacing_mm).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.constants import KEYPOINT_NAMES
from src.dataset import build_datasets, make_loader
from src.inference import get_device, load_model, run_inference
from src.io_utils import ensure_dir, read_gray, save_json
from src.metrics import compute_metrics, nme_per_measure
from src.postprocess import draw_landmarks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="data_processed")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pixel_spacing_mm", type=float, default=None,
                    help="overrides the spacing stored in the checkpoint")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_overlays", type=int, default=12, help="save the N worst cases")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = get_device(a.device)
    model, cfg = load_model(a.checkpoint, device)
    spacing = a.pixel_spacing_mm or cfg["data"].get("pixel_spacing_mm")
    out = Path(a.out_dir) if a.out_dir else Path(a.checkpoint).parent / f"eval_{a.split}"
    ensure_dir(out / "overlays_worst")

    ds = build_datasets(a.data_dir, cfg)[a.split]
    loader = make_loader(ds, a.batch_size, shuffle=False, num_workers=0)
    print(f"{len(ds)} images in '{a.split}' split | device {device}")

    res = run_inference(model, loader, device, cfg["data"]["heatmap_stride"])
    metrics, err = compute_metrics(res["pred"], res["gt"], spacing)
    metrics.update(split=a.split, checkpoint=a.checkpoint)

    per_image = {"image_name": res["names"]}
    for k, name in enumerate(KEYPOINT_NAMES):
        per_image[f"{name}_pred_x"] = res["pred"][:, k, 0]
        per_image[f"{name}_pred_y"] = res["pred"][:, k, 1]
        per_image[f"{name}_gt_x"] = res["gt"][:, k, 0]
        per_image[f"{name}_gt_y"] = res["gt"][:, k, 1]
        per_image[f"{name}_err_px"] = err[:, k]
    per_image["mean_err_px"] = err.mean(axis=1)
    nme = nme_per_measure(res["pred"], res["gt"])
    per_image["nme_ofd"], per_image["nme_bpd"] = nme["ofd"], nme["bpd"]
    per_image["ofd_swapped_better"] = nme["ofd_swap_better"]
    per_image["bpd_swapped_better"] = nme["bpd_swap_better"]
    pd.DataFrame(per_image).to_csv(out / "per_image.csv", index=False)

    summary = dict(metrics)
    summary["n_images"] = len(ds)
    save_json(summary, out / "summary.json")

    # worst cases by mean landmark error, drawn with prediction (filled) and ground truth (hollow)
    order = np.argsort(-err.mean(axis=1))[:a.n_overlays]
    for i in order:
        gray = read_gray(ds.records[i]["image_path"])
        name = res["names"][i]
        cv2_name = f"{err[i].mean():.2f}px_{Path(name).stem}.png"
        cv2.imwrite(str(out / "overlays_worst" / cv2_name), draw_landmarks(gray, res["pred"][i], res["gt"][i]))

    unit = "mm" if "mre_mm" in metrics else "px"
    key = "mre_mm" if unit == "mm" else "mre_px"
    print("\n" + "=" * 56 + f"\nRESULTS ({a.split}, n={len(ds)})\n" + "=" * 56)
    print(f"landmark MRE            : {metrics[key]:.3f} {unit}  (median {metrics.get('median_mm' if unit == 'mm' else 'median_px'):.3f})")
    if unit == "mm":
        print("success within 2 mm     : " + f"{metrics['success_mm']['<= 2.0mm']:.3f}")
    print(f"NME (order-invariant)   : OFD {metrics['nme_ofd_mean']:.4f} | BPD {metrics['nme_bpd_mean']:.4f}  (benchmark definition)")
    print(f"endpoints swapped better: OFD {100*metrics['ofd_swap_better_frac']:.1f} % | BPD {100*metrics['bpd_swap_better_frac']:.1f} % of images")
    print(f"OFD length abs error    : {metrics['ofd_abs_err_px']:.2f} px  ({metrics['ofd_pct_err']:.2f} %)")
    print(f"BPD length abs error    : {metrics['bpd_abs_err_px']:.2f} px  ({metrics['bpd_pct_err']:.2f} %)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
