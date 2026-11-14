#!/usr/bin/env python
"""STEP 4 - Evaluate on a split: segmentation metrics + biometry (ellipse, 2 axis end-point pairs, HC).

Example:
    python tools/quick_evaluate.py --checkpoint runs/unet_base32/best.pt --split test
Outputs in --out_dir (default runs/<run>/eval_<split>/):
    per_image.csv, summary.json, overlays_worst/*.png
Metrics are computed at ORIGINAL image resolution. Distances are in mm if the split CSV has
pixel sizes, otherwise in pixels (the unit is recorded in the summary).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.inference import get_device, load_model, predict_prob
from src.io_utils import read_gray, read_split_csv, save_json
from src.metrics import dice_iou_np, hd95
from src.postprocess import clean_mask, measure_mask, point_pair_error


def draw_overlay(img, gt_mask, pred_mask, pred_m, gt_m):
    ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for m, col in ((gt_mask, (0, 255, 0)), (pred_mask, (0, 0, 255))):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, col, 1)
    if gt_m:
        for p in gt_m["major_pts"]: cv2.circle(ov, tuple(int(round(v)) for v in p), 4, (0, 255, 0), -1)
    if pred_m:
        for p in pred_m["major_pts"]: cv2.circle(ov, tuple(int(round(v)) for v in p), 4, (0, 255, 255), -1)
        for p in pred_m["minor_pts"]: cv2.circle(ov, tuple(int(round(v)) for v in p), 4, (255, 255, 0), -1)
    return ov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="data_processed")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    ap.add_argument("--n_overlays", type=int, default=12, help="save the N worst cases")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    out = Path(a.out_dir) if a.out_dir else Path(a.checkpoint).parent / f"eval_{a.split}"
    (out / "overlays_worst").mkdir(parents=True, exist_ok=True)
    device = get_device(a.device)
    model, pre_cfg = load_model(a.checkpoint, device)
    records = read_split_csv(Path(a.data_dir) / f"{a.split}.csv")
    print(f"{len(records)} images in '{a.split}' split | device {device}")

    rows, cache = [], {}
    for r in tqdm(records):
        img = read_gray(r["image_path"])
        gt = (read_gray(r["mask_path"]) > 127).astype(np.uint8)
        prob = predict_prob(model, img, pre_cfg, device, a.tta)
        pred = clean_mask(prob > a.threshold)
        sp = r["pixel_size_mm"] or 1.0                     # mm/px, or 1 -> pixels
        dice, iou = dice_iou_np(pred, gt)
        pm, gm = measure_mask(pred), measure_mask(gt)
        row = dict(stem=r["stem"], dice=dice, iou=iou, hd95=hd95(pred, gt, sp), unit="mm" if r["pixel_size_mm"] else "px")
        if pm and gm:
            maj_err, (ma, mb) = point_pair_error(pm["major_pts"] * 1.0, gm["major_pts"] * 1.0)
            min_err, (na, nb) = point_pair_error(pm["minor_pts"] * 1.0, gm["minor_pts"] * 1.0)
            row.update(major_pt_err=maj_err * sp, major_pt_err_a=ma * sp, major_pt_err_b=mb * sp,
                       minor_pt_err=min_err * sp, minor_pt_err_a=na * sp, minor_pt_err_b=nb * sp,
                       major_len_pred=pm["major_len"] * sp, major_len_gt=gm["major_len"] * sp,
                       minor_len_pred=pm["minor_len"] * sp, minor_len_gt=gm["minor_len"] * sp,
                       hc_pred=pm["hc"] * sp, hc_gt=gm["hc"] * sp, hc_abs_diff=abs(pm["hc"] - gm["hc"]) * sp)
            if r["official_hc_mm"] and r["pixel_size_mm"]:
                row["hc_abs_diff_vs_official_mm"] = abs(pm["hc"] * sp - r["official_hc_mm"])
        rows.append(row)
        cache[r["stem"]] = (img, gt, pred, pm, gm)

    # ---- write per-image table
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    import csv
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # ---- summary
    summary = dict(checkpoint=a.checkpoint, split=a.split, n=len(rows), threshold=a.threshold, tta=a.tta,
                   distance_unit=rows[0]["unit"] if rows else "px", metrics={})
    for k in keys:
        if k in ("stem", "unit"):
            continue
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=float)
        if len(v):
            summary["metrics"][k] = dict(mean=float(v.mean()), std=float(v.std()), median=float(np.median(v)),
                                         min=float(v.min()), max=float(v.max()), n=int(len(v)))
    summary["n_failed_ellipse_fit"] = sum(1 for r in rows if "major_pt_err" not in r)
    save_json(summary, out / "summary.json")

    # ---- worst-case overlays
    for row in sorted(rows, key=lambda r: r["dice"])[:a.n_overlays]:
        img, gt, pred, pm, gm = cache[row["stem"]]
        cv2.imwrite(str(out / "overlays_worst" / f"{row['dice']:.3f}_{row['stem']}.png"), draw_overlay(img, gt, pred, pm, gm))

    u = summary["distance_unit"]
    print("\n" + "=" * 56 + f"\nRESULTS ({a.split}, n={len(rows)})\n" + "=" * 56)
    for k, label in (("dice", "Dice"), ("iou", "IoU"), ("hd95", f"HD95 [{u}]"),
                     ("major_pt_err", f"major-axis end-point error [{u}]"),
                     ("minor_pt_err", f"minor-axis end-point error [{u}]"), ("hc_abs_diff", f"HC abs diff [{u}]")):
        m = summary["metrics"].get(k)
        if m:
            print(f"{label:38s} {m['mean']:8.3f} +/- {m['std']:.3f}   (median {m['median']:.3f})")
    print(f"ellipse fit failed on {summary['n_failed_ellipse_fit']} images")
    print(f"saved -> {out}")
    print("overlay colours: green = GT contour / GT major points, red = prediction, yellow = pred major pts, cyan = pred minor pts")


if __name__ == "__main__":
    main()
