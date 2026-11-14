#!/usr/bin/env python
"""STEP 4b - Full evaluation of ONE model on a split. Run it once per model (attention and baseline).

Example:
    python scripts/evaluate.py --checkpoint runs/attn_default/best.pt --name attention_unet --split test
    python scripts/evaluate.py --checkpoint runs/base/best.pt       --name baseline_unet  --split test

Writes into  eval_full/<name>/ :
    <name>_subject_metrics.csv   one row per image, exactly the requested columns
    <name>_biometry_points.csv   one row per image: OFD/BPD end-point errors and HC error (Part B)
    <name>_summary.csv           mean / std / median / IQR / min / max of every metric
    <name>_summary.json          run info, failure counts, timing, parameter count
    pooled_curves.npz            pooled ROC and PR points (network grid) for the report figures
    overlays_worst/              the N worst cases by Dice (green = GT, red = prediction)

Conventions (also written into the summary JSON). All areas are foreground pixel counts:
    gt_area_original_px        GT area at ORIGINAL image resolution (reference)
    gt_area_network_px         GT area on the network input grid (letterboxed, image_size x image_size)
    pred_area_network_px       predicted area on the same network grid (threshold + cleaning applied)
    area_diff_network_px       pred_area_network_px - gt_area_network_px (signed; < 0 = under-segmentation)
    abs_area_diff_network_px   absolute value of area_diff_network_px
    diceCEloss                 per-image training loss (same loss as --loss in scripts/train.py), on the grid
    Distances are in mm when the split CSV has pixel_size_mm, otherwise in pixels (see summary).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.full_metrics import boundary_metrics, overlap_metrics, ranking_metrics
from src.inference import get_device, load_model, torch_load
from src.io_utils import read_gray, read_split_csv, save_json
from src.losses import get_loss
from src.postprocess import clean_mask, measure_mask, point_pair_error
from src.preprocessing import preprocess_image, preprocess_mask, unletterbox_prob

SUBJECT_COLUMNS = [
    "subject_id", "gt_area_original_px", "gt_area_network_px", "pred_area_network_px", "area_diff_network_px",
    "abs_area_diff_network_px", "diceCEloss", "dice", "iou", "sensitivity", "specificity", "precision",
    "f1_score", "roc_auc", "pr_auc", "hausdorff_distance_mm", "hausdorff_95_mm",
    "average_surface_distance_mm", "normalised_surface_distance",
]
BIOMETRY_COLUMNS = [
    "subject_id", "ellipse_fit_ok", "ofd_pred_mm", "ofd_gt_mm", "bpd_pred_mm", "bpd_gt_mm",
    "ofd_point_error_mm", "bpd_point_error_mm", "hc_pred_mm", "hc_gt_mm", "hc_abs_error_mm",
]


def evaluate_model(checkpoint, data_dir="data_processed", split="test", name=None, out_root="eval_full",
                   threshold=0.5, tolerance_mm=2.0, n_overlays=12, pooled_sample=20000, device="auto", seed=0):
    """Run the full evaluation and return (subject_df_rows, biometry_rows, summary dict)."""
    name = name or Path(checkpoint).parent.name
    out = Path(out_root) / name
    (out / "overlays_worst").mkdir(parents=True, exist_ok=True)
    dev = get_device(device)
    model, pre_cfg = load_model(checkpoint, dev)
    ck_args = torch_load(checkpoint, torch.device("cpu"))["config"]["args"]
    criterion = get_loss(ck_args.get("loss", "bce_dice"))
    n_params = sum(p.numel() for p in model.parameters())

    records = read_split_csv(Path(data_dir) / f"{split}.csv")
    if not records:
        raise ValueError(f"no records in {data_dir}/{split}.csv")
    has_mm = all(r["pixel_size_mm"] for r in records)
    rng = np.random.default_rng(seed)

    subj_rows, bio_rows, cache, pooled_y, pooled_s = [], [], {}, [], []
    times = []
    S = pre_cfg["image_size"]
    for r in tqdm(records, desc=f"{name} [{split}]"):
        img = read_gray(r["image_path"])
        gt = (read_gray(r["mask_path"]) > 127).astype(np.uint8)
        sp = r["pixel_size_mm"] if has_mm else 1.0

        # ---- forward pass on the network grid (timed)
        t0 = time.perf_counter()
        x, meta = preprocess_image(img, S, pre_cfg["clahe"], pre_cfg["norm"])
        with torch.no_grad():
            logits = model(torch.from_numpy(x)[None, None].to(dev))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        prob_sq = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        gt_grid = preprocess_mask(gt, meta)                              # S x S, same letterbox as training
        with torch.no_grad():
            loss = criterion(logits.float().cpu(), torch.from_numpy(gt_grid.astype(np.float32))[None, None]).item()

        pred_grid = clean_mask(prob_sq > threshold)
        gt_grid_fg = int(gt_grid.sum())
        pred_grid_fg = int(pred_grid.sum())

        # ---- evaluation at original resolution
        prob = unletterbox_prob(prob_sq, meta)
        pred = clean_mask(prob > threshold)
        ov = overlap_metrics(pred, gt)
        roc, pr = ranking_metrics(prob, gt)
        bnd = boundary_metrics(pred, gt, sp, tolerance_mm)
        gt_area_original = int(gt.sum())

        # pooled curves: subsample pixels on the network grid to keep the file small
        idx = rng.choice(prob_sq.size, size=min(pooled_sample // max(len(records), 1) + 1, prob_sq.size), replace=False)
        pooled_y.append(gt_grid.ravel()[idx]); pooled_s.append(prob_sq.ravel()[idx])

        subj_rows.append({
            "subject_id": r["stem"],
            "gt_area_original_px": gt_area_original,
            "gt_area_network_px": gt_grid_fg,
            "pred_area_network_px": pred_grid_fg,
            "area_diff_network_px": pred_grid_fg - gt_grid_fg,
            "abs_area_diff_network_px": abs(pred_grid_fg - gt_grid_fg),
            "diceCEloss": loss,
            "dice": ov["dice"], "iou": ov["iou"], "sensitivity": ov["sensitivity"],
            "specificity": ov["specificity"], "precision": ov["precision"], "f1_score": ov["f1_score"],
            "roc_auc": roc, "pr_auc": pr,
            "hausdorff_distance_mm": bnd["hausdorff_distance"],
            "hausdorff_95_mm": bnd["hausdorff_95"],
            "average_surface_distance_mm": bnd["average_surface_distance"],
            "normalised_surface_distance": bnd["normalised_surface_distance"],
        })

        # ---- biometry points (ellipse end-points), Part B
        pm, gm = measure_mask(pred), measure_mask(gt)
        brow = dict(subject_id=r["stem"], ellipse_fit_ok=bool(pm and gm))
        if pm and gm:
            ofd_err, _ = point_pair_error(pm["major_pts"] * 1.0, gm["major_pts"] * 1.0)
            bpd_err, _ = point_pair_error(pm["minor_pts"] * 1.0, gm["minor_pts"] * 1.0)
            brow.update(ofd_pred_mm=pm["major_len"] * sp, ofd_gt_mm=gm["major_len"] * sp,
                        bpd_pred_mm=pm["minor_len"] * sp, bpd_gt_mm=gm["minor_len"] * sp,
                        ofd_point_error_mm=ofd_err * sp, bpd_point_error_mm=bpd_err * sp,
                        hc_pred_mm=pm["hc"] * sp, hc_gt_mm=gm["hc"] * sp,
                        hc_abs_error_mm=abs(pm["hc"] - gm["hc"]) * sp)
        bio_rows.append(brow)
        cache[r["stem"]] = (img, gt, pred, pm, gm, ov["dice"])

    # ---- write subject-level CSV (exact column order)
    subj_path = out / f"{name}_subject_metrics.csv"
    with open(subj_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBJECT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(subj_rows)

    bio_path = out / f"{name}_biometry_points.csv"
    with open(bio_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BIOMETRY_COLUMNS, extrasaction="ignore", restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(bio_rows)

    # ---- summary statistics
    import pandas as pd
    df = pd.DataFrame(subj_rows)
    num = [c for c in SUBJECT_COLUMNS if c != "subject_id"]
    stats = []
    for c in num:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) == 0:
            continue
        stats.append(dict(metric=c, n=int(len(v)), mean=v.mean(), std=v.std(), median=v.median(),
                          q1=v.quantile(0.25), q3=v.quantile(0.75), iqr=v.quantile(0.75) - v.quantile(0.25),
                          min=v.min(), max=v.max()))
    pd.DataFrame(stats).to_csv(out / f"{name}_summary.csv", index=False)

    # pooled curves for ROC / PR figures
    from sklearn.metrics import precision_recall_curve, roc_curve
    yy, ss = np.concatenate(pooled_y), np.concatenate(pooled_s)
    fpr, tpr, _ = roc_curve(yy, ss)
    pr_p, pr_r, _ = precision_recall_curve(yy, ss)
    np.savez_compressed(out / "pooled_curves.npz", fpr=fpr, tpr=tpr, precision=pr_p, recall=pr_r)

    # ---- failure counts
    n_empty_pred = sum(1 for r in subj_rows if r["pred_area_network_px"] == 0)
    n_empty_gt = sum(1 for r in subj_rows if r["gt_area_original_px"] == 0)
    summary = dict(
        checkpoint=str(checkpoint), name=name, split=split, n_images=len(subj_rows),
        threshold=threshold, distance_unit="mm" if has_mm else "px (no pixel_size_mm in split CSV)",
        boundary_tolerance=f"{tolerance_mm} {'mm' if has_mm else 'px'}", loss=ck_args.get("loss", "bce_dice"),
        image_size=S, n_params=int(n_params),
        empty_prediction_images=int(n_empty_pred), empty_gt_images=int(n_empty_gt),
        ellipse_fit_failed=int(sum(1 for b in bio_rows if not b["ellipse_fit_ok"])),
        inference_ms_per_image_mean=float(np.mean(times) * 1000), inference_ms_per_image_std=float(np.std(times) * 1000),
        device=str(dev),
    )
    save_json(summary, out / f"{name}_summary.json")

    # ---- worst-case overlays
    for stem, (img, gt, pred, pm, gm, d) in sorted(cache.items(), key=lambda kv: kv[1][5])[:n_overlays]:
        cv2.imwrite(str(out / "overlays_worst" / f"{d:.3f}_{stem}.png"), _overlay(img, gt, pred, pm, gm))

    return subj_rows, bio_rows, summary


def _overlay(img, gt, pred, pm, gm):
    ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for m, col in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, col, 1)
    if pm:
        for p in pm["major_pts"]:
            cv2.circle(ov, tuple(int(round(v)) for v in p), 4, (0, 255, 255), -1)
        for p in pm["minor_pts"]:
            cv2.circle(ov, tuple(int(round(v)) for v in p), 4, (255, 255, 0), -1)
    return ov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", default=None, help="model name used in file names (default: run folder name)")
    ap.add_argument("--data_dir", default="data_processed")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tolerance_mm", type=float, default=2.0, help="tolerance for normalised surface distance")
    ap.add_argument("--n_overlays", type=int, default=12)
    ap.add_argument("--out_root", default="eval_full")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    _, _, s = evaluate_model(a.checkpoint, a.data_dir, a.split, a.name, a.out_root, a.threshold,
                             a.tolerance_mm, a.n_overlays, device=a.device)
    print(f"\n{s['name']} | {s['n_images']} images | unit {s['distance_unit']} | "
          f"{s['inference_ms_per_image_mean']:.1f} ms/img | {s['n_params']/1e6:.2f} M params")
    print(f"empty prediction: {s['empty_prediction_images']} | empty GT: {s['empty_gt_images']} | "
          f"ellipse fit failed: {s['ellipse_fit_failed']}")
    print(f"written to {a.out_root}/{s['name']}/")


if __name__ == "__main__":
    main()
