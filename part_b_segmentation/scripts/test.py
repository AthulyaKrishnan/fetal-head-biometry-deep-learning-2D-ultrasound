#!/usr/bin/env python
"""Evaluate a trained fetal head segmentation model on any test dataset (for external judges).

Works with either model file:
    * a training checkpoint        runs/<run>/best.pt
    * a full exported model        "Model Weights/hypothesis_1_full_saved_model.pth"  (from tools/export_model.py)

Inputs
    --weights    model file (see above)
    --images     folder with the ultrasound images (png, jpg, bmp, tif)
    --gt         folder with the ground-truth annotations (outlines or filled masks)
    --gt_suffix  text between an image name and its GT name: "_Annotation" by default ("" if names are equal)
    --pixel_csv  optional table with columns: filename, pixel size (mm), head circumference (mm).
                 If given for every image, all distances are reported in mm; otherwise in pixels.

Outputs, all written directly into --result_dir (no sub-folders):
    <name>_subject_metrics.csv     one row per image with the requested metric columns
    <name>_biometry_points.csv     OFD / BPD end-point errors and head circumference error per image
    <name>_summary.csv             mean, SD, median, IQR, min, max of every metric
    <name>_summary.json            run information (weights, settings, counts, timing)
    <name>_summary.txt             the summary printed to the console
    <name>_skipped.csv             images that could not be evaluated, with the reason (only if any)
    <name>_pred_<image>.png        predicted mask per image, original resolution  (unless --no_masks)
    <name>_worst_<rank>_dice<d>_<image>.png   worst cases: green = GT contour, red = prediction
    pooled_curves.npz              data for the ROC / PR figures
    <name>_figures/                report figures and results_table.md (only with --make_figures)

Example
    python scripts/test.py --weights "Model Weights/hypothesis_1_full_saved_model.pt" \\
        --images /data/judge_test/images --gt /data/judge_test/gt --gt_suffix _Annotation \\
        --pixel_csv /data/judge_test/pixel_sizes.csv --result_dir results/my_run --make_figures

Run it from the part's root folder (the one that contains src/ and scripts/):  python scripts/test.py ...
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]  # repo-part root (contains src/)
sys.path.insert(0, str(ROOT))

from src.full_metrics import boundary_metrics, overlap_metrics, ranking_metrics  # noqa: E402
from src.inference import get_device, torch_load  # noqa: E402
from src.io_utils import find_pairs, load_pixel_table, read_gray, save_json  # noqa: E402
from src.losses import get_loss  # noqa: E402
from src.model import UNet  # noqa: E402
from src.postprocess import clean_mask, measure_mask, point_pair_error  # noqa: E402
from src.preprocessing import outline_to_mask, preprocess_image, preprocess_mask, unletterbox_prob  # noqa: E402

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
# (column, label, higher_is_better) for the printed summary; {u} is replaced by the distance unit
SUMMARY_ROWS = [
    ("dice", "Dice", True),
    ("iou", "IoU", True),
    ("sensitivity", "Sensitivity", True),
    ("specificity", "Specificity", True),
    ("precision", "Precision", True),
    ("roc_auc", "ROC-AUC", True),
    ("pr_auc", "PR-AUC", True),
    ("hausdorff_95_mm", "HD95 [{u}]", False),
    ("average_surface_distance_mm", "Avg surface distance [{u}]", False),
    ("normalised_surface_distance", "Normalised surface distance", True),
    ("abs_area_diff_network_px", "Abs area error [px]", False),
    ("diceCEloss", "Loss (BCE+Dice)", False),
]


# ----------------------------------------------------------------------------- model
def load_model(path, device, loss_override=None):
    """Return (model, preprocessing config, loss name, origin text, number of parameters)."""
    obj = torch_load(path, device)
    if isinstance(obj, dict) and "model_state" in obj:                      # training checkpoint
        cfg = obj["config"]
        model = UNet(**cfg["model"])
        model.load_state_dict(obj["model_state"])
        pre = dict(cfg["preprocess"])
        loss = cfg.get("args", {}).get("loss", "bce_dice")
        origin = "training checkpoint"
    elif isinstance(obj, torch.nn.Module):                                  # exported full model
        model = obj
        pre = getattr(model, "preprocess_cfg", None)
        if pre is None:
            raise ValueError("this exported model has no preprocessing settings; "
                             "use the training checkpoint (best.pt) instead")
        pre = dict(pre)
        loss = "bce_dice"
        origin = "exported full model"
    else:
        raise ValueError(f"unrecognised model file: {path}")
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, pre, loss_override or loss, origin, n_params


@torch.no_grad()
def forward(model, img, pre, device):
    """uint8 image -> (logits on the network grid, letterbox meta)."""
    x, meta = preprocess_image(img, pre["image_size"], pre["clahe"], pre["norm"])
    logits = model(torch.from_numpy(x)[None, None].to(device))
    return logits.float().cpu(), meta


def overlay(img, gt, pred, pm):
    ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for m, col in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, col, 1)
    if pm:
        for p in pm["major_pts"]:
            cv2.circle(ov, tuple(int(round(v)) for v in p), 4, (0, 255, 255), -1)
    return ov


# ----------------------------------------------------------------------------- evaluation
def evaluate(args, model, pre, loss_name, pairs, pix, out, name):
    device = next(model.parameters()).device
    criterion = get_loss(loss_name)
    has_mm = bool(pix) and all(p.name in pix for p, _ in pairs)
    rng = np.random.default_rng(args.seed)

    subj, bio, skipped, times, pooled_y, pooled_s = [], [], [], [], [], []
    worst = []                                              # (dice, stem, img, gt, pred, pm) for the n worst

    for img_p, gt_p in tqdm(pairs, desc=f"evaluating {name}"):
        img = read_gray(img_p)
        gt_raw = read_gray(gt_p)
        if gt_raw.shape != img.shape:
            skipped.append(dict(subject_id=img_p.stem, reason="ground-truth size differs from image"))
            continue
        gt, status = outline_to_mask(gt_raw)
        if status in ("empty", "open_outline"):
            skipped.append(dict(subject_id=img_p.stem, reason=f"ground truth unusable: {status}"))
            continue
        stem = img_p.stem
        sp = pix[img_p.name]["pixel_size_mm"] if has_mm else 1.0

        t0 = time.perf_counter()
        logits, meta = forward(model, img, pre, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        prob_sq = torch.sigmoid(logits)[0, 0].numpy()
        gt_grid = preprocess_mask(gt, meta)
        loss = criterion(logits, torch.from_numpy(gt_grid.astype(np.float32))[None, None]).item()
        pred_grid = clean_mask(prob_sq > args.threshold)

        prob = unletterbox_prob(prob_sq, meta)
        pred = clean_mask(prob > args.threshold)
        ov = overlap_metrics(pred, gt)
        roc, pr = ranking_metrics(prob, gt)
        bnd = boundary_metrics(pred, gt, sp, args.tolerance_mm)
        gt_fg, pr_fg = int(gt_grid.sum()), int(pred_grid.sum())

        idx = rng.choice(prob_sq.size, size=min(4000, prob_sq.size), replace=False)
        pooled_y.append(gt_grid.ravel()[idx]); pooled_s.append(prob_sq.ravel()[idx])

        subj.append({
            "subject_id": stem,
            "gt_area_original_px": int(gt.sum()),
            "gt_area_network_px": gt_fg,
            "pred_area_network_px": pr_fg,
            "area_diff_network_px": pr_fg - gt_fg,
            "abs_area_diff_network_px": abs(pr_fg - gt_fg),
            "diceCEloss": loss,
            "dice": ov["dice"], "iou": ov["iou"], "sensitivity": ov["sensitivity"],
            "specificity": ov["specificity"], "precision": ov["precision"], "f1_score": ov["f1_score"],
            "roc_auc": roc, "pr_auc": pr,
            "hausdorff_distance_mm": bnd["hausdorff_distance"],
            "hausdorff_95_mm": bnd["hausdorff_95"],
            "average_surface_distance_mm": bnd["average_surface_distance"],
            "normalised_surface_distance": bnd["normalised_surface_distance"],
        })

        pm, gm = measure_mask(pred), measure_mask(gt)
        brow = dict(subject_id=stem, ellipse_fit_ok=bool(pm and gm))
        if pm and gm:
            ofd_err, _ = point_pair_error(pm["major_pts"] * 1.0, gm["major_pts"] * 1.0)
            bpd_err, _ = point_pair_error(pm["minor_pts"] * 1.0, gm["minor_pts"] * 1.0)
            brow.update(ofd_pred_mm=pm["major_len"] * sp, ofd_gt_mm=gm["major_len"] * sp,
                        bpd_pred_mm=pm["minor_len"] * sp, bpd_gt_mm=gm["minor_len"] * sp,
                        ofd_point_error_mm=ofd_err * sp, bpd_point_error_mm=bpd_err * sp,
                        hc_pred_mm=pm["hc"] * sp, hc_gt_mm=gm["hc"] * sp,
                        hc_abs_error_mm=abs(pm["hc"] - gm["hc"]) * sp)
        bio.append(brow)

        if not args.no_masks:
            cv2.imwrite(str(out / f"{name}_pred_{stem}.png"), (pred * 255).astype(np.uint8))
        if args.n_overlays > 0:
            worst.append((ov["dice"], stem, img, gt, pred, pm))
            worst.sort(key=lambda r: r[0])
            del worst[args.n_overlays:]

    if not subj:
        raise SystemExit("no image could be evaluated - check the file names and the ground-truth folder")

    # ---- subject-level and biometry tables
    with open(out / f"{name}_subject_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBJECT_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(subj)
    with open(out / f"{name}_biometry_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BIOMETRY_COLUMNS, extrasaction="ignore", restval="", lineterminator="\n")
        w.writeheader(); w.writerows(bio)
    if skipped:
        with open(out / f"{name}_skipped.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subject_id", "reason"], lineterminator="\n")
            w.writeheader(); w.writerows(skipped)

    # ---- pooled curves (for ROC / PR figures)
    from sklearn.metrics import precision_recall_curve, roc_curve
    yy, ss = np.concatenate(pooled_y), np.concatenate(pooled_s)
    fpr, tpr, _ = roc_curve(yy, ss)
    pp, rr, _ = precision_recall_curve(yy, ss)
    np.savez_compressed(out / "pooled_curves.npz", fpr=fpr, tpr=tpr, precision=pp, recall=rr)

    # ---- summary statistics
    df = pd.DataFrame(subj)
    stats = []
    for c in SUBJECT_COLUMNS[1:]:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v):
            stats.append(dict(metric=c, n=int(len(v)), mean=v.mean(), std=v.std(ddof=1) if len(v) > 1 else 0.0,
                              median=v.median(), q1=v.quantile(0.25), q3=v.quantile(0.75),
                              iqr=v.quantile(0.75) - v.quantile(0.25), min=v.min(), max=v.max()))
    pd.DataFrame(stats).to_csv(out / f"{name}_summary.csv", index=False)

    unit = "mm" if has_mm else "px"
    summary = dict(
        name=name, weights=str(args.weights), model_origin=args.origin, loss=loss_name,
        image_size=pre["image_size"], n_parameters=args.n_params,
        images_found=len(pairs), n_images=len(subj), n_skipped=len(skipped),
        threshold=args.threshold, boundary_tolerance=f"{args.tolerance_mm} {unit}",
        distance_unit=unit if has_mm else "px (no pixel sizes given)",
        empty_prediction_images=int(sum(1 for r in subj if r["pred_area_network_px"] == 0)),
        ellipse_fit_failed=int(sum(1 for b in bio if not b["ellipse_fit_ok"])),
        inference_ms_per_image_mean=float(np.mean(times) * 1000),
        inference_ms_per_image_std=float(np.std(times) * 1000),
        device=str(device), evaluated_at=datetime.now().isoformat(timespec="seconds"),
    )
    save_json(summary, out / f"{name}_summary.json")

    # ---- worst-case overlays
    for rank, (d, stem, img, gt, pred, pm) in enumerate(worst, start=1):
        cv2.imwrite(str(out / f"{name}_worst_{rank:02d}_dice{d:.3f}_{stem}.png"), overlay(img, gt, pred, pm))

    return df, summary


def summary_text(df, summary, out_dir):
    u = "mm" if summary["distance_unit"] == "mm" else "px"
    w = 78
    L = ["=" * w,
         f" EVALUATION SUMMARY  -  {summary['name']}",
         f" images evaluated: {summary['n_images']}   skipped: {summary['n_skipped']}   "
         f"distance unit: {summary['distance_unit']}",
         f" weights: {Path(summary['weights']).name} ({summary['model_origin']}, {summary['n_parameters']/1e6:.2f} M params)",
         "=" * w,
         f" {'metric':<34}{'mean ± SD':>20}{'median [IQR]':>24}",
         "-" * w]
    for col, label, higher in SUMMARY_ROWS:
        v = pd.to_numeric(df[col], errors="coerce").dropna() if col in df else pd.Series(dtype=float)
        if len(v) == 0:
            continue
        arrow = "↑" if higher else "↓"
        name = f"{label.format(u=u)} {arrow}"
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        ms = f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}"
        if abs(v.mean()) >= 100:
            ms = f"{v.mean():,.0f} ± {v.std(ddof=1) if len(v) > 1 else 0:,.0f}"
        md = f"{v.median():.3f} [{q1:.3f}, {q3:.3f}]"
        if abs(v.median()) >= 100:
            md = f"{v.median():,.0f} [{q1:,.0f}, {q3:,.0f}]"
        L.append(f" {name:<34}{ms:>20}{md:>24}")
    L += ["-" * w,
          f" empty predictions: {summary['empty_prediction_images']}   "
          f"ellipse fit failed: {summary['ellipse_fit_failed']}   "
          f"inference: {summary['inference_ms_per_image_mean']:.0f} ms/image on {summary['device']}",
          f" arrows: ↑ higher is better, ↓ lower is better (mean ± standard deviation; median [interquartile range])",
          "=" * w,
          f" all results saved to: {out_dir}"]
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="training checkpoint (.pt) or exported full model (.pth)")
    ap.add_argument("--images", required=True, help="folder with the ultrasound images")
    ap.add_argument("--gt", required=True, help="folder with the ground-truth annotations")
    ap.add_argument("--gt_suffix", default="_Annotation", help="suffix of the GT file name ('' if the names match)")
    ap.add_argument("--pixel_csv", default=None, help="optional table: filename, pixel size (mm), HC (mm)")
    ap.add_argument("--name", default=None, help="model name used in file names (default: weights file name)")
    ap.add_argument("--result_dir", default="results",
                    help="folder that receives all result files (created if missing; no sub-folders are made)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tolerance_mm", type=float, default=2.0, help="tolerance for the normalised surface distance")
    ap.add_argument("--loss", default=None, help="loss used for the diceCEloss column (default: from the weights)")
    ap.add_argument("--n_overlays", type=int, default=12, help="number of worst cases saved as overlays")
    ap.add_argument("--no_masks", action="store_true", help="do not save the predicted masks")
    ap.add_argument("--make_figures", action="store_true", help="also write the report figures to <result_dir>/<name>_figures/")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    name = args.name or weights.stem
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{name}_subject_metrics.csv").exists():
        print(f"note: existing results for '{name}' in {out} will be overwritten")

    device = get_device(args.device)
    model, pre, loss_name, origin, n_params = load_model(weights, device, args.loss)
    args.origin, args.n_params, args.weights = origin, n_params, str(weights)
    print(f"model: {name} ({origin}) | image size {pre['image_size']} | device {device}")

    pairs, missing, orphans = find_pairs(args.images, args.gt, args.gt_suffix)
    if not pairs:
        raise SystemExit(f"no image / ground-truth pairs found (images: {args.images}, gt: {args.gt}, "
                         f"suffix '{args.gt_suffix}')")
    if missing:
        print(f"warning: {len(missing)} image(s) have no ground truth and are not evaluated")
    if orphans:
        print(f"warning: {len(orphans)} ground-truth file(s) have no matching image")
    pix = load_pixel_table(args.pixel_csv)
    print(f"{len(pairs)} image / ground-truth pairs | distances in "
          f"{'mm' if pix and all(p.name in pix for p, _ in pairs) else 'pixels'}")

    df, summary = evaluate(args, model, pre, loss_name, pairs, pix, out, name)

    if args.make_figures:
        figs = out / f"{name}_figures"                     # one figure folder per model, so models do not overwrite each other
        cmd = [sys.executable, str(ROOT / "tools" / "make_report_figures.py"),
               "--evals", f"{name}={out}", "--out", str(figs)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"figures written to {figs}")
        else:
            print("figure generation failed:\n" + res.stderr[-1500:])

    text = summary_text(df, summary, out)
    print("\n" + text)
    (out / f"{name}_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()