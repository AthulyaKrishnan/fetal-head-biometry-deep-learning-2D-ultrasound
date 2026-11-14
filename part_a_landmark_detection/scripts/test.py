#!/usr/bin/env python
"""Evaluate a trained fetal head landmark model on a test set you provide. All results in millimetres.

Example:
    python scripts/test.py --weights weights/best.pt --images /path/to/test_images \
        --annotations /path/to/test_annotations.csv --pixel_spacing_mm 0.1 --result_dir results/my_test

Inputs
  --images            folder of ultrasound images (png, jpg, tif, bmp)
  --annotations       CSV with column image_name and the eight coordinate columns
                      ofd_1_x, ofd_1_y, ofd_2_x, ofd_2_y, bpd_1_x, bpd_1_y, bpd_2_x, bpd_2_y
                      (pixels in the original image). A folder of such CSVs also works.
  --weights           weights/best.pt (checkpoint) or a full exported model (.pth)
  --pixel_spacing_mm  millimetres per pixel of THIS test set. Required for mm results.
                      If omitted, the spacing stored with the training run is used with a warning.
                      --allow_pixels produces a pixel-only diagnostic run that is labelled as such.
  --result_dir        result directory (alias --out, default results/). All outputs of the run are
                      written directly into it: summary.txt, summary.json, config.json, predictions.csv,
                      flags.csv, skipped_rows.csv, per_image_mm.csv, metrics_mm.csv, report.md, figures/,
                      overlays/, overlays_worst/. Files from an earlier run in it are replaced.
  --eval_dir          optional: folder to search for make_report.py and mm_metrics.py if they are
                      not found automatically. Not needed in the normal layout.
  --no_report         skip the figure and report generation (summary and tables are still written).

Locations are found automatically: the project root is the folder that contains src/inference.py, and
mm_metrics.py and make_report.py are searched for below it. Nothing is hard-coded.

Every annotated row whose image can be read is evaluated. Rows that cannot be read are listed in
skipped_rows.csv. Rows whose landmarks are implausible for a fetal head are flagged in flags.csv and
reported again as a separate subset.

Metrics (all in mm unless stated)
  Landmark error      mean radial error (MRE) per image, its median, IQR and 90th percentile, with
                      bootstrap 95 % confidence intervals; per-landmark MRE (Chen et al., MICCAI 2019).
  SDR                 successful detection rate: share of landmarks within 2, 2.5, 3 and 4 mm.
  SDR-AUC             area under the SDR curve from 0 to 10 mm, divided by 10.
  NME                 endpoint-order-invariant normalised error per biometry (Di Vece et al., 2026).
  Length agreement    OFD and BPD in mm: absolute error, bias, 95 % limits of agreement (Bland and Altman),
                      share within 5 %, ICC(2,1) with absolute agreement.
  Head circumference  HC = 1.62 x (BPD + OFD) (ISUOG 2022), absolute error and agreement.
  Failure analysis    images with error above 3 x the median, and the share of total error in the worst 10 %.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================================ locating the code
def _is_project_root(d):
    return (d / "src" / "inference.py").is_file()


def find_project_root():
    """The folder that contains src/inference.py. Searched from the script's folder and from the
    current working directory, and their parents and immediate sub-folders."""
    bases = []
    for start in (SCRIPT_DIR, Path.cwd().resolve()):
        bases += [start, *start.parents[:2]]
    for base in bases:
        if _is_project_root(base):
            return base
        for pattern in ("*/src/inference.py", "*/*/src/inference.py"):
            for hit in sorted(base.glob(pattern)):
                return hit.parent.parent
    raise SystemExit("Could not find the project code (src/inference.py). Run scripts/test.py from the folder "
                     "that contains src/, or place it inside that folder.")


def find_file(name, search_roots):
    """First file called `name` below any of the roots, preferring paths that contain 'src' or 'evaluation'."""
    hits = []
    for root in search_roots:
        if root and Path(root).is_dir():
            hits += [p for p in Path(root).rglob(name) if len(p.relative_to(root).parts) <= 4]
    if not hits:
        return None
    hits.sort(key=lambda p: (not ("src" in p.parts or "evaluation" in p.parts), len(p.parts)))
    return hits[0]


def load_module_from_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import COORD_COLUMNS, KEYPOINT_NAMES  # noqa: E402
from src.dataset import FetalLandmarkDataset, make_loader  # noqa: E402
from src.inference import get_device, run_inference, torch_load  # noqa: E402
from src.io_utils import GEOMETRY_THRESHOLDS, load_annotations, patient_id, row_problems  # noqa: E402
from src.model import build_model  # noqa: E402
from src.postprocess import draw_landmarks  # noqa: E402


# ============================================================================ model
def load_weights(path, device):
    """Accepts a checkpoint written by scripts/train.py or a full model written by tools/export_model.py.
    Returns the model, its input size and heatmap stride, the preprocessing settings, a description,
    and the pixel spacing stored in the training configuration (None for a full exported model)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"weights not found: {path}\nPlace best.pt in weights/ or pass --weights.")
    obj = torch_load(str(path), device)
    if isinstance(obj, dict) and "model_state" in obj:
        cfg = obj["config"]
        model = build_model(cfg, pretrained=False).to(device)
        model.load_state_dict(obj["model_state"])
        info = dict(source="checkpoint", model_type=cfg["model"].get("type", "heatmap"),
                    backbone=cfg["model"]["backbone"], input_size=cfg["data"]["input_size"],
                    heatmap_stride=cfg["data"]["heatmap_stride"], trained_epochs=obj.get("epoch"))
        spacing = cfg["data"].get("pixel_spacing_mm")
        prep, input_size, stride = cfg["preprocess"], cfg["data"]["input_size"], cfg["data"]["heatmap_stride"]
    elif isinstance(obj, torch.nn.Module):
        model = obj.to(device)
        input_size, stride = model.input_size, getattr(model, "stride", 4)
        prep = getattr(model, "preprocess_cfg", {"use_clahe": True, "clip_limit": 2.0, "tile_grid": 8})
        info = dict(source="full exported model", model_type=getattr(model, "output_type", "heatmap"),
                    input_size=input_size, heatmap_stride=stride)
        spacing = None
    else:
        raise SystemExit(f"{path} is not a recognised model file")
    model.eval()
    return model, input_size, stride, prep, info, spacing


# ============================================================================ data
def read_test_set(images_dir, ann):
    """One record per annotated row whose image can be read, plus skipped rows and plausibility flags."""
    records, skipped, flags = [], [], []
    seen = set()
    for _, r in ann.iterrows():
        name = r["image_name"]
        if name in seen:
            skipped.append(dict(image_name=name, reason="duplicate annotation (first row kept)"))
            continue
        seen.add(name)
        path = Path(images_dir) / name
        if not path.is_file():
            skipped.append(dict(image_name=name, reason="image file not found"))
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append(dict(image_name=name, reason="image could not be read"))
            continue
        pts = pd.to_numeric(r[COORD_COLUMNS], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(pts)):
            skipped.append(dict(image_name=name, reason="coordinates missing or non-numeric"))
            continue
        pts = pts.reshape(-1, 2).astype(np.float32)
        h, w = img.shape
        reasons = [d for _, d in row_problems(pts, w, h, **GEOMETRY_THRESHOLDS)]
        records.append(dict(stem=Path(name).stem, patient_id=patient_id(Path(name).stem), image_name=name,
                            image_path=str(path), height=h, width=w, gt=pts, plausible=not reasons))
        flags.append(dict(image_name=name, readable=True, plausible=not reasons, reasons=" | ".join(reasons)))
    return records, skipped, pd.DataFrame(flags)


# ============================================================================ output folder
# Files and folders this tool writes into the results folder. Only these are replaced on a new run.
OUTPUT_FILES = ["summary.txt", "summary.json", "config.json", "per_image_mm.csv", "predictions.csv",
                "flags.csv", "skipped_rows.csv", "report.md", "metrics_mm.csv",
                "metrics_mm_plausible_subset.csv", "paired_tests.csv", "make_report.log"]
OUTPUT_DIRS = ["figures", "overlays", "overlays_worst"]


def clear_previous_results(out):
    """Remove the files and folders written by an earlier run, and nothing else."""
    for name in OUTPUT_FILES:
        (out / name).unlink(missing_ok=True)
    for name in OUTPUT_DIRS:
        shutil.rmtree(out / name, ignore_errors=True)
    for path in out.glob("summary_mm_*.json"):
        path.unlink(missing_ok=True)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(PROJECT_ROOT / "weights" / "best.pt"))
    ap.add_argument("--images", required=True, help="folder with the test images")
    ap.add_argument("--annotations", required=True, help="CSV file (or folder of CSVs) with the landmarks")
    ap.add_argument("--pixel_spacing_mm", type=float, default=None, help="mm per pixel of the test images")
    ap.add_argument("--allow_pixels", action="store_true",
                    help="diagnostic only: if no spacing is known, report in pixels and say so")
    ap.add_argument("--eval_dir", default=None, help="optional folder to search for make_report.py and mm_metrics.py")
    ap.add_argument("--no_report", action="store_true", help="skip figures and report.md")
    ap.add_argument("--result_dir", "--out", dest="out", default=str(PROJECT_ROOT / "results"),
                    help="result directory: all outputs of this run are written directly into it "
                         "(created if needed; files from an earlier run in it are replaced)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    a = ap.parse_args()

    # ----- locate the shared metrics module and the report generator (no hard-coded paths)
    search_roots = [PROJECT_ROOT, SCRIPT_DIR] + ([Path(a.eval_dir)] if a.eval_dir else [])
    metrics_path = find_file("mm_metrics.py", search_roots)
    if metrics_path is None:
        raise SystemExit("mm_metrics.py was not found. Keep it in src/, or pass --eval_dir <folder>.")
    M = load_module_from_file(metrics_path, "mm_metrics")
    report_path = None if a.no_report else find_file("make_report.py", search_roots)
    if report_path is None and not a.no_report:
        print("NOTE: make_report.py not found, so figures and report.md will not be generated.")

    device = get_device(a.device)
    model, input_size, stride, prep, info, ckpt_spacing = load_weights(a.weights, device)
    print(f"model loaded: {info['model_type']} from {a.weights} on {device}")

    # ----- pixel spacing: required for millimetre results
    if a.pixel_spacing_mm:
        spacing, spacing_text = a.pixel_spacing_mm, f"{a.pixel_spacing_mm} mm/px (given with --pixel_spacing_mm)"
    elif ckpt_spacing:
        spacing = float(ckpt_spacing)
        spacing_text = f"{spacing} mm/px (from the training run; pass --pixel_spacing_mm to set the test set's own value)"
        print(f"WARNING: using the training spacing {spacing} mm/px. Check that the test images have the same spacing.")
    elif a.allow_pixels:
        spacing, spacing_text = None, "NONE. Results are in PIXELS (diagnostic run, not millimetres)"
        print("WARNING: no pixel spacing; reporting in pixels because --allow_pixels was given.")
    else:
        raise SystemExit("No pixel spacing is known, so results cannot be reported in millimetres.\n"
                         "Pass the spacing of the test images, for example:\n"
                         "    python scripts/test.py ... --pixel_spacing_mm 0.1\n"
                         "(or --allow_pixels for a diagnostic run in pixels).")
    f = spacing or 1.0
    unit = "mm" if spacing else "px"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clear_previous_results(out)
    (out / "overlays").mkdir()
    (out / "overlays_worst").mkdir()

    ann = load_annotations(a.annotations)
    records, skipped, flags = read_test_set(a.images, ann)
    if not records:
        raise SystemExit("no usable rows: check that image_name matches the file names in --images")
    print(f"annotated rows {len(ann)} | evaluated {len(records)} | skipped {len(skipped)} | "
          f"implausible (flagged, still evaluated) {int((~flags['plausible']).sum())}")

    ds = FetalLandmarkDataset(records, input_size=input_size, stride=stride, sigma=2.0,
                              transform=None, preprocess_cfg=prep)
    loader = make_loader(ds, a.batch_size, shuffle=False, num_workers=a.num_workers)
    res = run_inference(model, loader, device, stride)
    pred, gt, names = res["pred"], res["gt"], res["names"]
    plausible = np.array([r["plausible"] for r in records])

    s = M.summarize(pred, gt, f)
    s_plaus = M.summarize(pred[plausible], gt[plausible], f) if plausible.sum() >= 3 else None

    # ----- per-image table, in mm
    err_mm = np.linalg.norm(pred - gt, axis=-1) * f
    per = {"image_name": names}
    for k, n in enumerate(KEYPOINT_NAMES):
        per[f"{n}_err_mm"] = err_mm[:, k]
    per["mean_err_mm"] = err_mm.mean(axis=1)
    per["ofd_len_pred_mm"] = np.linalg.norm(pred[:, 1] - pred[:, 0], axis=-1) * f
    per["ofd_len_true_mm"] = np.linalg.norm(gt[:, 1] - gt[:, 0], axis=-1) * f
    per["bpd_len_pred_mm"] = np.linalg.norm(pred[:, 3] - pred[:, 2], axis=-1) * f
    per["bpd_len_true_mm"] = np.linalg.norm(gt[:, 3] - gt[:, 2], axis=-1) * f
    per["plausible"] = plausible
    per_df = pd.DataFrame(per)
    per_df.to_csv(out / "per_image_mm.csv", index=False)

    # Intermediate table for the report generator only (pixel coordinates, in a temporary folder that is
    # deleted at the end). The flags are copied next to it, where the generator looks for them.
    tmp_root = Path(tempfile.mkdtemp(prefix="judge_report_"))
    eval_dir = tmp_root / "eval_judges"
    eval_dir.mkdir()
    px = {"image_name": names}
    for k, n in enumerate(KEYPOINT_NAMES):
        px[f"{n}_pred_x"], px[f"{n}_pred_y"] = pred[:, k, 0], pred[:, k, 1]
        px[f"{n}_gt_x"], px[f"{n}_gt_y"] = gt[:, k, 0], gt[:, k, 1]
        px[f"{n}_err_px"] = np.linalg.norm(pred[:, k] - gt[:, k], axis=-1)
    px["mean_err_px"] = np.linalg.norm(pred - gt, axis=-1).mean(axis=1)
    pd.DataFrame(px).to_csv(eval_dir / "per_image.csv", index=False)
    flags.to_csv(tmp_root / "flags.csv", index=False)

    pd.DataFrame({"image_name": names, **{c: pred.reshape(len(pred), -1)[:, i]
                                          for i, c in enumerate(COORD_COLUMNS)}}).round(2).to_csv(
        out / "predictions.csv", index=False)
    flags.to_csv(out / "flags.csv", index=False)
    pd.DataFrame(skipped, columns=["image_name", "reason"]).to_csv(out / "skipped_rows.csv", index=False)

    # ----- overlays
    worst = set(np.argsort(-per_df["mean_err_mm"].to_numpy())[:12])
    for i, rec in enumerate(records):
        gray = cv2.imread(rec["image_path"], cv2.IMREAD_GRAYSCALE)
        img = draw_landmarks(gray, pred[i], gt[i])
        cv2.imwrite(str(out / "overlays" / rec["image_name"]), img)
        if i in worst:
            cv2.imwrite(str(out / "overlays_worst" / f"{per_df.at[i, 'mean_err_mm']:07.3f}mm_{rec['stem']}.png"), img)

    # ----- configuration and summaries
    meta = dict(model_type=info["model_type"], source=info["source"], input_size=input_size,
                weights=str(a.weights), n_skipped=len(skipped), spacing_text=spacing_text)
    (out / "config.json").write_text(json.dumps(
        {"model": {"type": info["model_type"]}, "data": {"pixel_spacing_mm": spacing},
         "weights": str(a.weights), "model_info": info, "spacing_source": spacing_text,
         "evaluated_rows": len(records), "demo": False}, indent=2, default=str))
    (out / "summary.json").write_text(json.dumps(
        {"units": unit, "all_rows": s, "plausible_subset": s_plaus, "n_plausible": int(plausible.sum()),
         "skipped": len(skipped), "pixel_spacing_mm": spacing}, indent=2, default=str))

    # ----- report (figures and tables) from the evaluation code, written straight into the results folder
    report_ok = False
    if report_path is not None:
        cmd = [sys.executable, str(report_path), "--runs", str(eval_dir),
               "--names", "judge", "--out", str(out)]
        if spacing:
            cmd += ["--pixel_spacing_mm", str(spacing)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        report_ok = proc.returncode == 0 and (out / "report.md").exists()
        if not report_ok:
            (out / "make_report.log").write_text(proc.stdout + proc.stderr)
    shutil.rmtree(tmp_root, ignore_errors=True)

    title = ("EVALUATION SUMMARY  (all values in millimetres)" if unit == "mm"
             else "EVALUATION SUMMARY  (PIXELS, diagnostic run)")
    meta_lines = [f"model             : {meta['model_type']} ({meta['source']}), input {meta['input_size']} px",
                  f"weights           : {meta['weights']}",
                  f"pixel spacing     : {meta['spacing_text']}",
                  f"images evaluated  : {s['n_images']}   (rows skipped: {meta['n_skipped']})"]
    text = M.summary_text(s, title, meta_lines)
    if s_plaus is not None:
        text += (f"\n\nPLAUSIBLE SUBSET ({int(plausible.sum())} of {s['n_images']} rows; implausible labels "
                 f"excluded, see flags.csv)\n" + M.summary_text(s_plaus, "plausible subset", []))
    (out / "summary.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"\nresults folder : {out}")
    if report_ok:
        print(f"figures        : {out / 'figures'}")
        print(f"start with     : {out / 'report.md'}")
    else:
        print(f"figures        : not generated (log: {out / 'make_report.log'})" if report_path
              else "figures        : not generated")
        print(f"start with     : {out / 'summary.txt'}")


if __name__ == "__main__":
    main()