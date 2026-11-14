#!/usr/bin/env python
"""STEP 2 - Prepare data: GT outlines -> filled masks, patient-wise train/val/test split.

Output (--out_dir, default data_processed/):
    images/<stem>.png         copy of the original image
    masks/<stem>.png          filled binary mask (0/255) at ORIGINAL resolution
    overlays/                 visual check (red contour on image)  <-- LOOK AT THESE
    train.csv val.csv test.csv
    prepare_report.json
Split is by PATIENT (leading digits of the file name) so no patient is in two splits.
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.io_utils import find_pairs, load_pixel_table, patient_id, save_json
from src.preprocessing import outline_to_mask

COLS = ["stem", "patient_id", "image_path", "mask_path", "height", "width", "pixel_size_mm", "official_hc_mm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--gt_suffix", default="_Annotation")
    ap.add_argument("--pixel_csv", default=None)
    ap.add_argument("--out_dir", default="data_processed")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--n_overlays", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out_dir)
    for d in ("images", "masks", "overlays"):
        (out / d).mkdir(parents=True, exist_ok=True)

    pairs, missing, orphans = find_pairs(a.img_dir, a.gt_dir, a.gt_suffix)
    pix = load_pixel_table(a.pixel_csv)
    print(f"{len(pairs)} pairs ({len(missing)} images without GT, {len(orphans)} GT without image)")

    rows, skipped, status_ct = [], [], {}
    rng = random.Random(a.seed)
    overlay_ids = set(rng.sample(range(len(pairs)), min(a.n_overlays, len(pairs))))
    for i, (img_p, gt_p) in enumerate(pairs):
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None or img.shape != gt.shape:
            skipped.append((img_p.name, "unreadable or size mismatch")); continue
        mask, status = outline_to_mask(gt)
        status_ct[status] = status_ct.get(status, 0) + 1
        if status in ("empty", "open_outline"):
            skipped.append((img_p.name, status)); continue

        cv2.imwrite(str(out / "masks" / f"{img_p.stem}.png"), mask * 255)
        cv2.imwrite(str(out / "images" / f"{img_p.stem}.png"), img)      # lossless re-save as png
        info = pix.get(img_p.name, {})
        rows.append(dict(stem=img_p.stem, patient_id=patient_id(img_p.stem),
                         image_path=f"images/{img_p.stem}.png", mask_path=f"masks/{img_p.stem}.png",
                         height=img.shape[0], width=img.shape[1],
                         pixel_size_mm=info.get("pixel_size_mm", ""), official_hc_mm=info.get("official_hc_mm") or ""))
        if i in overlay_ids or status == "repaired":
            ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnts, -1, (0, 0, 255), 2)
            cv2.imwrite(str(out / "overlays" / f"{img_p.stem}.png"), ov)

    # ---- patient-wise split
    pids = sorted({r["patient_id"] for r in rows})
    rng.shuffle(pids)
    n = len(pids)
    n_test, n_val = max(1, round(a.test_frac * n)), max(1, round(a.val_frac * n))
    test_p, val_p = set(pids[:n_test]), set(pids[n_test:n_test + n_val])
    split_of = lambda r: "test" if r["patient_id"] in test_p else "val" if r["patient_id"] in val_p else "train"
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        splits[split_of(r)].append(r)
    for name, rs in splits.items():
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rs)

    # ---- leakage check + report
    pid_sets = {k: {r["patient_id"] for r in v} for k, v in splits.items()}
    assert not (pid_sets["train"] & pid_sets["val"] or pid_sets["train"] & pid_sets["test"]
                or pid_sets["val"] & pid_sets["test"]), "patient leakage between splits!"
    report = dict(n_pairs=len(pairs), n_used=len(rows), n_skipped=len(skipped), skipped=skipped,
                  gt_status=status_ct, n_patients=n,
                  split_images={k: len(v) for k, v in splits.items()},
                  split_patients={k: len(v) for k, v in pid_sets.items()}, seed=a.seed)
    save_json(report, out / "prepare_report.json")
    print(f"used {len(rows)} | skipped {len(skipped)} | GT status {status_ct}")
    print(f"images  train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"patients train/val/test = {len(pid_sets['train'])}/{len(pid_sets['val'])}/{len(pid_sets['test'])}  (no leakage)")
    print(f"CHECK the overlays in {out/'overlays'} before training.")


if __name__ == "__main__":
    main()
