#!/usr/bin/env python
"""STEP 2 - Prepare data: remove implausible annotations, copy usable images, split by patient.

A row is REMOVED when
  * its landmarks are not on a fetal head in an axial image (automatic geometry rule):
      - the OFD and BPD lines are more than 30 degrees away from perpendicular,
      - the midpoints of the two lines are more than 150 px apart,
      - the OFD or the BPD is shorter than 100 px,
      - a landmark lies outside the image;
  * it is listed in --exclude_list (labels found wrong during debugging; names may be .txt or .csv);
  * the coordinates are missing or non-numeric, or the image is missing or unreadable;
  * it duplicates an earlier row (the first is kept).
Removal is the only place samples are dropped. The split then assigns every remaining patient,
and therefore every remaining image, to exactly one of train / val / test (src/splitting.py):
no patient is shared between splits, the image shares match the targets as closely as patient
sizes allow, and image size x OFD-length groups are spread evenly across the splits.

Output (--out_dir, default data_processed/):
    images/<image_name>          copy of each kept image (grayscale, lossless)
    overlays/                    kept images with landmarks drawn (random sample)
    overlays_removed/            every removed image with landmarks drawn  <-- CHECK THESE
    train.csv val.csv test.csv   image paths (relative), landmarks in ORIGINAL pixels
    removed_samples.csv          every removed row with the reasons and measurements
    prepare_report.json          counts, removal reasons, split balance
"""
import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.constants import COORD_COLUMNS
from src.io_utils import (GEOMETRY_THRESHOLDS, geometry_measures, load_annotations, load_exclude_list,
                          patient_id, row_problems, save_json)
from src.postprocess import draw_landmarks
from src.splitting import size_strata, split_quality, stratified_group_split

COLS = ["stem", "patient_id", "image_name", "image_path", "height", "width", "stratum"] + COORD_COLUMNS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--csv", required=True, help="annotation CSV file, or a folder of CSV files")
    ap.add_argument("--out_dir", default="data_processed")
    ap.add_argument("--exclude_list", default=None,
                    help="names of wrong labels to remove (.txt one per line, or .csv with image_name)")
    ap.add_argument("--expect_excluded", type=int, default=None,
                    help="stop with an error unless exactly this many listed images are found")
    ap.add_argument("--max_perp_dev", type=float, default=GEOMETRY_THRESHOLDS["max_perp_dev"])
    ap.add_argument("--max_midpoint_gap", type=float, default=GEOMETRY_THRESHOLDS["max_midpoint_gap"])
    ap.add_argument("--min_len_px", type=float, default=GEOMETRY_THRESHOLDS["min_len_px"])
    ap.add_argument("--length_bins", type=int, default=3,
                    help="OFD-length groups used to balance the splits (1 = stratify by image size only)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--n_overlays", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    thresholds = dict(max_perp_dev=a.max_perp_dev, max_midpoint_gap=a.max_midpoint_gap, min_len_px=a.min_len_px)

    out = Path(a.out_dir)
    for d in ("images", "overlays", "overlays_removed"):          # clear stale files from earlier runs
        shutil.rmtree(out / d, ignore_errors=True)
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    # ---- manual exclusions (checked against the annotations)
    listed = load_exclude_list(a.exclude_list)
    ann = load_annotations(a.csv).reset_index(drop=True)
    is_listed = ann["image_name"].map(lambda n: n in listed or Path(n).stem in listed)
    listed_hit = set(ann.loc[is_listed, "image_name"])
    if a.exclude_list:
        known = set(ann["image_name"]) | {Path(n).stem for n in ann["image_name"]}
        missing = sorted(n for n in listed if n not in known)
        print(f"exclude list: {len(listed_hit)} annotation rows matched")
        if missing:
            print(f"  {len(missing)} listed name(s) not found in the annotations, e.g.: {missing[:5]}")
    if a.expect_excluded is not None and len(listed_hit) != a.expect_excluded:
        sys.exit(f"ERROR: expected {a.expect_excluded} excluded images but matched {len(listed_hit)}. "
                 "Check that the names match the annotation image_name column.")

    # ---- 1. validate every row
    kept, removed, seen = [], [], set()
    for i, r in ann.iterrows():
        name = r["image_name"]
        pts = r[COORD_COLUMNS].to_numpy(dtype=float).reshape(-1, 2)
        img_path = Path(a.img_dir) / name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) if img_path.is_file() else None
        h, w = img.shape if img is not None else (0, 0)

        reasons = []
        if name in seen:
            reasons.append(("duplicate_annotation", "listed more than once; first row kept"))
        seen.add(name)
        if name in listed_hit:
            reasons.append(("excluded_by_list", f"listed in {a.exclude_list}"))
        if not img_path.is_file():
            reasons.append(("missing_image", "image file not found"))
        elif img is None:
            reasons.append(("unreadable_image", "OpenCV could not decode the file"))
        if img is not None:                                   # record every reason, not just the first
            reasons += row_problems(pts, w, h, **thresholds)

        if reasons:
            m = geometry_measures(pts) if np.all(np.isfinite(pts)) else {}
            removed.append(dict(image_name=name, reasons=";".join(t for t, _ in reasons),
                                details=" | ".join(d for _, d in reasons), width=w, height=h,
                                **{k: round(v, 2) for k, v in m.items()}, _pts=pts, _img=img))
        else:
            kept.append(dict(name=name, img=img, pts=pts, h=h, w=w, measures=geometry_measures(pts)))

    # ---- 2. write kept images, overlays and the removed-sample review
    for k in kept:
        cv2.imwrite(str(out / "images" / k["name"]), k["img"])
    sample = set(rng.sample(range(len(kept)), min(a.n_overlays, len(kept))))
    for j, k in enumerate(kept):
        if j in sample:
            cv2.imwrite(str(out / "overlays" / k["name"]), draw_landmarks(k["img"], k["pts"]))
    for r in removed:
        if r["_img"] is not None and np.all(np.isfinite(r["_pts"])):
            cv2.imwrite(str(out / "overlays_removed" / r["image_name"]), draw_landmarks(r["_img"], r["_pts"]))
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in removed]).to_csv(
        out / "removed_samples.csv", index=False)

    # ---- 3. patient-level strata: image size x OFD-length group (median per patient)
    pat = {}
    for k in kept:
        pid = patient_id(Path(k["name"]).stem)
        pat.setdefault(pid, dict(n=0, size=f"{k['h']}x{k['w']}", ofd=[]))
        pat[pid]["n"] += 1
        pat[pid]["ofd"].append(k["measures"]["ofd_px"])
    n_img = {p: d["n"] for p, d in pat.items()}
    ofd_med = {p: float(np.median(d["ofd"])) for p, d in pat.items()}
    len_bin = size_strata(ofd_med, a.length_bins) if a.length_bins > 1 else {p: "0" for p in pat}
    strata = {p: f"{d['size']}|ofd{len_bin[p]}" for p, d in pat.items()}
    fracs = {"train": 1 - a.val_frac - a.test_frac, "val": a.val_frac, "test": a.test_frac}

    # ---- 4. optimal patient split (assigns every patient, so every kept image)
    assign = stratified_group_split(n_img, strata, fracs, seed=a.seed)
    quality = split_quality(assign, n_img, strata, fracs)

    rows = {s: [] for s in fracs}
    for k in kept:
        pid = patient_id(Path(k["name"]).stem)
        stem = Path(k["name"]).stem
        row = dict(stem=stem, patient_id=pid, image_name=k["name"], image_path=f"images/{k['name']}",
                   height=k["h"], width=k["w"], stratum=strata[pid])
        row.update({c: float(v) for c, v in zip(COORD_COLUMNS, k["pts"].reshape(-1))})
        rows[assign[pid]].append(row)
    for s, rs in rows.items():
        with open(out / f"{s}.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=COLS)
            wr.writeheader()
            wr.writerows(rs)

    # ---- 5. checks: no loss, no leakage
    n_written = sum(len(v) for v in rows.values())
    assert n_written == len(kept), "sample accounting failed: a kept image was not written to any split"
    pid_sets = {s: {r["patient_id"] for r in v} for s, v in rows.items()}
    names = list(fracs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (pid_sets[names[i]] & pid_sets[names[j]]), "patient leakage between splits!"

    report = dict(
        n_rows=int(len(ann)), n_kept=len(kept), n_removed=len(removed),
        removal_reasons=dict(Counter(t for r in removed for t in r["reasons"].split(";") if t)),
        removal_thresholds=thresholds, excluded_list=a.exclude_list, length_bins=a.length_bins,
        split_images={s: len(v) for s, v in rows.items()},
        split_patients={s: len(v) for s, v in pid_sets.items()},
        split_quality=quality, seed=a.seed)
    save_json(report, out / "prepare_report.json")

    print(f"rows {len(ann)} | kept {len(kept)} | removed {len(removed)} {report['removal_reasons']}")
    imgs = report["split_images"]
    print(f"images   train/val/test = {imgs['train']}/{imgs['val']}/{imgs['test']}  "
          f"(shares {quality['train']['share']:.3f}/{quality['val']['share']:.3f}/{quality['test']['share']:.3f}, "
          f"targets {quality['train']['target_share']:.3f}/{quality['val']['target_share']:.3f}/{quality['test']['target_share']:.3f})")
    pats = report["split_patients"]
    print(f"patients train/val/test = {pats['train']}/{pats['val']}/{pats['test']}  (no leakage)")
    print(f"every kept image placed in a split: {n_written == len(kept)}")
    print(f"CHECK removed rows in {out/'overlays_removed'} and {out/'removed_samples.csv'}")


if __name__ == "__main__":
    main()
