#!/usr/bin/env python
"""STEP 1 - Dataset quantity & quality check (run BEFORE preparing data).

Reports: how many annotation rows and images, missing or unreadable images, duplicate and
non-numeric annotations, keypoints outside the image, degenerate OFD/BPD lines, length outliers,
image sizes, intensity statistics, images per patient.
Writes reports/dataset_check.json, reports/dataset_problems.csv, reports/dataset_overview.png
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.constants import COORD_COLUMNS
from src.io_utils import (GEOMETRY_THRESHOLDS, ellipse_axes_problems, geometry_measures, load_annotations,
                          patient_id, row_problems, save_json, point_length)

NAME_PATTERN = re.compile(r"^\d+_HC\.(png|PNG)$")
ERROR_TYPES = {"missing_image", "unreadable_image", "nonnumeric_coords", "duplicate_annotation",
               "out_of_bounds", "short_ofd", "short_bpd", "not_perpendicular", "midpoint_gap"}


def stat(x):
    x = np.asarray(x, dtype=float)
    return dict(min=float(x.min()), mean=float(x.mean()), max=float(x.max()), std=float(x.std())) if x.size else {}


def severity_of(kind):
    if kind in ERROR_TYPES:
        return "error"
    if kind in ("unexpected_filename", "rare_image_size"):
        return "info"
    return "warning"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--csv", required=True, help="annotation CSV file, or a folder of CSV files")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--max_perp_dev", type=float, default=GEOMETRY_THRESHOLDS["max_perp_dev"])
    ap.add_argument("--max_midpoint_gap", type=float, default=GEOMETRY_THRESHOLDS["max_midpoint_gap"])
    ap.add_argument("--min_len_px", type=float, default=GEOMETRY_THRESHOLDS["min_len_px"])
    ap.add_argument("--z_thresh", type=float, default=4.0)
    ap.add_argument("--dark_mean", type=float, default=10.0)
    ap.add_argument("--flat_std", type=float, default=5.0)
    ap.add_argument("--max_center_gap", type=float, default=0.15)
    ap.add_argument("--min_line_angle", type=float, default=60.0)
    a = ap.parse_args()
    thresholds = dict(max_perp_dev=a.max_perp_dev, max_midpoint_gap=a.max_midpoint_gap, min_len_px=a.min_len_px)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ann = load_annotations(a.csv)
    problems = []                                   # (type, severity, file, detail)

    def add(kind, name, detail=""):
        problems.append((kind, severity_of(kind), name, detail))

    for name in ann.loc[ann["image_name"].duplicated(keep=False), "image_name"].unique():
        add("duplicate_annotation", name, "listed more than once in the annotations")
    for name in ann["image_name"]:
        if not NAME_PATTERN.match(name):
            add("unexpected_filename", name, "does not match NNN_HC.png")

    coords = ann[COORD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    sizes, means, stds = Counter(), [], []
    ofd_lens, bpd_lens, ids = [], [], []
    n_present = n_readable = 0
    img_dir = Path(a.img_dir)

    for i, r in ann.iterrows():
        name = r["image_name"]
        path = img_dir / name
        if not path.is_file():
            add("missing_image", name, "annotated image not found")
            continue
        n_present += 1
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            add("unreadable_image", name, "OpenCV could not decode the file")
            continue
        n_readable += 1
        h, w = img.shape
        sizes[(h, w)] += 1
        xy = coords.loc[i].to_numpy(dtype=float).reshape(-1, 2)

        for kind, detail in row_problems(xy, w, h, **thresholds):          # removal criteria (errors)
            add(kind, name, detail)
        if not np.all(np.isfinite(xy)):
            continue
        m = geometry_measures(xy)
        ofd_lens.append(m["ofd_px"])
        bpd_lens.append(m["bpd_px"])
        ids.append(name)
        for kind, detail in ellipse_axes_problems(xy, a.max_center_gap, a.min_line_angle):   # report only
            add(kind, name, detail)
        if min(point_length(xy[p], xy[q]) for p in range(4) for q in range(p + 1, 4)) < 1.0:
            add("coincident_keypoints", name, "two landmarks are almost identical")
        means.append(float(img.mean()))
        stds.append(float(img.std()))
        if img.mean() < a.dark_mean:
            add("near_black_image", name, f"mean intensity={img.mean():.1f}")
        if img.std() < a.flat_std:
            add("flat_image", name, f"intensity std={img.std():.1f}")

    # length outliers (z-score per biometry)
    for key, lens in (("ofd", ofd_lens), ("bpd", bpd_lens)):
        if len(lens) > 1 and np.std(lens) > 0:
            z = (np.asarray(lens) - np.mean(lens)) / np.std(lens)
            for j in np.flatnonzero(np.abs(z) > a.z_thresh):
                add(f"{key}_length_outlier", ids[j], f"z={z[j]:.2f} length={lens[j]:.1f}px")

    n_valid = max(1, len(ids))
    for (h, w), c in sizes.items():
        if c / n_valid < 0.01:
            add("rare_image_size", f"{h}x{w}", f"size seen {c} time(s)")

    pids = Counter(patient_id(Path(n).stem) for n in ann["image_name"])
    with open(out / "dataset_problems.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["type", "severity", "file", "detail"])
        wr.writerows(problems)

    severity = Counter(p[1] for p in problems)
    report = dict(
        n_rows=int(len(ann)), n_images_present=n_present, n_images_readable=n_readable,
        image_sizes={f"{h}x{w}": c for (h, w), c in sizes.items()},
        n_unique_patients=len(pids), images_per_patient=dict(Counter(pids.values())),
        ofd_length_px=stat(ofd_lens), bpd_length_px=stat(bpd_lens),
        intensity_mean=stat(means), intensity_std=stat(stds),
        issue_counts=dict(Counter(p[0] for p in problems)), severity_counts=dict(severity),
        n_problems=len(problems))
    save_json(report, out / "dataset_check.json")

    print("=" * 60); print("DATASET CHECK"); print("=" * 60)
    print(f"annotation rows        : {len(ann)}")
    print(f"images present/readable: {n_present} / {n_readable}")
    print(f"image sizes            : {report['image_sizes']}")
    print(f"unique patients        : {len(pids)}  (images/patient: {report['images_per_patient']})")
    if ids:
        print(f"OFD length (px)        : {report['ofd_length_px']['min']:.0f} - {report['ofd_length_px']['max']:.0f}")
        print(f"BPD length (px)        : {report['bpd_length_px']['min']:.0f} - {report['bpd_length_px']['max']:.0f}")
    print(f"problems               : {dict(severity)}  -> {out/'dataset_problems.csv'}")
    n_test = max(1, int(0.1 * len(pids)))
    print(f"\nsuggested 80/10/10 patient split: ~{int(.8*len(pids))} / ~{n_test} / ~{n_test} patients")
    if len(ids) < 100:
        print("WARNING: <100 usable rows - expect to need heavy augmentation and a small model.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax = ax.ravel()
    if ofd_lens: ax[0].hist(ofd_lens, bins=30); ax[0].set_title("OFD length (px)")
    if bpd_lens: ax[1].hist(bpd_lens, bins=30); ax[1].set_title("BPD length (px)")
    if ofd_lens and bpd_lens:
        ax[2].hist(np.asarray(bpd_lens) / np.asarray(ofd_lens), bins=30); ax[2].set_title("BPD / OFD")
    if means: ax[3].hist(means, bins=30); ax[3].set_title("image mean intensity")
    per_patient = Counter(pids.values())
    ax[4].bar([str(k) for k in per_patient.keys()], list(per_patient.values()))
    ax[4].set_title("patients by number of images (x = images)")
    ax[5].bar(list(severity.keys()) or ["-"], list(severity.values()) or [0]); ax[5].set_title("problems by severity")
    fig.tight_layout(); fig.savefig(out / "dataset_overview.png", dpi=110)
    print(f"overview plot          -> {out/'dataset_overview.png'}")


if __name__ == "__main__":
    main()
