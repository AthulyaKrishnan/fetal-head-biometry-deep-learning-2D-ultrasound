#!/usr/bin/env python
"""STEP 1 - Dataset quantity & quality check (run BEFORE preparing data).

Reports: how many images / GT files, which are unmatched, image sizes, GT type
(outline vs filled), unreadable or open outlines, mask-area and ellipse statistics,
intensity statistics, images per patient, optional pixel-size coverage.
Writes reports/dataset_check.json, reports/dataset_problems.csv, reports/dataset_overview.png
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo-part root, so that `src` is importable

from src.io_utils import find_pairs, list_images, load_pixel_table, patient_id, save_json
from src.postprocess import measure_mask
from src.preprocessing import outline_to_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--gt_suffix", default="_Annotation", help="GT = <image stem><suffix>.png ('' if same name)")
    ap.add_argument("--pixel_csv", default=None, help="optional HC18 pixel-size csv")
    ap.add_argument("--out_dir", default="reports")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    n_img, n_gt = len(list_images(a.img_dir)), len(list_images(a.gt_dir))
    pairs, missing_gt, orphan_gt = find_pairs(a.img_dir, a.gt_dir, a.gt_suffix)
    problems = [("missing_gt", p.name, "") for p in missing_gt] + [("orphan_gt", p.name, "") for p in orphan_gt]

    sizes, status_ct, areas, majors, minors, ratios, means, stds = Counter(), Counter(), [], [], [], [], [], []
    for img_p, gt_p in pairs:
        img, gt = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE), cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            problems.append(("unreadable", img_p.name, "image or GT cannot be read")); continue
        sizes[img.shape] += 1
        if img.shape != gt.shape:
            problems.append(("size_mismatch", img_p.name, f"img {img.shape} vs gt {gt.shape}")); continue
        mask, status = outline_to_mask(gt)
        status_ct[status] += 1
        if status in ("empty", "open_outline"):
            problems.append((status, img_p.name, "GT could not be turned into a filled mask")); continue
        if status == "repaired":
            problems.append(("repaired_gap", img_p.name, "outline had a gap; closed automatically (check overlay)"))
        areas.append(mask.mean())
        m = measure_mask(mask)
        if m:
            majors.append(m["major_len"]); minors.append(m["minor_len"]); ratios.append(m["minor_len"] / m["major_len"])
        means.append(img.mean()); stds.append(img.std())

    pids = Counter(patient_id(p.stem) for p, _ in pairs)
    pix = load_pixel_table(a.pixel_csv)
    pix_missing = [p.name for p, _ in pairs if p.name not in pix] if pix else None

    def stat(x):
        return dict(min=float(np.min(x)), mean=float(np.mean(x)), max=float(np.max(x)), std=float(np.std(x))) if len(x) else {}

    report = dict(
        n_images=n_img, n_gt_files=n_gt, n_matched_pairs=len(pairs),
        n_images_without_gt=len(missing_gt), n_gt_without_image=len(orphan_gt),
        image_sizes={f"{h}x{w}": c for (h, w), c in sizes.items()},
        gt_status=dict(status_ct), n_unique_patients=len(pids),
        images_per_patient=dict(Counter(pids.values())),
        mask_area_fraction=stat(areas), major_axis_px=stat(majors), minor_axis_px=stat(minors),
        axis_ratio_minor_over_major=stat(ratios), image_mean_intensity=stat(means), image_std_intensity=stat(stds),
        pixel_csv_entries=len(pix) if pix else None,
        images_missing_pixel_size=len(pix_missing) if pix_missing is not None else None,
        n_problems=len(problems))
    save_json(report, out / "dataset_check.json")
    with open(out / "dataset_problems.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["type", "file", "detail"]); w.writerows(problems)

    # ----------------------------------------------------------------- console summary
    print("=" * 60); print("DATASET CHECK"); print("=" * 60)
    print(f"images found           : {n_img}\nground-truth files     : {n_gt}\nmatched pairs          : {len(pairs)}")
    print(f"images without GT      : {len(missing_gt)}\nGT without image       : {len(orphan_gt)}")
    print(f"image sizes            : {report['image_sizes']}")
    print(f"GT type                : {dict(status_ct)}")
    print(f"unique patients        : {len(pids)}  (images/patient: {report['images_per_patient']})")
    if areas:
        print(f"mask area fraction     : {report['mask_area_fraction']['min']:.3f} - {report['mask_area_fraction']['max']:.3f} (mean {report['mask_area_fraction']['mean']:.3f})")
        print(f"major axis (px)        : {report['major_axis_px']['min']:.0f} - {report['major_axis_px']['max']:.0f}")
    if pix_missing is not None:
        print(f"pixel-size coverage    : {len(pairs) - len(pix_missing)}/{len(pairs)} images")
    print(f"problems logged        : {len(problems)}  -> {out/'dataset_problems.csv'}")
    n_test = max(1, int(0.1 * len(pids)))
    print(f"\nsuggested 80/10/10 patient split: ~{int(.8*len(pids))} / ~{n_test} / ~{n_test} patients")
    if len(pairs) < 100:
        print("WARNING: <100 pairs - expect to need heavy augmentation and a small model (base_channels 16-32).")

    # ------------------------------------------------------------------ overview plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax = ax.ravel()
    if areas: ax[0].hist(areas, bins=30); ax[0].set_title("mask area / image area")
    if majors: ax[1].hist(majors, bins=30); ax[1].set_title("major axis (px)")
    if ratios: ax[2].hist(ratios, bins=30); ax[2].set_title("minor / major axis")
    if means: ax[3].hist(means, bins=30); ax[3].set_title("image mean intensity")
    ax[4].bar(list(map(str, Counter(pids.values()).keys())), list(Counter(pids.values()).values()))
    ax[4].set_title("images per patient (x = count)")
    ax[5].bar(list(status_ct.keys()) or ["-"], list(status_ct.values()) or [0]); ax[5].set_title("GT status")
    fig.tight_layout(); fig.savefig(out / "dataset_overview.png", dpi=110)
    print(f"overview plot          -> {out/'dataset_overview.png'}")


if __name__ == "__main__":
    main()
