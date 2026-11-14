#!/usr/bin/env python
"""Compare two evaluated models on the SAME test images (paired comparison).

    python tools/compare_models.py runs/hrnet_run1/eval_test runs/coord_baseline/eval_test

For each measure (landmark error, OFD length error, BPD length error) it reports mean and median of
each model, the paired difference A - B with a 95 % bootstrap confidence interval (for the mean and
the median), and the Wilcoxon signed-rank p-value. The error distributions are skewed, so the paired
Wilcoxon test is used instead of a t-test. Negative differences favour model A.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load(run_dir):
    df = pd.read_csv(Path(run_dir) / "per_image.csv")
    if "ofd_1_pred_x" not in df.columns:
        raise SystemExit(f"{run_dir}/per_image.csv has no predictions; run scripts/evaluate.py first")
    ofd_p = np.hypot(df.ofd_1_pred_x - df.ofd_2_pred_x, df.ofd_1_pred_y - df.ofd_2_pred_y)
    ofd_g = np.hypot(df.ofd_1_gt_x - df.ofd_2_gt_x, df.ofd_1_gt_y - df.ofd_2_gt_y)
    bpd_p = np.hypot(df.bpd_1_pred_x - df.bpd_2_pred_x, df.bpd_1_pred_y - df.bpd_2_pred_y)
    bpd_g = np.hypot(df.bpd_1_gt_x - df.bpd_2_gt_x, df.bpd_1_gt_y - df.bpd_2_gt_y)
    return pd.DataFrame({"image_name": df.image_name,
                         "landmark": df.mean_err_px,
                         "ofd_len": np.abs(ofd_p - ofd_g),
                         "bpd_len": np.abs(bpd_p - bpd_g)}).set_index("image_name")


def bootstrap_ci(d, stat, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    vals = stat(d[idx], axis=1)
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    a = ap.parse_args()

    A, B = load(a.run_a), load(a.run_b)
    common = A.index.intersection(B.index)
    if len(common) != len(A) or len(common) != len(B):
        print(f"warning: comparing {len(common)} common images (A has {len(A)}, B has {len(B)})")
    A, B = A.loc[common], B.loc[common]
    print(f"paired comparison on {len(common)} test images\n")
    print(f"{'measure':<14}{a.name_a:>12}{a.name_b:>12}{'A - B':>10}{'95% CI (mean)':>22}{'95% CI (median)':>22}{'p (Wilcoxon)':>14}")
    for key, label in (("landmark", "landmark px"), ("ofd_len", "OFD len px"), ("bpd_len", "BPD len px")):
        x, y = A[key].to_numpy(), B[key].to_numpy()
        d = x - y
        ci_mean = bootstrap_ci(d, np.mean)
        ci_med = bootstrap_ci(d, np.median)
        p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
        print(f"{label:<14}{x.mean():>12.2f}{y.mean():>12.2f}{d.mean():>10.2f}"
              f"{f'[{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]':>22}{f'[{ci_med[0]:.2f}, {ci_med[1]:.2f}]':>22}{p:>14.4f}")
    print(f"\nmedian landmark px: {a.name_a} {A['landmark'].median():.2f} | {a.name_b} {B['landmark'].median():.2f}")
    print("Holm-correct the p-values when several comparisons are reported together.")


if __name__ == "__main__":
    main()
