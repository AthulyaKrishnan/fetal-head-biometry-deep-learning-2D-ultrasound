# #!/usr/bin/env python
# """Figures, tables and statistics for the report, from the full-evaluation outputs of two (or more) models.

# Example (after running scripts/evaluate.py once per model):
#     python tools/make_report_figures.py \
#         --evals attention=eval_full/attention_unet baseline=eval_full/baseline_unet \
#         --runs  attention=runs/attn_default   baseline=runs/baseline \
#         --out   report_figures

# Writes into --out:
#     learning_curves.png        validation Dice and loss per epoch (from runs/<run>/log.csv)
#     metric_boxplots.png        distribution of the main per-image metrics, one panel per metric
#     dice_per_subject.png       paired per-subject Dice (sorted), shows where the models disagree
#     area_bland_altman.png      Bland-Altman of predicted vs ground-truth area (bias and limits of agreement)
#     area_scatter.png           predicted vs ground-truth area with the identity line
#     roc_pr_pooled.png          pooled ROC and precision-recall curves
#     biometry_errors.png        OFD / BPD end-point errors and HC error (Part B)
#     results_table.md           mean +/- std and median [IQR] per model, ready to paste into the report
#     stats_comparison.csv       paired Wilcoxon signed-rank test between models, per metric
# """
# import argparse
# from pathlib import Path

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt  # noqa: E402
# import numpy as np  # noqa: E402
# import pandas as pd  # noqa: E402
# from scipy.stats import wilcoxon  # noqa: E402

# MAIN_METRICS = ["dice", "iou", "hausdorff_95_mm", "average_surface_distance_mm",
#                 "normalised_surface_distance", "sensitivity", "specificity", "precision", "roc_auc", "pr_auc"]
# LOWER_IS_BETTER = {"hausdorff_95_mm", "average_surface_distance_mm", "hausdorff_distance_mm", "diceCEloss"}


# def parse_pairs(items):
#     out = {}
#     for it in items:
#         k, v = it.split("=", 1)
#         out[k] = Path(v)
#     return out


# def load_subjects(eval_dirs):
#     dfs = {}
#     for name, d in eval_dirs.items():
#         f = next(d.glob("*_subject_metrics.csv"))
#         dfs[name] = pd.read_csv(f)
#     return dfs


# def fmt(mean, std):
#     return f"{mean:.3f} ± {std:.3f}"


# def learning_curves(run_dirs, out):
#     fig, ax = plt.subplots(1, 2, figsize=(11, 4))
#     for name, d in run_dirs.items():
#         log = pd.read_csv(d / "log.csv")
#         ax[0].plot(log["epoch"], log["val_loss"], label=f"{name} (val)")
#         ax[0].plot(log["epoch"], log["train_loss"], "--", label=f"{name} (train)")
#         ax[1].plot(log["epoch"], log["val_dice"], label=name)
#     ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend(fontsize=8)
#     ax[1].set(title="Validation Dice", xlabel="epoch"); ax[1].legend()
#     fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=150); plt.close(fig)


# def metric_boxplots(dfs, out):
#     metrics = MAIN_METRICS
#     n = len(metrics)
#     cols = 5
#     rows = int(np.ceil(n / cols))
#     fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
#     for ax, m in zip(axes.ravel(), metrics):
#         data = [pd.to_numeric(df[m], errors="coerce").dropna() for df in dfs.values()]
#         ax.boxplot(data, tick_labels=list(dfs.keys()), showmeans=True)
#         ax.set_title(m.replace("_", " "), fontsize=9)
#         ax.tick_params(axis="x", labelrotation=20, labelsize=8)
#     for ax in axes.ravel()[n:]:
#         ax.axis("off")
#     fig.tight_layout(); fig.savefig(out / "metric_boxplots.png", dpi=150); plt.close(fig)


# def dice_per_subject(dfs, out):
#     names = list(dfs.keys())
#     base = dfs[names[0]][["subject_id", "dice"]].rename(columns={"dice": names[0]})
#     merged = base
#     for n in names[1:]:
#         merged = merged.merge(dfs[n][["subject_id", "dice"]].rename(columns={"dice": n}), on="subject_id")
#     merged = merged.sort_values(names[0]).reset_index(drop=True)
#     fig, ax = plt.subplots(figsize=(9, 4))
#     for n in names:
#         ax.plot(merged.index, merged[n], marker=".", lw=0.8, label=n)
#     ax.set(xlabel="subjects (sorted by first model's Dice)", ylabel="Dice", title="Paired per-subject Dice")
#     ax.legend(); fig.tight_layout(); fig.savefig(out / "dice_per_subject.png", dpi=150); plt.close(fig)


# def area_plots(dfs, out):
#     fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
#     for n, df in dfs.items():
#         gt = df["gt_area_network_px"].astype(float)
#         pr = df["pred_area_network_px"].astype(float)
#         mean_area = (gt + pr) / 2
#         diff = pr - gt
#         ax[0].scatter(mean_area, diff, s=10, alpha=0.6, label=n)
#         ax[1].scatter(gt, pr, s=10, alpha=0.6, label=n)
#     titles = []
#     for n, df in dfs.items():                         # bias and 95% limits of agreement
#         gt = df["gt_area_network_px"].astype(float)
#         pr = df["pred_area_network_px"].astype(float)
#         d = pr - gt
#         bias, sd = d.mean(), d.std()
#         ax[0].axhline(bias, lw=1, ls="--", color="gray")
#         ax[0].axhline(bias + 1.96 * sd, lw=0.8, ls=":", color="gray")
#         ax[0].axhline(bias - 1.96 * sd, lw=0.8, ls=":", color="gray")
#         titles.append(f"{n}: bias {bias:.0f} px, LoA ±{1.96 * sd:.0f} px")
#     ax[0].set_title("Bland-Altman | " + " ; ".join(titles), fontsize=8)
#     lim = max(dfs[list(dfs)[0]]["gt_area_network_px"].max(), dfs[list(dfs)[0]]["pred_area_network_px"].max()) * 1.05
#     ax[1].plot([0, lim], [0, lim], "k--", lw=1)
#     ax[0].set(xlabel="mean of predicted and GT area (pixels, network grid)", ylabel="predicted - GT (pixels)")
#     ax[1].set(xlabel="ground-truth area (pixels)", ylabel="predicted area (pixels)", title="Predicted vs GT area")
#     ax[0].legend(); ax[1].legend()
#     fig.tight_layout(); fig.savefig(out / "area_bland_altman_and_scatter.png", dpi=150); plt.close(fig)


# def roc_pr(eval_dirs, out):
#     fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
#     for n, d in eval_dirs.items():
#         z = np.load(d / "pooled_curves.npz")
#         ax[0].plot(z["fpr"], z["tpr"], label=n)
#         ax[1].plot(z["recall"], z["precision"], label=n)
#     ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
#     ax[0].set(title="Pooled ROC", xlabel="FPR", ylabel="TPR"); ax[1].set(title="Pooled PR", xlabel="recall", ylabel="precision")
#     ax[0].legend(); ax[1].legend()
#     fig.tight_layout(); fig.savefig(out / "roc_pr_pooled.png", dpi=150); plt.close(fig)


# def biometry(eval_dirs, out):
#     names, data_ofd, data_bpd, data_hc = [], [], [], []
#     for n, d in eval_dirs.items():
#         b = pd.read_csv(next(d.glob("*_biometry_points.csv")))
#         b = b[b["ellipse_fit_ok"].astype(str).str.lower() == "true"]
#         names.append(n)
#         data_ofd.append(b["ofd_point_error_mm"].dropna())
#         data_bpd.append(b["bpd_point_error_mm"].dropna())
#         data_hc.append(b["hc_abs_error_mm"].dropna())
#     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#     for a, data, t in zip(ax, [data_ofd, data_bpd, data_hc],
#                           ["OFD end-point error (mm)", "BPD end-point error (mm)", "HC absolute error (mm)"]):
#         a.boxplot(data, tick_labels=names, showmeans=True); a.set_title(t, fontsize=10)
#     fig.tight_layout(); fig.savefig(out / "biometry_errors.png", dpi=150); plt.close(fig)


# def tables(dfs, out):
#     lines = ["| metric | " + " | ".join(f"{n} mean ± std" for n in dfs) + " | " +
#              " | ".join(f"{n} median [IQR]" for n in dfs) + " |",
#              "|---|" + "---|" * (2 * len(dfs))]
#     for m in MAIN_METRICS + ["hausdorff_distance_mm", "abs_area_diff_network_px", "area_diff_network_px", "diceCEloss"]:
#         if m not in next(iter(dfs.values())).columns:
#             continue
#         ms, md = [], []
#         for df in dfs.values():
#             v = pd.to_numeric(df[m], errors="coerce").dropna()
#             ms.append(fmt(v.mean(), v.std()))
#             md.append(f"{v.median():.3f} [{v.quantile(0.25):.3f}, {v.quantile(0.75):.3f}]")
#         lines.append(f"| {m} | " + " | ".join(ms) + " | " + " | ".join(md) + " |")
#     (out / "results_table.md").write_text("\n".join(lines) + "\n")


# def stats(dfs, out):
#     names = list(dfs)
#     if len(names) != 2:
#         print("stats_comparison: needs exactly two models, skipped")
#         return
#     a, b = names
#     merged = dfs[a].merge(dfs[b], on="subject_id", suffixes=("_a", "_b"))
#     rows = []
#     for m in MAIN_METRICS + ["diceCEloss", "abs_area_diff_network_px"]:
#         x = pd.to_numeric(merged[f"{m}_a"], errors="coerce")
#         y = pd.to_numeric(merged[f"{m}_b"], errors="coerce")
#         ok = x.notna() & y.notna()
#         x, y = x[ok], y[ok]
#         d = x - y
#         p = wilcoxon(x, y).pvalue if len(d) > 0 and (d != 0).any() else float("nan")
#         rows.append(dict(metric=m, n_pairs=int(ok.sum()), median_a=x.median(), median_b=y.median(),
#                          median_diff_a_minus_b=d.median(), mean_diff_a_minus_b=d.mean(),
#                          wilcoxon_p=p, lower_is_better=m in LOWER_IS_BETTER))
#     pd.DataFrame(rows).to_csv(out / "stats_comparison.csv", index=False)


# def main():
#     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--evals", nargs="+", required=True, help="name=eval_dir (output of scripts/evaluate.py)")
#     ap.add_argument("--runs", nargs="*", default=[], help="name=run_dir (contains log.csv)")
#     ap.add_argument("--out", default="report_figures")
#     a = ap.parse_args()
#     out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

#     eval_dirs = parse_pairs(a.evals)
#     dfs = load_subjects(eval_dirs)
#     if a.runs:
#         learning_curves(parse_pairs(a.runs), out)
#     metric_boxplots(dfs, out)
#     dice_per_subject(dfs, out)
#     area_plots(dfs, out)
#     roc_pr(eval_dirs, out)
#     biometry(eval_dirs, out)
#     tables(dfs, out)
#     stats(dfs, out)
#     print(f"written to {out}/:")
#     for f in sorted(out.iterdir()):
#         print("  ", f.name)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
"""Report figures with the numbers printed on them, from the full-evaluation outputs of two or more models.

Run scripts/evaluate.py once per model first, then:

    python tools/make_report_figures.py \
        --evals attention=eval_full/attention_unet baseline=eval_full/baseline_unet \
        --runs  attention=runs/attn_default baseline=runs/baseline \
        --out   report_figures

Output files (numbered in the order they should appear in the report):
    01_learning_curves.png     loss and validation Dice per epoch, best epoch marked
    02_metric_panels.png       one panel per metric: every image as a dot, mean, SD, median, IQR, n
    03_dice_distribution.png   histogram of per-image Dice with the share of images above a threshold
    04_paired_dice.png         per-image Dice of model A against model B (two models only)
    05_area_bland_altman.png   agreement of predicted and GT area: bias and 95% limits, in percent
    06_roc_pr_pooled.png       pooled pixel-level ROC and precision-recall curves with AUC values
    07_biometry_errors.png     OFD / BPD end-point errors and HC error, with the share within tolerance
    results_table.md           mean ± SD and median [IQR] per model (paste-ready)
    stats_comparison.csv       paired Wilcoxon test and bootstrap CI for the difference (two models)
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import auc  # noqa: E402

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

PRETTY = {
    "dice": "Dice coefficient",
    "iou": "IoU (Jaccard index)",
    "sensitivity": "Sensitivity (recall)",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1_score": "F1 score",
    "roc_auc": "ROC-AUC (pixel level)",
    "pr_auc": "PR-AUC (pixel level)",
    "hausdorff_distance_mm": "Hausdorff distance",
    "hausdorff_95_mm": "HD95 (95th percentile Hausdorff)",
    "average_surface_distance_mm": "Average surface distance",
    "normalised_surface_distance": "Normalised surface distance",
    "diceCEloss": "Training loss (BCE + Dice)",
    "area_diff_network_px": "Area error (predicted - GT)",
    "abs_area_diff_network_px": "Absolute area error",
}
HIGHER_IS_BETTER = {"dice", "iou", "sensitivity", "specificity", "precision", "f1_score",
                    "roc_auc", "pr_auc", "normalised_surface_distance"}
# metrics shown in the metric panel figure and in the statistics (order = order in the figure)
GRID = ["dice", "iou", "sensitivity", "specificity", "precision", "roc_auc", "pr_auc",
        "hausdorff_95_mm", "average_surface_distance_mm", "normalised_surface_distance",
        "abs_area_diff_network_px", "diceCEloss"]

CAPTIONS = {
    "learning": (
        "How to read: dashed lines are training, solid lines are validation. The star marks the epoch with the "
        "highest validation Dice; that checkpoint (best.pt) is the one used on the test set. Validation loss rising "
        "while training loss keeps falling is a sign of overfitting."),
    "metrics": (
        "How to read: each dot is one test image, spread sideways so dots do not overlap. The diamond is the mean, "
        "the vertical bar is ±1 standard deviation, and the short black line is the median. Under each name: mean ± SD, "
        "median, IQR (middle 50% of images, from the 25th to the 75th percentile) and n. The vertical axis covers the "
        "1st to 99th percentile so a few extreme images do not compress the plot; the number of hidden points is printed "
        "in the corner. Where two models are compared, the paired Wilcoxon p-value (same images) is shown in the top-left "
        "corner of each panel; p < 0.05 means the difference is unlikely to be chance."),
    "dice": (
        "How to read: each bar counts the test images whose Dice falls in that range. The solid line is the mean and the "
        "dotted line the median. The dashed black line is the threshold named in the legend. Images to the left of it are "
        "the failure cases to inspect in the overlays_worst/ folder."),
    "paired": (
        "How to read: each dot is one test image. Its x-position is the first model's Dice and its y-position the second "
        "model's Dice. Dots above the dashed diagonal are images where the second model did better. The 95% CI comes from "
        "2000 bootstrap resamples of the images; if it does not contain 0, the mean difference is unlikely to be chance."),
    "ba": (
        "How to read (Bland-Altman): each dot is one image. x = average of the predicted and ground-truth area, y = "
        "(predicted - GT) as a percentage of the GT area. The solid line is the bias (average over- or under-segmentation). "
        "The dashed lines are the 95% limits of agreement (bias ± 1.96 SD): about 95% of images are expected to lie between "
        "them. A cloud with no slope across x means the error does not depend on head size."),
    "roc": (
        "How to read: the ROC curve shows how well the probability map separates head pixels from background at every "
        "threshold; the closer to the top-left corner, the better (AUC 1.0 is perfect, 0.5 is chance). The precision-recall "
        "curve is more informative when the head occupies a small part of the image: higher is better. Pixels from all test "
        "images are pooled, on the network grid and subsampled."),
    "biometry": (
        "How to read: as in the metric panels, each dot is one image. OFD and BPD errors are the distances between predicted "
        "and ground-truth ellipse end-points (OFD = occipito-frontal diameter, BPD = biparietal diameter). The head "
        "circumference (HC) error is the absolute difference of the ellipse perimeter. The dashed line is the tolerance "
        "(--bio_tol_mm); the label under each name gives the share of images within it. Images where the ellipse could "
        "not be fitted are excluded."),
}


def parse_pairs(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = Path(v)
    return out


def display_unit(col, distance_unit):
    if col.endswith("_mm"):
        return distance_unit
    if col.endswith("_px"):
        return "px"
    return ""


def values(df, col):
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)


def fmt(x):
    if not np.isfinite(x):
        return "nan"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3f}"


def boot_ci(d, n=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def finish(fig, path, caption, suptitle=None, dpi=150):
    """Lay out the figure, put the reading guide underneath it, and save."""
    text, bottom = "", 0.02
    if caption:
        text = "\n".join(textwrap.wrap(caption, width=int(fig.get_figwidth() * 12.5)))
        bottom = 0.03 + 0.16 * (text.count("\n") + 1) / fig.get_figheight()
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, bottom, 1, 0.93 if suptitle else 0.97])
    if caption:
        fig.text(0.01, 0.005, text, fontsize=8, va="bottom", ha="left", color="#333333")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- statistics
def paired_stats(dfs, seed=0):
    names = list(dfs)
    if len(names) != 2:
        return pd.DataFrame(), {}
    a, b = names
    m = dfs[a].merge(dfs[b], on="subject_id", suffixes=("_a", "_b"))
    rows, pvals = [], {}
    for col in GRID:
        x = pd.to_numeric(m[f"{col}_a"], errors="coerce")
        y = pd.to_numeric(m[f"{col}_b"], errors="coerce")
        ok = x.notna() & y.notna()
        x, y = x[ok].to_numpy(float), y[ok].to_numpy(float)
        if len(x) < 2:
            continue
        d = y - x                                    # second model minus first model
        p = wilcoxon(x, y).pvalue if (d != 0).any() else float("nan")
        lo, hi = boot_ci(d, seed=seed)
        higher = col in HIGHER_IS_BETTER
        better = b if ((d.mean() > 0) == higher) else a
        pvals[col] = p
        rows.append(dict(metric=col, direction="higher better" if higher else "lower better", n_pairs=len(x),
                        median_a=np.median(x), median_b=np.median(y),
                        mean_diff_b_minus_a=d.mean(), ci95_low=lo, ci95_high=hi,
                        wilcoxon_p=p, significant_0_05=bool(p < 0.05) if np.isfinite(p) else False,
                        better_model=better))
    return pd.DataFrame(rows), pvals


# --------------------------------------------------------------------- panels
def dot_panel(ax, groups, names, title, higher_better, unit, bounded=False, threshold=None, pval=None, seed=0):
    """One dot per image for each model, with mean ± SD, median and IQR printed under the model name."""
    rng = np.random.default_rng(seed)
    nonempty = [g for g in groups if len(g)]
    if not nonempty:
        ax.set_axis_off()
        return
    pooled = np.concatenate(nonempty)
    lo, hi = np.percentile(pooled, [1, 99])
    pad = 0.12 * max(hi - lo, 1e-9)
    lo, hi = lo - pad, hi + pad
    if bounded:
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    hidden, ticks, labels = 0, [], []
    for i, (v, name) in enumerate(zip(groups, names)):
        ticks.append(i)
        if len(v) == 0:
            labels.append(f"{name}\nno data")
            continue
        c = COLORS[i % len(COLORS)]
        inside = (v >= lo) & (v <= hi)
        hidden += int((~inside).sum())
        jitter = rng.uniform(-0.14, 0.14, size=len(v))
        ax.scatter(i + jitter[inside], v[inside], s=12, color=c, alpha=0.45, edgecolor="none", zorder=2)
        mean = v.mean()
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        med = np.median(v)
        q1, q3 = np.percentile(v, [25, 75])
        ax.errorbar(i + 0.3, mean, yerr=sd, fmt="D", color=c, mec="black", ms=8, capsize=6, lw=2, zorder=4)
        ax.hlines(med, i + 0.22, i + 0.38, colors="black", lw=2.5, zorder=5)
        label = (f"{name}\nmean {fmt(mean)} ± {fmt(sd)}\nmedian {fmt(med)}\n"
                 f"IQR {fmt(q1)} to {fmt(q3)}\nn = {len(v)}")
        if threshold is not None:
            value, text = threshold
            label += f"\n{(v <= value).mean() * 100:.0f}% {text}"
        labels.append(label)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.6, len(groups) - 0.5)
    ax.set_ylim(lo, hi)
    ax.set_title(f"{title}\n{'↑ higher is better' if higher_better else '↓ lower is better'}", fontsize=10)
    if unit:
        ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=0.3)
    if threshold is not None:
        ax.axhline(threshold[0], ls="--", color="gray", lw=1)
    if pval is not None and np.isfinite(pval):
        ax.text(0.02, 0.98, f"paired Wilcoxon p = {pval:.2g}", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    if hidden:
        ax.text(0.98, 0.02, f"{hidden} point(s) beyond axis, not drawn", transform=ax.transAxes, ha="right",
                fontsize=7, color="firebrick")


def metric_panels(dfs, distance_unit, pvals, out):
    cols = [c for c in GRID if all(c in df.columns for df in dfs.values())]
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 5.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    names = list(dfs)
    for ax, c in zip(axes, cols):
        dot_panel(ax, [values(df, c) for df in dfs.values()], names, PRETTY[c], c in HIGHER_IS_BETTER,
                  display_unit(c, distance_unit), bounded=c in HIGHER_IS_BETTER, pval=pvals.get(c))
    for ax in axes[len(cols):]:
        ax.set_axis_off()
    finish(fig, out / "02_metric_panels.png", CAPTIONS["metrics"],
           suptitle="Per-image test results: each dot is one test image")


def dice_distribution(dfs, thr, out):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    all_v = np.concatenate([values(df, "dice") for df in dfs.values()])
    lo = max(0.0, np.floor(all_v.min() * 20) / 20 - 0.02)
    bins = np.linspace(lo, 1.0, 41)
    for i, (name, df) in enumerate(dfs.items()):
        v = values(df, "dice")
        c = COLORS[i % len(COLORS)]
        share = (v >= thr).mean() * 100
        ax.hist(v, bins=bins, color=c, alpha=0.35, edgecolor=c,
                label=f"{name}: mean {v.mean():.3f}, median {np.median(v):.3f}; "
                      f"{share:.0f}% of images have Dice ≥ {thr:g}")
        ax.axvline(v.mean(), color=c, lw=2.2)
        ax.axvline(np.median(v), color=c, lw=1.5, ls=":")
    ax.axvline(thr, color="black", ls="--", lw=1)
    ax.set(xlabel="Dice per test image", ylabel="number of images", xlim=(lo, 1.0))
    ax.set_title("Distribution of per-image Dice\n(solid line = mean, dotted line = median, dashed = threshold)",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    finish(fig, out / "03_dice_distribution.png", CAPTIONS["dice"])


def paired_dice(dfs, out, seed=0):
    names = list(dfs)
    if len(names) != 2:
        return
    a, b = names
    m = dfs[a][["subject_id", "dice"]].merge(dfs[b][["subject_id", "dice"]], on="subject_id",
                                             suffixes=(f"_{a}", f"_{b}"))
    x, y = m[f"dice_{a}"].to_numpy(float), m[f"dice_{b}"].to_numpy(float)
    d = y - x
    lo_ci, hi_ci = boot_ci(d, seed=seed)
    # zoom to the bulk of the data; the few failure cases outside the zoom still count in all statistics
    lo = max(0.0, np.percentile(np.concatenate([x, y]), 5) - 0.02)
    keep = (x >= lo) & (y >= lo)
    hidden = int((~keep).sum())
    fig, ax = plt.subplots(figsize=(6.8, 6.6))
    ax.scatter(x[keep], y[keep], s=22, alpha=0.6, color="#444444", edgecolor="none", zorder=2)
    ax.plot([lo, 1.0], [lo, 1.0], "k--", lw=1, zorder=1)
    ax.set(xlim=(lo, 1.0), ylim=(lo, 1.0), xlabel=f"Dice, {a}", ylabel=f"Dice, {b}")
    nb, na = int((y > x).sum()), int((x > y).sum())
    nt = len(x) - nb - na
    txt = (f"{b} higher on {nb} images\n{a} higher on {na} images"
           + (f"\ntied on {nt} images" if nt else "")
           + f"\n\nmean difference ({b} − {a}) = {d.mean():+.4f}\n"
             f"95% CI (bootstrap) [{lo_ci:+.4f}, {hi_ci:+.4f}]"
           + (f"\n\n{hidden} image(s) with Dice below {lo:.2f} are outside the zoom;\n"
              f"they are included in all numbers" if hidden else ""))
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"Paired Dice per image (n = {len(x)})\npoints above the dashed line: {b} is better", fontsize=11)
    finish(fig, out / "04_paired_dice.png", CAPTIONS["paired"])


def area_bland_altman(dfs, out):
    names = list(dfs)
    pct_all = []
    for df in dfs.values():
        ok = df["gt_area_network_px"] > 0
        g = df.loc[ok, "gt_area_network_px"].astype(float)
        p = df.loc[ok, "pred_area_network_px"].astype(float)
        pct_all.append(((p - g) / g * 100).to_numpy())
    lo, hi = np.percentile(np.concatenate(pct_all), [1, 99])
    pad = 0.15 * (hi - lo)
    fig, axes = plt.subplots(1, len(names), figsize=(6.4 * len(names), 5.6), squeeze=False)
    for i, (ax, name, df) in enumerate(zip(axes[0], names, dfs.values())):
        ok = df["gt_area_network_px"] > 0
        g = df.loc[ok, "gt_area_network_px"].astype(float).to_numpy()
        p = df.loc[ok, "pred_area_network_px"].astype(float).to_numpy()
        mean_area = (g + p) / 2
        pct = (p - g) / g * 100
        bias, sd = pct.mean(), pct.std(ddof=1)
        up, low = bias + 1.96 * sd, bias - 1.96 * sd
        c = COLORS[i % len(COLORS)]
        ax.scatter(mean_area, pct, s=14, alpha=0.5, color=c, edgecolor="none", zorder=2)
        ax.axhline(0, color="black", lw=0.8, zorder=1)
        ax.axhline(bias, color=c, lw=2, zorder=3)
        ax.axhline(up, color=c, ls="--", lw=1.2, zorder=3)
        ax.axhline(low, color=c, ls="--", lw=1.2, zorder=3)
        tr = ax.get_yaxis_transform()
        ax.text(1.01, up, f"upper limit\n{up:+.1f}%", transform=tr, va="center", fontsize=8.5, color=c)
        ax.text(1.01, bias, f"bias\n{bias:+.1f}%", transform=tr, va="center", fontsize=9, color=c,
                fontweight="bold")
        ax.text(1.01, low, f"lower limit\n{low:+.1f}%", transform=tr, va="center", fontsize=8.5, color=c)
        ax.text(0.02, 0.98, "above 0 = prediction too large\nbelow 0 = prediction too small",
                transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.6"))
        ax.set(title=f"{name}  (n = {len(pct)} images)",
               xlabel="mean of GT and predicted area (pixels, network grid)",
               ylabel="difference, % of GT area", ylim=(lo - pad, hi + pad))
        ax.grid(alpha=0.3)
    finish(fig, out / "05_area_bland_altman.png", CAPTIONS["ba"],
           suptitle="Agreement between predicted and ground-truth head area")


def roc_pr(eval_dirs, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
    for i, (name, d) in enumerate(eval_dirs.items()):
        z = np.load(d / "pooled_curves.npz")
        c = COLORS[i % len(COLORS)]
        ax[0].plot(z["fpr"], z["tpr"], color=c, lw=2.2, label=f"{name}: AUC = {auc(z['fpr'], z['tpr']):.4f}")
        ax[1].plot(z["recall"], z["precision"], color=c, lw=2.2,
                   label=f"{name}: AUC = {auc(z['recall'], z['precision']):.4f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance: AUC = 0.5")
    ax[0].set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
              title="ROC curve (all test pixels pooled)", xlim=(0, 1), ylim=(0, 1.01))
    ax[1].set(xlabel="Recall (sensitivity)", ylabel="Precision",
              title="Precision-recall curve (all test pixels pooled)", xlim=(0, 1), ylim=(0, 1.01))
    for a_ in ax:
        a_.legend(fontsize=9, loc="lower left")
        a_.grid(alpha=0.3)
    finish(fig, out / "06_roc_pr_pooled.png", CAPTIONS["roc"])


def learning_curves(run_dirs, out):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for i, (name, d) in enumerate(run_dirs.items()):
        log = pd.read_csv(d / "log.csv")
        c = COLORS[i % len(COLORS)]
        ax[0].plot(log["epoch"], log["train_loss"], "--", color=c, lw=1.5, label=f"{name}: training")
        ax[0].plot(log["epoch"], log["val_loss"], "-", color=c, lw=2, label=f"{name}: validation")
        ax[1].plot(log["epoch"], log["val_dice"], color=c, lw=2, label=name)
        best = log.loc[log["val_dice"].idxmax()]
        ax[1].scatter(best["epoch"], best["val_dice"], marker="*", s=230, color=c, edgecolor="black", zorder=5)
        ax[1].annotate(f"{name}: best {best['val_dice']:.3f} at epoch {int(best['epoch'])}",
                       (best["epoch"], best["val_dice"]), xytext=(12, -30 - 22 * i), textcoords="offset points",
                       fontsize=8.5, color=c, arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
    ax[0].set(title="Loss (lower is better)", xlabel="epoch", ylabel="loss")
    ax[1].set(title="Validation Dice (higher is better)", xlabel="epoch", ylabel="Dice")
    for a_ in ax:
        a_.legend(fontsize=8.5)
        a_.grid(alpha=0.3)
    finish(fig, out / "01_learning_curves.png", CAPTIONS["learning"])


def biometry(eval_dirs, unit, tol_mm, out):
    names = list(eval_dirs)
    specs = [("ofd_point_error_mm", "OFD end-point error"),
             ("bpd_point_error_mm", "BPD end-point error"),
             ("hc_abs_error_mm", "Head circumference absolute error")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
    for ax, (col, title) in zip(axes, specs):
        groups = []
        for d in eval_dirs.values():
            b = pd.read_csv(next(d.glob("*_biometry_points.csv")))
            ok = b["ellipse_fit_ok"].astype(str).str.lower() == "true"
            groups.append(pd.to_numeric(b.loc[ok, col], errors="coerce").dropna().to_numpy(float))
        dot_panel(ax, groups, names, title, False, unit, threshold=(tol_mm, f"within {tol_mm:g} {unit}"))
    finish(fig, out / "07_biometry_errors.png", CAPTIONS["biometry"],
           suptitle="Biometry: distance between predicted and ground-truth measurements")


def results_table(dfs, out):
    names = list(dfs)
    lines = ["Mean ± SD and median [IQR] over the test images. ↑ higher is better, ↓ lower is better.", "",
             "| metric | " + " | ".join(f"{n}: mean ± SD" for n in names) + " | "
             + " | ".join(f"{n}: median [IQR]" for n in names) + " |",
             "|---|" + "---|" * (2 * len(names))]
    cols = GRID + ["area_diff_network_px"]
    for c in cols:
        if not all(c in df.columns for df in dfs.values()):
            continue
        unit = display_unit(c, "mm" if c.endswith("_mm") else "")
        arrow = "↑" if c in HIGHER_IS_BETTER else ("↓" if c.endswith(("_mm", "_px")) or c == "diceCEloss" else "")
        label = f"{PRETTY[c]}" + (f" [{unit}]" if unit else "") + (f" {arrow}" if arrow else "")
        ms, md = [], []
        for df in dfs.values():
            v = values(df, c)
            ms.append(f"{fmt(v.mean())} ± {fmt(v.std(ddof=1))}")
            md.append(f"{fmt(np.median(v))} [{fmt(np.percentile(v, 25))}, {fmt(np.percentile(v, 75))}]")
        lines.append(f"| {label} | " + " | ".join(ms) + " | " + " | ".join(md) + " |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", nargs="+", required=True, help="name=eval_dir (output of scripts/evaluate.py)")
    ap.add_argument("--runs", nargs="*", default=[], help="name=run_dir (folder containing log.csv)")
    ap.add_argument("--out", default="report_figures")
    ap.add_argument("--dice_threshold", type=float, default=0.9)
    ap.add_argument("--bio_tol_mm", type=float, default=2.0, help="tolerance for the biometry 'share within' label")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_dirs = parse_pairs(a.evals)
    dfs = {n: pd.read_csv(next(d.glob("*_subject_metrics.csv"))) for n, d in eval_dirs.items()}
    summaries = {n: json.loads(next(d.glob("*_summary.json")).read_text()) for n, d in eval_dirs.items()}
    unit = "px" if str(next(iter(summaries.values()))["distance_unit"]).startswith("px") else "mm"

    stats_df, pvals = paired_stats(dfs, a.seed)
    if a.runs:
        learning_curves(parse_pairs(a.runs), out)
    metric_panels(dfs, unit, pvals, out)
    dice_distribution(dfs, a.dice_threshold, out)
    paired_dice(dfs, out, a.seed)
    area_bland_altman(dfs, out)
    roc_pr(eval_dirs, out)
    biometry(eval_dirs, unit, a.bio_tol_mm, out)
    results_table(dfs, out)
    if len(dfs) == 2:
        stats_df.to_csv(out / "stats_comparison.csv", index=False)

    print(f"written to {out}/:")
    for f in sorted(out.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
