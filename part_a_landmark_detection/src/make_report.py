#!/usr/bin/env python
"""Evaluation report for one or more models: tables, paired comparison, figures and a written report.

One model:
    python make_report.py --runs runs/hrnet_w32_base/eval_test --names heatmap --out report/heatmap

Two or more models (the first two are compared head-to-head):
    python make_report.py --runs runs/hrnet_run1/eval_test runs/coord_baseline/eval_test \
        --names heatmap coord --splits_dir data_processed --out report/comparison

Each --runs entry is an evaluation folder written by scripts/evaluate.py (per_image.csv, summary.json); the
training log and configuration are read from the parent run folder when present.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# works whether this file sits at the package root or inside src/ (then the package root is its parent)
ROOT = HERE.parent if HERE.name == "src" else HERE
sys.path.insert(0, str(ROOT))

import figures as F  # noqa: E402
from report import build_models, metrics_table, model_summary, paired_comparison, write_report  # noqa: E402
from mm_metrics import summary_text  # noqa: E402


GUIDE = {
    "fig01_training_curves.png": ("Training curves", "Read: is validation error still falling at the end, and where is the best epoch?"),
    "fig02_error_distribution.png": ("Landmark error distribution", "Read: how far apart the mean and median are, and how long the right tail is."),
    "fig03_success_curve.png": ("Cumulative success curve", "Read: the share of landmarks accurate to each threshold; the clinical reference is marked."),
    "fig04_error_per_landmark.png": ("Error per landmark", "Read: whether one endpoint is consistently harder than the other."),
    "fig05_bland_altman.png": ("Biometry agreement (Bland-Altman)", "Read: bias near zero and narrow limits of agreement indicate clinically usable lengths."),
    "fig06_length_error.png": ("Length error", "Read: the share of images whose length is within 5 % of the true value."),
    "fig07_along_across.png": ("Error along versus across the line", "Read: most error across the line means lengths are more reliable than the landmark error suggests."),
    "fig08_error_vs_size.png": ("Error versus head size", "Read: a rising line means small heads are relatively harder."),
    "fig09_swap_diagnostic.png": ("Endpoint order diagnostic", "Read: a high swap share means a labelling-convention problem to fix before modelling."),
    "fig10_subgroups.png": ("Error by subgroup", "Read: whether any head size or image size group is clearly worse for a model."),
    "fig11_head_to_head.png": ("Head-to-head comparison", "Read: the paired differences and whether their intervals exclude zero."),
    "fig12_error_concentration.png": ("Error concentration", "Read: how much of the total error comes from the worst images."),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="evaluation folders (runs/<run>/eval_<split>)")
    ap.add_argument("--names", nargs="+", default=None, help="labels for the models, same order as --runs")
    ap.add_argument("--splits_dir", default=None, help="data_processed/ for image size and patient groups")
    ap.add_argument("--pixel_spacing_mm", type=float, default=None,
                    help="mm per pixel. Required: all results are reported in millimetres")
    ap.add_argument("--allow_pixels", action="store_true", help="diagnostic only: report in pixels if no spacing is known")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    if a.names and len(a.names) != len(a.runs):
        ap.error("--names must have the same number of entries as --runs")

    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    models, unit, demo = build_models(a.runs, a.names, a.pixel_spacing_mm, a.splits_dir)
    if not unit["mm_per_px"] and not a.allow_pixels:
        sys.exit("No pixel spacing is known, so results cannot be reported in millimetres.\n"
                 "Pass --pixel_spacing_mm (the spacing of these images), or --allow_pixels for a pixel-only diagnostic.")
    if not unit["mm_per_px"]:
        print("WARNING: no pixel spacing; figures and tables are in PIXELS (diagnostic run).")
    note = ("DEMO DATA: synthetic, for checking the tooling only" if demo else
            (f"Units: mm (pixel spacing {unit['mm_per_px']} mm/px)" if unit["mm_per_px"] else "Units: pixels (diagnostic)"))
    note += f" | n = {', '.join(str(len(m['df'])) + ' ' + m['name'] for m in models)}"
    fd = out / "figures"

    figs = [
        ("fig02_error_distribution.png", F.fig_error_distribution),
        ("fig03_success_curve.png", F.fig_success_curve),
        ("fig04_error_per_landmark.png", F.fig_per_landmark),
        ("fig05_bland_altman.png", F.fig_bland_altman),
        ("fig06_length_error.png", F.fig_length_error),
        ("fig07_along_across.png", F.fig_error_decomposition),
        ("fig08_error_vs_size.png", F.fig_error_vs_size),
        ("fig09_swap_diagnostic.png", F.fig_swap_diagnostic),
        ("fig10_subgroups.png", F.fig_strata),
        ("fig12_error_concentration.png", F.fig_concentration),
    ]
    for name, fn in figs:
        fn(models, unit, fd / name, note)
    if any(m["run"]["log"] is not None for m in models):
        F.fig_training(models, unit, fd / "fig01_training_curves.png", note)

    comparison, paired_table, n_common, u = None, None, None, None
    if len(models) >= 2:
        paired_table, fin, n_common, u = paired_comparison(models, unit, a.n_boot)
        comparison = True
        F.fig_paired(models, unit, fd / "fig11_head_to_head.png", note, fin)
        paired_table.to_csv(out / "paired_tests.csv", index=False)

    figs_done = [(fn, GUIDE[fn][0], GUIDE[fn][1]) for fn in sorted(GUIDE) if (fd / fn).exists()]
    write_report(models, unit, demo, out, a.n_boot, comparison, figs_done, paired_table, n_common, u)
    for m in models:
        s = model_summary(m, unit)
        title = f"EVALUATION SUMMARY: {m['name']} ({m['kind']})  [all values in millimetres]"
        print("\n" + summary_text(s, title, [f"images evaluated  : {s['n_images']}",
                                              f"pixel spacing    : {unit['mm_per_px']} mm/px"]))
    print(f"\nreport written to {out/'report.md'}")
    print(f"tables: {out/'metrics_mm.csv'}")
    print(f"figures in {fd}")


if __name__ == "__main__":
    main()