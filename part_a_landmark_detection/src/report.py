"""Millimetre summary table, paired comparison, figures and the written report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mm_metrics as M
from derived import derive
from load import join_splits, load_run, model_type, pixel_spacing
from stats import holm, paired_diff_ci, wilcoxon_p


def load_plausible(eval_dir):
    """Plausibility per image from the tester's flags.csv (in the results folder above eval_judges)."""
    p = Path(eval_dir).parent / "flags.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("image_name")["plausible"].astype(bool)


def build_models(run_dirs, names=None, spacing_override=None, splits_dir=None):
    meta = join_splits(splits_dir)
    models, spacings = [], []
    for i, d in enumerate(run_dirs):
        run = load_run(d, name=names[i] if names else None)
        df = derive(run, meta)
        lm_err = df[[c for c in df.columns if c.startswith("err_") and c.endswith("_px")]].to_numpy()
        plaus_map = load_plausible(d)
        plausible = None
        if plaus_map is not None:
            plausible = np.array([bool(plaus_map.get(n, False)) for n in run["names"]])
        models.append(dict(name=run["name"], run=run, df=df, lm_err=lm_err,
                           kind=model_type(run), plausible=plausible))
        spacings.append(pixel_spacing(run, spacing_override))
    known = [s for s in spacings if s]
    if known and len({round(s, 6) for s in known}) > 1:
        raise SystemExit(f"runs use different pixel spacings {known}; evaluate them with one spacing")
    mm = known[0] if known else None
    demo = any(m["run"]["config"].get("demo") for m in models)
    return models, dict(mm_per_px=mm), demo


def model_summary(m, unit, subset=None):
    run = m["run"]
    pred, gt = run["pred"], run["gt"]
    if subset is not None:
        pred, gt = pred[subset], gt[subset]
    return M.summarize(pred, gt, unit["mm_per_px"])


def metrics_table(models, unit):
    """Table of the same measures as the tester: rows = measures, columns = models (all in mm)."""
    cols, plaus_rows = {}, None
    for m in models:
        cols[m["name"]] = dict(M.rows(model_summary(m, unit)))
    labels = list(next(iter(cols.values())).keys())
    tbl = pd.DataFrame({k: pd.Series(v) for k, v in cols.items()}).reindex(labels)
    tbl.index.name = "measure (mm unless stated)"
    if all(m["plausible"] is not None for m in models):
        sub_cols = {}
        for m in models:
            s = model_summary(m, unit, m["plausible"])
            sub_cols[m["name"]] = dict(M.rows(s))
        sub = pd.DataFrame({k: pd.Series(v) for k, v in sub_cols.items()}).reindex(labels)
        sub.index.name = "plausible subset"
        plaus_rows = sub
    return tbl, plaus_rows


def markdown_table(df):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def paired_comparison(models, unit, n_boot=5000):
    """First two models on the images they share, in mm."""
    a, b = models[0]["df"], models[1]["df"]
    A = a[a["image_name"].isin(b["image_name"])].set_index("image_name").sort_index()
    B = b[b["image_name"].isin(A.index)].set_index("image_name").sort_index()
    f = unit["mm_per_px"]
    measures = [("landmark error (mean per image)", "mean_err_px"),
                ("OFD length abs error", "ofd_len_abs_err_px"),
                ("BPD length abs error", "bpd_len_abs_err_px")]
    out, pvals = [], []
    for label, col in measures:
        x, y = A[col].to_numpy() * f, B[col].to_numpy() * f
        lo, hi = paired_diff_ci(x, y, n=n_boot)
        p = wilcoxon_p(x, y)
        pvals.append(p)
        out.append(dict(measure=label, a_mean=x.mean(), b_mean=y.mean(), diff=(x - y).mean(),
                        ci_lo=lo, ci_hi=hi, p_raw=p, n=len(x)))
    for o, pa in zip(out, holm(pvals)):
        o["p_holm"] = pa
    table = pd.DataFrame(out)
    fig_rows = [dict(label=o["measure"] + " (mm)", diff=o["diff"], lo=o["ci_lo"], hi=o["ci_hi"],
                     p_holm=o["p_holm"]) for o in out]
    figure_input = dict(a_err=A["mean_err_px"].to_numpy() * f, b_err=B["mean_err_px"].to_numpy() * f,
                        rows=fig_rows)
    return table, figure_input, len(A), "mm"


def write_report(models, unit, demo, out, n_boot, comparison, figs_done, paired_table=None, n_common=None, u=None):
    out = Path(out)
    tbl, plaus = metrics_table(models, unit)
    counts = ", ".join(m["name"] + " " + str(len(m["df"])) for m in models)
    lines = ["# Evaluation report: fetal head landmark models (millimetres)\n"]
    if demo:
        lines.append("> **DEMO DATA.** At least one run is a synthetic demonstration. The numbers and figures show "
                     "that the tooling works and are not results.\n")
    lines.append(f"All values are in millimetres (pixel spacing {unit['mm_per_px']} mm/px). "
                 f"Images per model: {counts}.\n")

    lines.append("## 1. Summary, all evaluated images\n")
    lines.append(markdown_table(tbl) + "\n")
    if plaus is not None:
        lines.append("## 2. Plausible subset\n")
        lines.append("Rows whose landmarks are implausible for a fetal head (see flags.csv of the tester) are excluded.\n")
        lines.append(markdown_table(plaus) + "\n")

    lines.append("## 3. What the numbers say\n")
    for m in models:
        s = model_summary(m, unit)
        e = s["landmark_error_mm"]
        lines.append(f"**{m['name']}** ({m['kind']}). Median landmark error {e['median']:.2f} mm "
                     f"(mean {e['mean']:.2f} mm). Share of landmarks within 2 mm: {100*s['sdr']['<= 2.0 mm']:.1f} %. "
                     f"OFD length error median {s['OFD']['abs_error_pct_median']:.2f} % and BPD "
                     f"{s['BPD']['abs_error_pct_median']:.2f} % of the true length; HC ICC "
                     f"{s['HC']['icc_2_1']:.3f}.")
        swap = max(s["OFD"]["endpoints_swapped_better"], s["BPD"]["endpoints_swapped_better"])
        if swap >= 0.2:
            lines.append(f"Endpoints fit better in swapped order for {100*swap:.0f} % of images. Part of the error is "
                         "label order rather than localisation (Di Vece et al., 2026). Check the annotation convention.")
        if s["failures"]["catastrophic_share"] >= 0.05:
            lines.append(f"{100*s['failures']['catastrophic_share']:.0f} % of images exceed three times the median error; "
                         "inspect the worst overlays, because these failures dominate the mean.")
        lines.append("")

    if comparison is not None:
        lines.append("## 4. Head-to-head comparison (mm)\n")
        lines.append(f"Paired on {n_common} shared test images, {n_boot} bootstrap resamples, Wilcoxon signed-rank tests, "
                     f"Holm correction across the three measures. Negative differences favour {models[0]['name']}.\n")
        t = paired_table.copy()
        t.index = t["measure"]
        t = t.drop(columns=["measure", "n"]).round(4)
        t.index.name = "measure"
        lines.append(markdown_table(t) + "\n")
        for _, r in paired_table.iterrows():
            sig = "a difference" if r["p_holm"] < 0.05 else "no clear difference"
            better = models[0]["name"] if r["diff"] < 0 else models[1]["name"]
            lines.append(f"- {r['measure']}: {sig} (Holm p = {r['p_holm']:.3g}); the point estimate favours {better}.")
        lines.append("")

    lines.append("## 5. Figures (all in millimetres)\n")
    for fname, title, guide in figs_done:
        lines.append(f"### {title}\n\n![{title}](figures/{fname})\n\n{guide}\n")

    lines.append("## 6. Method notes\n")
    lines.append(
        "Landmark error is the straight-line distance between predicted and true points, averaged over the four "
        "landmarks of each image. The mean radial error and the successful detection rate at 2, 2.5, 3 and 4 mm follow "
        "Chen et al. (MICCAI 2019). The SDR-AUC is the area under the SDR curve from 0 to 10 mm, divided by 10. The "
        "endpoint-order-invariant NME follows Di Vece et al. (2026). Length agreement uses Bland and Altman limits of "
        "agreement and ICC(2,1) with absolute agreement (McGraw and Wong, 1996). Head circumference is "
        "1.62 x (BPD + OFD), as in the ISUOG practice guideline (2022). Confidence intervals are bootstrap intervals "
        "over images. Subset statistics use only rows that the tester flagged as plausible.\n")
    lines.append("## 7. Limits of this evaluation\n")
    lines.append(
        "A single test split gives wide intervals, so differences of a few pixels or a fraction of a millimetre are not "
        "reliable from one split. Repeat over seeds or patient-grouped folds before drawing a conclusion. The reference "
        "is an expert annotation, and the report measures agreement with it. Head-capacity differences between the "
        "heatmap head and the flattened regression head are not removed by the comparison.\n")
    (out / "report.md").write_text("\n".join(lines))
    tbl.to_csv(out / "metrics_mm.csv")
    if plaus is not None:
        plaus.to_csv(out / "metrics_mm_plausible_subset.csv")
    for m in models:
        (out / f"summary_mm_{m['name']}.json").write_text(json.dumps(model_summary(m, unit), indent=2, default=float))