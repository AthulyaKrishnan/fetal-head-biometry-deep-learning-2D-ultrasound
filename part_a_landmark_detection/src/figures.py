"""Figures for the evaluation report. Every figure has a title, a one-line 'how to read' subtitle,
and a footnote that states the unit and the data source, so it can be read without the text."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:  # normal case: imported as part of the src package
    from .load import KEYS
    from . import mm_metrics as _M
except ImportError:  # fallback: this file was loaded directly, with src/ on sys.path
    from load import KEYS
    import mm_metrics as _M

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
LABEL = {"ofd_1": "OFD end 1", "ofd_2": "OFD end 2", "bpd_1": "BPD end 1", "bpd_2": "BPD end 2"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold"})


def _save(fig, path, title, how, note):
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.text(0.5, 0.955, title, ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.925, how, ha="center", va="top", fontsize=9, color="0.25", wrap=True)
    fig.text(0.01, 0.005, note, fontsize=7.5, color="0.4", va="bottom")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _u(unit):
    return ("mm", unit["mm_per_px"]) if unit["mm_per_px"] else ("px", 1.0)


def fig_training(models, unit, out, note):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    unit_name, f = _u(unit)
    for m, c in zip(models, COLORS):
        log = m["run"]["log"]
        if log is None:
            continue
        ax[0].plot(log["epoch"], log["train_loss"], color=c, ls="--", lw=1.2, label=f"{m['name']} train")
        ax[0].plot(log["epoch"], log["val_loss"], color=c, lw=1.8, label=f"{m['name']} validation")
        col = "val_mre_mm" if unit_name == "mm" and "val_mre_mm" in log and log["val_mre_mm"].notna().any() else "val_mre_px"
        ax[1].plot(log["epoch"], log[col], color=c, lw=1.8, label=m["name"])
        best = log[col].idxmin()
        ax[1].scatter(log.loc[best, "epoch"], log.loc[best, col], s=160, marker="*", color=c, zorder=5,
                      edgecolor="black", linewidth=0.5)
    ax[0].set_yscale("log")
    ax[0].set_title("Training and validation loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (log scale)")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].set_title(f"Validation landmark error ({unit_name})")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel(f"mean radial error [{unit_name}]")
    ax[1].legend(fontsize=8, frameon=False)
    _save(fig, out, "Training curves",
          "Left: losses are NOT comparable across model types (different loss functions). "
          "Right: the star marks the best epoch, the checkpoint that is kept. Flat curves at the end mean the run converged.",
          note)


def fig_error_distribution(models, unit, out, note):
    unit_name, f = _u(unit)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    allv = np.concatenate([m["df"]["mean_err_px"].to_numpy() for m in models]) * f
    bins = np.linspace(0, np.quantile(allv, 0.99), 40)
    for m, c in zip(models, COLORS):
        v = m["df"]["mean_err_px"].to_numpy() * f
        ax[0].hist(v, bins=bins, histtype="step", lw=2, color=c, label=f"{m['name']} (median {np.median(v):.2f})")
        ax[0].axvline(np.median(v), color=c, ls=":", lw=1)
    ax[0].set_title("Distribution of per-image mean landmark error")
    ax[0].set_xlabel(f"mean landmark error per image [{unit_name}]"); ax[0].set_ylabel("number of images")
    ax[0].legend(fontsize=8, frameon=False)
    data = [m["df"]["mean_err_px"].to_numpy() * f for m in models]
    bp = ax[1].boxplot(data, labels=[m["name"] for m in models], showfliers=True, whis=(5, 95), patch_artist=True)
    for patch, c in zip(bp["boxes"], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    for i, v in enumerate(data, 1):
        ax[1].text(i, np.mean(v), f"  mean {np.mean(v):.2f}", fontsize=8, va="center")
    ax[1].set_title("Spread per model (box = IQR, whiskers = 5 to 95 %)")
    ax[1].set_ylabel(f"[{unit_name}]")
    _save(fig, out, "Landmark error distribution",
          "A long right tail with mean well above median means a few images fail badly while most are accurate. "
          "Compare the mean with the median and read the whiskers for the tail.", note)


def fig_success_curve(models, unit, out, note):
    unit_name, f = _u(unit)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ref = 2.0 if unit_name == "mm" else 5.0
    top = max(np.quantile(m["df"][[c for c in m["df"].columns if c.startswith("err_") and c.endswith("_px")]].to_numpy().ravel(), 0.95) for m in models) * f
    xs = np.linspace(0, top * 1.05, 400)
    for m, c in zip(models, COLORS):
        e = m["lm_err"].ravel() * f
        ys = [(e <= x).mean() for x in xs]
        ax.plot(xs, ys, color=c, lw=2.2, label=m["name"])
        if unit_name == "mm":
            for thr in _M.SDR_MM:
                ax.scatter([thr], [(e <= thr).mean()], color=c, zorder=5, s=28)
            ax.annotate(f"SDR 2 mm {100*(e <= 2.0).mean():.1f} %   SDR-AUC {_M.sdr_auc(e):.3f}",
                        (2.0, (e <= 2.0).mean()), textcoords="offset points", xytext=(10, -14 - 14 * COLORS.index(c)),
                        color=c, fontsize=9)
        else:
            ax.scatter([ref], [(e <= ref).mean()], color=c, zorder=5, s=40)
    ax.axvline(ref, color="0.4", ls="--", lw=1)
    ax.set_xlabel(f"landmark error threshold [{unit_name}]")
    ax.set_ylabel("fraction of landmarks within threshold")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False)
    _save(fig, out, "Cumulative success curve",
          f"Read across: at each error threshold, the fraction of landmarks that are at least that accurate. "
          f"Higher and further left is better. Dashed line = {ref:g} {unit_name}, the clinical reference.", note)


def fig_per_landmark(models, unit, out, note):
    unit_name, f = _u(unit)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    w = 0.8 / len(models)
    x = np.arange(len(KEYS))
    for i, (m, c) in enumerate(zip(models, COLORS)):
        med = [np.median(m["df"][f"err_{k}_px"]) * f for k in KEYS]
        lo = [np.quantile(m["df"][f"err_{k}_px"], 0.25) * f for k in KEYS]
        hi = [np.quantile(m["df"][f"err_{k}_px"], 0.75) * f for k in KEYS]
        err = np.vstack([np.array(med) - np.array(lo), np.array(hi) - np.array(med)])
        ax.bar(x + (i - (len(models) - 1) / 2) * w, med, w, color=c, alpha=0.85, label=m["name"],
               yerr=err, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[k] for k in KEYS])
    ax.set_ylabel(f"median landmark error [{unit_name}] (bars: IQR)")
    ax.legend(frameon=False)
    _save(fig, out, "Error per landmark",
          "Bars are medians, whiskers the interquartile range. A consistently higher error at one end of a line "
          "points to that endpoint's annotation or appearance, not to the whole measurement.", note)


def fig_bland_altman(models, unit, out, note):
    unit_name, f = _u(unit)
    fig, axes = plt.subplots(2, len(models), figsize=(4.6 * len(models), 8), squeeze=False)
    for col, (m, c) in enumerate(zip(models, COLORS)):
        for row, name in enumerate(("ofd", "bpd")):
            ax = axes[row, col]
            d = m["df"]
            mean_len = (d[f"{name}_len_pred_px"] + d[f"{name}_len_gt_px"]).to_numpy() / 2 * f
            diff = d[f"{name}_len_bias_px"].to_numpy() * f
            bias, sd = diff.mean(), diff.std(ddof=1)
            ax.scatter(mean_len, diff, s=14, alpha=0.6, color=c)
            ax.axhline(bias, color="black", lw=1.5)
            ax.axhline(bias + 1.96 * sd, color="0.3", ls="--", lw=1)
            ax.axhline(bias - 1.96 * sd, color="0.3", ls="--", lw=1)
            ax.axhline(0, color="0.8", lw=0.8)
            ax.set_title(f"{m['name']}: {name.upper()}", fontsize=10)
            icc = _M.icc_2_1(d[f"{name}_len_pred_px"].to_numpy() * f, d[f"{name}_len_gt_px"].to_numpy() * f)
            ax.text(0.02, 0.97, f"bias {bias:+.2f} {unit_name}\nLoA [{bias-1.96*sd:+.2f}, {bias+1.96*sd:+.2f}] {unit_name}\nICC(2,1) {icc:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.8"))
            ax.set_xlabel(f"mean of predicted and true length [{unit_name}]")
            ax.set_ylabel(f"predicted minus true [{unit_name}]")
    _save(fig, out, "Biometry agreement (Bland-Altman)",
          "Each dot is one image. The solid line is the bias (systematic over- or under-measurement); dashed lines "
          "are the 95 % limits of agreement. A useful model has a bias near zero and narrow limits.", note)


def fig_length_error(models, unit, out, note):
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(models)
    for i, (m, c) in enumerate(zip(models, COLORS)):
        for j, name in enumerate(("ofd", "bpd")):
            v = m["df"][f"{name}_len_pct_err"].to_numpy()
            pos = j + (i - (len(models) - 1) / 2) * width
            bp = ax.boxplot([v], positions=[pos], widths=width * 0.85, patch_artist=True, whis=(5, 95),
                            showfliers=False)
            bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.35)
            bp["medians"][0].set_color("black")
            ax.text(pos, np.quantile(v, 0.95) + 0.3, f"{100*(v <= 5).mean():.0f}% ≤5%", ha="center", fontsize=7.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["OFD length", "BPD length"])
    ax.set_ylabel("absolute length error [% of true length]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for c in COLORS[:len(models)]]
    ax.legend(handles, [m["name"] for m in models], frameon=False)
    _save(fig, out, "Length error as a percentage of the true length",
          "Box = IQR, black line = median, whiskers = 5 to 95 %. Labels give the share of images within 5 %. "
          "Lengths are what clinicians use for gestational age, so this is the most relevant panel.", note)


def fig_error_decomposition(models, unit, out, note):
    unit_name, f = _u(unit)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.8 / len(models)
    x = np.arange(4)
    names = ["OFD along", "OFD across", "BPD along", "BPD across"]
    for i, (m, c) in enumerate(zip(models, COLORS)):
        vals = [m["df"]["ofd_along_px"].mean() * f, m["df"]["ofd_across_px"].mean() * f,
                m["df"]["bpd_along_px"].mean() * f, m["df"]["bpd_across_px"].mean() * f]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, color=c, alpha=0.85, label=m["name"])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel(f"mean absolute error [{unit_name}]")
    ax.legend(frameon=False)
    _save(fig, out, "Error along versus across the measurement line",
          "'Along' errors move an endpoint along the line and change the length. 'Across' errors move it sideways "
          "and change the length little. If most error is across the line, length accuracy is better than the "
          "landmark error suggests.", note)


def fig_error_vs_size(models, unit, out, note):
    unit_name, f = _u(unit)
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 4.6), squeeze=False, sharey=True)
    for ax, m, c in zip(axes[0], models, COLORS):
        d = m["df"]
        size = d["ofd_len_gt_px"].to_numpy() * f
        rel = d["mean_err_pct_of_ofd"].to_numpy()
        ax.scatter(size, rel, s=14, alpha=0.6, color=c)
        edges = np.quantile(size, np.linspace(0, 1, 7))
        centers, medians = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (size >= lo) & (size <= hi)
            if sel.any():
                centers.append(size[sel].mean()); medians.append(np.median(rel[sel]))
        ax.plot(centers, medians, color="black", lw=2)
        ax.set_title(m["name"], fontsize=10)
        ax.set_xlabel(f"true OFD length [{unit_name}]")
    axes[0][0].set_ylabel("mean landmark error [% of OFD]")
    _save(fig, out, "Does error depend on head size?",
          "Each dot is an image. The black line is the median in size groups. A rising line means small heads are "
          "relatively harder; a flat line means the error scales with the head.", note)


def fig_swap_diagnostic(models, unit, out, note):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(models))
    w = 0.38
    ofd = [100 * m["df"]["ofd_swap_better"].mean() for m in models]
    bpd = [100 * m["df"]["bpd_swap_better"].mean() for m in models]
    ax[0].bar(x - w / 2, ofd, w, label="OFD", color="#1f77b4")
    ax[0].bar(x + w / 2, bpd, w, label="BPD", color="#ff7f0e")
    for i, (a, b) in enumerate(zip(ofd, bpd)):
        ax[0].text(i - w / 2, a + 1, f"{a:.0f}%", ha="center", fontsize=8)
        ax[0].text(i + w / 2, b + 1, f"{b:.0f}%", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([m["name"] for m in models])
    ax[0].set_ylabel("% of images where swapped endpoints fit better")
    ax[0].set_title("Label-order diagnostic"); ax[0].legend(frameon=False)
    data, labels = [], []
    for m in models:
        for name in ("ofd", "bpd"):
            data.append(m["df"][f"{name}_nme"].to_numpy()); labels.append(f"{m['name']}\n{name.upper()}")
    bp = ax[1].boxplot(data, labels=labels, whis=(5, 95), showfliers=False, patch_artist=True)
    for k, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[(k // 2) % len(COLORS)]); patch.set_alpha(0.35)
    ax[1].set_ylabel("NME (order-invariant, lower is better)")
    ax[1].set_title("Endpoint-order-invariant error (Di Vece et al., 2026)")
    _save(fig, out, "Endpoint order diagnostic",
          "Left: a high share means the endpoints are often placed in the wrong order, a labelling-convention "
          "issue. Right: NME ignores order, so it measures localisation alone. If NME is much lower than the "
          "landmark error, order is responsible for part of the error.", note)


def fig_strata(models, unit, out, note):
    unit_name, f = _u(unit)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    pooled = np.concatenate([m["df"]["ofd_len_gt_px"].to_numpy() for m in models])
    edges = np.quantile(pooled, [0, 1 / 3, 2 / 3, 1])
    names = ["small", "medium", "large"]
    for i, (m, c) in enumerate(zip(models, COLORS)):
        d = m["df"]
        groups = []
        for lo, hi, nm in zip(edges[:-1], edges[1:], names):
            sel = (d["ofd_len_gt_px"] >= lo) & (d["ofd_len_gt_px"] <= hi)
            groups.append(d.loc[sel, "mean_err_px"].to_numpy() * f)
        pos = np.arange(3) + (i - (len(models) - 1) / 2) * 0.8 / len(models)
        bp = ax[0].boxplot(groups, positions=pos, widths=0.8 / len(models) * 0.85, patch_artist=True,
                           whis=(5, 95), showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(c); patch.set_alpha(0.35)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["small heads", "medium", "large heads"])
    ax[0].set_ylabel(f"mean landmark error [{unit_name}]")
    ax[0].set_title("Error by head size (tertiles of true OFD)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for c in COLORS[:len(models)]]
    ax[0].legend(handles, [m["name"] for m in models], frameon=False)
    if "size" in models[0]["df"].columns and models[0]["df"]["size"].notna().any():
        sizes = sorted(models[0]["df"]["size"].dropna().unique())
        for i, (m, c) in enumerate(zip(models, COLORS)):
            groups = [m["df"].loc[m["df"]["size"] == s, "mean_err_px"].to_numpy() * f for s in sizes]
            pos = np.arange(len(sizes)) + (i - (len(models) - 1) / 2) * 0.8 / len(models)
            bp = ax[1].boxplot(groups, positions=pos, widths=0.8 / len(models) * 0.85, patch_artist=True,
                               whis=(5, 95), showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(c); patch.set_alpha(0.35)
        ax[1].set_xticks(range(len(sizes))); ax[1].set_xticklabels(sizes, rotation=15)
        ax[1].set_title("Error by image size (from the split files)")
        ax[1].set_ylabel(f"mean landmark error [{unit_name}]")
    else:
        ax[1].text(0.5, 0.5, "image size not available\n(pass --splits_dir data_processed)",
                   ha="center", va="center", transform=ax[1].transAxes)
        ax[1].set_axis_off()
    _save(fig, out, "Error by subgroup",
          "Look for subgroups where a model is clearly worse, for example small heads or one image size. "
          "Overlapping boxes mean no evidence of a difference in that subgroup.", note)


def fig_paired(models, unit, out, note, comparison):
    unit_name, f = _u(unit)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    a, b = models[0], models[1]
    x = comparison["a_err"] * f
    y = comparison["b_err"] * f
    lo_v = max(min(x.min(), y.min()), 1e-2)
    top = max(np.quantile(x, 0.995), np.quantile(y, 0.995)) * 1.2
    ax[0].scatter(np.maximum(x, lo_v), np.maximum(y, lo_v), s=18, alpha=0.6, color="0.2")
    ax[0].plot([lo_v, top], [lo_v, top], color="0.6", ls="--")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(lo_v, top); ax[0].set_ylim(lo_v, top)
    ax[0].set_xlabel(f"{a['name']} mean landmark error [{unit_name}]")
    ax[0].set_ylabel(f"{b['name']} mean landmark error [{unit_name}]")
    ax[0].set_title("Same image, two models (log scale)")
    ax[0].text(0.03, 0.95, f"points below the dashed line:\n{b['name']} better on "
               f"{100*(y < x).mean():.0f}% of images", transform=ax[0].transAxes, va="top", fontsize=8.5,
               bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    rows = comparison["rows"]
    ypos = np.arange(len(rows))[::-1]
    for yi, r in zip(ypos, rows):
        d, lo, hi = r["diff"], r["lo"], r["hi"]          # already in the report unit
        color = "#2ca02c" if hi < 0 else "#d62728" if lo > 0 else "0.5"
        ax[1].plot([lo, hi], [yi, yi], color=color, lw=2.5)
        ax[1].scatter([d], [yi], color=color, s=60, zorder=5)
        ax[1].text(hi, yi, f"  p(Holm)={r['p_holm']:.3f}", va="center", fontsize=8)
    ax[1].axvline(0, color="black", lw=1)
    ax[1].set_yticks(ypos); ax[1].set_yticklabels([r["label"] for r in rows])
    ax[1].set_xlabel(f"{a['name']} minus {b['name']} [{unit_name}]  (negative favours {a['name']})")
    ax[1].set_title("Paired differences with 95 % bootstrap CI")
    _save(fig, out, "Head-to-head comparison on the same test images",
          "Left: one dot per image; dots below the dashed line are better for the second model. Right: a bar that "
          "stays on one side of zero is evidence of a difference (green favours the first model, red the second). "
          "Holm-corrected p-values account for testing several measures.", note)


def fig_concentration(models, unit, out, note):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for m, c in zip(models, COLORS):
        e = np.sort(m["df"]["mean_err_px"].to_numpy())[::-1]
        share = np.cumsum(e) / e.sum()
        frac = np.arange(1, len(e) + 1) / len(e)
        ax.plot(frac * 100, share * 100, color=c, lw=2.2, label=m["name"])
        k = max(1, int(round(0.1 * len(e))))
        ax.annotate(f"worst 10% = {100*share[k-1]:.0f}% of total error", (frac[k-1] * 100, share[k-1] * 100),
                    textcoords="offset points", xytext=(8, -12), color=c, fontsize=8.5)
    ax.plot([0, 100], [0, 100], color="0.6", ls="--", label="equal share")
    ax.set_xlabel("worst images, % of the test set"); ax.set_ylabel("% of total error they account for")
    ax.legend(frameon=False)
    _save(fig, out, "Error concentration",
          "Curves above the dashed line mean a few images carry most of the error. A steep early rise points to "
          "catastrophic failures to inspect in overlays_worst/ rather than general inaccuracy.", note)