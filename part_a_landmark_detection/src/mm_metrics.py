"""Millimetre metrics for landmark evaluation. The definitions here are the same as in the judges'
tester (scripts/test.py), so a number means the same thing in the tester summary and in the report.

Inputs are pixel coordinates (N, 4, 2) and the pixel spacing f (mm per pixel). Every length and error
is converted to mm; NME is dimensionless and computed from pixels (the ratio does not depend on units).

Sources: MRE and SDR at 2, 2.5, 3, 4 mm, Chen et al. (MICCAI 2019); order-invariant NME and Bland-Altman
agreement, Di Vece et al. (2026); HC = 1.62 x (BPD + OFD), ISUOG practice guideline (2022);
ICC(2,1) absolute agreement, McGraw and Wong (1996); SDR-AUC is common practice in landmark challenges.
"""
import numpy as np

SDR_MM = (2.0, 2.5, 3.0, 4.0)
AUC_MAX_MM = 10.0
KEYS = ["ofd_1", "ofd_2", "bpd_1", "bpd_2"]


def bootstrap_ci(x, stat=np.mean, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    v = stat(x[idx], axis=1)
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def icc_2_1(a, b):
    """ICC(2,1): two-way random effects, single measurement, absolute agreement."""
    X = np.stack([np.asarray(a, float), np.asarray(b, float)], axis=1)
    n, k = X.shape
    if n < 3:
        return float("nan")
    gm = X.mean()
    row, col = X.mean(axis=1), X.mean(axis=0)
    ssr = k * np.sum((row - gm) ** 2)
    ssc = n * np.sum((col - gm) ** 2)
    sst = np.sum((X - gm) ** 2)
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom > 0 else float("nan")


def agreement(pred_mm, true_mm):
    diff = np.asarray(pred_mm, float) - np.asarray(true_mm, float)
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    pct = 100 * np.abs(diff) / np.maximum(np.asarray(true_mm, float), 1e-6)
    return dict(abs_error_mean=float(np.abs(diff).mean()), abs_error_median=float(np.median(np.abs(diff))),
                abs_error_pct_median=float(np.median(pct)), bias=float(diff.mean()),
                limits_of_agreement=[float(diff.mean() - 1.96 * sd), float(diff.mean() + 1.96 * sd)],
                within_5pct=float((pct <= 5).mean()), icc_2_1=icc_2_1(pred_mm, true_mm))


def sdr_auc(err_mm):
    t = np.linspace(0, AUC_MAX_MM, 401)
    sdr = np.array([(err_mm <= x).mean() for x in t])
    return float(np.sum((sdr[1:] + sdr[:-1]) / 2 * np.diff(t)) / AUC_MAX_MM)


def summarize(pred, gt, f):
    """All numbers of the report for one model or one subset. f = mm per pixel."""
    err_mm = np.linalg.norm(pred - gt, axis=-1) * f
    per_img = err_mm.mean(axis=1)
    med = float(np.median(per_img))
    s = dict(n_images=int(len(pred)))
    s["landmark_error_mm"] = dict(
        mean=float(per_img.mean()), mean_ci95=bootstrap_ci(per_img),
        median=med, median_ci95=bootstrap_ci(per_img, np.median),
        iqr=[float(np.quantile(per_img, .25)), float(np.quantile(per_img, .75))],
        p90=float(np.quantile(per_img, .9)), max=float(per_img.max()))
    s["per_landmark_mm"] = {n: dict(mean=float(err_mm[:, k].mean()), median=float(np.median(err_mm[:, k])),
                                    sdr_2mm=float((err_mm[:, k] <= 2.0).mean()))
                            for k, n in enumerate(KEYS)}
    s["sdr"] = {f"<= {t} mm": float((err_mm.ravel() <= t).mean()) for t in SDR_MM}
    s["sdr_auc_0_10mm"] = sdr_auc(err_mm.ravel())
    k = max(1, int(round(0.1 * len(per_img))))
    s["failures"] = dict(catastrophic_share=float((per_img > 3 * med).mean()),
                         worst10_share_of_error=float(np.sort(per_img)[::-1][:k].sum() / per_img.sum()))
    lengths = {}
    for name, (a, b) in (("OFD", (0, 1)), ("BPD", (2, 3))):
        pa, pb, ga, gb = pred[:, a], pred[:, b], gt[:, a], gt[:, b]
        lp = np.linalg.norm(pb - pa, axis=-1) * f
        lg = np.linalg.norm(gb - ga, axis=-1) * f
        lengths[name] = (lp, lg)
        dstd = np.linalg.norm(pa - ga, axis=-1) + np.linalg.norm(pb - gb, axis=-1)
        dswap = np.linalg.norm(pa - gb, axis=-1) + np.linalg.norm(pb - ga, axis=-1)
        nme = np.minimum(dstd, dswap) / (2 * np.maximum(np.linalg.norm(gb - ga, axis=-1), 1e-6))
        s[name] = dict(agreement(lp, lg), nme_mean=float(nme.mean()),
                       endpoints_swapped_better=float((dswap < dstd).mean()))
    hc_p = 1.62 * (lengths["OFD"][0] + lengths["BPD"][0])
    hc_g = 1.62 * (lengths["OFD"][1] + lengths["BPD"][1])
    s["HC"] = agreement(hc_p, hc_g)
    return s


def _m(x):
    return f"{x:.2f}"


def rows(s):
    """Ordered (label, text) pairs for one model. Labels are shared by the table and the summary text."""
    e = s["landmark_error_mm"]
    out = [
        ("images", f"{s['n_images']}"),
        ("landmark error, mean per image (mm)", f"{e['mean']:.3f}  [95% CI {e['mean_ci95'][0]:.3f}, {e['mean_ci95'][1]:.3f}]"),
        ("landmark error, median per image (mm)", f"{e['median']:.3f}  [95% CI {e['median_ci95'][0]:.3f}, {e['median_ci95'][1]:.3f}]"),
        ("landmark error, IQR (mm)", f"{e['iqr'][0]:.3f} to {e['iqr'][1]:.3f}"),
        ("landmark error, 90th percentile (mm)", f"{e['p90']:.3f}"),
        ("landmark error, max (mm)", f"{e['max']:.3f}"),
    ]
    for t in SDR_MM:
        out.append((f"SDR within {t:g} mm (share of landmarks)", f"{100*s['sdr'][f'<= {t} mm']:.1f} %"))
    out.append(("SDR-AUC, 0 to 10 mm (1 = perfect)", f"{s['sdr_auc_0_10mm']:.3f}"))
    for n in KEYS:
        out.append((f"{n}, mean error (mm)", f"{s['per_landmark_mm'][n]['mean']:.3f}"))
    for name in ("OFD", "BPD"):
        d = s[name]
        out += [
            (f"{name} length, mean absolute error (mm)", _m(d["abs_error_mean"])),
            (f"{name} length, median absolute error (% of true)", f"{d['abs_error_pct_median']:.2f} %"),
            (f"{name} length, bias (mm, + = overestimate)", f"{d['bias']:+.2f}"),
            (f"{name} length, 95% limits of agreement (mm)", f"[{d['limits_of_agreement'][0]:+.2f}, {d['limits_of_agreement'][1]:+.2f}]"),
            (f"{name} length, within 5 % of true", f"{100*d['within_5pct']:.1f} %"),
            (f"{name} length, ICC(2,1)", f"{d['icc_2_1']:.3f}"),
            (f"{name}, NME (order-invariant, dimensionless)", f"{d['nme_mean']:.4f}"),
            (f"{name}, endpoints swapped better", f"{100*d['endpoints_swapped_better']:.1f} %"),
        ]
    h = s["HC"]
    out += [
        ("HC = 1.62 x (BPD + OFD), mean absolute error (mm)", _m(h["abs_error_mean"])),
        ("HC, median absolute error (% of true)", f"{h['abs_error_pct_median']:.2f} %"),
        ("HC, within 5 % of true", f"{100*h['within_5pct']:.1f} %"),
        ("HC, ICC(2,1)", f"{h['icc_2_1']:.3f}"),
        ("images with error above 3 x median", f"{100*s['failures']['catastrophic_share']:.1f} %"),
        ("share of total error in worst 10 % of images", f"{100*s['failures']['worst10_share_of_error']:.1f} %"),
    ]
    return out


def summary_text(s, title, meta_lines=()):
    line = "=" * 78
    L = [line, title, line]
    L += list(meta_lines)
    L.append("")
    for label, text in rows(s):
        L.append(f"  {label:<52} {text}")
    L += [line]
    return "\n".join(L)