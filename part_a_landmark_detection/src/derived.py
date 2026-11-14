"""Per-image quantities derived from the predicted and true landmarks. Everything is in pixels;
the reporting layer converts to millimetres with the pixel spacing."""
import numpy as np
import pandas as pd

from load import KEYS


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def derive(run, meta=None):
    P, G = run["pred"], run["gt"]
    d = pd.DataFrame({"image_name": run["names"]})

    err = np.linalg.norm(P - G, axis=-1)                       # (N, 4) landmark radial errors
    for k, name in enumerate(KEYS):
        d[f"err_{name}_px"] = err[:, k]
    d["mean_err_px"] = err.mean(axis=1)
    d["median_lm_err_px"] = np.median(err, axis=1)

    for name, (a, b) in (("ofd", (0, 1)), ("bpd", (2, 3))):
        pa, pb, ga, gb = P[:, a], P[:, b], G[:, a], G[:, b]
        lp = np.linalg.norm(pb - pa, axis=-1)
        lg = np.linalg.norm(gb - ga, axis=-1)
        d[f"{name}_len_gt_px"] = lg
        d[f"{name}_len_pred_px"] = lp
        d[f"{name}_len_bias_px"] = lp - lg                       # signed: + means overestimate
        d[f"{name}_len_abs_err_px"] = np.abs(lp - lg)
        d[f"{name}_len_pct_err"] = 100 * np.abs(lp - lg) / np.maximum(lg, 1e-6)
        dstd = np.linalg.norm(pa - ga, axis=-1) + np.linalg.norm(pb - gb, axis=-1)
        dswap = np.linalg.norm(pa - gb, axis=-1) + np.linalg.norm(pb - ga, axis=-1)
        d[f"{name}_nme"] = np.minimum(dstd, dswap) / (2 * np.maximum(lg, 1e-6))   # benchmark NME
        d[f"{name}_swap_better"] = dswap < dstd
        # error components relative to the TRUE measurement line: along it (changes the length)
        # or across it (barely changes the length)
        u = _unit(gb - ga)
        # use the pairing that fits better, so a swapped labelling does not appear as a large error
        swapped = (dswap < dstd)[:, None]
        pa_use, pb_use = np.where(swapped, pb, pa), np.where(swapped, pa, pb)
        ea, eb = pa_use - ga, pb_use - gb
        along = (np.abs(np.sum(ea * u, -1)) + np.abs(np.sum(eb * u, -1))) / 2
        across = (np.abs(ea[:, 0] * u[:, 1] - ea[:, 1] * u[:, 0])
                  + np.abs(eb[:, 0] * u[:, 1] - eb[:, 1] * u[:, 0])) / 2
        d[f"{name}_along_px"] = along
        d[f"{name}_across_px"] = across

    # head circumference from the two lengths, HC = 1.62 x (BPD + OFD) (ISUOG 2022 practice guideline)
    hc_g = 1.62 * (d["ofd_len_gt_px"] + d["bpd_len_gt_px"])
    hc_p = 1.62 * (d["ofd_len_pred_px"] + d["bpd_len_pred_px"])
    d["hc_gt_px"], d["hc_pred_px"] = hc_g, hc_p
    d["hc_abs_err_px"] = np.abs(hc_p - hc_g)
    d["hc_pct_err"] = 100 * np.abs(hc_p - hc_g) / np.maximum(hc_g, 1e-6)

    d["mean_err_pct_of_ofd"] = 100 * d["mean_err_px"] / np.maximum(d["ofd_len_gt_px"], 1e-6)
    median = d["mean_err_px"].median()
    d["catastrophic"] = d["mean_err_px"] > 3 * median           # more than 3x the median image error

    if meta is not None:
        d = d.merge(meta, left_on="image_name", right_index=True, how="left")
    return d
