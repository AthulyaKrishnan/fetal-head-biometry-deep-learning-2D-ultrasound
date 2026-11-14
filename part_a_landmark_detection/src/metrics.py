"""Landmark and biometry metrics. All inputs are (N, 4, 2) arrays in original pixels."""
import numpy as np

from .constants import BPD_IDX, OFD_IDX
from .postprocess import biometry_lengths


def radial_errors(pred, gt):
    return np.linalg.norm(pred - gt, axis=-1)  # (N, K)


def nme_per_measure(pred, gt):
    """Endpoint-order-invariant normalised mean error per biometry, as defined in the multicentre
    benchmark (Di Vece et al., 2026): NME = min(d_std, d_swap) / (2 |g1 - g2|), where d_std and
    d_swap are the summed endpoint errors under the standard and the swapped pairing. Returns
    per-image arrays and the fraction of images where the swapped pairing fits better (a label-order
    diagnostic: a high fraction means the errors come from endpoint order, not localisation)."""
    out = {}
    for name, (i, j) in (("ofd", OFD_IDX), ("bpd", BPD_IDX)):
        p1, p2, g1, g2 = pred[:, i], pred[:, j], gt[:, i], gt[:, j]
        d_std = np.linalg.norm(p1 - g1, axis=-1) + np.linalg.norm(p2 - g2, axis=-1)
        d_swap = np.linalg.norm(p1 - g2, axis=-1) + np.linalg.norm(p2 - g1, axis=-1)
        length = np.linalg.norm(g1 - g2, axis=-1)
        out[name] = np.minimum(d_std, d_swap) / (2 * np.maximum(length, 1e-6))
        out[f"{name}_swap_better"] = d_swap < d_std
    return out


def compute_metrics(pred, gt, pixel_spacing_mm=None,
                    px_thresholds=(5, 10, 15, 20), mm_thresholds=(2.0, 2.5, 3.0, 4.0)):
    """Returns (metrics dict, per-landmark radial error array in px)."""
    err_px = radial_errors(pred, gt)
    m = {
        "num_images": int(len(pred)),
        "mre_px": float(err_px.mean()),
        "median_px": float(np.median(err_px)),
        "std_px": float(err_px.std()),
        "mre_px_per_landmark": [round(float(v), 3) for v in err_px.mean(axis=0)],
        "success_px": {f"<= {t}px": float((err_px <= t).mean()) for t in px_thresholds},
    }

    nme = nme_per_measure(pred, gt)
    for name in ("ofd", "bpd"):
        m[f"nme_{name}_mean"] = float(nme[name].mean())
        m[f"nme_{name}_std"] = float(nme[name].std())
        m[f"{name}_swap_better_frac"] = float(nme[f"{name}_swap_better"].mean())

    ofd_p, bpd_p = biometry_lengths(pred)
    ofd_g, bpd_g = biometry_lengths(gt)
    ofd_abs = np.abs(ofd_p - ofd_g)
    bpd_abs = np.abs(bpd_p - bpd_g)
    m["ofd_abs_err_px"] = float(ofd_abs.mean())
    m["bpd_abs_err_px"] = float(bpd_abs.mean())
    m["ofd_pct_err"] = float((ofd_abs / np.maximum(ofd_g, 1e-6)).mean() * 100)
    m["bpd_pct_err"] = float((bpd_abs / np.maximum(bpd_g, 1e-6)).mean() * 100)

    if pixel_spacing_mm:
        s = float(pixel_spacing_mm)
        err_mm = err_px * s
        m["pixel_spacing_mm"] = s
        m["mre_mm"] = float(err_mm.mean())
        m["median_mm"] = float(np.median(err_mm))
        m["std_mm"] = float(err_mm.std())
        m["success_mm"] = {f"<= {t}mm": float((err_mm <= t).mean()) for t in mm_thresholds}
        m["ofd_abs_err_mm"] = float(ofd_abs.mean() * s)
        m["bpd_abs_err_mm"] = float(bpd_abs.mean() * s)
    return m, err_px
