"""File discovery, reading, CSV helpers, seeding and small utilities."""
import csv
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .constants import COORD_COLUMNS, KEYPOINT_NAMES

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


def patient_id(stem: str) -> str:
    """'001_HC' -> '001'. Falls back to the whole stem if there are no leading digits."""
    m = re.match(r"(\d+)", stem)
    return m.group(1) if m else stem


def load_annotations(path):
    """Load one CSV file, or every *.csv in a folder, into one frame with image_name + 8 coordinates."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV annotation files found at {p}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    required = ["image_name"] + COORD_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation file(s) missing columns: {missing}")
    df["image_name"] = df["image_name"].astype(str).str.strip()
    return df[required].reset_index(drop=True)


def point_length(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


# Thresholds for a landmark set that is NOT on the fetal head in an axial image.
# Shared by tools/check_dataset.py and tools/prepare_data.py so that the check and the removal agree.
GEOMETRY_THRESHOLDS = dict(max_perp_dev=30.0,        # OFD and BPD should be roughly perpendicular (deg)
                           max_midpoint_gap=150.0,   # the two line midpoints should nearly coincide (px)
                           min_len_px=100.0)         # a fetal OFD or BPD is never this short (px)


def geometry_measures(xy):
    """OFD/BPD lengths, OFD-BPD angle deviation from 90 degrees, and midpoint gap, for xy (4, 2)."""
    ofd_v, bpd_v = xy[1] - xy[0], xy[3] - xy[2]
    ofd_len, bpd_len = float(np.linalg.norm(ofd_v)), float(np.linalg.norm(bpd_v))
    cos = abs(float(np.dot(ofd_v, bpd_v))) / (ofd_len * bpd_len + 1e-9)
    acute = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
    return dict(ofd_px=ofd_len, bpd_px=bpd_len, perp_dev_deg=abs(90.0 - acute),
                midpoint_gap_px=float(np.linalg.norm(xy[:2].mean(0) - xy[2:].mean(0))))


def geometry_problems(xy, w, h, max_perp_dev=30.0, max_midpoint_gap=150.0, min_len_px=100.0):
    """Reasons why the landmarks are implausible for a fetal head. Returns (type, detail) pairs."""
    out = []
    if ((xy[:, 0] < 0) | (xy[:, 0] >= w) | (xy[:, 1] < 0) | (xy[:, 1] >= h)).any():
        out.append(("out_of_bounds", f"points={xy.round(1).tolist()} image={w}x{h}"))
    m = geometry_measures(xy)
    if m["ofd_px"] < min_len_px:
        out.append(("short_ofd", f"OFD {m['ofd_px']:.1f}px < {min_len_px:.0f}px"))
    if m["bpd_px"] < min_len_px:
        out.append(("short_bpd", f"BPD {m['bpd_px']:.1f}px < {min_len_px:.0f}px"))
    if m["perp_dev_deg"] > max_perp_dev:
        out.append(("not_perpendicular", f"OFD/BPD deviate {m['perp_dev_deg']:.1f} deg from 90"))
    if m["midpoint_gap_px"] > max_midpoint_gap:
        out.append(("midpoint_gap", f"line midpoints {m['midpoint_gap_px']:.1f}px apart"))
    return out


def row_problems(xy, w, h, **thresholds):
    """All error-level problems of one annotation row (xy: (4, 2), w and h: image size).
    An empty list means the row is usable."""
    if not np.all(np.isfinite(xy)):
        return [("nonnumeric_coords", "NaN or non-numeric coordinate")]
    return geometry_problems(xy, w, h, **thresholds)


def read_split_csv(csv_path):
    """Read a split CSV written by tools/prepare_data.py. Paths become absolute and the four
    landmarks become a float array 'gt' of shape (4, 2) in ORIGINAL image pixels."""
    root = Path(csv_path).parent
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rec = dict(stem=row["stem"], patient_id=row["patient_id"], image_name=row["image_name"],
                       image_path=str(root / row["image_path"]),
                       height=int(row["height"]), width=int(row["width"]))
            rec["gt"] = np.array([[float(row[f"{n}_x"]), float(row[f"{n}_y"])] for n in KEYPOINT_NAMES],
                                 dtype=np.float32)
            records.append(rec)
    return records


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def line_angle_deg(u, v):
    """Angle between two lines (0..90 degrees), so direction signs do not matter."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    cos = abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def ellipse_axes_problems(xy, max_center_gap=0.15, min_line_angle=60.0):
    """Checks that the 4 landmarks form the OFD and BPD axes of one head ellipse.

    The two axes of an ellipse share a centre and are perpendicular. A landmark placed off the
    head breaks one of these properties, so such rows are flagged:
      centre_mismatch         midpoints of the OFD and BPD lines are further apart than
                              max_center_gap x OFD length
      axes_not_perpendicular  the two lines meet at less than min_line_angle degrees
    """
    if not np.all(np.isfinite(xy)):
        return []
    o1, o2, b1, b2 = xy
    ofd = point_length(o1, o2)
    out = []
    gap = point_length((o1 + o2) / 2, (b1 + b2) / 2) / max(ofd, 1e-6)
    if gap > max_center_gap:
        out.append(("centre_mismatch",
                    f"OFD and BPD midpoints {gap:.2f} OFD-lengths apart (limit {max_center_gap})"))
    angle = line_angle_deg(o2 - o1, b2 - b1)
    if angle < min_line_angle:
        out.append(("axes_not_perpendicular", f"OFD and BPD lines meet at {angle:.0f} deg (min {min_line_angle})"))
    return out


def robust_outlier_mask(values, k=4.0):
    """True where a value is more than k robust standard deviations (median / MAD) from the median."""
    v = np.asarray(values, float)
    if v.size < 3:
        return np.zeros(v.size, bool)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    if mad == 0:
        return np.zeros(v.size, bool)
    return np.abs(v - med) > k * mad


def load_exclude_list(path):
    """Image names to remove. Accepts a CSV with an image_name column (or its first column), or a
    text file with one name per line; '#' starts a comment. Stems ('001_HC') also match."""
    if not path:
        return set()
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = "image_name" if "image_name" in df.columns else df.columns[0]
        names = df[col].astype(str).str.strip().tolist()
    else:
        names = [ln.strip() for ln in p.read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
    keys = set()
    for n in names:
        keys.add(n)
        keys.add(Path(n).stem)
    return keys
