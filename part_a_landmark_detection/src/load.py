"""Load evaluated runs written by scripts/evaluate.py, plus optional split metadata."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["ofd_1", "ofd_2", "bpd_1", "bpd_2"]


def load_run(eval_dir, name=None):
    """eval_dir = runs/<run>/eval_<split>. Training log and config come from the run folder."""
    eval_dir = Path(eval_dir)
    run_dir = eval_dir.parent
    per = pd.read_csv(eval_dir / "per_image.csv")
    needed = [f"{k}_{s}_{a}" for k in KEYS for s in ("pred", "gt") for a in ("x", "y")]
    missing = [c for c in needed if c not in per.columns]
    if missing:
        raise SystemExit(f"{eval_dir}/per_image.csv lacks columns {missing[:4]} ... run scripts/evaluate.py first")

    def points(prefix):
        return np.stack([np.stack([per[f"{k}_{prefix}_x"], per[f"{k}_{prefix}_y"]], axis=-1) for k in KEYS],
                        axis=1)

    log_path, cfg_path = run_dir / "log.csv", run_dir / "config.json"
    return dict(
        name=name or run_dir.name,
        names=per["image_name"].to_numpy(),
        pred=points("pred"),
        gt=points("gt"),
        log=pd.read_csv(log_path) if log_path.exists() else None,
        config=json.loads(cfg_path.read_text()) if cfg_path.exists() else {},
    )


def pixel_spacing(run, override=None):
    """mm per pixel: command-line override, else the value saved in the run's configuration."""
    if override:
        return float(override)
    cfg = run["config"]
    value = cfg.get("data", {}).get("pixel_spacing_mm") or cfg.get("args", {}).get("pixel_spacing_mm")
    return float(value) if value else None


def model_type(run):
    cfg = run["config"]
    kind = cfg.get("model", {}).get("type", "heatmap")
    return f"{kind}" + (f" ({cfg['model'].get('head')} head)" if kind == "regression" else "")


def join_splits(splits_dir):
    """Image size and patient for every image, from the split CSVs written by tools/prepare_data.py."""
    if not splits_dir:
        return None
    frames = []
    for s in ("train", "val", "test"):
        p = Path(splits_dir) / f"{s}.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=lambda c: c in ("image_name", "patient_id", "height", "width"))
            df["split"] = s
            frames.append(df)
    if not frames:
        return None
    meta = pd.concat(frames, ignore_index=True)
    meta["size"] = meta["height"].astype(str) + "x" + meta["width"].astype(str)
    return meta.drop_duplicates("image_name").set_index("image_name")
