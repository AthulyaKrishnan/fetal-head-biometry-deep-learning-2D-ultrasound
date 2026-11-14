"""File discovery, reading, patient-id parsing, small helpers."""
import csv
import json
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np

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


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def find_pairs(img_dir, gt_dir, gt_suffix="_Annotation"):
    """Match every image with its ground-truth file.

    GT name is tried as  <stem><gt_suffix>.<any ext>  and then  <stem>.<any ext>.
    Returns (pairs, images_without_gt, gt_without_image).
    """
    images = list_images(img_dir)
    gts = list_images(gt_dir)
    gt_by_stem = {}
    for g in gts:
        gt_by_stem.setdefault(g.stem, g)

    pairs, missing, used = [], [], set()
    for img in images:
        cand = [img.stem + gt_suffix, img.stem] if gt_suffix else [img.stem]
        hit = next((c for c in cand if c in gt_by_stem), None)
        if hit is None:
            missing.append(img)
        else:
            pairs.append((img, gt_by_stem[hit]))
            used.add(hit)
    orphans = [g for g in gts if g.stem not in used]
    return pairs, missing, orphans


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


def patient_id(stem: str) -> str:
    """'001_2HC' -> '001'. Falls back to the whole stem if no leading digits."""
    m = re.match(r"(\d+)", stem)
    return m.group(1) if m else stem


def load_pixel_table(csv_path):
    """Optional HC18-style table: filename, pixel size(mm), head circumference (mm).

    Returns {filename: {"pixel_size_mm": float, "official_hc_mm": float|None}}.
    """
    table = {}
    if not csv_path:
        return table
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        name_col = cols[0]
        pix_col = next((c for c in cols if "pixel" in c.lower()), None)
        hc_col = next((c for c in cols if "circumference" in c.lower()), None)
        for row in reader:
            if pix_col is None:
                continue
            try:
                table[row[name_col].strip()] = {
                    "pixel_size_mm": float(row[pix_col]),
                    "official_hc_mm": float(row[hc_col]) if hc_col and row[hc_col] else None,
                }
            except ValueError:
                continue
    return table


def read_split_csv(csv_path):
    """Read a split CSV written by tools/prepare_data.py; paths are made absolute."""
    root = Path(csv_path).parent
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row["image_path"] = str(root / row["image_path"])
            row["mask_path"] = str(root / row["mask_path"])
            row["height"], row["width"] = int(row["height"]), int(row["width"])
            row["pixel_size_mm"] = float(row["pixel_size_mm"]) if row["pixel_size_mm"] else None
            row["official_hc_mm"] = float(row["official_hc_mm"]) if row["official_hc_mm"] else None
            records.append(row)
    return records


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
