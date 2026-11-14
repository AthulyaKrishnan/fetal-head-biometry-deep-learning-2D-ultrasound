"""Datasets for training/evaluation (with labels) and prediction (images only)."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentation import build_train_augmentation
from .heatmap import generate_heatmaps
from .io_utils import read_split_csv
from .preprocessing import preprocess_gray, to_model_input


class FetalLandmarkDataset(Dataset):
    """records: list of dicts from read_split_csv (image_path, gt (4,2) in original pixels)."""

    def __init__(self, records, input_size, stride, sigma, transform=None, preprocess_cfg=None):
        self.records = records
        self.S = input_size
        self.stride = stride
        self.sigma = sigma
        self.transform = transform
        self.pre = preprocess_cfg or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img = cv2.imread(rec["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(rec["image_path"])
        H0, W0 = img.shape
        S = self.S

        gt = rec["gt"].astype(np.float32)                                   # original px
        kp = gt * np.array([S / W0, S / H0], dtype=np.float32)              # input px

        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)
        img = preprocess_gray(img, **self.pre)

        if self.transform is not None:
            out = self.transform(image=img, keypoints=[(float(x), float(y)) for x, y in kp])
            img = out["image"]
            kp = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 2)

        heatmaps = generate_heatmaps(kp, S // self.stride, self.stride, self.sigma)
        return {
            "image": torch.from_numpy(to_model_input(img)),
            "heatmap": torch.from_numpy(heatmaps),
            "coords_norm": torch.from_numpy((kp / S).astype(np.float32)),   # target for the regression baseline
            "scale": torch.tensor([W0 / S, H0 / S], dtype=torch.float32),   # input px -> original px
            "gt_orig": torch.from_numpy(gt),
            "image_name": rec["image_name"],
        }


class ImageFolderDataset(Dataset):
    """Images only (no annotations), for prediction."""

    def __init__(self, paths, input_size, preprocess_cfg=None):
        self.paths = [Path(p) for p in paths]
        self.S = input_size
        self.pre = preprocess_cfg or {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        H0, W0 = img.shape
        S = self.S
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)
        img = preprocess_gray(img, **self.pre)
        return {
            "image": torch.from_numpy(to_model_input(img)),
            "scale": torch.tensor([W0 / S, H0 / S], dtype=torch.float32),
            "image_name": path.name,
            "image_path": str(path),
        }


def build_datasets(data_dir, cfg, max_samples=None):
    """Train/val/test datasets from the split CSVs written by tools/prepare_data.py."""
    d = cfg["data"]
    common = dict(input_size=d["input_size"], stride=d["heatmap_stride"], sigma=d["sigma"],
                  preprocess_cfg=cfg["preprocess"])
    out = {}
    for split in ("train", "val", "test"):
        records = read_split_csv(Path(data_dir) / f"{split}.csv")
        if max_samples:
            records = records[:max_samples]
        transform = build_train_augmentation(cfg["augment"]) if split == "train" else None
        out[split] = FetalLandmarkDataset(records, transform=transform, **common)
    return out


def make_loader(ds, batch_size, shuffle, num_workers=2, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)
