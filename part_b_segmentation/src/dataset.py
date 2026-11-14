"""PyTorch dataset: reads a split CSV, preprocesses once, caches in RAM."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentation import Augmenter
from .io_utils import read_gray, read_split_csv
from .preprocessing import preprocess_image, preprocess_mask


class FetalHeadDataset(Dataset):
    def __init__(self, csv_path, image_size=256, train=False, clahe=False, norm="zscore",
                 augment=True, max_samples=None):
        self.records = read_split_csv(csv_path)
        if max_samples:
            self.records = self.records[:max_samples]
        self.train = train
        self.augmenter = Augmenter() if (train and augment) else None

        self.images, self.masks, self.metas = [], [], []
        for r in self.records:                       # preprocess once, keep in RAM
            gray = read_gray(r["image_path"])
            mask = (read_gray(r["mask_path"]) > 127).astype(np.uint8)
            x, meta = preprocess_image(gray, image_size, clahe, norm)
            self.images.append(x)
            self.masks.append(preprocess_mask(mask, meta))
            self.metas.append(meta)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        img, mask = self.images[i], self.masks[i]
        if self.augmenter is not None:
            img, mask = self.augmenter(img, mask)
        return {"image": torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None],
                "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None],
                "index": i}


def make_loader(ds, batch_size, shuffle, num_workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)
