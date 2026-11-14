import csv
import time
from pathlib import Path

import torch


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.n += n

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


class CSVLogger:

    FIELDS = ["epoch", "train_loss", "train_dice", "val_loss", "val_dice", "val_iou", "lr", "hd_weight", "seconds"]

    def __init__(self, path, resume=False):
        self.path = Path(path)
        if not (resume and self.path.exists()):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in self.FIELDS])


def read_log(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_curves(log_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_log(log_path)
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    ax[0].plot(ep, [float(r["val_loss"]) for r in rows], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [float(r["train_dice"]) for r in rows], label="train")
    ax[1].plot(ep, [float(r["val_dice"]) for r in rows], label="val")
    ax[1].set_title("Dice"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice, config, bad_epochs=0):
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "epoch": epoch, "best_dice": best_dice, "bad_epochs": bad_epochs,
                "config": config}, path)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self):
        t = time.time() - self.t0
        self.t0 = time.time()
        return t
