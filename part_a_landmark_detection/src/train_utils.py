"""Training helpers: learning-rate schedule, model selection score, checkpoints and curves."""
import math

import torch


def build_scheduler(optimizer, total_epochs, warmup_epochs):
    """Linear warmup for `warmup_epochs`, then cosine decay to zero. Stepped once per epoch."""
    def lr_lambda(step):
        if step < warmup_epochs:
            return (step + 1) / warmup_epochs
        progress = (step - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def selection_metric(metrics):
    """Lower is better. Uses mm error when a pixel spacing is known, otherwise px."""
    return metrics.get("mre_mm", metrics["mre_px"])


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, config, bad_epochs=0):
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "epoch": epoch, "best_score": best_score, "bad_epochs": bad_epochs,
                "config": config}, path)


def plot_curves(log_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    log = pd.read_csv(log_path)
    if log.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(log["epoch"], log["train_loss"], label="train")
    ax[0].plot(log["epoch"], log["val_loss"], label="val")
    ax[0].set_title("Heatmap loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(log["epoch"], log["val_mre_px"], label="val MRE [px]")
    if "val_mre_mm" in log and log["val_mre_mm"].notna().any():
        ax[1].plot(log["epoch"], log["val_mre_mm"], label="val MRE [mm]")
    ax[1].set_title("Validation landmark error"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
