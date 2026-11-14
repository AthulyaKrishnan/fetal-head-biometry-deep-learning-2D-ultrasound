"""Checkpoint loading and batched inference (network output -> landmarks in original pixels).

Works for both models: a heatmap model is decoded by argmax, a coordinate-regression model's
outputs are already the coordinates. Both are then mapped back to original pixels with the scale
factors of each image."""
import numpy as np
import torch

from .heatmap import decode_heatmaps
from .model import build_model


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def torch_load(path, device):
    """torch.load that works across versions (newer ones default to weights_only=True)."""
    try:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def load_model(ckpt_path, device):
    """Rebuild the model from the configuration stored inside the checkpoint."""
    ck = torch_load(ckpt_path, device)
    cfg = ck["config"]
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg


def get_target(model, batch):
    """The training target that matches the model's output: heatmaps or normalised coordinates."""
    key = "coords_norm" if getattr(model, "output_type", "heatmap") == "coords" else "heatmap"
    return batch[key]


@torch.no_grad()
def run_inference(model, loader, device, stride=None, criterion=None, amp=False):
    """Returns names, predicted landmarks (N, 4, 2) in original pixels, ground truth if present,
    and the mean loss if a criterion is given."""
    model.eval()
    coords = getattr(model, "output_type", "heatmap") == "coords"
    names, preds, gts = [], [], []
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(img)
        out = out.float()
        if criterion is not None:
            loss_sum += criterion(out, get_target(model, batch).to(device)).item()
            n_batches += 1
        if coords:
            pts_input = out.cpu().numpy() * model.input_size
        else:
            pts_input = decode_heatmaps(out.cpu().numpy(), model.stride)
        scale = batch["scale"].numpy()
        preds.append(pts_input * scale[:, None, :])
        if "gt_orig" in batch:
            gts.append(batch["gt_orig"].numpy())
        names.extend(list(batch["image_name"]))
    return {
        "names": names,
        "pred": np.concatenate(preds),
        "gt": np.concatenate(gts) if gts else None,
        "loss": loss_sum / n_batches if n_batches else None,
    }
