"""Load a trained checkpoint and segment an image at its ORIGINAL resolution."""
import numpy as np
import torch

from .model import UNet
from .preprocessing import preprocess_image, unletterbox_prob


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
    """Rebuild the model from the config stored inside the checkpoint."""
    ck = torch_load(ckpt_path, device)
    cfg = ck["config"]
    model = UNet(**cfg["model"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg["preprocess"]


@torch.no_grad()
def predict_prob(model, gray_u8, pre_cfg, device, tta=False):
    """uint8 HxW image -> float32 HxW probability map (original resolution)."""
    x, meta = preprocess_image(gray_u8, pre_cfg["image_size"], pre_cfg["clahe"], pre_cfg["norm"])
    t = torch.from_numpy(x)[None, None].to(device)
    p = torch.sigmoid(model(t))
    if tta:                                            # horizontal-flip test-time augmentation
        p = (p + torch.sigmoid(model(t.flip(-1))).flip(-1)) / 2
    return unletterbox_prob(p[0, 0].float().cpu().numpy(), meta).astype(np.float32)
