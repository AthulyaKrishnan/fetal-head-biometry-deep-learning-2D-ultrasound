"""Default settings. The command-line flags of scripts/train.py override these values, and the full
configuration is stored inside every checkpoint so that evaluation uses exactly the same settings."""
import copy

DEFAULT_CONFIG = {
    "seed": 42,
    "data": {
        "input_size": 384,          # network input is input_size x input_size
        "heatmap_stride": 4,        # heatmap resolution = input_size / heatmap_stride
        "sigma": 2.0,               # Gaussian sigma in heatmap pixels
        "pixel_spacing_mm": None,   # mm per image pixel; set to also report mm metrics
    },
    "preprocess": {"use_clahe": True, "clip_limit": 2.0, "tile_grid": 8},
    "augment": {
        "enabled": True,            # training split only
        "hflip_p": 0.5,
        "affine_p": 0.7,
        "scale_limit": 0.1,
        "shift_limit": 0.05,
        "rotate_limit": 15,
        "brightness_contrast_p": 0.5,
        "gamma_p": 0.3,
        "blur_noise_p": 0.3,
    },
    "model": {
        "backbone": "hrnet_w32",    # any timm HRNet name, e.g. hrnet_w18, hrnet_w48
        "pretrained": True,
        "num_keypoints": 4,
        "head_channels": 256,
        "type": "heatmap",          # "heatmap" (proposed) or "regression" (direct coordinate baseline)
        "head": "flatten",          # regression only: "flatten" keeps spatial layout, "gap" pools it away
    },
    "train": {
        "epochs": 150,
        "batch_size": 16,
        "num_workers": 2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "grad_clip": 5.0,
        "amp": False,
        "pos_weight": 0.0,          # >0 up-weights heatmap peaks in the MSE loss
        "patience": 30,             # early stopping on validation error (0 = off)
    },
}


def default_config():
    return copy.deepcopy(DEFAULT_CONFIG)
