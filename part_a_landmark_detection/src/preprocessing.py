"""Image preprocessing shared by training, evaluation, prediction and the dataset check.

The image is resized (stretched) to a square network input. Landmark coordinates are scaled
with the same factors, and predictions are mapped back to original pixels by the inverse scale.
"""
import cv2
import numpy as np

# Approximate ImageNet statistics for grayscale ultrasound.
IMG_MEAN = 0.449
IMG_STD = 0.226


def preprocess_gray(gray, use_clahe=True, clip_limit=2.0, tile_grid=8):
    """Operates on a uint8 grayscale image. CLAHE enhances local contrast."""
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        gray = clahe.apply(gray)
    return gray


def to_model_input(gray_uint8):
    """uint8 (H, W) -> normalised float32 (3, H, W) for ImageNet-pretrained backbones."""
    x = gray_uint8.astype(np.float32) / 255.0
    x = (x - IMG_MEAN) / IMG_STD
    return np.repeat(x[None], 3, axis=0)
