"""Keypoint-aware training augmentations (albumentations).

Geometric transforms move the landmarks together with the image. Horizontal flip keeps labels
unchanged because each biometry's two end-points remain end-points of the same line.
"""
import albumentations as A


def build_train_augmentation(cfg):
    if not cfg.get("enabled", True):
        return None
    return A.Compose(
        [
            A.HorizontalFlip(p=cfg["hflip_p"]),
            A.Affine(
                scale=(1 - cfg["scale_limit"], 1 + cfg["scale_limit"]),
                translate_percent=(-cfg["shift_limit"], cfg["shift_limit"]),
                rotate=(-cfg["rotate_limit"], cfg["rotate_limit"]),
                p=cfg["affine_p"],
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=cfg["brightness_contrast_p"]
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=cfg["gamma_p"]),
            A.OneOf(
                [A.GaussianBlur(blur_limit=(3, 5)), A.GaussNoise()],
                p=cfg["blur_noise_p"],
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
