"""Light, ultrasound-appropriate augmentation using only numpy + OpenCV.

Applied to the already letterboxed, already normalised (z-score) image so that
the image and mask always stay perfectly aligned. No vertical flip: ultrasound
probe orientation is fixed (top = probe), so only horizontal flips are realistic.
"""
import random

import cv2
import numpy as np


class Augmenter:
    def __init__(self, p_flip=0.5, p_affine=0.8, max_rot=15.0, scale=(0.85, 1.15),
                 max_shift=0.08, p_intensity=0.7, p_blur=0.2, p_noise=0.3):
        self.p_flip, self.p_affine = p_flip, p_affine
        self.max_rot, self.scale, self.max_shift = max_rot, scale, max_shift
        self.p_intensity, self.p_blur, self.p_noise = p_intensity, p_blur, p_noise

    def __call__(self, img, mask):
        s = img.shape[0]
        if random.random() < self.p_flip:
            img, mask = img[:, ::-1], mask[:, ::-1]

        if random.random() < self.p_affine:
            ang = random.uniform(-self.max_rot, self.max_rot)
            sc = random.uniform(*self.scale)
            tx = random.uniform(-self.max_shift, self.max_shift) * s
            ty = random.uniform(-self.max_shift, self.max_shift) * s
            M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, sc)
            M[:, 2] += (tx, ty)
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (s, s),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            m = cv2.warpAffine(np.ascontiguousarray(mask).astype(np.float32), M, (s, s),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = (m > 0.5).astype(np.uint8)

        img = np.ascontiguousarray(img, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

        if random.random() < self.p_intensity:           # contrast / brightness (z-score space)
            img = img * random.uniform(0.8, 1.25) + random.uniform(-0.3, 0.3)
        if random.random() < self.p_blur:
            sigma = random.uniform(0.3, 1.2)
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        if random.random() < self.p_noise:
            img = img + np.random.RandomState(random.getrandbits(32)).normal(
                0, random.uniform(0.02, 0.1), img.shape).astype(np.float32)
        return img, mask
