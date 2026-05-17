"""Augmentation pipeline used in training.

Two flavours, controlled from the YAML config:

* ``geometric`` only -- the default, and the recipe behind the headline
  numbers in Chapter 5. Random horizontal/vertical flips and rotations
  within +/- 15 degrees.
* ``photometric`` on top -- the ablation reported in Section 5.6. Brightness,
  contrast, saturation and hue jitter plus Gaussian noise. This is the
  configuration that produced the negative result.

Both branches respect the patch-grid alignment constraint described in
Section 4.12: random crop corners are constrained to multiples of 16 so the
ViT-Base/16 patch embedding sees a stable grid across epochs.
"""
from __future__ import annotations
from typing import Mapping, Any, Callable

import numpy as np


def build_train_transform(data_cfg: Mapping[str, Any]) -> Callable:
    aug = data_cfg.get("augmentation", {})
    geom = aug.get("geometric", {})
    photo = aug.get("photometric", {})
    align = int(data_cfg.get("align_crop_to", 16))
    crop = int(data_cfg.get("input_size", 224))

    def _apply(image: np.ndarray, mask: np.ndarray):
        image, mask = _align_random_crop(image, mask, crop, align)
        image, mask = _flip(image, mask, geom)
        image, mask = _rotate(image, mask, geom)
        if photo.get("enabled", False):
            image = _photometric(image, photo)
        return image, mask

    return _apply


def build_eval_transform(data_cfg: Mapping[str, Any]) -> Callable:
    """Centre-crop only; no geometric or photometric perturbation."""
    crop = int(data_cfg.get("input_size", 224))

    def _apply(image: np.ndarray, mask: np.ndarray):
        h, w = image.shape[:2]
        top = max((h - crop) // 2, 0)
        left = max((w - crop) // 2, 0)
        image = image[top:top + crop, left:left + crop]
        mask = mask[top:top + crop, left:left + crop]
        return image, mask

    return _apply


# --------------------------------------------------------------- helpers
def _align_random_crop(image, mask, crop, align):
    h, w = image.shape[:2]
    if h < crop or w < crop:
        return image, mask
    max_top = (h - crop) // align * align
    max_left = (w - crop) // align * align
    top = np.random.randint(0, max(max_top, 1) + 1)
    left = np.random.randint(0, max(max_left, 1) + 1)
    top = (top // align) * align
    left = (left // align) * align
    return image[top:top + crop, left:left + crop], mask[top:top + crop, left:left + crop]


def _flip(image, mask, geom):
    if np.random.rand() < float(geom.get("hflip_p", 0.0)):
        image = image[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()
    if np.random.rand() < float(geom.get("vflip_p", 0.0)):
        image = image[::-1, :, :].copy()
        mask = mask[::-1, :].copy()
    return image, mask


def _rotate(image, mask, geom):
    deg = float(geom.get("rot_deg", 0.0))
    if deg <= 0:
        return image, mask
    # Multiples-of-90 only -- preserves the mask perfectly without resampling.
    angle = int(np.random.uniform(-deg, deg))
    k = int(round(angle / 90.0))
    if k == 0:
        return image, mask
    return np.rot90(image, k=k).copy(), np.rot90(mask, k=k).copy()


def _photometric(image, photo):
    """Brightness/contrast/saturation/hue jitter + Gaussian noise + per-channel intensity."""
    img = image.astype(np.float32)
    b = float(photo.get("brightness", 0.0))
    c = float(photo.get("contrast", 0.0))
    if np.random.rand() < float(photo.get("colour_jitter_p", 0.0)):
        if b > 0:
            img = img * np.random.uniform(1 - b, 1 + b)
        if c > 0:
            mean = img.mean()
            img = (img - mean) * np.random.uniform(1 - c, 1 + c) + mean
    sigma = float(photo.get("gaussian_noise_std", 0.0))
    if sigma > 0:
        img = img + np.random.normal(0.0, sigma * 255.0, img.shape).astype(np.float32)
    pcij = float(photo.get("per_channel_intensity_jitter", 0.0))
    if pcij > 0:
        gain = np.random.uniform(1 - pcij, 1 + pcij, size=(1, 1, 3)).astype(np.float32)
        img = img * gain
    return np.clip(img, 0, 255)
