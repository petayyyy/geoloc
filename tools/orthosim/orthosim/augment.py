"""Query-image augmentations for OrthoSim (T10 §3).

Every function maps an ``(H, W, 3)`` float64 image in ``[0, 255]`` to a new
array and is a no-op at its identity parameter (T10-U-05). Randomness flows only
through the passed ``rng`` (a ``np.random.Generator``) so a fixed seed gives a
bit-identical result (P0 rule 3). Geometry perturbations (attitude/AGL/time
sync) are deliberately NOT image ops -- they change the pipeline's *input pose*,
so they are applied at render time and never touch the ground truth.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

_MEAN = 128.0


def gamma(img: np.ndarray, value: float) -> np.ndarray:
    if abs(value - 1.0) < 1e-12:
        return img.copy()
    return np.clip((img / 255.0) ** (1.0 / value) * 255.0, 0, 255)


def white_balance(img: np.ndarray, gains) -> np.ndarray:
    g = np.asarray(gains, dtype=np.float64)
    if np.allclose(g, 1.0):
        return img.copy()
    return np.clip(img * g, 0, 255)


def contrast(img: np.ndarray, value: float) -> np.ndarray:
    if abs(value - 1.0) < 1e-12:
        return img.copy()
    return np.clip((img - _MEAN) * value + _MEAN, 0, 255)


def haze(img: np.ndarray, value: float) -> np.ndarray:
    if value <= 0.0:
        return img.copy()
    return np.clip(img * (1.0 - value) + 180.0 * value, 0, 255)


def gaussian_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return img.copy()
    return np.clip(img + rng.normal(0.0, sigma, size=img.shape), 0, 255)


def motion_blur(img: np.ndarray, length: int) -> np.ndarray:
    length = int(length)
    if length <= 1:
        return img.copy()
    k = np.ones(length) / length
    out = img.copy()
    for _ in range(2):
        out = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 1, out)
    return np.clip(out, 0, 255)


def vignette(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return img.copy()
    h, w = img.shape[:2]
    yy, xx = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.7071
    gain = 1.0 - strength * np.clip(r, 0, 1) ** 2
    return np.clip(img * gain[..., None], 0, 255)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return img.copy()
    buf = io.BytesIO()
    arr8 = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(arr8).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)


def snow(img: np.ndarray, rng: np.random.Generator, coverage: float) -> np.ndarray:
    """Naive snow: replace a fraction of pixels with near-white (T10 §3 season)."""
    if coverage <= 0.0:
        return img.copy()
    h, w = img.shape[:2]
    mask = rng.random((h, w)) < coverage
    out = img.copy()
    white = 235.0 + rng.random((h, w)) * 20.0
    out[mask] = white[mask, None]
    return out


def season_shift(img: np.ndarray, value: float) -> np.ndarray:
    """Coarse seasonal colour transfer: mix toward a cold (winter) tint."""
    if abs(value) < 1e-12:
        return img.copy()
    out = img.copy()
    out[..., 0] = out[..., 0] * (1.0 - value) + (out[..., 0] * 0.9 + 25.0) * value
    out[..., 1] = out[..., 1] * (1.0 - value) + (out[..., 1] * 0.85) * value
    out[..., 2] = out[..., 2] * (1.0 - value) + (out[..., 2] * 0.7 + 30.0) * value
    return np.clip(out, 0, 255)


_AUGMENTORS = {
    "gamma": gamma,
    "white_balance": white_balance,
    "contrast": contrast,
    "haze": haze,
    "motion_blur": motion_blur,
    "vignette": vignette,
    "jpeg": jpeg,
    "season_shift": season_shift,
}


def apply_augmentations(img: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply the configured augmentations in a fixed order.

    ``params`` is a flat dict of ``name -> value``. Noise and snow draw from
    ``rng``; every other operation is deterministic given its value.
    """
    out = img.astype(np.float64)
    for name in (
        "gamma",
        "white_balance",
        "contrast",
        "haze",
        "season_shift",
        "snow",
        "motion_blur",
        "vignette",
        "gaussian_noise",
        "jpeg",
    ):
        if name not in params:
            continue
        value = params[name]
        if name == "snow":
            out = snow(out, rng, float(value))
        elif name == "gaussian_noise":
            out = gaussian_noise(out, rng, float(value))
        else:
            out = _AUGMENTORS[name](out, value)
    return np.clip(out, 0, 255)
