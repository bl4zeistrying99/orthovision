"""Pure-numpy preprocessing so it can be unit-tested without torch."""
from __future__ import annotations

import numpy as np

# ImageNet statistics, since all four encoders use ImageNet-pretrained weights.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_intensity(vol: np.ndarray) -> np.ndarray:
    """Per-series min-max scaling to [0, 1]. vol: (S, H, W) any numeric dtype."""
    vol = vol.astype(np.float32)
    lo, hi = float(vol.min()), float(vol.max())
    if hi - lo < 1e-8:
        return np.zeros_like(vol)
    return (vol - lo) / (hi - lo)


def resize_slices(vol: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize every slice to (size, size). vol: (S, H, W) float32."""
    if vol.shape[1] == size and vol.shape[2] == size:
        return vol
    from scipy.ndimage import zoom

    zh, zw = size / vol.shape[1], size / vol.shape[2]
    return zoom(vol, (1, zh, zw), order=1).astype(np.float32)


def to_three_channel(vol: np.ndarray) -> np.ndarray:
    """(S, H, W) in [0,1] -> (S, 3, H, W), ImageNet-normalised."""
    x = np.repeat(vol[:, None, :, :], 3, axis=1)
    x = (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    return x.astype(np.float32)


def preprocess_series(vol: np.ndarray, size: int = 224) -> np.ndarray:
    """Full pipeline for one plane of one exam: (S,H,W) -> (S,3,size,size)."""
    return to_three_channel(resize_slices(normalize_intensity(vol), size))
