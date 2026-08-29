from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import imageio.v3 as iio
import numpy as np


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] > 3:
        # Composite RGBA onto white to avoid black backgrounds when alpha exists.
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3].astype(np.float32)
        # Normalize alpha to [0, 1] even if dtype is float or uint8.
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        alpha = alpha[..., None]
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)
        image = rgb
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def load_image(path: Path, flip_udlr: bool = False) -> np.ndarray:
    """Load an image and optionally flip both axes (COLMAP-style)."""
    image = ensure_rgb(iio.imread(path))
    if flip_udlr:
        image = np.flipud(np.fliplr(image))
    return image


def add_alpha_channel(image: np.ndarray, opacity: float) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if image.shape[2] == 4:
        out = image.copy().astype(np.float32)
        out[:, :, 3] = out[:, :, 3] * opacity
        return np.clip(out, 0, 255).astype(np.uint8)
    alpha = np.full((*image.shape[:2], 1), int(round(opacity * 255)), dtype=np.uint8)
    return np.concatenate([image.astype(np.uint8), alpha], axis=2)


def image_bbox_from_mask(image: np.ndarray, threshold: int = 3) -> Tuple[int, int, int, int]:
    mask = (image > threshold).any(axis=2)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        h, w = image.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resize_for_strip(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    target_width = max(1, int(round(w * target_height / float(h))))
    y_idx = np.linspace(0, h - 1, target_height).astype(np.int64)
    x_idx = np.linspace(0, w - 1, target_width).astype(np.int64)
    return image[y_idx][:, x_idx]


def build_timeline_strip(
    image_paths: Sequence[Path], target_height: int = 96, pad: int = 2
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    thumbs: List[np.ndarray] = []
    widths: List[int] = []
    for path in image_paths:
        img = ensure_rgb(iio.imread(path))
        thumb = resize_for_strip(img, target_height)
        thumbs.append(thumb)
        widths.append(thumb.shape[1])

    total_width = sum(widths) + pad * (len(widths) - 1)
    strip = np.full((target_height, total_width, 3), 245, dtype=np.uint8)
    spans: List[Tuple[int, int]] = []

    cursor = 0
    for thumb in thumbs:
        h, w = thumb.shape[:2]
        strip[:h, cursor : cursor + w] = thumb
        spans.append((cursor, cursor + w))
        cursor += w + pad

    return strip, spans


def highlight_strip(base_strip: np.ndarray, spans: Sequence[Tuple[int, int]], index: int) -> np.ndarray:
    index = int(max(0, min(len(spans) - 1, index)))
    strip = base_strip.copy()
    start, end = spans[index]
    start = max(0, start)
    end = min(strip.shape[1], end)
    color = np.array([220, 40, 40], dtype=np.uint8)
    thickness = 3
    strip[:thickness, start:end] = color
    strip[-thickness:, start:end] = color
    strip[:, start : min(end, start + thickness)] = color
    strip[:, max(start, end - thickness) : end] = color
    return strip
