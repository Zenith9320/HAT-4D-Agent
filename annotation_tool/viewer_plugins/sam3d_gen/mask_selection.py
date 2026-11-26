from __future__ import annotations

import numpy as np


def select_prompt_consistent_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    points_xy: np.ndarray,
    point_labels: np.ndarray,
) -> int:
    """Select a SAM candidate using click consistency before model confidence."""
    candidates = np.asarray(masks, dtype=bool)
    if candidates.ndim == 2:
        candidates = candidates[None, ...]
    model_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
    if candidates.shape[0] <= 1 or points.shape[0] == 0:
        return int(np.argmax(model_scores[: candidates.shape[0]])) if model_scores.size else 0

    h, w = candidates.shape[-2:]
    xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, max(0, w - 1))
    ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, max(0, h - 1))
    positive = labels == 1
    negative = labels == 0

    ranked: list[tuple[float, float, float, float, int]] = []
    for idx, mask in enumerate(candidates):
        pos_cov = float(mask[ys[positive], xs[positive]].mean()) if positive.any() else 1.0
        neg_cov = float(mask[ys[negative], xs[negative]].mean()) if negative.any() else 0.0
        area_ratio = float(mask.mean())
        confidence = float(model_scores[idx]) if idx < model_scores.size else 0.0
        # Click agreement is the hard signal. For equally consistent masks, prefer
        # the tighter instance mask, then SAM's own confidence.
        ranked.append((pos_cov, -neg_cov, -area_ratio, confidence, -idx))
    return int(max(range(len(ranked)), key=ranked.__getitem__))
