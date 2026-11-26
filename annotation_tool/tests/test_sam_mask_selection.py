from __future__ import annotations

import numpy as np

from viewer_plugins.sam3d_gen.mask_selection import select_prompt_consistent_mask


def test_prefers_prompt_consistent_object_over_higher_score_background() -> None:
    background = np.ones((10, 10), dtype=bool)
    background[4:7, 4:7] = False
    target = np.zeros((10, 10), dtype=bool)
    target[4:7, 4:7] = True

    selected = select_prompt_consistent_mask(
        np.stack([background, target]),
        np.asarray([0.99, 0.70]),
        np.asarray([[5, 5], [0, 0]]),
        np.asarray([1, 0]),
    )

    assert selected == 1


def test_prefers_tighter_mask_when_click_consistency_matches() -> None:
    large = np.zeros((10, 10), dtype=bool)
    large[1:9, 1:9] = True
    tight = np.zeros((10, 10), dtype=bool)
    tight[4:7, 4:7] = True

    selected = select_prompt_consistent_mask(
        np.stack([large, tight]),
        np.asarray([0.95, 0.60]),
        np.asarray([[5, 5]]),
        np.asarray([1]),
    )

    assert selected == 1
