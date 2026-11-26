from __future__ import annotations

import numpy as np

from viewer_plugins.vlm_assistor.agents.core import _AgentBase


def test_yellow_target_points_are_moved_into_yellow_interior() -> None:
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    image[20:60, 35:75] = np.asarray([230, 205, 45], dtype=np.uint8)

    points = _AgentBase._refine_color_grounded_points(
        "黄色香蕉", image, [(0, 0)] * 8, 8
    )

    assert len(points) == 8
    assert all(35 <= x < 75 and 20 <= y < 60 for x, y in points)


def test_non_yellow_target_keeps_vlm_points() -> None:
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    original = [(2, 3), (4, 5)]
    assert _AgentBase._refine_color_grounded_points("绿色切菜刀", image, original, 2) == original


def test_yellow_mask_contract_accepts_high_support_recall() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[30:60, 35:65] = np.asarray([230, 205, 45], dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[28:62, 33:67] = True

    result = _AgentBase._color_mask_contract("黄色香蕉", image, mask)

    assert result["applicable"] is True
    assert result["ok"] is True
    assert result["support_recall"] == 1.0


def test_yellow_mask_contract_rejects_partial_mask() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[30:60, 35:65] = np.asarray([230, 205, 45], dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:40, 35:65] = True

    result = _AgentBase._color_mask_contract("黄色香蕉", image, mask)

    assert result["ok"] is False
