from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from viewer_plugins.vlm_assistor.vlm.client import VLMBackendClient


def test_structured_inventory_overrides_contradictory_containment(monkeypatch) -> None:
    app = SimpleNamespace(
        vlm_agent0_backend_mode="local_openai_compatible",
        vlm_local_base_url="http://example.invalid/v1",
        vlm_local_model="test-model",
        vlm_local_api_key="",
        agent1_mask_eval_language="zh",
    )
    client = VLMBackendClient(app)
    response = {
        "accurate": False,
        "inside_mask_objects": ["毛绒小熊"],
        "outside_mask_objects": ["橙色细长物体"],
        "target_fully_contained": False,
        "extra_objects": [],
        "missing_parts": [],
        "occluded_target_parts": [],
        "wrong_positive_points": [],
        "wrong_negative_points": [],
        "uncertain": False,
        "uncertainty_reason": "",
        "violates_instance_policy": False,
        "consistency_note": "",
        "summary": "结果：失败。",
    }
    monkeypatch.setattr(
        client,
        "_post_chat_completions",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}]
        },
    )

    result = client.evaluate_mask_with_vlm(
        object_name="毛绒小熊",
        image_rgb_u8=np.zeros((16, 16, 3), dtype=np.uint8),
        mask_bool=np.ones((16, 16), dtype=bool),
        non_target_objects=["橙色细长物体"],
        backend_mode="local_openai_compatible",
    )

    assert result["ok"] is True
    assert result["target_complete_in_mask"] is True
    assert result["inventory_policy"]["target_fully_contained_overridden"] is True
    assert "target_fully_contained_overridden_by_inventory" in result["reason_codes"]
    assert "pass" in result["summary"].lower() or "通过" in result["summary"]
