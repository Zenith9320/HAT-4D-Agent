from __future__ import annotations

from types import SimpleNamespace

from viewer_plugins.vlm_assistor.agents.core import VideoUnderstandingAgent
from viewer_plugins.vlm_assistor.vlm.client import VLMBackendClient


def test_auto_sampling_keeps_all_frames_for_short_clip() -> None:
    assert VideoUnderstandingAgent._auto_sample_indices(32) == list(range(32))


def test_auto_sampling_bounds_long_clip_cost() -> None:
    assert VideoUnderstandingAgent._auto_sample_indices(52) == list(range(0, 52, 4))


def test_agent0_default_segments_are_short_enough_for_temporal_comparison(monkeypatch) -> None:
    monkeypatch.delenv("VLM_AGENT0_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("VLM_AGENT0_CHUNK_OVERLAP", raising=False)
    client = VLMBackendClient(SimpleNamespace())

    assert client._segment_config() == (12, 3)
    assert client._segment_ranges(32, 12, 3) == [(0, 12), (9, 21), (18, 30), (27, 32)]
