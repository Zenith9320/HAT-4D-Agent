from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch.nn as nn

from sgm.modules.attention import CrossAttention, MemoryEfficientCrossAttention


@dataclass(frozen=True)
class DeepCacheOptions:
    enabled: bool = False
    mode: str = "kv"
    start_step: int = 0
    interval: int = 1
    profile: bool = False
    precompute: bool = False

    @staticmethod
    def from_dict(options: Optional[Dict[str, Any]]) -> "DeepCacheOptions":
        if not options:
            return DeepCacheOptions()
        return DeepCacheOptions(
            enabled=bool(options.get("enabled", False)),
            mode=str(options.get("mode", "kv")),
            start_step=int(options.get("start_step", 0) or 0),
            interval=int(options.get("interval", 1) or 1),
            profile=bool(options.get("profile", False)),
            precompute=bool(options.get("precompute", False)),
        )


def _iter_cross_attention_modules(module: nn.Module) -> Iterable[nn.Module]:
    for m in module.modules():
        if isinstance(m, (CrossAttention, MemoryEfficientCrossAttention)):
            yield m


def enable_deepcache(module: nn.Module, options: Optional[Dict[str, Any]]) -> DeepCacheOptions:
    parsed = DeepCacheOptions.from_dict(options)
    if not parsed.enabled:
        return parsed
    modes = {m.strip() for m in parsed.mode.split("+") if m.strip()}
    supported = {"kv", "feature"}
    if not modes.issubset(supported):
        raise ValueError(
            f"Unsupported DeepCache mode: {parsed.mode!r} (supported: 'kv', 'feature', or 'kv+feature')"
        )

    if "kv" in modes:
        for attn in _iter_cross_attention_modules(module):
            attn.kv_cache_enabled = True
            if parsed.profile:
                attn.kv_cache_profile = True
                attn.kv_cache_stats = {"hits": 0, "misses": 0, "kv_proj_s": 0.0}

    if "feature" in modes:
        from sgm.modules.spacetime_attention import (
            PostHocSpatialTransformerWithTimeMixingAndMotion,
        )

        for m in module.modules():
            if isinstance(m, PostHocSpatialTransformerWithTimeMixingAndMotion):
                m.deepcache_feature_enabled = True
                m.deepcache_start_step = parsed.start_step
                m.deepcache_interval = max(1, parsed.interval)
                m.deepcache_precompute_enabled = bool(parsed.precompute)
                m.deepcache_profile = parsed.profile
                if hasattr(m, "clear_deepcache"):
                    m.clear_deepcache()

    if "kv" in modes and "feature" not in modes:
        from sgm.modules.spacetime_attention import (
            PostHocSpatialTransformerWithTimeMixingAndMotion,
        )

        for m in module.modules():
            if isinstance(m, PostHocSpatialTransformerWithTimeMixingAndMotion):
                m.deepcache_precompute_enabled = bool(parsed.precompute)
                m.deepcache_profile = parsed.profile
                if hasattr(m, "clear_deepcache"):
                    m.clear_deepcache()
    return parsed


def disable_deepcache(
    module: nn.Module, options: DeepCacheOptions, *, clear: bool = True
) -> None:
    if not options.enabled:
        return
    modes = {m.strip() for m in options.mode.split("+") if m.strip()}
    if "kv" in modes:
        for attn in _iter_cross_attention_modules(module):
            attn.kv_cache_enabled = False
            attn.kv_cache_profile = False
            attn.kv_cache_stats = None
            if clear and hasattr(attn, "clear_kv_cache"):
                attn.clear_kv_cache()

    if "feature" in modes:
        from sgm.modules.spacetime_attention import (
            PostHocSpatialTransformerWithTimeMixingAndMotion,
        )

        for m in module.modules():
            if isinstance(m, PostHocSpatialTransformerWithTimeMixingAndMotion):
                m.deepcache_feature_enabled = False
                m.deepcache_precompute_enabled = False
                m.deepcache_profile = False
                if clear and hasattr(m, "clear_deepcache"):
                    m.clear_deepcache()

    # Always clear any leftover feature/precompute state to avoid cross-run leakage.
    if "feature" not in modes:
        try:
            from sgm.modules.spacetime_attention import (
                PostHocSpatialTransformerWithTimeMixingAndMotion,
            )

            for m in module.modules():
                if isinstance(m, PostHocSpatialTransformerWithTimeMixingAndMotion):
                    m.deepcache_feature_enabled = False
                    m.deepcache_precompute_enabled = False
                    m.deepcache_profile = False
                    if clear and hasattr(m, "clear_deepcache"):
                        m.clear_deepcache()
        except Exception:
            pass


def collect_deepcache_stats(module: nn.Module) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    kv_hits = 0
    kv_misses = 0
    kv_proj_s = 0.0
    for attn in _iter_cross_attention_modules(module):
        d = getattr(attn, "kv_cache_stats", None)
        if isinstance(d, dict):
            kv_hits += int(d.get("hits", 0))
            kv_misses += int(d.get("misses", 0))
            kv_proj_s += float(d.get("kv_proj_s", 0.0))
    stats["kv"] = {
        "hits": kv_hits,
        "misses": kv_misses,
        "hit_rate": (kv_hits / max(1, kv_hits + kv_misses)),
        "kv_proj_s": kv_proj_s,
    }

    try:
        from sgm.modules.spacetime_attention import (
            PostHocSpatialTransformerWithTimeMixingAndMotion,
        )

        prep_hits = 0
        prep_misses = 0
        prep_s = 0.0
        feat_cam_hits = 0
        feat_cam_misses = 0
        feat_mot_hits = 0
        feat_mot_misses = 0
        for m in module.modules():
            if not isinstance(m, PostHocSpatialTransformerWithTimeMixingAndMotion):
                continue
            d = getattr(m, "_deepcache_stats", None)
            if isinstance(d, dict):
                prep_hits += int(d.get("prep_hits", 0))
                prep_misses += int(d.get("prep_misses", 0))
                prep_s += float(d.get("prep_s", 0.0))
                feat_cam_hits += int(d.get("cam_hits", 0))
                feat_cam_misses += int(d.get("cam_misses", 0))
                feat_mot_hits += int(d.get("mot_hits", 0))
                feat_mot_misses += int(d.get("mot_misses", 0))
        stats["precompute"] = {
            "hits": prep_hits,
            "misses": prep_misses,
            "hit_rate": (prep_hits / max(1, prep_hits + prep_misses)),
            "prep_s": prep_s,
        }
        stats["feature"] = {
            "cam_hits": feat_cam_hits,
            "cam_misses": feat_cam_misses,
            "mot_hits": feat_mot_hits,
            "mot_misses": feat_mot_misses,
        }
    except Exception:
        pass

    return stats
