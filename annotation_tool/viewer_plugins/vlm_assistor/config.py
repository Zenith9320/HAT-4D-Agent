from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RELATION_HINT = "Identify physically implausible or inconsistent object relations in the scene."
DEFAULT_LOCAL_VLM_BASE_URL = "http://127.0.0.1:22003/v1"
DEFAULT_LOCAL_VLM_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
DEFAULT_LOCAL_VLM_API_KEY = ""
DEFAULT_AGENT1_MASK_EVAL_LANGUAGE = "auto"


def _safe_int(raw: str | None, fallback: int) -> int:
    if raw is None:
        return int(fallback)
    try:
        return int(raw)
    except Exception:
        return int(fallback)


def _safe_bool(raw: str | None, fallback: bool) -> bool:
    if raw is None:
        return bool(fallback)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(fallback)


def _safe_float(raw: str | None, fallback: float) -> float:
    if raw is None:
        return float(fallback)
    try:
        return float(raw)
    except Exception:
        return float(fallback)


def resolve_input_frames_source(app: Any | None) -> str:
    """Return file://<input_dir> when app has a valid frame directory."""
    if app is None:
        return ""
    raw = getattr(app, "input_dir", None)
    if raw is None:
        return ""
    try:
        p = Path(raw).expanduser().resolve()
    except Exception:
        return ""
    if not p.exists() or not p.is_dir():
        return ""
    return f"file://{p}"


@dataclass(frozen=True)
class VLMAssistorDefaults:
    relation_hint: str
    agent0_backend_mode: str
    agent0_max_frames: int
    agent0_render_size: int
    agent0_local_base_url: str
    agent0_local_model: str
    agent0_local_api_key: str
    agent0_local_video_url: str
    agent0_ikg_enabled: bool
    agent1_points_per_object: int
    agent1_max_retries: int
    agent1_mask_min_area_ratio: float
    agent1_visualize_seconds: float
    agent1_debug_save_eval: bool
    save_intermediate_results: bool
    agent1_mask_eval_language: str
    agent2_x_step: float
    agent2_num_candidates: int
    agent2_max_rounds: int
    agent2_x_min: float
    agent2_x_max: float
    agent2_pose_opt_steps: int
    agent2_pose_opt_lr: float
    agent2_pose_opt_min_side: int
    agent2_pose_opt_rgb_weight: float
    agent2_pose_opt_timeout_sec: float
    agent2_debug_save_views: bool
    use_pose_adjust_v2: bool
    agent4_history_window: int
    agent4_views_per_frame: int
    agent4_max_images: int
    agent4_history_capacity: int
    agent4_memory_bank_capacity: int
    agent4_memory_enabled: bool
    agent4_validator_enabled: bool
    agent4_use_smolagents: bool


def build_defaults(app: Any | None = None) -> VLMAssistorDefaults:
    backend_mode = os.getenv("VLM_AGENT0_BACKEND_MODE") or "local_openai_compatible"
    if backend_mode not in {"official_api", "local_openai_compatible"}:
        backend_mode = "local_openai_compatible"

    raw_mask_eval_lang = str(
        os.getenv("VLM_AGENT1_MASK_EVAL_LANGUAGE") or DEFAULT_AGENT1_MASK_EVAL_LANGUAGE
    ).strip().lower()
    if raw_mask_eval_lang in {"zh", "cn", "chinese"}:
        mask_eval_lang = "zh"
    elif raw_mask_eval_lang in {"en", "english"}:
        mask_eval_lang = "en"
    else:
        mask_eval_lang = "auto"

    if app is not None and hasattr(app, "use_pose_adjust_v2"):
        use_pose_adjust_v2 = bool(getattr(app, "use_pose_adjust_v2"))
    else:
        use_pose_adjust_v2 = _safe_bool(os.getenv("VLM_USE_POSE_ADJUST_V2"), False)

    return VLMAssistorDefaults(
        relation_hint=os.getenv("VLM_RELATION_HINT") or DEFAULT_RELATION_HINT,
        agent0_backend_mode=backend_mode,
        # 0 keeps clips up to 48 frames intact, then samples every 4 frames.
        agent0_max_frames=max(0, _safe_int(os.getenv("VLM_AGENT0_MAX_FRAMES"), 0)),
        agent0_render_size=max(128, _safe_int(os.getenv("VLM_AGENT0_RENDER_SIZE"), 384)),
        agent0_local_base_url=os.getenv("VLM_LOCAL_BASE_URL") or DEFAULT_LOCAL_VLM_BASE_URL,
        agent0_local_model=os.getenv("VLM_LOCAL_MODEL") or DEFAULT_LOCAL_VLM_MODEL,
        agent0_local_api_key=os.getenv("VLM_LOCAL_API_KEY") or DEFAULT_LOCAL_VLM_API_KEY,
        agent0_local_video_url=os.getenv("VLM_LOCAL_VIDEO_URL") or resolve_input_frames_source(app),
        agent0_ikg_enabled=_safe_bool(os.getenv("VLM_AGENT0_IKG_ENABLED"), True),
        agent1_points_per_object=max(1, min(8, _safe_int(os.getenv("VLM_AGENT1_POINTS_PER_OBJECT"), 3))),
        agent1_max_retries=max(1, min(10, _safe_int(os.getenv("VLM_AGENT1_MAX_RETRIES"), 5))),
        agent1_mask_min_area_ratio=max(0.0, min(0.5, _safe_float(os.getenv("VLM_AGENT1_MASK_MIN_AREA_RATIO"), 0.001))),
        agent1_visualize_seconds=max(0.0, min(5.0, _safe_float(os.getenv("VLM_AGENT1_VISUALIZE_SECONDS"), 1.2))),
        agent1_debug_save_eval=_safe_bool(os.getenv("VLM_AGENT1_DEBUG_SAVE_EVAL"), False),
        save_intermediate_results=_safe_bool(os.getenv("VLM_SAVE_INTERMEDIATE_RESULTS"), False),
        agent1_mask_eval_language=mask_eval_lang,
        agent2_x_step=max(0.01, min(0.5, _safe_float(os.getenv("VLM_AGENT2_X_STEP"), 0.1))),
        agent2_num_candidates=max(1, min(8, _safe_int(os.getenv("VLM_AGENT2_NUM_CANDIDATES"), 4))),
        agent2_max_rounds=max(1, min(6, _safe_int(os.getenv("VLM_AGENT2_MAX_ROUNDS"), 3))),
        agent2_x_min=max(-5.0, min(5.0, _safe_float(os.getenv("VLM_AGENT2_X_MIN"), -1.0))),
        agent2_x_max=max(-5.0, min(5.0, _safe_float(os.getenv("VLM_AGENT2_X_MAX"), 1.0))),
        agent2_pose_opt_steps=max(10, min(1200, _safe_int(os.getenv("VLM_AGENT2_POSE_OPT_STEPS"), 160))),
        agent2_pose_opt_lr=max(1e-4, min(1.0, _safe_float(os.getenv("VLM_AGENT2_POSE_OPT_LR"), 0.05))),
        agent2_pose_opt_min_side=max(64, min(2048, _safe_int(os.getenv("VLM_AGENT2_POSE_OPT_MIN_SIDE"), 512))),
        agent2_pose_opt_rgb_weight=max(0.0, min(1.0, _safe_float(os.getenv("VLM_AGENT2_POSE_OPT_RGB_WEIGHT"), 0.4))),
        agent2_pose_opt_timeout_sec=max(5.0, min(7200.0, _safe_float(os.getenv("VLM_AGENT2_POSE_OPT_TIMEOUT_SEC"), 1500.0))),
        agent2_debug_save_views=_safe_bool(os.getenv("VLM_AGENT2_DEBUG_SAVE_VIEWS"), False),
        use_pose_adjust_v2=bool(use_pose_adjust_v2),
        agent4_history_window=max(0, _safe_int(os.getenv("VLM_AGENT4_HISTORY_WINDOW"), 6)),
        agent4_views_per_frame=max(1, _safe_int(os.getenv("VLM_AGENT4_VIEWS_PER_FRAME"), 3)),
        agent4_max_images=max(1, _safe_int(os.getenv("VLM_AGENT4_MAX_IMAGES"), 12)),
        agent4_history_capacity=max(1, _safe_int(os.getenv("VLM_AGENT4_HISTORY_CAPACITY"), 64)),
        agent4_memory_bank_capacity=max(1, _safe_int(os.getenv("VLM_AGENT4_MEMORY_BANK_CAPACITY"), 8)),
        agent4_memory_enabled=_safe_bool(os.getenv("VLM_AGENT4_MEMORY_ENABLED"), True),
        agent4_validator_enabled=_safe_bool(os.getenv("VLM_AGENT4_VALIDATOR_ENABLED"), True),
        agent4_use_smolagents=_safe_bool(os.getenv("VLM_AGENT4_USE_SMOLAGENTS"), False),
    )
