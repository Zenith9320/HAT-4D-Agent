from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import VLMAssistorDefaults, build_defaults


@dataclass
class AgentLogEntry:
    ts: str
    agent: str
    status: str
    message: str


class VLMAssistorState:
    def __init__(self, defaults: VLMAssistorDefaults | None = None) -> None:
        defaults = defaults or build_defaults()
        self.lock = threading.Lock()

        self.enabled: bool = True
        self.scene_graph: dict[str, Any] = {}
        self.object_description: str = ""
        self.relation_hint: str = str(defaults.relation_hint)

        self.last_video_understanding: dict[str, Any] = {}
        self.last_click_result: dict[str, Any] = {}
        self.last_gen_validation: dict[str, Any] = {}
        self.last_pose_adjust: dict[str, Any] = {}
        self.last_4d_validation: dict[str, Any] = {}
        self.last_4d_memory_result: dict[str, Any] = {}

        self.logs: list[AgentLogEntry] = []
        self.max_logs: int = 40

        self.artifact_root: Path | None = None
        self.artifact_base_root: Path | None = None
        self.save_intermediate_results: bool = bool(getattr(defaults, "save_intermediate_results", False))
        self.intermediate_run_root: Path | None = None
        self.intermediate_run_id: str = ""
        self.depth_scan_min: float = -0.35
        self.depth_scan_max: float = 0.35
        self.depth_scan_steps: int = 9
        self.agent0_use_gpt4o: bool = True
        self.agent0_backend_mode: str = str(defaults.agent0_backend_mode)  # official_api | local_openai_compatible
        self.agent0_max_frames: int = int(defaults.agent0_max_frames)
        self.agent0_render_size: int = int(defaults.agent0_render_size)
        self.agent0_local_base_url: str = str(defaults.agent0_local_base_url)
        self.agent0_local_model: str = str(defaults.agent0_local_model)
        self.agent0_local_api_key: str = str(defaults.agent0_local_api_key)
        self.agent0_local_video_url: str = str(defaults.agent0_local_video_url)
        self.agent0_ikg_enabled: bool = bool(getattr(defaults, "agent0_ikg_enabled", True))
        self.agent1_points_per_object: int = int(defaults.agent1_points_per_object)
        self.agent1_max_retries: int = int(defaults.agent1_max_retries)
        self.agent1_mask_min_area_ratio: float = float(defaults.agent1_mask_min_area_ratio)
        self.agent1_debug_save_eval: bool = bool(defaults.agent1_debug_save_eval)
        self.agent1_mask_eval_language: str = str(getattr(defaults, "agent1_mask_eval_language", "auto"))
        self.agent1_targets: list[str] = []
        self.agent1_target_index: int = 0
        self.agent1_generated_objects: list[str] = []
        self.agent1_generated_object_indices: dict[str, int] = {}
        self.agent2_x_step: float = float(getattr(defaults, "agent2_x_step", 0.1))
        self.agent2_num_candidates: int = int(getattr(defaults, "agent2_num_candidates", 4))
        self.agent2_max_rounds: int = int(getattr(defaults, "agent2_max_rounds", 3))
        self.agent2_x_min: float = float(getattr(defaults, "agent2_x_min", -1.0))
        self.agent2_x_max: float = float(getattr(defaults, "agent2_x_max", 1.0))
        self.agent2_pose_opt_steps: int = int(getattr(defaults, "agent2_pose_opt_steps", 160))
        self.agent2_pose_opt_lr: float = float(getattr(defaults, "agent2_pose_opt_lr", 0.05))
        self.agent2_pose_opt_min_side: int = int(getattr(defaults, "agent2_pose_opt_min_side", 512))
        self.agent2_pose_opt_rgb_weight: float = float(getattr(defaults, "agent2_pose_opt_rgb_weight", 0.4))
        self.agent2_pose_opt_timeout_sec: float = float(getattr(defaults, "agent2_pose_opt_timeout_sec", 1500.0))
        self.agent2_debug_save_views: bool = bool(getattr(defaults, "agent2_debug_save_views", False))
        self.use_pose_adjust_v2: bool = bool(getattr(defaults, "use_pose_adjust_v2", False))
        self.agent0_last_source: str = ""
        self.agent0_last_frame_count: int = 0
        self.agent4_history_window: int = int(defaults.agent4_history_window)
        self.agent4_views_per_frame: int = int(defaults.agent4_views_per_frame)
        self.agent4_max_images: int = int(defaults.agent4_max_images)
        self.agent4_history_capacity: int = int(defaults.agent4_history_capacity)
        self.agent4_memory_bank_capacity: int = int(getattr(defaults, "agent4_memory_bank_capacity", 8))
        self.agent4_memory_enabled: bool = bool(getattr(defaults, "agent4_memory_enabled", True))
        self.agent4_validator_enabled: bool = bool(getattr(defaults, "agent4_validator_enabled", True))
        self.agent4_use_smolagents: bool = bool(defaults.agent4_use_smolagents)
        self.agent4_history_records: list[dict[str, Any]] = []
        self.agent4_history_memory: str = ""

        # UI handles
        self.status_handle: Any = None
        self.object_description_handle: Any = None
        self.log_handle: Any = None
        self.scene_graph_handle: Any = None
        self.scene_graph_detail_handle: Any = None
        self.agent4_result_handle: Any = None
