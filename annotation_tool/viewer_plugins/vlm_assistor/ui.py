from __future__ import annotations

import os

from .config import (
    DEFAULT_LOCAL_VLM_BASE_URL,
    DEFAULT_LOCAL_VLM_MODEL,
    resolve_input_frames_source,
)


class VLMAssistorUI:
    def __init__(self, ctx, ops, state):
        self.ctx = ctx
        self.ops = ops
        self.state = state
        self.folder = None

    def build(self, client, parent_folder=None):
        if parent_folder is None:
            folder = client.gui.add_folder("VLM Assistor", expand_by_default=False)
        else:
            with parent_folder:
                folder = client.gui.add_folder("VLM Assistor", expand_by_default=False)
        self.folder = folder

        with folder:
            self.enable_toggle = client.gui.add_checkbox(
                "Enable VLM Assistor",
                initial_value=bool(self.state.enabled),
            )
            self.object_text = client.gui.add_text(
                "Object description",
                initial_value=str(self.state.object_description),
                hint="Comma-separated objects or the current target object name.",
            )
            self.state.object_description_handle = self.object_text
            self.relation_text = client.gui.add_text(
                "Relation hint",
                initial_value=str(self.state.relation_hint),
                hint="Example: knife should contact banana in cutting stage.",
            )
            mode_init = str(getattr(self.state, "agent0_backend_mode", "official_api") or "official_api")
            if mode_init not in {"official_api", "local_openai_compatible"}:
                mode_init = "official_api"
            self.agent0_backend_mode = client.gui.add_dropdown(
                "Agent0 backend mode",
                options=["official_api", "local_openai_compatible"],
                initial_value=mode_init,
                hint=(
                    "official_api: toolkit/shared-dir + GPT-4o official API fallback. "
                    "local_openai_compatible: call your local OpenAI-compatible /v1 endpoint."
                ),
            )
            self.agent0_use_gpt4o = client.gui.add_checkbox(
                "Official mode: enable GPT-4o/toolkit path",
                initial_value=bool(self.state.agent0_use_gpt4o),
                hint=(
                    "Used only when backend mode is official_api. "
                    "If VLM_SHARED_DIR is set, Agent0 first uses VLM_toolkit_cross_gpfs."
                ),
            )
            local_base_init = str(
                getattr(self.state, "agent0_local_base_url", "")
                or getattr(self.ctx.app, "vlm_local_base_url", None)
                or os.getenv("VLM_LOCAL_BASE_URL")
                or DEFAULT_LOCAL_VLM_BASE_URL
            )
            local_model_init = str(
                getattr(self.state, "agent0_local_model", "")
                or getattr(self.ctx.app, "vlm_local_model", None)
                or os.getenv("VLM_LOCAL_MODEL")
                or DEFAULT_LOCAL_VLM_MODEL
            )
            local_video_url_init = str(
                getattr(self.state, "agent0_local_video_url", "")
                or getattr(self.ctx.app, "vlm_local_video_url", None)
                or os.getenv("VLM_LOCAL_VIDEO_URL")
                or resolve_input_frames_source(self.ctx.app)
            )
            self.agent0_local_base_url = client.gui.add_text(
                "Local VLM base URL",
                initial_value=local_base_init,
                hint='Example: http://127.0.0.1:22002/v1',
            )
            self.agent0_local_model = client.gui.add_text(
                "Local VLM model",
                initial_value=local_model_init,
                hint='Example: Qwen/Qwen3-VL-235B-A22B-Instruct',
            )
            self.agent0_local_video_url = client.gui.add_text(
                "Local video URL (optional)",
                initial_value=local_video_url_init,
                hint=(
                    "Supports: file://video.mp4 / http(s)://video, or file://<input_dir>. "
                    "If it points to a folder, Agent0 uses image-frame mode from that directory."
                ),
            )
            self.agent0_max_frames = client.gui.add_slider(
                "Agent0 frame samples",
                min=0,
                max=50,
                step=1,
                initial_value=int(self.state.agent0_max_frames),
                hint="0 = auto (every 4 frames pick 1, ~25% of total). >0 = explicit max sampled frames.",
            )
            self.agent0_render_size = client.gui.add_slider(
                "Agent0 render size",
                min=192,
                max=768,
                step=32,
                initial_value=int(self.state.agent0_render_size),
                hint="Rendered frame size used by GPT-4o.",
            )
            self.agent1_points_per_object = client.gui.add_slider(
                "Agent1 points per object",
                min=1,
                max=8,
                step=1,
                initial_value=int(getattr(self.state, "agent1_points_per_object", 3)),
                hint="How many positive click points Agent1 asks VLM for each object.",
            )
            self.agent1_max_retries = client.gui.add_slider(
                "Agent1 max retries",
                min=1,
                max=10,
                step=1,
                initial_value=int(getattr(self.state, "agent1_max_retries", 5)),
                hint="Maximum retries for Agent1 point-plan -> segment -> evaluate loop.",
            )
            self.agent1_debug_save_eval = client.gui.add_checkbox(
                "Agent1 debug: save eval masks",
                initial_value=bool(getattr(self.state, "agent1_debug_save_eval", False)),
                hint="When enabled, each Agent1 segmentation/evaluation attempt is saved under vlm_assistor/debug_agent1_eval.",
            )
            self.save_intermediate_results = client.gui.add_checkbox(
                "Save detailed intermediate results",
                initial_value=bool(getattr(self.state, "save_intermediate_results", False)),
                hint=(
                    "When enabled, each button run creates its own folder and saves detailed "
                    "intermediate outputs/images/evaluation artifacts."
                ),
            )
            mask_eval_lang_init = str(getattr(self.state, "agent1_mask_eval_language", "auto") or "auto").strip().lower()
            if mask_eval_lang_init not in {"auto", "zh", "en"}:
                mask_eval_lang_init = "auto"
            with self.state.lock:
                self.state.agent1_mask_eval_language = mask_eval_lang_init
            self.agent1_mask_eval_lang_status = client.gui.add_text(
                "Agent1 mask eval language",
                initial_value="auto (follow model)",
                hint="Controls mask evaluator prompt language and free-text reply language.",
            )
            self.agent1_mask_eval_lang_toggle_btn = client.gui.add_button("Mask Eval Language: Switch to 中文")
            agent2_folder = client.gui.add_folder("Agent2 X-Pose", expand_by_default=False)
            with agent2_folder:
                self.agent2_x_step = client.gui.add_slider(
                    "x step",
                    min=0.01,
                    max=0.5,
                    step=0.01,
                    initial_value=float(getattr(self.state, "agent2_x_step", 0.1)),
                    hint="Translation delta per candidate step along WORLD X.",
                )
                self.agent2_num_candidates = client.gui.add_slider(
                    "candidate count",
                    min=1,
                    max=8,
                    step=1,
                    initial_value=int(getattr(self.state, "agent2_num_candidates", 4)),
                )
                self.agent2_max_rounds = client.gui.add_slider(
                    "max rounds",
                    min=1,
                    max=6,
                    step=1,
                    initial_value=int(getattr(self.state, "agent2_max_rounds", 3)),
                )
                self.agent2_x_min = client.gui.add_slider(
                    "x min bound",
                    min=-3.0,
                    max=0.0,
                    step=0.05,
                    initial_value=float(getattr(self.state, "agent2_x_min", -1.0)),
                )
                self.agent2_x_max = client.gui.add_slider(
                    "x max bound",
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    initial_value=float(getattr(self.state, "agent2_x_max", 1.0)),
                )
                self.agent2_pose_opt_steps = client.gui.add_slider(
                    "pose-opt steps",
                    min=10,
                    max=800,
                    step=10,
                    initial_value=int(getattr(self.state, "agent2_pose_opt_steps", 160)),
                )
                self.agent2_pose_opt_lr = client.gui.add_slider(
                    "pose-opt lr",
                    min=1e-3,
                    max=5e-1,
                    step=1e-3,
                    initial_value=float(getattr(self.state, "agent2_pose_opt_lr", 0.05)),
                )
                self.agent2_pose_opt_min_side = client.gui.add_slider(
                    "pose-opt min side",
                    min=128,
                    max=1024,
                    step=2,
                    initial_value=int(getattr(self.state, "agent2_pose_opt_min_side", 512)),
                )
                self.agent2_pose_opt_rgb_weight = client.gui.add_slider(
                    "pose-opt rgb weight",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=float(getattr(self.state, "agent2_pose_opt_rgb_weight", 0.4)),
                )
                self.agent2_pose_opt_timeout_sec = client.gui.add_slider(
                    "pose-opt timeout(s)",
                    min=60,
                    max=3600,
                    step=10,
                    initial_value=float(getattr(self.state, "agent2_pose_opt_timeout_sec", 1500.0)),
                )
                self.agent2_debug_save_views = client.gui.add_checkbox(
                    "Agent2 debug: save 6-view images",
                    initial_value=bool(getattr(self.state, "agent2_debug_save_views", False)),
                    hint="When enabled, Agent2 saves per-round/per-candidate 6-view renders + overlay under vlm_assistor/agent2_xpose.",
                )
            self.agent4_history_window = client.gui.add_slider(
                "Agent4 past frame window",
                min=0,
                max=30,
                step=1,
                initial_value=int(getattr(self.state, "agent4_history_window", 6)),
                hint="Each run uses current frame + past N frames.",
            )
            self.agent4_views_per_frame = client.gui.add_slider(
                "Agent4 views per frame",
                min=1,
                max=8,
                step=1,
                initial_value=int(getattr(self.state, "agent4_views_per_frame", 3)),
                hint="How many multiview images to read per frame from session_root/multi_view.",
            )
            self.agent4_use_smolagents = client.gui.add_checkbox(
                "Agent4 use smolagents refinement",
                initial_value=bool(getattr(self.state, "agent4_use_smolagents", False)),
                hint="Optional post-processing; requires smolagents + SMOLAGENTS_MODEL_ID.",
            )

            depth_folder = client.gui.add_folder("Depth scan", expand_by_default=False)
            with depth_folder:
                self.depth_min = client.gui.add_slider(
                    "min offset",
                    min=-1.0,
                    max=0.0,
                    step=0.01,
                    initial_value=float(self.state.depth_scan_min),
                )
                self.depth_max = client.gui.add_slider(
                    "max offset",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=float(self.state.depth_scan_max),
                )
                self.depth_steps = client.gui.add_slider(
                    "steps",
                    min=3,
                    max=31,
                    step=2,
                    initial_value=int(self.state.depth_scan_steps),
                )

            run_folder = client.gui.add_folder("Agents", expand_by_default=True)
            with run_folder:
                self.agent0_btn = client.gui.add_button("Run Agent 0: Video Understanding")
                self.agent1_prev_btn = client.gui.add_button("Agent1: Previous Object")
                self.agent1_next_btn = client.gui.add_button("Agent1: Next Object")
                self.agent4_btn = client.gui.add_button("Run Agent 4: 4D Memory Judge")
                self.agent1_btn = client.gui.add_button("Run Agent 1: Click Agent")
                self.agent1_eval_current_btn = client.gui.add_button("Agent1: Eval Current Mask (Manual Points)")
                self.agent2_btn = client.gui.add_button("Run Agent 2: X-Axis Pose Adjuster")
                self.agent2_v2_btn = client.gui.add_button("Run Agent 2 V2: Direct Shift + PoseOpt")
                self.agentm1_btn = client.gui.add_button("Run Agent -4: 4D Validator")
                self.pipeline_btn = client.gui.add_button("Run Full 3D Gen Pipeline")
                self.pipeline4d_btn = client.gui.add_button("Run Full 4D Gen Pipeline")

            self.status_text = client.gui.add_text("VLM Assistor status", "idle")
            self.scene_graph_text = client.gui.add_markdown("scene_graph: (empty)")
            self.scene_graph_detail_text = client.gui.add_markdown("scene_graph_json: (empty)")
            self.agent4_result_text = client.gui.add_markdown("agent4_result: (empty)")
            self.log_text = client.gui.add_markdown("(no logs)")

            self.state.status_handle = self.status_text
            self.state.scene_graph_handle = self.scene_graph_text
            self.state.scene_graph_detail_handle = self.scene_graph_detail_text
            self.state.agent4_result_handle = self.agent4_result_text
            self.state.log_handle = self.log_text

        def _sync_backend_to_app() -> None:
            app = self.ctx.app
            app.vlm_agent0_backend_mode = str(self.agent0_backend_mode.value)
            app.vlm_local_base_url = str(self.agent0_local_base_url.value).strip()
            app.vlm_local_model = str(self.agent0_local_model.value).strip()
            app.vlm_local_api_key = str(
                getattr(self.state, "agent0_local_api_key", "") or ""
            ).strip()
            app.vlm_local_video_url = str(self.agent0_local_video_url.value).strip()
            with self.state.lock:
                app.vlm_agent1_mask_eval_language = str(
                    getattr(self.state, "agent1_mask_eval_language", "auto") or "auto"
                )

        def _agent1_mask_eval_lang_label(mode: str) -> str:
            s = str(mode or "auto").strip().lower()
            if s == "zh":
                return "zh (中文)"
            if s == "en":
                return "en (English)"
            return "auto (follow model)"

        def _refresh_mask_eval_language_widgets() -> None:
            with self.state.lock:
                current = str(getattr(self.state, "agent1_mask_eval_language", "auto") or "auto").strip().lower()
            if current not in {"auto", "zh", "en"}:
                current = "auto"
            try:
                self.agent1_mask_eval_lang_status.value = _agent1_mask_eval_lang_label(current)
            except Exception:
                pass
            next_label = "中文" if current != "zh" else "English"
            try:
                self.agent1_mask_eval_lang_toggle_btn.label = f"Mask Eval Language: Switch to {next_label}"
            except Exception:
                pass

        def _refresh_backend_widgets() -> None:
            use_local = str(self.agent0_backend_mode.value) == "local_openai_compatible"
            try:
                self.agent0_use_gpt4o.disabled = bool(use_local)
                self.agent0_local_base_url.disabled = (not use_local)
                self.agent0_local_model.disabled = (not use_local)
                self.agent0_local_video_url.disabled = (not use_local)
            except Exception:
                pass

        _sync_backend_to_app()
        _refresh_backend_widgets()
        _refresh_mask_eval_language_widgets()

        @self.enable_toggle.on_update
        def _(_evt):
            with self.state.lock:
                self.state.enabled = bool(self.enable_toggle.value)

        @self.object_text.on_update
        def _(_evt):
            with self.state.lock:
                self.state.object_description = str(self.object_text.value)

        @self.relation_text.on_update
        def _(_evt):
            with self.state.lock:
                self.state.relation_hint = str(self.relation_text.value)

        @self.agent0_backend_mode.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent0_backend_mode = str(self.agent0_backend_mode.value)
            _sync_backend_to_app()
            _refresh_backend_widgets()

        @self.agent0_use_gpt4o.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent0_use_gpt4o = bool(self.agent0_use_gpt4o.value)

        @self.agent0_local_base_url.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent0_local_base_url = str(self.agent0_local_base_url.value).strip()
            _sync_backend_to_app()

        @self.agent0_local_model.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent0_local_model = str(self.agent0_local_model.value).strip()
            _sync_backend_to_app()

        @self.agent0_local_video_url.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent0_local_video_url = str(self.agent0_local_video_url.value).strip()
            _sync_backend_to_app()

        @self.agent0_max_frames.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent0_max_frames = int(self.agent0_max_frames.value)

        @self.agent0_render_size.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent0_render_size = int(self.agent0_render_size.value)

        @self.agent1_points_per_object.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent1_points_per_object = int(self.agent1_points_per_object.value)

        @self.agent1_max_retries.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent1_max_retries = int(self.agent1_max_retries.value)

        @self.agent1_debug_save_eval.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent1_debug_save_eval = bool(self.agent1_debug_save_eval.value)

        @self.save_intermediate_results.on_update
        def _(_evt):
            with self.state.lock:
                self.state.save_intermediate_results = bool(self.save_intermediate_results.value)

        @self.agent2_x_step.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_x_step = float(self.agent2_x_step.value)

        @self.agent2_num_candidates.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_num_candidates = int(self.agent2_num_candidates.value)

        @self.agent2_max_rounds.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_max_rounds = int(self.agent2_max_rounds.value)

        @self.agent2_x_min.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_x_min = float(self.agent2_x_min.value)

        @self.agent2_x_max.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_x_max = float(self.agent2_x_max.value)

        @self.agent2_pose_opt_steps.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_pose_opt_steps = int(self.agent2_pose_opt_steps.value)

        @self.agent2_pose_opt_lr.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_pose_opt_lr = float(self.agent2_pose_opt_lr.value)

        @self.agent2_pose_opt_min_side.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_pose_opt_min_side = int(self.agent2_pose_opt_min_side.value)

        @self.agent2_pose_opt_rgb_weight.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_pose_opt_rgb_weight = float(self.agent2_pose_opt_rgb_weight.value)

        @self.agent2_pose_opt_timeout_sec.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_pose_opt_timeout_sec = float(self.agent2_pose_opt_timeout_sec.value)

        @self.agent2_debug_save_views.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent2_debug_save_views = bool(self.agent2_debug_save_views.value)

        def _toggle_agent1_mask_eval_language(_evt) -> None:
            with self.state.lock:
                cur = str(getattr(self.state, "agent1_mask_eval_language", "auto") or "auto").strip().lower()
                nxt = "en" if cur == "zh" else "zh"
                self.state.agent1_mask_eval_language = nxt
            _sync_backend_to_app()
            _refresh_mask_eval_language_widgets()
            try:
                client.add_notification(
                    "Agent1",
                    f"Mask evaluator language switched to {_agent1_mask_eval_lang_label(nxt)}",
                    color="blue",
                    auto_close_seconds=3,
                )
            except Exception:
                pass

        self.agent1_mask_eval_lang_toggle_btn.on_click(_toggle_agent1_mask_eval_language)

        @self.agent4_history_window.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent4_history_window = int(self.agent4_history_window.value)

        @self.agent4_views_per_frame.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent4_views_per_frame = int(self.agent4_views_per_frame.value)

        @self.agent4_use_smolagents.on_update
        def _(_evt):
            with self.state.lock:
                self.state.agent4_use_smolagents = bool(self.agent4_use_smolagents.value)

        @self.depth_min.on_update
        def _(_evt):
            with self.state.lock:
                self.state.depth_scan_min = float(self.depth_min.value)

        @self.depth_max.on_update
        def _(_evt):
            with self.state.lock:
                self.state.depth_scan_max = float(self.depth_max.value)

        @self.depth_steps.on_update
        def _(_evt):
            with self.state.lock:
                self.state.depth_scan_steps = int(self.depth_steps.value)

        self.agent0_btn.on_click(
            lambda _evt: self.ops.run_video_agent(
                client,
                object_text=str(self.object_text.value),
                relation_hint=str(self.relation_text.value),
            )
        )
        self.agent1_prev_btn.on_click(lambda _evt: self.ops.select_prev_agent1_object(client))
        self.agent1_next_btn.on_click(lambda _evt: self.ops.select_next_agent1_object(client))
        self.agent4_btn.on_click(
            lambda _evt: self.ops.run_4d_memory_judge(
                client,
                window_size=int(self.agent4_history_window.value),
            )
        )
        self.agent1_btn.on_click(
            lambda _evt: self.ops.run_click_agent(client, object_text=str(self.object_text.value))
        )
        self.agent1_eval_current_btn.on_click(
            lambda _evt: self.ops.run_click_agent_evaluate_current(client, object_text=str(self.object_text.value))
        )
        self.agent2_btn.on_click(
            lambda _evt: self.ops.run_agent2_xpose_adjuster(client, object_name=str(self.object_text.value))
        )
        self.agent2_v2_btn.on_click(
            lambda _evt: self.ops.run_agent2_xpose_adjuster_v2(client, object_name=str(self.object_text.value))
        )
        self.agentm1_btn.on_click(
            lambda _evt: self.ops.run_4d_validator(
                client,
                window_size=int(self.agent4_history_window.value),
            )
        )
        self.pipeline_btn.on_click(
            lambda _evt: self.ops.run_full_pipeline(
                client,
                object_text=str(self.object_text.value),
                relation_hint=str(self.relation_text.value),
            )
        )
        self.pipeline4d_btn.on_click(
            lambda _evt: self.ops.run_full_4d_pipeline(
                client,
                object_text=str(self.object_text.value),
                relation_hint=str(self.relation_text.value),
            )
        )

        self.ops.sync_ui()
