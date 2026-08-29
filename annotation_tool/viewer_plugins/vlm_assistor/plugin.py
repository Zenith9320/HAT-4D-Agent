from __future__ import annotations

from pathlib import Path

from .config import build_defaults
from .context import ViewerContext
from .ops import VLMAssistorOps
from .state import VLMAssistorState
from .ui import VLMAssistorUI


class VLMAssistorPlugin:
    def __init__(self, app, sam3d_plugin):
        self.app = app
        self.ctx = ViewerContext(app, sam3d_plugin)
        defaults = build_defaults(app)
        self.state = VLMAssistorState(defaults=defaults)
        self.app.vlm_agent0_backend_mode = self.state.agent0_backend_mode
        self.app.vlm_local_base_url = self.state.agent0_local_base_url
        self.app.vlm_local_model = self.state.agent0_local_model
        self.app.vlm_local_api_key = self.state.agent0_local_api_key
        self.app.vlm_local_video_url = self.state.agent0_local_video_url
        self.app.vlm_agent1_mask_eval_language = self.state.agent1_mask_eval_language
        self.app.use_pose_adjust_v2 = bool(self.state.use_pose_adjust_v2)
        self.ops = VLMAssistorOps(self.ctx, self.state)
        self.ui = VLMAssistorUI(self.ctx, self.ops, self.state)

    def install(self, client, workspace, parent_folder=None):
        self.workspace = workspace
        base_root = Path(workspace.root) / "vlm_assistor"
        with self.state.lock:
            self.state.artifact_base_root = base_root
            self.state.artifact_root = base_root
        self.ui.build(client, parent_folder=parent_folder)
