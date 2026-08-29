from __future__ import annotations

from .context import ViewerContext
from .ops import Sam3DGenOps
from .state import Sam3DGenState
from .ui import Sam3DGenUI


class Sam3DGenPlugin:
    def __init__(self, app):
        self.app = app
        self.ctx = ViewerContext(app)
        self.state = Sam3DGenState()
        try:
            self.state.display_max_points = int(getattr(app, "gs_display_max_points", self.state.display_max_points))
        except Exception:
            pass
        try:
            self.state.display_sample = str(getattr(app, "gs_display_sample", self.state.display_sample))
        except Exception:
            pass
        self.ops = Sam3DGenOps(self.ctx, self.state)
        self.ui = Sam3DGenUI(self.ctx, self.ops, self.state)

    def install(self, client, workspace, parent_folder=None):
        self.workspace = workspace
        self.ui.build(client, parent_folder=parent_folder)
        self.ops.install_client_hooks(client)
