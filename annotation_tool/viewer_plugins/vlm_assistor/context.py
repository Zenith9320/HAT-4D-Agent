from __future__ import annotations


class ViewerContext:
    def __init__(self, app, sam3d_plugin):
        self.app = app
        self.sam3d_plugin = sam3d_plugin

    @property
    def sam3d_ops(self):
        try:
            return self.sam3d_plugin.ops
        except Exception:
            return None

    @property
    def sam3d_state(self):
        try:
            return self.sam3d_plugin.state
        except Exception:
            return None
