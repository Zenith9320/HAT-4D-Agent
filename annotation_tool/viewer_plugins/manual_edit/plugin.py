# plugins/manual_edit/plugin.py
from .context import ViewerContext
from .state import ManualEditState
from .ops import ManualEditOps
from .ui import ManualEditUI
from utils.viser_compat import register_key_handler


class ManualEditPlugin:
    def __init__(self, app):
        self.app = app
        self.ctx = ViewerContext(app)
        self.state = ManualEditState()
        self.ops = ManualEditOps(self.ctx, self.state)
        self.ui = ManualEditUI(self.ctx, self.ops, self.state)

    def install(self, client, workspace):
        self.ui.build(client)
        self.workspace = workspace

        def _handle_key(event):
            if event.event_type != "keydown" or event.repeat:
                return

            key_display = event.key
            if key_display == " ":
                key_display = "Space"
            if self.state.key_status_handle is not None:
                try:
                    self.state.key_status_handle.value = f"Click Key '{key_display}'"
                except Exception:
                    pass

            action = None
            key = event.key.lower()
            if key == "u":
                action = "Add Selection (Union)"
                self.ops.add_selection(client)
            elif key == "i":
                action = "Intersect Selection"
                self.ops.intersect_selection(client)
            elif key in ("-", "_") or event.code == "Minus":
                action = "Subtract Selection"
                self.ops.subtract_selection(client)
            elif key == "c":
                action = "Clear Selection"
                self.ops.clear_selection(client)
            elif key == "b":
                new_visible = not bool(self.state.bbox_visible)
                action = f"Toggle BBox Gizmo ({'On' if new_visible else 'Off'})"
                self.ops.set_bbox_visible(client, new_visible)
                try:
                    self.ui.show_bbox_toggle.value = new_visible
                except Exception:
                    pass
            elif key == "a":
                action = "Apply Transform"
                self.ops.apply_transform(client)
            elif key in ("1", "2", "3"):
                if not self.state.bbox_visible:
                    action = "Transform Mode (requires BBox)"
                else:
                    mode = {"1": "Translate", "2": "Scale", "3": "Rotate"}[key]
                    action = f"Transform Mode: {mode}"
                    self.ops.update_transform_mode(mode)
                    try:
                        self.ui.transform_mode.value = mode
                    except Exception:
                        pass

            if self.state.action_status_handle is not None:
                try:
                    self.state.action_status_handle.value = action or "No action"
                except Exception:
                    pass

        register_key_handler(client, _handle_key)
