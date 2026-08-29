from __future__ import annotations


class RecordOpsUI:
    def __init__(self, ctx, ops, state):
        self.ctx = ctx
        self.ops = ops
        self.state = state
        self.folder = None

    def build(self, client, parent_folder=None):
        if parent_folder is None:
            folder = client.gui.add_folder("Record Ops", expand_by_default=False)
        else:
            with parent_folder:
                folder = client.gui.add_folder("Record Ops", expand_by_default=False)
        self.folder = folder

        with folder:
            self.recording_toggle = client.gui.add_checkbox(
                "Enable Recording",
                initial_value=bool(self.state.recording_enabled),
                hint="When enabled, records SAM3D operations for VLM training data."
            )
            self.status_text = client.gui.add_text("Status", "idle")
            self.state.toggle_handle = self.recording_toggle
            self.state.status_handle = self.status_text

        @self.recording_toggle.on_update
        def _(_evt):
            enabled = bool(self.recording_toggle.value)
            self.ops.set_recording_enabled(enabled)
            self._update_status()

        self._update_status()

    def _update_status(self):
        try:
            with self.state.lock:
                enabled = self.state.recording_enabled
                active = self.state.session_active
                obj_count = len(self.state.objects)

            if not enabled:
                status = "disabled"
            elif not active:
                status = "enabled (waiting)"
            else:
                status = f"recording: {obj_count} objs"

            if self.state.status_handle is not None:
                self.state.status_handle.value = status
        except Exception:
            pass

    def update_status(self):
        self._update_status()
