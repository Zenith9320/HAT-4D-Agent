# plugins/manual_edit/ui.py

class ManualEditUI:
    def __init__(self, ctx, ops, state):
        self.ctx = ctx
        self.ops = ops
        self.state = state
        self.root_folder = None

    def build(self, client):
        folder = client.gui.add_folder("Edit GS (Manual)", expand_by_default=True)
        self.root_folder = folder

        with folder:
            selection_folder = client.gui.add_folder("Selection", expand_by_default=True)
            with selection_folder:
                self.start_select_btn = client.gui.add_button("Start Box Select")
                self.union_btn = client.gui.add_button("Add Selection (Union)")
                self.intersect_btn = client.gui.add_button("Intersect Selection")
                self.subtract_btn = client.gui.add_button("Subtract Selection")
                self.clear_btn = client.gui.add_button("Clear Selection")
                self.show_overlay_toggle = client.gui.add_checkbox(
                    "Show Selection Overlay", initial_value=False
                )
                self.selection_status = client.gui.add_text("Selection Status", "None")
                self.state.selection_status_handle = self.selection_status
                self.key_status = client.gui.add_text("Last key", "None")
                self.state.key_status_handle = self.key_status
                self.action_status = client.gui.add_text("Last action", "None")
                self.state.action_status_handle = self.action_status

            attrs_folder = client.gui.add_folder("Edit Attributes", expand_by_default=False)
            with attrs_folder:
                self.color_r = client.gui.add_slider(
                    "Color R", min=0.0, max=1.0, step=0.01, initial_value=1.0
                )
                self.color_g = client.gui.add_slider(
                    "Color G", min=0.0, max=1.0, step=0.01, initial_value=1.0
                )
                self.color_b = client.gui.add_slider(
                    "Color B", min=0.0, max=1.0, step=0.01, initial_value=1.0
                )
                self.opacity_slider = client.gui.add_slider(
                    "Opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0
                )
                self.scale_x = client.gui.add_slider(
                    "Scale X", min=0.01, max=10.0, step=0.01, initial_value=1.0
                )
                self.scale_y = client.gui.add_slider(
                    "Scale Y", min=0.01, max=10.0, step=0.01, initial_value=1.0
                )
                self.scale_z = client.gui.add_slider(
                    "Scale Z", min=0.01, max=10.0, step=0.01, initial_value=1.0
                )
                self.uniform_scale = client.gui.add_slider(
                    "Uniform Scale", min=0.01, max=10.0, step=0.01, initial_value=1.0
                )
                self.rot_x = client.gui.add_slider(
                    "Rotation X", min=-180.0, max=180.0, step=1.0, initial_value=0.0
                )
                self.rot_y = client.gui.add_slider(
                    "Rotation Y", min=-180.0, max=180.0, step=1.0, initial_value=0.0
                )
                self.rot_z = client.gui.add_slider(
                    "Rotation Z", min=-180.0, max=180.0, step=1.0, initial_value=0.0
                )
                self.sh_intensity = client.gui.add_slider(
                    "SH intensity", min=0.0, max=2.0, step=0.01, initial_value=1.0
                )
                self.apply_attrs_btn = client.gui.add_button("Apply Attributes")

            delete_folder = client.gui.add_folder("Delete", expand_by_default=False)
            with delete_folder:
                self.delete_selected_btn = client.gui.add_button("Delete Selected GS")
                self.delete_unselected_btn = client.gui.add_button(
                    "Invert Delete (Delete Unselected)"
                )

            bbox_folder = client.gui.add_folder("BBox Transform", expand_by_default=False)
            with bbox_folder:
                self.gen_bbox_btn = client.gui.add_button("Generate BBox from Selection")
                self.show_bbox_toggle = client.gui.add_checkbox(
                    "Show BBox Gizmo", initial_value=False
                )
                self.transform_mode = client.gui.add_dropdown(
                    "Transform Mode",
                    options=["Translate", "Scale", "Rotate"],
                    initial_value="Translate",
                )
                self.axis_mode = client.gui.add_dropdown(
                    "Axis Mode",
                    options=["World", "Local"],
                    initial_value="World",
                )
                self.axis_x = client.gui.add_checkbox("Enable Axis Constraint X", False)
                self.axis_y = client.gui.add_checkbox("Enable Axis Constraint Y", False)
                self.axis_z = client.gui.add_checkbox("Enable Axis Constraint Z", False)
                self.apply_transform_btn = client.gui.add_button("Apply Transform")

            history_folder = client.gui.add_folder("History", expand_by_default=False)
            with history_folder:
                self.undo_btn = client.gui.add_button("Undo")
                self.redo_btn = client.gui.add_button("Redo")
                self.snapshot_btn = client.gui.add_button("Save Snapshot")

        if hasattr(self.ops, "start_box_select"):
            self.start_select_btn.on_click(lambda _evt: self.ops.start_box_select(client))
        if hasattr(self.ops, "add_selection"):
            self.union_btn.on_click(lambda _evt: self.ops.add_selection(client))
        if hasattr(self.ops, "intersect_selection"):
            self.intersect_btn.on_click(lambda _evt: self.ops.intersect_selection(client))
        if hasattr(self.ops, "subtract_selection"):
            self.subtract_btn.on_click(lambda _evt: self.ops.subtract_selection(client))
        if hasattr(self.ops, "clear_selection"):
            self.clear_btn.on_click(lambda _evt: self.ops.clear_selection(client))
        if hasattr(self.ops, "toggle_selection_overlay"):
            @self.show_overlay_toggle.on_update
            def _(_evt):
                self.ops.toggle_selection_overlay(client, bool(self.show_overlay_toggle.value))
        if hasattr(self.ops, "delete_selected"):
            self.delete_selected_btn.on_click(lambda _evt: self.ops.delete_selected(client))
        if hasattr(self.ops, "delete_unselected"):
            self.delete_unselected_btn.on_click(lambda _evt: self.ops.delete_unselected(client))
        if hasattr(self.ops, "apply_attributes"):
            self.apply_attrs_btn.on_click(
                lambda _evt: self.ops.apply_attributes(
                    client,
                    (self.color_r.value, self.color_g.value, self.color_b.value),
                    self.opacity_slider.value,
                    (self.scale_x.value, self.scale_y.value, self.scale_z.value),
                    self.uniform_scale.value,
                    (self.rot_x.value, self.rot_y.value, self.rot_z.value),
                    self.sh_intensity.value,
                )
            )
        if hasattr(self.ops, "generate_bbox_from_selection"):
            self.gen_bbox_btn.on_click(lambda _evt: self.ops.generate_bbox_from_selection(client))
        if hasattr(self.ops, "set_bbox_visible"):
            @self.show_bbox_toggle.on_update
            def _(_evt):
                self.ops.set_bbox_visible(client, bool(self.show_bbox_toggle.value))
        if hasattr(self.ops, "apply_transform"):
            self.apply_transform_btn.on_click(lambda _evt: self.ops.apply_transform(client))
        if hasattr(self.ops, "update_transform_mode"):
            @self.transform_mode.on_update
            def _(_evt):
                self.ops.update_transform_mode(str(self.transform_mode.value))
        if hasattr(self.ops, "update_axis_mode"):
            @self.axis_mode.on_update
            def _(_evt):
                self.ops.update_axis_mode(str(self.axis_mode.value))
        if hasattr(self.ops, "update_axis_constraint"):
            @self.axis_x.on_update
            def _(_evt):
                self.ops.update_axis_constraint(
                    bool(self.axis_x.value),
                    bool(self.axis_y.value),
                    bool(self.axis_z.value),
                )
            @self.axis_y.on_update
            def _(_evt):
                self.ops.update_axis_constraint(
                    bool(self.axis_x.value),
                    bool(self.axis_y.value),
                    bool(self.axis_z.value),
                )
            @self.axis_z.on_update
            def _(_evt):
                self.ops.update_axis_constraint(
                    bool(self.axis_x.value),
                    bool(self.axis_y.value),
                    bool(self.axis_z.value),
                )
        if hasattr(self.ops, "undo"):
            self.undo_btn.on_click(lambda _evt: self.ops.undo(client))
        if hasattr(self.ops, "redo"):
            self.redo_btn.on_click(lambda _evt: self.ops.redo(client))
        if hasattr(self.ops, "save_snapshot"):
            self.snapshot_btn.on_click(lambda _evt: self.ops.save_snapshot(client))
