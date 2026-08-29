from __future__ import annotations

import numpy as np


class Sam3DGenUI:
    def __init__(self, ctx, ops, state):
        self.ctx = ctx
        self.ops = ops
        self.state = state
        self.folder = None
        self._local_active = False

    def build(self, client, parent_folder=None):
        if parent_folder is None:
            folder = client.gui.add_folder("SAM 3D Gen", expand_by_default=True)
        else:
            with parent_folder:
                folder = client.gui.add_folder("SAM 3D Gen", expand_by_default=True)
        self.folder = folder

        with folder:
            # Label is client-local; keep it consistent with state when possible.
            initial_label = "End SAM3D Gen" if bool(getattr(self.state, "gen_active", False)) else "Begin SAM3D Gen"
            self._local_active = bool(getattr(self.state, "gen_active", False))
            self.begin_btn = client.gui.add_button(initial_label)
            self.total_cover_toggle = client.gui.add_checkbox("Total cover (clear session PLY on begin)", initial_value=False)
            self.pad_square_toggle = client.gui.add_checkbox(
                "Pad image to square (frustum + SAM3D)", initial_value=True
            )
            self.pose_opt_after_gen_toggle = client.gui.add_checkbox(
                "Pose optimize after GEN", initial_value=True
            )
            self.interaction_mode = client.gui.add_dropdown(
                "Interaction Mode",
                options=["Segment (SAM)", "Edit Transform"],
                initial_value="Segment (SAM)",
            )
            self.frustum_opacity = client.gui.add_slider(
                "Frustum Opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0
            )
            self.display_max_points_slider = client.gui.add_slider(
                "SAM3D Display Max Points",
                min=0,
                max=2_000_000,
                step=1_000,
                initial_value=int(getattr(self.state, "display_max_points", 150_000)),
                hint="0 disables downsample for generated objects.",
            )
            self.display_sample_mode = client.gui.add_dropdown(
                "SAM3D Display Sample",
                options=["opacity", "random"],
                initial_value=str(getattr(self.state, "display_sample", "opacity")),
            )

            self.pick_mode = client.gui.add_checkbox("Pixel Pick Mode", initial_value=False)
            self.negative_toggle = client.gui.add_checkbox("Negative Point", initial_value=False)
            self.eraser_toggle = client.gui.add_checkbox("Eraser Point", initial_value=False)
            self.eraser_radius = client.gui.add_slider(
                "Eraser Radius (px)", min=3, max=80, step=1, initial_value=18
            )

            obj_folder = client.gui.add_folder("Object", expand_by_default=True)
            with obj_folder:
                self.new_obj_btn = client.gui.add_button("New Object")
                self.prev_obj_btn = client.gui.add_button("Prev Object")
                self.next_obj_btn = client.gui.add_button("Next Object")
                self.undo_point_btn = client.gui.add_button("Undo Point")
                self.clear_points_btn = client.gui.add_button("Clear Points")
                self.generate_btn = client.gui.add_button("Generate 3D (Current Mask)")
                self.add_object_btn = client.gui.add_button(
                    "Add All Objects & Exit",
                    hint="Append ALL objects' GS into the current frame's session PLY, then auto-exit SAM3D Gen mode.",
                )

                self.object_status = client.gui.add_text("Current object", "obj=0")
                self.mask_status = client.gui.add_text("Mask", "mask=None")
                self.gen_status = client.gui.add_text("Generation", "idle")
                self.state.object_status_handle = self.object_status
                self.state.mask_status_handle = self.mask_status
                self.state.gen_status_handle = self.gen_status

            transform_folder = client.gui.add_folder("Transform", expand_by_default=False)
            with transform_folder:
                self.show_bbox_toggle = client.gui.add_checkbox("Show BBox Gizmo", initial_value=False)
                self.transform_mode = client.gui.add_dropdown(
                    "Transform Mode",
                    options=["Translate", "Rotate", "Scale"],
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
                self.reset_transform_btn = client.gui.add_button("Reset Transform")
                # Scale controls (moved from Pose Opt folder)
                self.manual_scale_text = client.gui.add_text("Object Scale", "1.000")
                self.scale_down_btn = client.gui.add_button("Scale -", hint="Click: step down; Hold: continuous")
                self.scale_up_btn = client.gui.add_button("Scale +", hint="Click: step up; Hold: continuous")

            poseopt_folder = client.gui.add_folder("Pose Opt", expand_by_default=False)
            with poseopt_folder:
                self.pose_opt_steps = client.gui.add_slider(
                    "Pose opt steps", min=10, max=800, step=10, initial_value=160
                )
                self.pose_opt_lr = client.gui.add_slider(
                    "Pose opt lr", min=1e-2, max=5e-1, step=1e-5, initial_value=5e-2
                )
                self.pose_opt_min_side = client.gui.add_slider(
                    "Pose opt render min side", min=128, max=1024, step=2, initial_value=512
                )
                self.pose_opt_rgb_weight = client.gui.add_slider(
                    "Pose opt rgb weight", min=0.0, max=1.0, step=0.01, initial_value=0.4
                )
                self.pose_opt_center_reg = client.gui.add_slider(
                    "Pose opt center reg", min=0.0, max=1e-1, step=1e-4, initial_value=1e-2
                )
                self.pose_opt_pose_reg = client.gui.add_slider(
                    "Pose opt pose reg", min=0.0, max=1e-2, step=1e-5, initial_value=1e-3
                )
                self.pose_opt_scale_reg = client.gui.add_slider(
                    "Pose opt scale reg", min=0.0, max=1e-3, step=1e-6, initial_value=1e-5
                )
                self.pose_opt_box_reg = client.gui.add_slider(
                    "Pose opt box reg", min=0.0, max=1.0, step=1e-3, initial_value=1e-1
                )
                self.pose_opt_debug = client.gui.add_checkbox("Pose opt debug", initial_value=False)
                self.pose_opt_optimize_rotation = client.gui.add_checkbox(
                    "Pose opt optimize rotation",
                    initial_value=bool(getattr(self.state, "pose_opt_optimize_rotation", False)),
                )
                self.pose_opt_lock_tx = client.gui.add_checkbox("Pose opt hard lock world X", initial_value=True)
                self.pose_opt_lock_ty = client.gui.add_checkbox("Pose opt hard lock world Y", initial_value=False)
                self.pose_opt_lock_tz = client.gui.add_checkbox("Pose opt hard lock world Z", initial_value=False)
                self.pose_opt_lock_debug = client.gui.add_text("Pose opt lock debug", "n/a")
                # Preview the final mask overlay after backend pose optimization.
                self.pose_opt_overlay = client.gui.add_image(
                    np.zeros((256, 256, 3), dtype=np.uint8),
                    label="PoseOpt mask overlay",
                )
                try:
                    self.state.pose_opt_overlay_handles[client] = self.pose_opt_overlay
                    self.state.pose_opt_debug_handles[client] = self.pose_opt_lock_debug
                except Exception:
                    pass
                self.pose_opt_btn = client.gui.add_button("Optimize Pose (Mask Align)")

            self.status = client.gui.add_text("SAM3D status", "idle")
            self.state.status_handle = self.status
            self.sam_status = client.gui.add_text("SAM status", "uninitialized")
            self.sam3d_status = client.gui.add_text("SAM3D pipeline", "uninitialized")
            self.state.sam_status_handle = self.sam_status
            self.state.sam3d_status_handle = self.sam3d_status

        # Wire callbacks
        def _set_active(active: bool) -> None:
            self._local_active = bool(active)
            print(f"Setting local active: {self._local_active}")
            self.begin_btn.label = "End SAM3D Gen" if self._local_active else "Begin SAM3D Gen"

        def _sync_pose_opt_settings() -> None:
            # Keep state in sync so auto pose-opt (after GEN) uses current UI values.
            try:
                with self.state.lock:
                    self.state.pose_opt_steps = int(self.pose_opt_steps.value)
                    self.state.pose_opt_lr = float(self.pose_opt_lr.value)
                    self.state.pose_opt_min_side = int(self.pose_opt_min_side.value)
                    self.state.pose_opt_rgb_weight = float(self.pose_opt_rgb_weight.value)
                    self.state.pose_opt_center_reg_weight = float(self.pose_opt_center_reg.value)
                    self.state.pose_opt_pose_reg_weight = float(self.pose_opt_pose_reg.value)
                    self.state.pose_opt_scale_reg_weight = float(self.pose_opt_scale_reg.value)
                    self.state.pose_opt_box_reg_weight = float(self.pose_opt_box_reg.value)
                    self.state.pose_opt_debug_mode = bool(self.pose_opt_debug.value)
                    self.state.pose_opt_optimize_rotation = bool(self.pose_opt_optimize_rotation.value)
                    self.state.pose_opt_lock_tx = bool(self.pose_opt_lock_tx.value)
                    self.state.pose_opt_lock_ty = bool(self.pose_opt_lock_ty.value)
                    self.state.pose_opt_lock_tz = bool(self.pose_opt_lock_tz.value)
            except Exception:
                pass

        def _reset_controls_for_end() -> None:
            # Keep UI + ops consistent when leaving the mode.
            self.pick_mode.value = False
            self.negative_toggle.value = False
            self.eraser_toggle.value = False
            self.show_bbox_toggle.value = False
            self.interaction_mode.value = "Segment (SAM)"

        def _sync_transform_controls_from_ui() -> None:
            self.ops.update_transform_mode(str(self.transform_mode.value))
            self.ops.update_axis_mode(str(self.axis_mode.value))
            self.ops.update_axis_constraint(
                bool(self.axis_x.value),
                bool(self.axis_y.value),
                bool(self.axis_z.value),
            )

        def _toggle_begin_end(_evt) -> None:
            if not self._local_active:
                try:
                    with self.state.lock:
                        self.state.total_cover = bool(self.total_cover_toggle.value)
                        self.state.pad_square = bool(self.pad_square_toggle.value)
                        self.state.pose_opt_after_gen = bool(self.pose_opt_after_gen_toggle.value)
                except Exception:
                    pass
                print("Beginning SAM3D Gen mode")
                self.ops.begin(client)
                _sync_transform_controls_from_ui()
                print("After ops.begin()")
                _set_active(True)
            else:
                self.ops.end(client)
                _reset_controls_for_end()
                _set_active(False)


        self.begin_btn.on_click(_toggle_begin_end)

        @self.total_cover_toggle.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.total_cover = bool(self.total_cover_toggle.value)
            except Exception:
                pass

        @self.pose_opt_after_gen_toggle.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.pose_opt_after_gen = bool(self.pose_opt_after_gen_toggle.value)
            except Exception:
                pass

        @self.pad_square_toggle.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.pad_square = bool(self.pad_square_toggle.value)
            except Exception:
                pass

        @self.pose_opt_lock_tx.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.pose_opt_lock_tx = bool(self.pose_opt_lock_tx.value)
            except Exception:
                pass

        @self.pose_opt_lock_ty.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.pose_opt_lock_ty = bool(self.pose_opt_lock_ty.value)
            except Exception:
                pass

        @self.pose_opt_lock_tz.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.pose_opt_lock_tz = bool(self.pose_opt_lock_tz.value)
            except Exception:
                pass

        @self.pose_opt_optimize_rotation.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.pose_opt_optimize_rotation = bool(self.pose_opt_optimize_rotation.value)
            except Exception:
                pass

        def _set_manual_scale_text(v: float) -> None:
            try:
                self.manual_scale_text.value = f"{float(v):.3f}"
            except Exception:
                pass

        def _nudge_manual_scale(mult: float) -> None:
            try:
                v = self.ops.nudge_manual_object_scale(client, float(mult))
                _set_manual_scale_text(v)
            except Exception as exc:
                try:
                    client.add_notification(
                        "SAM3D",
                        f"Manual scale failed: {exc}",
                        color="red",
                        auto_close_seconds=6,
                    )
                except Exception:
                    pass

        self.scale_down_btn.on_click(lambda _evt: _nudge_manual_scale(0.95))
        self.scale_up_btn.on_click(lambda _evt: _nudge_manual_scale(1.05))

        try:
            self.scale_down_btn.on_hold(lambda _evt: _nudge_manual_scale(0.995), callback_hz=20.0)
            self.scale_up_btn.on_hold(lambda _evt: _nudge_manual_scale(1.005), callback_hz=20.0)
        except Exception:
            pass

        @self.interaction_mode.on_update
        def _(_evt):
            mode = str(self.interaction_mode.value)
            self.ops.set_interaction_mode("transform" if "Transform" in mode else "segment")

        @self.frustum_opacity.on_update
        def _(_evt):
            self.ops.set_frustum_opacity(float(self.frustum_opacity.value))

        def _refresh_object_display() -> None:
            ensure_fn = getattr(self.ops, "_ensure_object_handle", None)
            if not callable(ensure_fn):
                return
            try:
                with self.state.lock:
                    obj_indices = list(range(len(getattr(self.state, "objects", []) or [])))
            except Exception:
                obj_indices = []
            for i in obj_indices:
                try:
                    ensure_fn(int(i))
                except Exception:
                    pass

        @self.display_max_points_slider.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.display_max_points = int(self.display_max_points_slider.value)
            except Exception:
                pass
            _refresh_object_display()

        @self.display_sample_mode.on_update
        def _(_evt):
            try:
                with self.state.lock:
                    self.state.display_sample = str(self.display_sample_mode.value)
            except Exception:
                pass
            _refresh_object_display()


        @self.pick_mode.on_update
        def _(_evt):
            self.ops.set_pixel_pick_mode(client, bool(self.pick_mode.value))

        @self.negative_toggle.on_update
        def _(_evt):
            self.ops.set_negative_point(bool(self.negative_toggle.value))

        @self.eraser_toggle.on_update
        def _(_evt):
            self.ops.set_eraser_mode(client, bool(self.eraser_toggle.value))

        @self.eraser_radius.on_update
        def _(_evt):
            self.ops.set_eraser_radius(int(self.eraser_radius.value))

        def _add_all_objects_and_update_ui(_evt) -> None:
            self.ops.add_object_to_session_ply(client)
            # Since add_object_to_session_ply now auto-exits, update UI state accordingly.
            _reset_controls_for_end()
            _set_active(False)

        self.new_obj_btn.on_click(lambda _evt: (self.ops.new_object(client), _sync_transform_controls_from_ui()))
        self.prev_obj_btn.on_click(lambda _evt: (self.ops.select_prev_object(client), _sync_transform_controls_from_ui()))
        self.next_obj_btn.on_click(lambda _evt: (self.ops.select_next_object(client), _sync_transform_controls_from_ui()))
        self.undo_point_btn.on_click(lambda _evt: self.ops.undo_point(client))
        self.clear_points_btn.on_click(lambda _evt: self.ops.clear_points(client))
        self.generate_btn.on_click(lambda _evt: (_sync_pose_opt_settings(), self.ops.generate_current_object_async(client)))
        self.add_object_btn.on_click(_add_all_objects_and_update_ui)

        @self.show_bbox_toggle.on_update
        def _(_evt):
            self.ops.set_bbox_visible(client, bool(self.show_bbox_toggle.value))

        @self.transform_mode.on_update
        def _(_evt):
            self.ops.update_transform_mode(str(self.transform_mode.value))

        @self.axis_mode.on_update
        def _(_evt):
            self.ops.update_axis_mode(str(self.axis_mode.value))

        def _update_constraints():
            self.ops.update_axis_constraint(
                bool(self.axis_x.value), bool(self.axis_y.value), bool(self.axis_z.value)
            )

        @self.axis_x.on_update
        def _(_evt):
            _update_constraints()

        @self.axis_y.on_update
        def _(_evt):
            _update_constraints()

        @self.axis_z.on_update
        def _(_evt):
            _update_constraints()

        self.apply_transform_btn.on_click(lambda _evt: self.ops.apply_transform(client))
        self.reset_transform_btn.on_click(lambda _evt: self.ops.reset_transform(client))

        self.pose_opt_btn.on_click(
            lambda _evt: self.ops.optimize_pose_current_object_async(
                client,
                steps=int(self.pose_opt_steps.value),
                lr=float(self.pose_opt_lr.value),
                min_side=int(self.pose_opt_min_side.value),
                rgb_weight=float(self.pose_opt_rgb_weight.value),
                center_reg_weight=float(self.pose_opt_center_reg.value) * 1e12,
                pose_reg_weight=float(self.pose_opt_pose_reg.value),
                scale_reg_weight=float(self.pose_opt_scale_reg.value),
                box_reg_weight=float(self.pose_opt_box_reg.value),
                debug_mode=bool(self.pose_opt_debug.value),
                lock_translation_x=bool(self.pose_opt_lock_tx.value),
                lock_translation_y=bool(self.pose_opt_lock_ty.value),
                lock_translation_z=bool(self.pose_opt_lock_tz.value),
            )
        )
