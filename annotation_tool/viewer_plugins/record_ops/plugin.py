"""Record Ops Plugin - records SAM3D operations for VLM training data.

This plugin observes SAM3D events non-invasively by wrapping key methods
and records user operations including clicks, masks, and generated objects.
"""
from __future__ import annotations

from functools import wraps
import threading
from typing import Any, Optional

import numpy as np

from .context import ViewerContext
from .ops import RecordOps
from .state import RecordOpsState
from .ui import RecordOpsUI


class RecordOpsPlugin:
    """Plugin that records SAM3D operations for VLM training data generation."""

    def __init__(self, app, sam3d_plugin):
        self.app = app
        self.sam3d_plugin = sam3d_plugin
        self.ctx = ViewerContext(app)
        self.state = RecordOpsState()
        self.ops = RecordOps(self.ctx, self.state)
        self.ui = RecordOpsUI(self.ctx, self.ops, self.state)
        self._hooks_installed = False

    def install(self, client, workspace, parent_folder=None):
        """Install the plugin UI and hook into SAM3D methods."""
        self.workspace = workspace
        self.ops.workspace = workspace
        self.ui.build(client, parent_folder=parent_folder)

        # Install method hooks on SAM3D ops
        if not self._hooks_installed:
            self._install_method_hooks()
            self._hooks_installed = True

    def _install_method_hooks(self) -> None:
        """Wrap SAM3D ops methods to observe events."""
        sam3d_ops = self.sam3d_plugin.ops
        print("[RecordOps] Installing method hooks on SAM3D ops...")

        # Hook begin() - session start
        original_begin = sam3d_ops.begin

        @wraps(original_begin)
        def hooked_begin(client):
            print("[RecordOps] hooked_begin called")
            # Call original first
            result = original_begin(client)
            # Then trigger our callback (async to avoid blocking UI).
            print("[RecordOps] Scheduling on_sam3d_begin callback")
            try:
                app = self.app
                frame_token = app.sequence.frame_tokens[app.current_frame]
                frame_idx = app.current_frame

                def _run_async():
                    try:
                        self.ops.on_sam3d_begin(client, frame_token, frame_idx)
                        self.ui.update_status()
                    except Exception as e:
                        print(f"[RecordOps] on_sam3d_begin failed: {e}")
                        import traceback
                        traceback.print_exc()

                threading.Thread(target=_run_async, daemon=True).start()
            except Exception as e:
                print(f"[RecordOps] on_sam3d_begin schedule failed: {e}")
            return result

        sam3d_ops.begin = hooked_begin
        print("[RecordOps] Hooked begin()")

        # Hook _handle_pick() - pixel click
        original_handle_pick = sam3d_ops._handle_pick

        @wraps(original_handle_pick)
        def hooked_handle_pick(client, u0: int, v0: int, u_disp: int, v_disp: int):
            # Determine click label before calling original
            try:
                label = 0 if sam3d_ops.state.negative_point else 1
                # Check eraser mode - don't record eraser clicks
                is_eraser = sam3d_ops.state.eraser_mode
            except Exception:
                label = 1
                is_eraser = False

            # Call original
            result = original_handle_pick(client, u0, v0, u_disp, v_disp)

            # Record the click (only non-eraser clicks)
            if not is_eraser:
                try:
                    self.ops.on_pixel_pick(client, u0, v0, label)
                    self.ui.update_status()
                except Exception as e:
                    print(f"[RecordOps] on_pixel_pick failed: {e}")

            return result

        sam3d_ops._handle_pick = hooked_handle_pick
        print("[RecordOps] Hooked _handle_pick()")

        # Hook new_object() - object completion + new object start
        original_new_object = sam3d_ops.new_object

        @wraps(original_new_object)
        def hooked_new_object(client):
            # Before creating new object, record the current object as complete
            try:
                self._record_current_object_complete(client)
            except Exception as e:
                print(f"[RecordOps] on_object_complete failed: {e}")

            # Call original
            result = original_new_object(client)

            # Notify that a new object was created
            try:
                self.ops.on_new_object(client)
                self.ui.update_status()
            except Exception as e:
                print(f"[RecordOps] on_new_object failed: {e}")

            return result

        sam3d_ops.new_object = hooked_new_object
        print("[RecordOps] Hooked new_object()")

        # Hook add_object_to_session_ply() - session completion
        original_add_object = sam3d_ops.add_object_to_session_ply

        @wraps(original_add_object)
        def hooked_add_object(client):
            # Record current object as complete
            try:
                self._record_current_object_complete(client)
            except Exception as e:
                print(f"[RecordOps] on_object_complete failed: {e}")

            # Call original
            result = original_add_object(client)

            # Record session completion
            try:
                frame_idx = self.app.current_frame
                self.ops.on_session_complete(client, frame_idx)
                self.ui.update_status()
            except Exception as e:
                print(f"[RecordOps] on_session_complete failed: {e}")

            return result

        sam3d_ops.add_object_to_session_ply = hooked_add_object
        print("[RecordOps] Hooked add_object_to_session_ply()")
        print("[RecordOps] All hooks installed successfully")

    def _record_current_object_complete(self, client) -> None:
        """Record the current object's completion data."""
        sam3d_state = self.sam3d_plugin.state
        with sam3d_state.lock:
            obj_idx = sam3d_state.current_object_idx
            if obj_idx >= len(sam3d_state.objects):
                return
            sam3d_obj = sam3d_state.objects[obj_idx]

            # Get mask path
            mask_path = sam3d_obj.mask_path

            # Get frustum RGB image (the current view shown in frustum)
            rgb_image = None
            if sam3d_state.image_rgb_u8 is not None:
                rgb_image = sam3d_state.image_rgb_u8.copy()

            # Get bbox from sam3d object state
            bbox_center = None
            bbox_dimensions = None
            bbox_wxyz = None

            if hasattr(sam3d_obj, "bbox_base_center") and sam3d_obj.bbox_base_center is not None:
                bbox_center = np.asarray(sam3d_obj.bbox_base_center, dtype=np.float32)
            if hasattr(sam3d_obj, "bbox_dimensions") and sam3d_obj.bbox_dimensions is not None:
                bbox_dimensions = np.asarray(sam3d_obj.bbox_dimensions, dtype=np.float32)
            if hasattr(sam3d_obj, "bbox_base_wxyz") and sam3d_obj.bbox_base_wxyz is not None:
                bbox_wxyz = np.asarray(sam3d_obj.bbox_base_wxyz, dtype=np.float32)

        # Call the ops method
        self.ops.on_object_complete(
            client,
            obj_idx=obj_idx,
            mask_path=mask_path,
            rgb_image=rgb_image,
            bbox_center=bbox_center,
            bbox_dimensions=bbox_dimensions,
            bbox_wxyz=bbox_wxyz,
        )
