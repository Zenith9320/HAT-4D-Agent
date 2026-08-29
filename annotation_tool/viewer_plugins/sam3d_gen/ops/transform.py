"""BBox transform mixin for SAM3D generation operations.

This module contains transform-related methods for bounding box manipulation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utils.ply_utils import SplatData

from ..state import Sam3DObjectState

if TYPE_CHECKING:
    pass


class TransformMixin:
    """Mixin class providing BBox transform functionality."""

    # BBox transform (object-local)
    def _rotation_matrix_from_wxyz(self, wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in wxyz]
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float32,
        )

    def _face_handle_position(self, obj: Sam3DObjectState, key: str) -> np.ndarray:
        local_dir, axis = obj.bbox_face_defs[key]
        dims = obj.bbox_dimensions
        assert dims is not None
        return local_dir * (dims[axis] * 0.5)

    def _update_face_handles(self, obj: Sam3DObjectState) -> None:
        if not obj.bbox_face_handles or obj.bbox_dimensions is None:
            return
        obj.bbox_updating = True
        try:
            for key, handle in obj.bbox_face_handles.items():
                handle.position = tuple(self._face_handle_position(obj, key).tolist())
        finally:
            obj.bbox_updating = False

    def _apply_bbox_transform_live(self, obj_idx: int) -> None:
        # Block live transform while pose opt is running
        if getattr(self.state, "pose_opt_running", False):
            return
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            if obj.bbox_updating:
                return
            if obj.bbox_frame_handle is None:
                return
            if obj.bbox_base_splat is None or obj.splat is None:
                return
            base = obj.bbox_base_splat
            splat = obj.splat
            c0 = obj.bbox_base_center
            wxyz0 = obj.bbox_base_wxyz
            dims0 = obj.bbox_base_dims
            dims = obj.bbox_dimensions
            if c0 is None or wxyz0 is None or dims0 is None or dims is None:
                return
            pos = np.array(obj.bbox_frame_handle.position, dtype=np.float32)
            wxyz = np.array(obj.bbox_frame_handle.wxyz, dtype=np.float32)
            mode = obj.transform_mode

        if mode == "Scale":
            pos = c0
            # In scale mode, let user rotate the frame to choose scaling axes.
            with self.state.lock:
                obj = self.state.objects[obj_idx]
                obj.bbox_base_wxyz = wxyz
                wxyz0 = wxyz
            wxyz = wxyz0

        R0 = self._rotation_matrix_from_wxyz(wxyz0)
        R = self._rotation_matrix_from_wxyz(wxyz)
        S = np.diag((dims / np.maximum(dims0, 1e-6)).astype(np.float32))

        centers0 = base.centers
        local0 = (centers0 - c0) @ R0
        local1 = local0 @ S
        centers_new = local1 @ R.T + pos

        cov0 = base.covariances
        cov_local = np.einsum("ij,njk,kl->nil", R0.T, cov0, R0)
        cov_scaled = np.einsum("ij,njk,kl->nil", S, cov_local, S)
        cov_new = np.einsum("ij,njk,kl->nil", R, cov_scaled, R.T)

        with self.state.lock:
            obj = self.state.objects[obj_idx]
            if obj.splat is None:
                return
            obj.splat.centers = centers_new.astype(np.float32)
            obj.splat.covariances = cov_new.astype(np.float32)
            splat_updated = obj.splat

        # Push update to scene.
        self._ensure_object_handle(obj_idx)
        if splat_updated is not None:
            pass

    def _ensure_bbox_for_object(self, client, obj_idx: int) -> None:
        app = self._app()
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            splat = obj.splat
            if splat is None:
                return
            # If bbox was created under an older node path, recreate it under camera_frame.
            if obj.bbox_frame_handle is not None:
                bbox_name = getattr(obj.bbox_frame_handle, "name", "")
                if "/camera_frame/" in str(bbox_name):
                    return
                try:
                    if obj.bbox_frame_handle is not None:
                        obj.bbox_frame_handle.remove()
                except Exception:
                    pass
                try:
                    if obj.bbox_box_handle is not None:
                        obj.bbox_box_handle.remove()
                except Exception:
                    pass
                for h in list(obj.bbox_face_handles.values()):
                    try:
                        h.remove()
                    except Exception:
                        pass
                obj.bbox_frame_handle = None
                obj.bbox_box_handle = None
                obj.bbox_face_handles = {}
                obj.bbox_face_defs = {}
            xyz = splat.centers

            # Also migrate gaussian node if it was created under the old path (without camera_frame).
            if obj.gs_handle is not None:
                gs_name = getattr(obj.gs_handle, "name", "")
                if "/camera_frame/" not in str(gs_name):
                    try:
                        obj.gs_handle.remove()
                    except Exception:
                        pass
                    obj.gs_handle = None
            if obj.frame_handle is not None:
                fr_name = getattr(obj.frame_handle, "name", "")
                if "/camera_frame" not in str(fr_name):
                    try:
                        obj.frame_handle.remove()
                    except Exception:
                        pass
                    obj.frame_handle = None
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        dims = np.maximum(maxs - mins, 1e-4).astype(np.float32)
        center = ((mins + maxs) * 0.5).astype(np.float32)

        frame_token = self.state.frame_token
        if frame_token is None:
            return
        base = f"/sam3d_gen/{frame_token}/objects/obj_{obj_idx}/camera_frame/bbox"
        frame = app.server.scene.add_transform_controls(base, position=tuple(center.tolist()), depth_test=False)
        if hasattr(frame, "wxyz"):
            frame.wxyz = (1.0, 0.0, 0.0, 0.0)

        box = app.server.scene.add_box(
            f"{base}/box",
            dimensions=tuple(dims.tolist()),
            wireframe=True,
            color=(255, 220, 60),
            opacity=0.9,
            position=(0.0, 0.0, 0.0),
        )

        face_defs = {
            "px": (np.array([+1, 0, 0], np.float32), 0),
            "nx": (np.array([-1, 0, 0], np.float32), 0),
            "py": (np.array([0, +1, 0], np.float32), 1),
            "ny": (np.array([0, -1, 0], np.float32), 1),
            "pz": (np.array([0, 0, +1], np.float32), 2),
            "nz": (np.array([0, 0, -1], np.float32), 2),
        }

        def _make_limits(axis: int):
            big = (-1000.0, 1000.0)
            lock = (0.0, 0.0)
            limits = [lock, lock, lock]
            limits[axis] = big
            return (tuple(limits[0]), tuple(limits[1]), tuple(limits[2]))

        face_handles = {}
        for k, (local_dir, axis) in face_defs.items():
            h = app.server.scene.add_transform_controls(
                f"{base}/face_{k}",
                scale=0.35,
                disable_rotations=True,
                disable_sliders=True,
                active_axes=(axis == 0, axis == 1, axis == 2),
                translation_limits=_make_limits(axis),
                position=tuple((local_dir * (dims[axis] * 0.5)).tolist()),
                depth_test=False,
            )
            face_handles[k] = h

        with self.state.lock:
            obj = self.state.objects[obj_idx]
            obj.bbox_frame_handle = frame
            obj.bbox_box_handle = box
            obj.bbox_face_handles = face_handles
            obj.bbox_face_defs = face_defs
            obj.bbox_dimensions = dims.copy()
            obj.bbox_base_splat = SplatData(
                centers=splat.centers.copy(),
                rgbs=splat.rgbs.copy(),
                opacities=splat.opacities.copy(),
                covariances=splat.covariances.copy(),
                scales=splat.scales.copy(),
                rotations=splat.rotations.copy(),
                sh=splat.sh.copy(),
                opacity_raw=splat.opacity_raw.copy(),
                centers_raw=splat.centers_raw.copy(),
            )
            obj.bbox_base_center = center.copy()
            obj.bbox_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obj.bbox_base_dims = dims.copy()
            obj.bbox_visible = True

        self._attach_bbox_callbacks(client, obj_idx)
        self._apply_transform_mode_controls(obj_idx)
        self.set_bbox_visible(client, True)

    def _attach_bbox_callbacks(self, client, obj_idx: int) -> None:
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            frame = obj.bbox_frame_handle
            face_handles = dict(obj.bbox_face_handles)
        if frame is None:
            return

        @frame.on_update
        def _(_event):
            self._apply_bbox_transform_live(obj_idx)

        for key, handle in face_handles.items():
            @handle.on_update
            def _evt(_event, key=key):
                with self.state.lock:
                    obj = self.state.objects[obj_idx]
                    if obj.bbox_updating:
                        return
                self._update_bbox_from_face(obj_idx, key)
                with self.state.lock:
                    obj = self.state.objects[obj_idx]
                    self._update_face_handles(obj)
                self._apply_bbox_transform_live(obj_idx)

    def _update_bbox_from_face(self, obj_idx: int, key: str) -> None:
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            dims = obj.bbox_dimensions
            handle = obj.bbox_face_handles.get(key)
            face_defs = obj.bbox_face_defs
        if dims is None or handle is None:
            return
        pos = np.array(handle.position, dtype=np.float32)
        local_dir, axis = face_defs[key]
        val = float(pos[axis])
        if key[0] == "p":
            dims[axis] = max(val * 2.0, 1e-4)
        else:
            dims[axis] = max(-val * 2.0, 1e-4)
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            obj.bbox_dimensions = dims
            if obj.bbox_box_handle is not None:
                obj.bbox_box_handle.dimensions = tuple(dims.tolist())

    def _apply_transform_mode_controls(self, obj_idx: int) -> None:
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            frame = obj.bbox_frame_handle
            mode = obj.transform_mode
            x_on, y_on, z_on = obj.axis_constraint
            face_handles = dict(obj.bbox_face_handles)
            face_defs = dict(obj.bbox_face_defs)
        if frame is None:
            return

        if mode == "Translate":
            frame.disable_axes = False
            frame.disable_sliders = False
            frame.disable_rotations = True
        elif mode == "Rotate":
            frame.disable_axes = True
            frame.disable_sliders = True
            frame.disable_rotations = False
        elif mode == "Scale":
            frame.disable_axes = True
            frame.disable_sliders = True
            frame.disable_rotations = False

        use_constraint = any((x_on, y_on, z_on))
        frame.active_axes = (
            (x_on if use_constraint else True),
            (y_on if use_constraint else True),
            (z_on if use_constraint else True),
        )

        show_faces = mode == "Scale"
        for key, handle in face_handles.items():
            axis = face_defs[key][1]
            if not use_constraint:
                handle.visible = show_faces
            else:
                handle.visible = show_faces and [x_on, y_on, z_on][axis]

    def set_bbox_visible(self, client, visible: bool) -> None:
        with self.state.lock:
            obj = self._cur_object()
            obj.bbox_visible = bool(visible)
            frame = obj.bbox_frame_handle
            box = obj.bbox_box_handle
            faces = list(obj.bbox_face_handles.values())
        if frame is not None:
            frame.visible = bool(visible)
        if box is not None:
            box.visible = bool(visible)
        for h in faces:
            h.visible = bool(visible)
        if bool(visible):
            self._apply_transform_mode_controls(self.state.current_object_idx)

    def _hide_bbox_for_object(self, obj_idx: int) -> None:
        """Hide bbox gizmo for a specific object (used when switching objects)."""
        with self.state.lock:
            if obj_idx < 0 or obj_idx >= len(self.state.objects):
                return
            obj = self.state.objects[obj_idx]
            obj.bbox_visible = False
            frame = obj.bbox_frame_handle
            box = obj.bbox_box_handle
            faces = list(obj.bbox_face_handles.values())
        if frame is not None:
            try:
                frame.visible = False
            except Exception:
                pass
        if box is not None:
            try:
                box.visible = False
            except Exception:
                pass
        for h in faces:
            try:
                h.visible = False
            except Exception:
                pass

    def update_transform_mode(self, mode: str) -> None:
        mode = mode if mode in ("Translate", "Rotate", "Scale") else "Translate"
        with self.state.lock:
            obj = self._cur_object()
            obj.transform_mode = mode
        self._apply_transform_mode_controls(self.state.current_object_idx)

    def update_axis_mode(self, mode: str) -> None:
        with self.state.lock:
            obj = self._cur_object()
            obj.axis_mode = str(mode)

    def update_axis_constraint(self, x: bool, y: bool, z: bool) -> None:
        with self.state.lock:
            obj = self._cur_object()
            obj.axis_constraint = (bool(x), bool(y), bool(z))
        self._apply_transform_mode_controls(self.state.current_object_idx)

    def _reset_bbox_to_current_splat(self, client, obj_idx: int) -> None:
        """Reset the bounding-box gizmo to the current post-optimization splat."""
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            # Remove existing bounding-box handles.
            if obj.bbox_frame_handle is not None:
                try:
                    obj.bbox_frame_handle.remove()
                except Exception:
                    pass
                obj.bbox_frame_handle = None
            if obj.bbox_box_handle is not None:
                try:
                    obj.bbox_box_handle.remove()
                except Exception:
                    pass
                obj.bbox_box_handle = None
            for h in list(obj.bbox_face_handles.values()):
                try:
                    h.remove()
                except Exception:
                    pass
            obj.bbox_face_handles = {}
            obj.bbox_face_defs = {}
            # Clear the transform baseline.
            obj.bbox_base_splat = None
            obj.bbox_base_center = None
            obj.bbox_base_wxyz = None
            obj.bbox_base_dims = None
            obj.bbox_dimensions = None
        # Recreate the bounding box from the current splat.
        self._ensure_bbox_for_object(client, obj_idx)

    def apply_transform(self, client) -> None:
        # Check if pose opt is running
        if getattr(self.state, "pose_opt_running", False):
            client.add_notification("SAM3D", "Cannot modify while pose opt is running.", color="orange", auto_close_seconds=3)
            return
        with self.state.lock:
            obj = self._cur_object()
            if obj.bbox_frame_handle is None or obj.splat is None:
                client.add_notification("SAM3D", "No bbox/object to apply.", color="orange", auto_close_seconds=4)
                return
            pos = np.array(obj.bbox_frame_handle.position, dtype=np.float32)
            wxyz = np.array(obj.bbox_frame_handle.wxyz, dtype=np.float32)
            dims = obj.bbox_dimensions if obj.bbox_dimensions is not None else obj.bbox_base_dims
            obj.bbox_base_splat = SplatData(
                centers=obj.splat.centers.copy(),
                rgbs=obj.splat.rgbs.copy(),
                opacities=obj.splat.opacities.copy(),
                covariances=obj.splat.covariances.copy(),
                scales=obj.splat.scales.copy(),
                rotations=obj.splat.rotations.copy(),
                sh=obj.splat.sh.copy(),
                opacity_raw=obj.splat.opacity_raw.copy(),
                centers_raw=obj.splat.centers_raw.copy(),
            )
            obj.bbox_base_center = pos
            obj.bbox_base_wxyz = wxyz
            if dims is not None:
                obj.bbox_base_dims = np.array(dims, dtype=np.float32)
            # Use the transformed object as the next pose-optimization baseline.
            import torch
            obj.gs_base_xyz = torch.from_numpy(obj.splat.centers.copy()).float()
            obj.gs_base_rot = torch.from_numpy(obj.splat.rotations.copy()).float()
            obj.gs_base_scale = torch.from_numpy(obj.splat.scales.copy()).float()
            if obj.splat.scales is not None and len(obj.splat.scales) > 0:
                obj.gs_base_min_kernel = float(obj.splat.scales.min())
        client.add_notification("SAM3D", "Transform applied.", color="green", auto_close_seconds=4)

    def reset_transform(self, client) -> None:
        # Check if pose opt is running
        if getattr(self.state, "pose_opt_running", False):
            client.add_notification("SAM3D", "Cannot modify while pose opt is running.", color="orange", auto_close_seconds=3)
            return
        with self.state.lock:
            obj = self._cur_object()
            if obj.bbox_frame_handle is None or obj.bbox_base_splat is None:
                client.add_notification("SAM3D", "No transform to reset.", color="orange", auto_close_seconds=4)
                return
            if obj.bbox_base_center is not None:
                obj.bbox_frame_handle.position = tuple(obj.bbox_base_center.tolist())
            if obj.bbox_base_wxyz is not None:
                obj.bbox_frame_handle.wxyz = tuple(obj.bbox_base_wxyz.tolist())
            if obj.bbox_base_dims is not None:
                obj.bbox_dimensions = obj.bbox_base_dims.copy()
                if obj.bbox_box_handle is not None:
                    obj.bbox_box_handle.dimensions = tuple(obj.bbox_dimensions.tolist())
        with self.state.lock:
            self._update_face_handles(self._cur_object())
        self._apply_bbox_transform_live(self.state.current_object_idx)
        client.add_notification("SAM3D", "Transform reset.", color="green", auto_close_seconds=4)
