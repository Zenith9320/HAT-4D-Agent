"""Base mixin class for SAM3D generation operations.

This module contains the Sam3DGenOpsBase class with core initialization,
state management, and basic UI control methods.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.ply_utils import SplatData

from ..state import Sam3DGenState, Sam3DObjectState
from .helpers import (
    _apply_uniform_scale_about_center,
    _load_rgb_alpha_u8,
    _pad_alpha_to_square,
    _pad_rgb_to_square,
)


class Sam3DGenOpsBase:
    """Base mixin providing core SAM3D generation operations.

    This class provides initialization, state management, camera control,
    status helpers, UI setters, and basic session management methods.

    Methods that call into other mixins (e.g., _refresh_frustum_image,
    _build_or_update_frustum, etc.) are left as self.method() calls
    since the final class will inherit from all mixins.
    """

    def __init__(self, ctx, state: Sam3DGenState):
        self.ctx = ctx
        self.state = state

    def _app(self):
        return self.ctx.app

    @staticmethod
    def _client_key(client) -> str:
        try:
            return str(client.client_id)
        except Exception:
            return str(id(client))

    def _clone_data(self, splat: SplatData) -> SplatData:
        """Deep-copy a SplatData so interactive edits are not cumulative."""
        return SplatData(
            centers=np.array(splat.centers, copy=True),
            rgbs=np.array(splat.rgbs, copy=True),
            opacities=np.array(splat.opacities, copy=True),
            covariances=np.array(splat.covariances, copy=True),
            scales=np.array(splat.scales, copy=True),
            rotations=np.array(splat.rotations, copy=True),
            sh=np.array(splat.sh, copy=True),
            opacity_raw=np.array(splat.opacity_raw, copy=True),
            centers_raw=np.array(splat.centers_raw, copy=True),
        )

    def _prepare_sam3d_image_and_mask(
        self,
        *,
        frame_token: str,
        mask_path: Path,
        mask_bool: np.ndarray | None = None,
        points_xy: list[tuple[int, int]] | None = None,
        point_labels: list[int] | None = None,
    ) -> tuple[Path, Path]:
        """
        Prepare image/mask paths for SAM3D generation/pose-opt.
        If display uses flip_udlr, unflip the mask and use the raw (unflipped) image.
        Also applies square padding to the raw image/mask when enabled.
        """
        app = self._app()
        img_raw = app.workspace.image_path_for_view(frame_token, "input")
        flip_udlr = bool(getattr(self.state, "image_flip_udlr", False))

        pad_square = bool(getattr(self.state, "pad_square", False))

        # Fast path: no flip/rotation, use existing padded path if any.
        if flip_udlr:
            raise NotImplementedError("SAM3D preparation does not support flipped input images")
        if pad_square and self.state.padded_image_path:
            return Path(self.state.padded_image_path), mask_path
        return img_raw, mask_path

    @staticmethod
    def _pad_mask_to_match(
        mask: np.ndarray,
        *,
        target_shape: tuple[int, int],
        pad_info: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        """Pad a HxW mask to target_shape using provided pad_info (top,bottom,left,right)."""
        h_t, w_t = int(target_shape[0]), int(target_shape[1])
        h, w = mask.shape[:2]
        if h == h_t and w == w_t:
            return mask
        out = np.zeros((h_t, w_t), dtype=mask.dtype)
        if pad_info is not None:
            pt, pb, pl, pr = [int(x) for x in pad_info]
            out[pt : pt + h, pl : pl + w] = mask
            return out
        # Center pad if pad_info missing.
        pad_top = max((h_t - h) // 2, 0)
        pad_left = max((w_t - w) // 2, 0)
        out[pad_top : pad_top + h, pad_left : pad_left + w] = mask
        return out

    def install_client_hooks(self, client) -> None:
        self._snapshot_camera(client)
        self._set_sam_status("uninitialized")
        self._set_sam3d_status("uninitialized")
        if self._backend_enabled():
            self._backend_start_status_poll(client)

        @client.camera.on_update
        def _(_cam: Any) -> None:
            if self._is_pixel_pick_mode(client):
                self._apply_locked_pose(client)

    def _is_pixel_pick_mode(self, client) -> bool:
        try:
            return bool(self.state.pixel_pick_mode_by_client.get(self._client_key(client), False))
        except Exception:
            return False

    # Camera locking
    def _snapshot_camera(self, client) -> None:
        self.state.locked_pose[client.client_id] = {
            "wxyz": np.asarray(client.camera.wxyz, dtype=np.float32).copy(),
            "position": np.asarray(client.camera.position, dtype=np.float32).copy(),
            "look_at": np.asarray(client.camera.look_at, dtype=np.float32).copy(),
            "up_direction": np.asarray(client.camera.up_direction, dtype=np.float32).copy(),
            "fov": float(client.camera.fov),
        }

    def _apply_locked_pose(self, client) -> None:
        pose = self.state.locked_pose.get(client.client_id)
        if pose is None:
            return
        with client.atomic():
            client.camera.wxyz = pose["wxyz"]
            client.camera.position = pose["position"]
            client.camera.look_at = pose["look_at"]
            client.camera.up_direction = pose["up_direction"]
            client.camera.fov = pose["fov"]

    # Status helpers
    def _set_sam_status(self, status: str) -> None:
        self.state.sam_status = str(status)
        if self.state.sam_status_handle is not None:
            try:
                self.state.sam_status_handle.value = self.state.sam_status
            except Exception:
                pass

    def _set_sam3d_status(self, status: str) -> None:
        self.state.sam3d_status = str(status)
        if self.state.sam3d_status_handle is not None:
            try:
                self.state.sam3d_status_handle.value = self.state.sam3d_status
            except Exception:
                pass

    # UI setters
    def set_interaction_mode(self, mode: str) -> None:
        mode = "transform" if mode == "transform" else "segment"
        self.state.interaction_mode = mode
        if self.state.frustum_handle is not None:
            if mode == "transform":
                try:
                    self.state.frustum_handle.remove_click_callback("all")
                except Exception:
                    pass
            else:
                self._ensure_frustum_click_callback()

    def set_frustum_opacity(self, opacity: float) -> None:
        self.state.frustum_opacity = float(np.clip(opacity, 0.0, 1.0))
        self._refresh_frustum_image()

    def set_pixel_pick_mode(self, client, enabled: bool) -> None:
        try:
            self.state.pixel_pick_mode_by_client[self._client_key(client)] = bool(enabled)
        except Exception:
            pass
        if self._is_pixel_pick_mode(client):
            self._snapshot_camera(client)

    def set_negative_point(self, enabled: bool) -> None:
        self.state.negative_point = bool(enabled)

    def set_eraser_mode(self, client, enabled: bool) -> None:
        self.state.eraser_mode = bool(enabled)
        if self.state.eraser_mode:
            try:
                self.state.pixel_pick_mode_by_client[self._client_key(client)] = True
            except Exception:
                pass
            self._snapshot_camera(client)

    def set_eraser_radius(self, radius_px: int) -> None:
        self.state.eraser_radius_px = int(max(1, radius_px))
        self._refresh_frustum_image()

    # Public actions
    def begin(self, client) -> None:
        app = self._app()
        import time as _time
        t0 = _time.time()
        print("[SAM3D begin] start")
        # Clean up any previous session nodes (frustum/object previews) to avoid leaks.
        try:
            app.server.scene.remove_by_name("/sam3d_gen")
        except Exception:
            pass

        frame_token = app.sequence.frame_tokens[app.current_frame]
        img_path = app.workspace.image_path_for_view(frame_token, "input")
        if not img_path.exists():
            client.add_notification(
                "SAM3D",
                f"Input image not found: {img_path}",
                color="red",
                auto_close_seconds=6,
            )
            return
        print(f"[SAM3D begin] image_path ok ({_time.time() - t0:.2f}s)")

        # Optional: clear current frame session PLY so subsequent edits can "overwrite" cleanly.
        try:
            with self.state.lock:
                do_cover = bool(getattr(self.state, "total_cover", False))
        except Exception:
            do_cover = False
        if do_cover:
            try:
                gs_path = Path(app.sequence.gs_path(app.current_frame))
            except Exception:
                gs_path = app.workspace.gs_dir / f"{frame_token}.ply"
            try:
                if gs_path.exists():
                    gs_path.unlink()
                # Invalidate cache and refresh current frame view.
                try:
                    if hasattr(app.sequence, "cache") and app.current_frame in app.sequence.cache:
                        app.sequence.cache.pop(app.current_frame, None)
                except Exception:
                    pass
                try:
                    preview_handle = app.preview_handles.get(client)
                    if preview_handle is not None:
                        app._apply_frame(int(app.current_frame), preview_handle)
                except Exception:
                    pass
                client.add_notification(
                    "SAM3D",
                    f"Total cover: cleared session PLY for frame {frame_token}",
                    color="green",
                    auto_close_seconds=4,
                )
            except Exception as exc:
                client.add_notification(
                    "SAM3D",
                    f"Total cover failed to clear PLY: {exc}",
                    color="orange",
                    auto_close_seconds=8,
                )
        print(f"[SAM3D begin] total_cover step done ({_time.time() - t0:.2f}s)")

        with self.state.lock:
            self.state.frame_token = frame_token
            self.state.gen_active = True
            # Match viewer_frontend input orientation (uses flip_colmap).
            flip_udlr = bool(getattr(app, "flip_colmap", True))
            self.state.image_flip_udlr = flip_udlr
            print("flip_udlr:", flip_udlr)
            image_rgb, alpha_u8 = _load_rgb_alpha_u8(img_path, flip_udlr=flip_udlr)
            print("image_path:", img_path)
            self.state.pad_info = None
            self.state.padded_image_path = None
            self.state.image_alpha_u8 = alpha_u8
            if bool(getattr(self.state, "pad_square", False)):
                image_rgb, pad_info = _pad_rgb_to_square(image_rgb, pad_value=int(self.state.pad_value))
                self.state.pad_info = pad_info
                if alpha_u8 is not None:
                    alpha_u8, _ = _pad_alpha_to_square(alpha_u8, pad_value=0)
                    self.state.image_alpha_u8 = alpha_u8
                # Persist padded image for backend usage (if needed).
                try:
                    out_dir = app.workspace.output_dir_for(frame_token, "input")
                    pad_dir = out_dir / "sam3d_gen" / "padded"
                    pad_dir.mkdir(parents=True, exist_ok=True)
                    pad_file = pad_dir / f"{frame_token}_padded.png"
                    Image.fromarray(image_rgb).save(pad_file)
                    self.state.padded_image_path = str(pad_file)
                except Exception:
                    self.state.padded_image_path = None
            self.state.image_rgb_u8 = image_rgb
            self.state.image_disp_rgb_u8 = self.state.image_rgb_u8

            # Reset object prompts when starting a new frame session.
            self.state.objects = [Sam3DObjectState()]
            self.state.current_object_idx = 0
            # Reset init notifications per session/frame.
            self.state._sam_init_notified = False
            self.state._sam3d_init_notified = False
        print(f"[SAM3D begin] state init done ({_time.time() - t0:.2f}s)")

        # Build frustum at current "input" view pose.
        self._build_or_update_frustum()
        # Apply current opacity right away (useful if user adjusted it previously).
        self._refresh_frustum_image()
        print(f"[SAM3D begin] frustum built ({_time.time() - t0:.2f}s)")
        if self._backend_enabled():
            print("[SAM3D begin] backend enabled: calling _backend_set_image")
            self._backend_set_image(client)
            print(f"[SAM3D begin] backend set_image done ({_time.time() - t0:.2f}s)")
        else:
            print("[SAM3D begin] local mode: calling _ensure_sam_ready")
            self._ensure_sam_ready(client)
            print(f"[SAM3D begin] local sam ready done ({_time.time() - t0:.2f}s)")

        # Default behavior: jump to frustum view, then let user toggle pick mode.
        self._jump_camera_to_frustum(client)
        self._update_status_handles()
        print(f"[SAM3D begin] done ({_time.time() - t0:.2f}s)")

    def end(self, client) -> None:
        """Exit SAM3D generation mode and remove temporary scene nodes.

        Leaves the workspace/session GS state intact (e.g. any objects appended to the
        frame PLY remain visible after the next frame reload).
        """
        app = self._app()
        try:
            app.server.scene.remove_by_name("/sam3d_gen")
        except Exception:
            pass

        with self.state.lock:
            self.state.gen_active = False
            self.state.frame_token = None
            self.state.image_rgb_u8 = None
            self.state.image_disp_rgb_u8 = None
            self.state.frustum_handle = None
            self.state.frustum_wxyz = None
            self.state.frustum_position = None
            self.state.pad_info = None
            self.state.padded_image_path = None
            self.state.image_alpha_u8 = None
            self.state.interaction_mode = "segment"
            self.state.pixel_pick_mode_by_client.clear()
            self.state.negative_point = False
            self.state.eraser_mode = False
            self.state.objects = [Sam3DObjectState()]
            self.state.current_object_idx = 0
            # Reset init notifications per "session".
            self.state._sam_init_notified = False
            self.state._sam3d_init_notified = False

        self._update_status_handles()
        client.add_notification(
            "SAM3D",
            "Exited SAM3D Gen mode (frustum + gizmos hidden).",
            color="green",
            auto_close_seconds=4,
        )

    def new_object(self, client) -> None:
        # Hide bbox of previous object before switching
        prev_idx = self.state.current_object_idx
        self._hide_bbox_for_object(prev_idx)

        with self.state.lock:
            self.state.objects.append(Sam3DObjectState())
            self.state.current_object_idx = len(self.state.objects) - 1
        self._refresh_frustum_image()
        self._update_status_handles()
        client.add_notification(
            "SAM3D",
            f"New object: idx={self.state.current_object_idx}",
            color="green",
            auto_close_seconds=4,
        )

    def select_prev_object(self, client) -> None:
        # Hide bbox of current object before switching
        prev_idx = self.state.current_object_idx
        self._hide_bbox_for_object(prev_idx)

        with self.state.lock:
            self.state.current_object_idx = max(0, self.state.current_object_idx - 1)
        self._refresh_frustum_image()
        self._update_status_handles()

    def select_next_object(self, client) -> None:
        # Hide bbox of current object before switching
        prev_idx = self.state.current_object_idx
        self._hide_bbox_for_object(prev_idx)

        with self.state.lock:
            self.state.current_object_idx = min(len(self.state.objects) - 1, self.state.current_object_idx + 1)
        self._refresh_frustum_image()
        self._update_status_handles()

    def undo_point(self, client) -> None:
        with self.state.lock:
            obj = self._cur_object()
            if obj.points_xy:
                obj.points_xy.pop()
                obj.point_labels.pop()
                obj.logits = None
                obj.logits_key = None
                obj.mask = None
                obj.mask_path = None
        self._run_sam_and_update_mask(client)

    def clear_points(self, client) -> None:
        with self.state.lock:
            obj = self._cur_object()
            obj.points_xy.clear()
            obj.point_labels.clear()
            obj.logits = None
            obj.logits_key = None
            obj.mask = None
            obj.mask_path = None
        self._refresh_frustum_image()
        self._update_status_handles()

    def set_manual_object_scale(self, client, scale: float) -> None:
        """Manually adjust current object's overall size (uniform scale about center)."""
        # Check if pose opt is running
        if getattr(self.state, "pose_opt_running", False):
            client.add_notification("SAM3D", "Cannot modify while pose opt is running.", color="orange", auto_close_seconds=3)
            return
        s = float(scale)
        s = float(np.clip(s, 0.01, 100.0))
        with self.state.lock:
            obj_idx = int(self.state.current_object_idx)
            obj = self.state.objects[obj_idx]
            if obj.splat is None:
                obj.manual_scale = s
                return
            # Keep an immutable base so slider moves are not cumulative.
            if obj.manual_scale_base_splat is None:
                obj.manual_scale_base_splat = self._clone_data(obj.splat)
            obj.manual_scale = s
            base = obj.manual_scale_base_splat
        try:
            splat_scaled = _apply_uniform_scale_about_center(base, s)
        except Exception:
            return
        with self.state.lock:
            if 0 <= obj_idx < len(self.state.objects):
                self.state.objects[obj_idx].splat = splat_scaled
                # Sync to gs_base state for pose opt
                import torch
                self.state.objects[obj_idx].gs_base_xyz = torch.from_numpy(splat_scaled.centers.copy()).float()
                self.state.objects[obj_idx].gs_base_rot = torch.from_numpy(splat_scaled.rotations.copy()).float()
                self.state.objects[obj_idx].gs_base_scale = torch.from_numpy(splat_scaled.scales.copy()).float()
                if splat_scaled.scales is not None and len(splat_scaled.scales) > 0:
                    self.state.objects[obj_idx].gs_base_min_kernel = float(splat_scaled.scales.min())
                # Sync bbox_base_splat so transform controller uses the scaled splat as base
                obj = self.state.objects[obj_idx]
                if obj.bbox_frame_handle is not None:
                    obj.bbox_base_splat = self._clone_data(splat_scaled)
                    # Update bbox dimensions to match new splat bounds
                    xyz = splat_scaled.centers
                    mins = xyz.min(axis=0)
                    maxs = xyz.max(axis=0)
                    dims = np.maximum(maxs - mins, 1e-4).astype(np.float32)
                    center = ((mins + maxs) * 0.5).astype(np.float32)
                    obj.bbox_base_center = center.copy()
                    obj.bbox_base_dims = dims.copy()
                    obj.bbox_dimensions = dims.copy()
                    # Update bbox frame position and box dimensions
                    obj.bbox_frame_handle.position = tuple(center.tolist())
                    obj.bbox_frame_handle.wxyz = (1.0, 0.0, 0.0, 0.0)
                    obj.bbox_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    if obj.bbox_box_handle is not None:
                        obj.bbox_box_handle.dimensions = tuple(dims.tolist())
        try:
            self._ensure_object_handle(obj_idx)
            # Update face handles positions if bbox exists
            with self.state.lock:
                obj = self.state.objects[obj_idx]
                if obj.bbox_frame_handle is not None:
                    self._update_face_handles(obj)
            if not hasattr(self.state.objects[obj_idx], 'bbox_frame_handle') or self.state.objects[obj_idx].bbox_frame_handle is None:
                self._ensure_bbox_for_object(client, obj_idx)
        except Exception:
            pass

    def nudge_manual_object_scale(self, client, mult: float) -> float:
        """Multiply current manual scale by `mult` and apply; returns new scale."""
        # Check if pose opt is running
        if getattr(self.state, "pose_opt_running", False):
            client.add_notification("SAM3D", "Cannot modify while pose opt is running.", color="orange", auto_close_seconds=3)
            with self.state.lock:
                obj = self._cur_object()
                return float(getattr(obj, "manual_scale", 1.0))
        m = float(mult)
        if not np.isfinite(m) or m <= 0:
            return 1.0
        with self.state.lock:
            obj = self._cur_object()
            cur = float(getattr(obj, "manual_scale", 1.0))
        new = float(np.clip(cur * m, 0.01, 100.0))
        self.set_manual_object_scale(client, new)
        return new

    # Helpers
    def _cur_object(self) -> Sam3DObjectState:
        return self.state.objects[self.state.current_object_idx]

    def _update_status_handles(self) -> None:
        with self.state.lock:
            idx = int(self.state.current_object_idx)
            obj = self.state.objects[idx]
            npts = len(obj.points_xy)
            has_mask = obj.mask is not None
            frame = self.state.frame_token
        if self.state.object_status_handle is not None:
            try:
                self.state.object_status_handle.value = f"obj={idx} points={npts}"
            except Exception:
                pass
        if self.state.mask_status_handle is not None:
            try:
                self.state.mask_status_handle.value = "mask=ok" if has_mask else "mask=None"
            except Exception:
                pass
        if self.state.status_handle is not None:
            try:
                self.state.status_handle.value = f"frame={frame}" if frame else "idle"
            except Exception:
                pass
