"""SAM-related methods for SAM3D generation operations.

This module contains the SamMixin class with methods for:
- Frustum building and click handling
- Camera control and ray picking
- SAM model initialization and mask prediction
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import viser.transforms as tf

from utils.image_utils import add_alpha_channel

from .helpers import _ensure_sam3d_thirdparty_on_path, _overlay_mask_and_points


class SamMixin:
    """Mixin class providing SAM segmentation and frustum interaction methods."""

    # ----------------------------
    # Frustum + picking
    # ----------------------------
    def _build_or_update_frustum(self) -> None:
        app = self._app()
        assert self.state.image_disp_rgb_u8 is not None
        img = self.state.image_disp_rgb_u8

        H, W = img.shape[:2]
        self.state.frustum_aspect = W / float(H)

        # Use the viewer's canonical "input" view pose so pose optimization/rendering aligns
        # with the input image camera (az=0, el=0 by default).
        # Match the main viewer's default input view distance (see viser_frontend._move_camera_to_view).
        dist = 1.5
        eye, wxyz = app._eye_wxyz_for_view("input", float(dist))

        wxyz = np.asarray(wxyz, dtype=np.float32)

        self.state.frustum_wxyz = np.asarray(wxyz, dtype=np.float32)
        self.state.frustum_position = np.asarray(eye, dtype=np.float32)

        # Recreate the node to guarantee click callback registration and to refresh image buffer.
        if self.state.frustum_handle is not None:
            try:
                self.state.frustum_handle.remove()
            except Exception:
                pass
            self.state.frustum_handle = None

        image_rgba = add_alpha_channel(img, float(self.state.frustum_opacity))
        frustum_variant = "filled"
        self.state.frustum_handle = app.server.scene.add_camera_frustum(
            f"/sam3d_gen/{self.state.frame_token}/input_frustum",
            fov=np.deg2rad(app.fov_deg),
            aspect=float(self.state.frustum_aspect),
            scale=float(self.state.frustum_scale),
            image=image_rgba,
            wxyz=tuple(self.state.frustum_wxyz.tolist()),
            position=tuple(self.state.frustum_position.tolist()),
            variant=frustum_variant,
            cast_shadow=False,
            receive_shadow=False,
        )
        # Only make frustum clickable in segment mode. In transform mode, we want gizmo
        # interactions to go "through" the image plane.
        if self.state.interaction_mode != "transform":
            self._ensure_frustum_click_callback()
        else:
            try:
                self.state.frustum_handle.remove_click_callback("all")
            except Exception:
                pass

    def _ensure_frustum_click_callback(self) -> None:
        frustum = self.state.frustum_handle
        if frustum is None:
            return
        try:
            frustum.remove_click_callback("all")
        except Exception:
            pass

        @frustum.on_click
        def _on_frustum_click(event) -> None:
            if self.state.interaction_mode == "transform":
                return

            if not self._is_pixel_pick_mode(event.client):
                self._jump_camera_to_frustum(event.client)
                self._snapshot_camera(event.client)
                return

            xy = self._ray_to_pixel_xy(event)
            if xy is None:
                return
            u0, v0, u_disp, v_disp = xy
            self._handle_pick(event.client, u0, v0, u_disp, v_disp)

    def _jump_camera_to_frustum(self, client) -> None:
        if self.state.frustum_wxyz is None or self.state.frustum_position is None:
            return
        app = self._app()
        try:
            R_wc = tf.SO3(self.state.frustum_wxyz).as_matrix().astype(np.float32)
            forward_w = R_wc[:, 2]
            with client.atomic():
                client.camera.position = self.state.frustum_position
                client.camera.wxyz = self.state.frustum_wxyz
                client.camera.look_at = self.state.frustum_position + forward_w
                client.camera.up_direction = np.asarray(getattr(app, "up_vec", [0.0, 1.0, 0.0]), dtype=np.float32)
                client.camera.fov = np.deg2rad(app.fov_deg)
        except Exception:
            with client.atomic():
                client.camera.position = self.state.frustum_position
                client.camera.wxyz = self.state.frustum_wxyz
                client.camera.up_direction = np.asarray(getattr(app, "up_vec", [0.0, 1.0, 0.0]), dtype=np.float32)
                client.camera.fov = np.deg2rad(app.fov_deg)

    def _ray_to_pixel_xy(self, event) -> Optional[Tuple[int, int, int, int]]:
        """Map frustum click ray -> (u0,v0) in original image, plus (u_disp,v_disp) in display image."""
        if self.state.frustum_wxyz is None or self.state.frustum_position is None:
            return None
        if self.state.image_rgb_u8 is None or self.state.image_disp_rgb_u8 is None:
            return None
        if not (hasattr(event, "ray_origin") and hasattr(event, "ray_direction")):
            return None

        app = self._app()
        fov_y = float(app.fov_deg)

        img_disp = self.state.image_disp_rgb_u8
        Hd, Wd = img_disp.shape[:2]
        H0, W0 = self.state.image_rgb_u8.shape[:2]

        R_wc = tf.SO3(self.state.frustum_wxyz).as_matrix().astype(np.float64)
        t_wc = self.state.frustum_position.astype(np.float64)

        ray_o_w = np.asarray(event.ray_origin, dtype=np.float64)
        ray_d_w = np.asarray(event.ray_direction, dtype=np.float64)
        ray_d_w /= np.linalg.norm(ray_d_w) + 1e-12

        R_cw = R_wc.T
        ray_o_c = R_cw @ (ray_o_w - t_wc)
        ray_d_c = R_cw @ ray_d_w
        ray_d_c /= np.linalg.norm(ray_d_c) + 1e-12

        z_plane = float(self.state.frustum_scale)
        denom = float(ray_d_c[2])
        if abs(denom) < 1e-8:
            return None
        t = (z_plane - float(ray_o_c[2])) / denom
        if t <= 0:
            return None
        P_c = ray_o_c + t * ray_d_c

        half_h = z_plane * math.tan(math.radians(fov_y) / 2.0)
        half_w = half_h * float(Wd / float(Hd))
        u01 = (float(P_c[0]) / (2.0 * half_w)) + 0.5
        v01 = (float(P_c[1]) / (2.0 * half_h)) + 0.5
        if not (0.0 <= u01 <= 1.0 and 0.0 <= v01 <= 1.0):
            return None

        u_disp = int(round(u01 * (Wd - 1)))
        v_disp = int(round(v01 * (Hd - 1)))
        u0 = int(round(u01 * (W0 - 1)))
        v0 = int(round(v01 * (H0 - 1)))
        return u0, v0, u_disp, v_disp

    def _handle_pick(self, client, u0: int, v0: int, u_disp: int, v_disp: int) -> None:
        ds = int(max(1, self.state.image_downsample))

        if self.state.eraser_mode:
            radius_disp = int(self.state.eraser_radius_px)
            radius_orig = radius_disp * ds
            r2 = float(radius_orig * radius_orig)
            with self.state.lock:
                obj = self._cur_object()
                kept_xy = []
                kept_labels = []
                for (px, py), lab in zip(obj.points_xy, obj.point_labels):
                    if (px - u0) ** 2 + (py - v0) ** 2 > r2:
                        kept_xy.append((px, py))
                        kept_labels.append(lab)
                obj.points_xy = kept_xy
                obj.point_labels = kept_labels
                obj.logits = None
                obj.logits_key = None
                obj.mask = None
                obj.mask_path = None
            self._run_sam_and_update_mask(client)
            return

        label = 0 if self.state.negative_point else 1
        with self.state.lock:
            obj = self._cur_object()
            obj.points_xy.append((int(u0), int(v0)))
            obj.point_labels.append(int(label))
            obj.mask = None
            obj.mask_path = None
        self._run_sam_and_update_mask(client)

    # ----------------------------
    # SAM
    # ----------------------------
    def _ensure_sam_ready(self, client) -> bool:
        _ensure_sam3d_thirdparty_on_path()
        try:
            from omegaconf import OmegaConf  # noqa: F401
        except Exception:
            pass

        if self.state.image_rgb_u8 is None:
            return False

        if self.state._sam_predictor is None:
            self._set_sam_status("initializing")
            if not self.state._sam_init_notified:
                self.state._sam_init_notified = True
                client.add_notification(
                    "SAM3D",
                    "Initializing SAM... (first time may take a while)",
                    color="orange",
                    auto_close_seconds=6,
                )
            # Resolve default checkpoint if user didn't configure one.
            if self.state.sam_checkpoint is None:
                self.state.sam_checkpoint = os.environ.get("SAM_CHECKPOINT")

            if self.state.sam_checkpoint is None or not Path(self.state.sam_checkpoint).exists():
                self._set_sam_status("checkpoint_missing")
                client.add_notification(
                    "SAM3D",
                    "SAM checkpoint not found. Set SAM_CHECKPOINT to a valid .pth file.",
                    color="red",
                    auto_close_seconds=8,
                )
                return False

            try:
                from segment_anything import SamPredictor, sam_model_registry  # type: ignore
            except Exception as exc:
                self._set_sam_status("import_failed")
                client.add_notification(
                    "SAM3D",
                    f"Failed to import segment_anything: {exc}",
                    color="red",
                    auto_close_seconds=8,
                )
                return False

            try:
                sam = sam_model_registry[self.state.sam_model_type](checkpoint=self.state.sam_checkpoint)
                sam.to(device=self.state.sam_device)
                self.state._sam_model = sam
                self.state._sam_predictor = SamPredictor(sam)
            except Exception as exc:
                self._set_sam_status("init_failed")
                client.add_notification(
                    "SAM3D",
                    f"Failed to init SAM: {exc}",
                    color="red",
                    auto_close_seconds=10,
                )
                return False
            self._set_sam_status("ready")
            client.add_notification(
                "SAM3D",
                "SAM is ready.",
                color="green",
                auto_close_seconds=3,
            )

        # Set image (per frame)
        try:
            self.state._sam_predictor.set_image(self.state.image_rgb_u8)
        except Exception as exc:
            self._set_sam_status("set_image_failed")
            client.add_notification(
                "SAM3D",
                f"Failed to set SAM image: {exc}",
                color="red",
                auto_close_seconds=8,
            )
            return False
        if self.state._sam_predictor is not None:
            self._set_sam_status("ready(image_set)")
        return True

    def _run_sam_and_update_mask(self, client) -> None:
        if self._backend_enabled():
            self._run_sam_and_update_mask_backend(client)
            return

        if self.state._sam_predictor is None:
            if not self._ensure_sam_ready(client):
                return

        with self.state.lock:
            obj = self._cur_object()
            points = np.array(obj.points_xy, dtype=np.float32) if obj.points_xy else None
            labels = np.array(obj.point_labels, dtype=np.int32) if obj.point_labels else None
            prev_logits = obj.logits

        if points is None or labels is None or len(points) == 0:
            with self.state.lock:
                obj = self._cur_object()
                obj.mask = None
                obj.logits = None
                obj.logits_key = None
                obj.mask_path = None
            self._refresh_frustum_image()
            self._update_status_handles()
            return

        try:
            if prev_logits is None:
                masks, scores, logits = self.state._sam_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                mask = masks[best].astype(bool)
                logits_best = logits[best]
            else:
                masks, _scores, logits = self.state._sam_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    mask_input=prev_logits[None, :, :],
                    multimask_output=False,
                )
                mask = masks[0].astype(bool)
                logits_best = logits[0]
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"SAM predict failed: {exc}",
                color="red",
                auto_close_seconds=8,
            )
            return

        with self.state.lock:
            obj = self._cur_object()
            obj.mask = mask
            obj.logits = logits_best
            obj.logits_key = None
            obj.mask_path = None

        self._refresh_frustum_image()
        self._update_status_handles()

    def _run_sam_and_update_mask_backend(self, client) -> None:
        app = self._app()
        with self.state.lock:
            obj = self._cur_object()
            points = np.array(obj.points_xy, dtype=np.float32) if obj.points_xy else None
            labels = np.array(obj.point_labels, dtype=np.int32) if obj.point_labels else None
            prev_logits_key = obj.logits_key

        if points is None or labels is None or len(points) == 0:
            with self.state.lock:
                obj = self._cur_object()
                obj.mask = None
                obj.logits_key = None
                obj.mask_path = None
            self._refresh_frustum_image()
            self._update_status_handles()
            return

        resp = self._backend_sam_predict(client, points, labels, prev_logits_key)
        if resp is None:
            return
        mask_path = Path(str(resp.get("mask_path", "")))
        logits_key = str(resp.get("logits_key") or "")
        if not mask_path.exists():
            client.add_notification(
                "SAM3D",
                f"Backend returned missing mask_path: {mask_path}",
                color="red",
                auto_close_seconds=8,
            )
            return
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bool = (mask > 0).astype(bool)
        with self.state.lock:
            obj = self._cur_object()
            obj.mask = mask_bool
            obj.mask_path = str(mask_path)
            obj.logits_key = logits_key if logits_key else None
            obj.logits = None

        self._refresh_frustum_image()
        self._update_status_handles()

    def _refresh_frustum_image(self) -> None:
        if self.state.image_disp_rgb_u8 is None:
            return
        img = self.state.image_disp_rgb_u8

        ds = int(max(1, self.state.image_downsample))

        with self.state.lock:
            masks: list[tuple[np.ndarray, tuple[int, int, int]]] = []
            for i, obj in enumerate(self.state.objects):
                if obj.mask is None:
                    continue
                mask = obj.mask
                # If image was padded to square, pad mask to match the padded image first.
                if self.state.image_rgb_u8 is not None:
                    try:
                        mask = self._pad_mask_to_match(
                            mask,
                            target_shape=self.state.image_rgb_u8.shape[:2],
                            pad_info=self.state.pad_info,
                        )
                    except Exception:
                        pass
                if ds > 1:
                    mask = np.array(
                        Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (img.shape[1], img.shape[0]), resample=Image.NEAREST
                        )
                    ).astype(bool)
                color = (255, 255, 0) if i != self.state.current_object_idx else (0, 255, 0)
                masks.append((mask, color))

            current = self._cur_object()
            points_xy = []
            for (x, y), label in zip(current.points_xy, current.point_labels):
                if ds > 1:
                    points_xy.append((int(round(x / ds)), int(round(y / ds)), label))
                else:
                    points_xy.append((x, y, label))

        eraser_circle = None

        overlay = _overlay_mask_and_points(img, masks=masks, points_xy=points_xy, eraser_circle=eraser_circle)
        opacity = float(self.state.frustum_opacity)
        alpha_u8 = None
        try:
            alpha_u8 = self.state.image_alpha_u8
        except Exception:
            alpha_u8 = None
        if alpha_u8 is not None:
            # Preserve per-pixel alpha (e.g. transparent padding), scaled by frustum opacity.
            if alpha_u8.shape[:2] != overlay.shape[:2]:
                alpha_u8 = np.array(
                    Image.fromarray(alpha_u8).resize(
                        (overlay.shape[1], overlay.shape[0]), resample=Image.NEAREST
                    )
                )
            a = (alpha_u8.astype(np.float32) * float(np.clip(opacity, 0.0, 1.0))).clip(0, 255).astype(np.uint8)
            overlay = np.concatenate([overlay, a[..., None]], axis=-1)
        elif opacity < 1.0:
            a = int(round(np.clip(opacity, 0.0, 1.0) * 255.0))
            alpha = np.full((overlay.shape[0], overlay.shape[1], 1), a, dtype=np.uint8)
            overlay = np.concatenate([overlay, alpha], axis=-1)

        if self.state.frustum_handle is not None and hasattr(self.state.frustum_handle, "image"):
            try:
                self.state.frustum_handle.image = overlay
                return
            except Exception:
                pass

        # Fallback: rebuild frustum.
        try:
            self._build_or_update_frustum()
        except Exception:
            pass
