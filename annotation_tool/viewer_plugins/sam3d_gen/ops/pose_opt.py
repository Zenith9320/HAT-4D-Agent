"""Pose optimization mixin for SAM3D generation operations.

This module contains pose optimization methods extracted from Sam3DGenOps.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .helpers import _ensure_sam3d_thirdparty_on_path

if TYPE_CHECKING:
    pass


class PoseOptMixin:
    """Mixin class providing pose optimization methods for SAM3D objects."""

    def _pose_opt_center_local_world(
        self,
        obj,
        frustum_wxyz: np.ndarray | None,
        frustum_pos: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        center_local = None
        # For pose-opt lock/debug, prefer a fixed point derived from splat geometry.
        # BBox AABB center can shift under rotation even when translation-X is locked.
        if obj.splat is not None:
            try:
                center_local = np.asarray(obj.splat.centers, dtype=np.float32).mean(axis=0)
            except Exception:
                center_local = None
        if center_local is None and obj.bbox_frame_handle is not None:
            try:
                center_local = np.array(obj.bbox_frame_handle.position, dtype=np.float32)
            except Exception:
                center_local = None
        if center_local is None:
            return None, None
        if frustum_wxyz is None or frustum_pos is None:
            return center_local, None
        R_wc = self._rotation_matrix_from_wxyz(np.asarray(frustum_wxyz, dtype=np.float32))
        center_world = center_local @ R_wc.T + np.asarray(frustum_pos, dtype=np.float32)
        return center_local, center_world

    def _update_pose_opt_lock_debug(self, client, msg: str) -> None:
        try:
            with self.state.lock:
                h = self.state.pose_opt_debug_handles.get(client)
            if h is not None:
                h.value = str(msg)
        except Exception:
            pass

    # ----------------------------
    # Pose optimization (best-effort)
    # ----------------------------
    def optimize_pose_current_object_async(
        self,
        client,
        *,
        object_idx: int | None = None,
        mask_path: str | None = None,
        steps: int,
        lr: float,
        min_side: int,
        rgb_weight: float,
        center_reg_weight: float,
        pose_reg_weight: float,
        scale_reg_weight: float,
        box_reg_weight: float,
        debug_mode: bool,
        lock_translation_x: bool,
        lock_translation_y: bool,
        lock_translation_z: bool,
    ) -> None:
        # Set running state
        with self.state.lock:
            self.state.pose_opt_running = True
            n_obj = int(len(getattr(self.state, "objects", []) or []))
            cur_idx = int(getattr(self.state, "current_object_idx", 0))
        target_obj_idx = int(cur_idx if object_idx is None else object_idx)
        if target_obj_idx < 0 or target_obj_idx >= n_obj:
            with self.state.lock:
                self.state.pose_opt_running = False
            client.add_notification(
                "SAM3D",
                f"Invalid pose-opt object idx: {target_obj_idx}",
                color="orange",
                auto_close_seconds=8,
            )
            return

        if self._backend_enabled():
            with self.state.lock:
                obj = self.state.objects[target_obj_idx]
                mask_path_use = str(mask_path).strip() if mask_path is not None else str(getattr(obj, "mask_path", "") or "").strip()
            if not mask_path_use or not Path(str(mask_path_use)).exists():
                with self.state.lock:
                    self.state.pose_opt_running = False
                client.add_notification(
                    "SAM3D",
                    "Mask path missing (backend mode). Re-run SAM predict first.",
                    color="orange",
                    auto_close_seconds=8,
                )
                return

            def _run_backend_and_cleanup():
                try:
                    self._optimize_pose_current_object_backend(
                        client,
                        int(target_obj_idx),
                        str(mask_path_use),
                        int(steps),
                        float(lr),
                        int(min_side),
                        float(rgb_weight),
                        float(center_reg_weight),
                        float(pose_reg_weight),
                        float(scale_reg_weight),
                        float(box_reg_weight),
                        bool(debug_mode),
                        bool(lock_translation_x),
                        bool(lock_translation_y),
                        bool(lock_translation_z),
                    )
                finally:
                    with self.state.lock:
                        self.state.pose_opt_running = False

            th = threading.Thread(target=_run_backend_and_cleanup, daemon=True)
            th.start()
            return

        # Best-effort: requires torch + pytorch3d.
        def _run_local_and_cleanup():
            try:
                self._optimize_pose_current_object(
                    client,
                    int(target_obj_idx),
                    int(steps),
                    float(lr),
                    int(min_side),
                    float(rgb_weight),
                    bool(debug_mode),
                    bool(lock_translation_x),
                    bool(lock_translation_y),
                    bool(lock_translation_z),
                )
            finally:
                with self.state.lock:
                    self.state.pose_opt_running = False

        th = threading.Thread(target=_run_local_and_cleanup, daemon=True)
        th.start()

    def _optimize_pose_current_object_backend(
        self,
        client,
        obj_idx: int,
        mask_path: str,
        steps: int,
        lr: float,
        min_side: int,
        rgb_weight: float,
        center_reg_weight: float,
        pose_reg_weight: float,
        scale_reg_weight: float,
        box_reg_weight: float,
        debug_mode: bool,
        lock_translation_x: bool,
        lock_translation_y: bool,
        lock_translation_z: bool,
    ) -> None:
        app = self._app()
        frame_token = app.sequence.frame_tokens[app.current_frame]
        img_path = app.workspace.image_path_for_view(frame_token, "input")
        try:
            with self.state.lock:
                if bool(getattr(self.state, "pad_square", False)) and self.state.padded_image_path:
                    img_path = Path(self.state.padded_image_path)
        except Exception:
            pass
        if not img_path.exists():
            client.add_notification(
                "SAM3D",
                f"Input image not found: {img_path}",
                color="red",
                auto_close_seconds=8,
            )
            return

        # Use the input-view frustum camera (not the current viewer camera).
        with self.state.lock:
            frustum_pos = None if self.state.frustum_position is None else self.state.frustum_position.copy()
            frustum_wxyz = None if self.state.frustum_wxyz is None else self.state.frustum_wxyz.copy()
            manual_scale = float(self._cur_object().manual_scale) if hasattr(self._cur_object(), "manual_scale") else 1.0
        if frustum_pos is None or frustum_wxyz is None:
            client.add_notification(
                "SAM3D",
                "Frustum pose missing. Click 'Begin SAM3D Gen' first.",
                color="red",
                auto_close_seconds=8,
            )
            return

        # Log frustum camera params (radius/az/el) + intrinsics (fov/image_size).
        try:
            from utils.camera_utils import vec_to_spherical

            target = np.asarray(app.camera_target, dtype=np.float32)
            r, yaw, pitch = vec_to_spherical(frustum_pos - target)
            az_deg = float(np.rad2deg(yaw))
            el_deg = float(np.rad2deg(pitch))
            fov_y_deg = float(getattr(app, "fov_deg", 49.1))
            img_size = int(getattr(app, "image_size", 512))
            sc_r = float(getattr(app, "scene_radius", 1.0))
            print(
                "[SAM3D PoseOpt] frustum(input) "
                f"cam_r={r:.4f} az_deg={az_deg:.2f} el_deg={el_deg:.2f} "
                f"fov_y_deg={fov_y_deg:.2f} image_size={img_size} scene_radius={sc_r:.4f}"
            )
        except Exception:
            pass
        try:
            base = app._sam3d_backend_base_url()
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"sam3d_backend_url not set: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            return

        out_dir = app.workspace.output_dir_for(frame_token, "input")
        sid = str(app.workspace.root.name) + "_" + str(frame_token)
        target = np.asarray(app.camera_target, dtype=np.float32)
        up = np.asarray(app.up_vec, dtype=np.float32)
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            pts = list(obj.points_xy)
            labs = list(obj.point_labels)
            mask_bool = None if obj.mask is None else obj.mask.copy()
            out = obj.sam3d_output
        try:
            img_path_pre, mask_path_pre = self._prepare_sam3d_image_and_mask(
                frame_token=frame_token,
                mask_path=Path(mask_path),
                mask_bool=mask_bool,
                points_xy=pts,
                point_labels=labs,
            )
        except Exception:
            img_path_pre, mask_path_pre = img_path, Path(mask_path)
        payload = {
            "image_path": str(img_path_pre),
            "flip_udlr": False,
            "mask_path": str(mask_path_pre),
            "output_dir": str(out_dir),
            "session_id": sid,
            "object_id": int(obj_idx),
            "steps": int(steps),
            "lr": float(lr),
            "min_side": int(min_side),
            "rgb_weight": float(rgb_weight),
            "debug_mode": bool(debug_mode),
            "align_pytorch3d_camera_to_viser": bool(self.state.align_pytorch3d_camera_to_viser),
            "center_reg_weight": float(center_reg_weight),
            "pose_reg_weight": float(pose_reg_weight),
            "scale_reg_weight": float(scale_reg_weight),
            "box_reg_weight": float(box_reg_weight),
            "init_scale_mult": float(manual_scale),
            "lock_translation_x": bool(lock_translation_x),
            "lock_translation_y": bool(lock_translation_y),
            "lock_translation_z": bool(lock_translation_z),
            # Default behavior: optimize translation/scale only; keep rotation fixed.
            "optimize_rotation": bool(getattr(self.state, "pose_opt_optimize_rotation", False)),
            # Match frustum (input view) intrinsics/extrinsics for projection.
            "camera_fov_deg": float(getattr(app, "fov_deg", 49.1)),
            "camera_wxyz": frustum_wxyz.reshape(-1).astype(float).tolist(),
            "camera_position": frustum_pos.reshape(-1).astype(float).tolist(),
            "camera_look_at": target.reshape(-1).astype(float).tolist(),
            "camera_up": up.reshape(-1).astype(float).tolist(),
        }
        target_center_world_np = None
        try:
            center_local, center_world = self._pose_opt_center_local_world(obj, frustum_wxyz, frustum_pos)
            if center_local is not None and center_world is not None:
                center_local_send = np.asarray(center_local, dtype=np.float32)
                if bool(self.state.align_pytorch3d_camera_to_viser):
                    R_fix = np.array(
                        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float32,
                    )
                    center_local_send = center_local_send @ R_fix
                anchor_local = None
                if out is not None:
                    try:
                        quat0 = out.get("rotation", None)
                        t0 = out.get("translation", None)
                        s0 = out.get("scale", None)
                        if quat0 is not None and t0 is not None and s0 is not None:
                            q0 = np.asarray(quat0.detach().cpu().numpy(), dtype=np.float32).reshape(4)
                            t0_np = np.asarray(t0.detach().cpu().numpy(), dtype=np.float32).reshape(3)
                            s0_np = np.asarray(s0.detach().cpu().numpy(), dtype=np.float32).reshape(3)
                            R0 = self._rotation_matrix_from_wxyz(q0)
                            anchor_local = (center_local_send - t0_np) @ R0.T
                            anchor_local = anchor_local / s0_np
                    except Exception:
                        anchor_local = None
                payload["target_center_world"] = center_world.reshape(-1).astype(float).tolist()
                if anchor_local is not None:
                    payload["lock_anchor_local"] = np.asarray(anchor_local, dtype=np.float32).reshape(-1).astype(float).tolist()
                target_center_world_np = np.asarray(center_world, dtype=np.float32)
        except Exception:
            pass
        if target_center_world_np is not None:
            self._update_pose_opt_lock_debug(
                client,
                "lock_target_world={:.4f},{:.4f},{:.4f}".format(
                    float(target_center_world_np[0]),
                    float(target_center_world_np[1]),
                    float(target_center_world_np[2]),
                ),
            )
            print(
                "[SAM3D PoseOpt][frontend][before] "
                f"obj={int(obj_idx)} world_x={float(target_center_world_np[0]):+.6f} "
                f"lock=(x={bool(lock_translation_x)},y={bool(lock_translation_y)},z={bool(lock_translation_z)})"
            )
        client.add_notification(
            "SAM3D",
            "Pose optimizing on backend... (may take a while)",
            color="orange",
            auto_close_seconds=6,
        )
        try:
            resp = app._post_json(base + "/sam3d/pose_opt", payload, timeout=3600)
        except Exception as exc:
            msg = str(exc)
            # Common backend-side failure: cache got cleared (backend restarted) so pose opt
            # can't reuse the previous SAM3D outputs. Best-effort: regenerate then retry.
            if "No cached SAM3D output" in msg:
                client.add_notification(
                    "SAM3D",
                    "Backend cache missing; regenerating 3D then retrying pose opt...",
                    color="orange",
                    auto_close_seconds=8,
                )
                gen_payload = {
                    "image_path": str(img_path_pre),
                    "flip_udlr": False,
                    "mask_path": str(mask_path_pre),
                    "output_dir": str(out_dir),
                    "session_id": sid,
                    "object_id": int(obj_idx),
                    "seed": int(self.state.sam3d_seed),
                    "align_pytorch3d_camera_to_viser": bool(self.state.align_pytorch3d_camera_to_viser),
                }
                try:
                    _ = app._post_json(base + "/sam3d/generate", gen_payload, timeout=3600)
                except Exception as exc2:
                    client.add_notification(
                        "SAM3D",
                        f"Backend pose opt failed (cache miss) and regenerate also failed: {exc2}",
                        color="red",
                        auto_close_seconds=14,
                    )
                    return
                try:
                    resp = app._post_json(base + "/sam3d/pose_opt", payload, timeout=3600)
                except Exception as exc3:
                    client.add_notification(
                        "SAM3D",
                        f"Backend pose opt retry failed: {exc3}",
                        color="red",
                        auto_close_seconds=14,
                    )
                    return
            else:
                client.add_notification(
                    "SAM3D",
                    f"Backend pose opt failed: {exc}",
                    color="red",
                    auto_close_seconds=12,
                )
                return
        ply_path = Path(str(resp.get("ply_path", "")))
        if not ply_path.exists():
            client.add_notification(
                "SAM3D",
                f"Backend returned missing ply_path: {ply_path}",
                color="red",
                auto_close_seconds=12,
            )
            return
        try:
            splat = self._load_splat_from_ply_aligned(ply_path)
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"Failed to load pose-opt PLY: {exc}",
                color="red",
                auto_close_seconds=12,
            )
            return
        with self.state.lock:
            if 0 <= obj_idx < len(self.state.objects):
                self.state.objects[obj_idx].splat = splat
                self.state.objects[obj_idx].manual_scale = 1.0
                self.state.objects[obj_idx].manual_scale_base_splat = None
        self._ensure_object_handle(obj_idx)
        self._reset_bbox_to_current_splat(client, obj_idx)
        self._update_status_handles()
        try:
            t_lock = resp.get("lock_target_world", None)
            f_lock = resp.get("lock_final_world", None)
            if t_lock is not None and f_lock is not None and len(t_lock) >= 3 and len(f_lock) >= 3:
                tx = float(t_lock[0])
                fx = float(f_lock[0])
                dx_backend = float(fx - tx)
                print(
                    "[SAM3D PoseOpt][frontend][after-backend-lock] "
                    f"obj={int(obj_idx)} world_x_before={tx:+.6f} "
                    f"world_x_after={fx:+.6f} dx={dx_backend:+.6f}"
                )
                self._update_pose_opt_lock_debug(
                    client,
                    "backend_lock_x target={:.4f} final={:.4f} dx={:+.6f}".format(
                        tx,
                        fx,
                        dx_backend,
                    ),
                )
        except Exception:
            pass
        try:
            with self.state.lock:
                obj = self.state.objects[obj_idx]
            center_local, center_world = self._pose_opt_center_local_world(obj, frustum_wxyz, frustum_pos)
            if center_world is not None and target_center_world_np is not None:
                dx = float(center_world[0] - target_center_world_np[0])
                print(
                    "[SAM3D PoseOpt][frontend][after] "
                    f"obj={int(obj_idx)} world_x_before={float(target_center_world_np[0]):+.6f} "
                    f"world_x_after={float(center_world[0]):+.6f} dx={dx:+.6f}"
                )
                self._update_pose_opt_lock_debug(
                    client,
                    "lock_target_x={:.4f} final_x={:.4f} dx={:+.6f}".format(
                        float(target_center_world_np[0]),
                        float(center_world[0]),
                        dx,
                    ),
                )
        except Exception:
            pass

        # Update pose-opt overlay preview in the UI (best-effort).
        try:
            overlay_path = resp.get("mask_overlay_path")
            if overlay_path:
                p = Path(str(overlay_path))
                if p.exists():
                    import imageio.v3 as iio

                    overlay_u8 = iio.imread(p)
                    with self.state.lock:
                        h = self.state.pose_opt_overlay_handles.get(client)
                    if h is not None:
                        try:
                            h.image = overlay_u8
                        except Exception:
                            pass
        except Exception:
            pass

        client.add_notification(
            "SAM3D",
            f"Pose opt done for obj={obj_idx}",
            color="green",
            auto_close_seconds=6,
        )

    def _optimize_pose_current_object(
        self,
        client,
        object_idx: int,
        steps: int,
        lr: float,
        min_side: int,
        rgb_weight: float,
        debug_mode: bool,
        lock_translation_x: bool,
        lock_translation_y: bool,
        lock_translation_z: bool,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as F
            from pytorch3d.renderer import (
                BlendParams,
                MeshRasterizer,
                MeshRenderer,
                PerspectiveCameras,
                RasterizationSettings,
                SoftSilhouetteShader,
            )
            from pytorch3d.structures import Meshes
            from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, so3_exp_map
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"Pose opt missing deps (torch/pytorch3d): {exc}",
                color="orange",
                auto_close_seconds=10,
            )
            return

        app = self._app()

        with self.state.lock:
            obj_idx = int(object_idx)
            if obj_idx < 0 or obj_idx >= len(self.state.objects):
                client.add_notification(
                    "SAM3D",
                    f"Invalid pose-opt object idx: {obj_idx}",
                    color="orange",
                    auto_close_seconds=8,
                )
                return
            obj = self.state.objects[obj_idx]
            out = obj.sam3d_output
            mask_np = None if obj.mask is None else obj.mask.copy()
            base_xyz = obj.gs_base_xyz
            base_rot = obj.gs_base_rot
            base_scale = obj.gs_base_scale
            base_min = obj.gs_base_min_kernel
            frustum_wxyz = None if self.state.frustum_wxyz is None else self.state.frustum_wxyz.copy()
            frustum_position = None if self.state.frustum_position is None else self.state.frustum_position.copy()
            do_align = bool(self.state.align_pytorch3d_camera_to_viser)
            center_local0, center_world0 = self._pose_opt_center_local_world(obj, frustum_wxyz, frustum_position)
            if center_world0 is not None:
                print(
                    "[SAM3D PoseOpt][frontend-local][before] "
                    f"obj={int(obj_idx)} world_x={float(center_world0[0]):+.6f} "
                    f"lock=(x={bool(lock_translation_x)},y={bool(lock_translation_y)},z={bool(lock_translation_z)})"
                )

        if out is None or mask_np is None:
            client.add_notification(
                "SAM3D",
                "Pose opt requires a generated object and a mask.",
                color="orange",
                auto_close_seconds=8,
            )
            return

        mesh_obj = out.get("glb", None)
        if mesh_obj is None or not hasattr(mesh_obj, "vertices") or not hasattr(mesh_obj, "faces"):
            client.add_notification(
                "SAM3D",
                "Pose opt requires mesh output (missing `out['glb']`).",
                color="orange",
                auto_close_seconds=10,
            )
            return

        if base_xyz is None or base_rot is None or base_scale is None or base_min is None:
            client.add_notification(
                "SAM3D",
                "Pose opt missing cached base gaussian state (regenerate object).",
                color="orange",
                auto_close_seconds=10,
            )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------
        # Target mask (downsample)
        # -----------------------
        gt = torch.from_numpy(mask_np.astype(np.float32))[None, None].to(device=device)
        H0, W0 = int(gt.shape[-2]), int(gt.shape[-1])
        scale_factor = float(min_side) / float(max(1, min(H0, W0)))
        Hr = max(32, int(round(H0 * scale_factor)))
        Wr = max(32, int(round(W0 * scale_factor)))
        if (Hr, Wr) != (H0, W0):
            gt = F.interpolate(gt, size=(Hr, Wr), mode="bilinear", align_corners=False)
            gt = (gt > 0.5).float()

        # -----------------------
        # Intrinsics (fallback from app fov)
        # -----------------------
        fov_y = float(app.fov_deg)
        fy_norm = 0.5 / max(1e-6, math.tan(math.radians(fov_y) * 0.5))
        fx_norm = fy_norm * (float(Hr) / max(1e-6, float(Wr)))
        fx = float(fx_norm * Wr)
        fy = float(fy_norm * Hr)
        cx = float(0.5 * Wr)
        cy = float(0.5 * Hr)

        cameras = PerspectiveCameras(
            focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
            principal_point=torch.tensor([[cx, cy]], device=device, dtype=torch.float32),
            image_size=torch.tensor([[Hr, Wr]], device=device, dtype=torch.float32),
            in_ndc=False,
            device=device,
        )
        raster_settings = RasterizationSettings(
            image_size=(Hr, Wr),
            blur_radius=1e-6,
            faces_per_pixel=30,
            max_faces_per_bin=50_000,
            # Avoid coarse rasterization bin overflows on dense meshes.
            bin_size=0,
        )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
        renderer_sil = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )

        # -----------------------
        # Mesh verts/faces (Y-up -> Z-up)
        # -----------------------
        verts_y = np.asarray(mesh_obj.vertices, dtype=np.float32)
        faces_np = np.asarray(mesh_obj.faces, dtype=np.int64)
        M_y2z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
        verts_z = verts_y @ M_y2z
        verts_z_t = torch.from_numpy(verts_z).to(device=device, dtype=torch.float32)
        faces_t = torch.from_numpy(faces_np).to(device=device, dtype=torch.int64)

        quat0 = out.get("rotation", None)
        t0 = out.get("translation", None)
        s0 = out.get("scale", None)
        if quat0 is None or t0 is None or s0 is None:
            client.add_notification(
                "SAM3D",
                "Pose opt output missing rotation/translation/scale.",
                color="orange",
                auto_close_seconds=10,
            )
            return

        quat0 = quat0.detach().to(device=device, dtype=torch.float32)  # (1,4) wxyz
        t0 = t0.detach().to(device=device, dtype=torch.float32)  # (1,3)
        s0 = s0.detach().to(device=device, dtype=torch.float32)  # (1,3)
        R0 = quaternion_to_matrix(quat0)  # (1,3,3)

        # Build local(camera)->world mapping used by the viewer:
        # local -> optional align fix -> frustum(camera_frame) -> world.
        if do_align:
            R_fix = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                device=device,
                dtype=torch.float32,
            )
        else:
            R_fix = torch.eye(3, device=device, dtype=torch.float32)
        if frustum_wxyz is not None and frustum_position is not None:
            cam_q = torch.tensor(frustum_wxyz, device=device, dtype=torch.float32).view(1, 4)
            R_cw = quaternion_to_matrix(cam_q)[0]
            t_cw = torch.tensor(frustum_position, device=device, dtype=torch.float32).view(1, 3)
        else:
            R_cw = torch.eye(3, device=device, dtype=torch.float32)
            t_cw = torch.zeros((1, 3), device=device, dtype=torch.float32)
        B_cam_to_world = torch.matmul(R_fix.T, R_cw.T)

        base_xyz_t_for_anchor = base_xyz.to(device=device, dtype=torch.float32)
        base_mu = base_xyz_t_for_anchor.mean(dim=0, keepdim=True)
        if center_local0 is not None and int(np.asarray(center_local0).shape[0]) == 3:
            center_local_use = np.asarray(center_local0, dtype=np.float32)
            if do_align:
                R_fix_np = np.array(
                    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                center_local_use = center_local_use @ R_fix_np
            center_cam = torch.tensor(center_local_use, device=device, dtype=torch.float32).view(1, 3)
            # Convert camera-space center into Gaussian local coordinates to keep locking consistent.
            anchor_local = torch.matmul(center_cam - t0, R0.transpose(1, 2))
            anchor_local = anchor_local / s0
            base_mu = anchor_local.detach()
        lock_world_x = bool(lock_translation_x)
        lock_world_y = bool(lock_translation_y)
        lock_world_z = bool(lock_translation_z)
        has_world_lock = bool(lock_world_x or lock_world_y or lock_world_z)

        omega = torch.nn.Parameter(torch.zeros((3,), device=device, dtype=torch.float32))
        log_s = torch.nn.Parameter(torch.zeros((1,), device=device, dtype=torch.float32))

        def _center_world_from_pose(R: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            mu_cam = torch.bmm((base_mu * s).unsqueeze(1), R).squeeze(1) + t
            return torch.matmul(mu_cam, B_cam_to_world) + t_cw

        with torch.no_grad():
            target_center_world = _center_world_from_pose(R0, t0, s0).detach()
            if center_world0 is not None:
                target_center_world = torch.tensor(center_world0, device=device, dtype=torch.float32).view(1, 3)
        try:
            if center_world0 is not None:
                self._update_pose_opt_lock_debug(
                    client,
                    "lock_target_world={:.4f},{:.4f},{:.4f}".format(
                        float(center_world0[0]),
                        float(center_world0[1]),
                        float(center_world0[2]),
                    ),
                )
        except Exception:
            pass

        def _anchor_world_from_pose(R: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            mu_cam = torch.bmm((base_mu * s).unsqueeze(1), R).squeeze(1)
            return torch.matmul(mu_cam, B_cam_to_world) + t_cw

        free_axes = [i for i, locked in enumerate([lock_world_x, lock_world_y, lock_world_z]) if not locked]
        delta_free = None
        trans_param: torch.nn.Parameter | None = None
        optimize_rotation = bool(getattr(self.state, "pose_opt_optimize_rotation", False))
        if has_world_lock:
            with torch.no_grad():
                anchor_world0 = _anchor_world_from_pose(R0, s0)
                delta_world0 = _center_world_from_pose(R0, t0, s0) - anchor_world0
                init_free = delta_world0[0, free_axes] if free_axes else torch.zeros((0,), device=device)
            if free_axes:
                delta_free = torch.nn.Parameter(init_free.clone().detach().to(device=device, dtype=torch.float32))
                trans_param = delta_free
        else:
            delta_t = torch.nn.Parameter(torch.zeros((3,), device=device, dtype=torch.float32))
            trans_param = delta_t

        # Prioritize translation (unlocked axes) + scale first, then only mild rotation refinement.
        trans_lr = float(lr)
        scale_lr = float(lr) * 0.85
        rot_lr = float(lr) * 0.12
        param_groups = [{"params": [log_s], "lr": scale_lr}]
        if bool(optimize_rotation):
            param_groups.insert(0, {"params": [omega], "lr": rot_lr})
        if trans_param is not None:
            param_groups.insert(1, {"params": [trans_param], "lr": trans_lr})
        opt = torch.optim.Adam(param_groups)
        warmup_steps = max(1, int(round(int(steps) * 0.65)))
        rot_max_rad = float(math.radians(6.0))

        def _compose_pose():
            if bool(optimize_rotation):
                R_delta = so3_exp_map(omega[None])  # (1,3,3)
                R = torch.bmm(R_delta, R0)
            else:
                R = R0
            s = s0 * torch.exp(log_s)[None].expand_as(s0)
            if not has_world_lock:
                t = t0 + delta_t[None]
                return R, t, s
            anchor_world = _anchor_world_from_pose(R, s)
            delta_world = torch.zeros_like(anchor_world)
            if lock_world_x:
                delta_world[:, 0] = target_center_world[:, 0] - anchor_world[:, 0]
            if lock_world_y:
                delta_world[:, 1] = target_center_world[:, 1] - anchor_world[:, 1]
            if lock_world_z:
                delta_world[:, 2] = target_center_world[:, 2] - anchor_world[:, 2]
            if delta_free is not None:
                for j, axis in enumerate(free_axes):
                    delta_world[:, axis] = delta_free[j]
            t = torch.matmul(delta_world, B_cam_to_world.T)
            return R, t, s

        def _render_silhouette(R: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            v = verts_z_t[None] * s[:, None, :]
            v = torch.bmm(v, R) + t[:, None, :]
            mesh = Meshes(verts=[v[0]], faces=[faces_t])
            out_img = renderer_sil(mesh)
            return out_img[..., 3].clamp(0.0, 1.0)  # (1,H,W)

        def _dice_loss(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            eps = 1e-6
            num = 2.0 * (p * g).sum()
            den = (p + g).sum() + eps
            return 1.0 - (num + eps) / den

        if self.state.gen_status_handle is not None:
            try:
                self.state.gen_status_handle.value = "pose_opt_running"
            except Exception:
                pass

        # Prepare for real-time visualization
        try:
            _ensure_sam3d_thirdparty_on_path()
            from pytorch3d.transforms import quaternion_invert, quaternion_multiply, quaternion_to_matrix as quat_to_mat
            from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
            can_realtime_viz = True
        except Exception:
            can_realtime_viz = False

        for it in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            Rn, tn, sn = _compose_pose()
            pred = _render_silhouette(Rn, tn, sn)
            loss_dice = _dice_loss(pred, gt)
            loss_reg_rot = (omega * omega).sum()
            if not bool(optimize_rotation):
                loss_reg_rot = torch.tensor(0.0, device=device, dtype=torch.float32)
            loss_reg_scale = (log_s * log_s).sum()
            loss_reg_trans = torch.tensor(0.0, device=device, dtype=torch.float32)
            if trans_param is not None:
                loss_reg_trans = (trans_param * trans_param).sum()
            # Penalize large camera-Z drift so silhouette-size mismatch is not solved only by pushing/pulling depth.
            loss_reg_depth = (tn[:, 2] - t0[:, 2]).pow(2).mean()
            loss = (
                loss_dice
                + 4e-3 * loss_reg_rot
                + 6e-4 * loss_reg_scale
                + 2e-4 * loss_reg_trans
                + 5e-4 * loss_reg_depth
            )
            loss.backward()
            # Stage-1: freeze rotation updates so translation/scale converge first.
            if bool(optimize_rotation) and it < warmup_steps and omega.grad is not None:
                omega.grad.zero_()
            opt.step()
            with torch.no_grad():
                if bool(optimize_rotation) and it < warmup_steps:
                    omega.zero_()
                elif bool(optimize_rotation):
                    n = torch.linalg.norm(omega)
                    if torch.isfinite(n) and n > rot_max_rad:
                        omega.mul_(rot_max_rad / (n + 1e-8))

            # Real-time visualization every 10 iterations
            if can_realtime_viz and (it % 10 == 0 or it == int(steps) - 1):
                try:
                    with torch.no_grad():
                        Rn_viz, tn_viz, sn_viz = _compose_pose()
                        quat_viz = matrix_to_quaternion(Rn_viz)
                        gs_live = out["gaussian"][0]
                        R_pose = quat_to_mat(quat_viz)
                        l2c = compose_transform(scale=sn_viz, rotation=R_pose, translation=tn_viz)
                        pts_scene = l2c.transform_points(base_xyz.unsqueeze(0))[0]
                        gs_live.from_xyz(pts_scene)

                        q_inv = quaternion_invert(quat_viz).expand(base_rot.shape[0], -1)
                        gs_live.from_rotation(quaternion_multiply(q_inv, base_rot))

                        adjusted_scale = base_scale * sn_viz.expand(base_scale.shape[0], -1)
                        kernel_scale = float(sn_viz[0].mean().item())
                        gs_live.mininum_kernel_size = float(base_min) * kernel_scale
                        adjusted_scale = torch.maximum(
                            adjusted_scale,
                            torch.tensor(gs_live.mininum_kernel_size * 1.1, device=adjusted_scale.device, dtype=adjusted_scale.dtype),
                        )
                        gs_live.from_scaling(adjusted_scale)

                        if bool(self.state.align_pytorch3d_camera_to_viser):
                            self._sam3d_align_pytorch3d_camera_to_viser_in_place(gs_live)

                        out["gs"] = gs_live

                    # Update displayed splat
                    splat_live = self._splat_from_sam3d_output(out)
                    with self.state.lock:
                        self.state.objects[obj_idx].splat = splat_live
                    self._ensure_object_handle(obj_idx)
                except Exception:
                    pass

            # Update progress status
            if self.state.gen_status_handle is not None:
                try:
                    progress = int(100.0 * (it + 1) / steps)
                    self.state.gen_status_handle.value = f"pose_opt {progress}% loss={float(loss.item()):.4f}"
                except Exception:
                    pass

        with torch.no_grad():
            Rn, tn, sn = _compose_pose()
            quat_new = matrix_to_quaternion(Rn)  # (1,4) wxyz

        # Commit pose into the output dict.
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            if obj.sam3d_output is None:
                return
            obj.sam3d_output["rotation"] = quat_new.detach()
            obj.sam3d_output["translation"] = tn.detach()
            obj.sam3d_output["scale"] = sn.detach()

        # Re-apply refined pose deterministically to gaussian using cached base.
        try:
            _ensure_sam3d_thirdparty_on_path()
            from pytorch3d.transforms import quaternion_invert, quaternion_multiply, quaternion_to_matrix
            from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

            gs_live = out["gaussian"][0]
            R_pose = quaternion_to_matrix(quat_new)
            l2c = compose_transform(scale=sn, rotation=R_pose, translation=tn)
            pts_scene = l2c.transform_points(base_xyz.unsqueeze(0))[0]
            gs_live.from_xyz(pts_scene)

            q_inv = quaternion_invert(quat_new).expand(base_rot.shape[0], -1)
            gs_live.from_rotation(quaternion_multiply(q_inv, base_rot))

            adjusted_scale = base_scale * sn.expand(base_scale.shape[0], -1)
            kernel_scale = float(sn[0].mean().item())
            gs_live.mininum_kernel_size = float(base_min) * kernel_scale
            adjusted_scale = torch.maximum(
                adjusted_scale,
                torch.tensor(gs_live.mininum_kernel_size * 1.1, device=adjusted_scale.device, dtype=adjusted_scale.dtype),
            )
            gs_live.from_scaling(adjusted_scale)

            if bool(self.state.align_pytorch3d_camera_to_viser):
                self._sam3d_align_pytorch3d_camera_to_viser_in_place(gs_live)

            out["gs"] = gs_live
            with self.state.lock:
                obj = self.state.objects[obj_idx]
                obj.pose_applied = True
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"Pose opt apply failed: {exc}",
                color="orange",
                auto_close_seconds=10,
            )
            return

        # Update displayed splats.
        try:
            splat = self._splat_from_sam3d_output(out)
            with self.state.lock:
                self.state.objects[obj_idx].splat = splat
            self._ensure_object_handle(obj_idx)
            self._reset_bbox_to_current_splat(client, obj_idx)
            with self.state.lock:
                obj = self.state.objects[obj_idx]
            center_local2, center_world2 = self._pose_opt_center_local_world(obj, frustum_wxyz, frustum_position)
            if center_world0 is not None and center_world2 is not None:
                dx = float(center_world2[0] - center_world0[0])
                print(
                    "[SAM3D PoseOpt][frontend-local][after] "
                    f"obj={int(obj_idx)} world_x_before={float(center_world0[0]):+.6f} "
                    f"world_x_after={float(center_world2[0]):+.6f} dx={dx:+.6f}"
                )
                self._update_pose_opt_lock_debug(
                    client,
                    "lock_target_x={:.4f} final_x={:.4f} dx={:+.6f}".format(
                        float(center_world0[0]),
                        float(center_world2[0]),
                        dx,
                    ),
                )
        except Exception:
            pass

        if self.state.gen_status_handle is not None:
            try:
                self.state.gen_status_handle.value = "pose_opt_done"
            except Exception:
                pass
        client.add_notification(
            "SAM3D",
            "Pose opt done.",
            color="green",
            auto_close_seconds=6,
        )
