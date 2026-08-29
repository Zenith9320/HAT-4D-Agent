"""SAM3D generation mixin class.

This module contains the Sam3DMixin class with methods for SAM3D-based
3D object generation from segmentation masks.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

import viser.transforms as tf

from utils.constants import COORD_ALIGN_MAT, SH_C0
from utils.geometry_utils import transform_gaussians
from utils.ply_utils import read_ply_vertices
from utils.ply_utils import SplatData
from utils.ply_utils import inverse_transform_gaussians, write_ply_vertices

from .helpers import (
    _logit,
    _pack_splat_buffer,
    _downsample_splat_for_display,
    _default_ply_header_lines,
    _default_ply_dtype,
    _fill_vertices_from_gaussians,
    _ensure_sam3d_thirdparty_on_path,
    _repo_root,
    _sam3d_repo_root,
)


class Sam3DMixin:
    """Mixin class providing SAM3D generation methods."""

    def add_object_to_session_ply(self, client) -> None:
        """Append ALL objects' gaussians into the session PLY for the current frame, then auto-exit."""
        app = self._app()
        with self.state.lock:
            frame_token = self.state.frame_token
            all_objects = list(self.state.objects)

        if frame_token is None:
            client.add_notification(
                "SAM3D",
                "No active frame. Click 'Begin SAM3D Gen' first.",
                color="orange",
                auto_close_seconds=6,
            )
            return

        # Collect all objects that have splats
        objects_with_splat = []
        for idx, obj in enumerate(all_objects):
            if obj.splat is not None:
                objects_with_splat.append((idx, obj.splat))

        if not objects_with_splat:
            client.add_notification(
                "SAM3D",
                "No objects with generated GS. Generate 3D first.",
                color="orange",
                auto_close_seconds=6,
            )
            return

        if self.state.frustum_wxyz is None or self.state.frustum_position is None:
            client.add_notification(
                "SAM3D",
                "Frustum pose missing. Click 'Begin SAM3D Gen' first.",
                color="red",
                auto_close_seconds=8,
            )
            return

        # Resolve the session PLY path for the current frame.
        try:
            ply_path = app.sequence.gs_path(app.current_frame)
        except Exception:
            ply_path = app.workspace.gs_dir / f"{frame_token}.ply"
        ply_path = Path(ply_path)
        ply_path.parent.mkdir(parents=True, exist_ok=True)

        # Camera transform for all objects
        R_wc = tf.SO3(self.state.frustum_wxyz).as_matrix().astype(np.float32)
        t_wc = self.state.frustum_position.astype(np.float32)

        # Collect all transformed gaussian data from all objects
        all_centers_src = []
        all_rotations_src = []
        all_scales_src_log = []
        all_opacity_raw = []
        all_sh_dc = []

        for obj_idx, splat_local in objects_with_splat:
            # 1) Local(camera) -> world (aligned viewer coords): apply camera_frame transform.
            centers_w, rotations_w, scales_w, cov_w = transform_gaussians(
                centers=splat_local.centers,
                rotations=splat_local.rotations,
                scales=splat_local.scales,
                A=R_wc,
                return_cov=True,
            )
            centers_w = centers_w + t_wc[None, :]

            # 2) World(aligned) -> source PLY coords: inverse COORD_ALIGN_MAT.
            centers_src, rotations_src, scales_src = inverse_transform_gaussians(
                centers=centers_w,
                rotations=rotations_w,
                scales=scales_w,
                A=COORD_ALIGN_MAT,
            )
            scales_src_log = np.log(np.clip(scales_src, 1e-8, None)).astype(np.float32)
            opacity_raw = splat_local.opacity_raw.astype(np.float32).reshape(-1)
            sh_dc = splat_local.sh.astype(np.float32)

            all_centers_src.append(centers_src)
            all_rotations_src.append(rotations_src)
            all_scales_src_log.append(scales_src_log)
            all_opacity_raw.append(opacity_raw)
            all_sh_dc.append(sh_dc)

        # Concatenate all objects' data
        centers_src = np.concatenate(all_centers_src, axis=0)
        rotations_src = np.concatenate(all_rotations_src, axis=0)
        scales_src_log = np.concatenate(all_scales_src_log, axis=0)
        opacity_raw = np.concatenate(all_opacity_raw, axis=0)
        sh_dc = np.concatenate(all_sh_dc, axis=0)

        # Load existing PLY (if present) and append; otherwise create a new minimal PLY.
        if ply_path.exists():
            try:
                _prop_names, verts_old, header_lines = read_ply_vertices(ply_path, return_header=True)
            except Exception as exc:
                client.add_notification(
                    "SAM3D",
                    f"Failed to read existing PLY: {exc}",
                    color="red",
                    auto_close_seconds=10,
                )
                return
            n_old = int(len(verts_old))
            n_new = int(len(centers_src))
            verts_all = np.zeros(n_old + n_new, dtype=verts_old.dtype)
            verts_all[:n_old] = verts_old
            # Defaults for any extra fields remain zeros.
            _fill_vertices_from_gaussians(
                verts_all[n_old:],
                centers_src=centers_src,
                rotations_src_wxyz=rotations_src,
                scales_src_log=scales_src_log,
                opacity_raw=opacity_raw,
                sh_dc=sh_dc,
            )
            try:
                write_ply_vertices(header_lines, verts_all, ply_path)
            except Exception as exc:
                client.add_notification(
                    "SAM3D",
                    f"Failed to write appended PLY: {exc}",
                    color="red",
                    auto_close_seconds=10,
                )
                return
        else:
            n_new = int(len(centers_src))
            header_lines = _default_ply_header_lines(n_new)
            verts_all = np.zeros(n_new, dtype=_default_ply_dtype())
            _fill_vertices_from_gaussians(
                verts_all,
                centers_src=centers_src,
                rotations_src_wxyz=rotations_src,
                scales_src_log=scales_src_log,
                opacity_raw=opacity_raw,
                sh_dc=sh_dc,
            )
            try:
                write_ply_vertices(header_lines, verts_all, ply_path)
            except Exception as exc:
                client.add_notification(
                    "SAM3D",
                    f"Failed to write new PLY: {exc}",
                    color="red",
                    auto_close_seconds=10,
                )
                return

        # Invalidate cache + reload current frame so user sees the merged scene.
        try:
            if hasattr(app.sequence, "cache") and app.current_frame in app.sequence.cache:
                app.sequence.cache.pop(app.current_frame, None)
        except Exception:
            pass
        try:
            preview_handle = app.preview_handles.get(client)
            if preview_handle is not None:
                # Applying a frame can be heavy for large GS (PLY parse + display subset + websocket buffer).
                # Schedule it in the app's worker thread so the UI doesn't block/white-screen.
                try:
                    app._schedule_apply_frame(client, int(app.current_frame), preview_handle, debounce=False)
                except Exception:
                    app._apply_frame(int(app.current_frame), preview_handle)
        except Exception:
            pass

        added_indices = [idx for idx, _ in objects_with_splat]
        client.add_notification(
            "SAM3D",
            f"Appended {len(objects_with_splat)} objects (idx={added_indices}) to PLY: {ply_path.name}",
            color="green",
            auto_close_seconds=6,
        )

        # Auto-exit SAM3D Gen mode after adding all objects
        self.end(client)

    def _ensure_sam3d_pipeline_ready(self, client) -> bool:
        _ensure_sam3d_thirdparty_on_path()

        if self.state._sam3d_pipeline is not None:
            self._set_sam3d_status("ready")
            return True
        self._set_sam3d_status("initializing")
        if not self.state._sam3d_init_notified:
            self.state._sam3d_init_notified = True
            client.add_notification(
                "SAM3D",
                "Initializing SAM3D pipeline... (first time may take a while)",
                color="orange",
                auto_close_seconds=6,
            )

        # Ask hydra to include chained exceptions, and also preflight-import the target
        # so we can surface the real ImportError (Hydra otherwise shows "Error locating target ...").
        os.environ.setdefault("HYDRA_FULL_ERROR", "1")
        try:
            import importlib

            mod = importlib.import_module("sam3d_objects.pipeline.inference_pipeline_pointmap")
            getattr(mod, "InferencePipelinePointMap")
        except Exception as exc:
            self._set_sam3d_status("target_import_failed")
            client.add_notification(
                "SAM3D",
                f"Failed to import pipeline target: {type(exc).__name__}: {exc}",
                color="red",
                auto_close_seconds=12,
            )
            return False

        sam3d_root = _sam3d_repo_root()
        if self.state.sam3d_pipeline_config is None:
            for default_pipeline in (
                sam3d_root / "checkpoints" / "hf" / "pipeline.yaml",
                sam3d_root / "checkpoints" / "pipeline.yaml",
            ):
                if default_pipeline.exists():
                    self.state.sam3d_pipeline_config = str(default_pipeline)
                    break

        if self.state.sam3d_pipeline_config is None or not Path(self.state.sam3d_pipeline_config).exists():
            self._set_sam3d_status("config_missing")
            client.add_notification(
                "SAM3D",
                "SAM3D pipeline config not found. Set `state.sam3d_pipeline_config` to a valid pipeline.yaml.",
                color="red",
                auto_close_seconds=10,
            )
            return False

        try:
            import torch
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
        except Exception as exc:
            self._set_sam3d_status("deps_missing")
            client.add_notification(
                "SAM3D",
                f"Missing deps for SAM3D pipeline: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            return False

        try:
            config = OmegaConf.load(str(self.state.sam3d_pipeline_config))
            if self.state.sam3d_depth_model_path:
                config.depth_model.model.pretrained_model_name_or_path = self.state.sam3d_depth_model_path
            config.rendering_engine = "pytorch3d"
            config.compile_model = bool(self.state.sam3d_compile)
            config.workspace_dir = str(Path(self.state.sam3d_pipeline_config).parent)
            self.state._sam3d_pipeline = instantiate(config)
            _ = torch  # keep reference
        except Exception as exc:
            self._set_sam3d_status("init_failed")
            # Try to surface the root cause (Hydra can hide the underlying ImportError).
            msg = str(exc)
            cause = exc
            for _ in range(6):
                nxt = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
                if nxt is None:
                    break
                cause = nxt
            if cause is not exc:
                msg = msg + f" | Root cause: {type(cause).__name__}: {cause}"
            if "Error locating target" in msg and "sam3d_objects.pipeline" in msg:
                msg = (
                    msg
                    + " | Hint: `sam3d_objects/__init__.py` requires `LIDRA_SKIP_INIT=true` "
                    + "(plugin sets this automatically; restart the server if code was updated)."
                )
            client.add_notification(
                "SAM3D",
                f"Failed to init SAM3D pipeline: {msg}",
                color="red",
                auto_close_seconds=12,
            )
            return False
        self._set_sam3d_status("ready")
        client.add_notification(
            "SAM3D",
            "SAM3D pipeline is ready.",
            color="green",
            auto_close_seconds=3,
        )
        return True

    def generate_current_object_async(self, client) -> None:
        with self.state.lock:
            obj = self._cur_object()
            mask = None if obj.mask is None else obj.mask.copy()
            mask_path = obj.mask_path
        if mask is None:
            client.add_notification(
                "SAM3D",
                "No mask yet. Add prompt points first.",
                color="orange",
                auto_close_seconds=6,
            )
            return

        if self._backend_enabled():
            if not mask_path:
                client.add_notification(
                    "SAM3D",
                    "Mask path missing (backend mode). Re-run SAM predict.",
                    color="red",
                    auto_close_seconds=8,
                )
                return
            if self.state.gen_status_handle is not None:
                try:
                    self.state.gen_status_handle.value = "running(backend)"
                except Exception:
                    pass
            th = threading.Thread(
                target=self._generate_object_backend,
                args=(client, self.state.current_object_idx, str(mask_path)),
                daemon=True,
            )
            th.start()
            return

        if not self._ensure_sam3d_pipeline_ready(client):
            return

        if self.state.gen_status_handle is not None:
            try:
                self.state.gen_status_handle.value = "running"
            except Exception:
                pass

        th = threading.Thread(target=self._generate_object, args=(client, self.state.current_object_idx, mask), daemon=True)
        th.start()

    def _generate_object_backend(self, client, obj_idx: int, mask_path: str) -> None:
        app = self._app()
        frame_token = app.sequence.frame_tokens[app.current_frame]
        img_path = app.workspace.image_path_for_view(frame_token, "input")
        if not img_path.exists():
            return
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            pts = list(obj.points_xy)
            labs = list(obj.point_labels)
            mask_bool = None if obj.mask is None else obj.mask.copy()
        try:
            img_path, mask_path_prepared = self._prepare_sam3d_image_and_mask(
                frame_token=frame_token,
                mask_path=Path(mask_path),
                mask_bool=mask_bool,
                points_xy=pts,
                point_labels=labs,
            )
        except Exception:
            mask_path_prepared = Path(mask_path)

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
        st = self._backend_fetch_status(client, timeout=2.0)
        if isinstance(st, dict):
            self._backend_apply_status(client, st, notify=True)
        client.add_notification(
            "SAM3D",
            "Generating 3D on backend... (this can take a while)",
            color="orange",
            auto_close_seconds=6,
        )

        out_dir = app.workspace.output_dir_for(frame_token, "input")
        sid = str(app.workspace.root.name) + "_" + str(frame_token)
        payload = {
            "image_path": str(img_path),
            "flip_udlr": False,
            "mask_path": str(mask_path_prepared),
            "output_dir": str(out_dir),
            "session_id": sid,
            "object_id": int(obj_idx),
            "seed": int(self.state.sam3d_seed),
            "align_pytorch3d_camera_to_viser": bool(self.state.align_pytorch3d_camera_to_viser),
        }
        try:
            resp = app._post_json(base + "/sam3d/generate", payload, timeout=3600)
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"Backend generate failed: {exc}",
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
                f"Failed to load generated PLY: {exc}",
                color="red",
                auto_close_seconds=12,
            )
            return
        with self.state.lock:
            self.state.objects[obj_idx].splat = splat
            self.state.objects[obj_idx].manual_scale = 1.0
            self.state.objects[obj_idx].manual_scale_base_splat = None
        self._ensure_object_handle(obj_idx)
        self._ensure_bbox_for_object(client, obj_idx)
        if self.state.gen_status_handle is not None:
            try:
                self.state.gen_status_handle.value = "done(backend)"
            except Exception:
                pass
        self._update_status_handles()
        client.add_notification(
            "SAM3D",
            f"Generated 3D for obj={obj_idx}",
            color="green",
            auto_close_seconds=6,
        )

        # Auto-run pose optimization if requested (backend mode).
        try:
            with self.state.lock:
                auto = bool(getattr(self.state, "pose_opt_after_gen", False))
        except Exception:
            auto = False
        if auto:
            try:
                with self.state.lock:
                    steps = int(getattr(self.state, "pose_opt_steps", 160))
                    lr = float(getattr(self.state, "pose_opt_lr", 1e-3))
                    min_side = int(getattr(self.state, "pose_opt_min_side", 512))
                    rgb_weight = float(getattr(self.state, "pose_opt_rgb_weight", 0.4))
                    center_reg_weight = float(getattr(self.state, "pose_opt_center_reg_weight", 1e-2))
                    pose_reg_weight = float(getattr(self.state, "pose_opt_pose_reg_weight", 1e-3))
                    scale_reg_weight = float(getattr(self.state, "pose_opt_scale_reg_weight", 1e-5))
                    box_reg_weight = float(getattr(self.state, "pose_opt_box_reg_weight", 1e-1))
                    debug_mode = bool(getattr(self.state, "pose_opt_debug_mode", False))
                    lock_tx = bool(getattr(self.state, "pose_opt_lock_tx", False))
                    lock_ty = bool(getattr(self.state, "pose_opt_lock_ty", False))
                    lock_tz = bool(getattr(self.state, "pose_opt_lock_tz", False))
                self.optimize_pose_current_object_async(
                    client,
                    object_idx=int(obj_idx),
                    mask_path=str(mask_path),
                    steps=steps,
                    lr=lr,
                    min_side=min_side,
                    rgb_weight=rgb_weight,
                    center_reg_weight=center_reg_weight,
                    pose_reg_weight=pose_reg_weight,
                    scale_reg_weight=scale_reg_weight,
                    box_reg_weight=box_reg_weight,
                    debug_mode=debug_mode,
                    lock_translation_x=lock_tx,
                    lock_translation_y=lock_ty,
                    lock_translation_z=lock_tz,
                )
            except Exception:
                pass

    def _load_splat_from_ply_aligned(self, path: Path) -> SplatData:
        _, v = read_ply_vertices(path)
        centers = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        sh = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
        rgbs = np.clip(0.5 + SH_C0 * sh, 0.0, 1.0).astype(np.float32)

        opacity_raw = v["opacity"].astype(np.float32)
        opacities = (1.0 / (1.0 + np.exp(-opacity_raw)))[:, None].astype(np.float32)

        scales_raw = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
        scales = np.exp(scales_raw).astype(np.float32)
        rotations = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

        centers2, rotations2, scales2, cov = transform_gaussians(
            centers=centers,
            rotations=rotations,
            scales=scales,
            A=np.eye(3, dtype=np.float32),
            return_cov=True,
        )
        return SplatData(
            centers=centers2,
            rgbs=rgbs,
            opacities=opacities,
            covariances=cov,
            scales=scales2,
            rotations=rotations2,
            sh=sh,
            opacity_raw=opacity_raw,
            centers_raw=centers,
        )

    def _generate_object(self, client, obj_idx: int, mask: np.ndarray) -> None:
        if self.state.image_rgb_u8 is None:
            return
        pipeline = self.state._sam3d_pipeline
        if pipeline is None:
            return

        rgba = np.concatenate(
            [self.state.image_rgb_u8[..., :3], (mask.astype(np.uint8) * 255)[..., None]], axis=-1
        )

        try:
            out = pipeline.run(
                rgba,
                None,
                int(self.state.sam3d_seed),
                stage1_only=False,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=True,
                use_vertex_color=True,
                stage1_inference_steps=None,
                pointmap=None,
            )
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"SAM3D generation failed: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            if self.state.gen_status_handle is not None:
                try:
                    self.state.gen_status_handle.value = "failed"
                except Exception:
                    pass
            return

        with self.state.lock:
            if obj_idx < 0 or obj_idx >= len(self.state.objects):
                return
            obj = self.state.objects[obj_idx]
            obj.sam3d_output = out

        # Cache base gaussian state (Gaussian frame) for later pose refinement.
        try:
            if out is not None and "gaussian" in out and out["gaussian"]:
                gs_local = out["gaussian"][0]
                import torch

                with self.state.lock:
                    obj = self.state.objects[obj_idx]
                    obj.gs_base_xyz = gs_local.get_xyz.detach().clone()
                    obj.gs_base_rot = gs_local.get_rotation.detach().clone()
                    obj.gs_base_scale = gs_local.get_scaling.detach().clone()
                    obj.gs_base_min_kernel = float(gs_local.mininum_kernel_size)
                    obj.pose_applied = False
                _ = torch
        except Exception:
            pass

        # Apply pose (and optionally align camera frame) so the gaussian is in the frustum camera coordinates.
        try:
            with self.state.lock:
                obj = self.state.objects[obj_idx]
            self._ensure_output_pose_applied(out, obj)
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"Failed to apply pose to SAM3D gaussian: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            if self.state.gen_status_handle is not None:
                try:
                    self.state.gen_status_handle.value = "failed"
                except Exception:
                    pass
            return

        try:
            splat = self._splat_from_sam3d_output(out)
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"Failed to convert SAM3D gaussian to splats: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            if self.state.gen_status_handle is not None:
                try:
                    self.state.gen_status_handle.value = "failed"
                except Exception:
                    pass
            return

        with self.state.lock:
            self.state.objects[obj_idx].splat = splat
            self.state.objects[obj_idx].manual_scale = 1.0
            self.state.objects[obj_idx].manual_scale_base_splat = None

        self._ensure_object_handle(obj_idx)
        self._ensure_bbox_for_object(client, obj_idx)

        if self.state.gen_status_handle is not None:
            try:
                self.state.gen_status_handle.value = "done"
            except Exception:
                pass
        self._update_status_handles()
        client.add_notification(
            "SAM3D",
            f"Generated 3D for obj={obj_idx}",
            color="green",
            auto_close_seconds=6,
        )

        # Auto-run pose optimization if requested (local mode).
        try:
            with self.state.lock:
                auto = bool(getattr(self.state, "pose_opt_after_gen", False))
        except Exception:
            auto = False
        if auto:
            try:
                with self.state.lock:
                    steps = int(getattr(self.state, "pose_opt_steps", 160))
                    lr = float(getattr(self.state, "pose_opt_lr", 1e-3))
                    min_side = int(getattr(self.state, "pose_opt_min_side", 512))
                    rgb_weight = float(getattr(self.state, "pose_opt_rgb_weight", 0.4))
                    center_reg_weight = float(getattr(self.state, "pose_opt_center_reg_weight", 1e-2))
                    pose_reg_weight = float(getattr(self.state, "pose_opt_pose_reg_weight", 1e-3))
                    scale_reg_weight = float(getattr(self.state, "pose_opt_scale_reg_weight", 1e-5))
                    box_reg_weight = float(getattr(self.state, "pose_opt_box_reg_weight", 1e-1))
                    debug_mode = bool(getattr(self.state, "pose_opt_debug_mode", False))
                    lock_tx = bool(getattr(self.state, "pose_opt_lock_tx", False))
                    lock_ty = bool(getattr(self.state, "pose_opt_lock_ty", False))
                    lock_tz = bool(getattr(self.state, "pose_opt_lock_tz", False))
                self.optimize_pose_current_object_async(
                    client,
                    object_idx=int(obj_idx),
                    steps=steps,
                    lr=lr,
                    min_side=min_side,
                    rgb_weight=rgb_weight,
                    center_reg_weight=center_reg_weight,
                    pose_reg_weight=pose_reg_weight,
                    scale_reg_weight=scale_reg_weight,
                    box_reg_weight=box_reg_weight,
                    debug_mode=debug_mode,
                    lock_translation_x=lock_tx,
                    lock_translation_y=lock_ty,
                    lock_translation_z=lock_tz,
                )
            except Exception:
                pass

    def _ensure_output_pose_applied(self, out: dict, obj) -> None:
        if obj.pose_applied:
            return
        gs = self._sam3d_apply_pose_to_output_gaussian_in_place(out)
        if bool(self.state.align_pytorch3d_camera_to_viser):
            self._sam3d_align_pytorch3d_camera_to_viser_in_place(gs)
        obj.pose_applied = True

    def _sam3d_apply_pose_to_output_gaussian_in_place(self, output: dict) -> Any:
        import torch
        from pytorch3d.transforms import quaternion_invert, quaternion_multiply, quaternion_to_matrix

        _ensure_sam3d_thirdparty_on_path()
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

        if "gaussian" not in output or not output["gaussian"]:
            raise KeyError("output missing gaussian")
        gs = output["gaussian"][0]

        quat_l2c = output.get("rotation", None)
        trans_l2c = output.get("translation", None)
        scale_l2c = output.get("scale", None)
        if quat_l2c is None or trans_l2c is None or scale_l2c is None:
            raise KeyError("output missing pose keys (rotation/translation/scale)")

        points_local = gs.get_xyz.unsqueeze(0)  # (1,N,3)
        R_l2c = quaternion_to_matrix(quat_l2c)
        l2c = compose_transform(scale=scale_l2c, rotation=R_l2c, translation=trans_l2c)
        points_scene = l2c.transform_points(points_local)[0]
        gs.from_xyz(points_scene)

        gs.from_rotation(quaternion_multiply(quaternion_invert(quat_l2c), gs.get_rotation))

        scale = gs.get_scaling
        adjusted_scale = scale * scale_l2c
        gs.mininum_kernel_size *= float(scale_l2c[0, 0].item())
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(gs.mininum_kernel_size * 1.1, device=adjusted_scale.device),
        )
        gs.from_scaling(adjusted_scale)

        output["gs"] = gs
        return gs

    def _sam3d_align_pytorch3d_camera_to_viser_in_place(self, gs: Any) -> Any:
        import torch
        from pytorch3d.transforms import quaternion_multiply

        device = gs._xyz.device
        dtype = gs._xyz.dtype
        R_fix = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        xyz = gs.get_xyz
        gs.from_xyz(xyz @ R_fix.T)

        q_fix = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 4)
        q_old = gs.get_rotation
        gs.from_rotation(quaternion_multiply(q_fix.expand(q_old.shape[0], -1), q_old))
        return gs

    def _splat_from_sam3d_output(self, out: dict) -> SplatData:
        if out is None or "gaussian" not in out or not out["gaussian"]:
            raise ValueError("SAM3D output missing gaussian")
        gs = out.get("gs", None) or out["gaussian"][0]

        import torch

        xyz = gs.get_xyz.detach().float().cpu().numpy().astype(np.float32)
        scales = gs.get_scaling.detach().float().cpu().numpy().astype(np.float32)
        rotations = gs.get_rotation.detach().float().cpu().numpy().astype(np.float32)
        opacities = gs.get_opacity.detach().float().cpu().numpy().astype(np.float32)

        feats = gs._features_dc.detach().float()
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        rgbs = torch.sigmoid(feats).cpu().numpy().astype(np.float32)
        rgbs = np.clip(rgbs, 0.0, 1.0).astype(np.float32)

        xyz, rotations, scales, covs = transform_gaussians(
            centers=xyz,
            rotations=rotations,
            scales=scales,
            A=np.eye(3, dtype=np.float32),
            return_cov=True,
        )

        sh = ((rgbs - 0.5) / float(SH_C0)).astype(np.float32)
        opacity_raw = _logit(opacities.reshape(-1)).astype(np.float32)

        return SplatData(
            centers=xyz.astype(np.float32),
            rgbs=rgbs.astype(np.float32),
            opacities=opacities.reshape(-1, 1).astype(np.float32),
            covariances=covs.astype(np.float32),
            scales=scales.astype(np.float32),
            rotations=rotations.astype(np.float32),
            sh=sh.astype(np.float32),
            opacity_raw=opacity_raw.astype(np.float32),
            centers_raw=xyz.astype(np.float32),
        )

    def _ensure_object_handle(self, obj_idx: int) -> None:
        app = self._app()
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            splat = obj.splat
        if splat is None:
            return
        if self.state.frustum_wxyz is None or self.state.frustum_position is None:
            return

        frame_token = self.state.frame_token
        if frame_token is None:
            return

        # Reduce viewer stalls by capping points sent over the websocket.
        with self.state.lock:
            max_pts = int(getattr(self.state, "display_max_points", 150_000))
            sample_mode = str(getattr(self.state, "display_sample", "opacity"))
        splat_disp = _downsample_splat_for_display(
            splat,
            max_points=max_pts,
            seed=int(obj_idx) + 1337,
            sample_mode=sample_mode,
        )

        # Create a shared camera-frame parent so bbox gizmo and gaussians live in the same
        # coordinate system. This fixes gizmo offsets (e.g. appearing far above the object).
        frame_name = f"/sam3d_gen/{frame_token}/objects/obj_{obj_idx}/camera_frame"
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            frame = obj.frame_handle
        if frame is None or getattr(frame, "name", None) != frame_name:
            try:
                if frame is not None:
                    frame.remove()
            except Exception:
                pass
            frame = app.server.scene.add_frame(
                frame_name,
                show_axes=False,
                wxyz=tuple(self.state.frustum_wxyz.tolist()),
                position=tuple(self.state.frustum_position.tolist()),
            )
            with self.state.lock:
                obj = self.state.objects[obj_idx]
                obj.frame_handle = frame
        else:
            try:
                frame.wxyz = tuple(self.state.frustum_wxyz.tolist())
                frame.position = tuple(self.state.frustum_position.tolist())
            except Exception:
                pass

        name = f"{frame_name}/gs"
        with self.state.lock:
            obj = self.state.objects[obj_idx]
            gs = obj.gs_handle
        if gs is None or getattr(gs, "name", None) != name:
            try:
                if gs is not None:
                    gs.remove()
            except Exception:
                pass
            gs = app.server.scene.add_gaussian_splats(
                name,
                centers=splat_disp.centers,
                covariances=splat_disp.covariances,
                rgbs=splat_disp.rgbs,
                opacities=splat_disp.opacities,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            # Prefer packed buffer update to avoid multiple full-buffer websocket messages.
            try:
                if hasattr(gs, "buffer"):
                    gs.buffer = _pack_splat_buffer(splat_disp)
            except Exception:
                pass
            with self.state.lock:
                obj = self.state.objects[obj_idx]
                obj.gs_handle = gs
        else:
            # IMPORTANT: setting these individually triggers multiple websocket buffer updates.
            # Use a packed buffer if available.
            try:
                if hasattr(gs, "buffer"):
                    gs.buffer = _pack_splat_buffer(splat_disp)
                    return
            except Exception:
                pass
            gs.centers = splat_disp.centers
            gs.covariances = splat_disp.covariances
            gs.rgbs = splat_disp.rgbs
            gs.opacities = splat_disp.opacities
