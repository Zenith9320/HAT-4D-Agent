#!/usr/bin/env python3
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys

import numpy as np
import torch
from PIL import Image

# Support both package and script execution (avoid relative-import errors)
_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
for _p in (_THIS_DIR, _PARENT):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

try:
    from .helpers import (
        BUILD_ID,
        _repo_root,
        _sam3d_repo_root,
        _ensure_sam3d_import_path,
        _load_rgb_u8,
        _load_mask_bool,
        _write_mask_png,
        _align_fix_from_mode,
        _save_debug_inputs,
        _y_up_mesh_to_z_up,
    )
except ImportError:  # script mode fallback
    from helpers import (
        BUILD_ID,
        _repo_root,
        _sam3d_repo_root,
        _ensure_sam3d_import_path,
        _load_rgb_u8,
        _load_mask_bool,
        _write_mask_png,
        _align_fix_from_mode,
        _save_debug_inputs,
        _y_up_mesh_to_z_up,
    )


@dataclass
class BackendConfig:
    host: str = "0.0.0.0"
    port: int = 8015
    sam_checkpoint: str | None = None
    sam_model_type: str = "vit_h"
    sam_device: str = "cuda"
    sam3d_pipeline_config: str | None = None
    sam3d_depth_model_path: str | None = None
    sam3d_compile: bool = False
    sam3d_seed: int = 42
    align_pytorch3d_camera_to_viser: bool = False
    debug_save_inputs: bool = False


class Sam3DBackend:
    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._sam_init_lock = threading.Lock()
        self._sam3d_init_lock = threading.Lock()
        self._pose_opt_lock = threading.Lock()

        self._sam_predictor: Any = None
        self._sam3d_pipeline: Any = None

        self._sam_current_image_key: Tuple[str, bool] | None = None
        self._sam_logits_store: Dict[str, np.ndarray] = {}
        # Cache latest SAM3D outputs per (session_id, object_id) so pose opt can reuse
        # the mesh + base gaussian state without re-running the full pipeline.
        self._sam3d_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

        self.sam_status: str = "initializing"
        self.sam3d_status: str = "initializing"
        self.sam_error: str | None = None
        self.sam3d_error: str | None = None

        # Eager init synchronously: users reported background threads don't reliably
        # initialize before the first request in some environments.
        self._init_sam_sync()
        self._init_sam3d_sync()

    def _init_sam_sync(self) -> None:
        try:
            self._ensure_sam()
            with self._lock:
                self.sam_status = "ready"
                self.sam_error = None
        except Exception as exc:
            with self._lock:
                self.sam_status = "failed"
                self.sam_error = f"{type(exc).__name__}: {exc}"

    def _init_sam3d_sync(self) -> None:
        try:
            self._ensure_sam3d()
            with self._lock:
                self.sam3d_status = "ready"
                self.sam3d_error = None
        except Exception as exc:
            with self._lock:
                self.sam3d_status = "failed"
                self.sam3d_error = f"{type(exc).__name__}: {exc}"

    # -----------------------
    # Init
    # -----------------------
    def _ensure_sam(self) -> None:
        with self._sam_init_lock:
            if self._sam_predictor is not None:
                return
            _ensure_sam3d_import_path()

            if self.cfg.sam_checkpoint is None:
                self.cfg.sam_checkpoint = os.environ.get("SAM_CHECKPOINT")
            if self.cfg.sam_checkpoint is None:
                raise RuntimeError(
                    "SAM checkpoint is not configured; pass --sam-checkpoint or set SAM_CHECKPOINT"
                )
            ckpt = Path(self.cfg.sam_checkpoint)
            if not ckpt.exists():
                raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

            from segment_anything import SamPredictor, sam_model_registry  # type: ignore

            sam = sam_model_registry[self.cfg.sam_model_type](checkpoint=str(ckpt))
            sam.to(device=self.cfg.sam_device)
            self._sam_predictor = SamPredictor(sam)

    def _ensure_sam3d(self) -> None:
        with self._sam3d_init_lock:
            if self._sam3d_pipeline is not None:
                return
            _ensure_sam3d_import_path()

            if self.cfg.sam3d_pipeline_config is None:
                checkpoint_root = _sam3d_repo_root() / "checkpoints"
                for default_pipeline in (
                    checkpoint_root / "hf" / "pipeline.yaml",
                    checkpoint_root / "pipeline.yaml",
                ):
                    if default_pipeline.exists():
                        self.cfg.sam3d_pipeline_config = str(default_pipeline)
                        break
            if self.cfg.sam3d_pipeline_config is None or not Path(self.cfg.sam3d_pipeline_config).exists():
                raise FileNotFoundError("sam3d_pipeline_config is not set or missing")

            import importlib

            # Preflight import for clearer errors.
            mod = importlib.import_module("sam3d_objects.pipeline.inference_pipeline_pointmap")
            getattr(mod, "InferencePipelinePointMap")

            import torch
            from hydra.utils import instantiate
            from omegaconf import OmegaConf

            config_file = Path(self.cfg.sam3d_pipeline_config)
            config = OmegaConf.load(str(config_file))
            if self.cfg.sam3d_depth_model_path:
                config.depth_model.model.pretrained_model_name_or_path = self.cfg.sam3d_depth_model_path
            config.rendering_engine = "pytorch3d"
            config.compile_model = bool(self.cfg.sam3d_compile)
            config.workspace_dir = str(config_file.parent)
            self._sam3d_pipeline = instantiate(config)
            _ = torch

    # -----------------------
    # Public endpoints
    # -----------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "build_id": BUILD_ID,
                "sam_ready": self._sam_predictor is not None,
                "sam3d_ready": self._sam3d_pipeline is not None,
                "sam_status": self.sam_status,
                "sam3d_status": self.sam3d_status,
                "sam_error": self.sam_error,
                "sam3d_error": self.sam3d_error,
                "sam_image_key": None if self._sam_current_image_key is None else list(self._sam_current_image_key),
                "logits_cached": len(self._sam_logits_store),
            }

    def sam_set_image(self, *, image_path: str, flip_udlr: bool) -> Dict[str, Any]:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"image_path not found: {p}")
        self._ensure_sam()
        rgb = _load_rgb_u8(p, flip_udlr=bool(flip_udlr))
        with self._lock:
            key = (str(p.resolve()), bool(flip_udlr))
            if self._sam_current_image_key != key:
                self._sam_predictor.set_image(rgb)
                self._sam_current_image_key = key
        return {"image_size": [int(rgb.shape[0]), int(rgb.shape[1])]}

    def sam_predict(
        self,
        *,
        image_path: str,
        flip_udlr: bool,
        points_xy: list[list[int]] | list[tuple[int, int]],
        point_labels: list[int],
        prev_logits_key: str | None,
        output_dir: str | None,
        session_id: str | None,
        object_id: int | None,
    ) -> Dict[str, Any]:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"image_path not found: {p}")
        self._ensure_sam()

        # Ensure predictor is set to this image (cheap if unchanged).
        self.sam_set_image(image_path=image_path, flip_udlr=bool(flip_udlr))

        pts = np.asarray(points_xy, dtype=np.float32)
        labs = np.asarray(point_labels, dtype=np.int32)
        if pts.size == 0:
            raise ValueError("points_xy is empty")
        if pts.shape[0] != labs.shape[0]:
            raise ValueError("points_xy and point_labels size mismatch")

        mask: np.ndarray
        logits_best: np.ndarray
        with self._lock:
            prev_logits = None
            if prev_logits_key:
                prev_logits = self._sam_logits_store.get(prev_logits_key)

        if prev_logits is None:
            masks, scores, logits = self._sam_predictor.predict(
                point_coords=pts,
                point_labels=labs,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = masks[best].astype(bool)
            logits_best = logits[best]
        else:
            masks, _scores, logits = self._sam_predictor.predict(
                point_coords=pts,
                point_labels=labs,
                mask_input=prev_logits[None, :, :],
                multimask_output=False,
            )
            mask = masks[0].astype(bool)
            logits_best = logits[0]

        logits_key = uuid.uuid4().hex
        with self._lock:
            self._sam_logits_store[logits_key] = logits_best

        # Save mask as PNG (so frontend can load without shipping big arrays over HTTP).
        if output_dir is None:
            output_dir = str((_repo_root() / "tmp").resolve())
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sid = session_id or "default"
        oid = int(object_id or 0)
        mask_path = out_dir / "sam3d_gen" / "masks" / sid / f"obj_{oid}.png"
        _write_mask_png(mask, mask_path)
        return {"mask_path": str(mask_path), "logits_key": logits_key}

    def sam3d_generate(
        self,
        *,
        image_path: str,
        flip_udlr: bool,
        mask_path: str,
        output_dir: str,
        session_id: str | None,
        object_id: int,
        seed: int | None,
        align_pytorch3d_camera_to_viser: bool | None,
        align_mode: str | None = None,
    ) -> Dict[str, Any]:
        p_img = Path(image_path)
        if not p_img.exists():
            raise FileNotFoundError(f"image_path not found: {p_img}")
        p_mask = Path(mask_path)
        if not p_mask.exists():
            raise FileNotFoundError(f"mask_path not found: {p_mask}")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_sam3d()

        rgb = _load_rgb_u8(p_img, flip_udlr=bool(flip_udlr))
        mask = _load_mask_bool(p_mask, flip_udlr=False)  # mask already in display coords
        rgba = np.concatenate([rgb, (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
        if bool(self.cfg.debug_save_inputs):
            _save_debug_inputs(out_dir, rgb=rgb, mask=mask)

        pipeline = self._sam3d_pipeline
        assert pipeline is not None

        out = pipeline.run(
            rgba,
            None,
            int(seed if seed is not None else self.cfg.sam3d_seed),
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=None,
        )

        # Apply pose + optional camera alignment (same as sam3d_objects_viser_app).
        gs = out.get("gaussian", [None])[0]
        if gs is None:
            raise RuntimeError("SAM3D output missing gaussian")

        # Cache base gaussian state (Gaussian frame) for pose refinement.
        try:
            import torch

            base_xyz = gs.get_xyz.detach().clone().cpu()
            base_rot = gs.get_rotation.detach().clone().cpu()
            base_sc = gs.get_scaling.detach().clone().cpu()
            base_min = float(gs.mininum_kernel_size)
            base_opacity = gs.get_opacity.detach().clone().cpu()
            feats = gs._features_dc.detach().clone()
            if feats.dim() == 3:
                feats = feats[:, 0, :]
            base_rgb = torch.sigmoid(feats).detach().cpu()
            _ = torch
        except Exception as exc:
            raise RuntimeError(f"Failed to cache base gaussian state: {exc}") from exc

        try:
            from pytorch3d.transforms import quaternion_invert, quaternion_multiply, quaternion_to_matrix
            from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
        except Exception as exc:
            raise RuntimeError(f"Missing deps for pose apply: {exc}") from exc

        quat_l2c = out.get("rotation", None)
        trans_l2c = out.get("translation", None)
        scale_l2c = out.get("scale", None)
        if quat_l2c is None or trans_l2c is None or scale_l2c is None:
            raise KeyError("output missing pose keys (rotation/translation/scale)")

        points_local = gs.get_xyz.unsqueeze(0)
        R_l2c = quaternion_to_matrix(quat_l2c)
        l2c = compose_transform(scale=scale_l2c, rotation=R_l2c, translation=trans_l2c)
        points_scene = l2c.transform_points(points_local)[0]
        gs.from_xyz(points_scene)
        gs.from_rotation(quaternion_multiply(quaternion_invert(quat_l2c), gs.get_rotation))

        adjusted_scale = gs.get_scaling * scale_l2c
        gs.mininum_kernel_size *= float(scale_l2c[0, 0].item())
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(gs.mininum_kernel_size * 1.1, device=adjusted_scale.device),
        )
        gs.from_scaling(adjusted_scale)

        do_align = bool(
            self.cfg.align_pytorch3d_camera_to_viser
            if align_pytorch3d_camera_to_viser is None
            else align_pytorch3d_camera_to_viser
        )

        if do_align:
            from pytorch3d.transforms import quaternion_multiply

            device = gs._xyz.device
            dtype = gs._xyz.dtype
            R_fix, q_fix = _align_fix_from_mode(align_mode, device=device, dtype=dtype)
            gs.from_xyz(gs.get_xyz @ R_fix.T)
            q_old = gs.get_rotation
            gs.from_rotation(quaternion_multiply(q_fix.expand(q_old.shape[0], -1), q_old))

        sid = session_id or "default"
        ply_path = out_dir / "sam3d_gen" / "objects" / sid / f"obj_{int(object_id):03d}.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        gs.save_ply(str(ply_path))

        # Cache everything needed for pose optimization in backend mode.
        try:
            mesh_obj = out.get("glb", None)
            verts_y = None
            faces = None
            vcols = None
            if mesh_obj is not None and hasattr(mesh_obj, "vertices") and hasattr(mesh_obj, "faces"):
                verts_y = np.asarray(mesh_obj.vertices, dtype=np.float32)
                faces = np.asarray(mesh_obj.faces, dtype=np.int64)
                try:
                    if hasattr(mesh_obj, "visual") and hasattr(mesh_obj.visual, "vertex_colors"):
                        vcols = np.asarray(mesh_obj.visual.vertex_colors)
                except Exception:
                    vcols = None

            with self._lock:
                self._sam3d_cache[(sid, int(object_id))] = {
                    "out": out,
                    "ply_path": str(ply_path),
                    "base_xyz": base_xyz,
                    "base_rot": base_rot,
                    "base_sc": base_sc,
                    "base_min": float(base_min),
                    "base_opacity": base_opacity,
                    "base_rgb": base_rgb,
                    "mesh_verts_y": verts_y,
                    "mesh_faces": faces,
                    "mesh_vcols": vcols,
                    "normK": None,
                }
        except Exception:
            # Cache failures should not block generation.
            pass
        return {"ply_path": str(ply_path)}

    def sam3d_pose_opt(
        self,
        *,
        image_path: str,
        flip_udlr: bool,
        mask_path: str,
        output_dir: str,
        session_id: str | None,
        object_id: int,
        steps: int,
        lr: float,
        min_side: int,
        rgb_weight: float,
        debug_mode: bool,
        align_pytorch3d_camera_to_viser: bool | None,
        align_mode: str | None = None,
        center_reg_weight: float | None = None,
        pose_reg_weight: float | None = None,
        scale_reg_weight: float | None = None,
        box_reg_weight: float | None = None,
        init_scale_mult: float | None = None,
        camera_fov_deg: float | None = None,
        camera_wxyz: list[float] | None = None,
        camera_position: list[float] | None = None,
        camera_look_at: list[float] | None = None,
        camera_up: list[float] | None = None,
        lock_translation_x: bool | None = None,
        lock_translation_y: bool | None = None,
        lock_translation_z: bool | None = None,
        optimize_rotation: bool | None = None,
        target_center_world: list[float] | None = None,
        lock_anchor_local: list[float] | None = None,
    ) -> Dict[str, Any]:
        """
        Mask-aligned pose optimization in backend mode (ported from sam3d_objects_viser_app.py).
        Returns a refined PLY path.
        """
        p_img = Path(image_path)
        if not p_img.exists():
            raise FileNotFoundError(f"image_path not found: {p_img}")
        p_mask = Path(mask_path)
        if not p_mask.exists():
            raise FileNotFoundError(f"mask_path not found: {p_mask}")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sid = session_id or "default"
        oid = int(object_id)

        self._ensure_sam3d()
        pipeline = self._sam3d_pipeline
        assert pipeline is not None

        # Pull cached output + base state.
        with self._lock:
            cached = self._sam3d_cache.get((sid, oid))
        if cached is None:
            raise RuntimeError("No cached SAM3D output for this object. Run /sam3d/generate first.")

        out = cached.get("out", None)
        base_xyz = cached.get("base_xyz", None)
        base_rot = cached.get("base_rot", None)
        base_sc = cached.get("base_sc", None)
        base_min = cached.get("base_min", None)
        verts_y = cached.get("mesh_verts_y", None)
        faces_np = cached.get("mesh_faces", None)
        vcols = cached.get("mesh_vcols", None)
        base_opacity = cached.get("base_opacity", None)
        base_rgb = cached.get("base_rgb", None)

        if out is None or base_xyz is None or base_rot is None or base_sc is None or base_min is None:
            raise RuntimeError("Cached SAM3D base gaussian state is missing (restart backend and regenerate).")

        # Global lock: one pose opt at a time (GPU heavy).
        with self._pose_opt_lock:
            try:
                import math
                import torch
                import torch.nn.functional as F
                from gsplat.rendering import rasterization as gsplat_rasterization  # type: ignore
                from pytorch3d.transforms import (
                    matrix_to_quaternion,
                    quaternion_invert,
                    quaternion_multiply,
                    quaternion_to_matrix,
                    so3_exp_map,
                )
                from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
            except Exception as exc:
                raise RuntimeError(f"Missing deps for pose optimization: {exc}") from exc

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Build GT mask/RGB in the original image view (the same one used by the frontend).
            # This avoids relying on pipeline.compute_pointmap() intrinsics, and makes the objective
            # "rendered silhouette/RGB vs. original image mask/RGB" explicit.
            rgb_u8 = _load_rgb_u8(p_img, flip_udlr=bool(flip_udlr))
            mask_bool = _load_mask_bool(p_mask, flip_udlr=False)

            H0, W0 = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
            if mask_bool.shape[0] != H0 or mask_bool.shape[1] != W0:
                mask_bool = (
                    np.array(
                        Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((W0, H0), resample=Image.NEAREST)
                    )
                    > 0
                )

            scale_factor = float(min_side) / float(max(1, min(H0, W0)))
            Hr = max(32, int(round(H0 * scale_factor)))
            Wr = max(32, int(round(W0 * scale_factor)))

            if (Hr, Wr) != (H0, W0):
                gt_mask_u8 = np.array(
                    Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((Wr, Hr), resample=Image.NEAREST)
                )
                gt_rgb_u8 = np.array(
                    Image.fromarray(rgb_u8).resize((Wr, Hr), resample=Image.BILINEAR),
                    dtype=np.uint8,
                )
            else:
                gt_mask_u8 = (mask_bool.astype(np.uint8) * 255)
                gt_rgb_u8 = rgb_u8

            gt = (torch.from_numpy(gt_mask_u8).to(device=device, dtype=torch.float32) / 255.0)[None, None]
            gt = (gt > 0.5).float()
            gt_rgb = None
            if float(rgb_weight) > 0.0:
                gt_rgb = (
                    torch.from_numpy(gt_rgb_u8)
                    .to(device=device, dtype=torch.float32)
                    .permute(2, 0, 1)[None]
                    / 255.0
                ).clamp(0.0, 1.0)

            # Distance-transform term (optional): if SciPy is unavailable, skip this loss.
            # The dice loss alone is usually sufficient for mask alignment.
            dt_t = torch.zeros_like(gt[0, 0], device=device, dtype=torch.float32)
            w_dt = 0.0
            try:
                from scipy.ndimage import distance_transform_edt  # type: ignore

                dt = distance_transform_edt((gt[0, 0].detach().cpu().numpy() < 0.5).astype(np.uint8)).astype(
                    np.float32
                )
                dt_t = torch.from_numpy(dt).to(device=device, dtype=torch.float32)
                dt_t = dt_t / (dt_t.max() + 1e-6)
                w_dt = 0.25
            except Exception:
                # Keep dt_t as zeros and weight as 0.0.
                pass

            do_align = bool(
                self.cfg.align_pytorch3d_camera_to_viser
                if align_pytorch3d_camera_to_viser is None
                else align_pytorch3d_camera_to_viser
            )
            R_fix = None
            q_fix = None
            if do_align:
                R_fix, q_fix = _align_fix_from_mode(align_mode, device=device, dtype=torch.float32)

            # Camera matrices for gsplat (OpenGL-like pinhole model as used in viser_frontend._rasterize_current_view).
            width = int(Wr)
            height = int(Hr)
            fov_y_deg = float(camera_fov_deg if camera_fov_deg is not None else 49.1)
            tan_half_fov = math.tan(0.5 * math.radians(fov_y_deg))
            f = float(width) / max(1e-6, 2.0 * tan_half_fov)
            K = torch.tensor(
                [[f, 0.0, float(width) / 2.0], [0.0, f, float(height) / 2.0], [0.0, 0.0, 1.0]],
                device=device,
                dtype=torch.float32,
            )
            Ks = K[None]

            # Camera extrinsics (world_from_camera) from frontend frustum. If missing, fall back to identity.
            viewmats = torch.eye(4, device=device, dtype=torch.float32)[None]
            R_cw = torch.eye(3, device=device, dtype=torch.float32)
            t_cw = torch.zeros((3,), device=device, dtype=torch.float32)
            if camera_wxyz is not None and camera_position is not None:
                cam_q = torch.tensor(camera_wxyz, device=device, dtype=torch.float32).view(1, 4)
                R_cw = quaternion_to_matrix(cam_q)[0]
                t_cw = torch.tensor(camera_position, device=device, dtype=torch.float32).view(3)
                T_cw = torch.eye(4, device=device, dtype=torch.float32)
                T_cw[:3, :3] = R_cw
                T_cw[:3, 3] = t_cw
                viewmats = torch.linalg.inv(T_cw)[None]

            # Base gaussian attributes for gsplat render.
            if base_opacity is None or base_rgb is None:
                try:
                    gs0 = out.get("gaussian", [None])[0]
                    if gs0 is not None:
                        base_opacity = gs0.get_opacity.detach().clone().cpu()
                        feats = gs0._features_dc.detach().clone()
                        if feats.dim() == 3:
                            feats = feats[:, 0, :]
                        base_rgb = torch.sigmoid(feats).detach().cpu()
                except Exception:
                    pass
            if base_opacity is None or base_rgb is None:
                raise RuntimeError("Pose opt(gsplat) missing cached base_opacity/base_rgb (restart and regenerate).")

            quat0 = out.get("rotation", None)
            t0 = out.get("translation", None)
            s0 = out.get("scale", None)
            if quat0 is None or t0 is None or s0 is None:
                raise RuntimeError("Cached SAM3D output missing pose keys (rotation/translation/scale)")
            quat0 = quat0.detach().to(device=device, dtype=torch.float32)
            t0 = t0.detach().to(device=device, dtype=torch.float32)
            s0 = s0.detach().to(device=device, dtype=torch.float32)
            if init_scale_mult is not None:
                s0 = s0 * float(init_scale_mult)
            R0 = quaternion_to_matrix(quat0)

            # Move base gaussian tensors to the render device once.
            base_xyz_t = base_xyz.to(device=device, dtype=torch.float32)
            base_rot_t = base_rot.to(device=device, dtype=torch.float32)
            base_sc_t = base_sc.to(device=device, dtype=torch.float32)
            base_op_t = base_opacity.to(device=device, dtype=torch.float32)
            if base_op_t.dim() > 1:
                base_op_t = base_op_t.reshape(-1)
            base_rgb_t = base_rgb.to(device=device, dtype=torch.float32)
            if base_rgb_t.dim() == 3:
                base_rgb_t = base_rgb_t[:, 0, :]

            # Optional subsampling for speed/stability in interactive optimization.
            try:
                n_all = int(base_xyz_t.shape[0])
                max_pts = 150_000
                if n_all > max_pts:
                    g = torch.Generator(device="cpu")
                    g.manual_seed(int(oid) * 10007 + 1337)
                    idx = torch.randperm(n_all, generator=g, device="cpu")[:max_pts].to(device=device)
                    base_xyz_t = base_xyz_t.index_select(0, idx)
                    base_rot_t = base_rot_t.index_select(0, idx)
                    base_sc_t = base_sc_t.index_select(0, idx)
                    base_op_t = base_op_t.index_select(0, idx)
                    base_rgb_t = base_rgb_t.index_select(0, idx)
            except Exception:
                pass

            omega = torch.nn.Parameter(torch.zeros((3,), device=device, dtype=torch.float32))
            log_s = torch.nn.Parameter(torch.zeros((1,), device=device, dtype=torch.float32))
            lock_world_x = bool(lock_translation_x)
            lock_world_y = bool(lock_translation_y)
            lock_world_z = bool(lock_translation_z)
            optimize_rotation = bool(optimize_rotation)
            has_world_lock = bool(lock_world_x or lock_world_y or lock_world_z)

            R_pre_world = R_fix.T if R_fix is not None else torch.eye(3, device=device, dtype=torch.float32)
            B_cam_to_world = torch.matmul(R_pre_world, R_cw.T)  # local(camera) -> world rotation
            t_world_offset = t_cw.view(1, 3)

            w_center = base_op_t.detach()
            if w_center.dim() != 1:
                w_center = w_center.reshape(-1)
            w_center = w_center.clamp(min=0.0) + 1e-6
            w_center = w_center / w_center.sum()
            base_mu = (base_xyz_t * w_center[:, None]).sum(dim=0, keepdim=True)
            anchor_local = base_mu
            if lock_anchor_local is not None and len(lock_anchor_local) >= 3:
                anchor_local = torch.tensor(
                    lock_anchor_local[:3],
                    device=device,
                    dtype=torch.float32,
                ).view(1, 3)

            def _center_world_from_pose(R: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                mu_cam = torch.bmm((anchor_local * s).unsqueeze(1), R).squeeze(1) + t
                return torch.matmul(mu_cam, B_cam_to_world) + t_world_offset

            def _anchor_world_from_pose(R: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                mu_cam = torch.bmm((anchor_local * s).unsqueeze(1), R).squeeze(1)
                return torch.matmul(mu_cam, B_cam_to_world) + t_world_offset

            with torch.no_grad():
                target_center_world_t = _center_world_from_pose(R0, t0, s0).detach()
                if target_center_world is not None and len(target_center_world) >= 3:
                    target_center_world_t = torch.tensor(
                        target_center_world[:3],
                        device=device,
                        dtype=torch.float32,
                    ).view(1, 3)
            lock_target_world_np = target_center_world_t.detach().cpu().numpy().reshape(3).astype(np.float32)
            lock_final_world_np = None

            free_axes = [i for i, locked in enumerate([lock_world_x, lock_world_y, lock_world_z]) if not locked]
            delta_free = None
            trans_param: torch.nn.Parameter | None = None
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

            # Prioritize translation (unlocked axes) + scale, keep rotation as mild refinement.
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
                    R_delta = so3_exp_map(omega[None])
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
                    delta_world[:, 0] = target_center_world_t[:, 0] - anchor_world[:, 0]
                if lock_world_y:
                    delta_world[:, 1] = target_center_world_t[:, 1] - anchor_world[:, 1]
                if lock_world_z:
                    delta_world[:, 2] = target_center_world_t[:, 2] - anchor_world[:, 2]
                if delta_free is not None:
                    for j, axis in enumerate(free_axes):
                        delta_world[:, axis] = delta_free[j]
                t = torch.matmul(delta_world, B_cam_to_world.T)
                return R, t, s

            bg = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
            backgrounds = bg.view(1, 3)

            def _render_pred(R: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                # Apply pose to base gaussians in the PyTorch3D camera frame.
                v = base_xyz_t[None] * s[:, None, :]
                v = torch.bmm(v, R) + t[:, None, :]  # (1,N,3)

                # Convert to Viser camera convention if enabled.
                if R_fix is not None:
                    v = torch.matmul(v, R_fix.T)

                # Camera -> world using the provided frustum pose.
                # We keep viewmats/K consistent with viser_frontend._rasterize_current_view.
                # NOTE: points are row-vectors; for camera-to-world rotation R_cw, use right-multiply by R_cw^T.
                v_world = torch.matmul(v, R_cw.T) + t_cw.view(1, 1, 3)

                quat_new = matrix_to_quaternion(R)  # (1,4) wxyz
                q_inv = quaternion_invert(quat_new).expand(base_rot_t.shape[0], -1)
                rot_cam = quaternion_multiply(q_inv, base_rot_t)
                if q_fix is not None:
                    rot_cam = quaternion_multiply(q_fix.expand(rot_cam.shape[0], -1), rot_cam)

                R_rot_cam = quaternion_to_matrix(rot_cam)  # (N,3,3)
                # Same convention as above: row-vector rotations compose via right-multiply.
                R_rot_world = torch.matmul(R_rot_cam, R_cw.T)  # (N,3,3)
                rot_world = matrix_to_quaternion(R_rot_world)

                scales_world = base_sc_t * s.expand(base_sc_t.shape[0], -1)

                rendered_image_all, rendered_alpha_all, _ = gsplat_rasterization(
                    means=v_world[0],
                    quats=rot_world,
                    scales=scales_world,
                    opacities=base_op_t,
                    colors=base_rgb_t,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    near_plane=0.1,
                    far_plane=100.0,
                    packed=False,
                    backgrounds=backgrounds,
                    render_mode="RGB",
                )
                pred_rgb = rendered_image_all[0].clamp(0.0, 1.0)  # (H,W,3)
                pred_a = rendered_alpha_all[0]
                if pred_a.dim() == 3 and pred_a.shape[-1] == 1:
                    pred_a = pred_a[..., 0]
                pred_a = pred_a.clamp(0.0, 1.0)  # (H,W)
                return pred_rgb, pred_a

            def _dice_loss(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
                eps = 1e-6
                num = 2.0 * (p * g).sum()
                den = (p + g).sum() + eps
                return 1.0 - (num + eps) / den

            def _weighted_center(v_world_n3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """Opacity-weighted center in world coordinates.

                Returns (center_xyz, dist_to_origin).
                """
                w = base_op_t.detach()
                if w.dim() != 1:
                    w = w.reshape(-1)
                w = w.clamp(min=0.0).to(device=v_world_n3.device, dtype=v_world_n3.dtype)
                w = w + 1e-6
                center = (v_world_n3 * w[:, None]).sum(dim=0) / w.sum()
                dist = torch.linalg.norm(center)
                return center, dist

            def _center_reg_loss(v_world_n3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """Regularize object center toward origin.

                Returns (loss, dist) where dist is the weighted-center distance to origin.
                The loss grows super-linearly with distance (stronger when farther away).
                """
                center, dist = _weighted_center(v_world_n3)
                # Super-linear growth: ~||c||^3 (zero gradient at origin, strong far away).
                loss = (dist * dist) * (1.0 + dist)
                return loss, dist

            def _x_reg_loss(v_world_n3: torch.Tensor, x_target: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
                """Regularize object center's X coordinate toward x_target.

                Returns (loss, x_dist) where x_dist is the weighted-center X distance to target.
                Only constrains the X axis, leaving Y and Z free for 2D mask alignment.
                """
                center, _ = _weighted_center(v_world_n3)
                x_dist = torch.abs(center[0] - x_target)
                # Quadratic loss on X deviation only
                loss = x_dist * x_dist
                return loss, x_dist

            w_dice = 1.0
            # Rotation regularizer is stronger than translation to keep rotation change mild.
            w_reg_rot = float(pose_reg_weight) if pose_reg_weight is not None else 2e-3
            w_reg_trans = max(1e-6, 0.2 * w_reg_rot)
            # Keep depth drift modest so apparent-size alignment is solved by scale instead of only Z-translation.
            w_reg_depth = max(1e-4, 2.5 * w_reg_trans)
            # Weaker scale regularizer so scale can adapt when center is constrained.
            w_reg_scale = float(scale_reg_weight) if scale_reg_weight is not None else 1e-5
            # X-axis regularizer: keeps object center's X near 0, leaves Y/Z free for 2D alignment.
            # Use center_reg_weight to control this (repurposed from full-center reg).
            w_x_reg = 0.0 if lock_world_x else (float(center_reg_weight) if center_reg_weight is not None else 1e-2)
            # Bounding-box regularizer: disabled by default to allow free 2D alignment.
            w_box = float(box_reg_weight) if box_reg_weight is not None else 0.0
            w_rgb = float(rgb_weight)
            gt_mask = gt[:, 0]

            # Record initial X position for reference (no forced pre-centering).
            init_x_target = 0.0
            try:
                with torch.no_grad():
                    v0 = base_xyz_t[None] * s0[:, None, :]
                    v0 = torch.bmm(v0, R0) + t0[:, None, :]
                    if R_fix is not None:
                        v0 = torch.matmul(v0, R_fix.T)
                    v0_world = torch.matmul(v0, R_cw.T) + t_cw.view(1, 1, 3)
                    c0, _ = _weighted_center(v0_world[0])
                    if torch.isfinite(c0).all():
                        # Use initial X as the target (or 0 if you prefer absolute constraint)
                        # init_x_target = float(c0[0].item())  # keep near initial X
                        init_x_target = 0.0  # keep near X=0
            except Exception:
                pass

            # Progress + debug dumps
            dbg_dir = None
            dbg_stride = max(1, int(steps) // 20)
            if bool(debug_mode):
                dbg_dir = out_dir / "sam3d_gen" / "pose_opt_debug" / sid / f"obj_{oid:03d}"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from PIL import Image as _Img

                    _Img.fromarray(gt_mask_u8.astype(np.uint8), mode="L").save(dbg_dir / "gt_mask.png")
                    _Img.fromarray(gt_rgb_u8.astype(np.uint8), mode="RGB").save(dbg_dir / "gt_rgb.png")
                except Exception:
                    pass

            try:
                from tqdm import tqdm  # type: ignore

                iters = tqdm(range(int(steps)), desc=f"[PoseOpt] obj={oid}", leave=False, dynamic_ncols=True)
            except Exception:
                iters = range(int(steps))

            for it in iters:
                opt.zero_grad(set_to_none=True)
                R, t, s = _compose_pose()
                pred_rgb_hw3, pred = _render_pred(R, t, s)
                loss_dice = _dice_loss(pred[None], gt_mask)
                loss_dt = (pred * dt_t).mean()
                loss_rgb = torch.tensor(0.0, device=device, dtype=torch.float32)
                if w_rgb > 0.0 and gt_rgb is not None:
                    pred_rgb = pred_rgb_hw3.permute(2, 0, 1)[None]
                    w = gt
                    eps = 1e-6
                    loss_rgb = F.smooth_l1_loss(pred_rgb * w, gt_rgb * w, reduction="sum") / (w.sum() * 3.0 + eps)

                # X-axis regularization: only constrain X coordinate, leave Y/Z free for 2D mask alignment.
                # NOTE: v_world depends on (R,t,s); recompute minimally here.
                v = base_xyz_t[None] * s[:, None, :]
                v = torch.bmm(v, R) + t[:, None, :]  # (1,N,3)
                if R_fix is not None:
                    v = torch.matmul(v, R_fix.T)
                v_world = torch.matmul(v, R_cw.T) + t_cw.view(1, 1, 3)  # (1,N,3)
                loss_x_reg, x_dist = _x_reg_loss(v_world[0], init_x_target)
                # Soft box constraint (disabled by default, w_box=0).
                overflow = torch.relu(v_world[0].abs() - 1.0)
                loss_box = (overflow * overflow).mean()

                loss_reg_rot = (omega * omega).sum()
                if not bool(optimize_rotation):
                    loss_reg_rot = torch.tensor(0.0, device=device, dtype=torch.float32)
                loss_reg_trans = torch.tensor(0.0, device=device, dtype=torch.float32)
                if trans_param is not None:
                    loss_reg_trans = (trans_param * trans_param).sum()
                loss_reg_depth = (t[:, 2] - t0[:, 2]).pow(2).mean()
                loss_reg_scale = (log_s * log_s).sum()
                loss = (
                    w_dice * loss_dice
                    + w_dt * loss_dt
                    + w_rgb * loss_rgb
                    + w_x_reg * loss_x_reg
                    + w_box * loss_box
                    + w_reg_rot * loss_reg_rot
                    + w_reg_trans * loss_reg_trans
                    + w_reg_depth * loss_reg_depth
                    + w_reg_scale * loss_reg_scale
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

                # tqdm live metrics
                try:
                    if hasattr(iters, "set_postfix"):
                        iters.set_postfix(
                            loss=float(loss.detach().cpu().item()),
                            dice=float(loss_dice.detach().cpu().item()),
                            dt=float(loss_dt.detach().cpu().item()),
                            rgb=float(loss_rgb.detach().cpu().item()),
                            x=float(x_dist.detach().cpu().item()),
                            box=float(loss_box.detach().cpu().item()),
                        )
                except Exception:
                    pass

                if dbg_dir is not None and (it == 0 or it == int(steps) - 1 or (it % dbg_stride == 0)):
                    try:
                        from PIL import Image as _Img

                        pr_mask_u8 = (pred.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                        pr_rgb_u8 = (
                            pred_rgb_hw3.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0
                        ).astype(np.uint8)

                        _Img.fromarray(pr_mask_u8, mode="L").save(dbg_dir / f"pred_mask_{int(it):05d}.png")
                        _Img.fromarray(pr_rgb_u8, mode="RGB").save(dbg_dir / f"pred_rgb_{int(it):05d}.png")

                        # Quick overlay for sanity: GT in green, pred in red.
                        ov = np.zeros((pr_mask_u8.shape[0], pr_mask_u8.shape[1], 3), dtype=np.uint8)
                        ov[..., 0] = pr_mask_u8
                        ov[..., 1] = gt_mask_u8.astype(np.uint8)
                        _Img.fromarray(ov, mode="RGB").save(dbg_dir / f"mask_overlay_{int(it):05d}.png")
                    except Exception:
                        pass

            # Always emit a stable "final" overlay path for the frontend to display.
            pose_opt_preview_dir = out_dir / "sam3d_gen" / "pose_opt_preview" / sid / f"obj_{oid:03d}"
            if dbg_dir is not None:
                pose_opt_preview_dir = dbg_dir
            pose_opt_preview_dir.mkdir(parents=True, exist_ok=True)
            mask_overlay_path = pose_opt_preview_dir / "mask_overlay_final.png"
            try:
                from PIL import Image as _Img

                pr_mask_u8 = (pred.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                pr_rgb_u8 = (pred_rgb_hw3.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                _Img.fromarray(pr_mask_u8, mode="L").save(pose_opt_preview_dir / "pred_mask_final.png")
                _Img.fromarray(pr_rgb_u8, mode="RGB").save(pose_opt_preview_dir / "pred_rgb_final.png")
                try:
                    _Img.fromarray(gt_mask_u8.astype(np.uint8), mode="L").save(pose_opt_preview_dir / "gt_mask.png")
                    _Img.fromarray(gt_rgb_u8.astype(np.uint8), mode="RGB").save(pose_opt_preview_dir / "gt_rgb.png")
                except Exception:
                    pass
                ov = np.zeros((pr_mask_u8.shape[0], pr_mask_u8.shape[1], 3), dtype=np.uint8)
                ov[..., 0] = pr_mask_u8
                ov[..., 1] = gt_mask_u8.astype(np.uint8)
                _Img.fromarray(ov, mode="RGB").save(mask_overlay_path)
            except Exception:
                # Best-effort; still return the intended path.
                pass

            # Commit refined pose + re-apply to gaussian deterministically using cached base state.
            with torch.no_grad():
                Rn, tn, sn = _compose_pose()
                quat_new = matrix_to_quaternion(Rn)
                lock_final_world_np = _center_world_from_pose(Rn, tn, sn).detach().cpu().numpy().reshape(3).astype(np.float32)

                out["rotation"] = quat_new.detach()
                out["translation"] = tn.detach()
                out["scale"] = sn.detach()

                # Re-apply to gaussian in-place (Gaussian frame -> PyTorch3D camera frame).
                if "gaussian" not in out or not out["gaussian"]:
                    raise RuntimeError("Cached output missing gaussian")
                gs_live = out["gaussian"][0]

                base_xyz_t = base_xyz.to(device=device, dtype=torch.float32)
                base_rot_t = base_rot.to(device=device, dtype=torch.float32)
                base_sc_t = base_sc.to(device=device, dtype=torch.float32)

                R_pose = quaternion_to_matrix(quat_new)
                l2c = compose_transform(scale=sn, rotation=R_pose, translation=tn)
                pts_scene = l2c.transform_points(base_xyz_t.unsqueeze(0))[0]
                gs_live.from_xyz(pts_scene)

                q_inv = quaternion_invert(quat_new).expand(base_rot_t.shape[0], -1)
                gs_live.from_rotation(quaternion_multiply(q_inv, base_rot_t))

                adjusted_scale = base_sc_t * sn.expand(base_sc_t.shape[0], -1)
                kernel_scale = float(sn[0].mean().item())
                gs_live.mininum_kernel_size = float(base_min) * kernel_scale
                adjusted_scale = torch.maximum(
                    adjusted_scale,
                    torch.tensor(gs_live.mininum_kernel_size * 1.1, device=adjusted_scale.device, dtype=adjusted_scale.dtype),
                )
                gs_live.from_scaling(adjusted_scale)

                if do_align:
                    from pytorch3d.transforms import quaternion_multiply as _qm

                    R_fix, q_fix = _align_fix_from_mode(align_mode, device=device, dtype=torch.float32)
                    gs_live.from_xyz(gs_live.get_xyz @ R_fix.T)
                    q_old = gs_live.get_rotation
                    gs_live.from_rotation(_qm(q_fix.expand(q_old.shape[0], -1), q_old))

            ply_path = out_dir / "sam3d_gen" / "objects" / sid / f"obj_{oid:03d}_poseopt.ply"
            ply_path.parent.mkdir(parents=True, exist_ok=True)
            out["gaussian"][0].save_ply(str(ply_path))
            with self._lock:
                if (sid, oid) in self._sam3d_cache:
                    self._sam3d_cache[(sid, oid)]["ply_path"] = str(ply_path)
            return {
                "ply_path": str(ply_path),
                "mask_overlay_path": str(mask_overlay_path),
                "pose_opt_preview_dir": str(pose_opt_preview_dir),
                "lock_target_world": None if lock_target_world_np is None else lock_target_world_np.astype(float).tolist(),
                "lock_final_world": None if lock_final_world_np is None else lock_final_world_np.astype(float).tolist(),
            }
