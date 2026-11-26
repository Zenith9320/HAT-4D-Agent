#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import threading
import urllib.error
import urllib.request
import shutil
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
from scipy.spatial.transform import Rotation as R

import imageio.v3 as iio
import numpy as np
import time
import traceback
import math as _math

ROOT = Path(__file__).resolve().parent
VISER_SRC = ROOT / "thirdparty" / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.append(str(VISER_SRC))

import viser  # noqa: E402
import viser.transforms as tf  # noqa: E402
from viser._assignable_props_api import colors_to_uint8  # noqa: E402
from http_security import client_headers, is_loopback_host  # noqa: E402

from viewer_plugins.manual_edit import ManualEditPlugin  # noqa: E402
from viewer_plugins.sam3d_gen import Sam3DGenPlugin  # noqa: E402
from viewer_plugins.record_ops import RecordOpsPlugin  # noqa: E402
from viewer_plugins.vlm_assistor import VLMAssistorPlugin  # noqa: E402

from utils.constants import COORD_ALIGN_MAT, SH_C0
from utils.geometry_utils import (
    compute_center_radius,
    project_bbox_size,
    solve_depth_for_bbox,
    transform_gaussians
)
from utils.image_utils import (
    add_alpha_channel,
    build_timeline_strip,
    ensure_rgb,
    highlight_strip,
    image_bbox_from_mask,
    load_image,
)
from utils.ply_utils import read_ply_vertices
from utils.ply_utils import SplatData, load_splat_from_ply as load_gaussian_from_ply
from utils.viser_compat import register_key_handler
from utils.camera_utils import (
    direction_for_view,
    direction_from_angles,
    eye_wxyz_for_view,
    eye_wxyz_from_angles,
    up_vector,
)

CAMERA_ZNEAR = 0.05
CAMERA_ZFAR = 150.0


def _list_frame_tokens(input_dir: Path) -> List[str]:
    image_suffixes = {".png", ".jpg", ".jpeg"}
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    digit_named = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_suffixes and p.stem.isdigit()
    ]
    if digit_named:
        return [p.stem for p in sorted(digit_named, key=lambda p: int(p.stem))]
    generic = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes]
    if not generic:
        raise FileNotFoundError(
            "No input images found in input_dir. "
            "Expected .png/.jpg/.jpeg files directly under the directory. "
            f"Got: {input_dir}"
        )
    return [p.stem for p in sorted(generic, key=lambda p: p.name)]


# `SplatData` and `load_gaussian_from_ply` are provided by `utils.ply_utils`.


def pack_splat_buffer(data: SplatData) -> np.ndarray:
    """Pack splat attributes into the uint32 buffer layout expected by Viser."""
    num = data.centers.shape[0]
    cov_triu = data.covariances.reshape((-1, 9))[:, np.array([0, 1, 2, 4, 5, 8])]
    buffer = np.concatenate(
        [
            data.centers.astype(np.float32).view(np.uint8),
            np.zeros((num, 4), dtype=np.uint8),
            cov_triu.astype(np.float16).copy().view(np.uint8),
            colors_to_uint8(data.rgbs),
            colors_to_uint8(data.opacities),
        ],
        axis=-1,
    ).view(np.uint32)
    assert buffer.shape == (num, 8)
    return buffer


def compute_bounds(splats: SplatData) -> Tuple[np.ndarray, float]:
    center, radius = compute_center_radius(splats.centers)
    return center, radius


def _quat_wxyz_to_rotmat(wxyz: np.ndarray) -> np.ndarray:
    """Vectorized quaternion (wxyz) -> rotation matrix, returns (N,3,3)."""
    q = wxyz.astype(np.float32, copy=False)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    Rm = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    Rm[:, 0, 0] = ww + xx - yy - zz
    Rm[:, 0, 1] = 2.0 * (xy - wz)
    Rm[:, 0, 2] = 2.0 * (xz + wy)
    Rm[:, 1, 0] = 2.0 * (xy + wz)
    Rm[:, 1, 1] = ww - xx + yy - zz
    Rm[:, 1, 2] = 2.0 * (yz - wx)
    Rm[:, 2, 0] = 2.0 * (xz - wy)
    Rm[:, 2, 1] = 2.0 * (yz + wx)
    Rm[:, 2, 2] = ww - xx - yy + zz
    return Rm


def _covariance_from_rot_scale(
    rotations_wxyz: np.ndarray, scales: np.ndarray, A: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Compute aligned covariance A * (R diag(s^2) R^T) * A^T without eigen decomposition."""
    Rm = _quat_wxyz_to_rotmat(rotations_wxyz)
    S2 = np.square(scales.astype(np.float32, copy=False))
    Sigma = (Rm * S2[:, None, :]) @ Rm.transpose(0, 2, 1)
    A32 = A.astype(np.float32, copy=False)
    Sigma_new = A32[None] @ Sigma @ A32.T[None]
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.transpose(0, 2, 1))
    # keep SPD-ish
    with np.errstate(invalid="ignore"):
        eigmin = np.linalg.eigvalsh(Sigma_new).min()
    if not np.isfinite(eigmin):
        eigmin = 0.0
    if float(eigmin) < float(eps):
        Sigma_new = Sigma_new + np.eye(3, dtype=np.float32)[None] * (float(eps) - float(eigmin))
    return Sigma_new.astype(np.float32, copy=False)


class GaussianSequence:
    def __init__(
        self,
        ply_dir: Path,
        frame_tokens: Sequence[str] | None = None,
        cache_size: int = 3,
    ):
        self.ply_dir = ply_dir
        self.cache_size = max(1, cache_size)
        self.cache: OrderedDict[int, SplatData] = OrderedDict()
        self._vertex_count_cache: Dict[int, int] = {}

        def _token_from_ply_name(p: Path) -> str | None:
            # Common patterns:
            # - 000000.ply -> token=000000
            # - wo_interp_000000_127.ply -> token=000127 (last digit group)
            groups = re.findall(r"\d+", p.stem)
            if not groups:
                return None
            last = groups[-1]
            try:
                return f"{int(last):06d}"
            except ValueError:
                return None

        if frame_tokens is None:
            ply_paths = sorted(ply_dir.glob("*.ply"))
            if not ply_paths:
                raise FileNotFoundError(f"No PLY files found in {ply_dir}")

            keyed: List[Tuple[int, str, Path]] = []
            for i, p in enumerate(ply_paths):
                tok = _token_from_ply_name(p)
                if tok is None:
                    tok = f"{i:06d}"
                keyed.append((int(tok), tok, p))
            keyed.sort(key=lambda t: (t[0], t[2].name))

            self.frame_tokens = [t[1] for t in keyed]
            self.ply_paths = [t[2] for t in keyed]
        else:
            self.frame_tokens = list(frame_tokens)
            self.ply_paths = [ply_dir / f"{t}.ply" for t in self.frame_tokens]

    def load(self, index: int) -> SplatData:
        if index in self.cache:
            data = self.cache.pop(index)
            self.cache[index] = data
            return data

        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)

        path = self.ply_paths[index]
        if not path.exists():
            raise FileNotFoundError(f"GS PLY not found for frame {index}: {path}")
        data = load_gaussian_from_ply(path)
        self.cache[index] = data
        return data

    def gs_path(self, index: int) -> Path:
        return self.ply_paths[index]

    def vertex_count(self, index: int) -> int:
        """Fast(ish) vertex count from PLY header; cached per index."""
        if index in self._vertex_count_cache:
            return int(self._vertex_count_cache[index])
        path = self.ply_paths[index]
        if not path.exists():
            raise FileNotFoundError(f"GS PLY not found for frame {index}: {path}")
        n = 0
        try:
            with path.open("rb") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith(b"element vertex"):
                        try:
                            n = int(line.strip().split()[-1])
                        except Exception:
                            n = 0
                    if line.strip() == b"end_header":
                        break
        except Exception:
            n = 0
        self._vertex_count_cache[index] = int(n)
        return int(n)


class WorkspaceManager:
    def __init__(
        self,
        root: Path,
        gs_dir: Path,
        input_dir: Path,
        eval_dir: Path,
        *,
        create_empty_gs_sequence: bool = False,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = root / f"workspace_{timestamp}"
        self.gs_dir = self.root / "gs_sequence"
        self.images_root = self.root / "images"
        self.selections_dir = self.root / "selections"
        self.results_dir = self.root / "results"
        self.meta_path = self.root / "meta.json"

        self.root.mkdir(parents=True, exist_ok=True)
        self.images_root.mkdir(exist_ok=True)
        self.selections_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        if bool(create_empty_gs_sequence):
            # Create a real directory (not a symlink) so each new workspace starts empty and isolated.
            self.gs_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._stage_gs_sequence(self.gs_dir, gs_dir)
        self._link(self.images_root / "input", input_dir)
        self._link(self.images_root / "eval_gt", eval_dir)

        meta = {
            "created_at": timestamp,
            "gs_source": str(gs_dir.resolve()),
            "input_source": str(input_dir.resolve()),
            "eval_source": str(eval_dir.resolve()),
            "start_empty": bool(create_empty_gs_sequence),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_existing(cls, root: Path) -> "WorkspaceManager":
        if not root.exists():
            raise FileNotFoundError(f"workspace_root not found: {root}")
        obj = cls.__new__(cls)
        obj.root = root
        obj.gs_dir = root / "gs_sequence"
        obj.images_root = root / "images"
        obj.selections_dir = root / "selections"
        obj.results_dir = root / "results"
        obj.meta_path = root / "meta.json"
        obj.images_root.mkdir(exist_ok=True)
        obj.selections_dir.mkdir(exist_ok=True)
        obj.results_dir.mkdir(exist_ok=True)
        return obj

    def _link(self, target: Path, source: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"Source path for workspace link is missing: {source}")
        if target.exists():
            return
        target.symlink_to(source.resolve(), target_is_directory=True)

    def _stage_gs_sequence(self, target: Path, source: Path) -> None:
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Gaussian source directory is missing: {source}")
        target.mkdir(parents=True, exist_ok=False)
        for ply_path in sorted(source.glob("*.ply")):
            if ply_path.is_file():
                shutil.copy2(ply_path, target / ply_path.name)

    def relink_gs_dir(self, new_gs_dir: Path) -> None:
        """Retarget the workspace GS symlink to a new directory."""
        if not new_gs_dir.exists():
            raise FileNotFoundError(f"new_gs_dir missing: {new_gs_dir}")
        if self.gs_dir.is_symlink():
            self.gs_dir.unlink()
        elif self.gs_dir.exists():
            # If gs_dir is a real directory (e.g. created via --start-empty), only replace it if empty.
            if self.gs_dir.is_dir():
                try:
                    if any(self.gs_dir.iterdir()):
                        raise RuntimeError(
                            f"workspace gs_dir is a non-empty directory, refusing to replace: {self.gs_dir}"
                        )
                    self.gs_dir.rmdir()
                except Exception as exc:
                    raise RuntimeError(f"Failed to replace workspace gs_dir: {exc}") from exc
            else:
                raise RuntimeError(f"workspace gs_dir exists but is not a directory/symlink: {self.gs_dir}")
        if not self.gs_dir.exists():
            self.gs_dir.symlink_to(new_gs_dir.resolve(), target_is_directory=True)

    def image_path_for_view(self, frame_token: str, view: str) -> Path:
        if view == "input":
            base = self.images_root / "input"
        else:
            base = self.images_root / "eval_gt" / view
        return base / f"{frame_token}.png"

    def gs_path_for_frame(self, ply_path: Path) -> Path:
        return self.gs_dir / ply_path.name

    def output_dir_for(self, frame_token: str, view: str) -> Path:
        out_dir = self.results_dir / f"{frame_token}_{view}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def save_selection(
        self,
        frame_token: str,
        view: str,
        render_image: np.ndarray,
        roi_image: np.ndarray,
        payload: Dict,
    ) -> Dict[str, Path]:
        sel_dir = self.selections_dir / f"{frame_token}_{view}"
        sel_dir.mkdir(parents=True, exist_ok=True)

        render_path = sel_dir / "render.png"
        roi_path = sel_dir / "roi.png"
        iio.imwrite(render_path, render_image)
        if roi_image.size > 0:
            iio.imwrite(roi_path, roi_image)

        payload_path = sel_dir / "request.json"
        payload_path.write_text(json.dumps(payload, indent=2))

        return {"render": render_path, "roi": roi_path, "payload": payload_path}


class ViewerApp:
    def __init__(
        self,
        gs_dir: Path,
        input_dir: Path,
        eval_dir: Path,
        workspace_root: Path,
        resume_workspace: Path | None,
        backend_url: str | None,
        sam3d_backend_url: str | None = None,
        dispatcher_url: str | None = None,
        vlm_agent_url: str | None = None,
        vlm_shared_dir: str | None = None,
        mvtap_url: str | None = None,
        image_size: int = 512,
        fov_deg: float = 49.1,
        up_axis: str = "y",
        flip_colmap: bool = True,
        camera_target_origin: bool = False,
        start_empty: bool = False,
        metric_online: bool = False,
        no_flash_atten: bool = False,
        gs_display_max_points: int = 600_000,
        gs_display_sample: str = "opacity",
        gs_display_fixed_points: int = 0,
        selection_overlay_fixed_points: int = 50_000,
        frame_scrub_debounce_ms: int = 120,
        raster_preview_max_points: int = 200_000,
        raster_preview_enabled: bool = True,
        profile_frame_load: bool = False,
        profile_frame_threshold_ms: float = 150.0,
        use_pose_adjust_v2: bool = False,
        server_host: str = "127.0.0.1",
        server_port: int = 8897,
        http_get_retries: int = 2,
        http_post_retries: int = 0,
        http_retry_backoff_ms: int = 400,
    ):
        # Allow passing a single .ply path; treat its parent as the directory.
        if gs_dir.exists() and gs_dir.is_file() and gs_dir.suffix.lower() == ".ply":
            gs_dir = gs_dir.parent

        self.start_empty = bool(start_empty)
        self.source_gs_dir = gs_dir
        self.gs_dir = gs_dir
        self.input_dir = input_dir
        self.eval_dir = eval_dir
        self.dispatcher_url = dispatcher_url
        self.backend_url = backend_url
        self.sam3d_backend_url = sam3d_backend_url
        if self.dispatcher_url:
            if not self.backend_url:
                self.backend_url = self.dispatcher_url
            if not self.sam3d_backend_url:
                self.sam3d_backend_url = self.dispatcher_url
        self.vlm_agent_url = vlm_agent_url
        self.vlm_shared_dir = str(vlm_shared_dir or os.getenv("VLM_SHARED_DIR") or "").strip() or None
        self.mvtap_url = mvtap_url
        self.image_size = image_size
        self.fov_deg = fov_deg
        self.server_host = str(server_host)
        self.server_port = int(server_port)
        self.up_axis = up_axis
        self.up_vec = up_vector(up_axis)
        self.flip_colmap = flip_colmap
        self.camera_target_origin = camera_target_origin
        self.metric_online = bool(metric_online)
        self.no_flash_atten = bool(no_flash_atten)
        self.gs_display_max_points = int(gs_display_max_points)
        self.gs_display_sample = str(gs_display_sample)
        self.gs_display_fixed_points = int(gs_display_fixed_points)
        self.selection_overlay_fixed_points = int(selection_overlay_fixed_points)
        self.frame_scrub_debounce_s = max(0.0, float(frame_scrub_debounce_ms) / 1000.0)
        self.raster_preview_max_points = int(raster_preview_max_points)
        self.raster_preview_enabled = bool(raster_preview_enabled)
        self.profile_frame_load = bool(profile_frame_load)
        self.profile_frame_threshold_s = max(0.0, float(profile_frame_threshold_ms) / 1000.0)
        self.camera_znear = float(CAMERA_ZNEAR)
        self.camera_zfar = float(CAMERA_ZFAR)
        self.use_pose_adjust_v2 = bool(use_pose_adjust_v2)
        self.http_get_retries = max(0, int(http_get_retries))
        self.http_post_retries = max(0, int(http_post_retries))
        self.http_retry_backoff_s = max(0.0, float(http_retry_backoff_ms) / 1000.0)

        # If resuming an existing workspace, reuse its paths when available.
        self.workspace = None
        if resume_workspace is not None:
            self.workspace = WorkspaceManager.from_existing(resume_workspace)
            if self.workspace.gs_dir.exists():
                self.gs_dir = self.workspace.gs_dir
            input_candidate = self.workspace.images_root / "input"
            if input_candidate.exists():
                self.input_dir = input_candidate
            eval_candidate = self.workspace.images_root / "eval_gt"
            if eval_candidate.exists():
                self.eval_dir = eval_candidate
        else:
            # Create a new workspace up-front so we can start with a real empty gs_sequence directory
            # (not a symlink to a shared workspace_root/_empty_gs folder).
            self.workspace = WorkspaceManager(
                workspace_root,
                self.gs_dir,
                self.input_dir,
                self.eval_dir,
                create_empty_gs_sequence=self.start_empty,
            )
            self.gs_dir = self.workspace.gs_dir

        # If gs_dir already has PLYs, infer token ordering from filenames.
        # Otherwise, use input frames as tokens and allow missing PLYs until generated.
        has_existing_plys = bool(list(self.gs_dir.glob("*.ply")))
        if has_existing_plys and not self.start_empty:
            self.sequence = GaussianSequence(self.gs_dir)
        else:
            frame_tokens = _list_frame_tokens(self.input_dir)
            self.sequence = GaussianSequence(self.gs_dir, frame_tokens=frame_tokens)

        # Bootstrap bounds if frame 0 exists; otherwise fall back to a safe default.
        try:
            first_frame = self.sequence.load(0)
            self.scene_center, self.scene_radius = compute_bounds(first_frame)
            self._bounds_initialized = True
        except (FileNotFoundError, IndexError):
            self.scene_center = np.zeros(3, dtype=np.float32)
            self.scene_radius = 1.0
            self._bounds_initialized = False
        self.camera_target = np.zeros(3) if self.camera_target_origin else self.scene_center.copy()

        # (workspace is always initialized above)
        self.selection_handle = None
        self.selection_info: Dict | None = None
        self.selection_overlay_enabled = False
        self.selection_ui_hooks: List[Callable[[], None]] = []
        self.depth_margin_ratio = 0.05
        self.min_depth_margin = 0.01
        self.last_roi_payload: Dict | None = None
        self.last_refined_ply: Path | None = None
        self.live_last_path: Path | None = None
        self.live_preview_active = False
        self.live_preview_blocked = False
        self.live_preview_hidden = False
        self.loss_history: List[float] = []
        self.opt_metric_history: List[float] = []
        self.opt_metric_label: str | None = None
        self.show_loss_curve = False
        self.metric_frames: List[int] = []
        self.metric_psnr: List[float] = []
        self.metric_lpips: List[float] = []
        self.metric_seen_frames: set[int] = set()
        self.show_metric_curve = False
        self.render_video_only = False
        self.video_samples = 16
        self.tracklet_num_points = 128
        self.later_gen_frames = 16
        self.l4gm_init_mode = "imagedream_init"
        self.auto_vlm_eval_enabled = False
        self.auto_vlm_eval_history_window = 6
        self.memory_debug_ref_video = False
        self.multi_view_start = 0
        self.multi_view_end = max(0, len(self.sequence.frame_tokens) - 1)
        self.memory_selected_frames: List[int] = []
        self.loss_handles: Dict[viser.ClientHandle, viser.GuiImageHandle] = {}
        self.metric_handles: Dict[viser.ClientHandle, viser.GuiImageHandle] = {}
        self.fix_image_handles: List[viser.GuiImageHandle] = []
        self.archor_image_handles: List[viser.GuiImageHandle] = []
        self.frustum_opacity = 0.6
        self.frustum_wireframe = False
        self.show_world_axes = True
        self.black_background = False
        self.sv4d_sigma_start = 250.0
        self.sv4d_num_refine_steps = 25
        self.sv4d_optimize = True
        self.sv4d_mask_dilate = 0
        self.sv4d_lr_color = 0.0025
        self.sv4d_lr_opacity = 0.05
        self.sv4d_lr_scale = 0.005
        self.sv4d_lr_means = 0.00016
        self.sv4d_lr_rotation = 0.001
        self.sv4d_lr_final_factor = 0.02
        self.sv4d_split_enable = True
        self.sv4d_split_every = 20
        self.sv4d_split_start = 100
        self.sv4d_split_max_new = 512
        self.sv4d_split_topk = 128
        self.sv4d_split_score = "scale_opacity_grad"
        self.sv4d_split_thresh = 0.001
        self.sv4d_split_jitter = 0.3
        self.sv4d_split_scale_shrink = 0.6
        self.sv4d_split_opacity_shrink = 0.5
        self.sv4d_split_min_scale = 1e-4
        self.sv4d_split_max_scale = 0.2
        self.sv4d_split_cooldown = 3
        self.last_client: viser.ClientHandle | None = None
        self.preview_handles: Dict[viser.ClientHandle, viser.GuiImageHandle] = {}
        self.cancel_live_buttons: Dict[viser.ClientHandle, viser.GuiButtonHandle] = {}
        self.applied_backups: Dict[int, Path] = {}
        self.gs_node_name = "/gaussians"
        # raster preview handles per client
        self.raster_handles: Dict[viser.ClientHandle, viser.GuiImageHandle] = {}
        # camera poll threads/events per client
        self._camera_poll_threads: Dict[viser.ClientHandle, threading.Thread] = {}
        self._camera_poll_events: Dict[viser.ClientHandle, threading.Event] = {}
        # frame apply debouncing per client (prevents repeated heavy GS loads while scrubbing)
        self._frame_apply_events: Dict[viser.ClientHandle, threading.Event] = {}
        self._frame_apply_threads: Dict[viser.ClientHandle, threading.Thread] = {}
        self._frame_apply_latest_idx: Dict[viser.ClientHandle, int] = {}
        self._frame_apply_debounce_s: Dict[viser.ClientHandle, float] = {}
        self._frame_apply_inflight: Dict[viser.ClientHandle, bool] = {}
        self._ignore_frame_slider_updates: Dict[viser.ClientHandle, bool] = {}
        # raster preview debouncing per client (prevents spawning many expensive threads)
        self._raster_update_events: Dict[viser.ClientHandle, threading.Event] = {}
        self._raster_update_threads: Dict[viser.ClientHandle, threading.Thread] = {}
        self._raster_update_inflight: Dict[viser.ClientHandle, bool] = {}
        self._playing_flags: Dict[viser.ClientHandle, bool] = {}
        # cache for downsampled display splats
        self._display_cache: OrderedDict[int, SplatData] = OrderedDict()
        self._display_cache_size: int = 3
        self._notified_downsample_frames: set[int] = set()
        self._loss_curve_last_saved_at = 0.0
        self._loss_curve_last_path: Path | None = None

        self.session_id: str | None = None
        self.session_root: Path | None = None
        self.session_status_handles: Dict[viser.ClientHandle, viser.GuiTextHandle] = {}
        self.metric_status_handles: Dict[viser.ClientHandle, viser.GuiTextHandle] = {}
        self.metric_poll_events: Dict[viser.ClientHandle, threading.Event] = {}
        self.metric_poll_threads: Dict[viser.ClientHandle, threading.Thread] = {}
        self.frame_slider_handles: Dict[viser.ClientHandle, viser.GuiSliderHandle] = {}
        self.timeline_handles: Dict[viser.ClientHandle, viser.GuiImageHandle] = {}
        self.multi_view_status_handles: Dict[viser.ClientHandle, viser.GuiMarkdownHandle] = {}
        self.multi_view_start_sliders: Dict[viser.ClientHandle, viser.GuiSliderHandle] = {}
        self.multi_view_end_sliders: Dict[viser.ClientHandle, viser.GuiSliderHandle] = {}
        self.memory_status_handles: Dict[viser.ClientHandle, viser.GuiTextHandle] = {}
        self.memory_remove_dropdowns: Dict[viser.ClientHandle, viser.GuiDropdownHandle] = {}
        self.memory_remove_buttons: Dict[viser.ClientHandle, viser.GuiButtonHandle] = {}
        self.auto_vlm_eval_status_handles: Dict[viser.ClientHandle, viser.GuiTextHandle] = {}
        self.key_status_handles: Dict[viser.ClientHandle, viser.GuiTextHandle] = {}
        self.bg_toggle_handles: Dict[viser.ClientHandle, viser.GuiCheckboxHandle] = {}
        self.frustum_near_slider_handles: Dict[viser.ClientHandle, viser.GuiSliderHandle] = {}
        self.frustum_far_slider_handles: Dict[viser.ClientHandle, viser.GuiSliderHandle] = {}
        self.render_resolution_slider_handles: Dict[viser.ClientHandle, viser.GuiSliderHandle] = {}
        self._syncing_camera_clip_sliders = False
        self._syncing_render_resolution_sliders = False
        self.play_events: Dict[viser.ClientHandle, threading.Event] = {}
        self.play_threads: Dict[viser.ClientHandle, threading.Thread] = {}
        self.play_interval = 0.2  # seconds per frame for autoplay
        # auto-generate next segments
        self.gen_loop_events: Dict[viser.ClientHandle, threading.Event] = {}
        self.gen_loop_threads: Dict[viser.ClientHandle, threading.Thread] = {}
        self.gen_loop_active: Dict[viser.ClientHandle, bool] = {}

        self.timeline_base, self.timeline_spans = build_timeline_strip(
            [self.input_dir / f"{token}.png" for token in self.sequence.frame_tokens]
        )

        self.server = viser.ViserServer(host=self.server_host, port=self.server_port)
        self.server.scene.set_up_direction(f"+{self.up_axis}")
        self._set_world_axes_visible(self.show_world_axes)
        # Use ambient light helper available in bundled viser version.
        self.server.scene.add_light_ambient(
            "/lights/ambient", color=(255, 255, 255), intensity=0.5
        )

        self.gs_handle = None
        self.current_frame = 0
        self.current_view = "input"
        self.frustum_handles: List[viser.SceneNodeHandle] = []

        print("To setup client")
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self._setup_client(client)

        self.manual_edit_plugin = ManualEditPlugin(self)
        self.sam3d_gen_plugin = Sam3DGenPlugin(self)
        self.record_ops_plugin = RecordOpsPlugin(self, self.sam3d_gen_plugin)
        self.vlm_assistor_plugin = VLMAssistorPlugin(self, self.sam3d_gen_plugin)

    def _profile_log(self, label: str, dt_s: float, *, threshold_s: float | None = None) -> None:
        if not bool(self.profile_frame_load):
            return
        thr = self.profile_frame_threshold_s if threshold_s is None else float(threshold_s)
        if float(dt_s) < float(thr):
            return
        print(f"[PROFILE] {label}: {dt_s*1000.0:.1f} ms")

    def _set_world_axes_visible(self, visible: bool) -> None:
        self.show_world_axes = bool(visible)
        self.server.scene.world_axes.visible = self.show_world_axes
        for c in list(self.key_status_handles.keys()):
            try:
                c.scene.world_axes.visible = self.show_world_axes
                c.flush()
            except Exception:
                pass

    def _apply_scene_background(self, scene_api: object, black: bool) -> bool:
        set_bg = getattr(scene_api, "set_background_image", None)
        if set_bg is None:
            return False
        image = (
            np.zeros((1, 1, 3), dtype=np.uint8)
            if bool(black)
            else np.full((1, 1, 3), 255, dtype=np.uint8)
        )
        try:
            set_bg(image, format="png")
        except TypeError:
            set_bg(image)
        return True

    def _set_black_background(
        self,
        black: bool,
        source_client: viser.ClientHandle | None = None,
    ) -> None:
        self.black_background = bool(black)
        supported = False
        try:
            supported = self._apply_scene_background(self.server.scene, self.black_background) or supported
        except Exception:
            pass
        for c in list(self.key_status_handles.keys()):
            try:
                supported = self._apply_scene_background(c.scene, self.black_background) or supported
            except Exception:
                pass
        for c, toggle in list(self.bg_toggle_handles.items()):
            try:
                if bool(toggle.value) != self.black_background:
                    toggle.value = self.black_background
            except Exception:
                pass
        if source_client is not None and not supported:
            source_client.add_notification(
                "Background unsupported",
                "Current viser version does not expose set_background_image().",
                color="red",
                auto_close_seconds=6,
            )

    @staticmethod
    def _sanitize_camera_planes(znear: float, zfar: float) -> tuple[float, float]:
        n = float(max(1e-4, float(znear)))
        f = float(max(n + 1e-3, float(zfar)))
        return n, f

    @staticmethod
    def _sanitize_render_resolution(size: int) -> int:
        return int(max(128, min(4096, int(size))))

    def _set_camera_clip_planes(
        self,
        znear: float,
        zfar: float,
        source_client: viser.ClientHandle | None = None,
    ) -> None:
        n, f = self._sanitize_camera_planes(znear, zfar)
        self.camera_znear = float(n)
        self.camera_zfar = float(f)

        for c in list(self.key_status_handles.keys()):
            try:
                with c.atomic():
                    c.camera.near = float(n)
                    c.camera.far = float(f)
                c.flush()
            except Exception:
                pass

        self._syncing_camera_clip_sliders = True
        try:
            for h in list(self.frustum_near_slider_handles.values()):
                try:
                    if abs(float(h.value) - float(n)) > 1e-9:
                        h.value = float(n)
                except Exception:
                    pass
            for h in list(self.frustum_far_slider_handles.values()):
                try:
                    if abs(float(h.value) - float(f)) > 1e-9:
                        h.value = float(f)
                except Exception:
                    pass
        finally:
            self._syncing_camera_clip_sliders = False

    def _set_render_resolution(
        self,
        size: int,
        source_client: viser.ClientHandle | None = None,
    ) -> None:
        size_new = self._sanitize_render_resolution(size)
        changed = int(size_new) != int(self.image_size)
        self.image_size = int(size_new)

        self._syncing_render_resolution_sliders = True
        try:
            for h in list(self.render_resolution_slider_handles.values()):
                try:
                    if int(h.value) != int(size_new):
                        h.value = int(size_new)
                except Exception:
                    pass
        finally:
            self._syncing_render_resolution_sliders = False

        if changed:
            for c in list(self.raster_handles.keys()):
                try:
                    self._schedule_raster_update(c)
                except Exception:
                    pass
            if source_client is not None:
                source_client.add_notification(
                    "Render resolution updated",
                    f"image_size={int(size_new)} (used by raster preview/captures/backend render payloads).",
                    color="green",
                    auto_close_seconds=5,
                )


    @property
    def view_presets(self) -> Dict[str, Tuple[float, float]]:
        return {
            "input": (0.0, 0.0),
            "eval_0": (-75.0, 0.0),
            "eval_1": (15.0, 0.0),
            "eval_2": (105.0, 0.0),
            "eval_3": (195.0, 0.0),
        }
        # Viser and Blender camera azimuths differ by 180 degrees.


    def _setup_client(self, client: viser.ClientHandle) -> None:
        print("setup client")
        self.last_client = client
        client.camera.fov = np.deg2rad(self.fov_deg)
        client.camera.near = float(self.camera_znear)
        client.camera.far = float(self.camera_zfar)
        client.scene.world_axes.visible = self.show_world_axes
        if self.black_background:
            try:
                self._apply_scene_background(client.scene, True)
            except Exception:
                pass

        self._move_camera_to_view(client, self.current_view)

        key_status = client.gui.add_text(
            "Click Key",
            initial_value="None",
            disabled=True,
            order=-1000.0,
        )
        self.key_status_handles[client] = key_status

        # Show only the current input frame to save UI space (small timeline)
        try:
            frame_token = self.sequence.frame_tokens[self.current_frame]
            timeline_img_path = self.workspace.image_path_for_view(frame_token, self.current_view)
            timeline_img = ensure_rgb(load_image(timeline_img_path)) if timeline_img_path.exists() else np.zeros((self.image_size // 4, self.image_size // 4, 3), dtype=np.uint8)
        except Exception:
            timeline_img = np.zeros((self.image_size // 4, self.image_size // 4, 3), dtype=np.uint8)
        timeline_handle = client.gui.add_image(timeline_img, label="Current frame (preview only)")
        self.timeline_handles[client] = timeline_handle
        frame_slider = client.gui.add_slider(
            "Frame",
            min=0,
            max=len(self.sequence.ply_paths) - 1,
            step=1,
            initial_value=self.current_frame,
        )

        play_button = client.gui.add_button(
            "Play / Pause",
            hint="Autoplay frames in sequence (loops).",
        )

        if not self.sequence.ply_paths:
            try:
                frame_slider.disabled = True
            except Exception:
                pass
        self.frame_slider_handles[client] = frame_slider
        view_dropdown = client.gui.add_dropdown(
            "View",
            options=list(self.view_presets.keys()),
            initial_value=self.current_view,
        )

        # Primary operation groups: SAM3D -> L4GM -> Manual Edit.
        try:
            self.sam3d_gen_plugin.install(client, self.workspace, parent_folder=None)
        except Exception as exc:
            print(f"[SAM3D plugin install failed] {exc}")

        max_frame_idx = max(0, len(self.sequence.ply_paths) - 1)
        l4gm_folder = client.gui.add_folder(
            "L4GM_4DGEN",
            expand_by_default=True,
        )
        with l4gm_folder:
            init_mode_dropdown = client.gui.add_dropdown(
                "Init mode",
                options=["imagedream_init", "sv4d_init", "sv4d_overlook_init"],
                initial_value=self.l4gm_init_mode,
                disabled=(self.backend_url is None),
                hint="Choose init mode for first-frame generation.",
            )
            gen_first_button = client.gui.add_button(
                "Generate first frame (F)",
                disabled=(self.backend_url is None),
                hint="Create workspace/<session_id>/gs/000000.ply via backend and load it.",
            )
            later_frames_slider = client.gui.add_slider(
                "Later gen frames",
                min=1,
                max=32,
                step=1,
                initial_value=self.later_gen_frames,
                disabled=(self.backend_url is None),
                hint="Number of future frames to generate via /generate/later_gen.",
            )
            gen_later_button = client.gui.add_button(
                "Generate next segment",
                disabled=(self.backend_url is None),
                hint="Call /generate/later_gen after the current generated frame.",
            )
            gen_last_button = client.gui.add_button(
                "Generate last segment",
                disabled=(self.backend_url is None),
                hint="Call /generate/last_segment from current frame with reversed past frames.",
            )
            add_memory_button = client.gui.add_button(
                "Add Memory Frame",
                disabled=(self.backend_url is None),
                hint="Use current frame as memory. Requires this frame already has generated GS.",
            )
            memory_remove_dropdown = client.gui.add_dropdown(
                "Memory frames",
                options=["(empty)"],
                initial_value="(empty)",
                disabled=True,
                hint="Current memory frame IDs. Choose one then click remove.",
            )
            remove_memory_button = client.gui.add_button(
                "Remove Selected Memory Frame",
                disabled=True,
                hint="Remove the selected memory frame from memory bank.",
            )
            gen_later_memory_button = client.gui.add_button(
                "Generate Next Segment with memory",
                disabled=(self.backend_url is None),
                hint="Use memory bank as first n clip frames, then generate only the next 16-n GS frames.",
            )
            memory_debug_checkbox = client.gui.add_checkbox(
                "Generate with memory debug",
                initial_value=bool(self.memory_debug_ref_video),
                disabled=(self.backend_url is None),
                hint="Save a visualization video of ref_video_arr inputs for inspection.",
            )
            memory_status = client.gui.add_text(
                "Memory bank",
                initial_value="memory n=0 ids=[]",
                disabled=(self.backend_url is None),
            )
            auto_vlm_eval_toggle = client.gui.add_checkbox(
                "Auto VLM eval on new frames",
                initial_value=bool(self.auto_vlm_eval_enabled),
                disabled=(self.backend_url is None),
                hint=(
                    "When enabled, after each generate step it renders front/back/left/right "
                    "views for each new frame and runs Agent4 scoring."
                ),
            )
            auto_vlm_eval_window_slider = client.gui.add_slider(
                "Auto eval history window",
                min=0,
                max=30,
                step=1,
                initial_value=int(self.auto_vlm_eval_history_window),
                disabled=(self.backend_url is None),
                hint="Past N frames used by Agent4 memory when evaluating each generated frame.",
            )
            auto_vlm_eval_status = client.gui.add_text(
                "Auto VLM eval",
                initial_value="idle",
                disabled=(self.backend_url is None),
            )

        # Manual editing is independent from the SAM3D plugin.
        self.manual_edit_plugin.install(client, self.workspace)
        parent_folder = getattr(self.manual_edit_plugin.ui, "root_folder", None)
        # Record operations plugin (observes SAM3D)
        try:
            self.record_ops_plugin.install(client, self.workspace, parent_folder=parent_folder)
        except Exception as exc:
            print(f"[RecordOps plugin install failed] {exc}")

        # Secondary groups: collapsed by default.
        try:
            self.vlm_assistor_plugin.install(client, self.workspace, parent_folder=None)
        except Exception as exc:
            print(f"[VLM assistor plugin install failed] {exc}")

        frame_preview_folder = client.gui.add_folder(
            "Frame preview",
            expand_by_default=False,
        )
        with frame_preview_folder:
            preview_handle = client.gui.add_image(
                ensure_rgb(
                    iio.imread(
                        self.workspace.image_path_for_view(
                            self.sequence.frame_tokens[self.current_frame], self.current_view
                        )
                    )
                ),
                label="Frame preview",
            )
        self.preview_handles[client] = preview_handle

        raster_preview_folder = client.gui.add_folder(
            "Raster preview",
            expand_by_default=False,
        )
        with raster_preview_folder:
            # Rasterized GS preview (using gsplat.rendering.rasterization when available)
            placeholder_raster = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            try:
                raster_handle = client.gui.add_image(placeholder_raster, label="Raster preview")
            except Exception:
                raster_handle = client.gui.add_image(np.zeros((8, 8, 3), dtype=np.uint8), label="Raster preview")
        self.raster_handles[client] = raster_handle

        path_info_folder = client.gui.add_folder(
            "path_info",
            expand_by_default=False,
        )
        with path_info_folder:
            status_md = client.gui.add_markdown(
                f"Workspace: `{self.workspace.root}`\n\n"
                f"GS dir: `{self.workspace.gs_dir}`\n\n"
                f"Images: `{self.workspace.images_root}`"
            )

            session_status = client.gui.add_text(
                "4DGen session",
                initial_value="state=idle",
                disabled=(self.backend_url is None),
            )
            self.session_status_handles[client] = session_status

            metric_status = client.gui.add_text(
                "Metrics",
                initial_value="metric=online" if self.metric_online else "metric=offline",
                disabled=not self.metric_online,
            )
            self.metric_status_handles[client] = metric_status

        select_roi_folder = client.gui.add_folder(
            "Select ROI",
            expand_by_default=False,
        )
        with select_roi_folder:
            roi_button = client.gui.add_button("Start ROI box (rect-select)")
            margin_slider = client.gui.add_slider(
                "Depth margin (%)",
                min=0.0,
                max=20.0,
                step=0.5,
                initial_value=self.depth_margin_ratio * 100.0,
                hint="Depth tolerance relative to depth map (per request).",
                disabled=self.backend_url is None,
            )
            selection_toggle = client.gui.add_checkbox(
                "Show ROI Gaussians",
                initial_value=False,
                disabled=True,
                hint="Toggle overlay for the last ROI selection returned by backend.",
            )
            selection_status = client.gui.add_text(
                "ROI selection",
                initial_value="No ROI selection yet.",
                disabled=True,
            )

        fix_roi_folder = client.gui.add_folder(
            "Fix ROI",
            expand_by_default=False,
        )
        with fix_roi_folder:
            fix_button = client.gui.add_button(
                "Fix ROI (eval_gt)",
                disabled=(self.backend_url is None),
                hint="Project selection to eval_gt views and color-align selected splats.",
            )
            render_video_toggle = client.gui.add_checkbox(
                "Render interp video on Fix",
                initial_value=self.render_video_only,
                disabled=(self.backend_url is None),
                hint="When enabled, Fix will first render an interpolated view trajectory video (input -> ROI view) before optimization.",
            )
            video_samples_slider = client.gui.add_slider(
                "Video samples (frames)",
                min=2,
                max=64,
                step=2,
                initial_value=self.video_samples,
                disabled=(self.backend_url is None),
                hint="Number of interpolated viewpoints in the rendered trajectory video.",
            )
            fix_status = client.gui.add_markdown("Fix not started.")
            if not self.fix_image_handles:
                placeholder = np.zeros((8, 8, 3), dtype=np.uint8)
                self.fix_image_handles = [
                    client.gui.add_image(placeholder, label="Fixing View"),
                    client.gui.add_image(placeholder, label="Target View"),
                    client.gui.add_image(placeholder, label="Fix Target View"),
                ]

        archor_folder = client.gui.add_folder(
            "Anchor refine",
            expand_by_default=False,
        )
        with archor_folder:
            archor_view_dropdown = client.gui.add_dropdown(
                "Anchor view (archor_refine)",
                options=["input", "eval_0", "eval_1", "eval_2", "eval_3"],
                initial_value="input",
                disabled=self.backend_url is None,
                hint="Choose which view to anchor for archor_view_refine (view-consistency tuning).",
            )
            archor_button = client.gui.add_button(
                "Anchor refine (view consistency)",
                disabled=(self.backend_url is None),
                hint="Run archor_view_refine from the selected anchor view with slight tilts.",
            )
            archor_status = client.gui.add_markdown("Anchor refine not started.")
            if not self.archor_image_handles:
                placeholder_small = np.zeros((8, 8, 3), dtype=np.uint8)
                self.archor_image_handles = [
                    client.gui.add_image(placeholder_small, label="Anchor Before"),
                    client.gui.add_image(placeholder_small, label="Anchor After"),
                ]

        loss_folder = client.gui.add_folder(
            "Loss curve",
            expand_by_default=False,
        )
        with loss_folder:
            loss_toggle = client.gui.add_checkbox(
                "Show loss curve",
                initial_value=False,
                hint="Toggle to plot loss over iterations during Fix.",
            )
            loss_image = client.gui.add_image(
                np.zeros((64, 256, 3), dtype=np.uint8), label="Loss curve"
            )

        metrics_folder = client.gui.add_folder(
            "Metrics curve",
            expand_by_default=False,
        )
        with metrics_folder:
            metric_toggle = client.gui.add_checkbox(
                "Show per-frame metrics",
                initial_value=False,
                hint="Plot PSNR / LPIPS over generated frames (requires --metric-online).",
                disabled=not self.metric_online,
            )
            metric_image = client.gui.add_image(
                np.zeros((64, 256, 3), dtype=np.uint8), label="Metrics"
            )

        multiview_folder = client.gui.add_folder(
            "Multi-view",
            expand_by_default=False,
        )
        with multiview_folder:
            mv_start_slider = client.gui.add_slider(
                "Multi-view start frame",
                min=0,
                max=max_frame_idx,
                step=1,
                initial_value=min(self.multi_view_start, max_frame_idx),
                hint="Start frame index (inclusive) for azimuth sweep renders.",
            )
            mv_end_slider = client.gui.add_slider(
                "Multi-view end frame",
                min=0,
                max=max_frame_idx,
                step=1,
                initial_value=min(self.multi_view_end, max_frame_idx),
                hint="End frame index (inclusive) for azimuth sweep renders.",
            )
            multi_view_button = client.gui.add_button(
                "Render multiview RGB+mask",
                disabled=(self.backend_url is None),
                hint=(
                    "Render azimuths [0, 60, 120, 180, 240] at elevation 0 for frames"
                    " [start, end] to session_root/multi_view and session_root/multi_view_mask."
                ),
            )
            multi_view_status = client.gui.add_markdown("Multi-view export idle.")

        sv4d_folder = client.gui.add_folder(
            "SV4D inpaint",
            expand_by_default=False,
        )
        with sv4d_folder:
            sv4d_optimize_toggle = client.gui.add_checkbox(
                "Optimize GS after SV4D",
                initial_value=self.sv4d_optimize,
                disabled=(self.backend_url is None),
                hint="When enabled, run GS optimization using SV4D outputs as GT.",
            )
            sv4d_lr_folder = client.gui.add_folder(
                "LR parameters",
                expand_by_default=False,
            )
            with sv4d_lr_folder:
                sv4d_lr_color_slider = client.gui.add_slider(
                    "LR color",
                    min=1e-5,
                    max=0.02,
                    step=1e-5,
                    initial_value=self.sv4d_lr_color,
                    disabled=(self.backend_url is None),
                )
                sv4d_lr_opacity_slider = client.gui.add_slider(
                    "LR opacity",
                    min=1e-4,
                    max=0.2,
                    step=1e-4,
                    initial_value=self.sv4d_lr_opacity,
                    disabled=(self.backend_url is None),
                )
                sv4d_lr_scale_slider = client.gui.add_slider(
                    "LR scale",
                    min=1e-4,
                    max=0.02,
                    step=1e-4,
                    initial_value=self.sv4d_lr_scale,
                    disabled=(self.backend_url is None),
                )
                sv4d_lr_means_slider = client.gui.add_slider(
                    "LR means",
                    min=1e-6,
                    max=0.005,
                    step=1e-6,
                    initial_value=self.sv4d_lr_means,
                    disabled=(self.backend_url is None),
                )
                sv4d_lr_rotation_slider = client.gui.add_slider(
                    "LR rotation",
                    min=1e-5,
                    max=0.01,
                    step=1e-5,
                    initial_value=self.sv4d_lr_rotation,
                    disabled=(self.backend_url is None),
                )
                sv4d_lr_final_factor_slider = client.gui.add_slider(
                    "LR final factor",
                    min=0.001,
                    max=0.2,
                    step=0.001,
                    initial_value=self.sv4d_lr_final_factor,
                    disabled=(self.backend_url is None),
                )

            sv4d_split_folder = client.gui.add_folder(
                "GS split parameters",
                expand_by_default=False,
            )
            with sv4d_split_folder:
                sv4d_split_enable_toggle = client.gui.add_checkbox(
                    "Enable split",
                    initial_value=self.sv4d_split_enable,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_every_slider = client.gui.add_slider(
                    "Split every (iter)",
                    min=1,
                    max=200,
                    step=1,
                    initial_value=self.sv4d_split_every,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_start_slider = client.gui.add_slider(
                    "Split start (iter)",
                    min=0,
                    max=500,
                    step=1,
                    initial_value=self.sv4d_split_start,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_max_new_slider = client.gui.add_slider(
                    "Split max new",
                    min=0,
                    max=2048,
                    step=16,
                    initial_value=self.sv4d_split_max_new,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_topk_slider = client.gui.add_slider(
                    "Split topk",
                    min=0,
                    max=1024,
                    step=8,
                    initial_value=self.sv4d_split_topk,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_score_dropdown = client.gui.add_dropdown(
                    "Split score",
                    options=["scale_opacity_grad", "means_grad"],
                    initial_value=self.sv4d_split_score,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_thresh_slider = client.gui.add_slider(
                    "Split thresh",
                    min=0.0,
                    max=0.01,
                    step=0.0001,
                    initial_value=self.sv4d_split_thresh,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_jitter_slider = client.gui.add_slider(
                    "Split jitter",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=self.sv4d_split_jitter,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_scale_shrink_slider = client.gui.add_slider(
                    "Split scale shrink",
                    min=0.1,
                    max=1.0,
                    step=0.01,
                    initial_value=self.sv4d_split_scale_shrink,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_opacity_shrink_slider = client.gui.add_slider(
                    "Split opacity shrink",
                    min=0.1,
                    max=1.0,
                    step=0.01,
                    initial_value=self.sv4d_split_opacity_shrink,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_min_scale_slider = client.gui.add_slider(
                    "Split min scale",
                    min=1e-5,
                    max=1e-2,
                    step=1e-5,
                    initial_value=self.sv4d_split_min_scale,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_max_scale_slider = client.gui.add_slider(
                    "Split max scale",
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    initial_value=self.sv4d_split_max_scale,
                    disabled=(self.backend_url is None),
                )
                sv4d_split_cooldown_slider = client.gui.add_slider(
                    "Split cooldown",
                    min=0,
                    max=20,
                    step=1,
                    initial_value=self.sv4d_split_cooldown,
                    disabled=(self.backend_url is None),
                )
            sv4d_sigma_slider = client.gui.add_slider(
                "SV4D sigma_start",
                min=0.0,
                max=500.0,
                step=5.0,
                initial_value=self.sv4d_sigma_start,
                disabled=(self.backend_url is None),
                hint="SV4D2 sampling start sigma (higher=more noise).",
            )
            sv4d_refine_slider = client.gui.add_slider(
                "SV4D refine steps",
                min=0,
                max=50,
                step=1,
                initial_value=self.sv4d_num_refine_steps,
                disabled=(self.backend_url is None),
                hint="Number of SV4D2 refine steps (denoising passes).",
            )
            sv4d_mask_dilate_slider = client.gui.add_slider(
                "SV4D mask dilate (px)",
                min=0,
                max=20,
                step=1,
                initial_value=self.sv4d_mask_dilate,
                disabled=(self.backend_url is None),
                hint="Dilate SV4D optimization mask (0 = no dilation).",
            )
            sv4d_button = client.gui.add_button(
                "SV4D inpaint ROI",
                disabled=(self.backend_url is None),
                hint=(
                    "Render ROI multi-view RGB/mask for the current frame window (12 frames, 5 views)"
                    " and run SV4D2 masked denoising in the backend."
                ),
            )
            sv4d_status = client.gui.add_markdown("SV4D inpaint idle.")

        apply_folder = client.gui.add_folder(
            "Apply refined GS",
            expand_by_default=False,
        )
        with apply_folder:
            apply_refined_button = client.gui.add_button(
                "Apply refined GS",
                disabled=(self.last_refined_ply is None),
                hint="Replace current frame's GS with the last refined_ply returned by Fix.",
            )
            cancel_live_button = client.gui.add_button(
                "Hide live preview",
                disabled=(not self.live_preview_active),
                hint="Reload the original GS for this frame and discard the live preview.",
            )
            apply_status = client.gui.add_text(
                "Apply status",
                initial_value="No refined GS applied.",
                disabled=True,
            )
            undo_apply_button = client.gui.add_button(
                "Undo refined apply",
                disabled=True,
                hint="Restore the original GS file for this frame (before apply).",
            )
        self.cancel_live_buttons[client] = cancel_live_button
        self.loss_handles[client] = loss_image
        self.metric_handles[client] = metric_image
        self.multi_view_status_handles[client] = multi_view_status
        self.multi_view_start_sliders[client] = mv_start_slider
        self.multi_view_end_sliders[client] = mv_end_slider
        self.memory_status_handles[client] = memory_status
        self.memory_remove_dropdowns[client] = memory_remove_dropdown
        self.memory_remove_buttons[client] = remove_memory_button
        self.auto_vlm_eval_status_handles[client] = auto_vlm_eval_status
        if self.memory_selected_frames:
            ids = sorted(set(int(x) for x in self.memory_selected_frames))
            options = [str(int(x)) for x in ids]
            try:
                memory_remove_dropdown.options = options
                memory_remove_dropdown.value = options[0]
                memory_remove_dropdown.disabled = False
            except Exception:
                pass
            remove_memory_button.disabled = self.backend_url is None
            memory_status.value = f"memory n={len(ids)} ids=[{', '.join(options)}]"

        # Restore live preview/apply button state for reconnected clients.
        if self.live_preview_active:
            cancel_live_button.disabled = False
            cancel_live_button.label = "Show live preview" if self.live_preview_hidden else "Hide live preview"
        if self.last_refined_ply is not None:
            apply_refined_button.disabled = False

        frustum_folder = client.gui.add_folder(
            "Frustum",
            expand_by_default=False,
        )
        with frustum_folder:
            frustum_button = client.gui.add_button("Toggle camera frustums (with image)")
            frustum_znear_slider = client.gui.add_slider(
                "Camera znear",
                min=0.001,
                max=5.0,
                step=0.001,
                initial_value=float(self.camera_znear),
                hint="Near clipping plane used by viewer and plugin renders.",
            )
            frustum_zfar_slider = client.gui.add_slider(
                "Camera zfar",
                min=1.0,
                max=500.0,
                step=1.0,
                initial_value=float(self.camera_zfar),
                hint="Far clipping plane used by viewer and plugin renders.",
            )
            render_resolution_slider = client.gui.add_slider(
                "Render resolution",
                min=128,
                max=4096,
                step=16,
                initial_value=int(self.image_size),
                hint="Controls image_size used by raster preview, captures, ROI/multi-view backend payloads.",
            )
            render_resolution_apply = client.gui.add_button(
                "Apply render resolution",
                hint="Apply the slider value to current and future renders.",
            )
            frustum_opacity_slider = client.gui.add_slider(
                "Frustum image opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self.frustum_opacity,
                hint="Adjust overlay transparency for frustum images.",
            )
            frustum_wireframe_toggle = client.gui.add_checkbox(
                "Frustum wireframe only",
                initial_value=self.frustum_wireframe,
                hint="Disable frustum image (wireframe only) to avoid occluding splats.",
            )
            world_axes_toggle = client.gui.add_checkbox(
                "Show coordinate axes",
                initial_value=self.show_world_axes,
                hint="Toggle world coordinate axes visibility.",
            )
            bg_toggle = client.gui.add_checkbox(
                "Black background",
                initial_value=self.black_background,
                hint="Switch scene background between black and white.",
            )
        self.bg_toggle_handles[client] = bg_toggle
        self.frustum_near_slider_handles[client] = frustum_znear_slider
        self.frustum_far_slider_handles[client] = frustum_zfar_slider
        self.render_resolution_slider_handles[client] = render_resolution_slider

        self._apply_frame(self.current_frame, preview_handle)

        # Start a lightweight camera-poll thread to detect interactive camera changes
        # and update the raster preview accordingly.
        def _camera_poll() -> None:
            last_pos = None
            last_wxyz = None
            stop_evt = threading.Event()
            self._camera_poll_events[client] = stop_evt
            while not stop_evt.is_set():
                try:
                    pos = tuple(client.camera.position.tolist())
                    wxyz = tuple(client.camera.wxyz.tolist())
                    if last_pos != pos or last_wxyz != wxyz:
                        last_pos, last_wxyz = pos, wxyz
                        self._schedule_raster_update(client)
                except Exception:
                    pass
                stop_evt.wait(0.25)

        t = threading.Thread(target=_camera_poll, daemon=True)
        self._camera_poll_threads[client] = t
        t.start()

        def refresh_selection_ui() -> None:
            info = self.selection_info
            if info and info.get("frame") == self.current_frame:
                selection_toggle.disabled = False
                selection_toggle.value = self.selection_overlay_enabled
                selection_status.value = (
                    f"{len(info.get('indices', []))} splats "
                    f"(frame {info.get('frame')}, view {info.get('view')})"
                )
                fix_button.disabled = self.backend_url is None
            else:
                selection_toggle.value = False
                selection_toggle.disabled = True
                fix_button.disabled = True
                selection_status.value = "No ROI selection on this frame."
            if selection_toggle.disabled or not self.selection_overlay_enabled:
                self._clear_selection_overlay()
            apply_refined_button.disabled = self.last_refined_ply is None
            undo_apply_button.disabled = self.applied_backups.get(self.current_frame) is None
            apply_status.disabled = False

        self.selection_ui_hooks.append(refresh_selection_ui)
        refresh_selection_ui()

        def _stop_playback() -> None:
            evt = self.play_events.get(client)
            if evt is not None:
                evt.set()
            self._playing_flags[client] = False
            t = self.play_threads.get(client)
            if t is not None and t.is_alive():
                # Best-effort join to prevent accumulating multiple playback threads.
                try:
                    t.join(timeout=0.5)
                except Exception:
                    pass
        def _start_playback() -> None:
            if not self.sequence.frame_tokens:
                return
            # Prevent multiple playback threads per client (a common cause of "GS disappears until refresh").
            existing_thread = self.play_threads.get(client)
            existing_evt = self.play_events.get(client)
            if existing_thread is not None and existing_thread.is_alive():
                # If an old thread is still shutting down, wait briefly; otherwise treat as already playing.
                if existing_evt is not None and not existing_evt.is_set():
                    return
                if existing_evt is not None:
                    existing_evt.set()
                try:
                    existing_thread.join(timeout=0.5)
                except Exception:
                    pass
                if existing_thread.is_alive():
                    return
            stop_evt = threading.Event()
            self.play_events[client] = stop_evt
            self._playing_flags[client] = True

            def _run() -> None:
                try:
                    while not stop_evt.is_set():
                        try:
                            t_play0 = time.perf_counter()
                            next_idx = int(self.current_frame) + 1
                            # Skip missing GS files; stop when no further generated frames.
                            while next_idx < len(self.sequence.ply_paths) and not self.sequence.ply_paths[next_idx].exists():
                                next_idx += 1
                            if next_idx >= len(self.sequence.ply_paths):
                                stop_evt.set()
                                break
                            self.current_frame = next_idx
                            # Update slider without triggering a heavy synchronous frame apply.
                            self._ignore_frame_slider_updates[client] = True
                            try:
                                frame_slider.value = next_idx
                            finally:
                                self._ignore_frame_slider_updates[client] = False
                            # Update timeline preview for the new frame.
                            try:
                                frame_token = self.sequence.frame_tokens[next_idx]
                                img_path = self.workspace.image_path_for_view(frame_token, "input")
                                if img_path.exists():
                                    timeline_handle.image = ensure_rgb(load_image(img_path))
                                else:
                                    timeline_handle.image = np.zeros(
                                        (self.image_size // 4, self.image_size // 4, 3),
                                        dtype=np.uint8,
                                    )
                            except Exception:
                                pass
                            # Apply frame in a dedicated worker to avoid websocket thread-safety issues.
                            self._schedule_apply_frame(client, next_idx, preview_handle, debounce=False)

                            # Throttle playback so we don't enqueue multiple huge GS updates at once.
                            t0 = time.time()
                            while not stop_evt.is_set() and bool(self._frame_apply_inflight.get(client, False)):
                                if time.time() - t0 > 30.0:
                                    break
                                time.sleep(0.01)
                            self._profile_log("play_step_total", time.perf_counter() - t_play0, threshold_s=0.0)
                        except Exception:
                            stop_evt.set()
                            break
                        time.sleep(self.play_interval)
                finally:
                    self._playing_flags[client] = False

            t = threading.Thread(target=_run, daemon=True)
            self.play_threads[client] = t
            t.start()

        @frame_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            if bool(self._ignore_frame_slider_updates.get(client, False)):
                return
            idx = int(frame_slider.value)
            if not self.sequence.ply_paths:
                return
            max_idx = len(self.sequence.ply_paths) - 1
            if idx < 0 or idx > max_idx:
                idx = max(0, min(idx, max_idx))
                try:
                    frame_slider.value = idx
                except Exception:
                    pass
            self.current_frame = idx
            # Always schedule: applying synchronously here can block the websocket thread.
            self._schedule_apply_frame(client, idx, preview_handle, debounce=True)
            try:
                frame_token = self.sequence.frame_tokens[idx]
                img_path = self.workspace.image_path_for_view(frame_token, "input")
                if img_path.exists():
                    timeline_handle.image = ensure_rgb(load_image(img_path))
                else:
                    timeline_handle.image = np.zeros(
                        (self.image_size // 4, self.image_size // 4, 3),
                        dtype=np.uint8,
                    )
            except Exception:
                pass
            self._refresh_selection_ui()

        @play_button.on_click
        def _(_: viser.GuiEvent) -> None:
            evt = self.play_events.get(client)
            if evt and not evt.is_set():
                _stop_playback()
            else:
                _start_playback()

        def _handle_gen_first() -> None:
            if not self.backend_url:
                return
            self._ensure_session(client)
            self._start_metric_poll(client)
            resp = self._post_json(
                self._backend_base_url() + "/generate/first",
                {
                    "session_id": self.session_id,
                    "generator": "l4gm",
                    "init_mode": self.l4gm_init_mode,
                },
                timeout=300,
            )
            print("resp:", resp)
            if resp.get("status") != "success":
                raise RuntimeError(resp.get("message") or "generate/first failed (backend)")
            # Clear cached splats so frame 0 reloads.
            self.sequence.cache.clear()
            self.current_frame = 0
            frame_slider.value = 0

            self._apply_frame(0, preview_handle)
            session_status.value = "state=reviewing frame=0"
            if self.metric_online:
                metrics = resp.get("metrics") or {}
                per_frame = metrics.get("per_frame") or []
                self._apply_metric_entries(per_frame, default_frame_idx=0)
                if metrics:
                    psnr_avg = metrics.get("psnr_avg")
                    lpips_avg = metrics.get("lpips_avg")
                    for h in self.metric_status_handles.values():
                        h.value = f"metric=online psnr={psnr_avg:.3f} lpips={lpips_avg:.3f}" if psnr_avg is not None else "metric=online"
                    if self.show_metric_curve:
                        self._update_metric_curve_image(client)
                else:
                    for h in self.metric_status_handles.values():
                        h.value = "metric=online (no metrics returned)"
            client.add_notification(
                "Generated",
                f"Loaded {resp.get('ply_path', '000000.ply')}",
                color="green",
                auto_close_seconds=8,
            )
            self._run_auto_vlm_eval_for_frames(client, [0])

        @gen_first_button.on_click
        def _(_: viser.GuiEvent) -> None:
            _handle_gen_first()

        def _run_generate_segment(start_frame: int, num_frames: int) -> tuple[int, int]:
            if not self.backend_url:
                raise RuntimeError("backend_url is not set")
            self._ensure_session(client)
            session_status.value = f"state=generating next {num_frames} from {start_frame}"
            self._start_metric_poll(client)
            physics_plan = None
            physics_ops = getattr(getattr(self, "vlm_assistor_plugin", None), "ops", None)
            plan_fn = getattr(physics_ops, "plan_physics_for_segment_sync", None)
            if callable(plan_fn):
                try:
                    planned = plan_fn(start_frame=int(start_frame), num_frames=int(num_frames))
                    physics_plan = dict(planned.get("plan", {}) or {})
                except Exception as exc:
                    print(f"[PhysicsPlanningAgent] fallback: {type(exc).__name__}: {exc}")
            payload = {
                "session_id": self.session_id,
                "num_frames": int(num_frames),
                "start_frame": int(start_frame),
            }
            if physics_plan:
                payload["physics_plan"] = physics_plan
            resp = self._post_json(
                self._backend_base_url() + "/generate/later_gen",
                payload,
                timeout=600,
            )
            if resp.get("status") != "success":
                raise RuntimeError(resp.get("message") or "generate/later_gen failed (backend)")

            self.sequence.cache.clear()
            start = int(resp.get("start_frame", start_frame))
            end = int(resp.get("end_frame", start))
            start = max(0, min(start, len(self.sequence.ply_paths) - 1))
            end = max(start, min(end, len(self.sequence.ply_paths) - 1))
            self.current_frame = start
            frame_slider.value = start
            self._apply_frame(start, preview_handle)
            try:
                frame_token = self.sequence.frame_tokens[start]
                img_path = self.workspace.image_path_for_view(frame_token, "input")
                if img_path.exists():
                    timeline_handle.image = ensure_rgb(load_image(img_path))
                else:
                    timeline_handle.image = np.zeros(
                        (self.image_size // 4, self.image_size // 4, 3),
                        dtype=np.uint8,
                    )
            except Exception:
                pass
            session_status.value = f"state=reviewing frames {start}-{end}"

            if self.metric_online:
                metrics = resp.get("metrics") or {}
                per_frame = metrics.get("per_frame") or []
                self._apply_metric_entries(per_frame, default_frame_idx=start)
                if metrics:
                    psnr_avg = metrics.get("psnr_avg")
                    lpips_avg = metrics.get("lpips_avg")
                    for h in self.metric_status_handles.values():
                        if psnr_avg is not None:
                            h.value = f"metric=online psnr={psnr_avg:.3f} lpips={lpips_avg:.3f}"
                        else:
                            h.value = "metric=online"
                    if self.show_metric_curve:
                        self._update_metric_curve_image(client)
                else:
                    for h in self.metric_status_handles.values():
                        h.value = "metric=online (no metrics returned)"
            client.add_notification(
                "Segment generated",
                f"Frames {start}-{end} finished",
                color="green",
                auto_close_seconds=8,
            )
            self._run_auto_vlm_eval_for_frames(client, range(start, end + 1))
            critic_fn = getattr(physics_ops, "run_4d_validator", None)
            if callable(critic_fn) and os.environ.get("HAT4D_AGENT_CRITIC", "1") == "1":
                try:
                    critic_fn(client, window_size=max(1, end - start), frame_idx=end)
                except Exception as exc:
                    print(f"[PhysicsCriticAgent] fallback: {type(exc).__name__}: {exc}")
            return start, end

        def _run_generate_last_segment(start_frame: int, num_frames: int) -> tuple[int, int]:
            if not self.backend_url:
                raise RuntimeError("backend_url is not set")
            self._ensure_session(client)
            session_status.value = (
                f"state=generating previous {num_frames} from {start_frame} (reverse)"
            )
            self._start_metric_poll(client)
            physics_plan = None
            physics_ops = getattr(getattr(self, "vlm_assistor_plugin", None), "ops", None)
            plan_fn = getattr(physics_ops, "plan_physics_for_segment_sync", None)
            if callable(plan_fn):
                try:
                    planned = plan_fn(start_frame=int(start_frame), num_frames=int(num_frames))
                    physics_plan = dict(planned.get("plan", {}) or {})
                except Exception as exc:
                    print(f"[PhysicsPlanningAgent] backward fallback: {type(exc).__name__}: {exc}")
            payload = {
                "session_id": self.session_id,
                "num_frames": int(num_frames),
                "start_frame": int(start_frame),
            }
            if physics_plan:
                payload["physics_plan"] = physics_plan
            resp = self._post_json(
                self._backend_base_url() + "/generate/last_segment",
                payload,
                timeout=600,
            )
            if resp.get("status") != "success":
                raise RuntimeError(resp.get("message") or "generate/last_segment failed (backend)")

            self.sequence.cache.clear()
            generated = [int(i) for i in (resp.get("generated_frame_indices") or [])]
            if generated:
                oldest = min(generated)
                newest = max(generated)
            else:
                oldest = int(resp.get("start_frame", start_frame))
                newest = int(resp.get("end_frame", start_frame))
            oldest = max(0, min(oldest, len(self.sequence.ply_paths) - 1))
            newest = max(oldest, min(newest, len(self.sequence.ply_paths) - 1))

            self.current_frame = oldest
            frame_slider.value = oldest
            self._apply_frame(oldest, preview_handle)
            try:
                frame_token = self.sequence.frame_tokens[oldest]
                img_path = self.workspace.image_path_for_view(frame_token, "input")
                if img_path.exists():
                    timeline_handle.image = ensure_rgb(load_image(img_path))
                else:
                    timeline_handle.image = np.zeros(
                        (self.image_size // 4, self.image_size // 4, 3),
                        dtype=np.uint8,
                    )
            except Exception:
                pass
            session_status.value = f"state=reviewing frames {oldest}-{newest} (reverse)"

            if self.metric_online:
                metrics = resp.get("metrics") or {}
                per_frame = metrics.get("per_frame") or []
                self._apply_metric_entries(per_frame, default_frame_idx=oldest)
                if metrics:
                    psnr_avg = metrics.get("psnr_avg")
                    lpips_avg = metrics.get("lpips_avg")
                    for h in self.metric_status_handles.values():
                        if psnr_avg is not None:
                            h.value = f"metric=online psnr={psnr_avg:.3f} lpips={lpips_avg:.3f}"
                        else:
                            h.value = "metric=online"
                    if self.show_metric_curve:
                        self._update_metric_curve_image(client)
                else:
                    for h in self.metric_status_handles.values():
                        h.value = "metric=online (no metrics returned)"
            client.add_notification(
                "Previous segment generated",
                f"Frames {oldest}-{newest} finished",
                color="green",
                auto_close_seconds=8,
            )
            self._run_auto_vlm_eval_for_frames(client, range(oldest, newest + 1))
            return oldest, newest

        def _memory_frames_label(ids: List[int]) -> str:
            if not ids:
                return "[]"
            return "[" + ", ".join(str(int(x)) for x in ids) + "]"

        def _refresh_memory_widgets() -> None:
            ids = sorted(set(int(x) for x in self.memory_selected_frames))
            self.memory_selected_frames = ids
            options = [str(int(x)) for x in ids] if ids else ["(empty)"]
            has_memory = bool(ids)
            for handle in list(self.memory_status_handles.values()):
                try:
                    handle.value = f"memory n={len(ids)} ids={_memory_frames_label(ids)}"
                except Exception:
                    pass
            for dd in list(self.memory_remove_dropdowns.values()):
                try:
                    dd.options = options
                    if str(dd.value) not in options:
                        dd.value = options[0]
                    dd.disabled = not has_memory
                except Exception:
                    pass
            for btn in list(self.memory_remove_buttons.values()):
                try:
                    btn.disabled = (self.backend_url is None) or (not has_memory)
                except Exception:
                    pass

        def _adjust_memory_bank(indices: List[int], *, allow_empty: bool = False) -> Dict:
            if not self.backend_url:
                raise RuntimeError("backend_url is not set")
            ids = sorted(set(int(i) for i in indices))
            if (not ids) and (not allow_empty):
                raise ValueError("Please add at least one memory frame")
            self._ensure_session(client)
            self._start_metric_poll(client)
            session_status.value = f"state=adjusting memory n={len(ids)}"
            resp = self._post_json(
                self._backend_base_url() + "/memory/adjust_frames",
                {
                    "session_id": self.session_id,
                    "memory_frame_indices": ids,
                    "allow_empty": bool(allow_empty),
                },
                timeout=600,
            )
            if resp.get("status") != "success":
                raise RuntimeError(resp.get("message") or "memory/adjust_frames failed (backend)")
            mem_indices = [int(i) for i in (resp.get("memory_frame_indices") or ids)]
            self.memory_selected_frames = list(mem_indices)
            _refresh_memory_widgets()
            session_status.value = f"state=reviewing memory n={len(mem_indices)}"
            return resp

        def _run_generate_segment_with_memory(
            start_frame: int, memory_indices: List[int], total_clip_frames: int = 16
        ) -> tuple[int, int]:
            if not self.backend_url:
                raise RuntimeError("backend_url is not set")
            if not memory_indices:
                raise ValueError("Please adjust memory frames first")
            if len(memory_indices) >= int(total_clip_frames):
                raise ValueError(
                    f"memory frame count must be < {int(total_clip_frames)}, got {len(memory_indices)}"
                )
            self._ensure_session(client)
            session_status.value = (
                f"state=generating with memory n={len(memory_indices)} "
                f"future={int(total_clip_frames) - len(memory_indices)} from {start_frame}"
            )
            self._start_metric_poll(client)
            physics_plan = None
            physics_ops = getattr(getattr(self, "vlm_assistor_plugin", None), "ops", None)
            plan_fn = getattr(physics_ops, "plan_physics_for_segment_sync", None)
            if callable(plan_fn):
                try:
                    planned = plan_fn(
                        start_frame=int(start_frame),
                        num_frames=max(1, int(total_clip_frames) - len(memory_indices)),
                    )
                    physics_plan = dict(planned.get("plan", {}) or {})
                except Exception as exc:
                    print(f"[PhysicsPlanningAgent] memory fallback: {type(exc).__name__}: {exc}")
            payload = {
                "session_id": self.session_id,
                "start_frame": int(start_frame),
                "total_clip_frames": int(total_clip_frames),
                "memory_frame_indices": [int(i) for i in memory_indices],
                "debug_ref_video": bool(memory_debug_checkbox.value),
            }
            if physics_plan:
                payload["physics_plan"] = physics_plan
            resp = self._post_json(
                self._backend_base_url() + "/generate/later_gen_with_memory",
                payload,
                timeout=900,
            )
            if resp.get("status") != "success":
                raise RuntimeError(
                    resp.get("message") or "generate/later_gen_with_memory failed (backend)"
                )

            self.sequence.cache.clear()
            start = int(resp.get("start_frame", start_frame))
            end = int(resp.get("end_frame", start))
            start = max(0, min(start, len(self.sequence.ply_paths) - 1))
            end = max(start, min(end, len(self.sequence.ply_paths) - 1))
            self.current_frame = start
            frame_slider.value = start
            self._apply_frame(start, preview_handle)
            try:
                frame_token = self.sequence.frame_tokens[start]
                img_path = self.workspace.image_path_for_view(frame_token, "input")
                if img_path.exists():
                    timeline_handle.image = ensure_rgb(load_image(img_path))
                else:
                    timeline_handle.image = np.zeros(
                        (self.image_size // 4, self.image_size // 4, 3),
                        dtype=np.uint8,
                    )
            except Exception:
                pass
            session_status.value = f"state=reviewing frames {start}-{end} (memory)"

            if self.metric_online:
                metrics = resp.get("metrics") or {}
                per_frame = metrics.get("per_frame") or []
                self._apply_metric_entries(per_frame, default_frame_idx=start)
                if metrics:
                    psnr_avg = metrics.get("psnr_avg")
                    lpips_avg = metrics.get("lpips_avg")
                    for h in self.metric_status_handles.values():
                        if psnr_avg is not None:
                            h.value = f"metric=online psnr={psnr_avg:.3f} lpips={lpips_avg:.3f}"
                        else:
                            h.value = "metric=online"
                    if self.show_metric_curve:
                        self._update_metric_curve_image(client)
                else:
                    for h in self.metric_status_handles.values():
                        h.value = "metric=online (no metrics returned)"
            client.add_notification(
                "Segment generated (memory)",
                f"Frames {start}-{end} finished",
                color="green",
                auto_close_seconds=8,
            )
            debug_path = resp.get("debug_ref_video_path")
            if debug_path:
                client.add_notification(
                    "Memory debug saved",
                    f"{debug_path}",
                    color="blue",
                    auto_close_seconds=10,
                )
            self._run_auto_vlm_eval_for_frames(client, range(start, end + 1))
            return start, end

        def _handle_gen_later() -> None:
            if not self.backend_url:
                return
            gen_later_button.disabled = True
            try:
                num_frames = max(1, int(later_frames_slider.value))
                start_frame = int(self.current_frame)
                _run_generate_segment(start_frame, num_frames)
            except Exception as exc:
                session_status.value = "state=error"
                client.add_notification(
                    "Generate segment failed",
                    f"{exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=12,
                )
            finally:
                gen_later_button.disabled = self.backend_url is None

        def _handle_gen_last() -> None:
            if not self.backend_url:
                return
            gen_last_button.disabled = True
            try:
                num_frames = max(1, int(later_frames_slider.value))
                start_frame = int(self.current_frame)
                _run_generate_last_segment(start_frame, num_frames)
            except Exception as exc:
                session_status.value = "state=error"
                client.add_notification(
                    "Generate last segment failed",
                    f"{exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=12,
                )
            finally:
                gen_last_button.disabled = self.backend_url is None

        def _handle_add_memory() -> None:
            if not self.backend_url:
                return
            add_memory_button.disabled = True
            try:
                frame_idx = int(self.current_frame)
                gs_path = self.sequence.gs_path(frame_idx)
                if not gs_path.exists():
                    raise RuntimeError(
                        f"Frame {frame_idx} has no generated 3D yet. Please generate this frame first."
                    )
                if frame_idx in self.memory_selected_frames:
                    client.add_notification(
                        "Memory unchanged",
                        f"Frame {frame_idx} is already in memory.",
                        color="blue",
                        auto_close_seconds=4,
                    )
                    return
                ids = sorted(set(self.memory_selected_frames + [frame_idx]))
                _adjust_memory_bank(ids)
                client.add_notification(
                    "Memory updated",
                    f"Added frame {frame_idx} (n={len(self.memory_selected_frames)})",
                    color="green",
                    auto_close_seconds=6,
                )
            except Exception as exc:
                session_status.value = "state=error"
                memory_status.value = f"memory=error {exc}"
                client.add_notification(
                    "Add memory failed",
                    f"{exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=12,
                )
            finally:
                add_memory_button.disabled = self.backend_url is None

        def _handle_remove_memory() -> None:
            if not self.backend_url:
                return
            remove_memory_button.disabled = True
            try:
                value = str(memory_remove_dropdown.value)
                if value == "(empty)":
                    return
                remove_idx = int(value)
                ids = [int(x) for x in self.memory_selected_frames if int(x) != remove_idx]
                _adjust_memory_bank(ids, allow_empty=True)
                client.add_notification(
                    "Memory updated",
                    f"Removed frame {remove_idx} (n={len(self.memory_selected_frames)})",
                    color="green",
                    auto_close_seconds=6,
                )
            except Exception as exc:
                session_status.value = "state=error"
                memory_status.value = f"memory=error {exc}"
                client.add_notification(
                    "Remove memory failed",
                    f"{exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=12,
                )
            finally:
                remove_memory_button.disabled = (self.backend_url is None) or (not self.memory_selected_frames)

        def _handle_gen_later_with_memory() -> None:
            if not self.backend_url:
                return
            gen_later_memory_button.disabled = True
            try:
                if not self.memory_selected_frames:
                    raise RuntimeError("No memory frames selected. Use Add Memory Frame first.")
                start_frame = int(self.current_frame)
                _run_generate_segment_with_memory(
                    start_frame=start_frame,
                    memory_indices=list(self.memory_selected_frames),
                    total_clip_frames=16,
                )
            except Exception as exc:
                session_status.value = "state=error"
                client.add_notification(
                    "Generate with memory failed",
                    f"{exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=12,
                )
            finally:
                gen_later_memory_button.disabled = self.backend_url is None

        @add_memory_button.on_click
        def _(_: viser.GuiEvent) -> None:
            _handle_add_memory()

        @remove_memory_button.on_click
        def _(_: viser.GuiEvent) -> None:
            _handle_remove_memory()

        @gen_later_memory_button.on_click
        def _(_: viser.GuiEvent) -> None:
            _handle_gen_later_with_memory()

        @gen_last_button.on_click
        def _(_: viser.GuiEvent) -> None:
            _handle_gen_last()

        @gen_later_button.on_click
        def _(_: viser.GuiEvent) -> None:
            # Toggle auto-generate loop.
            if bool(self.gen_loop_active.get(client, False)):
                evt = self.gen_loop_events.get(client)
                if evt is not None:
                    evt.set()
                self.gen_loop_active[client] = False
                gen_later_button.label = "Generate Next Segment"
                return

            if not self.backend_url:
                return
            stop_evt = threading.Event()
            self.gen_loop_events[client] = stop_evt
            self.gen_loop_active[client] = True
            gen_later_button.label = "Pause Generate"

            def _run_loop() -> None:
                try:
                    while not stop_evt.is_set():
                        num_frames = max(1, int(later_frames_slider.value))
                        start_frame = int(self.current_frame)
                        _start, _end = _run_generate_segment(start_frame, num_frames)
                        # Continue from the last generated frame.
                        self.current_frame = int(_end)
                        # Respect pause quickly.
                        if stop_evt.is_set():
                            break
                except Exception as exc:
                    session_status.value = "state=error"
                    client.add_notification(
                        "Auto-generate failed",
                        f"{exc}",
                        color="red",
                        with_close_button=True,
                        auto_close_seconds=12,
                    )
                finally:
                    self.gen_loop_active[client] = False
                    gen_later_button.label = "Generate Next Segment"

            t = threading.Thread(target=_run_loop, daemon=True)
            self.gen_loop_threads[client] = t
            t.start()

        def _handle_key(event: object) -> None:
            if event.event_type != "keydown" or event.repeat:
                return
            display_key = event.key
            if display_key == " ":
                display_key = "Space"
            key_status.value = f"Click Key '{display_key}'"
            key = event.key.lower()
            if key == "f":
                _handle_gen_first()
                return
            if key == "p":
                _handle_gen_later()
                return
            if key == "s":
                _start_roi_select()
                return
            if key in (" ", "space"):
                evt = self.play_events.get(client)
                if evt and not evt.is_set():
                    _stop_playback()
                else:
                    _start_playback()

        register_key_handler(client, _handle_key)

        @multi_view_button.on_click
        def _(_: viser.GuiEvent) -> None:
            multi_view_button.disabled = True
            start_idx = int(self.multi_view_start)
            end_idx = int(self.multi_view_end)
            multi_view_status.content = f"Multi-view rendering frames {start_idx}-{end_idx}..."

            def _run() -> None:
                try:
                    self._render_multiview_range(
                        client,
                        start_idx,
                        end_idx,
                        multi_view_status,
                        multi_view_button,
                    )
                except Exception as exc:  # noqa: BLE001
                    multi_view_status.content = f"Multi-view failed: {exc}"
                    client.add_notification(
                        "Multi-view export failed",
                        f"{exc}",
                        color="red",
                        with_close_button=True,
                        auto_close_seconds=10,
                    )
                    multi_view_button.disabled = self.backend_url is None

            threading.Thread(target=_run, daemon=True).start()

        @sv4d_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._run_sv4d_inpaint(
                client,
                sv4d_status,
                self.fix_image_handles,
                apply_refined_button,
                cancel_live_button,
                archor_view_dropdown.value,
            )

        @view_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            view = view_dropdown.value
            print(f"look from {view}")
            self.current_view = view
            self._move_camera_to_view(client, view)
            self._update_preview(preview_handle)
            # programmatic camera move -> update raster preview
            self._schedule_raster_update(client)
            self._refresh_selection_ui()

        @selection_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self.selection_overlay_enabled = bool(selection_toggle.value)
            self._render_selection_overlay(self.selection_overlay_enabled)

        @fix_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._run_fix(client, fix_status, self.fix_image_handles, apply_refined_button, cancel_live_button)

        @archor_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._run_archor_refine(
                client,
                archor_status,
                self.archor_image_handles,
                archor_view_dropdown.value,
                apply_refined_button,
                cancel_live_button,
            )

        @render_video_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self.render_video_only = bool(render_video_toggle.value)

        @video_samples_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.video_samples = int(video_samples_slider.value)

        @later_frames_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.later_gen_frames = int(later_frames_slider.value)

        @auto_vlm_eval_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self.auto_vlm_eval_enabled = bool(auto_vlm_eval_toggle.value)
            if not self.auto_vlm_eval_enabled:
                self._set_auto_vlm_eval_status("idle")

        @auto_vlm_eval_window_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.auto_vlm_eval_history_window = int(auto_vlm_eval_window_slider.value)

        @init_mode_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            self.l4gm_init_mode = str(init_mode_dropdown.value)
        @sv4d_sigma_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_sigma_start = float(sv4d_sigma_slider.value)

        @sv4d_refine_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_num_refine_steps = int(sv4d_refine_slider.value)

        @sv4d_mask_dilate_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_mask_dilate = int(sv4d_mask_dilate_slider.value)

        @sv4d_lr_color_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_lr_color = float(sv4d_lr_color_slider.value)

        @sv4d_lr_opacity_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_lr_opacity = float(sv4d_lr_opacity_slider.value)

        @sv4d_lr_scale_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_lr_scale = float(sv4d_lr_scale_slider.value)

        @sv4d_lr_means_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_lr_means = float(sv4d_lr_means_slider.value)

        @sv4d_lr_rotation_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_lr_rotation = float(sv4d_lr_rotation_slider.value)

        @sv4d_lr_final_factor_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_lr_final_factor = float(sv4d_lr_final_factor_slider.value)

        @sv4d_split_enable_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_enable = bool(sv4d_split_enable_toggle.value)

        @sv4d_split_every_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_every = int(sv4d_split_every_slider.value)

        @sv4d_split_start_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_start = int(sv4d_split_start_slider.value)

        @sv4d_split_max_new_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_max_new = int(sv4d_split_max_new_slider.value)

        @sv4d_split_topk_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_topk = int(sv4d_split_topk_slider.value)

        @sv4d_split_score_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_score = str(sv4d_split_score_dropdown.value)

        @sv4d_split_thresh_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_thresh = float(sv4d_split_thresh_slider.value)

        @sv4d_split_jitter_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_jitter = float(sv4d_split_jitter_slider.value)

        @sv4d_split_scale_shrink_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_scale_shrink = float(sv4d_split_scale_shrink_slider.value)

        @sv4d_split_opacity_shrink_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_opacity_shrink = float(sv4d_split_opacity_shrink_slider.value)

        @sv4d_split_min_scale_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_min_scale = float(sv4d_split_min_scale_slider.value)

        @sv4d_split_max_scale_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_max_scale = float(sv4d_split_max_scale_slider.value)

        @sv4d_split_cooldown_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_split_cooldown = int(sv4d_split_cooldown_slider.value)

        @sv4d_optimize_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self.sv4d_optimize = bool(sv4d_optimize_toggle.value)

        @mv_start_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            max_idx = max(0, len(self.sequence.ply_paths) - 1)
            self.multi_view_start = max(0, min(int(mv_start_slider.value), max_idx))
            if self.multi_view_start > self.multi_view_end:
                self.multi_view_end = self.multi_view_start
                mv_end_slider.value = self.multi_view_end

        @mv_end_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            max_idx = max(0, len(self.sequence.ply_paths) - 1)
            self.multi_view_end = max(0, min(int(mv_end_slider.value), max_idx))
            if self.multi_view_end < self.multi_view_start:
                self.multi_view_start = self.multi_view_end
                mv_start_slider.value = self.multi_view_start

        @memory_debug_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self.memory_debug_ref_video = bool(memory_debug_checkbox.value)

        @margin_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.depth_margin_ratio = float(margin_slider.value) / 100.0
            self._rerun_last_roi(client)

        @frustum_opacity_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.frustum_opacity = float(frustum_opacity_slider.value)
            if self.frustum_handles:
                for h in self.frustum_handles:
                    h.remove()
                self.frustum_handles = self._build_frustums_for_frame(self.current_frame, client)

        @frustum_znear_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            if bool(self._syncing_camera_clip_sliders):
                return
            znear = float(frustum_znear_slider.value)
            zfar = float(frustum_zfar_slider.value)
            self._set_camera_clip_planes(znear, zfar, source_client=client)

        @frustum_zfar_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            if bool(self._syncing_camera_clip_sliders):
                return
            znear = float(frustum_znear_slider.value)
            zfar = float(frustum_zfar_slider.value)
            self._set_camera_clip_planes(znear, zfar, source_client=client)

        @render_resolution_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            if bool(self._syncing_render_resolution_sliders):
                return
            try:
                size = self._sanitize_render_resolution(int(render_resolution_slider.value))
                if int(render_resolution_slider.value) != int(size):
                    render_resolution_slider.value = int(size)
            except Exception:
                pass

        @render_resolution_apply.on_click
        def _(_: viser.GuiEvent) -> None:
            self._set_render_resolution(int(render_resolution_slider.value), source_client=client)

        @frustum_wireframe_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self.frustum_wireframe = bool(frustum_wireframe_toggle.value)
            if self.frustum_handles:
                for h in self.frustum_handles:
                    h.remove()
                self.frustum_handles = self._build_frustums_for_frame(self.current_frame, client)

        @world_axes_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self._set_world_axes_visible(bool(world_axes_toggle.value))

        @bg_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self._set_black_background(bool(bg_toggle.value), source_client=client)

        def _start_roi_select() -> None:
            if roi_button.disabled:
                return
            roi_button.disabled = True

            @client.scene.on_pointer_event(event_type="rect-select")
            def _(event: viser.ScenePointerEvent) -> None:
                client.scene.remove_pointer_callback()
                roi_button.disabled = False
                self._handle_roi(event, client)

            @client.scene.on_pointer_callback_removed
            def _() -> None:
                roi_button.disabled = False

        @roi_button.on_click
        def _(_: viser.GuiEvent) -> None:
            _start_roi_select()

        @frustum_button.on_click
        def _(_: viser.GuiEvent) -> None:
            if self.frustum_handles:
                for h in self.frustum_handles:
                    h.remove()
                self.frustum_handles.clear()
                return
            frustums = self._build_frustums_for_frame(self.current_frame)
            self.frustum_handles.extend(frustums)

        @apply_refined_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._apply_refined_gs(client, apply_status, undo_apply_button)

        @undo_apply_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._undo_apply_refined(client, apply_status, undo_apply_button)

        @cancel_live_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._toggle_live_preview(client, apply_status, cancel_live_button)

        @loss_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            self.show_loss_curve = bool(loss_toggle.value)
            if self.show_loss_curve:
                self._update_loss_curve_image(client)
            else:
                self.loss_history.clear()
                loss_image.image = np.zeros((64, 256, 3), dtype=np.uint8)

        @metric_toggle.on_update
        def _(_: viser.GuiEvent) -> None:
            if not self.metric_online:
                metric_toggle.value = False
                self.show_metric_curve = False
                self._update_metric_curve_image(client)
                return
            self.show_metric_curve = bool(metric_toggle.value)
            self._update_metric_curve_image(client)

    def _move_camera_to_view(self, client: viser.ClientHandle, view: str) -> None:
        direction = self._direction_for_view(view)
        distance = 1.5
        target = self.camera_target
        eye = target + direction * distance

        with client.atomic():
            client.camera.position = eye
            client.camera.look_at = target
            client.camera.up_direction = self.up_vec

    def _camera_pose_for_view(self, view: str) -> Tuple[np.ndarray, np.ndarray]:
        distance = 1.5
        return self._eye_wxyz_for_view(view, distance)

    def _direction_for_view(self, view: str) -> np.ndarray:
        return direction_for_view(view, self.view_presets, self.up_axis)

    def _eye_wxyz_for_view(self, view: str, distance: float) -> Tuple[np.ndarray, np.ndarray]:
        return eye_wxyz_for_view(view, distance, self.camera_target, self.view_presets, self.up_axis)

    def _direction_from_angles(self, az_deg: float, el_deg: float) -> np.ndarray:
        return direction_from_angles(az_deg, el_deg, self.up_axis)

    def _eye_wxyz_from_angles(self, az_deg: float, el_deg: float, distance: float) -> Tuple[np.ndarray, np.ndarray]:
        return eye_wxyz_from_angles(az_deg, el_deg, distance, self.camera_target, self.up_axis)

    def _set_client_camera_pose(
        self,
        client: viser.ClientHandle,
        eye: np.ndarray,
        wxyz: np.ndarray,
        look_at: np.ndarray | None = None,
        up: np.ndarray | None = None,
    ) -> None:
        with client.atomic():
            client.camera.position = eye
            client.camera.wxyz = wxyz
            client.camera.look_at = self.camera_target if look_at is None else look_at
            client.camera.up_direction = self.up_vec if up is None else up
            client.camera.fov = np.deg2rad(self.fov_deg)
            client.camera.near = float(self.camera_znear)
            client.camera.far = float(self.camera_zfar)

    def _downsample_splats_for_display(self, data: SplatData, *, max_points: int, seed: int) -> SplatData:
        n = int(data.centers.shape[0])
        k = int(max_points)
        if k <= 0 or n <= k:
            return data

        # Choose representative subset for interactive viewing. Avoid full sort (O(N log N)).
        if self.gs_display_sample == "random":
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(n, size=k, replace=False)
        else:
            # Default: keep top-k by opacity (fast argpartition).
            weights = data.opacities.reshape(-1)
            if weights.shape[0] != n:
                weights = np.ones((n,), dtype=np.float32)
            if k >= n:
                idx = np.arange(n, dtype=np.int64)
            else:
                idx = np.argpartition(weights, -k)[-k:].astype(np.int64)

        idx = np.asarray(idx, dtype=np.int64)
        return SplatData(
            centers=data.centers[idx],
            rgbs=data.rgbs[idx],
            opacities=data.opacities[idx],
            covariances=data.covariances[idx],
            scales=data.scales[idx],
            rotations=data.rotations[idx],
            sh=data.sh[idx],
            opacity_raw=data.opacity_raw[idx],
            centers_raw=data.centers_raw[idx],
        )

    def _pad_splatdata_to_n(self, data: SplatData, n: int) -> SplatData:
        """Pad to a fixed number of gaussians with fully-transparent entries.

        This keeps the client-side Gaussian texture size stable across frames,
        which helps avoid WebGL/worker issues during playback when sizes fluctuate.
        """
        n = int(n)
        m = int(data.centers.shape[0])
        if n <= 0 or m >= n:
            return data

        centers = np.zeros((n, 3), dtype=np.float32)
        rgbs = np.zeros((n, 3), dtype=np.float32)
        opacities = np.zeros((n, 1), dtype=np.float32)
        cov = np.zeros((n, 3, 3), dtype=np.float32)
        cov[m:, 0, 0] = 1e-4
        cov[m:, 1, 1] = 1e-4
        cov[m:, 2, 2] = 1e-4

        centers[:m] = data.centers.astype(np.float32, copy=False)
        rgbs[:m] = data.rgbs.astype(np.float32, copy=False)
        opacities[:m] = data.opacities.astype(np.float32, copy=False)
        cov[:m] = data.covariances.astype(np.float32, copy=False)

        scales = np.zeros((n, 3), dtype=np.float32)
        rotations = np.zeros((n, 4), dtype=np.float32)
        sh = np.zeros((n, 3), dtype=np.float32)
        opacity_raw = np.zeros((n,), dtype=np.float32)
        centers_raw = np.zeros((n, 3), dtype=np.float32)

        try:
            scales[:m] = data.scales.astype(np.float32, copy=False)
        except Exception:
            pass
        try:
            rotations[:m] = data.rotations.astype(np.float32, copy=False)
        except Exception:
            pass
        try:
            sh[:m] = data.sh.astype(np.float32, copy=False)
        except Exception:
            pass
        try:
            opacity_raw[:m] = data.opacity_raw.astype(np.float32, copy=False)
        except Exception:
            pass
        try:
            centers_raw[:m] = data.centers_raw.astype(np.float32, copy=False)
        except Exception:
            centers_raw[:m] = data.centers.astype(np.float32, copy=False)

        return SplatData(
            centers=centers,
            rgbs=rgbs,
            opacities=opacities,
            covariances=cov,
            scales=scales,
            rotations=rotations,
            sh=sh,
            opacity_raw=opacity_raw,
            centers_raw=centers_raw,
        )

    def _load_display_splats_from_ply(
        self, path: Path, *, max_points: int, seed: int
    ) -> SplatData:
        """Load a *display* subset from PLY without per-Gaussian eigen decompositions.

        This is much faster than `utils.ply_utils.load_splat_from_ply` for multi-million
        gaussians, and is sufficient for Viser rendering (centers + covariances + rgb + opacity).
        """
        t0 = time.perf_counter()
        _, v = read_ply_vertices(path)
        t1 = time.perf_counter()

        n = int(len(v))
        fixed_points = int(getattr(self, "gs_display_fixed_points", 0))
        k = int(max_points if fixed_points <= 0 else fixed_points)
        if k <= 0 or n <= k:
            idx = slice(None)
        else:
            if self.gs_display_sample == "random":
                rng = np.random.default_rng(int(seed))
                idx = rng.choice(n, size=k, replace=False)
            else:
                # Use raw opacity logit field for ranking (monotonic with sigmoid).
                weights = v["opacity"].astype(np.float32, copy=False)
                idx = np.argpartition(weights, -k)[-k:]
            idx = np.asarray(idx, dtype=np.int64)

        centers_raw = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32, copy=False)
        centers_raw = centers_raw[idx]
        centers = centers_raw @ COORD_ALIGN_MAT.T

        sh = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32, copy=False)[idx]
        rgbs = np.clip(0.5 + SH_C0 * sh, 0.0, 1.0).astype(np.float32, copy=False)

        opacity_raw = v["opacity"].astype(np.float32, copy=False)[idx]
        opacities = (1.0 / (1.0 + np.exp(-opacity_raw)))[:, None].astype(np.float32, copy=False)

        scales_raw = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32, copy=False)[idx]
        scales = np.exp(scales_raw).astype(np.float32, copy=False)

        rotations = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32, copy=False)[
            idx
        ]

        cov = _covariance_from_rot_scale(rotations, scales, COORD_ALIGN_MAT)
        t2 = time.perf_counter()
        self._profile_log(f"ply_read({path.name})", t1 - t0, threshold_s=0.0)
        self._profile_log(f"ply_build_display({path.name})", t2 - t1, threshold_s=0.0)

        # rotations/scales are not used by Viser splat rendering directly (buffer uses covariances),
        # but keep them populated for any downstream code paths that expect SplatData fields.
        out = SplatData(
            centers=centers,
            rgbs=rgbs,
            opacities=opacities,
            covariances=cov,
            scales=scales,
            rotations=rotations,
            sh=sh,
            opacity_raw=opacity_raw,
            centers_raw=centers_raw,
        )
        if fixed_points > 0:
            out = self._pad_splatdata_to_n(out, fixed_points)
        return out

    def _get_display_splats(self, frame_idx: int, data_full: SplatData) -> SplatData:
        """Return splats to send to the client for visualization.

        Keeps full-resolution data for backend ops, but may downsample the displayed
        points to keep the UI responsive when a frame has millions of gaussians.
        """
        fixed_points = int(getattr(self, "gs_display_fixed_points", 0))
        max_points = int(self.gs_display_max_points if fixed_points <= 0 else fixed_points)
        n = int(data_full.centers.shape[0])
        if max_points <= 0 or n <= max_points:
            # Also clear display cache for this frame to avoid stale.
            if frame_idx in self._display_cache:
                try:
                    self._display_cache.pop(frame_idx, None)
                except Exception:
                    pass
            return self._pad_splatdata_to_n(data_full, fixed_points) if fixed_points > 0 else data_full

        if frame_idx in self._display_cache:
            disp = self._display_cache.pop(frame_idx)
            self._display_cache[frame_idx] = disp
            return disp

        disp = self._downsample_splats_for_display(data_full, max_points=max_points, seed=frame_idx + 1337)
        if fixed_points > 0:
            disp = self._pad_splatdata_to_n(disp, fixed_points)
        self._display_cache[frame_idx] = disp
        while len(self._display_cache) > int(self._display_cache_size):
            self._display_cache.popitem(last=False)
        return disp

    def _update_gaussian_handle(self, data: SplatData) -> None:
        # IMPORTANT: updating `centers/covariances/rgbs/opacities` individually triggers
        # 4 full-buffer websocket updates (see `GaussianSplatHandle` setters). Always
        # pack a buffer once and assign it once for performance.
        try:
            if self.gs_handle is None:
                self.gs_handle = self.server.scene.add_gaussian_splats(
                    self.gs_node_name,
                    centers=data.centers,
                    covariances=data.covariances,
                    rgbs=data.rgbs,
                    opacities=data.opacities,
                )
                return

            t0 = time.perf_counter()
            buf = pack_splat_buffer(data)
            t1 = time.perf_counter()
            self.gs_handle.buffer = buf
            self.gs_handle.visible = True
            self._profile_log("gs_pack_buffer", t1 - t0, threshold_s=0.0)
        except Exception as e:
            print(f"[GS update failed] {e}")
            try:
                if self.gs_handle is not None:
                    self.gs_handle.remove()
            except Exception:
                pass
            self.gs_handle = None


    def _rasterize_current_view(self, client: viser.ClientHandle) -> np.ndarray | None:
        """Rasterize the currently-loaded GS using gsplat.rendering.rasterization.
        OpenGL camera model (aligned with L4GM).
        Returns an HxWx3 uint8 image or None.
        """
        try:
            data = self.sequence.cache.get(int(self.current_frame))
        except Exception:
            data = None
        if data is None:
            return None
        if int(self.raster_preview_max_points) > 0 and int(data.centers.shape[0]) > int(self.raster_preview_max_points):
            return None

        try:
            import torch
            from gsplat.rendering import rasterization
        except Exception:
            return None

        # Camera from viser (OpenGL)
        try:
            cam_pos = np.asarray(client.camera.position, dtype=np.float32)   # world
            cam_wxyz = np.asarray(client.camera.wxyz, dtype=np.float32)      # camera-to-world
            cam_near = float(self.camera_znear)
            cam_far = float(self.camera_zfar)
        except Exception:
            return None

        width = int(self.image_size)
        height = int(self.image_size)

        # Build GS attributes from PLY
        centers = data.centers.astype(np.float32, copy=False)
        rotations = data.rotations.astype(np.float32, copy=False)
        scales = data.scales.astype(np.float32, copy=False)
        opacities = data.opacities.reshape(-1).astype(np.float32, copy=False)
        rgbs = data.rgbs.astype(np.float32, copy=False)


        # Build OpenGL camera matrices
        # camera-to-world from viser
        R_cw = tf.SO3(cam_wxyz).as_matrix().astype(np.float32)
        t_cw = cam_pos.astype(np.float32)

        T_cw = np.eye(4, dtype=np.float32)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw

        # world-to-camera
        T_wc = np.linalg.inv(T_cw).astype(np.float32)
        viewmat = torch.from_numpy(T_wc[None]).float()

        # world-to-camera
        T_wc = np.linalg.inv(T_cw).astype(np.float32)

        viewmat = torch.from_numpy(T_wc[None]).float()
        # OpenGL projection (L4GM style)
        tan_half_fov = np.tan(0.5 * np.deg2rad(float(self.fov_deg)))
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1.0 / tan_half_fov
        proj[1, 1] = 1.0 / tan_half_fov
        proj[2, 2] = (cam_far + cam_near) / (cam_far - cam_near)
        proj[3, 2] = -(cam_far * cam_near) / (cam_far - cam_near)
        proj[2, 3] = 1.0

        # Intrinsics matrix for gsplat
        f = width / (2.0 * tan_half_fov)
        K = torch.tensor(
            [[f, 0.0, width / 2.0],
            [0.0, f, height / 2.0],
            [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        # Torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        means = torch.from_numpy(centers).to(device)
        quats = torch.from_numpy(rotations).to(device)
        scales_t = torch.from_numpy(scales).to(device)
        opacities_t = torch.from_numpy(opacities).to(device)
        colors_t = torch.from_numpy(rgbs).to(device)

        viewmats = viewmat.to(device)
        Ks = torch.stack([K]).to(device)
        bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        rendered_image_all, rendered_alpha_all, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales_t,
            opacities=opacities_t,
            colors=colors_t,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=cam_near,
            far_plane=cam_far,
            packed=False,
            backgrounds=torch.stack([bg]),
            render_mode="RGB",
        )



        if rendered_image_all is None or len(rendered_image_all) == 0:
            return None

        img = rendered_image_all[0]
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()

        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)

        return img



    def _update_raster_for_client(self, client: viser.ClientHandle) -> None:
        try:
            handle = self.raster_handles.get(client)
            if handle is None:
                return
            img = self._rasterize_current_view(client)

            if img is None:
                return
            try:
                handle.image = ensure_rgb(img)
            except Exception:
                handle.image = img
        except Exception:
            pass

    def _schedule_raster_update(self, client: viser.ClientHandle) -> None:
        if not bool(self.raster_preview_enabled):
            return
        # Skip raster preview for extremely large frames to avoid GPU stalls.
        try:
            cached = self.sequence.cache.get(int(self.current_frame))
            if cached is not None and int(cached.centers.shape[0]) > int(self.raster_preview_max_points) > 0:
                return
        except Exception:
            pass

        evt = self._raster_update_events.get(client)
        if evt is None:
            evt = threading.Event()
            self._raster_update_events[client] = evt

        if client not in self._raster_update_threads or not self._raster_update_threads[client].is_alive():
            def _worker() -> None:
                while True:
                    try:
                        evt.wait()
                        evt.clear()
                        if bool(self._raster_update_inflight.get(client, False)):
                            continue
                        self._raster_update_inflight[client] = True
                        try:
                            self._update_raster_for_client(client)
                        finally:
                            self._raster_update_inflight[client] = False
                    except Exception:
                        time.sleep(0.05)

            t = threading.Thread(target=_worker, daemon=True)
            self._raster_update_threads[client] = t
            t.start()

        evt.set()

    def _schedule_apply_frame(
        self,
        client: viser.ClientHandle,
        idx: int,
        preview_handle: viser.GuiImageHandle,
        *,
        debounce: bool = True,
    ) -> None:
        """Schedule a frame apply in a worker thread.

        - When `debounce=True`, coalesces rapid scrubbing updates.
        - When `debounce=False`, applies as soon as possible (used for playback).
        """
        if not self.sequence.ply_paths:
            return
        idx = int(idx)
        self._frame_apply_latest_idx[client] = idx
        self._frame_apply_debounce_s[client] = self.frame_scrub_debounce_s if bool(debounce) else 0.0
        evt = self._frame_apply_events.get(client)
        if evt is None:
            evt = threading.Event()
            self._frame_apply_events[client] = evt

        if client not in self._frame_apply_threads or not self._frame_apply_threads[client].is_alive():
            def _worker() -> None:
                while True:
                    try:
                        evt.wait()
                        evt.clear()

                        # Debounce: if more events arrive during the wait, restart.
                        debounce_s = float(self._frame_apply_debounce_s.get(client, self.frame_scrub_debounce_s))
                        while debounce_s > 1e-6:
                            time.sleep(debounce_s)
                            if evt.is_set():
                                evt.clear()
                                debounce_s = float(
                                    self._frame_apply_debounce_s.get(client, self.frame_scrub_debounce_s)
                                )
                                continue
                            break

                        latest = int(self._frame_apply_latest_idx.get(client, idx))
                        # Only apply if still the current frame (avoid stale updates).
                        if int(self.current_frame) != latest:
                            continue
                        if bool(self._frame_apply_inflight.get(client, False)):
                            continue
                        self._frame_apply_inflight[client] = True
                        try:
                            self._apply_frame(latest, preview_handle)
                        finally:
                            self._frame_apply_inflight[client] = False
                    except Exception:
                        # If anything goes wrong, keep the worker alive but avoid a tight loop.
                        time.sleep(0.05)

            t = threading.Thread(target=_worker, daemon=True)
            self._frame_apply_threads[client] = t
            t.start()

        evt.set()

    def _apply_frame(self, index: int, preview_handle: viser.GuiImageHandle) -> None:
        t_apply0 = time.perf_counter()
        try:
            gs_path = self.sequence.gs_path(index)
            n_hint = None
            try:
                n_hint = self.sequence.vertex_count(index)
            except Exception:
                n_hint = None

            # Fast path: if a frame is huge and we're going to downsample for display anyway,
            # avoid `load_splat_from_ply` (which performs per-point eigen decompositions).
            if (
                int(self.gs_display_max_points) > 0
                and n_hint is not None
                and int(n_hint) > int(self.gs_display_max_points)
            ):
                data_full = None
                data_display = self._load_display_splats_from_ply(
                    gs_path, max_points=int(self.gs_display_max_points), seed=int(index) + 1337
                )
            else:
                t0 = time.perf_counter()
                data_full = self.sequence.load(index)
                self._profile_log("sequence.load(full)", time.perf_counter() - t0, threshold_s=0.0)
                data_display = self._get_display_splats(index, data_full)
        except FileNotFoundError:
            if self.gs_handle is not None:
                self.gs_handle.remove()
                self.gs_handle = None
            self._update_preview(preview_handle)
            self._refresh_selection_ui()
            self._render_selection_overlay(False)
            return

        if (
            n_hint is not None
            and int(n_hint) > int(self.gs_display_max_points)
            and int(self.gs_display_max_points) > 0
            and index not in self._notified_downsample_frames
        ):
            self._notified_downsample_frames.add(int(index))
            try:
                if self.last_client is not None:
                    self.last_client.add_notification(
                        "GS",
                        f"Frame {index}: {int(n_hint):,} gaussians; displaying {len(data_display.centers):,} for performance.",
                        color="orange",
                        auto_close_seconds=6,
                    )
            except Exception:
                pass

        t1 = time.perf_counter()
        self._update_gaussian_handle(data_display)
        self._profile_log("update_gaussians", time.perf_counter() - t1, threshold_s=0.0)
        self._update_preview(preview_handle)
        self._refresh_selection_ui()
        self._render_selection_overlay(self.selection_overlay_enabled)

        # Update scene bounds only once after the first successful load (important when booting empty).
        if not bool(getattr(self, "_bounds_initialized", False)):
            try:
                self.scene_center, self.scene_radius = compute_bounds(data_display)
                if not self.camera_target_origin:
                    self.camera_target = self.scene_center.copy()
                self._bounds_initialized = True
            except Exception:
                pass

        self._profile_log(f"apply_frame({index})", time.perf_counter() - t_apply0, threshold_s=0.0)

    def _backend_base_url(self) -> str:
        dbase = self._dispatcher_base_url()
        if dbase:
            return dbase
        if not self.backend_url:
            raise RuntimeError("backend_url is not set")
        base = self.backend_url.rstrip("/")
        # If user provided a leaf endpoint like /optimize_roi, strip it.
        if "/" in base:
            tail = base.rsplit("/", 1)[-1]
            if tail in {"select_roi", "optimize_roi", "fix_roi"}:
                return base.rsplit("/", 1)[0]
        return base

    def _sam3d_backend_base_url(self) -> str:
        dbase = self._dispatcher_base_url()
        if dbase:
            return dbase
        if not self.sam3d_backend_url:
            raise RuntimeError("sam3d_backend_url is not set")
        base = self.sam3d_backend_url.rstrip("/")
        if "/" in base:
            tail = base.rsplit("/", 1)[-1]
            if tail in {"health", "status", "sam", "sam3d", "predict", "generate", "set_image"}:
                return base.rsplit("/", 1)[0]
        return base

    def _dispatcher_base_url(self) -> str | None:
        base = str(self.dispatcher_url or "").strip()
        if not base:
            return None
        base = base.rstrip("/")
        if "/" in base:
            tail = base.rsplit("/", 1)[-1]
            if tail in {"jobs", "submit", "status", "result", "health", "dispatcher"}:
                return base.rsplit("/", 1)[0]
        return base

    def _select_roi_endpoint_url(self) -> str:
        dbase = self._dispatcher_base_url()
        if dbase:
            return dbase + "/select_roi"
        if not self.backend_url:
            raise RuntimeError("backend_url is not set")
        base = self.backend_url.rstrip("/")
        if base.endswith("select_roi") or base.endswith("optimize_roi"):
            return base
        return self._backend_base_url() + "/select_roi"

    def _is_retryable_exception(self, exc: Exception) -> bool:
        if isinstance(exc, urllib.error.URLError):
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (TimeoutError, ConnectionRefusedError, socket.timeout)):
                return True
            txt = str(reason if reason is not None else exc).lower()
            if "timed out" in txt or "connection refused" in txt or "connection reset" in txt:
                return True
        if isinstance(exc, (TimeoutError, ConnectionResetError, socket.timeout)):
            return True
        return False

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        payload: Dict | None,
        timeout: float,
        retries: int,
    ) -> Dict:
        attempts = max(1, int(retries) + 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            data = None if payload is None else json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers=client_headers(include_json=data is not None),
                method=method.upper(),
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    return {"status": "error", "message": body}
                if isinstance(parsed, dict):
                    return parsed
                return {"status": "success", "data": parsed}
            except urllib.error.HTTPError as exc:
                raw = ""
                try:
                    raw = exc.read().decode("utf-8")
                except Exception:
                    raw = ""
                try:
                    parsed = json.loads(raw) if raw else {}
                except Exception:
                    parsed = {}
                err = parsed.get("error") or parsed.get("message") or (raw.strip() if raw else None)
                tb = parsed.get("traceback")
                detail = ""
                if err:
                    detail = str(err)
                if tb:
                    tb_s = str(tb)
                    if len(tb_s) > 1200:
                        tb_s = tb_s[-1200:]
                    detail = (detail + " | traceback: " + tb_s) if detail else ("traceback: " + tb_s)

                if attempt + 1 < attempts and int(getattr(exc, "code", 0)) in {408, 429, 500, 502, 503, 504}:
                    backoff = self.http_retry_backoff_s * (2 ** attempt)
                    if backoff > 0:
                        time.sleep(backoff)
                    continue

                raise RuntimeError(
                    f"{method.upper()} {url} failed: HTTP {getattr(exc, 'code', '?')} {getattr(exc, 'reason', '')}"
                    + (f"; {detail}" if detail else "")
                ) from exc
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < attempts and self._is_retryable_exception(exc):
                    backoff = self.http_retry_backoff_s * (2 ** attempt)
                    if backoff > 0:
                        time.sleep(backoff)
                    continue
                raise RuntimeError(f"{method.upper()} {url} failed: {type(exc).__name__}: {exc}") from exc

        raise RuntimeError(
            f"{method.upper()} {url} failed after {attempts} attempts"
            + (f": {type(last_exc).__name__}: {last_exc}" if last_exc is not None else "")
        )

    def _post_json(self, url: str, payload: Dict, timeout: float = 30, retries: int | None = None) -> Dict:
        return self._request_json(
            method="POST",
            url=url,
            payload=payload,
            timeout=timeout,
            retries=self.http_post_retries if retries is None else int(retries),
        )

    def _get_json(self, url: str, timeout: float = 10, retries: int | None = None) -> Dict:
        return self._request_json(
            method="GET",
            url=url,
            payload=None,
            timeout=timeout,
            retries=self.http_get_retries if retries is None else int(retries),
        )

    def _ensure_session(self, client: viser.ClientHandle) -> None:
        if self.session_id is not None:
            return
        base = self._backend_base_url()
        resp = self._post_json(
            base + "/session/start",
            {
                "input_dir": str(self.input_dir.resolve()),
                "workspace_root": str(self.workspace.root.resolve()),
                "use_existing_workspace": True,
                "chunk_size": 8,
                "eval_gt_dir": str(self.eval_dir.resolve()),
                "metric_online": self.metric_online,
            },
            timeout=60,
        )
        if resp.get("status") != "success":
            raise RuntimeError(resp.get("message") or "session/start failed")
        self.session_id = str(resp.get("session_id"))
        session_root = resp.get("paths", {}).get("session_root")
        if not session_root:
            raise RuntimeError("session/start did not return paths.session_root")
        self.session_root = Path(session_root)
        # Retarget workspace GS dir to this session so ROI/Fix use generated PLYs.
        # New convention: session outputs live under session_root/gs_sequence.
        # For backwards compatibility (older backends), fall back to session_root/gs.
        session_gs_dir = self.session_root / "gs_sequence"
        if not session_gs_dir.exists():
            legacy = self.session_root / "gs"
            if legacy.exists():
                session_gs_dir = legacy
        # If session_root is the current workspace root, gs_sequence is already the workspace gs_dir
        # (and backend may have replaced a symlink with a real directory). Avoid relinking to self.
        try:
            if session_gs_dir.resolve() != self.workspace.gs_dir.resolve():
                self.workspace.relink_gs_dir(session_gs_dir)
        except Exception:
            if session_gs_dir != self.workspace.gs_dir:
                self.workspace.relink_gs_dir(session_gs_dir)

        # Switch sequence tokens to input frames for generation workflow.
        new_tokens = _list_frame_tokens(self.input_dir)
        self.sequence.cache.clear()
        self.sequence.ply_dir = self.workspace.gs_dir
        self.sequence.frame_tokens = list(new_tokens)
        self.sequence.ply_paths = [self.workspace.gs_dir / f"{t}.ply" for t in self.sequence.frame_tokens]

        # Reset metric buffers when a new session starts.
        self.metric_frames.clear()
        self.metric_psnr.clear()
        self.metric_lpips.clear()
        self.metric_seen_frames.clear()

        # Refresh timeline + sliders for all connected clients.
        self.timeline_base, self.timeline_spans = build_timeline_strip(
            [self.input_dir / f"{token}.png" for token in self.sequence.frame_tokens]
        )
        max_idx = max(0, len(self.sequence.frame_tokens) - 1)
        for c, slider in list(self.frame_slider_handles.items()):
            try:
                slider.max = max_idx
                slider.disabled = len(self.sequence.frame_tokens) == 0
                if int(slider.value) > max_idx:
                    slider.value = max_idx
            except Exception:
                pass
        for c, t_handle in list(self.timeline_handles.items()):
            try:
                # Update to show only the current input frame instead of the full timeline strip.
                frame_token = self.sequence.frame_tokens[int(self.current_frame)]
                img_path = self.workspace.image_path_for_view(frame_token, self.current_view)
                if img_path.exists():
                    t_handle.image = ensure_rgb(load_image(img_path))
                else:
                    t_handle.image = np.zeros((self.image_size // 4, self.image_size // 4, 3), dtype=np.uint8)
            except Exception:
                pass
        for c, slider in list(self.multi_view_start_sliders.items()):
            try:
                slider.max = max_idx
                slider.value = min(slider.value, max_idx)
            except Exception:
                pass
        for c, slider in list(self.multi_view_end_sliders.items()):
            try:
                slider.max = max_idx
                slider.value = min(slider.value, max_idx)
            except Exception:
                pass
        self.multi_view_start = min(self.multi_view_start, max_idx)
        self.multi_view_end = min(max(self.multi_view_start, self.multi_view_end), max_idx)
        self.memory_selected_frames = []
        for handle in self.multi_view_status_handles.values():
            handle.content = f"Multi-view ready: frames 0-{max_idx}"
        for handle in self.memory_status_handles.values():
            handle.value = "memory n=0 ids=[]"
        for c, dd in list(self.memory_remove_dropdowns.items()):
            try:
                dd.options = ["(empty)"]
                dd.value = "(empty)"
                dd.disabled = True
            except Exception:
                pass
        for c, btn in list(self.memory_remove_buttons.items()):
            try:
                btn.disabled = True
            except Exception:
                pass
        for handle in self.session_status_handles.values():
            handle.value = "state=idle"
        for handle in self.metric_status_handles.values():
            handle.value = "metric=online" if self.metric_online else "metric=offline"
        client.add_notification(
            "Session started",
            f"session_id={self.session_id}",
            color="blue",
            auto_close_seconds=6,
        )

    def _set_auto_vlm_eval_status(self, text: str) -> None:
        for handle in list(self.auto_vlm_eval_status_handles.values()):
            try:
                handle.value = str(text)
            except Exception:
                pass

    def _run_auto_vlm_eval_for_frames(
        self,
        client: viser.ClientHandle,
        frame_indices: Sequence[int],
    ) -> None:
        if not bool(self.auto_vlm_eval_enabled):
            return

        ops = getattr(getattr(self, "vlm_assistor_plugin", None), "ops", None)
        if ops is None:
            self._set_auto_vlm_eval_status("skip: vlm_assistor ops unavailable")
            return
        st = None
        try:
            st = getattr(self.vlm_assistor_plugin, "state", None)
            if st is not None and (not bool(getattr(st, "enabled", True))):
                self._set_auto_vlm_eval_status("skip: VLM Assistor disabled")
                return
        except Exception:
            pass

        total_frames = len(self.sequence.ply_paths)
        uniq = sorted({int(x) for x in list(frame_indices or [])})
        frames = [x for x in uniq if 0 <= x < total_frames]
        if not frames:
            self._set_auto_vlm_eval_status("skip: no valid frames")
            return

        # Keep Agent4 input consistent with front/back/left/right.
        try:
            if st is not None and hasattr(st, "lock"):
                with st.lock:
                    st.agent4_views_per_frame = 4
        except Exception:
            pass

        self._set_auto_vlm_eval_status(f"running 0/{len(frames)}")
        for i, frame_idx in enumerate(frames, start=1):
            self._set_auto_vlm_eval_status(f"rendering frame={frame_idx} ({i}/{len(frames)})")
            try:
                self._render_multiview_range(
                    client,
                    frame_idx,
                    frame_idx,
                    status_handle=None,
                    trigger_button=None,
                    azimuths=[0, 90, 180, 270],
                    notify=False,
                    purge_existing=True,
                )
            except Exception as exc:
                self._set_auto_vlm_eval_status(f"render failed frame={frame_idx}: {exc}")
                continue

            self._set_auto_vlm_eval_status(f"evaluating frame={frame_idx} ({i}/{len(frames)})")
            try:
                if hasattr(ops, "run_4d_memory_judge_sync"):
                    out = ops.run_4d_memory_judge_sync(
                        client,
                        window_size=int(self.auto_vlm_eval_history_window),
                        frame_idx=frame_idx,
                    )
                    score = out.get("quality_score", "NA") if isinstance(out, dict) else "NA"
                    self._set_auto_vlm_eval_status(
                        f"done frame={frame_idx} score={score} ({i}/{len(frames)})"
                    )
                else:
                    ops.run_4d_memory_judge(
                        client,
                        window_size=int(self.auto_vlm_eval_history_window),
                    )
                    self._set_auto_vlm_eval_status(
                        f"submitted frame={frame_idx} (async) ({i}/{len(frames)})"
                    )
            except Exception as exc:
                self._set_auto_vlm_eval_status(f"eval failed frame={frame_idx}: {exc}")

        self._set_auto_vlm_eval_status("idle")

    def _update_preview(self, preview_handle: viser.GuiImageHandle) -> None:
        frame_token = self.sequence.frame_tokens[self.current_frame]
        img_path = self.workspace.image_path_for_view(frame_token, self.current_view)
        if img_path.exists():
            preview_handle.image = ensure_rgb(iio.imread(img_path))
        if self.frustum_handles:
            for h in self.frustum_handles:
                h.remove()
            self.frustum_handles = self._build_frustums_for_frame(
                self.current_frame, self.last_client
            )

    def _clear_selection_overlay(self) -> None:
        if self.selection_handle is not None:
            self.selection_handle.remove()
            self.selection_handle = None

    def _apply_roi_overlay(self, image: np.ndarray, bbox: Dict[str, int], alpha: float = 0.65) -> np.ndarray:
        """Dim everything outside the ROI with a gray overlay."""
        img = ensure_rgb(image)
        h, w = img.shape[:2]
        x0, x1 = int(bbox["x0"]), int(bbox["x1"])
        y0, y1 = int(bbox["y0"]), int(bbox["y1"])
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(y0 + 1, min(y1, h))
        overlay = np.full_like(img, 120, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        out = img.copy()
        out[~mask] = (alpha * overlay[~mask] + (1 - alpha) * img[~mask]).astype(np.uint8)
        return out

    def _render_selection_overlay(self, enable: bool) -> None:
        self._clear_selection_overlay()
        if not enable or not self.selection_info:
            return
        if self.selection_info.get("frame") != self.current_frame:
            return

        indices = np.asarray(self.selection_info.get("indices", []), dtype=np.int64)
        if indices.size == 0:
            return

        # If this frame is huge and we're displaying a downsampled subset, selection overlay indices
        # (which reference the full set) won't match. Avoid triggering a full load (very slow).
        try:
            n_hint = self.sequence.vertex_count(int(self.current_frame))
            if int(self.gs_display_max_points) > 0 and int(n_hint) > int(self.gs_display_max_points):
                return
        except Exception:
            pass

        data = self.sequence.load(self.current_frame)
        indices = indices[(indices >= 0) & (indices < data.centers.shape[0])]
        if indices.size == 0:
            return

        # Limit overlay size to avoid massive websocket transfers and client stalls.
        # Also keep a fixed size to avoid Viser client resource rebuilds when the number of
        # overlay gaussians changes (see doc/gs_viser_display.md).
        overlay_fixed = int(getattr(self, "selection_overlay_fixed_points", 50_000))
        overlay_k = overlay_fixed if overlay_fixed > 0 else int(getattr(self, "selection_overlay_max_points", 50_000))
        overlay_k = max(0, int(overlay_k))
        if overlay_k > 0 and indices.size > overlay_k:
            try:
                if str(getattr(self, "gs_display_sample", "opacity")) == "random":
                    rng = np.random.default_rng(int(self.current_frame) + 424242)
                    keep = rng.choice(indices.size, size=overlay_k, replace=False)
                    indices = indices[keep]
                else:
                    # Rank by opacity (already [0,1] in SplatData).
                    weights = data.opacities.reshape(-1).astype(np.float32, copy=False)
                    w_sel = weights[indices]
                    keep = np.argpartition(w_sel, -overlay_k)[-overlay_k:]
                    indices = indices[keep]
            except Exception:
                indices = indices[:overlay_k]

        centers_sel = data.centers[indices]
        cov_sel     = data.covariances[indices]
        rgbs_sel    = np.full_like(data.rgbs[indices], np.array([1.0, 0.4, 0.4], dtype=np.float32))
        opacities_sel = np.clip(data.opacities[indices] * 1.25, 0.0, 1.0)

        # For performance: build one packed buffer update (avoid 4x full-buffer updates).
        sel_data = SplatData(
            centers=centers_sel.astype(np.float32, copy=False),
            rgbs=rgbs_sel.astype(np.float32, copy=False),
            opacities=opacities_sel.astype(np.float32, copy=False),
            covariances=cov_sel.astype(np.float32, copy=False),
            scales=np.zeros((centers_sel.shape[0], 3), dtype=np.float32),
            rotations=np.zeros((centers_sel.shape[0], 4), dtype=np.float32),
            sh=np.zeros((centers_sel.shape[0], 3), dtype=np.float32),
            opacity_raw=np.zeros((centers_sel.shape[0],), dtype=np.float32),
            centers_raw=centers_sel.astype(np.float32, copy=False),
        )
        if overlay_fixed > 0:
            sel_data = self._pad_splatdata_to_n(sel_data, overlay_fixed)
        buf = pack_splat_buffer(sel_data)
        if not hasattr(self, "selection_handle") or self.selection_handle is None:
            # Create with exact size once; subsequent updates use buffer to resize automatically.
            self.selection_handle = self.server.scene.add_gaussian_splats(
                "/roi_selection",
                centers=sel_data.centers,
                covariances=sel_data.covariances,
                rgbs=sel_data.rgbs,
                opacities=sel_data.opacities,
            )
        try:
            self.selection_handle.buffer = buf
            self.selection_handle.visible = True
        except Exception as e:
            print(f"[Selection overlay update failed] {e}")
            try:
                self.selection_handle.visible = False
            except Exception:
                pass


    def _refresh_selection_ui(self) -> None:
        for cb in list(self.selection_ui_hooks):
            try:
                cb()
            except Exception as exc:
                print(f"Selection UI update failed: {exc}")

    def _set_selection_info(
        self, indices: List[int], frame: int, view: str, message: str | None = None
    ) -> None:
        # Preserve the user's overlay toggle.
        # Previously this forced overlay on for any non-empty selection, which can cause large
        # websocket transfers (and UI stalls) when selecting many splats.
        keep_overlay = bool(self.selection_overlay_enabled)
        self.selection_info = {
            "indices": list(indices),
            "frame": int(frame),
            "view": view,
            "message": message,
        }
        self.selection_overlay_enabled = keep_overlay and bool(indices)
        self._render_selection_overlay(self.selection_overlay_enabled)
        self._refresh_selection_ui()

    def _rerun_last_roi(self, client: viser.ClientHandle) -> None:
        if not self.backend_url or not self.last_roi_payload:
            return
        payload = json.loads(json.dumps(self.last_roi_payload))
        payload["depth_margin_ratio"] = self.depth_margin_ratio
        payload["min_depth_margin"] = self.min_depth_margin
        self.last_roi_payload = payload
        self._post_payload(payload, client)

    def _handle_roi(self, event: viser.ScenePointerEvent, client: viser.ClientHandle):
        if event.screen_pos is None or len(event.screen_pos) < 2:
            client.add_notification(
                "ROI failed",
                "No selection received.",
                color="red",
                auto_close_seconds=10,
            )
            return

        (x0n, y0n), (x1n, y1n) = event.screen_pos[:2]
        render_w, render_h = self._choose_capture_size(client)
        x0 = int(max(0, min(x0n, x1n) * render_w))
        x1 = int(min(1, max(x0n, x1n)) * render_w)
        y0 = int(max(0, min(y0n, y1n) * render_h))
        y1 = int(min(1, max(y0n, y1n)) * render_h)
        if x1 <= x0 or y1 <= y0:
            client.add_notification(
                "ROI skipped",
                "Selection is empty, try again.",
                color="orange",
                auto_close_seconds=10,
            )
            return

        render = self._capture_clean_render(client, render_w, render_h)
        roi = render[y0:y1, x0:x1]

        frame_token = self.sequence.frame_tokens[self.current_frame]
        gs_path = self.workspace.gs_path_for_frame(self.sequence.gs_path(self.current_frame))
        gt_path = self.workspace.image_path_for_view(frame_token, self.current_view)
        output_dir = self.workspace.output_dir_for(frame_token, self.current_view)

        camera = client.camera
        sel_dir = self.workspace.selections_dir / f"{frame_token}_{self.current_view}"
        payload = {
            "experiment_id": f"{self.workspace.root.name}_{frame_token}_{self.current_view}_{int(time.time())}",
            "frame_id": int(self.current_frame),
            "camera_id": self.current_view,
            "frame_token": frame_token,
            "image_size": [render_w, render_h],
            "bbox_2d": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "paths": {
                "gs_ply": str(gs_path),
                "gt_image": str(gt_path),
                "output_dir": str(output_dir),
                "workspace_root": str(self.workspace.root),
                "capture_image": str(sel_dir / "render.png"),
                "roi_image": str(sel_dir / "roi.png"),
                "eval_gt_dir": str(self.workspace.images_root / "eval_gt"),
            },
            "camera": {
                "wxyz": camera.wxyz.tolist(),
                "position": camera.position.tolist(),
                "look_at": camera.look_at.tolist(),
                "up": camera.up_direction.tolist(),
                "fov_deg": self.fov_deg,
                "near": float(camera.near),
                "far": float(camera.far),
            },
            "opt_params": {
                "iters": 5000,
                "render_resolution": self.image_size,
                "lr_color": 0.0025,
                "lr_opacity": 0.025,
                "lr_scale": 0.005,
                "lr_rotation": 0.001,
                "lr_final_factor": 0.01,
                "lambda_dssim": 0.2,
                "optimize_means": True,
                "optimize_rotation": True,
                "video_samples": self.video_samples,
                "render_video_only": self.render_video_only,
            },
            "depth_margin_ratio": self.depth_margin_ratio,
            "min_depth_margin": self.min_depth_margin,
            "video_samples": self.video_samples,
            "render_video_only": self.render_video_only,
        }

        saved = self.workspace.save_selection(frame_token, self.current_view, render, roi, payload)
        (output_dir / "request.json").write_text(json.dumps(payload, indent=2))
        self.last_roi_payload = payload

        # Update preview with dimmed overlay outside ROI.
        overlay_img = self._apply_roi_overlay(render, payload["bbox_2d"])
        handle = self.preview_handles.get(client)
        if handle is not None:
            handle.image = overlay_img

        client.add_notification(
            "ROI ready",
            f"Saved request to {saved['payload']}",
            color="green",
            with_close_button=True,
            auto_close_seconds=10,
        )

        # Fire MV-TAP tracklet request if configured.
        if self.mvtap_url:
            track_payload = {
                "experiment_id": payload.get("experiment_id"),
                "frame_id": int(self.current_frame),
                "camera_id": self.current_view,
                "frame_token": frame_token,
                "image_size": [render_w, render_h],
                "bbox_2d": payload["bbox_2d"],
                "num_points": int(self.tracklet_num_points),
                "paths": {
                    "workspace_root": str(self.workspace.root),
                    "input_dir": str(self.workspace.images_root / "input"),
                    "eval_gt_dir": str(self.workspace.images_root / "eval_gt"),
                    "output_dir": str(output_dir),
                },
            }
            self._post_mvtap(track_payload, client)

        if self.backend_url:
            self._post_payload(payload, client)

    def _post_payload(self, payload: Dict, client: viser.ClientHandle) -> None:
        try:
            parsed = self._post_json(self._select_roi_endpoint_url(), payload, timeout=30)
            body = json.dumps(parsed, ensure_ascii=False)

            if parsed and parsed.get("status") == "success":
                selection = parsed.get("selection", {})
                indices = selection.get("gs_indices") or selection.get("indices") or []
                indices = [int(i) for i in indices]
                self._set_selection_info(
                    indices,
                    frame=int(payload.get("frame_id", self.current_frame)),
                    view=str(payload.get("camera_id", self.current_view)),
                    message=body[:240],
                )
                client.add_notification(
                    "ROI selection",
                    f"Backend returned {len(indices)} splats.",
                    color="green",
                    with_close_button=True,
                    auto_close_seconds=10,
                )
                return

            client.add_notification(
                "Backend response",
                body[:240] + ("..." if len(body) > 240 else ""),
                color="blue",
                auto_close_seconds=10,
            )
        except Exception as exc:
            client.add_notification(
                "Backend error",
                f"{exc.__class__.__name__}: {exc}",
                color="red",
                with_close_button=True,
                auto_close_seconds=10,
            )

    def _fix_endpoint_url(self) -> str | None:
        if not self.backend_url:
            return None
        base = self._backend_base_url().rstrip("/")
        if base.endswith("select_roi") or base.endswith("optimize_roi") or base.endswith("archor_view_refine"):
            return base.rsplit("/", 1)[0] + "/fix_roi"
        return base + "/fix_roi"

    def _archor_endpoint_url(self) -> str | None:
        if not self.backend_url:
            return None
        base = self._backend_base_url().rstrip("/")
        if base.endswith("archor_view_refine"):
            return base
        if base.endswith("select_roi") or base.endswith("optimize_roi") or base.endswith("fix_roi"):
            return base.rsplit("/", 1)[0] + "/archor_view_refine"
        return base + "/archor_view_refine"

    def _post_mvtap(self, payload: Dict, client: viser.ClientHandle) -> None:
        if not self.mvtap_url:
            return
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.mvtap_url,
                data=data,
                headers=client_headers(include_json=True),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None

            if parsed and parsed.get("status") == "success":
                paths = parsed.get("paths", {})
                overlay_path = paths.get("overlay")
                msg = f"Tracklet saved (N={parsed.get('num_points')})"
                if overlay_path and Path(overlay_path).exists():
                    try:
                        overlay_img = ensure_rgb(iio.imread(overlay_path))
                        handle = self.preview_handles.get(client)
                        if handle is not None:
                            handle.image = overlay_img
                    except Exception:
                        pass
                    msg += f"; overlay={overlay_path}"
                client.add_notification(
                    "Tracklet ready",
                    msg,
                    color="green",
                    with_close_button=True,
                    auto_close_seconds=8,
                )
                return

            client.add_notification(
                "Tracklet response",
                body[:240] + ("..." if len(body) > 240 else ""),
                color="blue",
                auto_close_seconds=10,
            )
        except urllib.error.URLError as exc:
            client.add_notification(
                "Tracklet error",
                f"{exc.__class__.__name__}: {exc}",
                color="red",
                with_close_button=True,
                auto_close_seconds=10,
            )

    def _fix_progress_url(self, fix_url: str | None) -> str | None:
        if not fix_url:
            return None
        base = fix_url.rstrip("/")
        if base.endswith("fix_progress"):
            return base
        if base.endswith("select_roi") or base.endswith("optimize_roi") or base.endswith("fix_roi") or base.endswith("archor_view_refine"):
            return base.rsplit("/", 1)[0] + "/fix_progress"

        if base.endswith("sv4d/inpaint"):
            return base.replace("sv4d/inpaint", "fix_progress")
        return base + "/fix_progress"

    def _disable_live_buttons(self, disabled: bool = True) -> None:
        for btn in self.cancel_live_buttons.values():
            try:
                btn.disabled = disabled
            except Exception:
                pass

    def _preview_live_gs(self, live_ply: str | Path, client: viser.ClientHandle) -> None:
        path = Path(live_ply)
        if not path.exists():
            return
        try:
            data = load_gaussian_from_ply(path)
        except Exception as exc:  # noqa: BLE001
            print(f"Live preview load failed: {exc}")
            return
        self._update_gaussian_handle(data)
        self.live_last_path = path
        self.live_preview_active = True
        self.live_preview_blocked = False
        self.live_preview_hidden = False
        btn = self.cancel_live_buttons.get(client)
        if btn is not None:
            btn.disabled = False
            btn.label = "Hide live preview"
        self._update_preview(self.preview_handles.get(client))

    def _toggle_live_preview(
        self,
        client: viser.ClientHandle,
        status_handle: viser.GuiTextHandle,
        toggle_button: viser.GuiButtonHandle,
    ) -> None:
        if not self.live_preview_active or self.live_last_path is None:
            client.add_notification(
                "No live preview",
                "Live preview is not available yet.",
                color="orange",
                auto_close_seconds=6,
            )
            toggle_button.disabled = True
            return
        if not self.live_preview_hidden:
            # Hide: show original GS for the frame and stop applying new live updates.
            self.live_preview_hidden = True
            self.live_preview_blocked = True
            self._apply_frame(self.current_frame, self.preview_handles.get(client))
            toggle_button.label = "Show live preview"
            status_handle.value = "Live preview hidden; showing original GS."
        else:
            # Show: reapply latest live ply and resume updates.
            if not Path(self.live_last_path).exists():
                client.add_notification(
                    "Live preview missing",
                    f"Live ply not found: {self.live_last_path}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=8,
                )
                toggle_button.disabled = True
                return
            self.live_preview_hidden = False
            self.live_preview_blocked = False
            self._preview_live_gs(self.live_last_path, client)
            toggle_button.label = "Hide live preview"
            status_handle.value = "Live preview restored."

    def _update_loss_curve_image(self, client: viser.ClientHandle) -> None:
        handle = self.loss_handles.get(client)
        if handle is None:
            return
        if not self.show_loss_curve:
            handle.image = np.zeros((64, 256, 3), dtype=np.uint8)
            return
        if not self.loss_history:
            handle.image = np.zeros((64, 256, 3), dtype=np.uint8)
            return
        fig, ax = plt.subplots(figsize=(4, 1.5), dpi=96)
        ax.plot(np.arange(len(self.loss_history)), self.loss_history, color="tab:blue")
        ax.set_xlabel("iter", fontsize=8)
        ax.set_ylabel("loss", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        if self.opt_metric_history:
            ax2 = ax.twinx()
            metric_label = self.opt_metric_label or "metric"
            ax2.plot(
                np.arange(len(self.opt_metric_history)),
                self.opt_metric_history,
                color="tab:orange",
            )
            ax2.set_ylabel(metric_label, color="tab:orange", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="tab:orange")
        fig.tight_layout(pad=0.6)
        fig.canvas.draw()
        try:
            rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = rgba[:, :, :3].copy()
        except Exception:
            # Fallback for older backends
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)
        handle.image = img
        # Persist the loss curve image for debugging and print its path.
        try:
            now = time.time()
            if now - self._loss_curve_last_saved_at >= 1.0:
                out_dir = self.workspace.results_dir if self.workspace else (ROOT / "outputs")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "loss_curve.png"
                iio.imwrite(out_path, img)
                self._loss_curve_last_saved_at = now
                self._loss_curve_last_path = out_path
                print(f"[loss-curve] saved: {out_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[loss-curve] save failed: {exc}")

    def _update_metric_curve_image(self, client: viser.ClientHandle) -> None:
        handle = self.metric_handles.get(client)
        if handle is None:
            return
        if not self.metric_online:
            handle.image = np.zeros((64, 256, 3), dtype=np.uint8)
            return
        if not self.show_metric_curve:
            handle.image = np.zeros((64, 256, 3), dtype=np.uint8)
            return
        if not self.metric_frames:
            handle.image = np.zeros((64, 256, 3), dtype=np.uint8)
            return
        fig, ax1 = plt.subplots(figsize=(4, 1.5), dpi=96)
        ax1.plot(self.metric_frames, self.metric_psnr, color="tab:green", label="PSNR")
        ax1.set_xlabel("frame", fontsize=8)
        ax1.set_ylabel("PSNR", color="tab:green", fontsize=8)
        ax1.tick_params(axis="y", labelcolor="tab:green")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()
        ax2.plot(self.metric_frames, self.metric_lpips, color="tab:red", label="LPIPS")
        ax2.set_ylabel("LPIPS", color="tab:red", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.tight_layout(pad=0.6)
        fig.canvas.draw()
        try:
            rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = rgba[:, :, :3].copy()
        except Exception:
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)
        handle.image = img

    def _apply_metric_entries(self, entries: List[Dict], default_frame_idx: int | None = None) -> None:
        """Append new metric entries, avoiding duplicates and keeping plots ordered."""
        updated = False
        for entry in entries:
            frame_token = entry.get("frame")
            try:
                frame_idx = int(frame_token) if frame_token is not None and str(frame_token).isdigit() else None
            except Exception:
                frame_idx = None
            frame_idx = frame_idx if frame_idx is not None else default_frame_idx
            if frame_idx is None or frame_idx in self.metric_seen_frames:
                continue
            self.metric_seen_frames.add(frame_idx)
            self.metric_frames.append(frame_idx)
            self.metric_psnr.append(float(entry.get("psnr", 0.0)))
            self.metric_lpips.append(float(entry.get("lpips", 0.0)))
            updated = True

        if updated:
            sorted_triplets = sorted(zip(self.metric_frames, self.metric_psnr, self.metric_lpips))
            self.metric_frames = [t[0] for t in sorted_triplets]
            self.metric_psnr = [t[1] for t in sorted_triplets]
            self.metric_lpips = [t[2] for t in sorted_triplets]
            if self.show_metric_curve:
                for c in list(self.metric_handles.keys()):
                    self._update_metric_curve_image(c)

    def _stop_metric_poll(self, client: viser.ClientHandle) -> None:
        evt = self.metric_poll_events.get(client)
        if evt is not None:
            evt.set()
        t = self.metric_poll_threads.get(client)
        if t is not None and t.is_alive():
            t.join(timeout=0.1)
        self.metric_poll_events.pop(client, None)
        self.metric_poll_threads.pop(client, None)

    def _start_metric_poll(self, client: viser.ClientHandle) -> None:
        if not self.metric_online or not self.backend_url or not self.session_id:
            return
        self._stop_metric_poll(client)

        stop_evt = threading.Event()
        self.metric_poll_events[client] = stop_evt
        base_url = self._backend_base_url()
        url = f"{base_url}/metric_progress?session_id={self.session_id}"

        def _run() -> None:
            while not stop_evt.is_set():
                try:
                    resp = self._get_json(url, timeout=10)
                    status = resp.get("status", "pending")
                    per_frame = resp.get("per_frame") or []
                    self._apply_metric_entries(per_frame)
                    completed = int(resp.get("completed", 0))
                    total = int(resp.get("total", 0))
                    psnr_avg = resp.get("psnr_avg")
                    lpips_avg = resp.get("lpips_avg")
                    for h in self.metric_status_handles.values():
                        if status in {"success", "disabled"}:
                            if psnr_avg is not None:
                                h.value = f"metric={status} psnr={psnr_avg:.3f} lpips={lpips_avg:.3f}"
                            else:
                                h.value = f"metric={status}"
                        else:
                            if total > 0:
                                h.value = f"metric={status} {completed}/{total}"
                            else:
                                h.value = f"metric={status}"
                    if status in {"success", "disabled", "error"}:
                        break
                except Exception as exc:  # noqa: BLE001
                    for h in self.metric_status_handles.values():
                        h.value = f"metric=error {exc}"
                    break
                stop_evt.wait(1.0)
            self.metric_poll_events.pop(client, None)
            self.metric_poll_threads.pop(client, None)

        t = threading.Thread(target=_run, daemon=True)
        self.metric_poll_threads[client] = t
        t.start()

    def _run_archor_refine(
        self,
        client: viser.ClientHandle,
        status_handle: viser.GuiMarkdownHandle,
        image_handles: List[viser.GuiImageHandle],
        anchor_view: str,
        apply_button: viser.GuiButtonHandle | None = None,
        cancel_button: viser.GuiButtonHandle | None = None,
    ) -> None:
        archor_url = self._archor_endpoint_url()
        if not archor_url:
            client.add_notification(
                "Anchor refine skipped",
                "Backend URL not set.",
                color="orange",
                auto_close_seconds=8,
            )
            return

        frame_idx = int(self.current_frame)
        if frame_idx < 0 or frame_idx >= len(self.sequence.ply_paths):
            client.add_notification(
                "Anchor refine skipped",
                "Invalid frame index.",
                color="orange",
                auto_close_seconds=8,
            )
            return

        gs_path = self.sequence.gs_path(frame_idx)
        if not gs_path.exists():
            client.add_notification(
                "Anchor refine skipped",
                f"GS file missing: {gs_path}",
                color="orange",
                auto_close_seconds=10,
            )
            return

        frame_token = self.sequence.frame_tokens[frame_idx]
        payload: Dict = {
            "anchor_view": anchor_view,
            "frame_token": frame_token,
            "paths": {
                "gs_ply": str(gs_path),
                "eval_gt_dir": str(self.workspace.images_root / "eval_gt"),
                "workspace_root": str(self.workspace.root),
            },
            "image_size": [int(self.image_size), int(self.image_size)],
            "camera": {"fov_deg": float(self.fov_deg)},
            "experiment_id": f"{frame_token}_archor_{anchor_view}",
        }

        if self.selection_info and self.selection_info.get("frame") == frame_idx:
            payload.setdefault("selection", {})["gs_indices"] = self.selection_info.get("indices", [])

        status_handle.content = f"Anchor refine running... (view={anchor_view})"
        self.live_preview_blocked = False
        self.live_preview_active = False
        self.live_preview_hidden = False
        progress_url = self._fix_progress_url(archor_url)
        progress_stop: threading.Event | None = None

        def _start_progress_poll(exp_id: str) -> threading.Event | None:
            if not progress_url:
                return None
            stop_evt = threading.Event()

            def _poll() -> None:
                while not stop_evt.is_set():
                    try:
                        query_url = f"{progress_url}?experiment_id={exp_id}"
                        req = urllib.request.Request(query_url, headers=client_headers())
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            body = resp.read().decode("utf-8")
                        parsed = json.loads(body)
                        status = parsed.get("status")
                        iter_cur = parsed.get("iter")
                        iter_max = parsed.get("iter_max")
                        loss_val = parsed.get("loss")
                        if status == "running":
                            msg = f"Anchor refine running... iter {iter_cur}/{iter_max}"
                            if loss_val is not None:
                                msg += f" loss={loss_val:.4f}"
                            status_handle.content = msg
                        elif status == "pending":
                            status_handle.content = "Anchor refine pending..."
                        render_path = parsed.get("progress_render") or parsed.get("render_path")
                        live_ply = parsed.get("live_ply")
                        if render_path and Path(render_path).exists():
                            try:
                                image_handles[0].image = ensure_rgb(iio.imread(render_path))
                            except Exception as exc:  # noqa: BLE001
                                print(f"Anchor progress image load failed: {exc}")
                        if live_ply and not self.live_preview_blocked and Path(live_ply).exists():
                            self._preview_live_gs(live_ply, client)
                            print("live_ply:", live_ply)
                            if cancel_button is not None:
                                cancel_button.disabled = False
                        if status not in ("pending", "running"):
                            # Ensure final live_ply/render is applied once before exit.
                            break
                    except Exception as exc:  # noqa: BLE001
                        print(f"Anchor progress poll failed: {exc}")
                    stop_evt.wait(1.0)

            threading.Thread(target=_poll, daemon=True).start()
            return stop_evt

        progress_stop = _start_progress_poll(payload["experiment_id"])

        def _execute() -> None:
            nonlocal progress_stop
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                archor_url,
                data=data,
                headers=client_headers(include_json=True),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    body = resp.read().decode("utf-8")
            except Exception as exc:  # noqa: BLE001
                status_handle.content = f"Anchor refine failed: {exc}"
                if progress_stop:
                    progress_stop.set()
                client.add_notification(
                    "Anchor refine error",
                    f"{exc.__class__.__name__}: {exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=10,
                )
                return

            if progress_stop:
                progress_stop.set()

            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None

            if not parsed or parsed.get("status") != "success":
                status_handle.content = "Anchor refine failed: backend returned error."
                client.add_notification(
                    "Anchor refine response",
                    body[:240] + ("..." if len(body) > 240 else ""),
                    color="orange",
                    auto_close_seconds=10,
                )
                return

            refined = parsed.get("refined_ply")
            render_before = parsed.get("render_before")
            render_after = parsed.get("render_after")
            loss_curve = parsed.get("loss_curve", [])
            time_sec = parsed.get("time_sec")

            if render_before and Path(render_before).exists():
                try:
                    image_handles[0].image = ensure_rgb(iio.imread(render_before))
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to load anchor before: {exc}")
            if len(image_handles) > 1 and render_after and Path(render_after).exists():
                try:
                    image_handles[1].image = ensure_rgb(iio.imread(render_after))
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to load anchor after: {exc}")

            status_lines = [
                f"Refined PLY: {refined}" if refined else "Refined PLY: (none)",
                f"Iters: {len(loss_curve)} | Final loss: {loss_curve[-1]:.4f}" if loss_curve else "No loss curve returned.",
                f"Time: {time_sec:.2f}s" if time_sec is not None else "",
            ]
            status_handle.content = "Anchor refine result:\n\n" + "\n".join([s for s in status_lines if s])

            if refined:
                self.last_refined_ply = Path(refined)
                if apply_button is not None:
                    apply_button.disabled = False
                self._refresh_selection_ui()
            self.live_preview_active = False
            self.live_last_path = None

        threading.Thread(target=_execute, daemon=True).start()

    def _run_sv4d_inpaint(
        self,
        client: viser.ClientHandle,
        status_handle: viser.GuiMarkdownHandle,
        image_handles: List[viser.GuiImageHandle],
        apply_button: viser.GuiButtonHandle | None = None,
        cancel_button: viser.GuiButtonHandle | None = None,
        anchor_view: str = "input",
    ) -> None:
        if not self.backend_url:
            client.add_notification(
                "SV4D skipped",
                "Backend URL not set.",
                color="orange",
                auto_close_seconds=8,
            )
            return

        frame_idx = int(self.current_frame)
        if frame_idx < 0 or frame_idx >= len(self.sequence.ply_paths):
            client.add_notification(
                "SV4D skipped",
                "Invalid frame index.",
                color="orange",
                auto_close_seconds=8,
            )
            return

        self._ensure_session(client)
        exp_id = f"sv4d_{self.sequence.frame_tokens[frame_idx]}_{int(time.time())}"

        selection_indices = None
        if self.selection_info and self.selection_info.get("frame") == frame_idx:
            selection_indices = self.selection_info.get("indices")
        if not selection_indices:
            selection_indices = None

        payload = {
            "session_id": self.session_id,
            "frame": frame_idx,
            "anchor_view": anchor_view,
            "image_size": self.image_size,
            "fov_deg": self.fov_deg,
            "up_axis": self.up_axis,
            "sigma_start": float(self.sv4d_sigma_start),
            "num_refine_steps": int(self.sv4d_num_refine_steps),
            "mask_dilate": int(self.sv4d_mask_dilate),
            "optimize": bool(self.sv4d_optimize),
            "selection_indices": selection_indices,
            "experiment_id": exp_id,
            "opt_params": {
                "lr_color": float(self.sv4d_lr_color),
                "lr_opacity": float(self.sv4d_lr_opacity),
                "lr_scale": float(self.sv4d_lr_scale),
                "lr_means": float(self.sv4d_lr_means),
                "lr_rotation": float(self.sv4d_lr_rotation),
                "lr_final_factor": float(self.sv4d_lr_final_factor),
                "mask_dilate": int(self.sv4d_mask_dilate),
                "split_enable": bool(self.sv4d_split_enable),
                "split_every": int(self.sv4d_split_every),
                "split_start": int(self.sv4d_split_start),
                "split_max_new": int(self.sv4d_split_max_new),
                "split_topk": int(self.sv4d_split_topk),
                "split_score": str(self.sv4d_split_score),
                "split_thresh": float(self.sv4d_split_thresh),
                "split_jitter": float(self.sv4d_split_jitter),
                "split_scale_shrink": float(self.sv4d_split_scale_shrink),
                "split_opacity_shrink": float(self.sv4d_split_opacity_shrink),
                "split_min_scale": float(self.sv4d_split_min_scale),
                "split_max_scale": float(self.sv4d_split_max_scale),
                "split_cooldown": int(self.sv4d_split_cooldown),
            },
        }

        status_handle.content = "SV4D inpaint running..."
        self.live_preview_blocked = False
        self.live_preview_active = False
        self.live_preview_hidden = False
        self.loss_history.clear()
        self.opt_metric_history.clear()
        self.opt_metric_label = None
        if client in self.loss_handles:
            self.loss_handles[client].image = np.zeros((64, 256, 3), dtype=np.uint8)

        sv4d_url = self._backend_base_url() + "/sv4d/inpaint"
        progress_url = self._fix_progress_url(sv4d_url)

        def _start_progress_poll() -> threading.Event | None:
            if not progress_url:
                return None
            stop_evt = threading.Event()

            def _poll() -> None:
                while not stop_evt.is_set():
                    try:
                        query_url = f"{progress_url}?experiment_id={exp_id}"
                        req = urllib.request.Request(query_url, headers=client_headers())
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            body = resp.read().decode("utf-8")

                        parsed = json.loads(body)
                        status = parsed.get("status")
                        iter_cur = parsed.get("iter")
                        iter_max = parsed.get("iter_max")
                        loss_val = parsed.get("loss")
                        print("query_url:", query_url)
                        print("sv4d painting status:", status, iter_cur, iter_max, loss_val)
                        if status == "running":
                            msg = f"SV4D optimize running... iter {iter_cur}/{iter_max}"
                            if loss_val is not None:
                                msg += f" loss={loss_val:.4f}"
                            status_handle.content = msg
                            if loss_val is not None:
                                try:
                                    self.loss_history.append(float(loss_val))
                                except Exception:
                                    pass
                                if self.show_loss_curve:
                                    self._update_loss_curve_image(client)
                        elif status == "pending":
                            status_handle.content = "SV4D optimize pending..."

                        metric_val = None
                        metric_label = None
                        if parsed.get("psnr") is not None:
                            metric_val = parsed.get("psnr")
                            metric_label = "psnr"
                        elif parsed.get("lpips") is not None:
                            metric_val = parsed.get("lpips")
                            metric_label = "lpips"
                        elif parsed.get("metric") is not None:
                            metric_val = parsed.get("metric")
                            metric_label = "metric"
                        elif parsed.get("metric_val") is not None:
                            metric_val = parsed.get("metric_val")
                            metric_label = "metric"

                        if metric_val is not None:

                            self.opt_metric_history.append(float(metric_val))
                            if metric_label:
                                self.opt_metric_label = metric_label
                            if self.show_loss_curve and self.loss_history:
                                self._update_loss_curve_image(client)

                        render_path = parsed.get("progress_render") or parsed.get("render_path")
                        live_ply = parsed.get("live_ply")
                        if render_path and Path(render_path).exists():
                            try:
                                image_handles[0].image = ensure_rgb(iio.imread(render_path))
                            except Exception as exc:  # noqa: BLE001
                                print(f"SV4D progress image load failed: {exc}")
                        if live_ply and not self.live_preview_blocked and Path(live_ply).exists():
                            self._preview_live_gs(live_ply, client)
                            if cancel_button is not None:
                                cancel_button.disabled = False
                        if status not in ("pending", "running"):
                            break
                    except Exception as exc:  # noqa: BLE001
                        print(f"SV4D progress poll failed: {exc}")
                    stop_evt.wait(1.0)

            threading.Thread(target=_poll, daemon=True).start()
            return stop_evt

        progress_stop = _start_progress_poll()

        def _execute() -> None:
            nonlocal progress_stop
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                sv4d_url,
                data=data,
                headers=client_headers(include_json=True),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    body = resp.read().decode("utf-8")
            except Exception as exc:  # noqa: BLE001
                status_handle.content = f"SV4D failed: {exc}"
                if progress_stop:
                    progress_stop.set()
                client.add_notification(
                    "SV4D inpaint failed",
                    f"{exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=10,
                )
                return

            if progress_stop:
                progress_stop.set()

            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None

            if not parsed or parsed.get("status") != "success":
                status_handle.content = "SV4D failed: backend returned error."
                client.add_notification(
                    "SV4D response",
                    body[:240] + ("..." if len(body) > 240 else ""),
                    color="orange",
                    auto_close_seconds=10,
                )
                return

            out_dir = parsed.get("output_dir") or parsed.get("sv4d_result", {}).get("output_dir")
            opt_result = parsed.get("optimize_result") or {}
            # Try multiple locations for refined PLY (different backends/versions)
            refined = (
                opt_result.get("refined_ply")
                or parsed.get("refined_ply")
                or parsed.get("sv4d_result", {}).get("refined_ply")
            )
            render_before = (
                opt_result.get("outputs", {}).get("render_before")
                or opt_result.get("render_before")
                or parsed.get("render_before")
            )
            render_after = (
                opt_result.get("outputs", {}).get("render_after")
                or opt_result.get("render_after")
                or parsed.get("render_after")
            )
            loss_curve = opt_result.get("loss_curve", [])

            if loss_curve:
                try:
                    self.loss_history = [float(x) for x in loss_curve]
                except Exception:
                    self.loss_history = []
                if self.show_loss_curve:
                    self._update_loss_curve_image(client)

            if render_before and Path(render_before).exists():
                try:
                    image_handles[0].image = ensure_rgb(iio.imread(render_before))
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to load SV4D before: {exc}")
            if len(image_handles) > 1 and render_after and Path(render_after).exists():
                try:
                    image_handles[1].image = ensure_rgb(iio.imread(render_after))
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to load SV4D after: {exc}")

            status_handle.content = f"SV4D done. output={out_dir or 'n/a'}"
            resolved_refined: Path | None = None
            if refined:
                # Resolve relative paths against reported output_dir when possible
                try:
                    p = Path(refined)
                    if not p.exists() and out_dir:
                        cand = Path(out_dir) / p.name
                        if cand.exists():
                            p = cand
                    if p.exists():
                        resolved_refined = p
                        self.last_refined_ply = resolved_refined
                        # enable apply button in a thread-safe manner
                        try:
                            if apply_button is not None:
                                try:
                                    with client.atomic():
                                        apply_button.disabled = False
                                except Exception:
                                    apply_button.disabled = False
                        except Exception:
                            pass
                    else:
                        # keep last_refined_ply unset if file not present locally
                        print(f"Refined PLY reported but not found locally: {refined} (out_dir={out_dir})")
                except Exception as exc:  # noqa: BLE001
                    print(f"Error resolving refined ply path: {exc}")
            if resolved_refined:
                self._refresh_selection_ui()
            self.live_preview_active = False
            self.live_last_path = None

            client.add_notification(
                "SV4D inpaint finished",
                f"Frame {self.current_frame} output: {out_dir}",
                color="green",
                auto_close_seconds=8,
            )

        threading.Thread(target=_execute, daemon=True).start()

    def _apply_refined_gs(
        self,
        client: viser.ClientHandle,
        status_handle: viser.GuiTextHandle,
        undo_button: viser.GuiButtonHandle,
    ) -> None:
        if self.last_refined_ply is None or not Path(self.last_refined_ply).exists():
            client.add_notification(
                "Apply skipped",
                "No refined GS available (run Fix first).",
                color="orange",
                auto_close_seconds=8,
            )
            return
        frame_idx = self.current_frame
        target_path = self.sequence.gs_path(frame_idx)
        if not target_path.exists():
            client.add_notification(
                "Apply failed",
                f"GS path missing: {target_path}",
                color="red",
                with_close_button=True,
                auto_close_seconds=10,
            )
            return
        # Save backup once per frame so we can undo.
        if frame_idx not in self.applied_backups:
            backup_dir = self.workspace.results_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{target_path.stem}_orig.ply"
            try:
                shutil.copy2(target_path, backup_path)
                self.applied_backups[frame_idx] = backup_path
            except Exception as exc:  # noqa: BLE001
                client.add_notification(
                    "Apply warning",
                    f"Backup failed: {exc}",
                    color="orange",
                    auto_close_seconds=10,
                )
        try:
            shutil.copy2(self.last_refined_ply, target_path)
            pass
        except Exception as exc:  # noqa: BLE001
            client.add_notification(
                "Apply failed",
                f"Could not copy refined GS: {exc}",
                color="red",
                with_close_button=True,
                auto_close_seconds=10,
            )
            return
        # Drop cached splat data and reload.
        if frame_idx in self.sequence.cache:
            self.sequence.cache.pop(frame_idx)
        self._apply_frame(frame_idx, self.preview_handles.get(client))
        status_handle.value = f"Applied refined GS to frame {frame_idx}."
        undo_button.disabled = self.applied_backups.get(frame_idx) is None
        self.live_preview_active = False
        self.live_last_path = None
        self._disable_live_buttons(True)
        client.add_notification(
            "Apply success",
            f"Refined GS applied to {target_path.name}.",
            color="green",
            auto_close_seconds=6,
        )
        self._refresh_selection_ui()

    def _undo_apply_refined(
        self,
        client: viser.ClientHandle,
        status_handle: viser.GuiTextHandle,
        undo_button: viser.GuiButtonHandle,
    ) -> None:
        frame_idx = self.current_frame
        backup_path = self.applied_backups.get(frame_idx)
        if backup_path is None or not Path(backup_path).exists():
            client.add_notification(
                "Undo skipped",
                "No backup available for this frame.",
                color="orange",
                auto_close_seconds=8,
            )
            return
        target_path = self.sequence.gs_path(frame_idx)
        try:
            shutil.copy2(backup_path, target_path)
        except Exception as exc:  # noqa: BLE001
            client.add_notification(
                "Undo failed",
                f"Could not restore backup: {exc}",
                color="red",
                with_close_button=True,
                auto_close_seconds=10,
            )
            return
        if frame_idx in self.sequence.cache:
            self.sequence.cache.pop(frame_idx)
        self._apply_frame(frame_idx, self.preview_handles.get(client))
        status_handle.value = f"Restored original GS for frame {frame_idx}."
        undo_button.disabled = False
        self.live_preview_active = False
        self.live_last_path = None
        self.live_preview_blocked = False
        self.live_preview_hidden = False
        self._disable_live_buttons(True)
        client.add_notification(
            "Undo success",
            f"Restored original GS for {target_path.name}.",
            color="green",
            auto_close_seconds=6,
        )
        self._refresh_selection_ui()

    def _run_fix(
        self,
        client: viser.ClientHandle,
        status_handle: viser.GuiMarkdownHandle,
        image_handles: List[viser.GuiImageHandle],
        apply_button: viser.GuiButtonHandle | None = None,
        cancel_button: viser.GuiButtonHandle | None = None,
    ) -> None:
        if not self.selection_info or not self.last_roi_payload:
            client.add_notification(
                "Fix skipped",
                "No ROI selection to fix.",
                color="orange",
                auto_close_seconds=10,
            )
            return
        fix_url = self._fix_endpoint_url()
        if not fix_url:
            client.add_notification(
                "Fix skipped",
                "Backend URL not set.",
                color="orange",
                auto_close_seconds=10,
            )
            return

        payload = json.loads(json.dumps(self.last_roi_payload))
        payload.setdefault("selection", {})
        payload["selection"]["gs_indices"] = self.selection_info.get("indices", [])
        payload.setdefault("paths", {})
        payload["paths"]["eval_gt_dir"] = payload["paths"].get(
            "eval_gt_dir", str(self.workspace.images_root / "eval_gt")
        )
        payload["paths"]["workspace_root"] = payload["paths"].get("workspace_root", str(self.workspace.root))
        payload["experiment_id"] = f"{payload.get('experiment_id', 'exp')}_fix"
        payload["video_samples"] = int(self.video_samples)
        payload["render_video_only"] = bool(self.render_video_only)
        payload.setdefault("opt_params", {})
        payload["opt_params"].setdefault("video_samples", int(self.video_samples))
        payload["opt_params"].setdefault("render_video_only", bool(self.render_video_only))
        status_handle.content = "Fix running... (ROI-aware optimization)"
        exp_id_for_progress = payload["experiment_id"]
        progress_stop: threading.Event | None = None
        self.live_preview_blocked = False
        self.live_preview_active = False
        self.live_preview_hidden = False
        self._disable_live_buttons(True)
        self.loss_history.clear()
        self.opt_metric_history.clear()
        self.opt_metric_label = None
        if client in self.loss_handles:
            self.loss_handles[client].image = np.zeros((64, 256, 3), dtype=np.uint8)

        def _start_progress_poll(exp_id: str) -> threading.Event | None:
            progress_url = self._fix_progress_url(fix_url)
            if not progress_url:
                return None
            stop_event = threading.Event()

            def _poll_progress() -> None:
                last_iter_seen = -1
                first_poll = True
                while not stop_event.is_set():
                    try:
                        query_url = f"{progress_url}?experiment_id={exp_id}"
                        req = urllib.request.Request(query_url, headers=client_headers())
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            body = resp.read().decode("utf-8")
                        parsed = json.loads(body)
                        print(f"[Fix progress] {parsed}")
                        status = parsed.get("status")
                        iter_cur = parsed.get("iter")
                        iter_max = parsed.get("iter_max")
                        loss_val = parsed.get("loss")
                        if status == "pending":
                            status_handle.content = "Fix pending... waiting for backend to start"
                        elif status == "running":
                            msg = (
                                f"Fix running... iter {iter_cur}/{iter_max}, loss {loss_val:.4f}"
                                if loss_val is not None
                                else f"Fix running... iter {iter_cur}/{iter_max}"
                            )
                            status_handle.content = msg
                            if loss_val is not None and (iter_cur is None or iter_cur != last_iter_seen):
                                self.loss_history.append(float(loss_val))
                                if iter_cur is not None:
                                    last_iter_seen = int(iter_cur)
                                if self.show_loss_curve:
                                    self._update_loss_curve_image(client)
                        else:
                            status_handle.content = f"Fix status: {status}"
                        render_path = parsed.get("progress_render") or parsed.get("render_path")
                        target_path = parsed.get("progress_target") or parsed.get("oracle_path")
                        live_ply = parsed.get("live_ply")
                        if render_path and Path(render_path).exists():
                            try:
                                image_handles[0].image = ensure_rgb(iio.imread(render_path))
                            except Exception as exc:  # noqa: BLE001
                                print(f"Progress image load failed: {exc}")
                        if len(image_handles) > 1 and target_path and Path(target_path).exists():
                            try:
                                image_handles[1].image = ensure_rgb(iio.imread(target_path))
                            except Exception as exc:  # noqa: BLE001
                                print(f"Target image load failed: {exc}")
                        if live_ply and not self.live_preview_blocked and Path(live_ply).exists():
                            self._preview_live_gs(live_ply, client)
                            if cancel_button is not None:
                                cancel_button.disabled = False
                        if status not in ("pending", "running"):
                            break
                    except Exception as exc:  # noqa: BLE001
                        print(f"Fix progress poll failed: {exc}")
                    stop_event.wait(1.0 if not first_poll else 0.5)
                    first_poll = False

            threading.Thread(target=_poll_progress, daemon=True).start()
            return stop_event

        progress_stop = _start_progress_poll(exp_id_for_progress)

        def _execute_fix_request() -> None:
            nonlocal progress_stop, exp_id_for_progress
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                fix_url,
                data=data,
                headers=client_headers(include_json=True),
                method="POST",
            )
            try:
                # Allow long-running ROI fixes; backend can easily exceed 60s.
                with urllib.request.urlopen(req, timeout=300) as resp:
                    body = resp.read().decode("utf-8")
            except urllib.error.URLError as exc:
                status_handle.content = f"Fix error: {exc}"
                if progress_stop:
                    progress_stop.set()
                client.add_notification(
                    "Fix error",
                    f"{exc.__class__.__name__}: {exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=10,
                )
                return
            except Exception as exc:  # noqa: BLE001
                status_handle.content = f"Fix failed: {exc}"
                if progress_stop:
                    progress_stop.set()
                client.add_notification(
                    "Fix error",
                    f"{exc.__class__.__name__}: {exc}",
                    color="red",
                    with_close_button=True,
                    auto_close_seconds=10,
                )
                return

            if progress_stop:
                progress_stop.set()

            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None

            if not parsed or parsed.get("status") != "success":
                client.add_notification(
                    "Fix response",
                    body[:240] + ("..." if len(body) > 240 else ""),
                    color="orange",
                    auto_close_seconds=10,
                )
                status_handle.content = "Fix failed: backend returned error."
                return

            parsed_exp_id = parsed.get("experiment_id")
            if parsed_exp_id and parsed_exp_id != exp_id_for_progress:
                if progress_stop:
                    progress_stop.set()
                exp_id_for_progress = parsed_exp_id
                progress_stop = _start_progress_poll(exp_id_for_progress)

            refined = parsed.get("refined_ply")
            loss_curve = parsed.get("loss_curve", [])
            time_sec = parsed.get("time_sec")
            metrics = parsed.get("metrics") or {}
            if loss_curve:
                try:
                    self.loss_history = [float(x) for x in loss_curve]
                except Exception:
                    self.loss_history = []
                if self.show_loss_curve:
                    self._update_loss_curve_image(client)
            metric_line = None
            if metrics:
                psnr_after = metrics.get("psnr_after")
                psnr_delta = metrics.get("psnr_delta")
                lpips_after = metrics.get("lpips_after")
                lpips_delta = metrics.get("lpips_delta")
                metric_parts: List[str] = []
                if psnr_after is not None:
                    part = f"PSNR {psnr_after:.3f}"
                    if psnr_delta is not None:
                        part += f" ({psnr_delta:+.3f})"
                    metric_parts.append(part)
                if lpips_after is not None:
                    part = f"LPIPS {lpips_after:.4f}"
                    if lpips_delta is not None:
                        part += f" ({lpips_delta:+.4f})"
                    metric_parts.append(part)
                if metric_parts:
                    metric_line = "Metrics: " + ", ".join(metric_parts)
            desc_lines = [
                f"Refined PLY: {refined}" if refined else "Refined PLY: (none)",
                f"Iters: {len(loss_curve)} | Final loss: {loss_curve[-1]:.4f}" if loss_curve else "No loss curve returned.",
                f"Time: {time_sec:.2f}s" if time_sec is not None else "",
            ]
            if metric_line:
                desc_lines.append(metric_line)
            status_handle.content = "Fix result:\n\n" + "\n".join([d for d in desc_lines if d])

            outputs = parsed.get("outputs", {}) or {}
            render_before = outputs.get("render_before")
            render_after = outputs.get("render_after")
            oracle_patch = outputs.get("oracle_patch")
            progress_render = outputs.get("progress_render")
            progress_target = outputs.get("progress_target")

            # Final display: Fix Before, Fix End, Fix Target
            final_images = [render_before, render_after, oracle_patch]
            if progress_render and Path(progress_render).exists():
                # Update the progress handle once more in case the final render is absent.
                try:
                    image_handles[0].image = ensure_rgb(iio.imread(progress_render))
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to load final progress render {progress_render}: {exc}")
            if len(image_handles) > 1 and progress_target and Path(progress_target).exists():
                try:
                    image_handles[1].image = ensure_rgb(iio.imread(progress_target))
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to load final progress target {progress_target}: {exc}")

            for h, img_path, label in zip(
                image_handles,
                final_images,
                ["Fix Before View", "Fix End View", "Fix Target View"],
            ):
                if hasattr(h, "label"):
                    try:
                        h.label = label
                    except Exception:
                        pass
                if img_path and Path(img_path).exists():
                    try:
                        h.image = ensure_rgb(iio.imread(img_path))
                    except Exception as exc:  # noqa: BLE001
                        print(f"Failed to load fix output image {img_path}: {exc}")
            if refined:
                self.last_refined_ply = Path(refined)
                if apply_button is not None:
                    apply_button.disabled = False
                self._refresh_selection_ui()
            if progress_stop:
                progress_stop.set()

        threading.Thread(target=_execute_fix_request, daemon=True).start()

    def run(self) -> None:
        print(
            f"Workspace created at {self.workspace.root}. "
            f"Open http://{getattr(self, 'server_host', '127.0.0.1')}:{self.server_port} "
            "to start using the viewer."
        )
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()

    def _choose_capture_size(self, client: viser.ClientHandle) -> Tuple[int, int]:
        # Force square captures so ROI coords align with 512x512 GT.
        width = height = int(self.image_size)
        return width, height

    def _capture_clean_render(
        self, client: viser.ClientHandle, width: int, height: int
    ) -> np.ndarray:
        """Temporarily hide helper visuals (world axes) to keep captures clean."""
        # Hide both server- and client-side world axes so captured images stay clean.
        server_axes = self.server.scene.world_axes
        client_axes = client.scene.world_axes
        prev_server = server_axes.visible
        prev_client = client_axes.visible
        server_axes.visible = False
        client_axes.visible = False
        client.flush()
        # Give the client time to apply visibility before capturing.
        time.sleep(0.05)
        try:
            image = ensure_rgb(client.get_render(height=height, width=width))
        finally:
            server_axes.visible = prev_server
            client_axes.visible = prev_client
            client.flush()
            time.sleep(0.01)
        return image

    def _compute_mask_from_pose(
        self,
        centers: np.ndarray,
        eye: np.ndarray,
        wxyz: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        rotation_wc = tf.SO3(np.asarray(wxyz, dtype=np.float32)).as_matrix()
        rotation_cw = rotation_wc.T
        translated = np.asarray(centers, dtype=np.float32) - np.asarray(eye, dtype=np.float32)[None, :]
        points_cam = translated @ rotation_cw.T
        z = points_cam[:, 2]
        fx = width / (2.0 * np.tan(np.deg2rad(self.fov_deg) * 0.5))
        fy = height / (2.0 * np.tan(np.deg2rad(self.fov_deg) * 0.5))
        cx = width * 0.5
        cy = height * 0.5
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            x = points_cam[:, 0] / np.maximum(z, 1e-6)
            y = points_cam[:, 1] / np.maximum(z, 1e-6)
            u = fx * x + cx
            v = fy * y + cy
        depth = np.full((height, width), np.inf, dtype=np.float32)
        valid = (z > 0) & np.isfinite(z)
        if np.any(valid):
            ui = np.clip(np.round(u[valid]).astype(np.int64), 0, width - 1)
            vi = np.clip(np.round(v[valid]).astype(np.int64), 0, height - 1)
            flat = vi * width + ui
            depth_flat = depth.ravel()
            np.minimum.at(depth_flat, flat, z[valid].astype(np.float32))
            depth = depth_flat.reshape(height, width)
        mask = np.isfinite(depth) & (depth > 0)
        return (mask.astype(np.uint8) * 255)

    def _render_multiview_range(
        self,
        client: viser.ClientHandle,
        start_idx: int,
        end_idx: int,
        status_handle: viser.GuiMarkdownHandle | None,
        trigger_button: viser.GuiButtonHandle | None,
        *,
        azimuths: Sequence[float] | None = None,
        notify: bool = True,
        purge_existing: bool = False,
    ) -> None:
        width = height = int(self.image_size)
        total_frames = len(self.sequence.ply_paths)
        if total_frames == 0:
            if status_handle is not None:
                status_handle.content = "Multi-view skipped: no frames available."
            if trigger_button is not None:
                trigger_button.disabled = self.backend_url is None
            return

        if self.backend_url:
            self._ensure_session(client)
        session_root = self.session_root
        if session_root is None:
            if status_handle is not None:
                status_handle.content = "Multi-view failed: start a session first (Generate)."
            if trigger_button is not None:
                trigger_button.disabled = self.backend_url is None
            return
        rgb_dir = session_root / "multi_view"
        mask_dir = session_root / "multi_view_mask"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        start = max(0, min(int(start_idx), total_frames - 1))
        end = max(start, min(int(end_idx), total_frames - 1))
        azs = [float(x) for x in (list(azimuths) if azimuths is not None else [0, 60, 120, 180, 240])]

        # Prefer backend gsplat rendering when a backend is configured and a session is active.
        if self.backend_url and self.session_id:
            payload = {
                "session_id": self.session_id,
                "start_frame": start,
                "end_frame": end,
                "azimuths": azs,
                "elevation_deg": 0.0,
                "distance": 1.5,
                "image_size": width,
                "fov_deg": float(self.fov_deg),
                "up_axis": self.up_axis,
            }
            try:
                base = self._backend_base_url()
                resp = self._post_json(base + "/render/multiview", payload, timeout=300)
                if resp.get("status") != "success":
                    raise RuntimeError(resp.get("message") or "backend returned error")
                saved = resp.get("saved", [])
                rgb_dir_backend = resp.get("rgb_dir", str(rgb_dir))
                mask_dir_backend = resp.get("mask_dir", str(mask_dir))
                if status_handle is not None:
                    status_handle.content = (
                        f"Multi-view done (backend): frames {start}-{end} -> {rgb_dir_backend} / {mask_dir_backend}."
                    )
                if notify:
                    client.add_notification(
                        "Multi-view export (backend)",
                        f"Saved {len(saved)} renders to {rgb_dir_backend} and {mask_dir_backend}.",
                        color="green",
                        auto_close_seconds=8,
                    )
                if trigger_button is not None:
                    trigger_button.disabled = self.backend_url is None
                return
            except Exception as exc:  # noqa: BLE001
                if status_handle is not None:
                    status_handle.content = f"Backend multiview failed, falling back to client capture: {exc}"

        prev_frame = self.current_frame
        prev_view = self.current_view
        prev_cam = {
            "position": client.camera.position.copy(),
            "wxyz": client.camera.wxyz.copy(),
            "look_at": client.camera.look_at.copy(),
            "up": client.camera.up_direction.copy(),
        }
        prev_overlay = self.selection_overlay_enabled
        prev_frustum_vis = [(h, getattr(h, "visible", True)) for h in self.frustum_handles]

        try:
            # Hide overlays/frustums during capture for clean outputs.
            self.selection_overlay_enabled = False
            self._render_selection_overlay(False)
            for h, _ in prev_frustum_vis:
                try:
                    h.visible = False
                except Exception:
                    pass

            for frame_idx in range(start, end + 1):
                if frame_idx >= total_frames:
                    break
                try:
                    data = self.sequence.load(frame_idx)
                except FileNotFoundError:
                    if status_handle is not None:
                        status_handle.content = f"Skip frame {frame_idx}: GS missing."
                    continue

                self.current_frame = frame_idx
                try:
                    self._update_gaussian_handle(data)
                except Exception as exc:  # noqa: BLE001
                    print(f"Multi-view GS update failed for frame {frame_idx}: {exc}")

                frame_token = self.sequence.frame_tokens[frame_idx]
                if purge_existing:
                    for old in rgb_dir.glob(f"{frame_token}_az*.png"):
                        try:
                            old.unlink()
                        except Exception:
                            pass
                    for old in mask_dir.glob(f"{frame_token}_az*.png"):
                        try:
                            old.unlink()
                        except Exception:
                            pass
                for az in azs:
                    eye, wxyz = self._eye_wxyz_from_angles(float(az), 0.0, 1.5)
                    self._set_client_camera_pose(client, eye, wxyz)
                    client.flush()
                    render = self._capture_clean_render(client, width, height)
                    rgb_path = rgb_dir / f"{frame_token}_az{int(az):03d}.png"
                    iio.imwrite(rgb_path, render)
                    mask = self._compute_mask_from_pose(data.centers, eye, wxyz, width, height)
                    mask_path = mask_dir / f"{frame_token}_az{int(az):03d}.png"
                    iio.imwrite(mask_path, mask)
                if status_handle is not None:
                    status_handle.content = (
                        f"Multi-view progress: frame {frame_idx}/{end} saved to {rgb_dir.name}."
                    )

            if status_handle is not None:
                status_handle.content = (
                    f"Multi-view done: frames {start}-{end} -> {rgb_dir.name} / {mask_dir.name}."
                )
            if notify:
                client.add_notification(
                    "Multi-view export",
                    f"Saved frames {start}-{end} to {rgb_dir} and {mask_dir}.",
                    color="green",
                    auto_close_seconds=8,
                )
        finally:
            self.current_frame = prev_frame
            self.current_view = prev_view
            if prev_overlay:
                self.selection_overlay_enabled = True
                self._render_selection_overlay(True)
            for h, vis in prev_frustum_vis:
                try:
                    h.visible = vis
                except Exception:
                    pass
            try:
                self._set_client_camera_pose(
                    client,
                    prev_cam["position"],
                    prev_cam["wxyz"],
                    look_at=prev_cam["look_at"],
                    up=prev_cam["up"],
                )
                client.flush()
            except Exception:
                pass
            try:
                preview_handle = self.preview_handles.get(client)
                if preview_handle is not None:
                    self._apply_frame(prev_frame, preview_handle)
            except Exception:
                pass
            if trigger_button is not None:
                trigger_button.disabled = self.backend_url is None

    def _build_frustums_for_frame(
        self, frame_index: int, client: viser.ClientHandle | None = None
    ) -> List[viser.SceneNodeHandle]:
        handles: List[viser.SceneNodeHandle] = []
        frame_token = self.sequence.frame_tokens[frame_index]
        data = self.sequence.load(frame_index)
        centers = data.centers
        # Precompute depth for each view to ensure consistent camera-target distance.
        for view_name in self.view_presets.keys():
            img_path = self.workspace.image_path_for_view(frame_token, view_name)
            if not img_path.exists():
                continue
            image = load_image(img_path, flip_udlr=True)
            H, W = image.shape[:2]
            aspect = W / float(H)
            depth = 1.5
            eye, wxyz = self._eye_wxyz_for_view(view_name, depth)
            image_rgba = add_alpha_channel(image, self.frustum_opacity)
            frustum_variant = "wireframe" if self.frustum_wireframe else "filled"
            image_arg = None if self.frustum_wireframe else image_rgba
            try:
                frustum = self.server.scene.add_camera_frustum(
                    f"/frustums/{frame_token}/{view_name}",
                    fov=np.deg2rad(self.fov_deg),
                    aspect=aspect,
                    scale=max(depth * 0.25, 0.1),
                    image=image_arg,
                    wxyz=tuple(wxyz.tolist()),
                    position=tuple(eye.tolist()),
                    variant=frustum_variant,
                    cast_shadow=False,
                    receive_shadow=False,
                )
                if client is not None:
                    eye_local = eye.copy()
                    wxyz_local = wxyz.copy()

                    def _on_click(ev: viser.SceneNodePointerEvent, eye=eye_local, wxyz=wxyz_local) -> None:
                        self._set_client_camera_pose(ev.client, eye=eye, wxyz=wxyz)

                    frustum.on_click(_on_click)
                handles.append(frustum)
            except Exception as exc:
                print(f"Failed to add frustum for {view_name}: {exc}")
                continue
        return handles

    def _compute_view_depth(self, frame_index: int, view: str, centers: np.ndarray) -> float | None:
        frame_token = self.sequence.frame_tokens[frame_index]
        img_path = self.workspace.image_path_for_view(frame_token, view)
        if not img_path.exists():
            return None
        image = load_image(img_path, flip_udlr=self.flip_colmap)
        H, W = image.shape[:2]
        _, wxyz = self._eye_wxyz_for_view(view, 1.0)
        rotation_wc = tf.SO3(np.asarray(wxyz, dtype=np.float32)).as_matrix()
        bbox_gt = image_bbox_from_mask(image)
        fx = W / (2 * np.tan(np.deg2rad(self.fov_deg) * 0.5))
        fy = H / (2 * np.tan(np.deg2rad(self.fov_deg) * 0.5))
        target = self.camera_target
        depth = solve_depth_for_bbox(centers, rotation_wc, target, bbox_gt, fx, fy)
        if depth <= 0:
            return None

        # refine depth using projected bbox size
        direction = self._direction_for_view(view)
        for _ in range(3):
            eye = target + direction * depth
            bw, bh = project_bbox_size(centers, rotation_wc, eye, self.fov_deg, W, H)
            if bw <= 1e-3 or bh <= 1e-3:
                break
            gt_w = max(bbox_gt[2] - bbox_gt[0], 1e-3)
            gt_h = max(bbox_gt[3] - bbox_gt[1], 1e-3)
            scale_w = bw / gt_w
            scale_h = bh / gt_h
            scale = max(scale_w, scale_h)
            if abs(scale - 1.0) < 1e-3:
                break
            depth *= scale
        return float(depth)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Viser frontend for GS timeline + ROI selection"
    )
    parser.add_argument(
        "--gs-dir",
        type=Path,
        default=ROOT / "asset/demo/4dgs_ls/0354",
        help="Directory containing a sequence of GS PLY files.",
    )
    parser.add_argument(
        "--input-image-dir",
        type=Path,
        default=ROOT / "asset/demo/image/input/0013",
        help="Directory with input view images (used for the timeline strip).",
    )
    parser.add_argument(
        "--eval-gt-dir",
        type=Path,
        default=ROOT / "asset/demo/image/eval_gt/0205_vault_open",
        help="Root directory with eval_gt views (eval_0...eval_3).",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=ROOT / "workspace",
        help="Root folder to store staged GS, images, selections, and outputs.",
    )
    parser.add_argument(
        "--resume-workspace",
        type=Path,
        default=None,
        help="Use an existing workspace directory (e.g., workspace_YYYYMMDD_HHMMSS) for debugging.",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=None,
        help="Optional HTTP endpoint to POST ROI payloads to (e.g. http://localhost:8000/optimize_roi).",
    )
    parser.add_argument(
        "--sam3d-backend-url",
        type=str,
        default=None,
        help="Optional HTTP endpoint for SAM/SAM3D plugin backend (e.g. http://localhost:8015).",
    )
    parser.add_argument(
        "--dispatcher-url",
        type=str,
        default=None,
        help="Optional unified dispatcher endpoint (e.g. http://localhost:8100). If set, ROI+SAM3D calls are routed through dispatcher.",
    )
    parser.add_argument(
        "--vlm-agent-url",
        type=str,
        default=None,
        help="Optional HTTP endpoint for VLM/LLM agent backend (e.g. http://localhost:8020).",
    )
    parser.add_argument(
        "--vlm-shared-dir",
        type=str,
        default=None,
        help="Optional shared directory for VLM_toolkit_cross_gpfs (same path used by toolkit server/client).",
    )
    parser.add_argument(
        "--mvtap-url",
        type=str,
        default=None,
        help="Optional MV-TAP tracklet endpoint (e.g. http://localhost:8010/track_roi).",
    )
    parser.add_argument(
        "--http-get-retries",
        type=int,
        default=2,
        help="Retry count for GET requests to backend/dispatcher.",
    )
    parser.add_argument(
        "--http-post-retries",
        type=int,
        default=0,
        help="Retry count for POST requests to backend/dispatcher (default 0 to avoid duplicate side effects).",
    )
    parser.add_argument(
        "--http-retry-backoff-ms",
        type=int,
        default=400,
        help="Initial retry backoff in ms; subsequent retries use exponential backoff.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image resolution for captures and bbox coordinates.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=49.1,
        help="Vertical field of view used for the camera presets.",
    )
    parser.add_argument(
        "--up-axis",
        type=str,
        choices=["y", "z", "-y"],
        default="z",
        help="World up axis. Use 'z' for Blender-style right-handed (+Z up).",
    )
    parser.add_argument(
        "--flip-colmap",
        action="store_true",
        help="Flip COLMAP images (ud+lr) when building frustums for alignment.",
    )
    parser.add_argument(
        "--camera-target-origin",
        action="store_true",
        help="If set, camera look_at/target is world origin instead of GS center.",
    )
    parser.add_argument(
        "--start-empty",
        action="store_true",
        help="Start with an empty scene (do not load existing GS on startup).",
    )
    parser.add_argument(
        "--metric-online",
        "--online-metric",
        "--online_metric",
        dest="metric_online",
        action="store_true",
        help="Compute PSNR/LPIPS against eval_gt after each generate step and display in UI.",
    )
    parser.add_argument(
        "--no-flash-atten",
        "--no-flash-attn",
        dest="no_flash_atten",
        action="store_true",
        help="Force SAM3D to avoid flash-attn (use sdpa) even on H100/H200/A100.",
    )
    parser.add_argument(
        "--one-dino",
        dest="one_dino",
        action="store_true",
        help="Make SAM3D share a single DINO backbone instance across all Dino() modules (process-wide).",
    )
    parser.add_argument(
        "--gs-display-max-points",
        type=int,
        default=600_000,
        help="Max gaussians to send to the browser for visualization per frame (0=disable downsample).",
    )
    parser.add_argument(
        "--gs-display-sample",
        type=str,
        choices=["opacity", "random"],
        default="opacity",
        help="Downsample strategy when a frame exceeds --gs-display-max-points.",
    )
    parser.add_argument(
        "--gs-display-fixed-points",
        type=int,
        default=0,
        help=(
            "If >0, always send exactly this many gaussians to the browser (pad with invisible ones). "
            "Helps stability during playback when frame sizes vary."
        ),
    )
    parser.add_argument(
        "--selection-overlay-fixed-points",
        type=int,
        default=50_000,
        help=(
            "Fixed number of gaussians to send for the ROI selection overlay (pad with invisible ones). "
            "Use 0 to disable fixing; large values can stall the UI on big selections."
        ),
    )
    parser.add_argument(
        "--frame-scrub-debounce-ms",
        type=int,
        default=120,
        help="Debounce for timeline scrubbing to avoid repeated heavy GS loads.",
    )
    parser.add_argument(
        "--raster-preview-max-points",
        type=int,
        default=200_000,
        help="Disable raster preview if a frame has more than this many points (0=never disable).",
    )
    parser.add_argument(
        "--no-raster-preview",
        action="store_true",
        help="Disable gsplat-based raster preview (saves GPU/CPU).",
    )
    parser.add_argument(
        "--profile-frame-load",
        action="store_true",
        help="Print per-frame timing breakdowns for diagnosing slow frame switches.",
    )
    parser.add_argument(
        "--profile-frame-threshold-ms",
        type=float,
        default=150.0,
        help="Only print profile lines slower than this threshold (ms).",
    )
    parser.add_argument(
        "--use_pose_adjust_v2",
        "--use-pose-adjust-v2",
        dest="use_pose_adjust_v2",
        action="store_true",
        help="Use Agent2 V2 (Direct Shift + PoseOpt) inside Full 3D/4D generation pipelines.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Viser server host (default: loopback only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8897,
        help="Viser server port (default: 8897).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not is_loopback_host(args.host):
        raise ValueError(
            "The Viser frontend has no built-in authentication and must bind to a loopback address"
        )
    if bool(getattr(args, "no_flash_atten", False)):
        os.environ["SAM3D_NO_FLASH_ATTN"] = "1"
        # Explicitly select PyTorch SDPA attention backends for SAM3D.
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    if bool(getattr(args, "one_dino", False)):
        os.environ["SAM3D_ONE_DINO"] = "1"
    app = ViewerApp(
        gs_dir=args.gs_dir,
        input_dir=args.input_image_dir,
        eval_dir=args.eval_gt_dir,
        workspace_root=args.workspace_root,
        resume_workspace=args.resume_workspace,
        backend_url=args.backend_url,
        sam3d_backend_url=args.sam3d_backend_url,
        dispatcher_url=args.dispatcher_url,
        vlm_agent_url=args.vlm_agent_url,
        vlm_shared_dir=args.vlm_shared_dir,
        mvtap_url=args.mvtap_url,
        image_size=args.image_size,
        fov_deg=args.fov_deg,
        up_axis=args.up_axis,
        flip_colmap=args.flip_colmap,
        camera_target_origin=args.camera_target_origin,
        start_empty=args.start_empty,
        metric_online=args.metric_online,
        no_flash_atten=args.no_flash_atten,
        gs_display_max_points=args.gs_display_max_points,
        gs_display_sample=args.gs_display_sample,
        gs_display_fixed_points=args.gs_display_fixed_points,
        selection_overlay_fixed_points=args.selection_overlay_fixed_points,
        frame_scrub_debounce_ms=args.frame_scrub_debounce_ms,
        raster_preview_max_points=args.raster_preview_max_points,
        raster_preview_enabled=not bool(args.no_raster_preview),
        profile_frame_load=bool(args.profile_frame_load),
        profile_frame_threshold_ms=float(args.profile_frame_threshold_ms),
        use_pose_adjust_v2=bool(getattr(args, "use_pose_adjust_v2", False)),
        server_host=args.host,
        server_port=args.port,
        http_get_retries=int(getattr(args, "http_get_retries", 2)),
        http_post_retries=int(getattr(args, "http_post_retries", 0)),
        http_retry_backoff_ms=int(getattr(args, "http_retry_backoff_ms", 400)),
    )
    app.run()


if __name__ == "__main__":
    main()
