#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
import time
import shutil
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from good_optimizer import FixOptimizer
    from metric.evaluator import L4GMEvaluator
    from model4dgen.sv4d_inpaint_backend import SV4DInpaintingBackend

ROOT = Path(__file__).resolve().parent
VISER_SRC = ROOT / "thirdparty" / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.append(str(VISER_SRC))

import viser.transforms as tf  # noqa: E402

SESSION_GS_DIRNAME = "gs_sequence"
LEGACY_SESSION_GS_DIRNAME = "gs"

from utils.constants import COORD_ALIGN_MAT, SH_C0
from utils.geometry_utils import compute_center_radius, sanitize_points, transform_gaussians
from utils.image_utils import ensure_rgb, load_image
from utils.ply_utils import read_ply_vertices, write_ply_vertices
from utils.ply_utils import load_splat_from_ply, splatdata_to_backend_dict
from utils.camera_utils import direction_from_angles, eye_wxyz_from_angles
from utils.ply_utils import read_ply_vertices as read_ply_vertices_simple
from http_security import (
    DEFAULT_MAX_REQUEST_BYTES,
    RequestError,
    parse_allowed_roots,
    read_json_body,
    request_is_authorized,
    require_secure_bind,
    validate_identifier,
    validate_payload_paths,
)


try:
    from gsplat.rendering import rasterization as gsplat_rasterization
except Exception:  # noqa: BLE001
    gsplat_rasterization = None

VIEW_PRESETS = {
    "input": (0.0, 0.0),
    "eval_0": (-75.0, 0.0),
    "eval_1": (15.0, 0.0),
    "eval_2": (105.0, 0.0),
    "eval_3": (195.0, 0.0),
}

def draw_bbox_outline(
    image: np.ndarray, bbox: Dict[str, int], color: Tuple[int, int, int] = (255, 200, 0), thickness: int = 3
) -> np.ndarray:
    img = ensure_rgb(image)
    h, w = img.shape[:2]
    x0, x1 = int(bbox["x0"]), int(bbox["x1"])
    y0, y1 = int(bbox["y0"]), int(bbox["y1"])
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    img = img.copy()
    img[y0:y0 + thickness, x0:x1] = color
    img[max(y1 - thickness, 0):y1, x0:x1] = color
    img[y0:y1, x0:x0 + thickness] = color
    img[y0:y1, max(x1 - thickness, 0):x1] = color
    return img


def load_gaussian_centers(ply_path: Path) -> np.ndarray:
    """Return aligned Gaussian centers from a ply file."""
    _, vertices = read_ply_vertices(ply_path)
    centers = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]], axis=1
    ).astype(np.float32)
    centers = sanitize_points(centers)
    # Keep alignment consistent with viewer load (COORD_ALIGN_MAT).
    centers = (centers @ COORD_ALIGN_MAT.T).astype(np.float32)
    return centers


def camera_to_world_rotation(wxyz: List[float]) -> np.ndarray:
    return tf.SO3(np.asarray(wxyz, dtype=np.float32)).as_matrix()


def world_to_camera_points(
    centers: np.ndarray, rotation_wc: np.ndarray, position_w: np.ndarray
) -> np.ndarray:
    rotation_cw = sanitize_points(rotation_wc.T)
    translated = sanitize_points(centers) - sanitize_points(position_w[None, :])
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return translated @ rotation_cw.T


def project_points(
    points_cam: np.ndarray, width: int, height: int, fov_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-space points to pixel coordinates."""
    z = points_cam[:, 2]
    fx = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    fy = height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    cx = width * 0.5
    cy = height * 0.5

    x = points_cam[:, 0] / np.maximum(z, 1e-6)
    y = points_cam[:, 1] / np.maximum(z, 1e-6)
    u = fx * x + cx
    v = fy * y + cy
    return u, v, z


def build_depth_map(
    u: np.ndarray, v: np.ndarray, z: np.ndarray, width: int, height: int
) -> np.ndarray:
    depth = np.full((height, width), np.inf, dtype=np.float32)
    valid = (z > 0) & np.isfinite(z)
    if not np.any(valid):
        return depth
    ui = np.clip(np.round(u[valid]).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(v[valid]).astype(np.int64), 0, height - 1)
    flat = vi * width + ui
    depth_flat = depth.ravel()
    np.minimum.at(depth_flat, flat, z[valid].astype(np.float32))
    return depth_flat.reshape(height, width)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    d_min, d_max = float(finite.min()), float(finite.max())
    if math.isclose(d_min, d_max):
        return np.ones_like(depth, dtype=np.float32)
    out = depth.copy()
    out[~np.isfinite(out)] = d_max
    out = (out - d_min) / (d_max - d_min + 1e-8)
    return np.clip(out, 0.0, 1.0)


def draw_selection_overlay(
    width: int,
    height: int,
    bbox: Dict[str, int],
    u: np.ndarray,
    v: np.ndarray,
    selected_mask: np.ndarray,
    output_path: Path,
) -> None:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ui = np.round(u[valid]).astype(np.int64)
    vi = np.round(v[valid]).astype(np.int64)
    canvas[vi, ui] = np.array([120, 120, 120], dtype=np.uint8)

    sel = valid.copy()
    sel[valid] &= selected_mask
    if np.any(sel):
        ui_sel = np.round(u[sel]).astype(np.int64)
        vi_sel = np.round(v[sel]).astype(np.int64)
        canvas[vi_sel, ui_sel] = np.array([20, 220, 60], dtype=np.uint8)

    x0, x1 = int(bbox["x0"]), int(bbox["x1"])
    y0, y1 = int(bbox["y0"]), int(bbox["y1"])
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height))
    canvas[y0:y0 + 2, x0:x1] = np.array([240, 60, 60], dtype=np.uint8)
    canvas[max(y1 - 2, 0):y1, x0:x1] = np.array([240, 60, 60], dtype=np.uint8)
    canvas[y0:y1, x0:x0 + 2] = np.array([240, 60, 60], dtype=np.uint8)
    canvas[y0:y1, max(x1 - 2, 0):x1] = np.array([240, 60, 60], dtype=np.uint8)

    iio.imwrite(output_path, canvas)


def draw_points_on_image(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    path: Path,
    color: Tuple[int, int, int] = (20, 220, 60),
) -> None:
    overlay = image.copy()
    h, w = overlay.shape[:2]
    valid = (z > 0) & np.isfinite(z) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if np.any(valid):
        ui = np.round(u[valid]).astype(np.int64)
        vi = np.round(v[valid]).astype(np.int64)
        ui = np.clip(ui, 0, w - 1)
        vi = np.clip(vi, 0, h - 1)
        overlay[vi, ui] = np.array(color, dtype=np.uint8)
    iio.imwrite(path, overlay)


def view_direction(view: str, up_axis: str = "y") -> np.ndarray:
    az_deg, el_deg = VIEW_PRESETS.get(view, (0.0, 0.0))
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    if up_axis == "y":
        return np.array(
            [math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)],
            dtype=np.float32,
        )
    return np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float32,
    )


def camera_pose_for_view(
    view: str, center: np.ndarray, radius: float, up_axis: str = "y"
) -> Tuple[np.ndarray, np.ndarray]:
    direction = view_direction(view, up_axis)
    distance = 1.5
    eye = center + direction * distance
    up_vec = np.array([0.0, -1.0, 0.0] if up_axis == "y" else [0.0, 0.0, 1.0])
    z = center - eye
    z = z / (np.linalg.norm(z) + 1e-8)
    y = up_vec - np.dot(up_vec, z) * z
    y = y / (np.linalg.norm(y) + 1e-8)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    wxyz = tf.SO3.from_matrix(R).wxyz.astype(np.float32)
    return eye.astype(np.float32), wxyz


def _now_session_id(prefix: str = "workspace") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{prefix}_{ts}"


def _resolve_path_maybe_relative(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p)


def _count_input_frames(input_dir: Path) -> int:
    """Count frame images in a directory.

    MVP assumes `input_dir` is a directory of images (e.g., 000000.png ...).
    """

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"input_dir must be a directory of frames for MVP: {input_dir}")

    image_suffixes = {".png", ".jpg", ".jpeg"}

    # Prefer digit-named frames (000000.png, 000001.png, ...).
    candidates = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_suffixes and p.stem.isdigit()
    ]
    if not candidates:
        # Fallback: count any image files.
        candidates = [
            p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes
        ]
    return len(candidates)


def _list_input_frames(input_dir: Path) -> List[Path]:
    """List input frame images sorted by index/name.

    MVP: accepts a directory of images. Prefer digit-named stems (000000.png).
    """

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    image_suffixes = {".png", ".jpg", ".jpeg"}
    digit_named = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_suffixes and p.stem.isdigit()
    ]
    if digit_named:
        return sorted(digit_named, key=lambda p: int(p.stem))
    generic = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes]
    return sorted(generic, key=lambda p: p.name)


def _write_mock_compatible_gs_ply(path: Path, num_splats: int = 8000, seed: int = 0) -> None:
    """Write a viewer-compatible gaussian PLY.

    This is a minimal fallback generator for Milestone 2 so the end-to-end
    UI plumbing can be validated even when the L4GM weights/pipeline are not
    available in the current environment.
    """

    rng = np.random.default_rng(int(seed))
    n = int(max(1, num_splats))

    centers = rng.normal(loc=0.0, scale=0.4, size=(n, 3)).astype(np.float32)
    sh = rng.uniform(low=-0.6, high=0.6, size=(n, 3)).astype(np.float32)
    opacity_logits = rng.normal(loc=0.0, scale=1.0, size=(n,)).astype(np.float32)
    log_scales = rng.normal(loc=-1.6, scale=0.35, size=(n, 3)).astype(np.float32)

    # Random unit quaternions (wxyz) for rotations.
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q = (q / q_norm).astype(np.float32)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    vertices = np.empty((n,), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = centers[:, 0], centers[:, 1], centers[:, 2]
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = sh[:, 0], sh[:, 1], sh[:, 2]
    vertices["opacity"] = opacity_logits
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]
    vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    header = [
        b"ply\n",
        b"format binary_little_endian 1.0\n",
        f"element vertex {n}\n".encode("ascii"),
        b"property float x\n",
        b"property float y\n",
        b"property float z\n",
        b"property float f_dc_0\n",
        b"property float f_dc_1\n",
        b"property float f_dc_2\n",
        b"property float opacity\n",
        b"property float scale_0\n",
        b"property float scale_1\n",
        b"property float scale_2\n",
        b"property float rot_0\n",
        b"property float rot_1\n",
        b"property float rot_2\n",
        b"property float rot_3\n",
        b"end_header\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    write_ply_vertices(header, vertices, path)


class SessionManager:
    """Minimal session manager for Milestone 1 (session start + frame counting)."""

    def __init__(self, default_workspace_root: Path, l4gm_init_sv4d: bool = False) -> None:
        self.default_workspace_root = default_workspace_root
        self._session_root_by_id: Dict[str, Path] = {}


        self._metric_runner: L4GMEvaluator | None = None
        # Eagerly initialize SV4D inpaint backend so weights are loaded once at startup.
        sv4d_root = ROOT / "model4dgen" / "generative-models"
        sv4d_ckpt = Path(
            os.environ.get(
                "SV4D_CHECKPOINT",
                sv4d_root
                / "checkpoints"
                / "sv4d2.0"
                / "sv4d2_with_first_stage_encoder_v2.safetensors",
            )
        ).expanduser()
        try:
            from model4dgen.sv4d_inpaint_backend import SV4DInpaintingBackend

            self._sv4d_backend = SV4DInpaintingBackend(model_path=sv4d_ckpt)
        except FileNotFoundError as exc:
            print(f"[SessionManager] SV4D checkpoint missing: {exc}")
            self._sv4d_backend = None
        except Exception as exc:  # noqa: BLE001
            print(f"[SessionManager] SV4D init failed: {exc}")
            self._sv4d_backend = None
        self.metric_progress: Dict[str, Dict] = {}
        # Eagerly initialize L4GM so weights are loaded at backend startup.
        l4gm_root = ROOT / "model4dgen" / "L4GM_MoGA"

        if not l4gm_init_sv4d:
            print("Using Init MV GEN Mdel: mv pipeline")
            from model4dgen.l4gm_backend import L4GMGenerator
            self._l4gm: L4GMGenerator | None = L4GMGenerator(l4gm_root )
        else:
            print("Using Init MV GEN Mdel: sv4d 2.0")
            from model4dgen.l4gm_backend_sv4d_init import L4GMGenerator
            self._l4gm: L4GMGenerator | None = L4GMGenerator(l4gm_root, sv4d_backend= self._sv4d_backend )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def start(self, payload: Dict) -> Dict:
        input_dir_str = payload.get("input_dir", "asset/demo/image/input/0013")
        chunk_size = int(payload.get("chunk_size", 8))
        workspace_root_str = payload.get("workspace_root")
        eval_gt_dir_str = payload.get("eval_gt_dir")
        metric_online = bool(payload.get("metric_online", False))

        workspace_root = (
            _resolve_path_maybe_relative(workspace_root_str, ROOT)
            if workspace_root_str
            else self.default_workspace_root
        )
        input_dir = _resolve_path_maybe_relative(str(input_dir_str), ROOT)
        eval_gt_dir = _resolve_path_maybe_relative(str(eval_gt_dir_str), ROOT) if eval_gt_dir_str else None

        num_frames = _count_input_frames(input_dir)

        def _session_id_conflicts(candidate_id: str, candidate_root: Path) -> bool:
            existing = self._session_root_by_id.get(str(candidate_id))
            if existing is None:
                return False
            try:
                return existing.resolve() != candidate_root.resolve()
            except Exception:
                return str(existing) != str(candidate_root)

        def _alloc_session_id_for_root(base_id: str, session_root: Path) -> str:
            if not _session_id_conflicts(base_id, session_root):
                return str(base_id)
            sig = hashlib.sha1(str(session_root.resolve()).encode("utf-8")).hexdigest()[:10]
            rooted = f"{base_id}_{sig}"
            if not _session_id_conflicts(rooted, session_root):
                return rooted
            for i in range(1, 1000):
                candidate = f"{rooted}_{i}"
                if not _session_id_conflicts(candidate, session_root):
                    return candidate
            raise RuntimeError("Failed to allocate a globally unique session_id for workspace")

        use_existing_workspace = bool(payload.get("use_existing_workspace", False))
        if use_existing_workspace:
            session_root = workspace_root
            session_root.mkdir(parents=True, exist_ok=True)
            session_id = _alloc_session_id_for_root(str(session_root.name), session_root)
            gs_dir = session_root / SESSION_GS_DIRNAME
            if gs_dir.exists() and gs_dir.is_symlink():
                # Make session outputs isolated and writable (avoid writing into shared sources).
                gs_dir.unlink()
            gs_dir.mkdir(parents=True, exist_ok=True)
        else:
            session_id_base = _now_session_id("workspace")
            session_id = ""
            session_root = workspace_root / session_id_base

            # Avoid collisions across both filesystem and global in-memory session map.
            for attempt in range(1000):
                candidate = session_id_base if attempt == 0 else f"{session_id_base}_{attempt}"
                candidate_root = workspace_root / candidate
                if _session_id_conflicts(candidate, candidate_root):
                    continue
                try:
                    (candidate_root / SESSION_GS_DIRNAME).mkdir(parents=True, exist_ok=False)
                    session_id = candidate
                    session_root = candidate_root
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError("Failed to allocate a unique session_id")

        self._session_root_by_id[session_id] = session_root

        (session_root / "fix").mkdir(parents=True, exist_ok=True)
        (session_root / "preview").mkdir(parents=True, exist_ok=True)
        # Backwards compatibility: some tools/scripts may still look for session_root/gs.
        try:
            legacy = session_root / LEGACY_SESSION_GS_DIRNAME
            target = session_root / SESSION_GS_DIRNAME
            if not legacy.exists():
                legacy.symlink_to(target.resolve(), target_is_directory=True)
        except Exception:
            pass

        session_json = session_root / "session.json"
        if session_json.exists():
            try:
                session_record = json.loads(session_json.read_text())
            except Exception:
                session_record = {}
        else:
            session_record = {}

        session_record.update(
            {
                "session_id": session_id,
                "created_at": session_record.get("created_at")
                or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "input_dir": str(input_dir),
                "workspace_root": str(workspace_root),
                "eval_gt_dir": str(eval_gt_dir) if eval_gt_dir else None,
                "chunk_size": chunk_size,
                "num_frames": num_frames,
                "state": session_record.get("state", "idle"),
                "current_frame": session_record.get("current_frame", 0),
                "seed_frame": session_record.get("seed_frame"),
                "seed_ply": session_record.get("seed_ply"),
                "generated_until": session_record.get("generated_until", -1),
                "approved_frames": session_record.get("approved_frames", []),
                "fixed_frames": session_record.get("fixed_frames", {}),
                "model": payload.get("model", session_record.get("model", {})),
                "metric_online": metric_online,
                "metrics": session_record.get("metrics", {"frames": [], "psnr": [], "lpips": []}),
                "memory_bank": session_record.get(
                    "memory_bank",
                    {"frame_indices": [], "num_memory_frames": 0, "dir": None, "updated_at": None},
                ),
                "error": None,
            }
        )

        session_json.write_text(json.dumps(session_record, indent=2, ensure_ascii=False))

        return {
            "status": "success",
            "session_id": session_id,
            "num_frames": num_frames,
            "paths": {
                "session_root": str(session_root),
                "session_json": str(session_json),
            },
        }

    def status(self, session_id: str) -> Dict:
        session_id = validate_identifier(session_id, "session_id")
        session_root = self._session_root_by_id.get(session_id) or (self.default_workspace_root / session_id)
        session_json = session_root / "session.json"
        if not session_json.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        record = json.loads(session_json.read_text())
        record["status"] = "success"
        return record

    def _session_root(self, session_id: str) -> Path:
        session_id = validate_identifier(session_id, "session_id")
        return self._session_root_by_id.get(session_id) or (self.default_workspace_root / session_id)

    def _load_record(self, session_id: str) -> Tuple[Path, Dict]:
        session_root = self._session_root(session_id)
        session_json = session_root / "session.json"
        if not session_json.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        record = json.loads(session_json.read_text())
        return session_root, record

    def _save_record(self, session_root: Path, record: Dict) -> None:
        (session_root / "session.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

    def _get_metric_runner(self) -> L4GMEvaluator:
        if self._metric_runner is None:
            from metric.evaluator import L4GMEvaluator

            if self._l4gm is None:
                raise RuntimeError("L4GM is not initialized")
            self._metric_runner = L4GMEvaluator(self._l4gm)
        return self._metric_runner

    def _get_sv4d_backend(self) -> SV4DInpaintingBackend:
        if self._sv4d_backend is None:
            raise RuntimeError("SV4D backend not initialized; check checkpoint path or startup logs")
        return self._sv4d_backend

    def _run_metrics(self, record: Dict, session_root: Path, frame_indices: List[int]) -> Dict:
        if not record.get("metric_online"):
            session_id = str(record.get("session_id") or session_root.name)
            self.metric_progress[session_id] = {
                "status": "disabled",
                "completed": 0,
                "total": len(frame_indices),
                "per_frame": [],
            }
            print(f"metric_progress {session_id} entry:", self.metric_progress[session_id])
            return {"per_frame": [], "psnr_avg": None, "lpips_avg": None}

        eval_root = record.get("eval_gt_dir")
        if not eval_root:
            raise FileNotFoundError("metric_online enabled but eval_gt_dir is missing")
        eval_root_path = Path(eval_root)
        frames = _list_input_frames(Path(record.get("input_dir")))
        runner = self._get_metric_runner()

        session_id = str(record.get("session_id") or session_root.name)
        progress_entry = {
            "status": "running",
            "completed": 0,
            "total": len(frame_indices),
            "per_frame": [],
        }
        self.metric_progress[session_id] = progress_entry

        per_frame = []
        for idx in frame_indices:
            if idx < 0 or idx >= len(frames):
                continue
            token = frames[idx].stem
            ply_path = session_root / SESSION_GS_DIRNAME / f"{idx:06d}.ply"
            if not ply_path.exists():
                continue
            try:
                print("test metric for frame:", ply_path)
                preview_dir = session_root / "preview" / "metric_debug"
                score = runner.score_frame(ply_path, eval_root_path, token, preview_dir)
                per_frame.append(score)
                record.setdefault("metrics", {}).setdefault("frames", []).append(idx)
                record.setdefault("metrics", {}).setdefault("psnr", []).append(score["psnr"])
                record.setdefault("metrics", {}).setdefault("lpips", []).append(score["lpips"])
                progress_entry["completed"] = len(per_frame)
                progress_entry["per_frame"] = per_frame.copy()
                self.metric_progress[session_id] = progress_entry.copy()
                self._save_record(session_root, record)
            except Exception as exc:
                print(f"[metric] failed for frame {idx}: {exc}")
                continue

        psnr_vals = [x["psnr"] for x in per_frame]
        lpips_vals = [x["lpips"] for x in per_frame]
        psnr_avg = float(np.mean(psnr_vals)) if psnr_vals else None
        lpips_avg = float(np.mean(lpips_vals)) if lpips_vals else None
        progress_entry.update(
            {
                "status": "success",
                "psnr_avg": psnr_avg,
                "lpips_avg": lpips_avg,
            }
        )
        self.metric_progress[session_id] = progress_entry
        print(f"metric_progress {session_id} entry:", self.metric_progress[session_id])
        return {"per_frame": per_frame, "psnr_avg": psnr_avg, "lpips_avg": lpips_avg}

    def generate_first(self, payload: Dict) -> Dict:
        """Generate the first GS frame (000000.ply) into the session gs_sequence/ folder."""

        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")

        session_root, record = self._load_record(str(session_id))
        gs_dir = session_root / SESSION_GS_DIRNAME
        gs_dir.mkdir(parents=True, exist_ok=True)

        # Update state to generating.
        record["state"] = "generating"
        record["error"] = None
        self._save_record(session_root, record)

        input_dir = Path(record.get("input_dir"))
        frames = _list_input_frames(input_dir)
        if not frames:
            raise FileNotFoundError(f"No input frames found in {input_dir}")

        out_ply = gs_dir / "000000.ply"


        generator = str(payload.get("generator", "mock"))
        init_mode = str(payload.get("init_mode", "imagedream_init"))
        if generator == "mock":
            _write_mock_compatible_gs_ply(
                out_ply,
                num_splats=int(payload.get("num_splats", 8000)),
                seed=int(payload.get("seed", 0)),
            )
        elif generator == "l4gm":
            if self._l4gm is None:
                raise RuntimeError("L4GM is not initialized")
            result = self._l4gm.generate_from_image(
                frames[0],
                out_ply,
                init_mode=init_mode,
                input_frames=frames,
            )
        else:
            raise RuntimeError(
                f"Unsupported generator '{generator}'. "
                "Use generator='mock' (default) or generator='l4gm'."
            )

        record["state"] = "reviewing"
        record["current_frame"] = 0
        record["seed_frame"] = 0
        record["seed_ply"] = str(out_ply)
        record["init_mode"] = init_mode
        record["generated_until"] = max(int(record.get("generated_until", -1)), 0)

        metrics_result = self._run_metrics(record, session_root, [0])
        self._save_record(session_root, record)



        return {
            "status": "success",
            "session_id": session_id,
            "ply_path": str(out_ply),
            "state": record["state"],
            "current_frame": record["current_frame"],
            "metrics": metrics_result,
        }

    def generate_later(self, payload: Dict) -> Dict:
        """Generate a subsequent GS segment after the current generated_until.

        Uses L4GM to create multi-view cache and forward num_clip_frames (default 8).
        """
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")

        session_root, record = self._load_record(str(session_id))
        input_dir = Path(record.get("input_dir"))
        frames = _list_input_frames(input_dir)
        if not frames:
            raise FileNotFoundError(f"No input frames found in {input_dir}")

        record["state"] = "generating"
        record["error"] = None
        self._save_record(session_root, record)

        # Start from the current frame (default 0) so the existing seed GS can be reused.
        start_idx = int(payload.get("start_frame", record.get("current_frame", 0)))
        if start_idx < 0:
            start_idx = 0
        if start_idx >= len(frames):
            raise RuntimeError("No remaining frames to generate from the requested start_frame")

        max_frames = int(payload.get("num_frames", 8))
        if max_frames <= 0:
            raise ValueError("num_frames must be > 0")

        slice_frames = frames[start_idx : start_idx + max_frames]
        num_clip_frames = len(slice_frames)
        if num_clip_frames == 0:
            raise RuntimeError("No frames selected for generation")

        if self._l4gm is None:
            raise RuntimeError("L4GM is not initialized")
        result = self._l4gm.generate_segment(
            slice_frames,
            session_root,
            start_idx,
            num_clip_frames,
        )

        record["state"] = "reviewing"
        record["generated_until"] = start_idx + num_clip_frames - 1
        record["current_frame"] = start_idx
        metrics_result = self._run_metrics(record, session_root, list(range(start_idx, start_idx + num_clip_frames)))
        self._save_record(session_root, record)

        return {
            "status": "success",
            "session_id": session_id,
            "ply_paths": result.get("ply_paths", []),
            "start_frame": result.get("start_frame"),
            "end_frame": result.get("end_frame"),
            "mv_preview_dir": result.get("mv_preview_dir"),
            "state": record["state"],
            "metrics": metrics_result,
        }

    def generate_last(self, payload: Dict) -> Dict:
        """Generate a previous GS segment in reverse order from the current frame."""
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")

        session_root, record = self._load_record(str(session_id))
        input_dir = Path(record.get("input_dir"))
        frames = _list_input_frames(input_dir)
        if not frames:
            raise FileNotFoundError(f"No input frames found in {input_dir}")

        record["state"] = "generating"
        record["error"] = None
        self._save_record(session_root, record)

        start_idx = int(payload.get("start_frame", record.get("current_frame", 0)))
        start_idx = max(0, min(start_idx, len(frames) - 1))

        max_frames = int(payload.get("num_frames", 8))
        if max_frames <= 0:
            raise ValueError("num_frames must be > 0")

        end_idx = max(0, start_idx - max_frames + 1)
        target_indices = list(range(start_idx, end_idx - 1, -1))
        slice_frames = [frames[i] for i in target_indices]
        if not slice_frames:
            raise RuntimeError("No frames selected for backward generation")

        if self._l4gm is None:
            raise RuntimeError("L4GM is not initialized")
        if not hasattr(self._l4gm, "generate_last_segment"):
            raise RuntimeError("Current L4GM backend does not support backward segment generation")

        result = self._l4gm.generate_last_segment(
            slice_frames,
            session_root,
            start_idx,
            len(slice_frames),
        )

        generated_indices = [int(i) for i in (result.get("generated_frame_indices") or target_indices)]
        generated_min = min(generated_indices) if generated_indices else start_idx
        generated_max = max(generated_indices) if generated_indices else start_idx

        record["state"] = "reviewing"
        record["generated_until"] = max(int(record.get("generated_until", -1)), generated_max)
        record["current_frame"] = generated_min
        metrics_result = self._run_metrics(record, session_root, generated_indices)
        self._save_record(session_root, record)

        return {
            "status": "success",
            "session_id": session_id,
            "ply_paths": result.get("ply_paths", []),
            "generated_frame_indices": generated_indices,
            "start_frame": int(result.get("start_frame", generated_min)),
            "end_frame": int(result.get("end_frame", generated_max)),
            "mv_preview_dir": result.get("mv_preview_dir"),
            "state": record["state"],
            "metrics": metrics_result,
        }

    def _resolve_memory_frame_indices(
        self,
        payload: Dict,
        record: Dict,
        num_total_frames: int,
        *,
        allow_empty: bool = False,
    ) -> List[int]:
        raw_indices = payload.get("memory_frame_indices")
        if raw_indices is None:
            bank = record.get("memory_bank") or {}
            raw_indices = bank.get("frame_indices") or []
        if not isinstance(raw_indices, list):
            raise ValueError("memory_frame_indices must be a list of ints")

        uniq: List[int] = []
        seen = set()
        for item in raw_indices:
            idx = int(item)
            if idx in seen:
                continue
            if idx < 0 or idx >= num_total_frames:
                raise ValueError(f"memory frame index out of range: {idx} (valid: 0-{num_total_frames - 1})")
            seen.add(idx)
            uniq.append(idx)
        uniq.sort()
        if (not uniq) and (not allow_empty):
            raise ValueError("No valid memory frames selected")
        return uniq

    def adjust_memory_frames(self, payload: Dict) -> Dict:
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")

        session_root, record = self._load_record(str(session_id))
        input_dir = Path(record.get("input_dir"))
        frames = _list_input_frames(input_dir)
        if not frames:
            raise FileNotFoundError(f"No input frames found in {input_dir}")

        allow_empty = bool(payload.get("allow_empty", False))
        memory_indices = self._resolve_memory_frame_indices(
            payload, record, len(frames), allow_empty=allow_empty
        )
        if not memory_indices:
            memory_bank_dir = session_root / "memory_bank"
            if memory_bank_dir.exists():
                for old in memory_bank_dir.glob("*"):
                    try:
                        if old.is_file() or old.is_symlink():
                            old.unlink()
                    except Exception:
                        pass
            record["memory_bank"] = {
                "frame_indices": [],
                "num_memory_frames": 0,
                "dir": str(memory_bank_dir),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            self._save_record(session_root, record)
            return {
                "status": "success",
                "session_id": session_id,
                "memory_frame_indices": [],
                "num_memory_frames": 0,
                "memory_bank_dir": str(memory_bank_dir),
                "entries": [],
            }

        if self._l4gm is None:
            raise RuntimeError("L4GM is not initialized")
        if not hasattr(self._l4gm, "build_memory_bank"):
            raise RuntimeError("Current L4GM backend does not support memory bank operations")
        missing = []
        for idx in memory_indices:
            ply_path = session_root / SESSION_GS_DIRNAME / f"{idx:06d}.ply"
            if not ply_path.exists():
                missing.append(str(ply_path))
        if missing:
            raise FileNotFoundError(
                "Some selected memory frames have no generated GS yet: "
                + ", ".join(missing[:5])
                + (" ..." if len(missing) > 5 else "")
            )

        frame_paths = [frames[i] for i in memory_indices]
        result = self._l4gm.build_memory_bank(
            frame_paths=frame_paths,
            frame_indices=memory_indices,
            session_root=session_root,
        )

        record["memory_bank"] = {
            "frame_indices": memory_indices,
            "num_memory_frames": int(result.get("num_memory_frames", len(memory_indices))),
            "dir": str(result.get("memory_bank_dir", session_root / "memory_bank")),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        self._save_record(session_root, record)

        return {
            "status": "success",
            "session_id": session_id,
            "memory_frame_indices": memory_indices,
            "num_memory_frames": int(result.get("num_memory_frames", len(memory_indices))),
            "memory_bank_dir": result.get("memory_bank_dir", str(session_root / "memory_bank")),
            "entries": result.get("entries", []),
        }

    def generate_later_with_memory(self, payload: Dict) -> Dict:
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")
        if self._l4gm is None:
            raise RuntimeError("L4GM is not initialized")
        if not hasattr(self._l4gm, "generate_segment_with_memory"):
            raise RuntimeError("Current L4GM backend does not support memory-aware generation")

        session_root, record = self._load_record(str(session_id))
        input_dir = Path(record.get("input_dir"))
        frames = _list_input_frames(input_dir)
        if not frames:
            raise FileNotFoundError(f"No input frames found in {input_dir}")

        record["state"] = "generating"
        record["error"] = None
        self._save_record(session_root, record)

        start_idx = int(payload.get("start_frame", record.get("current_frame", 0)))
        start_idx = max(0, min(start_idx, len(frames) - 1))
        total_clip_frames = int(payload.get("total_clip_frames", 16))
        debug_ref_video = bool(payload.get("debug_ref_video", False))
        if total_clip_frames <= 0:
            raise ValueError("total_clip_frames must be > 0")

        memory_indices = self._resolve_memory_frame_indices(payload, record, len(frames))
        if len(memory_indices) >= total_clip_frames:
            raise ValueError(
                f"memory frame count ({len(memory_indices)}) must be smaller than total_clip_frames ({total_clip_frames})"
            )

        for idx in memory_indices:
            ply_path = session_root / SESSION_GS_DIRNAME / f"{idx:06d}.ply"
            if not ply_path.exists():
                raise FileNotFoundError(f"Memory frame has no generated GS yet: {ply_path}")

        memory_bank = record.get("memory_bank") or {}
        memory_bank_dir = Path(memory_bank.get("dir") or (session_root / "memory_bank"))
        bank_indices = [int(x) for x in (memory_bank.get("frame_indices") or [])]
        bank_json = memory_bank_dir / "memory_bank.json"
        need_refresh_bank = (bank_indices != memory_indices) or (not bank_json.exists())

        if need_refresh_bank:
            if not hasattr(self._l4gm, "build_memory_bank"):
                raise RuntimeError("Current L4GM backend does not support memory bank operations")
            prep = self._l4gm.build_memory_bank(
                frame_paths=[frames[i] for i in memory_indices],
                frame_indices=memory_indices,
                session_root=session_root,
            )
            memory_bank_dir = Path(prep.get("memory_bank_dir", session_root / "memory_bank"))
            record["memory_bank"] = {
                "frame_indices": memory_indices,
                "num_memory_frames": int(prep.get("num_memory_frames", len(memory_indices))),
                "dir": str(memory_bank_dir),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            self._save_record(session_root, record)

        num_future_frames = total_clip_frames - len(memory_indices)
        slice_frames = frames[start_idx : start_idx + num_future_frames]
        if not slice_frames:
            raise RuntimeError("No future frames to generate for the requested start_frame")

        result = self._l4gm.generate_segment_with_memory(
            frame_paths=slice_frames,
            session_root=session_root,
            start_frame_idx=start_idx,
            total_clip_frames=total_clip_frames,
            memory_bank_dir=memory_bank_dir,
            debug_ref_video=debug_ref_video,
        )

        ply_paths = result.get("ply_paths", []) or []
        generated_count = len(ply_paths)
        if generated_count <= 0:
            raise RuntimeError("Memory-aware generation produced no PLY outputs")

        end_idx = start_idx + generated_count - 1
        record["state"] = "reviewing"
        record["generated_until"] = max(int(record.get("generated_until", -1)), end_idx)
        record["current_frame"] = start_idx
        record["memory_bank"] = {
            "frame_indices": memory_indices,
            "num_memory_frames": len(memory_indices),
            "dir": str(memory_bank_dir),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        metrics_result = self._run_metrics(record, session_root, list(range(start_idx, end_idx + 1)))
        self._save_record(session_root, record)

        return {
            "status": "success",
            "session_id": session_id,
            "ply_paths": ply_paths,
            "start_frame": int(result.get("start_frame", start_idx)),
            "end_frame": int(result.get("end_frame", end_idx)),
            "mv_preview_dir": result.get("mv_preview_dir"),
            "memory_bank_dir": str(memory_bank_dir),
            "num_memory_frames": len(memory_indices),
            "num_generated_frames": generated_count,
            "total_clip_frames": int(result.get("total_clip_frames", total_clip_frames)),
            "debug_ref_video_path": result.get("debug_ref_video_path"),
            "state": record["state"],
            "metrics": metrics_result,
        }

    # --- Multi-view rendering (backend) ---------------------------------
    def _load_gaussian_tensors(self, ply_path: Path) -> Dict[str, np.ndarray]:
        # Use centralized ply loader that mirrors front-end transforms.
        splat = load_splat_from_ply(ply_path)
        return splatdata_to_backend_dict(splat)

    def _prepare_gsplat_inputs_tensor(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means_t = torch.as_tensor(data["means"], device=self.device, dtype=torch.float32)
        scales_t = torch.as_tensor(data["scales"], device=self.device, dtype=torch.float32)
        rotations_t = torch.as_tensor(data["rotations"], device=self.device, dtype=torch.float32)
        opacity_t = torch.as_tensor(data["opacity"], device=self.device, dtype=torch.float32)
        sh_t = torch.as_tensor(data.get("sh_dc", data.get("sh", None)), device=self.device, dtype=torch.float32)

        if rotations_t.ndim == 1:
            rotations_t = rotations_t.expand(means_t.shape[0], -1)
        if rotations_t.ndim == 3 and rotations_t.shape[1:] == (3, 3):
            rotations_t = tf.SO3(rotations_t).wxyz.to(device=self.device, dtype=torch.float32)
        if rotations_t.ndim == 2 and rotations_t.shape[1] == 3:
            ones = torch.ones((rotations_t.shape[0], 1), device=self.device, dtype=torch.float32)
            rotations_t = torch.cat([ones, rotations_t], dim=1)
        if rotations_t.shape[-1] != 4:
            rotations_t = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).expand(means_t.shape[0], 4)
        rotations_t = rotations_t / (rotations_t.norm(dim=1, keepdim=True) + 1e-8)

        # `load_splat_from_ply` already applied `exp` to scales and `sigmoid` to opacity,
        # and colors are computed from SH there. So don't re-apply transforms here.
        colors_full = torch.clamp(0.5 + SH_C0 * sh_t, 0.0, 1.0)
        return means_t, scales_t, rotations_t, opacity_t, colors_full

    def _build_camera_tensors(
        self, rotation_wc: np.ndarray, position_w: np.ndarray, width: int, height: int, fov_deg: float
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        fx = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        fy = height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        cx = width * 0.5
        cy = height * 0.5
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32)
        Ks = K.unsqueeze(0)

        R_cw = torch.as_tensor(rotation_wc.T, device=self.device, dtype=torch.float32)
        t_cw = (-R_cw @ torch.as_tensor(position_w, device=self.device, dtype=torch.float32)).view(3)
        viewmat = torch.eye(4, device=self.device, dtype=torch.float32)
        viewmat[:3, :3] = R_cw
        viewmat[:3, 3] = t_cw
        viewmats = viewmat.unsqueeze(0)

        near, far = 0.01, 100.0
        return viewmats, Ks, near, far

    def _gsplat_render(
        self,
        means_t: torch.Tensor,
        scales_t: torch.Tensor,
        rotations_t: torch.Tensor,
        opacity_t: torch.Tensor,
        colors_in: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        near: float,
        far: float,
        detach: bool = True,
        to_numpy: bool = True,
        backgrounds: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor | np.ndarray:
        if backgrounds is None:
            backgrounds = torch.ones((1, 3), device=means_t.device, dtype=colors_in.dtype)
        else:
            backgrounds = torch.as_tensor(backgrounds, device=means_t.device, dtype=colors_in.dtype)
        out = gsplat_rasterization(
            means=means_t,
            quats=rotations_t,
            scales=scales_t,
            opacities=opacity_t,
            colors=colors_in,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=near,
            far_plane=far,
            packed=False,
            backgrounds=backgrounds,
            render_mode="RGB",
        )
        img = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(img, dict):
            img = img.get("rgb") or img.get("color") or img.get("image")
        if img is None:
            raise RuntimeError("gsplat rasterization returned None")
        if img.dim() == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        if detach:
            img = img.detach()
        img = img.clamp(0.0, 1.0)
        if to_numpy:
            return img.cpu().numpy().astype(np.float32)
        return img

    def _render_rgb_mask_simple(
        self,
        data: Dict[str, np.ndarray],
        rotation_wc: np.ndarray,
        position_w: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        means = sanitize_points(data["means"])
        scales = data["scales"]
        opacity = data["opacity"]
        sh_dc = data["sh_dc"]

        pts_cam = (means - sanitize_points(position_w[None, :])) @ rotation_wc
        z = pts_cam[:, 2]
        fx = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        fy = height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        u = fx * (pts_cam[:, 0] / np.maximum(z, 1e-4)) + width * 0.5
        v = fy * (pts_cam[:, 1] / np.maximum(z, 1e-4)) + height * 0.5

        depth_map = np.full((height, width), np.inf, dtype=np.float32)
        index_map = np.full((height, width), -1, dtype=np.int64)
        valid = (z > 0) & np.isfinite(z)
        if np.any(valid):
            ui = np.round(u[valid]).astype(np.int64)
            vi = np.round(v[valid]).astype(np.int64)
            inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            ui = ui[inside]
            vi = vi[inside]
            idx_valid = np.nonzero(valid)[0][inside]
            depth_valid = z[idx_valid]
            for px, py, d, idx in zip(ui, vi, depth_valid, idx_valid):
                if d < depth_map[py, px]:
                    depth_map[py, px] = d
                    index_map[py, px] = idx

        chosen = index_map >= 0
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        if np.any(chosen):
            chosen_idx = index_map[chosen]
            alpha = 1.0 / (1.0 + np.exp(-opacity[chosen_idx]))
            colors = np.clip(0.5 + SH_C0 * sh_dc[chosen_idx], 0.0, 1.0)
            rgb[chosen] = colors * alpha[:, None]
        mask = (chosen.astype(np.uint8) * 255)
        return rgb, mask

    def _render_rgb_mask(
        self,
        data: Dict[str, np.ndarray],
        rotation_wc: np.ndarray,
        position_w: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if gsplat_rasterization is None:
            print("render sample")
            return self._render_rgb_mask_simple(data, rotation_wc, position_w, width, height, fov_deg)
        try:
            means_t, scales_t, rotations_t, opacity_t, colors_full = self._prepare_gsplat_inputs_tensor(data)
            viewmats, Ks, near, far = self._build_camera_tensors(rotation_wc, position_w, width, height, fov_deg)
            rgb = self._gsplat_render(
                means_t,
                scales_t,
                rotations_t,
                opacity_t,
                colors_full,
                viewmats,
                Ks,
                width,
                height,
                near,
                far,
                detach=True,
                to_numpy=True,
                backgrounds=np.ones((1, 3), dtype=np.float32),
            )
            mask_rgb = self._gsplat_render(
                means_t,
                scales_t,
                rotations_t,
                opacity_t,
                torch.ones_like(colors_full),
                viewmats,
                Ks,
                width,
                height,
                near,
                far,
                detach=True,
                to_numpy=True,
                backgrounds=np.zeros((1, 3), dtype=np.float32),
            )
            mask = (np.clip(mask_rgb.mean(axis=2), 0.0, 1.0) * 255.0).astype(np.uint8)

            rgb = np.rot90(rgb, 2)
            mask = np.rot90(mask, 2)
            return rgb, mask
        except Exception:
            traceback.print_exc()
            return self._render_rgb_mask_simple(data, rotation_wc, position_w, width, height, fov_deg)

    def _direction_from_angles(self, az_deg: float, el_deg: float, up_axis: str = "y") -> np.ndarray:
        return direction_from_angles(az_deg, el_deg, up_axis)

    def _eye_wxyz_from_angles(
        self, az_deg: float, el_deg: float, distance: float, center: np.ndarray, up_axis: str = "y"
    ) -> Tuple[np.ndarray, np.ndarray]:
        return eye_wxyz_from_angles(az_deg, el_deg, distance, center, up_axis)

    def render_multiview(self, payload: Dict) -> Dict:
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")

        session_root, record = self._load_record(str(session_id))
        input_dir = Path(record.get("input_dir"))
        frames = _list_input_frames(input_dir)
        total_frames = len(frames)
        if total_frames == 0:
            raise FileNotFoundError(f"No input frames found in {input_dir}")

        start_frame = int(payload.get("start_frame", record.get("current_frame", 0)))
        end_frame = int(payload.get("end_frame", start_frame))
        if start_frame < 0:
            start_frame = 0
        if end_frame < start_frame:
            end_frame = start_frame
        start_frame = min(start_frame, total_frames - 1)
        end_frame = min(end_frame, total_frames - 1)

        azimuths = payload.get("azimuths") or [0, 60, 120, 180, 240]
        azimuths = [float(a) for a in azimuths]
        elevation_deg = float(payload.get("elevation_deg", 0.0))
        distance = float(payload.get("distance", 1.5))
        width = height = int(payload.get("image_size", 512))
        fov_deg = float(payload.get("fov_deg", 49.1))
        up_axis = payload.get("up_axis", "y")

        rgb_dir = session_root / "multi_view"
        mask_dir = session_root / "multi_view_mask"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Dict] = []
        for frame_idx in range(start_frame, end_frame + 1):
            ply_path = session_root / SESSION_GS_DIRNAME / f"{frame_idx:06d}.ply"
            ply_path = (
                ply_path
                if ply_path.exists()
                else session_root / SESSION_GS_DIRNAME / f"wo_interp_000000_{frame_idx:03d}.ply"
            )
            if not ply_path.exists():
                print(f"[multiview] skip frame {frame_idx}: missing {ply_path}")
                continue
            data = self._load_gaussian_tensors(ply_path)
            center = np.zeros(3, dtype=np.float32)
            distance_use = 1.5
            token = frames[frame_idx].stem if frame_idx < len(frames) else f"{frame_idx:06d}"

            for az in azimuths:
                eye, wxyz = self._eye_wxyz_from_angles(float(az), elevation_deg, 1.5,  np.zeros(3) , up_axis=up_axis)
                rotation_wc = tf.SO3(np.asarray(wxyz, dtype=np.float32)).as_matrix()
                rgb, mask = self._render_rgb_mask(data, rotation_wc, eye, width, height, fov_deg)
                rgb_path = rgb_dir / f"{token}_az{int(round(az)) % 360:03d}.png"
                mask_path = mask_dir / f"{token}_az{int(round(az)) % 360:03d}.png"
                iio.imwrite(rgb_path, (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
                iio.imwrite(mask_path, mask)
                saved.append({"frame": frame_idx, "azimuth": float(az), "rgb": str(rgb_path), "mask": str(mask_path)})
                print("rgb_path:", rgb_path)
        return {
            "status": "success",
            "session_id": session_id,
            "saved": saved,
            "rgb_dir": str(rgb_dir),
            "mask_dir": str(mask_dir),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "azimuths": azimuths,
            "elevation_deg": elevation_deg,
        }

    def sv4d_inpaint(self, payload: Dict) -> Dict:
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")

        session_root, record = self._load_record(str(session_id))
        input_dir = Path(record.get("input_dir"))
        frames = _list_input_frames(input_dir)
        if not frames:
            raise FileNotFoundError(f"No input frames found in {input_dir}")

        target_frame = int(payload.get("frame", record.get("current_frame", 0)))
        target_frame = max(0, min(target_frame, len(frames) - 1))
        window = 12
        start_frame = max(0, target_frame - (window - 1))
        frame_indices = list(range(start_frame, target_frame + 1))
        if len(frame_indices) < window:
            frame_indices += [target_frame] * (window - len(frame_indices))

        anchor_view = payload.get("anchor_view", "input")
        anchor_az, anchor_el = VIEW_PRESETS.get(anchor_view, (0.0, 0.0))
        sv4d_runs = int(payload.get("sv4d_runs", 5))
        sv4d_batch_views = int(payload.get("sv4d_batch_views", 4))
        azimuths_rel = payload.get("azimuths")
        if azimuths_rel is None:
            total_novel = sv4d_runs * sv4d_batch_views
            step = 360.0 / max(total_novel, 1)
            azimuths_rel = [0.0] + [step * (i + 1) for i in range(total_novel)]
        azimuths = [float((float(a) + anchor_az) % 360.0) for a in azimuths_rel]

        if len(azimuths) > 1 and sv4d_batch_views > 0:
            anchor_azimuth = azimuths[0]
            novel_sorted = sorted(azimuths[1:])
            reordered: List[float] = []
            for group_id in range(int(sv4d_runs)):
                for offset in range(int(sv4d_batch_views)):
                    idx = offset * int(sv4d_runs) + group_id
                    if idx < len(novel_sorted):
                        reordered.append(novel_sorted[idx])
            azimuths = [anchor_azimuth] + reordered

        elevation_deg = float(payload.get("elevation_deg", anchor_el))
        width = height = int(payload.get("image_size", 576))
        fov_deg = float(payload.get("fov_deg", 49.1))
        up_axis = payload.get("up_axis", "y")

        elevation_range = payload.get("elevation_range", [-15.0, 15.0])
        try:
            elev_min = float(elevation_range[0])
            elev_max = float(elevation_range[1])
        except Exception:
            elev_min, elev_max = -5.0, 10.0
        if elev_min > elev_max:
            elev_min, elev_max = elev_max, elev_min
        seed_base = int(payload.get("seed", 23))
        rng = np.random.default_rng(seed_base + int(target_frame))
        elevations_deg = rng.uniform(elev_min, elev_max, size=len(azimuths)).astype(np.float32).tolist()

        base_dir = session_root / "tmp_sv4d" / f"frame_{target_frame:06d}"
        render_dir = base_dir / "render"
        mask_dir = base_dir / "mask"
        input_out_dir = base_dir / "input"
        output_dir = base_dir / "output"
        render_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        input_out_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for clip_pos, f_idx in enumerate(frame_indices):
            token = f"{clip_pos:06d}"
            src_idx = min(f_idx, len(frames) - 1)
            ply_path = session_root / SESSION_GS_DIRNAME / f"{src_idx:06d}.ply"
            if not ply_path.exists():
                ply_path = session_root / SESSION_GS_DIRNAME / f"wo_interp_000000_{src_idx:03d}.ply"
            if not ply_path.exists():
                raise FileNotFoundError(f"Missing GS ply for frame {src_idx}: {ply_path}")

            data = self._load_gaussian_tensors(ply_path)
            center = np.zeros(3)
            radius = 1.5
            distance_use = radius

            for az_idx, az in enumerate(azimuths):
                elev = elevations_deg[az_idx] if az_idx < len(elevations_deg) else elevation_deg
                eye, wxyz = self._eye_wxyz_from_angles(float(az), float(elev), distance_use, center, up_axis=up_axis)
                rotation_wc = tf.SO3(np.asarray(wxyz, dtype=np.float32)).as_matrix()
                rgb, mask = self._render_rgb_mask(data, rotation_wc, eye, width, height, fov_deg)
                rgb_path = render_dir / f"{token}_az{int(round(az)) % 360:03d}.png"
                mask_path = mask_dir / f"{token}_az{int(round(az)) % 360:03d}.png"
                iio.imwrite(rgb_path, (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
                iio.imwrite(mask_path, mask)

            src_img = frames[src_idx]
            dst_img = input_out_dir / f"{token}.png"
            shutil.copy(src_img, dst_img)

        debug_mode = bool(payload.get("debug_mode", True))
        frame_token = f"{target_frame:06d}"
        cached_png_root = output_dir / "png"

        def _has_cached_sv4d_outputs(png_root: Path, token: str) -> bool:
            if not png_root.exists():
                return False
            for eval_dir in png_root.glob("eval_*"):
                if (eval_dir / f"{token}.png").exists():
                    return True
            return False

        if debug_mode and _has_cached_sv4d_outputs(cached_png_root, frame_token):
            view_azimuths_cached = azimuths[1:] if len(azimuths) > 1 else azimuths
            result = types.SimpleNamespace(
                png_root=str(cached_png_root),
                view_azimuths=view_azimuths_cached,
                output_dir=str(output_dir),
            )
            print("Using cached SV4D outputs from:", cached_png_root)
        else:
            backend = self._get_sv4d_backend()
            result = backend.inpaint(
                render_root=render_dir,
                mask_root=mask_dir,
                input_root=input_out_dir,
                output_dir=output_dir,
                num_steps=int(payload.get("num_steps", 50)),
                img_size=width,
                seed=int(payload.get("seed", 23)),
                encoding_t=int(payload.get("encoding_t", 8)),
                decoding_t=int(payload.get("decoding_t", 4)),
                elevations_deg=elevations_deg,
                azimuths_deg=azimuths,
                cond_mv=bool(payload.get("cond_mv", False)),
                sigma_start=float(payload.get("sigma_start", 250.0)),
                num_refine_steps=int(payload.get("num_refine_steps", 25)),
                frame_token=frame_token,
                batch_runs=sv4d_runs,
                batch_views=sv4d_batch_views,
            )

        meta_path = base_dir / "meta.json"
        view_azimuths = result.view_azimuths if hasattr(result, "view_azimuths") else (azimuths[1:] if len(azimuths) > 1 else azimuths)
        elevation_lookup: Dict[float, float] = {}
        for idx, az in enumerate(azimuths):
            if az not in elevation_lookup and idx < len(elevations_deg):
                elevation_lookup[float(az)] = float(elevations_deg[idx])
        view_mappings = []
        for global_view_idx, az in enumerate(view_azimuths):
            view_mappings.append(
                {
                    "azimuth": float(az),
                    "elevation": float(elevation_lookup.get(float(az), elevation_deg)),
                    "global_view_idx": int(global_view_idx),
                    "eval_dir": f"eval_{global_view_idx}",
                    "video": f"sample_v{global_view_idx + 1:03d}.mp4",
                }
            )

        meta_payload = {
            "session_id": session_id,
            "frame": target_frame,
            "anchor_view": anchor_view,
            "anchor_az": anchor_az,
            "anchor_el": anchor_el,
            "azimuths_rel": azimuths_rel,
            "azimuths": azimuths,
            "elevation_range": [elev_min, elev_max],
            "elevations_deg": elevations_deg,
            "image_size": [width, height],
            "fov_deg": fov_deg,
            "seed": seed_base,
            "view_mappings": view_mappings,
        }
        try:
            meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print(f"[sv4d] meta save failed: {exc}")

        # Save per-view SV4D frames as GT candidates and camera overrides.
        png_root = Path(result.png_root)
        frame_output_dir = base_dir / "frame_output"
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        opt_eval_dir = frame_output_dir / "eval_sv4d"
        opt_eval_dir.mkdir(parents=True, exist_ok=True)

        ply_target = session_root / SESSION_GS_DIRNAME / f"{target_frame:06d}.ply"
        if not ply_target.exists():
            ply_target = session_root / SESSION_GS_DIRNAME / f"wo_interp_000000_{target_frame:03d}.ply"

        center_target = np.zeros(3)
        radius_target = 1.5
        distance_target = radius_target

        # Output views correspond to rendered azimuths (skip anchor view0).
        view_azimuths = view_azimuths
        view_mapping_by_az: Dict[float, Dict] = {}
        for item in view_mappings:
            try:
                view_mapping_by_az[float(item.get("azimuth"))] = item
            except Exception:
                continue
        camera_overrides: Dict[str, Dict] = {}
        for v_idx, az in enumerate(view_azimuths):
            mapped = view_mapping_by_az.get(float(az))
            eval_dir = mapped.get("eval_dir") if mapped else None
            print("eval_dir:", eval_dir, "azi:", az)
            src_img = png_root / eval_dir / f"{target_frame:06d}.png"
            print("src_img:", src_img, " aziL", az , " ele:",  elevation_deg)
            print("pred render up_axis:", up_axis)
            eye, wxyz = self._eye_wxyz_from_angles(float(az), elevation_deg, distance_target, center_target, up_axis=up_axis)
            out_name = f"azi_{int(round(az))}_ele_{int(round(elevation_deg))}_frame_{target_frame:06d}.png"
            out_path = frame_output_dir / out_name
            shutil.copy(src_img, out_path)

            view_name = f"sv4d_az{int(round(az))}_el{int(round(elevation_deg))}"
            view_dir = opt_eval_dir / view_name
            view_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_img, view_dir / f"{target_frame:06d}.png")
            camera_overrides[view_name] = {"eye": eye.tolist(), "wxyz": wxyz.tolist()}

        print("optimize_after:", payload.get("optimize", True))

        optimize_after = bool(payload.get("optimize", True))
        opt_result: Dict | None = None
        if optimize_after:
            ply_path_final = session_root / SESSION_GS_DIRNAME / f"{target_frame:06d}.ply"
            if not ply_path_final.exists():
                ply_path_final = session_root / SESSION_GS_DIRNAME / f"wo_interp_000000_{target_frame:03d}.ply"
            if not ply_path_final.exists():
                raise FileNotFoundError(f"Missing GS ply for optimization: {ply_path_final}")

            selection_indices = payload.get("selection_indices")
            if isinstance(selection_indices, dict):
                selection_indices = selection_indices.get("gs_indices") or selection_indices.get("indices")
            if selection_indices is not None:
                selection_indices = [int(i) for i in selection_indices]
                if not selection_indices:
                    selection_indices = None

            view_specs: List[Dict] = []
            for view_name, cam in camera_overrides.items():
                img_path = opt_eval_dir / view_name / f"{target_frame:06d}.png"
                if not img_path.exists():
                    continue
                view_specs.append(
                    {
                        "name": view_name,
                        "image_path": str(img_path),
                        "eye": cam["eye"],
                        "wxyz": cam["wxyz"],
                    }
                )

            opt_output_dir = output_dir / "optimize"
            # Use the shared fixer instance so progress is visible to /fix_progress polling
            optimizer = ROIRequestHandler.fixer
            opt_result = optimizer.optimize_with_views(
                gs_ply=ply_path_final,
                view_specs=view_specs,
                output_dir=opt_output_dir,
                image_size=(width, height),
                selection_indices=selection_indices,
                frame_token=f"{target_frame:06d}",
                experiment_id=payload.get("experiment_id"),
                opt_params=payload.get("opt_params"),
            )

        return {
            "status": "success",
            "session_id": session_id,
            "frame": target_frame,
            "frames_used": frame_indices,
            "render_root": str(render_dir),
            "mask_root": str(mask_dir),
            "input_root": str(input_out_dir),
            "sv4d_result": result.__dict__ if hasattr(result, "__dict__") else result,
            "output_dir": str(output_dir),
            "azimuths": azimuths,
            "elevation_deg": elevation_deg,
            "optimize_result": opt_result,
        }


class ROISelector:
    def __init__(self, depth_margin_ratio: float = 0.05, min_depth_margin: float = 0.01):
        self.depth_margin_ratio = depth_margin_ratio
        self.min_depth_margin = min_depth_margin

    def handle(self, payload: Dict) -> Dict:
        bbox = payload.get("bbox_2d")
        cam = payload.get("camera", {})
        image_size = payload.get("image_size", [512, 512])
        paths = payload.get("paths", {})
        if bbox is None or "gs_ply" not in paths:
            raise ValueError("Payload missing bbox_2d or paths.gs_ply")

        width, height = int(image_size[0]), int(image_size[1])
        gs_path = Path(paths["gs_ply"])
        output_dir = Path(paths.get("output_dir", ROOT / "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        centers = load_gaussian_centers(gs_path)
        depth_margin_ratio = float(payload.get("depth_margin_ratio", self.depth_margin_ratio))
        min_depth_margin = float(payload.get("min_depth_margin", self.min_depth_margin))
        rotation_wc = camera_to_world_rotation(cam.get("wxyz", [1, 0, 0, 0]))
        position_w = np.asarray(cam.get("position", [0, 0, 0]), dtype=np.float32)

        points_cam = world_to_camera_points(centers, rotation_wc, position_w)
        u, v, z = project_points(points_cam, width, height, float(cam.get("fov_deg", 49.1)))
        depth_map = build_depth_map(u, v, z, width, height)

        in_bbox = (
            (u >= bbox["x0"])
            & (u < bbox["x1"])
            & (v >= bbox["y0"])
            & (v < bbox["y1"])
            & (z > max(cam.get("near", 0.0), 0.0))
            & (z < cam.get("far", float("inf")))
        )

        valid_px = (
            (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
            & np.isfinite(z)
            & (z > 0)
        )
        depth_margin = np.maximum(min_depth_margin, np.abs(depth_map) * depth_margin_ratio)
        ui = np.clip(np.round(u).astype(np.int64), 0, width - 1)
        vi = np.clip(np.round(v).astype(np.int64), 0, height - 1)
        depth_at_px = depth_map[vi, ui]
        close_enough = z <= (depth_at_px + depth_margin[vi, ui])

        # Extra culling: constrain to near-front surface within ROI using ROI min depth.
        roi_depth = depth_map[
            int(max(0, bbox["y0"])) : int(min(height, bbox["y1"])),
            int(max(0, bbox["x0"])) : int(min(width, bbox["x1"])),
        ]
        finite_roi = roi_depth[np.isfinite(roi_depth) & (roi_depth > 0)]
        roi_min_depth = float(finite_roi.min()) if finite_roi.size > 0 else None
        roi_depth_cutoff = None
        if roi_min_depth is not None:
            roi_margin = max(min_depth_margin, roi_min_depth * depth_margin_ratio)
            roi_depth_cutoff = roi_min_depth + roi_margin
            selected = in_bbox & valid_px & close_enough & (z <= roi_depth_cutoff)
        else:
            selected = in_bbox & valid_px & close_enough

        indices = np.nonzero(selected)[0].astype(int).tolist()

        request_copy = output_dir / "request.json"
        depth_path = output_dir / "depth_map.png"
        overlay_path = output_dir / "selection_overlay.png"
        request_copy.write_text(json.dumps(payload, indent=2))
        depth_vis = (normalize_depth(depth_map) * 255.0).astype(np.uint8)
        iio.imwrite(depth_path, depth_vis)
        draw_selection_overlay(width, height, bbox, u, v, selected, overlay_path)

        return {
            "status": "success",
            "experiment_id": payload.get("experiment_id"),
            "selection": {
                "gs_indices": indices,
                "count": len(indices),
                "bbox": bbox,
                "depth_margin_ratio": depth_margin_ratio,
                "min_depth_margin": min_depth_margin,
                "roi_min_depth": roi_min_depth,
                "roi_depth_cutoff": roi_depth_cutoff,
            },
            "paths": {
                "request": str(request_copy),
                "depth_map": str(depth_path),
                "overlay": str(overlay_path),
            },
        }




def health_payload() -> Dict[str, str]:
    return {"status": "ok", "service": "roi_backend"}


class ROIRequestHandler(BaseHTTPRequestHandler):
    selector = ROISelector()
    fixer: FixOptimizer | None = None
    sessions: SessionManager | None = None
    api_token = ""
    max_request_bytes = DEFAULT_MAX_REQUEST_BYTES
    allowed_roots = parse_allowed_roots(ROOT.parent)

    def _authorize(self) -> bool:
        if request_is_authorized(self.headers, self.api_token):
            return True
        self._send_json({"status": "error", "message": "unauthorized"}, code=401)
        return False

    @classmethod
    def _require_sessions(cls) -> SessionManager:
        if cls.sessions is None:
            raise RuntimeError("ROI model services have not been initialized")
        return cls.sessions

    @classmethod
    def _require_fixer(cls) -> FixOptimizer:
        if cls.fixer is None:
            raise RuntimeError("ROI optimizer service has not been initialized")
        return cls.fixer

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in (
            "/select_roi",
            "/optimize_roi",
            "/fix_roi",
            "/archor_view_refine",
            "/session/start",
            "/generate/first",
            "/generate/later_gen",
            "/generate/last_segment",
            "/memory/adjust_frames",
            "/generate/later_gen_with_memory",
            "/render/multiview",
            "/sv4d/inpaint",
        ):
            self._send_json(
                {"status": "error", "message": "Unsupported endpoint"},
                code=404,
            )
            return
        if not self._authorize():
            return
        try:
            payload = read_json_body(self, self.max_request_bytes)
            validate_payload_paths(payload, self.allowed_roots)
        except RequestError as exc:
            self._send_json({"status": "error", "message": str(exc)}, code=exc.status)
            return

        try:
            if self.path == "/session/start":
                result = self._require_sessions().start(payload)
            elif self.path == "/generate/first":
                result = self._require_sessions().generate_first(payload)
            elif self.path == "/generate/later_gen":
                result = self._require_sessions().generate_later(payload)
            elif self.path == "/generate/last_segment":
                result = self._require_sessions().generate_last(payload)
            elif self.path == "/memory/adjust_frames":
                result = self._require_sessions().adjust_memory_frames(payload)
            elif self.path == "/generate/later_gen_with_memory":
                result = self._require_sessions().generate_later_with_memory(payload)
            elif self.path == "/render/multiview":
                result = self._require_sessions().render_multiview(payload)
            elif self.path == "/sv4d/inpaint":
                result = self._require_sessions().sv4d_inpaint(payload)
            elif self.path == "/fix_roi":
                result = self._require_fixer()._optimize_roi(payload)
            elif self.path == "/archor_view_refine":
                result = self._require_fixer().archor_view_refine(payload)
            else:
                result = self.selector.handle(payload)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                {
                    "status": "error",
                    "message": f"{type(exc).__name__}: request failed; see server logs",
                },
                code=500,
            )
            return
        self._send_json(result, code=200)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path in {"", "/", "/health"}:
            self._send_json(health_payload(), code=200)
            return
        if not self._authorize():
            return
        if parsed.path == "/session/status":
            session_id_list = qs.get("session_id")
            if not session_id_list:
                self._send_json({"status": "error", "message": "session_id is required"}, code=400)
                return
            session_id = session_id_list[0]
            try:
                payload = self._require_sessions().status(session_id)
            except Exception as exc:
                traceback.print_exc()
                self._send_json(
                    {
                        "status": "error",
                        "message": f"{type(exc).__name__}: session lookup failed; see server logs",
                    },
                    code=404,
                )
                return
            self._send_json(payload, code=200)
            return

        if parsed.path.startswith("/fix_progress"):
            exp_id_list = qs.get("experiment_id") or qs.get("exp_id")
            if not exp_id_list:
                self._send_json({"status": "error", "message": "experiment_id is required"}, code=400)
                return
            exp_id = exp_id_list[0]
            entry = self._require_fixer().progress.get(exp_id)
            if entry is None:
                self._send_json({"status": "pending", "experiment_id": exp_id}, code=202)
                return
            payload = {"status": entry.get("status", "unknown"), "experiment_id": exp_id}
            payload.update(entry)
            self._send_json(payload, code=200)
            return
        if parsed.path.startswith("/metric_progress"):
            session_id_list = qs.get("session_id") or qs.get("sid")
            if not session_id_list:
                self._send_json({"status": "error", "message": "session_id is required"}, code=400)
                return
            session_id = session_id_list[0]
            entry = self._require_sessions().metric_progress.get(session_id)
            if entry is None:
                self._send_json({"status": "pending", "session_id": session_id}, code=202)
                return
            payload = {"status": entry.get("status", "unknown"), "session_id": session_id}
            payload.update(entry)
            self._send_json(payload, code=200)
            return
        self._send_json({"status": "error", "message": "Unsupported endpoint"}, code=404)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        sys.stderr.write(
            "[ROIBackend] %s - - [%s] %s\n"
            % (self.client_address[0], self.log_date_time_string(), format % args)
        )

    def _send_json(self, payload: Dict, code: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            # Client closed connection (e.g., UI refresh); ignore to avoid crashing the server thread.
            pass


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ROI backend for GS selection.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-token", default=os.environ.get("HAT4D_API_TOKEN", ""))
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BYTES,
    )
    parser.add_argument("--depth-margin-ratio", type=float, default=0.05)
    parser.add_argument("--min-depth-margin", type=float, default=0.01)
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Serve health checks without loading model dependencies or checkpoints.",
    )

    parser.add_argument(
        "--l4gm-init-sv4d",
        action="store_true",
        help="Initialize L4GM session with SV4D enabled",
    )


    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    require_secure_bind(args.host, args.api_token)
    ROIRequestHandler.selector = ROISelector(
        depth_margin_ratio=args.depth_margin_ratio,
        min_depth_margin=args.min_depth_margin,
    )

    ROIRequestHandler.sessions = None
    ROIRequestHandler.fixer = None
    ROIRequestHandler.api_token = str(args.api_token or "")
    ROIRequestHandler.max_request_bytes = int(args.max_request_bytes)
    ROIRequestHandler.allowed_roots = parse_allowed_roots(ROOT.parent)
    if not args.health_only:
        from good_optimizer import FixOptimizer

        ROIRequestHandler.fixer = FixOptimizer()
        ROIRequestHandler.sessions = SessionManager(
            default_workspace_root=ROOT / "workspace",
            l4gm_init_sv4d=args.l4gm_init_sv4d,
        )

    server = ThreadingHTTPServer((args.host, args.port), ROIRequestHandler)
    print(
        f"ROI backend listening on {args.host}:{args.port} | "
        f"health_only={args.health_only} | l4gm_init_sv4d={args.l4gm_init_sv4d}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
