#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BUILD_ID = "sam3d_gen_backend_2026-01-31"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sam3d_repo_root() -> Path:
    return Path(__file__).resolve().parents[1] / "thirdparty" / "sam-3d-objects"


def _ensure_sam3d_import_path() -> None:
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    if "CUDA_HOME" not in os.environ and "CONDA_PREFIX" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
    # Make Hydra show chained exceptions.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    root = _sam3d_repo_root()
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    seg_any = root / "thirdparty" / "segment-anything"
    if seg_any.exists() and str(seg_any) not in sys.path:
        sys.path.insert(0, str(seg_any))


def _load_rgb_u8(image_path: Path, *, flip_udlr: bool) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    if flip_udlr:
        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(img, dtype=np.uint8)


def _load_mask_bool(mask_path: Path, *, flip_udlr: bool) -> np.ndarray:
    img = Image.open(mask_path)
    if flip_udlr:
        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    m = np.array(img)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(bool)


def _write_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)


def _align_fix_from_mode(mode: str | None, device, dtype):
    """
    Match sam3d_objects_viser_app.py:
    PyTorch3D camera frame (x-left, y-up, z-forward) -> Viser frustum camera
    (x-right, y-down, z-forward) by a 180° rotation about camera Z.

    Note: `mode` is accepted for API compatibility, but ignored by design.
    """
    R_fix = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    q_fix = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 4)
    _ = mode
    return R_fix, q_fix


def _save_debug_inputs(out_dir: Path, *, rgb: np.ndarray, mask: np.ndarray) -> None:
    dbg_dir = out_dir / "sam3d_gen" / "debug_inputs"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(rgb.astype(np.uint8)).save(dbg_dir / "input_rgb.png")
    except Exception:
        pass
    try:
        Image.fromarray((mask.astype(np.uint8) * 255)).save(dbg_dir / "input_mask.png")
    except Exception:
        pass
    try:
        rgba = np.concatenate([rgb.astype(np.uint8), (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
        Image.fromarray(rgba).save(dbg_dir / "input_rgba.png")
    except Exception:
        pass


def _y_up_mesh_to_z_up(verts_y: np.ndarray) -> np.ndarray:
    # See sam3d_objects_viser_app.py: GLB export applies Z-up -> Y-up via M_z2y,
    # so invert by right-multiplying M_y2z.
    M_y2z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    return np.asarray(verts_y, dtype=np.float32) @ M_y2z
