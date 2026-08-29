from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def sanitize_points(points: np.ndarray, max_abs: float = 1e6) -> np.ndarray:
    pts = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pts, -max_abs, max_abs).astype(np.float32, copy=False)


def sanitize_rotation(rotation: np.ndarray) -> np.ndarray:
    rot = np.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0)
    return rot.astype(np.float32, copy=False)


def compute_center_radius(points: np.ndarray) -> Tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).max())
    return center, max(radius, 1e-3)


def solve_depth_for_bbox(
    points_world: np.ndarray,
    rotation_wc: np.ndarray,
    center_world: np.ndarray,
    bbox: Tuple[int, int, int, int],
    fx: float,
    fy: float,
) -> float:
    pts_centered = sanitize_points(points_world - center_world[None, :])
    rot = sanitize_rotation(rotation_wc)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        pts_cam = (rot @ pts_centered.T).T
    obj_half_w = float(np.max(np.abs(pts_cam[:, 0])) + 1e-6)
    obj_half_h = float(np.max(np.abs(pts_cam[:, 1])) + 1e-6)
    x0, y0, x1, y1 = bbox
    gt_half_w = max((x1 - x0) / 2.0, 1e-3)
    gt_half_h = max((y1 - y0) / 2.0, 1e-3)
    depth_w = fx * obj_half_w / gt_half_w
    depth_h = fy * obj_half_h / gt_half_h
    return float(max(depth_w, depth_h))


def project_bbox_size(
    points_world: np.ndarray,
    rotation_wc: np.ndarray,
    eye: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """Project 3D points to image and return bbox width/height in pixels."""
    pts_cam = (sanitize_points(points_world) - sanitize_points(eye)[None, :]) @ sanitize_rotation(
        rotation_wc
    ).T
    z = pts_cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return 0.0, 0.0
    fx = width / (2 * math.tan(math.radians(fov_deg) * 0.5))
    fy = height / (2 * math.tan(math.radians(fov_deg) * 0.5))
    u = fx * (pts_cam[valid, 0] / z[valid]) + width * 0.5
    v = fy * (pts_cam[valid, 1] / z[valid]) + height * 0.5
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    return max(u_max - u_min, 0.0), max(v_max - v_min, 0.0)

import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_gaussians(
    centers: np.ndarray,     # (N,3)
    rotations: np.ndarray,   # (N,4) wxyz
    scales: np.ndarray,      # (N,3)
    A: np.ndarray,           # (3,3)
    eps: float = 1e-8,
    return_cov: bool = True,
):
    """
    Stable Gaussian transform via covariance.

    Build:     Σ = R diag(s^2) R^T
    Transform: x' = A x
               Σ' = A Σ A^T

    Then (optionally) decompose Σ' back to (R', s') via eigh:
        Σ' = V diag(λ) V^T,  s' = sqrt(λ),  R' = V

    Returns:
        new_centers:   (N,3)
        new_rotations: (N,4) wxyz
        new_scales:    (N,3)  (>=0)
        Sigma_new:     (N,3,3)  (if return_cov=True)
    """
    centers = centers.astype(np.float32)
    rotations = rotations.astype(np.float32)
    scales = scales.astype(np.float32)
    A = A.astype(np.float32)

    # --- centers ---
    new_centers = centers @ A.T  # row-vector convention

    # --- quat -> R ---
    quats_xyzw = rotations[:, [1, 2, 3, 0]]
    Rm = R.from_quat(quats_xyzw).as_matrix().astype(np.float32)  # (N,3,3)

    # --- Σ = R diag(s^2) R^T ---
    S2 = np.square(scales).astype(np.float32)                   # (N,3)
    Sigma = (Rm * S2[:, None, :]) @ Rm.transpose(0, 2, 1)       # (N,3,3)

    # --- Σ' = A Σ A^T ---
    Sigma_new = A[None] @ Sigma @ A.T[None]

    # symmetrize for numerical stability
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.transpose(0, 2, 1))

    # keep SPD (prevents degenerate screen ellipses)
    eigvals = np.linalg.eigvalsh(Sigma_new)
    min_eig = eigvals.min()
    if min_eig < eps:
        Sigma_new = Sigma_new + np.eye(3, dtype=np.float32)[None] * (eps - min_eig)

    # --- decompose Σ' back to (R', s') ---
    # eigh returns eigenvalues in ascending order; that’s fine.
    lam, V = np.linalg.eigh(Sigma_new)  # lam: (N,3), V: (N,3,3)
    lam = np.clip(lam, eps, None).astype(np.float32)
    new_scales = np.sqrt(lam).astype(np.float32)               # (N,3) >= 0
    R_new = V.astype(np.float32)

    # fix improper rotation (det < 0): flip one axis of V
    det = np.linalg.det(R_new)
    mask = det < 0
    if np.any(mask):
        R_new[mask, :, -1] *= -1

    # R -> quat (xyzw -> wxyz)
    quats_new_xyzw = R.from_matrix(R_new).as_quat().astype(np.float32)
    new_rotations = quats_new_xyzw[:, [3, 0, 1, 2]]

    if return_cov:
        return new_centers, new_rotations, new_scales, Sigma_new
    else:
        return new_centers, new_rotations, new_scales