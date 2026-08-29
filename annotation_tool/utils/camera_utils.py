from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import viser.transforms as tf


def up_vector(up_axis: str = "y") -> np.ndarray:
    if up_axis == "y":
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if up_axis == "-y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def direction_from_angles(az_deg: float, el_deg: float, up_axis: str = "y") -> np.ndarray:
    az = np.deg2rad(float(az_deg))
    el = np.deg2rad(float(el_deg))
    if up_axis == "y":
        return np.array(
            [
                np.cos(el) * np.cos(az),
                np.sin(el),
                np.cos(el) * np.sin(az),
            ],
            dtype=np.float32,
        )
    if up_axis == "-y":
        return np.array(
            [
                np.cos(el) * np.cos(az),
                -np.sin(el),
                np.cos(el) * np.sin(az),
            ],
            dtype=np.float32,
        )
    return np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float32,
    ) # "z"


def direction_for_view(
    view: str,
    view_presets: Dict[str, Tuple[float, float]],
    up_axis: str = "y",
) -> np.ndarray:
    az_deg, el_deg = view_presets.get(view, (0.0, 0.0))
    return direction_from_angles(az_deg, el_deg, up_axis)


def eye_wxyz_from_angles(
    az_deg: float,
    el_deg: float,
    distance: float,
    center: np.ndarray,
    up_axis: str = "y",
) -> Tuple[np.ndarray, np.ndarray]:
    direction = direction_from_angles(az_deg, el_deg, up_axis)
    eye = np.asarray(center, dtype=np.float32) + direction * float(distance)
    up = up_vector(up_axis)
    z = np.asarray(center, dtype=np.float32) - eye
    z = z / (np.linalg.norm(z) + 1e-8)
    y = up - np.dot(up, z) * z
    y = y / (np.linalg.norm(y) + 1e-8)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    wxyz = tf.SO3.from_matrix(R).wxyz.astype(np.float32)
    return eye.astype(np.float32), wxyz


def eye_wxyz_for_view(
    view: str,
    distance: float,
    center: np.ndarray,
    view_presets: Dict[str, Tuple[float, float]],
    up_axis: str = "y",
) -> Tuple[np.ndarray, np.ndarray]:
    az_deg, el_deg = view_presets.get(view, (0.0, 0.0))
    return eye_wxyz_from_angles(az_deg, el_deg, distance, center, up_axis)


def vec_to_spherical(vec: np.ndarray) -> Tuple[float, float, float]:
    r = float(np.linalg.norm(vec))
    if r < 1e-6:
        return 0.0, 0.0, 0.0
    yaw = np.arctan2(vec[2], vec[0])
    pitch = np.arcsin(np.clip(vec[1] / r, -1.0, 1.0))
    return r, float(yaw), float(pitch)


def spherical_to_vec(radius: float, yaw: float, pitch: float) -> np.ndarray:
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([radius * cp * cy, radius * sp, radius * cp * sy], dtype=np.float32)


def interp_angle(a0: float, a1: float, t: float) -> float:
    delta = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
    return float(a0 + t * delta)


def look_at_rotation(
    eye_vec: np.ndarray, center_target: np.ndarray, up_axis: str = "y"
) -> Tuple[np.ndarray, np.ndarray]:
    up = up_vector(up_axis)
    z = np.asarray(center_target, dtype=np.float32) - np.asarray(eye_vec, dtype=np.float32)
    z = z / (np.linalg.norm(z) + 1e-8)
    y = up - np.dot(up, z) * z
    y = y / (np.linalg.norm(y) + 1e-8)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    so3 = tf.SO3.from_matrix(R.astype(np.float32))
    return so3.as_matrix().astype(np.float32), so3.wxyz.astype(np.float32)
