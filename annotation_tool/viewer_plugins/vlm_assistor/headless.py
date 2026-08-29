from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

CAMERA_ZNEAR = 0.05
CAMERA_ZFAR = 150.0


@dataclass
class _WorldAxes:
    visible: bool = True


@dataclass
class _Scene:
    world_axes: _WorldAxes = field(default_factory=_WorldAxes)


class _Camera:
    def __init__(
        self,
        *,
        position: np.ndarray,
        wxyz: np.ndarray,
        look_at: np.ndarray,
        up_direction: np.ndarray,
        fov: float,
        near: float,
        far: float,
    ) -> None:
        self.position = np.asarray(position, dtype=np.float32)
        self.wxyz = np.asarray(wxyz, dtype=np.float32)
        self.look_at = np.asarray(look_at, dtype=np.float32)
        self.up_direction = np.asarray(up_direction, dtype=np.float32)
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)
        self._on_update_callbacks: list[Callable[[Any], None]] = []

    def on_update(self, fn: Callable[[Any], None]) -> Callable[[Any], None]:
        self._on_update_callbacks.append(fn)
        return fn


class HeadlessClient:
    """A minimal client handle for running VLM/SAM3D pipeline without GUI."""

    def __init__(
        self,
        *,
        client_id: str,
        position: np.ndarray,
        wxyz: np.ndarray,
        look_at: np.ndarray,
        up_direction: np.ndarray,
        fov: float,
        near: float = CAMERA_ZNEAR,
        far: float = CAMERA_ZFAR,
        verbose: bool = True,
    ) -> None:
        self.client_id = str(client_id)
        self.camera = _Camera(
            position=np.asarray(position, dtype=np.float32),
            wxyz=np.asarray(wxyz, dtype=np.float32),
            look_at=np.asarray(look_at, dtype=np.float32),
            up_direction=np.asarray(up_direction, dtype=np.float32),
            fov=float(fov),
            near=float(near),
            far=float(far),
        )
        self.scene = _Scene()
        self.verbose = bool(verbose)
        self.notifications: list[dict[str, Any]] = []

    @contextmanager
    def atomic(self):
        yield

    def flush(self) -> None:
        return

    def add_notification(
        self,
        title: str,
        body: str,
        *,
        color: str = "blue",
        auto_close_seconds: float = 4.0,
    ) -> None:
        payload = {
            "title": str(title),
            "body": str(body),
            "color": str(color),
            "auto_close_seconds": float(auto_close_seconds),
        }
        self.notifications.append(payload)
        if self.verbose:
            print(f"[HeadlessNotification][{payload['color']}] {payload['title']}: {payload['body']}")


def create_headless_client(app, *, client_id: str = "headless", verbose: bool = True) -> HeadlessClient:
    eye, wxyz = app._eye_wxyz_for_view("input", 1.5)
    look_at = np.asarray(getattr(app, "camera_target", np.zeros(3, dtype=np.float32)), dtype=np.float32)
    up = np.asarray(getattr(app, "up_vec", [0.0, 1.0, 0.0]), dtype=np.float32)
    return HeadlessClient(
        client_id=str(client_id),
        position=np.asarray(eye, dtype=np.float32),
        wxyz=np.asarray(wxyz, dtype=np.float32),
        look_at=look_at,
        up_direction=up,
        fov=float(np.deg2rad(float(getattr(app, "fov_deg", 49.1)))),
        near=float(getattr(app, "camera_znear", CAMERA_ZNEAR)),
        far=float(getattr(app, "camera_zfar", CAMERA_ZFAR)),
        verbose=bool(verbose),
    )
