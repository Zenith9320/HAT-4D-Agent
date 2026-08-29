"""Backend communication mixin for SAM3D generation operations.

Handles all HTTP calls to the SAM / SAM3D backend server, status polling,
image upload, and predict requests.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


class BackendClientMixin:
    """Mixin providing backend HTTP communication helpers.

    Expects the host class to expose:
        - self.state          (Sam3DGenState)
        - self._app()         (application reference)
        - self._set_sam_status(s)
        - self._set_sam3d_status(s)
    """

    def _backend_enabled(self) -> bool:
        try:
            return bool(getattr(self._app(), "sam3d_backend_url", None))
        except Exception:
            return False

    # Backend calls
    def _backend_fetch_status(self, client, *, timeout: float = 5.0) -> Optional[dict]:
        app = self._app()
        try:
            base = app._sam3d_backend_base_url()
        except Exception as exc:
            return {"sam_status": "backend_url_missing", "sam3d_status": "backend_url_missing", "error": str(exc)}
        try:
            return app._get_json(base + "/status", timeout=timeout)
        except Exception as exc:
            return {"sam_status": "backend_unreachable", "sam3d_status": "backend_unreachable", "error": str(exc)}

    def _backend_apply_status(self, client, status: dict, *, notify: bool) -> None:
        sam_status = str(status.get("sam_status", "unknown"))
        sam3d_status = str(status.get("sam3d_status", "unknown"))
        sam_err = status.get("sam_error") or status.get("error")
        sam3d_err = status.get("sam3d_error") or status.get("error")

        if sam_status == "failed" and sam_err:
            self._set_sam_status(f"failed: {sam_err}")
        else:
            self._set_sam_status(sam_status)
        if sam3d_status == "failed" and sam3d_err:
            self._set_sam3d_status(f"failed: {sam3d_err}")
        else:
            self._set_sam3d_status(sam3d_status)

        if not notify:
            return

        # SAM notifications
        if sam_status in ("uninitialized", "initializing") and not self.state._sam_init_notified:
            self.state._sam_init_notified = True
            client.add_notification(
                "SAM3D",
                f"SAM backend status: {sam_status} (may take a while on first run)",
                color="orange",
                auto_close_seconds=6,
            )
        if sam_status == "ready" and not self.state._sam_ready_notified:
            self.state._sam_ready_notified = True
            client.add_notification(
                "SAM3D",
                "SAM backend is ready.",
                color="green",
                auto_close_seconds=3,
            )
        if sam_status == "failed" and not self.state._sam_failed_notified:
            self.state._sam_failed_notified = True
            client.add_notification(
                "SAM3D",
                f"SAM backend failed: {sam_err}",
                color="red",
                auto_close_seconds=10,
            )

        # SAM3D notifications
        if sam3d_status in ("uninitialized", "initializing") and not self.state._sam3d_init_notified:
            self.state._sam3d_init_notified = True
            client.add_notification(
                "SAM3D",
                f"SAM3D backend status: {sam3d_status} (first init can take minutes)",
                color="orange",
                auto_close_seconds=6,
            )
        if sam3d_status == "ready" and not self.state._sam3d_ready_notified:
            self.state._sam3d_ready_notified = True
            client.add_notification(
                "SAM3D",
                "SAM3D backend is ready.",
                color="green",
                auto_close_seconds=3,
            )
        if sam3d_status == "failed" and not self.state._sam3d_failed_notified:
            self.state._sam3d_failed_notified = True
            client.add_notification(
                "SAM3D",
                f"SAM3D backend failed: {sam3d_err}",
                color="red",
                auto_close_seconds=12,
            )

    def _backend_start_status_poll(self, client, *, interval_s: float = 2.0) -> None:
        cid = int(getattr(client, "client_id", 0))
        prev_evt = self.state.backend_poll_events.get(cid)
        if prev_evt is not None:
            prev_evt.set()
        evt = threading.Event()
        self.state.backend_poll_events[cid] = evt

        def _loop() -> None:
            # Poll until ready/failed (or client disconnect / stop event).
            while not evt.is_set():
                status = self._backend_fetch_status(client, timeout=3.0)
                if isinstance(status, dict):
                    self._backend_apply_status(client, status, notify=True)
                    sam = str(status.get("sam_status", ""))
                    sam3d = str(status.get("sam3d_status", ""))
                    if sam in ("ready", "failed") and sam3d in ("ready", "failed"):
                        return
                time.sleep(float(interval_s))

        th = threading.Thread(target=_loop, daemon=True)
        self.state.backend_poll_threads[cid] = th
        th.start()

    def _backend_set_image(self, client) -> None:
        app = self._app()
        try:
            base = app._sam3d_backend_base_url()
        except Exception as exc:
            self._set_sam_status("backend_url_missing")
            self._set_sam3d_status("backend_url_missing")
            client.add_notification(
                "SAM3D",
                f"sam3d_backend_url not set: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            return

        frame_token = app.sequence.frame_tokens[app.current_frame]
        img_path = app.workspace.image_path_for_view(frame_token, "input")
        try:
            with self.state.lock:
                if bool(getattr(self.state, "pad_square", False)) and self.state.padded_image_path:
                    img_path = Path(self.state.padded_image_path)
        except Exception:
            pass
        payload = {
            "image_path": str(img_path),
            "flip_udlr": bool(getattr(self.state, "image_flip_udlr", False)),
        }
        # Update status + show init hints before the call (useful if backend still warming up).
        st = self._backend_fetch_status(client, timeout=2.0)
        if isinstance(st, dict):
            self._backend_apply_status(client, st, notify=True)
        try:
            resp = app._post_json(base + "/sam/set_image", payload, timeout=180)
        except Exception as exc:
            self._set_sam_status("backend_error")
            client.add_notification(
                "SAM3D",
                f"SAM backend set_image failed: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            return
        if resp.get("ok", True):
            st2 = self._backend_fetch_status(client, timeout=2.0)
            if isinstance(st2, dict):
                self._backend_apply_status(client, st2, notify=False)
            else:
                self._set_sam_status("backend_ready(image_set)")
        else:
            self._set_sam_status("backend_error")

    def _backend_sam_predict(self, client, points: np.ndarray, labels: np.ndarray, prev_logits_key: str | None):
        app = self._app()
        try:
            base = app._sam3d_backend_base_url()
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"sam3d_backend_url not set: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            return None
        st = self._backend_fetch_status(client, timeout=2.0)
        if isinstance(st, dict):
            self._backend_apply_status(client, st, notify=True)

        frame_token = app.sequence.frame_tokens[app.current_frame]
        img_path = app.workspace.image_path_for_view(frame_token, "input")
        try:
            with self.state.lock:
                if bool(getattr(self.state, "pad_square", False)) and self.state.padded_image_path:
                    img_path = Path(self.state.padded_image_path)
        except Exception:
            pass
        out_dir = app.workspace.output_dir_for(frame_token, "input")
        sid = str(app.workspace.root.name) + "_" + str(frame_token)
        with self.state.lock:
            obj_idx = int(self.state.current_object_idx)

        payload = {
            "image_path": str(img_path),
            "flip_udlr": bool(getattr(self.state, "image_flip_udlr", False)),
            "points_xy": points.astype(int).tolist(),
            "point_labels": labels.astype(int).tolist(),
            "prev_logits_key": prev_logits_key,
            "output_dir": str(out_dir),
            "session_id": sid,
            "object_id": int(obj_idx),
        }
        try:
            resp = app._post_json(base + "/sam/predict", payload, timeout=180)
        except Exception as exc:
            client.add_notification(
                "SAM3D",
                f"SAM backend predict failed: {exc}",
                color="red",
                auto_close_seconds=10,
            )
            return None
        if not resp.get("ok", True) and "mask_path" not in resp:
            client.add_notification(
                "SAM3D",
                f"SAM backend error: {resp.get('error', 'unknown')}",
                color="red",
                auto_close_seconds=10,
            )
            return None
        return resp
