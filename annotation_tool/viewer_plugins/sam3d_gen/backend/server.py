#!/usr/bin/env python3
"""HTTP server classes and CLI entry-point for the SAM3D backend."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

# Allow running as a standalone script (fix relative import issues).
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
TOOL_ROOT = ROOT.parents[2]
for p in (ROOT, PARENT, TOOL_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from core import BackendConfig, Sam3DBackend
from http_security import (
    DEFAULT_MAX_REQUEST_BYTES,
    RequestError,
    parse_allowed_roots,
    read_json_body,
    request_is_authorized,
    require_secure_bind,
    validate_payload_paths,
)


def health_payload() -> Dict[str, str]:
    return {"status": "ok", "service": "sam3d_backend"}


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        def _json_default(obj: Any) -> Any:
            # Be robust in /status responses even if we accidentally store numpy scalars, Paths, etc.
            try:
                if isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except Exception:
                pass
            return str(obj)

        body = json.dumps(payload, default=_json_default).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-response (common when frontend polls /status while GPU is busy).
            return

    def _authorize(self) -> bool:
        if request_is_authorized(self.headers, self.server.api_token):
            return True
        self._send_json(401, {"ok": False, "error": "unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(200, health_payload())
            return
        if not self._authorize():
            return
        if self.path.rstrip("/") == "/status":
            try:
                payload = self.server.backend.status()
                self._send_json(200, {"ok": True, "status": payload})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                traceback.print_exc()
                self._send_json(
                    500,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: status failed; see server logs",
                    },
                )
            return
        self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        t0 = time.time()
        if not self._authorize():
            return
        try:
            payload = read_json_body(self, self.server.max_request_bytes)
            validate_payload_paths(payload, self.server.allowed_roots)
            path = self.path.rstrip("/")
            if path == "/sam/set_image":
                out = self.server.backend.sam_set_image(
                    image_path=str(payload["image_path"]),
                    flip_udlr=bool(payload.get("flip_udlr", False)),
                )
                self._send_json(200, {"ok": True, **out, "dt_s": time.time() - t0})
                return
            if path == "/sam/predict":
                out = self.server.backend.sam_predict(
                    image_path=str(payload["image_path"]),
                    flip_udlr=bool(payload.get("flip_udlr", False)),
                    points_xy=payload.get("points_xy", []),
                    point_labels=payload.get("point_labels", []),
                    prev_logits_key=payload.get("prev_logits_key", None),
                    output_dir=payload.get("output_dir", None),
                    session_id=payload.get("session_id", None),
                    object_id=payload.get("object_id", None),
                )
                self._send_json(200, {"ok": True, **out, "dt_s": time.time() - t0})
                return
            if path == "/sam3d/generate":
                out = self.server.backend.sam3d_generate(
                    image_path=str(payload["image_path"]),
                    flip_udlr=bool(payload.get("flip_udlr", False)),
                    mask_path=str(payload["mask_path"]),
                    output_dir=str(payload["output_dir"]),
                    session_id=payload.get("session_id", None),
                    object_id=int(payload.get("object_id", 0)),
                    seed=payload.get("seed", None),
                    align_pytorch3d_camera_to_viser=payload.get("align_pytorch3d_camera_to_viser", None),
                    align_mode=payload.get("align_mode", None),
                )
                self._send_json(200, {"ok": True, **out, "dt_s": time.time() - t0})
                return
            if path == "/sam3d/pose_opt":
                out = self.server.backend.sam3d_pose_opt(
                    image_path=str(payload["image_path"]),
                    flip_udlr=bool(payload.get("flip_udlr", False)),
                    mask_path=str(payload["mask_path"]),
                    output_dir=str(payload["output_dir"]),
                    session_id=payload.get("session_id", None),
                    object_id=int(payload.get("object_id", 0)),
                    steps=int(payload.get("steps", 160)),
                    lr=float(payload.get("lr", 5e-3)),
                    min_side=int(payload.get("min_side", 518)),
                    rgb_weight=float(payload.get("rgb_weight", 0.0)),
                    debug_mode=bool(payload.get("debug_mode", False)),
                    align_pytorch3d_camera_to_viser=payload.get("align_pytorch3d_camera_to_viser", None),
                    align_mode=payload.get("align_mode", None),
                    camera_fov_deg=payload.get("camera_fov_deg", None),
                    camera_wxyz=payload.get("camera_wxyz", None),
                    camera_position=payload.get("camera_position", None),
                    camera_look_at=payload.get("camera_look_at", None),
                    camera_up=payload.get("camera_up", None),
                    lock_translation_x=payload.get("lock_translation_x", None),
                    lock_translation_y=payload.get("lock_translation_y", None),
                    lock_translation_z=payload.get("lock_translation_z", None),
                    optimize_rotation=payload.get("optimize_rotation", False),
                    target_center_world=payload.get("target_center_world", None),
                    lock_anchor_local=payload.get("lock_anchor_local", None),
                )
                self._send_json(200, {"ok": True, **out, "dt_s": time.time() - t0})
                return
            self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})
        except RequestError as exc:
            self._send_json(exc.status, {"ok": False, "error": str(exc)})
        except Exception as exc:
            try:
                if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                    return
                print(f"[SAM3D backend] {type(exc).__name__}: {exc}", file=sys.stderr)
                traceback.print_exc()
            except Exception:
                pass
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: request failed; see server logs",
                },
            )


class _Server(ThreadingHTTPServer):
    def __init__(
        self,
        addr: Tuple[str, int],
        backend: Sam3DBackend | None,
        *,
        api_token: str = "",
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ):
        super().__init__(addr, _Handler)
        self.backend = backend
        self.api_token = str(api_token)
        self.max_request_bytes = int(max_request_bytes)
        self.allowed_roots = parse_allowed_roots(TOOL_ROOT.parent)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM + SAM3D backend for 4DGsEditor viser plugin.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--api-token", default=os.environ.get("HAT4D_API_TOKEN", ""))
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--sam-model-type", type=str, default="vit_h")
    parser.add_argument("--sam-device", type=str, default="cuda")
    parser.add_argument("--sam3d-pipeline-config", type=str, default=None)
    parser.add_argument("--sam3d-depth-model-path", type=str, default=None)
    parser.add_argument("--sam3d-compile", action="store_true")
    parser.add_argument("--sam3d-seed", type=int, default=42)
    parser.add_argument("--debug-save-inputs", action="store_true")
    parser.add_argument("--align-pytorch3d-camera-to-viser", action="store_true")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--one-dino", action="store_true")
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Serve health checks without loading SAM or SAM3D.",
    )
    parser.add_argument(
        "--init-on-start",
        action="store_true",
        help="Deprecated: backend always initializes SAM and SAM3D on start (in background).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    require_secure_bind(args.host, args.api_token)
    if bool(args.no_flash_attn):
        os.environ["SAM3D_NO_FLASH_ATTN"] = "1"
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    if bool(args.one_dino):
        os.environ["SAM3D_ONE_DINO"] = "1"

    cfg = BackendConfig(
        host=str(args.host),
        port=int(args.port),
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=str(args.sam_model_type),
        sam_device=str(args.sam_device),
        sam3d_pipeline_config=args.sam3d_pipeline_config,
        sam3d_depth_model_path=args.sam3d_depth_model_path,
        sam3d_compile=bool(args.sam3d_compile),
        sam3d_seed=int(args.sam3d_seed),
        debug_save_inputs=bool(args.debug_save_inputs),
        align_pytorch3d_camera_to_viser=bool(args.align_pytorch3d_camera_to_viser),
    )
    backend = None if args.health_only else Sam3DBackend(cfg)

    server = _Server(
        (cfg.host, cfg.port),
        backend,
        api_token=args.api_token,
        max_request_bytes=args.max_request_bytes,
    )
    print(f"SAM3D backend listening on http://{cfg.host}:{cfg.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
