#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import imageio.v2 as iio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MVTAP_ROOT = ROOT / "MVTracker" / "MV-TAP"

torch.set_float32_matmul_precision("high")


def _load_rgb(path: Path) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


def _sorted_frames(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"frame directory not found: {dir_path}")
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not files:
        raise FileNotFoundError(f"no frame images in {dir_path}")
    digit_named = [p for p in files if p.stem.isdigit()]
    if digit_named:
        return sorted(digit_named, key=lambda p: int(p.stem))
    return sorted(files, key=lambda p: p.name)


def _sample_queries(
    bbox: Dict[str, int],
    frame_idx: int,
    num_points: int,
    render_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Sample a roughly uniform grid of query points inside the bbox.

    Returns an array of shape (N, 3) with (t, x, y) in pixel units of the target_size.
    """
    x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    render_w, render_h = render_size
    tgt_h, tgt_w = target_size

    nx = max(1, int(math.sqrt(num_points)))
    ny = max(1, int(math.ceil(num_points / nx)))
    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / max(nx, 1) * 0.5
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / max(ny, 1) * 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    pts = pts[:num_points]

    scale_x = tgt_w / float(render_w)
    scale_y = tgt_h / float(render_h)
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    t_col = np.full((pts.shape[0], 1), float(frame_idx), dtype=np.float32)
    return np.concatenate([t_col, pts.astype(np.float32)], axis=1)


def _draw_tracks(
    base: np.ndarray,
    coords: np.ndarray,
    vis: np.ndarray,
    color: Tuple[int, int, int] = (20, 220, 60),
) -> np.ndarray:
    """Overlay polyline tracks on an image."""
    overlay = base.copy()
    t_len, n_pts, _ = coords.shape
    for pid in range(n_pts):
        pts = []
        for t in range(t_len):
            if vis[t, pid] <= 0.5:
                continue
            x, y = coords[t, pid]
            pts.append((int(round(x)), int(round(y))))
        if len(pts) >= 2:
            cv2.polylines(overlay, [np.asarray(pts, dtype=np.int32)], isClosed=False, color=color, thickness=2)
        elif len(pts) == 1:
            cv2.circle(overlay, pts[0], radius=2, color=color, thickness=-1)
    return overlay


class MVTAPService:
    def __init__(self, ckpt_path: Path | None, device: str | None, disable_cam_embed: bool = True) -> None:
        if str(MVTAP_ROOT) not in sys.path:
            sys.path.append(str(MVTAP_ROOT))
        try:
            import hydra
            from omegaconf import OmegaConf
            from models.mvtap import MVTAP
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MV-TAP is optional and is not bundled. Install its implementation "
                "under annotation_tool/MVTracker/MV-TAP before starting this backend."
            ) from exc

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg_path = MVTAP_ROOT / "configs" / "model" / "mvtap.yaml"
        cfg = OmegaConf.load(cfg_path)
        if disable_cam_embed:
            cfg.model.use_cam_embed = False
            cfg.model.view_att = False
        self.model: Any = hydra.utils.instantiate(cfg.model)
        state_obj = torch.load(ckpt_path or (MVTAP_ROOT / "ckpt" / "MVTAP.ckpt"), map_location="cpu")
        state_dict = state_obj.get("state_dict", state_obj)
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k.replace("model.", "")] = v
            elif k.startswith("_forward_module.model."):
                cleaned[k.replace("_forward_module.model.", "")] = v
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[MVTAP] Missing keys: {missing}")
        if unexpected:
            print(f"[MVTAP] Unexpected keys: {unexpected}")
        self.model.to(self.device)
        self.model.eval()
        self.model_resolution = (self.model.model_resolution_H, self.model.model_resolution_W)
        print(f"[MVTAP] Model loaded on {self.device} with resolution {self.model_resolution}")

    def infer(self, video: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run MV-TAP inference.

        video: float32 array (1, V, T, 3, H, W)
        queries: float32 array (1, V, N, 3)
        """
        with torch.no_grad():
            video_t = torch.from_numpy(video).to(self.device)
            queries_t = torch.from_numpy(queries).to(self.device)
            coords, vis, conf, _ = self.model(video_t, queries_t, intrinsic=None, extrinsic=None, is_train=False)
            return coords.cpu().numpy(), vis.cpu().numpy(), conf.cpu().numpy()


def health_payload() -> Dict[str, str]:
    return {"status": "ok", "service": "mvtap_backend"}


class MVTAPRequestHandler(BaseHTTPRequestHandler):
    service: MVTAPService | None = None
    default_num_points: int = 128

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(health_payload())
            return
        self._send_json({"status": "error", "message": "Unsupported endpoint"}, code=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/track_roi", "/tracklet", "/mvtap"}:
            self._send_json(
                {"status": "error", "message": "Unsupported endpoint"},
                code=404,
            )
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json({"status": "error", "message": f"Invalid JSON: {exc}"}, code=400)
            return
        try:
            result = self._handle_track_request(payload)
        except Exception as exc:  # noqa: BLE001
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

    # --- helpers -----------------------------------------------------
    def _handle_track_request(self, payload: Dict) -> Dict:
        if self.service is None:
            raise RuntimeError("MV-TAP model service has not been initialized")
        bbox = payload.get("bbox_2d") or payload.get("bbox")
        if not bbox:
            raise ValueError("bbox_2d is required")
        frame_token = str(payload.get("frame_token") or f"{int(payload.get('frame_id', 0)):06d}")
        view = str(payload.get("camera_id", "input"))
        render_w, render_h = payload.get("image_size", [512, 512])
        num_points = int(payload.get("num_points", self.default_num_points))
        paths = payload.get("paths", {})
        workspace_root = Path(paths.get("workspace_root", ROOT / "workspace")).resolve()
        images_root = workspace_root / "images"

        if view == "input":
            view_dir = Path(paths.get("input_dir", images_root / "input"))
        else:
            eval_root = Path(paths.get("eval_gt_dir", images_root / "eval_gt"))
            view_dir = eval_root / view

        frame_paths = _sorted_frames(view_dir)
        frame_tokens = [p.stem for p in frame_paths]
        if frame_token not in frame_tokens:
            raise ValueError(f"frame_token {frame_token} not found in {view_dir}")
        frame_idx = frame_tokens.index(frame_token)

        raw_frames = [_load_rgb(p) for p in frame_paths]
        orig_h, orig_w = raw_frames[0].shape[:2]
        tgt_h, tgt_w = self.service.model_resolution
        resized = [cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA) for img in raw_frames]
        video_np = np.stack(resized, axis=0)  # T H W C
        video_np = video_np[None, None]  # 1 V T H W C
        video_np = np.transpose(video_np, (0, 1, 2, 5, 3, 4)).astype(np.float32)

        queries_np = _sample_queries(
            bbox=bbox,
            frame_idx=frame_idx,
            num_points=num_points,
            render_size=(render_w, render_h),
            target_size=(tgt_h, tgt_w),
        )
        queries_np = queries_np[None, None]  # 1 V N 3

        coords, vis, conf = self.service.infer(video_np, queries_np)
        coords = coords[0, 0]  # T N 2
        vis = vis[0, 0]
        conf = conf[0, 0]
        score = vis * conf

        scale_back_x = orig_w / float(tgt_w)
        scale_back_y = orig_h / float(tgt_h)
        coords[..., 0] *= scale_back_x
        coords[..., 1] *= scale_back_y

        track_dir = Path(paths.get("output_dir", workspace_root / "results"))
        track_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{frame_token}_{view}_tracklet"
        npy_path = track_dir / f"{stem}.npz"
        json_path = track_dir / f"{stem}.json"
        overlay_path = track_dir / f"{stem}_overlay.png"

        np.savez_compressed(
            npy_path,
            coords=coords,
            visibility=vis,
            confidence=conf,
            frame_tokens=np.array(frame_tokens),
            view=view,
            bbox=json.dumps(bbox),
        )

        track_json = {
            "status": "success",
            "view": view,
            "selected_frame": frame_token,
            "frames": frame_tokens,
            "bbox": bbox,
            "num_points": int(coords.shape[1]),
            "coords": coords.tolist(),
            "visibility": vis.tolist(),
            "confidence": conf.tolist(),
        }
        json_path.write_text(json.dumps(track_json, indent=2))

        overlay = _draw_tracks(raw_frames[frame_idx], coords, score)
        iio.imwrite(overlay_path, overlay.astype(np.uint8))

        return {
            "status": "success",
            "message": "Tracklet generated",
            "view": view,
            "frame_token": frame_token,
            "num_points": int(coords.shape[1]),
            "paths": {
                "npz": str(npy_path),
                "json": str(json_path),
                "overlay": str(overlay_path),
            },
        }

    def _send_json(self, payload: Dict, code: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MV-TAP backend for ROI-driven tracklets.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--ckpt", type=str, default=str(MVTAP_ROOT / "ckpt" / "MVTAP.ckpt"))
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or device id")
    parser.add_argument("--num-points", type=int, default=128, help="default points to sample inside bbox")
    parser.add_argument("--enable-cam-embed", action="store_true", help="keep camera embedding (requires intr/extr)")
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Serve health checks without loading the optional MV-TAP model.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    MVTAPRequestHandler.service = None
    if not args.health_only:
        MVTAPRequestHandler.service = MVTAPService(
            Path(args.ckpt),
            args.device,
            disable_cam_embed=not args.enable_cam_embed,
        )
    MVTAPRequestHandler.default_num_points = int(args.num_points)
    server = ThreadingHTTPServer((args.host, args.port), MVTAPRequestHandler)
    print(f"mvtap backend listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
