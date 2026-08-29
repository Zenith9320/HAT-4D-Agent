#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

from viser_frontend import ROOT, ViewerApp
from viewer_plugins.vlm_assistor.headless import create_headless_client


_RESULT_BEGIN = "__HEADLESS4D_RESULT_BEGIN__"
_RESULT_END = "__HEADLESS4D_RESULT_END__"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _set_if_not_none(obj: Any, field: str, value: Any) -> None:
    if value is None:
        return
    setattr(obj, field, value)


def _str_or_empty(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _resolve_backend_url(raw: Any) -> str:
    direct = _str_or_empty(raw)
    if direct:
        return direct
    for key in ("BACKEND_URL", "ROI_BACKEND_URL", "VLM_BACKEND_URL"):
        val = _str_or_empty(os.getenv(key))
        if val:
            return val
    return ""


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    eval_dir = args.eval_gt_dir if args.eval_gt_dir is not None else args.input_image_dir
    if args.gs_dir is not None:
        gs_dir = args.gs_dir
    elif args.resume_workspace is not None:
        gs_dir = args.resume_workspace / "gs_sequence"
    elif bool(args.start_empty):
        gs_dir = args.workspace_root / "_headless_empty_gs"
        gs_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("Missing --gs-dir. If you want an empty start, pass --start-empty.")
    return gs_dir, eval_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless Full 4D Gen pipeline runner (same logic as 'Run Full 4D Gen Pipeline')."
    )

    # Single-task mode arguments.
    parser.add_argument("--gs-dir", type=Path, default=None, help="Directory containing GS PLY sequence.")
    parser.add_argument("--input-image-dir", type=Path, default=None, help="Input image directory.")
    parser.add_argument("--eval-gt-dir", type=Path, default=None, help="Eval GT root; defaults to input dir.")
    parser.add_argument("--workspace-root", type=Path, default=ROOT / "workspace", help="Workspace root.")
    parser.add_argument("--resume-workspace", type=Path, default=None, help="Reuse an existing workspace.")
    parser.add_argument("--backend-url", type=str, default=None)
    parser.add_argument("--sam3d-backend-url", type=str, default=None)
    parser.add_argument("--vlm-agent-url", type=str, default=None)
    parser.add_argument("--vlm-shared-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=49.1)
    parser.add_argument("--up-axis", type=str, choices=["y", "z", "-y"], default="z")
    parser.add_argument("--flip-colmap", action="store_true")
    parser.add_argument("--camera-target-origin", action="store_true")
    parser.add_argument("--start-empty", action="store_true")
    parser.add_argument("--port", type=int, default=0, help="Internal viser runtime port; 0 means auto free port.")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--disable-plugin", action="store_true")
    parser.add_argument("--no-notification-log", action="store_true")
    parser.add_argument(
        "--agent0-object-text-hint",
        "--object-text",
        dest="object_text",
        type=str,
        default="",
    )
    parser.add_argument(
        "--agent0-relation-hint",
        "--relation-hint",
        dest="relation_hint",
        type=str,
        default="",
    )
    parser.add_argument("--agent0-backend-mode", type=str, choices=["official_api", "local_openai_compatible"], default=None)
    parser.add_argument("--agent0-max-frames", type=int, default=None)
    parser.add_argument("--agent0-render-size", type=int, default=None)
    parser.add_argument("--agent1-points-per-object", type=int, default=None)
    parser.add_argument("--agent1-max-retries", type=int, default=None)
    parser.add_argument("--agent1-mask-min-area-ratio", type=float, default=None)
    parser.add_argument("--agent4-history-window", type=int, default=None)
    parser.add_argument("--agent4-memory-bank-capacity", type=int, default=None)
    parser.add_argument("--vlm-local-base-url", type=str, default=None)
    parser.add_argument("--vlm-local-model", type=str, default=None)
    parser.add_argument("--vlm-local-video-url", type=str, default=None)

    # Legacy compatibility flags used by existing scripts.
    parser.add_argument("--segment-frames", type=int, default=None, help="Deprecated: currently fixed by pipeline internals.")
    parser.add_argument("--init-strategy", type=str, default=None, help="Deprecated compatibility flag.")
    parser.add_argument("--on-init-fail", type=str, default=None, help="Deprecated compatibility flag.")
    parser.add_argument("--agent-point-backend-mode", type=str, default=None, help="Deprecated compatibility flag.")
    parser.add_argument("--sam3d-pose-opt", action="store_true", help="Deprecated compatibility flag.")
    parser.add_argument("--sam-max-retries", type=int, default=None, help="Deprecated compatibility flag.")
    parser.add_argument("--mask-max-area-ratio", type=float, default=None, help="Deprecated compatibility flag.")
    parser.add_argument(
        "--mask-min-positive-coverage",
        type=float,
        default=None,
        help="Deprecated compatibility flag.",
    )
    parser.add_argument(
        "--mask-min-negative-outside-coverage",
        type=float,
        default=None,
        help="Deprecated compatibility flag.",
    )
    parser.add_argument("--sam-flip-udlr", action="store_true", help="Deprecated compatibility flag.")

    return parser


def _apply_agent_state(args: argparse.Namespace, app: ViewerApp) -> None:
    plugin = app.vlm_assistor_plugin
    with plugin.state.lock:
        plugin.state.artifact_root = Path(app.workspace.root) / "vlm_assistor"
        plugin.state.enabled = (not bool(args.disable_plugin))
        plugin.state.object_description = str(_str_or_empty(args.object_text))
        plugin.state.relation_hint = str(_str_or_empty(args.relation_hint))
        _set_if_not_none(plugin.state, "agent0_backend_mode", args.agent0_backend_mode)
        _set_if_not_none(plugin.state, "agent0_max_frames", args.agent0_max_frames)
        _set_if_not_none(plugin.state, "agent0_render_size", args.agent0_render_size)
        _set_if_not_none(plugin.state, "agent1_points_per_object", args.agent1_points_per_object)
        _set_if_not_none(plugin.state, "agent1_max_retries", args.agent1_max_retries)
        _set_if_not_none(plugin.state, "agent1_mask_min_area_ratio", args.agent1_mask_min_area_ratio)
        _set_if_not_none(plugin.state, "agent4_history_window", args.agent4_history_window)
        _set_if_not_none(plugin.state, "agent4_memory_bank_capacity", args.agent4_memory_bank_capacity)
        if args.vlm_local_base_url is not None:
            plugin.state.agent0_local_base_url = str(args.vlm_local_base_url).strip()
        if args.vlm_local_model is not None:
            plugin.state.agent0_local_model = str(args.vlm_local_model).strip()
        if args.vlm_local_video_url is not None:
            plugin.state.agent0_local_video_url = str(args.vlm_local_video_url).strip()

    # keep app-side mirrors aligned with plugin state
    app.vlm_agent0_backend_mode = str(getattr(plugin.state, "agent0_backend_mode", "official_api"))
    app.vlm_local_base_url = str(getattr(plugin.state, "agent0_local_base_url", ""))
    app.vlm_local_model = str(getattr(plugin.state, "agent0_local_model", ""))
    app.vlm_local_api_key = str(getattr(plugin.state, "agent0_local_api_key", ""))
    app.vlm_local_video_url = str(getattr(plugin.state, "agent0_local_video_url", ""))


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_image_dir is None:
        raise ValueError("Missing --input-image-dir")
    backend_url = _resolve_backend_url(args.backend_url)
    if not backend_url:
        raise ValueError(
            "Missing backend URL. Full 4D pipeline requires ROI/L4GM backend. "
            "Pass --backend-url (e.g. http://localhost:8002/select_roi) "
            "or set env BACKEND_URL/ROI_BACKEND_URL/VLM_BACKEND_URL."
        )
    gs_dir, eval_dir = _resolve_paths(args)
    port = int(args.port) if int(args.port) > 0 else _pick_free_port()
    app = ViewerApp(
        gs_dir=gs_dir,
        input_dir=args.input_image_dir,
        eval_dir=eval_dir,
        workspace_root=args.workspace_root,
        resume_workspace=args.resume_workspace,
        backend_url=backend_url,
        sam3d_backend_url=args.sam3d_backend_url,
        vlm_agent_url=args.vlm_agent_url,
        vlm_shared_dir=args.vlm_shared_dir,
        mvtap_url=None,
        image_size=int(args.image_size),
        fov_deg=float(args.fov_deg),
        up_axis=str(args.up_axis),
        flip_colmap=bool(args.flip_colmap),
        camera_target_origin=bool(args.camera_target_origin),
        start_empty=bool(args.start_empty),
        metric_online=False,
        no_flash_atten=False,
        server_port=int(port),
    )

    try:
        total_frames = int(len(app.sequence.frame_tokens))
        if total_frames <= 0:
            raise RuntimeError("No frame tokens found in input image dir.")
        app.current_frame = max(0, min(int(args.frame_idx), total_frames - 1))
        client = create_headless_client(
            app,
            client_id=f"headless4d_{port}",
            verbose=(not bool(args.no_notification_log)),
        )
        _apply_agent_state(args, app)
        plugin = app.vlm_assistor_plugin
        object_text = str(getattr(plugin.state, "object_description", "") or "")
        relation_hint = str(getattr(plugin.state, "relation_hint", "") or "")
        result = plugin.ops.run_full_4d_pipeline_sync(
            client,
            object_text=object_text,
            relation_hint=relation_hint,
        )
        packed = {
            "ok": bool(result.get("ok", False)),
            "result": result,
            "workspace_root": str(app.workspace.root),
            "session_id": str(getattr(app, "session_id", "") or ""),
            "frame_tokens": [str(x) for x in list(getattr(app.sequence, "frame_tokens", []) or [])],
            "input_image_dir": str(args.input_image_dir),
            "backend_url": str(backend_url),
            "object_text": object_text,
            "relation_hint": relation_hint,
        }
        save_dir = Path(app.workspace.root) / "vlm_assistor"
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_headless_4dgen_result.json"
        out_path.write_text(json.dumps(packed, ensure_ascii=False, indent=2), encoding="utf-8")
        packed["result_path"] = str(out_path)
        return packed
    finally:
        stop_fn = getattr(app.server, "stop", None)
        if not callable(stop_fn):
            stop_fn = getattr(app.server, "shutdown", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    out = _run_single(args)
    print(_RESULT_BEGIN)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(_RESULT_END)
    return 0 if bool(out.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
