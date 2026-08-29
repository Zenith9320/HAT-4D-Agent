#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime
from pathlib import Path

from viser_frontend import ROOT, ViewerApp
from viewer_plugins.vlm_assistor.headless import create_headless_client


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless Full 3D Gen pipeline runner (same logic as VLM button).")
    parser.add_argument("--gs-dir", type=Path, default=None, help="Directory containing GS PLY sequence.")
    parser.add_argument("--input-image-dir", type=Path, required=True, help="Input image directory.")
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
    parser.add_argument("--port", type=int, default=0, help="Viser port for internal runtime; 0 means auto free port.")

    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument(
        "--agent0-object-text-hint",
        "--object-text",
        dest="agent0_object_text_hint",
        type=str,
        default=None,
        help=(
            "Optional hint passed into Agent0. "
            "Default is empty so Agent0 infers objects from video frames."
        ),
    )
    parser.add_argument(
        "--agent0-relation-hint",
        "--relation-hint",
        dest="agent0_relation_hint",
        type=str,
        default=None,
        help=(
            "Optional relation hint passed into Agent0. "
            "Default is empty so Agent0 infers relations from video frames."
        ),
    )
    parser.add_argument("--disable-plugin", action="store_true")
    parser.add_argument("--no-notification-log", action="store_true")
    parser.add_argument(
        "--render-four-views",
        dest="render_four_views",
        action="store_true",
        default=True,
        help="After generation, render front/back/left/right views for generated objects.",
    )
    parser.add_argument(
        "--no-render-four-views",
        dest="render_four_views",
        action="store_false",
        help="Disable post-generation four-view rendering.",
    )
    parser.add_argument(
        "--render-view-size",
        type=int,
        default=512,
        help="Image size for post-generation four-view rendering.",
    )

    parser.add_argument(
        "--agent0-backend-mode",
        type=str,
        choices=["official_api", "local_openai_compatible"],
        default=None,
    )
    parser.add_argument("--agent0-max-frames", type=int, default=None)
    parser.add_argument("--agent0-render-size", type=int, default=None)
    parser.add_argument("--agent1-points-per-object", type=int, default=None)
    parser.add_argument("--agent1-max-retries", type=int, default=None)
    parser.add_argument("--agent1-mask-min-area-ratio", type=float, default=None)

    parser.add_argument("--vlm-local-base-url", type=str, default=None)
    parser.add_argument("--vlm-local-model", type=str, default=None)
    parser.add_argument("--vlm-local-video-url", type=str, default=None)
    return parser


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


def _set_if_not_none(obj, field: str, value) -> None:
    if value is None:
        return
    setattr(obj, field, value)


def _resolve_pipeline_hints(args: argparse.Namespace) -> tuple[str, str]:
    object_text = ""
    relation_hint = ""
    if args.agent0_object_text_hint is not None:
        object_text = str(args.agent0_object_text_hint).strip()
    if args.agent0_relation_hint is not None:
        relation_hint = str(args.agent0_relation_hint).strip()
    return object_text, relation_hint


def _safe_name(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return "object"
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "object"


def _collect_generated_objects(result: dict) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[int] = set()
    for item in list(result.get("items", []) or []):
        if not isinstance(item, dict):
            continue
        agent1 = dict(item.get("agent1", {}) or {})
        try:
            obj_idx = int(agent1.get("generated_object_idx", -1))
        except Exception:
            obj_idx = -1
        if obj_idx < 0 or obj_idx in seen:
            continue
        seen.add(obj_idx)
        out.append(
            {
                "object_idx": int(obj_idx),
                "target": str(item.get("target", f"obj_{obj_idx}")),
            }
        )
    return out


def _render_generated_four_views(plugin, client, result: dict, image_size: int) -> dict:
    registry = plugin.ops.bridge.registry
    render_root = (
        Path(plugin.state.artifact_root)
        / "headless_four_views"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    )
    render_root.mkdir(parents=True, exist_ok=True)

    targets = _collect_generated_objects(result)
    if not targets:
        listed = registry.call("object.list")
        if bool(listed.get("ok", False)):
            for obj in list(listed.get("objects", []) or []):
                if not isinstance(obj, dict):
                    continue
                if not bool(obj.get("has_splat", False)):
                    continue
                try:
                    idx = int(obj.get("object_idx", -1))
                except Exception:
                    idx = -1
                if idx < 0:
                    continue
                targets.append({"object_idx": int(idx), "target": f"obj_{idx}"})
    if not targets:
        return {
            "ok": False,
            "reason": "no_generated_objects_for_render",
            "save_root": str(render_root),
            "items": [],
        }

    items: list[dict[str, object]] = []
    for i, target in enumerate(targets):
        idx = int(target["object_idx"])
        target_name = str(target.get("target", f"obj_{idx}"))
        _ = registry.call("object.select", object_idx=int(idx), client=client, ensure_bbox=False)
        out = registry.call(
            "render.object_multiview",
            client=client,
            object_idx=int(idx),
            views=["front", "back", "left", "right"],
            image_size=int(max(64, int(image_size))),
            save_dir=str(render_root),
            prefix=f"{int(i + 1):02d}_{_safe_name(target_name)}",
        )
        items.append(
            {
                "object_idx": int(idx),
                "target": target_name,
                "ok": bool(out.get("ok", False)),
                "reason": str(out.get("reason", "")),
                "view_paths": dict(out.get("view_paths", {}) or {}),
            }
        )

    return {
        "ok": bool(all(bool(x.get("ok", False)) for x in items)),
        "save_root": str(render_root),
        "items": items,
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    gs_dir, eval_dir = _resolve_paths(args)
    port = int(args.port) if int(args.port) > 0 else _pick_free_port()

    object_text_hint, relation_hint_hint = _resolve_pipeline_hints(args)

    app = ViewerApp(
        gs_dir=gs_dir,
        input_dir=args.input_image_dir,
        eval_dir=eval_dir,
        workspace_root=args.workspace_root,
        resume_workspace=args.resume_workspace,
        backend_url=args.backend_url,
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
            client_id=f"headless_{port}",
            verbose=(not bool(args.no_notification_log)),
        )

        plugin = app.vlm_assistor_plugin
        with plugin.state.lock:
            plugin.state.artifact_root = Path(app.workspace.root) / "vlm_assistor"
            plugin.state.enabled = (not bool(args.disable_plugin))
            # Match UI button semantics: button reads current text-box values.
            # In headless mode we bind those values explicitly here.
            plugin.state.object_description = str(object_text_hint)
            plugin.state.relation_hint = str(relation_hint_hint)
            _set_if_not_none(plugin.state, "agent0_backend_mode", args.agent0_backend_mode)
            _set_if_not_none(plugin.state, "agent0_max_frames", args.agent0_max_frames)
            _set_if_not_none(plugin.state, "agent0_render_size", args.agent0_render_size)
            _set_if_not_none(plugin.state, "agent1_points_per_object", args.agent1_points_per_object)
            _set_if_not_none(plugin.state, "agent1_max_retries", args.agent1_max_retries)
            _set_if_not_none(plugin.state, "agent1_mask_min_area_ratio", args.agent1_mask_min_area_ratio)
            if args.vlm_local_base_url is not None:
                plugin.state.agent0_local_base_url = str(args.vlm_local_base_url).strip()
            if args.vlm_local_model is not None:
                plugin.state.agent0_local_model = str(args.vlm_local_model).strip()
            if args.vlm_local_video_url is not None:
                plugin.state.agent0_local_video_url = str(args.vlm_local_video_url).strip()

        app.vlm_agent0_backend_mode = str(getattr(plugin.state, "agent0_backend_mode", "official_api"))
        app.vlm_local_base_url = str(getattr(plugin.state, "agent0_local_base_url", ""))
        app.vlm_local_model = str(getattr(plugin.state, "agent0_local_model", ""))
        app.vlm_local_api_key = str(getattr(plugin.state, "agent0_local_api_key", ""))
        app.vlm_local_video_url = str(getattr(plugin.state, "agent0_local_video_url", ""))

        object_text = str(getattr(plugin.state, "object_description", "") or "")
        relation_hint = str(getattr(plugin.state, "relation_hint", "") or "")
        result = plugin.ops.headless_3d_gen(
            client,
            object_text=object_text,
            relation_hint=relation_hint,
            run_async=False,
        )
        if bool(args.render_four_views):
            result["headless_four_view_render"] = _render_generated_four_views(
                plugin,
                client,
                result=result,
                image_size=int(args.render_view_size),
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if bool(result.get("ok", False)) else 2
    finally:
        stop_fn = getattr(app.server, "stop", None)
        if not callable(stop_fn):
            stop_fn = getattr(app.server, "shutdown", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
