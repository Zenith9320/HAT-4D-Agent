from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .agents import (
    ClickAgent,
    FourDMemoryJudgeAgent,
    FourDValidatorAgent,
    Gen3DValidatorAgent,
    MemoryFrameSelector,
    PoseAdjusterAgent,
    PoseAdjusterAgentV2,
    VideoUnderstandingAgent,
)
from .operators import Sam3DOperatorBridge
from .vlm import VLMBackendClient
from .headless import create_headless_client


class VLMAssistorOps:
    def __init__(self, ctx, state):
        self.ctx = ctx
        self.state = state
        self.bridge = Sam3DOperatorBridge(ctx, state)
        self.vlm_client = VLMBackendClient(ctx.app)

        self.video_agent = VideoUnderstandingAgent(ctx, state, self.bridge, self.vlm_client)
        self.click_agent = ClickAgent(ctx, state, self.bridge, self.vlm_client)
        self.gen3d_validator = Gen3DValidatorAgent(ctx, state, self.bridge, self.vlm_client)
        self.pose_adjuster = PoseAdjusterAgent(ctx, state, self.bridge, self.vlm_client)
        self.pose_adjuster_v2 = PoseAdjusterAgentV2(ctx, state, self.bridge, self.vlm_client)
        self.fourd_validator = FourDValidatorAgent(ctx, state, self.bridge, self.vlm_client)
        self.fourd_memory_judge = FourDMemoryJudgeAgent(ctx, state, self.bridge, self.vlm_client)
        self.memory_selector = MemoryFrameSelector(ctx, state, self.bridge, self.vlm_client)

    def _set_status(self, text: str) -> None:
        if self.state.status_handle is not None:
            try:
                self.state.status_handle.value = str(text)
            except Exception:
                pass

    def _set_object_description(self, text: str) -> None:
        s = str(text or "")
        with self.state.lock:
            self.state.object_description = s
        handle = getattr(self.state, "object_description_handle", None)
        if handle is not None:
            try:
                handle.value = s
            except Exception:
                pass

    @staticmethod
    def _dedupe_names(names: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for x in names:
            s = str(x).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _scene_graph_object_names(self) -> list[str]:
        with self.state.lock:
            g = dict(self.state.scene_graph or {})
        to_en = getattr(self.vlm_client, "to_english_object_name", None)
        names: list[str] = []
        for x in g.get("objects", []):
            s = str(x).strip()
            if callable(to_en):
                try:
                    s = str(to_en(s)).strip()
                except Exception:
                    pass
            if s:
                names.append(s)
        for x in g.get("object_meta", []):
            if isinstance(x, dict):
                s = str(x.get("name", x.get("id", ""))).strip()
                if callable(to_en):
                    try:
                        s = str(to_en(s)).strip()
                    except Exception:
                        pass
                if s:
                    names.append(s)
        return self._dedupe_names(names)

    def _scene_graph_depth_order_near_to_far(self) -> list[str]:
        with self.state.lock:
            g = dict(self.state.scene_graph or {})
        depth_rows = list(g.get("x_depth_order_timeline", g.get("depth_order_timeline", [])) or [])
        if not depth_rows:
            return []

        app = getattr(self.ctx, "app", None)
        cur_idx = int(getattr(app, "current_frame", 0)) if app is not None else -1
        to_en = getattr(self.vlm_client, "to_english_object_name", None)

        def _normalize_order(raw: Any) -> list[str]:
            if isinstance(raw, str):
                arr = [x.strip() for x in raw.replace("->", ",").replace(">", ",").split(",") if x.strip()]
            elif isinstance(raw, list):
                arr = [str(x).strip() for x in raw if str(x).strip()]
            else:
                arr = []
            out: list[str] = []
            seen: set[str] = set()
            for x in arr:
                s = str(x).strip()
                if callable(to_en):
                    try:
                        s = str(to_en(s)).strip()
                    except Exception:
                        pass
                if not s:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
            return out

        def _entry_span(entry: dict[str, Any]) -> int:
            try:
                st = int(entry.get("start_frame_index", -1))
                ed = int(entry.get("end_frame_index", -1))
            except Exception:
                return 10**9
            if st < 0 or ed < 0 or ed < st:
                return 10**9
            return int(ed - st)

        def _covers_current(entry: dict[str, Any]) -> bool:
            if cur_idx < 0:
                return False
            try:
                st = int(entry.get("start_frame_index", -1))
                ed = int(entry.get("end_frame_index", -1))
            except Exception:
                return False
            if st < 0 or ed < 0 or ed < st:
                return False
            return bool(st <= cur_idx <= ed)

        candidates: list[dict[str, Any]] = []
        for row in depth_rows:
            if not isinstance(row, dict):
                continue
            order = _normalize_order(
                row.get("order_positive_to_negative", row.get("x_axis_positive_to_negative", []))
            )
            if not order:
                continue
            item = dict(row)
            item["_order"] = order
            candidates.append(item)
        if not candidates:
            return []

        covered = [x for x in candidates if _covers_current(x)]
        if covered:
            covered.sort(key=_entry_span)
            return list(covered[0].get("_order", []))
        return list(candidates[-1].get("_order", []))

    def _scene_graph_ordered_object_names(self) -> list[str]:
        names = self._scene_graph_object_names()
        if not names:
            return []
        depth_order = self._scene_graph_depth_order_near_to_far()
        if not depth_order:
            return names
        rank = {str(x).strip().lower(): i for i, x in enumerate(depth_order)}
        indexed = list(enumerate(names))
        indexed.sort(key=lambda x: (rank.get(str(x[1]).strip().lower(), 10**6), int(x[0])))
        return [x[1] for x in indexed]

    def _init_agent1_targets_from_scene_graph(self, *, reset_cursor: bool = True) -> list[str]:
        names = self._scene_graph_ordered_object_names()
        with self.state.lock:
            self.state.agent1_targets = list(names)
            if reset_cursor:
                self.state.agent1_target_index = 0
                self.state.agent1_generated_objects = []
                self.state.agent1_generated_object_indices = {}
        if names and reset_cursor:
            self._set_object_description(names[0])
        return names

    def _set_scene_graph_text(self) -> None:
        if self.state.scene_graph_handle is None:
            return
        g = dict(self.state.scene_graph or {})
        objs = g.get("objects", [])
        rels = g.get("relations", [])
        events = g.get("events", [])
        depth_timeline = g.get("x_depth_order_timeline", g.get("depth_order_timeline", []))
        lines = [
            f"objects: {objs}",
            f"relations: {len(rels)}",
            f"events: {len(events)}",
            f"x_depth_order_timeline: {len(depth_timeline) if isinstance(depth_timeline, list) else 0}",
        ]
        with self.state.lock:
            src = str(getattr(self.state, "agent0_last_source", ""))
            n_frames = int(getattr(self.state, "agent0_last_frame_count", 0))
        if src:
            lines.append(f"source: {src} ({n_frames} frames)")
        for r in rels[:8]:
            lines.append(
                f"- {r.get('subject')} --{r.get('predicate')}--> {r.get('obj')} ({r.get('distance_hint', 'unknown')})"
            )
        for e in events[:6]:
            lines.append(
                f"* event:{e.get('type')} {e.get('subject')}->{e.get('obj')} @ {e.get('frame_token', '')}"
            )
        if isinstance(depth_timeline, list):
            for d in depth_timeline[:4]:
                if not isinstance(d, dict):
                    continue
                fr = str(d.get("frame_range", "")).strip()
                order = list(d.get("order_positive_to_negative", d.get("x_axis_positive_to_negative", [])) or [])
                order = [str(x).strip() for x in order if str(x).strip()]
                if not order:
                    continue
                lines.append(f"# depth[{fr or '?'}] +X->-X: {' > '.join(order)}")
        try:
            self.state.scene_graph_handle.value = "\n".join(lines)
        except Exception:
            pass

    def _set_scene_graph_detail_text(self) -> None:
        handle = getattr(self.state, "scene_graph_detail_handle", None)
        if handle is None:
            return
        payload = dict(self.state.last_video_understanding or {})
        if not payload:
            payload = {"scene_graph": dict(self.state.scene_graph or {})}
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            text = str(payload)
        # Keep UI responsive when result gets very large.
        if len(text) > 24000:
            text = text[:24000] + "\n...<truncated>"
        try:
            handle.value = f"```json\n{text}\n```"
        except Exception:
            pass

    @staticmethod
    def _print_agent0_result(out: dict[str, Any]) -> None:
        try:
            text = json.dumps(out, ensure_ascii=False, indent=2)
        except Exception:
            text = str(out)
        print("[VLM Assistor][Agent0] Final structured understanding:")
        print(text)

    def _set_logs_text(self) -> None:
        if self.state.log_handle is None:
            return
        with self.state.lock:
            logs = list(self.state.logs)
        lines = [f"[{x.ts}] {x.agent} | {x.status} | {x.message}" for x in logs[-12:]]
        try:
            self.state.log_handle.value = "\n".join(lines) if lines else "(no logs)"
        except Exception:
            pass

    def _set_agent4_result_text(self) -> None:
        handle = getattr(self.state, "agent4_result_handle", None)
        if handle is None:
            return
        payload = dict(getattr(self.state, "last_4d_memory_result", {}) or {})
        if not payload:
            text = "agent4_result: (empty)"
        else:
            try:
                detail = json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception:
                detail = str(payload)
            if len(detail) > 12000:
                detail = detail[:12000] + "\n...<truncated>"
            text = f"```json\n{detail}\n```"
        try:
            handle.value = text
        except Exception:
            pass

    @staticmethod
    def _safe_notify(client, title: str, body: str, *, color: str = "blue", sec: float = 4.0) -> None:
        try:
            client.add_notification(
                str(title),
                str(body),
                color=str(color),
                auto_close_seconds=float(sec),
            )
        except Exception:
            pass

    def _make_detached_pipeline_client(self, client, *, tag: str) -> Any:
        """Build a websocket-independent client for long-running pipelines.

        This prevents Viser disconnects from aborting running jobs.
        """
        app = self.ctx.app
        hc = create_headless_client(app, client_id=f"{str(tag)}_headless", verbose=True)
        try:
            cam = getattr(client, "camera", None)
            if cam is not None:
                hc.camera.position = np.asarray(getattr(cam, "position"), dtype=np.float32).copy()
                hc.camera.wxyz = np.asarray(getattr(cam, "wxyz"), dtype=np.float32).copy()
                hc.camera.look_at = np.asarray(getattr(cam, "look_at"), dtype=np.float32).copy()
                hc.camera.up_direction = np.asarray(getattr(cam, "up_direction"), dtype=np.float32).copy()
                hc.camera.fov = float(getattr(cam, "fov", hc.camera.fov))
                hc.camera.near = float(getattr(cam, "near", hc.camera.near))
                hc.camera.far = float(getattr(cam, "far", hc.camera.far))
        except Exception as exc:
            self._dbg_full4d(
                "detached_client_copy_camera_failed",
                {"tag": str(tag), "error": f"{type(exc).__name__}:{exc}"},
            )
        try:
            hc.camera.near = float(getattr(app, "camera_znear", 0.05))
            hc.camera.far = float(getattr(app, "camera_zfar", 150.0))
        except Exception:
            pass
        self._dbg_full4d(
            "detached_client_ready",
            {
                "tag": str(tag),
                "client_id": str(getattr(hc, "client_id", "headless")),
                "reason": "avoid_frontend_disconnect_interrupt",
            },
        )
        return hc

    def _artifact_path(self, name: str) -> Path | None:
        with self.state.lock:
            root = self.state.artifact_root
        if root is None:
            return None
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"{ts}_{name}.json"

    def _dump_artifact(self, name: str, payload: dict[str, Any]) -> None:
        p = self._artifact_path(name)
        if p is None:
            return
        try:
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception:
            pass

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": [int(x) for x in list(value.shape)],
                "dtype": str(value.dtype),
            }
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, set):
            return list(value)
        return str(value)

    def _write_json_file(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=self._json_default),
                encoding="utf-8",
            )
        except Exception:
            pass

    @staticmethod
    def _safe_action_name(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return "action"
        chars: list[str] = []
        for ch in raw:
            if ch.isalnum() or ch in {"-", "_"}:
                chars.append(ch)
            else:
                chars.append("_")
        out = "".join(chars).strip("_")
        return out[:64] if out else "action"

    def _begin_intermediate_capture(self, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.state.lock:
            enabled = bool(getattr(self.state, "save_intermediate_results", False))
            base_root = getattr(self.state, "artifact_base_root", None) or getattr(self.state, "artifact_root", None)
            active_root = getattr(self.state, "intermediate_run_root", None)
        if (not enabled) or base_root is None:
            return {"enabled": False}
        if active_root is not None:
            return {"enabled": False, "nested": True, "active_root": str(active_root)}

        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + "_"
            + self._safe_action_name(str(action))
        )
        run_root = Path(base_root) / "intermediate_runs" / run_id
        try:
            run_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            return {"enabled": False}

        with self.state.lock:
            prev_artifact_root = getattr(self.state, "artifact_root", None)
            prev_agent1_debug = bool(getattr(self.state, "agent1_debug_save_eval", False))
            prev_agent2_debug = bool(getattr(self.state, "agent2_debug_save_views", False))
            self.state.artifact_root = run_root
            self.state.intermediate_run_root = run_root
            self.state.intermediate_run_id = str(run_id)
            # Force-save key visual diagnostics while detailed capture is enabled.
            self.state.agent1_debug_save_eval = True
            self.state.agent2_debug_save_views = True

        app = getattr(self.ctx, "app", None)
        run_meta = {
            "run_id": str(run_id),
            "action": str(action),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "params": dict(params or {}),
            "app_frame_idx": int(getattr(app, "current_frame", 0)) if app is not None else -1,
            "session_id": str(getattr(app, "session_id", "") or "") if app is not None else "",
            "session_root": str(getattr(app, "session_root", "") or "") if app is not None else "",
        }
        self._write_json_file(run_root / "run_meta.json", run_meta)
        return {
            "enabled": True,
            "run_id": str(run_id),
            "run_root": run_root,
            "prev_artifact_root": prev_artifact_root,
            "prev_agent1_debug": bool(prev_agent1_debug),
            "prev_agent2_debug": bool(prev_agent2_debug),
        }

    def _finish_intermediate_capture(
        self,
        capture_ctx: dict[str, Any] | None,
        *,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        ctx = dict(capture_ctx or {})
        if not bool(ctx.get("enabled", False)):
            return
        run_root_raw = ctx.get("run_root", None)
        run_root = Path(run_root_raw) if isinstance(run_root_raw, (str, Path)) else None

        with self.state.lock:
            logs = list(getattr(self.state, "logs", []) or [])
        log_rows = [
            {
                "ts": str(x.ts),
                "agent": str(x.agent),
                "status": str(x.status),
                "message": str(x.message),
            }
            for x in logs
        ]
        summary = {
            "run_id": str(ctx.get("run_id", "")),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "status": "error" if error is not None else "ok",
            "error": (f"{type(error).__name__}:{error}" if error is not None else ""),
            "logs": log_rows,
        }
        if isinstance(result, dict):
            summary["result"] = dict(result)
        if run_root is not None:
            self._write_json_file(run_root / "run_summary.json", summary)

        with self.state.lock:
            self.state.artifact_root = ctx.get("prev_artifact_root", self.state.artifact_root)
            self.state.agent1_debug_save_eval = bool(ctx.get("prev_agent1_debug", self.state.agent1_debug_save_eval))
            self.state.agent2_debug_save_views = bool(ctx.get("prev_agent2_debug", self.state.agent2_debug_save_views))
            self.state.intermediate_run_root = None
            self.state.intermediate_run_id = ""

    def _run_with_intermediate_capture(
        self,
        action: str,
        params: dict[str, Any] | None,
        fn,
    ) -> Any:
        capture_ctx = self._begin_intermediate_capture(action=action, params=params)
        try:
            out = fn()
            self._finish_intermediate_capture(capture_ctx, result=(out if isinstance(out, dict) else None))
            return out
        except Exception as exc:
            self._finish_intermediate_capture(capture_ctx, error=exc)
            raise

    def _run_async(self, fn):
        def _wrapped():
            try:
                fn()
            except Exception as exc:
                try:
                    self.fourd_validator.log("fail", f"runtime_exception: {type(exc).__name__}: {exc}")
                except Exception:
                    pass
                self._set_logs_text()
                self._set_status("idle")

        th = threading.Thread(target=_wrapped, daemon=True)
        th.start()

    def _is_enabled(self, client) -> bool:
        with self.state.lock:
            enabled = bool(self.state.enabled)
        if enabled:
            return True
        try:
            client.add_notification(
                "VLM Assistor",
                "Plugin is disabled. Enable 'Enable VLM Assistor' first.",
                color="orange",
                auto_close_seconds=4,
            )
        except Exception:
            pass
        return False

    def list_agent1_skills(self) -> list[dict[str, str]]:
        """Expose Agent1 callable skill metadata for external VLM agents."""
        try:
            bundle = getattr(self.click_agent, "skill_bundle", None)
            if bundle is None:
                return []
            registry = getattr(bundle, "registry", None)
            if registry is None:
                return []
            fn = getattr(registry, "describe", None)
            if callable(fn):
                out = fn()
                return list(out) if isinstance(out, list) else []
        except Exception:
            pass
        return []

    def call_agent1_skill(self, skill_name: str, *, client=None, **kwargs: Any) -> dict[str, Any]:
        """Call one Agent1 skill by name with runtime kwargs."""
        name = str(skill_name or "").strip()
        if not name:
            return {"ok": False, "reason": "empty_skill_name"}
        try:
            bundle = getattr(self.click_agent, "skill_bundle", None)
            if bundle is None:
                return {"ok": False, "reason": "agent1_skill_bundle_unavailable"}
            registry = getattr(bundle, "registry", None)
            if registry is None:
                return {"ok": False, "reason": "agent1_skill_registry_unavailable"}
            supported = [x.get("name", "") for x in self.list_agent1_skills()]
            if name not in supported:
                return {
                    "ok": False,
                    "reason": "unknown_skill_name",
                    "skill": name,
                    "supported_skills": supported,
                }
            payload = dict(kwargs or {})
            if client is not None and "client" not in payload:
                payload["client"] = client
            fn = getattr(registry, "call", None)
            if not callable(fn):
                return {"ok": False, "reason": "agent1_skill_call_unavailable", "skill": name}
            result = fn(name, **payload)
            if isinstance(result, dict):
                out = dict(result)
                out.setdefault("skill", name)
                return out
            return {"ok": True, "skill": name, "result": result}
        except TypeError as exc:
            return {"ok": False, "reason": f"invalid_skill_arguments:{type(exc).__name__}:{exc}", "skill": name}
        except Exception as exc:
            return {"ok": False, "reason": f"skill_call_failed:{type(exc).__name__}:{exc}", "skill": name}

    def run_video_agent(self, client, object_text: str, relation_hint: str) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: agent_0_video_understanding")
                out = self.video_agent.run(client, object_text=object_text, relation_hint=relation_hint)
                if bool(out.get("ok", False)):
                    names = self._init_agent1_targets_from_scene_graph(reset_cursor=True)
                    if names:
                        self.video_agent.log("info", f"agent1_targets_initialized:{names}")
                self._dump_artifact("agent0_video_understanding", out)
                self._print_agent0_result(out)
                self._set_scene_graph_text()
                self._set_scene_graph_detail_text()
                self._set_logs_text()
                self._set_status("idle")
                return out

            self._run_with_intermediate_capture(
                "agent0_video_understanding",
                {"object_text": str(object_text), "relation_hint": str(relation_hint)},
                _do,
            )

        self._run_async(_task)

    def select_next_agent1_object(self, client) -> None:
        if not self._is_enabled(client):
            return

        def _do() -> dict[str, Any]:
            names = self._scene_graph_ordered_object_names()
            if not names:
                try:
                    client.add_notification(
                        "Agent1",
                        "No objects from Agent0 scene_graph. Run Agent0 first.",
                        color="orange",
                        auto_close_seconds=5,
                    )
                except Exception:
                    pass
                out = {"ok": False, "reason": "no_objects_from_scene_graph"}
                self._dump_artifact("agent1_select_next_target", out)
                return out
            with self.state.lock:
                if self.state.agent1_targets != names:
                    self.state.agent1_targets = list(names)
                    self.state.agent1_target_index = 0
                    reached_end = False
                else:
                    old_idx = int(self.state.agent1_target_index)
                    self.state.agent1_target_index = min(
                        len(self.state.agent1_targets) - 1,
                        int(self.state.agent1_target_index) + 1,
                    )
                    reached_end = int(self.state.agent1_target_index) == old_idx and old_idx == (
                        len(self.state.agent1_targets) - 1
                    )
                idx = int(self.state.agent1_target_index)
                cur = self.state.agent1_targets[idx]
            self._set_object_description(cur)
            try:
                client.add_notification(
                    "Agent1",
                    (
                        f"Current target: {cur} ({idx + 1}/{len(names)})"
                        if not reached_end
                        else f"Already at last target: {cur} ({idx + 1}/{len(names)})"
                    ),
                    color="blue",
                    auto_close_seconds=4,
                )
            except Exception:
                pass
            out = {
                "ok": True,
                "current_target": str(cur),
                "target_index": int(idx),
                "target_count": int(len(names)),
                "reached_end": bool(reached_end),
            }
            self._dump_artifact("agent1_select_next_target", out)
            return out

        self._run_with_intermediate_capture("agent1_select_next_target", {}, _do)

    def select_prev_agent1_object(self, client) -> None:
        if not self._is_enabled(client):
            return

        def _do() -> dict[str, Any]:
            names = self._scene_graph_ordered_object_names()
            if not names:
                try:
                    client.add_notification(
                        "Agent1",
                        "No objects from Agent0 scene_graph. Run Agent0 first.",
                        color="orange",
                        auto_close_seconds=5,
                    )
                except Exception:
                    pass
                out = {"ok": False, "reason": "no_objects_from_scene_graph"}
                self._dump_artifact("agent1_select_prev_target", out)
                return out
            with self.state.lock:
                if self.state.agent1_targets != names:
                    self.state.agent1_targets = list(names)
                    self.state.agent1_target_index = 0
                    reached_start = True
                else:
                    old_idx = int(self.state.agent1_target_index)
                    self.state.agent1_target_index = max(0, int(self.state.agent1_target_index) - 1)
                    reached_start = int(self.state.agent1_target_index) == old_idx and old_idx == 0
                idx = int(self.state.agent1_target_index)
                cur = self.state.agent1_targets[idx]
            self._set_object_description(cur)
            try:
                client.add_notification(
                    "Agent1",
                    (
                        f"Current target: {cur} ({idx + 1}/{len(names)})"
                        if not reached_start
                        else f"Already at first target: {cur} ({idx + 1}/{len(names)})"
                    ),
                    color="blue",
                    auto_close_seconds=4,
                )
            except Exception:
                pass
            out = {
                "ok": True,
                "current_target": str(cur),
                "target_index": int(idx),
                "target_count": int(len(names)),
                "reached_start": bool(reached_start),
            }
            self._dump_artifact("agent1_select_prev_target", out)
            return out

        self._run_with_intermediate_capture("agent1_select_prev_target", {}, _do)

    def run_click_agent(self, client, object_text: str) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: agent_1_click")
                out = self.click_agent.run(client, object_text=object_text)
                self._dump_artifact("agent1_click", out)
                self._set_logs_text()
                self._set_status("idle")
                return out

            self._run_with_intermediate_capture(
                "agent1_click",
                {"object_text": str(object_text)},
                _do,
            )

        self._run_async(_task)

    def run_click_agent_evaluate_current(self, client, object_text: str) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: agent_1_eval_current_mask")
                out = self.click_agent.evaluate_current_mask(client, object_text=object_text)
                self._dump_artifact("agent1_eval_current_mask", out)
                self._set_logs_text()
                self._set_status("idle")
                return out

            self._run_with_intermediate_capture(
                "agent1_eval_current_mask",
                {"object_text": str(object_text)},
                _do,
            )

        self._run_async(_task)

    def run_generate_and_validate(self, client) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: generate+agent_2")
                gen = self.bridge.registry.call("sam3d.generate", client=client)
                if not gen.get("ok"):
                    self.gen3d_validator.log("fail", f"generate_failed: {gen.get('reason')}")
                    self._dump_artifact("generate_failed", gen)
                    self._set_logs_text()
                    self._set_status("idle")
                    return {"ok": False, "generate": gen}
                out = self.gen3d_validator.run(client)
                packed = {"ok": bool(out.get("ok", False)), "generate": gen, "validator": out}
                self._dump_artifact("agent2_gen3d_validator", packed)
                self._set_logs_text()
                self._set_status("idle")
                return packed

            self._run_with_intermediate_capture("generate_and_validate", {}, _do)

        self._run_async(_task)

    def _use_pose_adjust_v2_for_pipeline(self) -> bool:
        with self.state.lock:
            return bool(getattr(self.state, "use_pose_adjust_v2", False))

    def _run_agent2_for_pipeline(self, client, object_name: str) -> tuple[dict[str, Any], str]:
        if self._use_pose_adjust_v2_for_pipeline():
            return self.pose_adjuster_v2.run(client, object_name=object_name), "v2_direct_shift_poseopt"
        return self.pose_adjuster.run(client, object_name=object_name), "v1_candidate_search"

    def run_agent2_xpose_adjuster(self, client, object_name: str) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: agent_2_xpose_adjuster")
                out = self.pose_adjuster.run(client, object_name=object_name)
                self._dump_artifact("agent2_xpose_adjuster", out)
                self._set_logs_text()
                self._set_status("idle")
                return out

            self._run_with_intermediate_capture(
                "agent2_xpose_adjuster",
                {"object_name": str(object_name)},
                _do,
            )

        self._run_async(_task)

    def run_agent2_xpose_adjuster_v2(self, client, object_name: str) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: agent_2_xpose_adjuster_v2")
                out = self.pose_adjuster_v2.run(client, object_name=object_name)
                self._dump_artifact("agent2_xpose_adjuster_v2", out)
                self._set_logs_text()
                self._set_status("idle")
                return out

            self._run_with_intermediate_capture(
                "agent2_xpose_adjuster_v2",
                {"object_name": str(object_name)},
                _do,
            )

        self._run_async(_task)

    def run_pose_adjuster(self, client, object_name: str) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: agent_3_pose_adjuster_legacy")
                out = self.pose_adjuster.run(client, object_name=object_name)
                self._dump_artifact("agent3_pose_adjuster_legacy", out)
                self._set_logs_text()
                self._set_status("idle")
                return out

            self._run_with_intermediate_capture(
                "agent3_pose_adjuster_legacy",
                {"object_name": str(object_name)},
                _do,
            )

        self._run_async(_task)

    def run_4d_validator(self, client, window_size: int, frame_idx: int | None = None) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: agent_-1_4d_validator")
                out = self.fourd_validator.run(client, window_size=int(window_size), frame_idx=frame_idx)
                self._dump_artifact("agentm1_4d_validator", out)
                self._set_logs_text()
                self._set_status("idle")
                return out

            self._run_with_intermediate_capture(
                "agentm1_4d_validator",
                {"window_size": int(window_size), "frame_idx": (None if frame_idx is None else int(frame_idx))},
                _do,
            )

        self._run_async(_task)

    def run_4d_validator_sync(
        self,
        client,
        window_size: int,
        frame_idx: int | None = None,
    ) -> dict[str, Any]:
        if not self._is_enabled(client):
            return {"ok": False, "reason": "plugin_disabled"}
        out = self.fourd_validator.run(client, window_size=int(window_size), frame_idx=frame_idx)
        self._dump_artifact("agentm1_4d_validator", out)
        self._set_logs_text()
        return out

    def run_4d_memory_judge(self, client, window_size: int | None = None, frame_idx: int | None = None) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            def _do() -> dict[str, Any]:
                self._set_status("running: agent_4_4d_memory_judge")
                out = self.fourd_memory_judge.run(client, window_size=window_size, frame_idx=frame_idx)
                self._dump_artifact("agent4_4d_memory_judge", out)
                self._set_agent4_result_text()
                self._set_logs_text()
                self._set_status("idle")
                return out

            self._run_with_intermediate_capture(
                "agent4_4d_memory_judge",
                {
                    "window_size": (None if window_size is None else int(window_size)),
                    "frame_idx": (None if frame_idx is None else int(frame_idx)),
                },
                _do,
            )

        self._run_async(_task)

    def run_4d_memory_judge_sync(
        self,
        client,
        window_size: int | None = None,
        frame_idx: int | None = None,
    ) -> dict[str, Any]:
        if not self._is_enabled(client):
            return {"ok": False, "reason": "plugin_disabled"}

        def _do() -> dict[str, Any]:
            out = self.fourd_memory_judge.run(client, window_size=window_size, frame_idx=frame_idx)
            self._dump_artifact("agent4_4d_memory_judge", out)
            self._set_agent4_result_text()
            self._set_logs_text()
            return out

        return self._run_with_intermediate_capture(
            "agent4_4d_memory_judge_sync",
            {
                "window_size": (None if window_size is None else int(window_size)),
                "frame_idx": (None if frame_idx is None else int(frame_idx)),
            },
            _do,
        )

    @staticmethod
    def _dbg_full4d(msg: str, payload: dict[str, Any] | None = None) -> None:
        if payload is None:
            print(f"[VLM Assistor][Full4D] {msg}")
            return
        try:
            print(f"[VLM Assistor][Full4D] {msg}: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        except Exception:
            print(f"[VLM Assistor][Full4D] {msg}: {payload}")

    def _app_frame_count(self) -> int:
        app = self.ctx.app
        seq = getattr(app, "sequence", None)
        return int(len(getattr(seq, "frame_tokens", []) or []))

    def _set_current_frame_for_client(self, client, frame_idx: int, *, mirror_client=None) -> int:
        app = self.ctx.app
        total = self._app_frame_count()
        if total <= 0:
            return -1
        idx = max(0, min(int(frame_idx), total - 1))
        try:
            app.current_frame = int(idx)
        except Exception:
            pass

        def _apply_for(one_client, *, stream: str) -> None:
            if one_client is None:
                return
            try:
                slider = None
                sliders = getattr(app, "frame_slider_handles", None)
                if isinstance(sliders, dict):
                    slider = sliders.get(one_client)
                if slider is not None:
                    slider.value = int(idx)
            except Exception:
                pass
            try:
                preview = None
                previews = getattr(app, "preview_handles", None)
                if isinstance(previews, dict):
                    preview = previews.get(one_client)
                apply_fn = getattr(app, "_apply_frame", None)
                if callable(apply_fn) and preview is not None:
                    apply_fn(int(idx), preview)
            except Exception as exc:
                self._dbg_full4d(
                    "set_current_frame_apply_failed",
                    {
                        "frame_idx": int(idx),
                        "stream": str(stream),
                        "error": f"{type(exc).__name__}:{exc}",
                    },
                )

        _apply_for(client, stream="runtime")
        if mirror_client is not None and mirror_client is not client:
            _apply_for(mirror_client, stream="ui_mirror")
        return int(idx)

    def _ensure_l4gm_session(self, client) -> dict[str, Any]:
        app = self.ctx.app
        fn = getattr(app, "_ensure_session", None)
        if not callable(fn):
            return {"ok": False, "reason": "app_ensure_session_missing"}
        fn(client)
        return {
            "ok": True,
            "session_id": str(getattr(app, "session_id", "") or ""),
            "session_root": str(getattr(app, "session_root", "") or ""),
        }

    def _safe_post_json(
        self,
        post_fn: Any,
        url: str,
        payload: dict[str, Any],
        *,
        timeout: int,
        stage: str,
    ) -> dict[str, Any]:
        try:
            resp = post_fn(str(url), payload, timeout=int(timeout))
            return {"ok": True, "resp": resp}
        except Exception as exc:
            err = f"{type(exc).__name__}:{exc}"
            self._dbg_full4d(f"{stage}_http_exception", {"url": str(url), "error": err})
            return {
                "ok": False,
                "reason": f"{stage}_http_exception:{err}",
                "url": str(url),
            }

    def _run_l4gm_generate_segment(
        self,
        client,
        *,
        start_frame: int,
        num_frames: int,
        mirror_client=None,
    ) -> dict[str, Any]:
        app = self.ctx.app
        self._ensure_l4gm_session(client)
        post_fn = getattr(app, "_post_json", None)
        base_fn = getattr(app, "_backend_base_url", None)
        if not callable(post_fn) or not callable(base_fn):
            return {"ok": False, "reason": "app_backend_http_helpers_missing"}
        seed_check = self._ensure_l4gm_seed_frame_ready(
            client,
            start_frame=int(start_frame),
            mirror_client=mirror_client,
        )
        if not bool(seed_check.get("ok", False)):
            return {"ok": False, "reason": "ensure_seed_frame_failed", "seed_check": seed_check}
        payload = {
            "session_id": str(getattr(app, "session_id", "") or ""),
            "num_frames": int(max(1, int(num_frames))),
            "start_frame": int(max(0, int(start_frame))),
        }
        self._dbg_full4d("generate_segment_request", payload)
        req = self._safe_post_json(
            post_fn,
            str(base_fn()) + "/generate/later_gen",
            payload,
            timeout=900,
            stage="generate_later_gen",
        )
        if not bool(req.get("ok", False)):
            return {"ok": False, "reason": str(req.get("reason", "generate_later_gen_http_exception")), "raw": req}
        resp = dict(req.get("resp", {}) or {})
        if str(resp.get("status", "")).lower() != "success":
            return {"ok": False, "reason": str(resp.get("message", "generate/later_gen_failed")), "raw": resp}
        try:
            app.sequence.cache.clear()
        except Exception:
            pass
        start = int(resp.get("start_frame", int(start_frame)))
        end = int(resp.get("end_frame", int(start)))
        self._set_current_frame_for_client(client, start, mirror_client=mirror_client)
        out = {
            "ok": True,
            "mode": "plain",
            "start_frame": int(start),
            "end_frame": int(end),
            "raw": resp,
        }
        self._dbg_full4d("generate_segment_result", out)
        return out

    def _run_l4gm_generate_segment_with_memory(
        self,
        client,
        *,
        start_frame: int,
        memory_indices: list[int],
        total_clip_frames: int = 16,
        mirror_client=None,
    ) -> dict[str, Any]:
        app = self.ctx.app
        self._ensure_l4gm_session(client)
        post_fn = getattr(app, "_post_json", None)
        base_fn = getattr(app, "_backend_base_url", None)
        if not callable(post_fn) or not callable(base_fn):
            return {"ok": False, "reason": "app_backend_http_helpers_missing"}
        seed_check = self._ensure_l4gm_seed_frame_ready(
            client,
            start_frame=int(start_frame),
            mirror_client=mirror_client,
        )
        if not bool(seed_check.get("ok", False)):
            return {"ok": False, "reason": "ensure_seed_frame_failed", "seed_check": seed_check}
        mem = sorted({int(x) for x in list(memory_indices or []) if int(x) >= 0})
        if not mem:
            return {"ok": False, "reason": "memory_indices_empty"}
        payload = {
            "session_id": str(getattr(app, "session_id", "") or ""),
            "start_frame": int(max(0, int(start_frame))),
            "total_clip_frames": int(max(2, int(total_clip_frames))),
            "memory_frame_indices": mem,
            "debug_ref_video": bool(getattr(app, "memory_debug_ref_video", False)),
        }
        self._dbg_full4d("generate_segment_with_memory_request", payload)
        req = self._safe_post_json(
            post_fn,
            str(base_fn()) + "/generate/later_gen_with_memory",
            payload,
            timeout=1200,
            stage="generate_later_gen_with_memory",
        )
        if not bool(req.get("ok", False)):
            return {
                "ok": False,
                "reason": str(req.get("reason", "generate_later_gen_with_memory_http_exception")),
                "raw": req,
            }
        resp = dict(req.get("resp", {}) or {})
        if str(resp.get("status", "")).lower() != "success":
            return {"ok": False, "reason": str(resp.get("message", "generate/later_gen_with_memory_failed")), "raw": resp}
        try:
            app.sequence.cache.clear()
        except Exception:
            pass
        start = int(resp.get("start_frame", int(start_frame)))
        end = int(resp.get("end_frame", int(start)))
        self._set_current_frame_for_client(client, start, mirror_client=mirror_client)
        out = {
            "ok": True,
            "mode": "memory",
            "start_frame": int(start),
            "end_frame": int(end),
            "memory_frame_indices": mem,
            "raw": resp,
        }
        self._dbg_full4d("generate_segment_with_memory_result", out)
        return out

    def _adjust_memory_bank(self, client, memory_indices: list[int], *, allow_empty: bool = False) -> dict[str, Any]:
        app = self.ctx.app
        self._ensure_l4gm_session(client)
        post_fn = getattr(app, "_post_json", None)
        base_fn = getattr(app, "_backend_base_url", None)
        if not callable(post_fn) or not callable(base_fn):
            return {"ok": False, "reason": "app_backend_http_helpers_missing"}
        ids = sorted({int(x) for x in list(memory_indices or []) if int(x) >= 0})
        if (not ids) and (not bool(allow_empty)):
            return {"ok": False, "reason": "memory_indices_empty"}
        payload = {
            "session_id": str(getattr(app, "session_id", "") or ""),
            "memory_frame_indices": ids,
            "allow_empty": bool(allow_empty),
        }
        self._dbg_full4d("adjust_memory_bank_request", payload)
        req = self._safe_post_json(
            post_fn,
            str(base_fn()) + "/memory/adjust_frames",
            payload,
            timeout=900,
            stage="memory_adjust_frames",
        )
        if not bool(req.get("ok", False)):
            return {"ok": False, "reason": str(req.get("reason", "memory_adjust_frames_http_exception")), "raw": req}
        resp = dict(req.get("resp", {}) or {})
        if str(resp.get("status", "")).lower() != "success":
            return {"ok": False, "reason": str(resp.get("message", "memory/adjust_frames_failed")), "raw": resp}
        mem_indices = [int(x) for x in list(resp.get("memory_frame_indices", ids) or ids)]
        try:
            app.memory_selected_frames = list(mem_indices)
        except Exception:
            pass
        out = {"ok": True, "memory_frame_indices": mem_indices, "raw": resp}
        self._dbg_full4d("adjust_memory_bank_result", out)
        return out

    def _frame_has_generated_gs(self, frame_idx: int) -> bool:
        app = self.ctx.app
        seq = getattr(app, "sequence", None)
        gs_fn = getattr(seq, "gs_path", None)
        if not callable(gs_fn):
            return False
        try:
            p = Path(gs_fn(int(frame_idx)))
            return bool(p.exists())
        except Exception:
            return False

    def _ensure_l4gm_seed_frame_ready(self, client, *, start_frame: int, mirror_client=None) -> dict[str, Any]:
        """Ensure L4GM has at least one seed GS frame before later_gen."""
        app = self.ctx.app
        total = self._app_frame_count()
        if total <= 0:
            return {"ok": False, "reason": "frame_tokens_empty"}
        st = max(0, min(int(start_frame), total - 1))
        # If current start or any previous frame already has GS, later_gen can proceed.
        for i in range(st, -1, -1):
            if self._frame_has_generated_gs(i):
                return {"ok": True, "seed_ready": True, "seed_frame": int(i), "seed_mode": "existing_gs"}

        self._dbg_full4d(
            "seed_missing_auto_generate_first",
            {
                "start_frame": int(st),
                "reason": "no_existing_gs_in_prefix_frames",
            },
        )
        self._ensure_l4gm_session(client)
        post_fn = getattr(app, "_post_json", None)
        base_fn = getattr(app, "_backend_base_url", None)
        if not callable(post_fn) or not callable(base_fn):
            return {"ok": False, "reason": "app_backend_http_helpers_missing"}
        init_mode = str(getattr(app, "l4gm_init_mode", "imagedream_init") or "imagedream_init")
        payload = {
            "session_id": str(getattr(app, "session_id", "") or ""),
            "generator": "l4gm",
            "init_mode": init_mode,
        }
        self._dbg_full4d("generate_first_request", payload)
        req = self._safe_post_json(
            post_fn,
            str(base_fn()) + "/generate/first",
            payload,
            timeout=1800,
            stage="generate_first",
        )
        if not bool(req.get("ok", False)):
            return {"ok": False, "reason": str(req.get("reason", "generate_first_http_exception")), "raw": req}
        resp = dict(req.get("resp", {}) or {})
        if str(resp.get("status", "")).lower() != "success":
            return {"ok": False, "reason": str(resp.get("message", "generate/first_failed")), "raw": resp}
        try:
            app.sequence.cache.clear()
        except Exception:
            pass
        self._set_current_frame_for_client(client, 0, mirror_client=mirror_client)
        out = {"ok": True, "seed_ready": True, "seed_frame": 0, "seed_mode": "auto_generate_first", "raw": resp}
        self._dbg_full4d("generate_first_result", out)
        return out

    def _sam3d_add_all_objects_and_exit(self, client, *, stage: str) -> dict[str, Any]:
        """Persist SAM3D generated objects into session PLY and exit SAM3D mode."""
        out = self.bridge.registry.call("session.add_all_objects_and_exit", client=client)
        packed = {
            "ok": bool(out.get("ok", False)),
            "stage": str(stage),
            "detail": out,
        }
        self._dbg_full4d("sam3d_add_all_objects_and_exit", packed)
        return packed

    def _repair_frame_with_full3d(
        self,
        client,
        *,
        frame_idx: int,
        object_text: str,
        relation_hint: str,
        mirror_client=None,
    ) -> dict[str, Any]:
        self._dbg_full4d("repair_full3d_start", {"frame_idx": int(frame_idx)})
        self._set_current_frame_for_client(client, int(frame_idx), mirror_client=mirror_client)
        out = self._run_full_pipeline_task(client, object_text=object_text, relation_hint=relation_hint)
        add_all = self._sam3d_add_all_objects_and_exit(client, stage=f"repair_frame_{int(frame_idx)}")
        packed = {
            "ok": bool(out.get("ok", False)) and bool(add_all.get("ok", False)),
            "frame_idx": int(frame_idx),
            "pipeline_result": out,
            "sam3d_finalize": add_all,
        }
        self._dbg_full4d("repair_full3d_done", packed)
        return packed

    def _repair_frame_with_memory(
        self,
        client,
        *,
        frame_idx: int,
        issue_diagnosis: dict[str, Any],
        history_window: int,
        mirror_client=None,
    ) -> dict[str, Any]:
        app = self.ctx.app
        traits = [str(x).strip() for x in list(issue_diagnosis.get("memory_frame_characteristics", []) or []) if str(x).strip()]
        st = max(0, int(frame_idx) - max(4, int(history_window) * 4))
        candidate_indices = [i for i in range(st, int(frame_idx)) if self._frame_has_generated_gs(i)]
        if not candidate_indices:
            return {"ok": False, "reason": "no_candidate_history_frames", "frame_idx": int(frame_idx)}

        self._dbg_full4d(
            "repair_memory_select_start",
            {
                "frame_idx": int(frame_idx),
                "candidate_indices": [int(x) for x in candidate_indices],
                "traits": traits,
            },
        )
        sel = self.memory_selector.select_memory_frames(
            client,
            target_frame_idx=int(frame_idx),
            candidate_frame_indices=candidate_indices,
            desired_traits=traits,
            top_k=3,
        )
        if not bool(sel.get("ok", False)):
            return {"ok": False, "reason": "memory_selector_failed", "selector": sel}
        selected = [int(x) for x in list(sel.get("selected_frame_indices", []) or []) if int(x) >= 0]
        if not selected:
            return {"ok": False, "reason": "memory_selector_empty", "selector": sel}
        current_bank = sorted({int(x) for x in list(getattr(app, "memory_selected_frames", []) or []) if int(x) >= 0})
        new_bank = sorted(set(current_bank + selected))
        with self.state.lock:
            cap = int(getattr(self.state, "agent4_memory_bank_capacity", 8))
        cap = max(1, cap)
        if len(new_bank) > cap:
            new_bank = sorted(new_bank)[-cap:]
        adj = self._adjust_memory_bank(client, new_bank, allow_empty=False)
        if not bool(adj.get("ok", False)):
            return {"ok": False, "reason": "adjust_memory_bank_failed", "selector": sel, "adjust": adj}
        gen = self._run_l4gm_generate_segment_with_memory(
            client,
            start_frame=int(frame_idx),
            memory_indices=list(new_bank),
            total_clip_frames=16,
            mirror_client=mirror_client,
        )
        out = {
            "ok": bool(gen.get("ok", False)),
            "frame_idx": int(frame_idx),
            "selector": sel,
            "memory_bank": list(new_bank),
            "generate": gen,
            "end_frame": int(gen.get("end_frame", int(frame_idx))) if bool(gen.get("ok", False)) else int(frame_idx),
        }
        self._dbg_full4d("repair_memory_done", out)
        return out

    def _maybe_update_memory_bank_from_good_frame(self, client, *, frame_idx: int) -> dict[str, Any]:
        app = self.ctx.app
        existing = sorted({int(x) for x in list(getattr(app, "memory_selected_frames", []) or []) if int(x) >= 0})
        dec = self.memory_selector.decide_memory_bank_update(
            client,
            frame_idx=int(frame_idx),
            existing_memory_indices=list(existing),
        )
        if not bool(dec.get("ok", False)):
            return {"ok": False, "reason": "memory_update_decision_failed", "decision": dec}
        should_add = bool(dec.get("should_add", False))
        if not should_add:
            return {"ok": True, "updated": False, "decision": dec, "memory_frame_indices": list(existing)}
        replace_idx = int(dec.get("replace_frame_idx", -1))
        bank = list(existing)
        if replace_idx in bank:
            bank = [x for x in bank if int(x) != int(replace_idx)]
        if int(frame_idx) not in bank:
            bank.append(int(frame_idx))
        with self.state.lock:
            cap = int(getattr(self.state, "agent4_memory_bank_capacity", 8))
        cap = max(1, cap)
        bank = sorted(set(bank))
        if len(bank) > cap:
            bank = sorted(bank)[-cap:]
        adj = self._adjust_memory_bank(client, bank, allow_empty=False)
        return {
            "ok": bool(adj.get("ok", False)),
            "updated": bool(adj.get("ok", False)),
            "decision": dec,
            "memory_frame_indices": list(adj.get("memory_frame_indices", bank) or bank),
            "adjust": adj,
        }

    def _full4d_checkpoint_path(self) -> Path | None:
        with self.state.lock:
            root = self.state.artifact_root
        if root is None:
            return None
        root.mkdir(parents=True, exist_ok=True)
        return root / "pipeline_full_4d_checkpoint.json"

    def _load_full4d_checkpoint(self) -> dict[str, Any] | None:
        p = self._full4d_checkpoint_path()
        if p is None or (not p.exists()):
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            self._dbg_full4d("checkpoint_load_failed", {"path": str(p), "error": f"{type(exc).__name__}:{exc}"})
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _save_full4d_checkpoint(self, payload: dict[str, Any]) -> None:
        p = self._full4d_checkpoint_path()
        if p is None:
            return
        try:
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception as exc:
            self._dbg_full4d("checkpoint_save_failed", {"path": str(p), "error": f"{type(exc).__name__}:{exc}"})

    def _clear_full4d_checkpoint(self) -> None:
        p = self._full4d_checkpoint_path()
        if p is None:
            return
        try:
            if p.exists():
                p.unlink()
        except Exception as exc:
            self._dbg_full4d("checkpoint_clear_failed", {"path": str(p), "error": f"{type(exc).__name__}:{exc}"})

    def _run_full_4d_pipeline_task(self, client, object_text: str, relation_hint: str, *, ui_client=None) -> dict[str, Any]:
        self._set_status("running: full_pipeline_4d")
        app = self.ctx.app
        orig_frame = int(getattr(app, "current_frame", 0))
        with self.state.lock:
            history_window = int(getattr(self.state, "agent4_history_window", 6))
            memory_enabled = bool(getattr(self.state, "agent4_memory_enabled", True))
            validator_enabled = bool(getattr(self.state, "agent4_validator_enabled", True))
        if not memory_enabled:
            try:
                app.memory_selected_frames = []
            except Exception:
                pass
        ablation_flags: list[str] = []
        if not bool(validator_enabled):
            ablation_flags.append("wo_4dvalidator")
        if not bool(memory_enabled):
            ablation_flags.append("wo_memory")
        result: dict[str, Any] = {
            "ok": False,
            "agent": "full_4d_pipeline",
            "memory_enabled": bool(memory_enabled),
            "validator_enabled": bool(validator_enabled),
            "ablation_mode": "+".join(ablation_flags),
            "init_3d": {},
            "sam3d_finalize": {},
            "segments": [],
            "errors": [],
        }
        total_frames = int(self._app_frame_count())
        cursor = 0
        seg_id = 0
        # Allow enough segments to cover long videos even when each memory-based segment
        # only advances a small number of frames.
        max_segments = max(12, int(total_frames) + 4)
        init_3d_done = False
        sam3d_finalize_done = False
        in_progress_segment: dict[str, Any] | None = None
        if ui_client is not None and ui_client is not client:
            self._dbg_full4d(
                "ui_mirror_enabled",
                {
                    "runtime_client_id": str(getattr(client, "client_id", "runtime")),
                    "ui_client_id": str(getattr(ui_client, "client_id", "ui")),
                },
            )

        def _checkpoint_payload(*, status: str, stage: str) -> dict[str, Any]:
            return {
                "version": 1,
                "pipeline": "full_4d_pipeline",
                "status": str(status),
                "stage": str(stage),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "object_text": str(object_text),
                "relation_hint": str(relation_hint),
                "session_id": str(getattr(app, "session_id", "") or ""),
                "total_frames": int(total_frames),
                "target_last_frame": int(max(0, total_frames - 1)),
                "cursor": int(cursor),
                "seg_id": int(seg_id),
                "max_segments": int(max_segments),
                "memory_enabled": bool(memory_enabled),
                "validator_enabled": bool(validator_enabled),
                "history_window": int(history_window),
                "init_3d_done": bool(init_3d_done),
                "sam3d_finalize_done": bool(sam3d_finalize_done),
                "memory_bank": sorted(
                    {int(x) for x in list(getattr(app, "memory_selected_frames", []) or []) if int(x) >= 0}
                ),
                "in_progress_segment": dict(in_progress_segment or {}),
                "result": dict(result),
            }

        def _save_checkpoint(*, status: str, stage: str) -> None:
            self._save_full4d_checkpoint(_checkpoint_payload(status=status, stage=stage))

        try:
            sess = self._ensure_l4gm_session(client)
            if not bool(sess.get("ok", False)):
                result["errors"].append({"stage": "ensure_session", "detail": sess})
                self._dbg_full4d("abort_ensure_session_failed", {"detail": sess})
                _save_checkpoint(status="failed", stage="ensure_session_failed")
                return result

            total_frames = int(self._app_frame_count())
            if total_frames <= 0:
                result["errors"].append({"stage": "frame_tokens", "detail": "empty"})
                _save_checkpoint(status="failed", stage="frame_tokens_empty")
                return result
            max_segments = max(12, int(total_frames) + 4)

            cp = self._load_full4d_checkpoint()
            can_resume = False
            if isinstance(cp, dict):
                cp_ver = int(cp.get("version", 0))
                cp_pipeline = str(cp.get("pipeline", ""))
                cp_status = str(cp.get("status", "")).lower()
                cp_obj = str(cp.get("object_text", ""))
                cp_rel = str(cp.get("relation_hint", ""))
                params_match = (cp_obj == str(object_text)) and (cp_rel == str(relation_hint))
                cp_memory_enabled = bool(cp.get("memory_enabled", True))
                memory_mode_match = bool(cp_memory_enabled) == bool(memory_enabled)
                cp_validator_enabled = bool(cp.get("validator_enabled", True))
                validator_mode_match = bool(cp_validator_enabled) == bool(validator_enabled)
                if (
                    cp_ver == 1
                    and cp_pipeline == "full_4d_pipeline"
                    and cp_status in {"running", "failed"}
                    and params_match
                    and memory_mode_match
                    and validator_mode_match
                ):
                    can_resume = True

            if can_resume and isinstance(cp, dict):
                cp_result = dict(cp.get("result", {}) or {})
                if isinstance(cp_result, dict):
                    result["init_3d"] = dict(cp_result.get("init_3d", {}) or {})
                    result["sam3d_finalize"] = dict(cp_result.get("sam3d_finalize", {}) or {})
                    result["segments"] = list(cp_result.get("segments", []) or [])
                    old_errors = list(cp_result.get("errors", []) or [])
                    if old_errors:
                        result["resume_from_errors"] = old_errors
                    result["errors"] = []
                cursor = max(0, min(int(cp.get("cursor", 0)), max(0, total_frames - 1)))
                seg_id = max(0, int(cp.get("seg_id", 0)))
                init_3d_done = bool(cp.get("init_3d_done", bool(result.get("init_3d"))))
                sam3d_finalize_done = bool(cp.get("sam3d_finalize_done", bool(result.get("sam3d_finalize"))))
                maybe_seg = cp.get("in_progress_segment", {})
                in_progress_segment = dict(maybe_seg) if isinstance(maybe_seg, dict) and maybe_seg else None
                cp_mem = sorted({int(x) for x in list(cp.get("memory_bank", []) or []) if int(x) >= 0})
                if cp_mem:
                    try:
                        app.memory_selected_frames = list(cp_mem)
                    except Exception:
                        pass
                self._dbg_full4d(
                    "resume_checkpoint_loaded",
                    {
                        "cursor": int(cursor),
                        "seg_id": int(seg_id),
                        "init_3d_done": bool(init_3d_done),
                        "sam3d_finalize_done": bool(sam3d_finalize_done),
                        "has_in_progress_segment": bool(in_progress_segment),
                        "memory_bank": cp_mem,
                    },
                )
            else:
                _save_checkpoint(status="running", stage="start")

            if not init_3d_done:
                self._set_current_frame_for_client(client, 0, mirror_client=ui_client)
                init_3d = self._run_full_pipeline_task(client, object_text=object_text, relation_hint=relation_hint)
                result["init_3d"] = init_3d
                init_3d_done = True
                if not bool(init_3d.get("ok", False)):
                    self._dbg_full4d("init_3d_partial_or_failed", {"init_3d_ok": bool(init_3d.get("ok", False))})
                _save_checkpoint(status="running", stage="after_init_3d")

            if not sam3d_finalize_done:
                add_all = self._sam3d_add_all_objects_and_exit(client, stage="after_init_3d")
                result["sam3d_finalize"] = add_all
                sam3d_finalize_done = bool(add_all.get("ok", False))
                if not bool(add_all.get("ok", False)) and bool(result.get("init_3d", {}).get("ok", False)):
                    result["errors"].append({"stage": "sam3d_finalize_after_init_3d", "detail": add_all})
                    self._dbg_full4d("abort_sam3d_finalize_after_init_3d_failed", {"detail": add_all})
                    _save_checkpoint(status="failed", stage="sam3d_finalize_after_init_3d_failed")
                    return result
                _save_checkpoint(status="running", stage="after_sam3d_finalize")

            while seg_id < max_segments and cursor < total_frames - 1:
                if in_progress_segment is None:
                    seg_id += 1
                    memory_bank = (
                        sorted({int(x) for x in list(getattr(app, "memory_selected_frames", []) or []) if int(x) >= 0})
                        if bool(memory_enabled)
                        else []
                    )
                    if memory_bank:
                        gen = self._run_l4gm_generate_segment_with_memory(
                            client,
                            start_frame=int(cursor),
                            memory_indices=memory_bank,
                            total_clip_frames=16,
                            mirror_client=ui_client,
                        )
                    else:
                        gen = self._run_l4gm_generate_segment(
                            client,
                            start_frame=int(cursor),
                            num_frames=16,
                            mirror_client=ui_client,
                        )
                    if not bool(gen.get("ok", False)):
                        result["errors"].append({"stage": "generate", "segment_id": int(seg_id), "detail": gen})
                        self._dbg_full4d("abort_generate_failed", {"segment_id": int(seg_id), "detail": gen})
                        _save_checkpoint(status="failed", stage="generate_failed")
                        break

                    seg_start = int(gen.get("start_frame", cursor))
                    seg_end = int(gen.get("end_frame", seg_start))
                    if seg_end <= cursor and seg_id > 1:
                        result["errors"].append(
                            {
                                "stage": "generate_no_progress",
                                "segment_id": int(seg_id),
                                "cursor": int(cursor),
                                "start": int(seg_start),
                                "end": int(seg_end),
                            }
                        )
                        self._dbg_full4d(
                            "abort_generate_no_progress",
                            {"segment_id": int(seg_id), "cursor": int(cursor), "start": int(seg_start), "end": int(seg_end)},
                        )
                        _save_checkpoint(status="failed", stage="generate_no_progress")
                        break

                    seg_log: dict[str, Any] = {
                        "segment_id": int(seg_id),
                        "mode": str(gen.get("mode", "plain")),
                        "start_frame": int(seg_start),
                        "end_frame": int(seg_end),
                        "frames": [],
                    }
                    f = int(seg_start)
                    dyn_end = int(seg_end)
                    in_progress_segment = {
                        "segment_id": int(seg_id),
                        "seg_log": dict(seg_log),
                        "next_frame": int(f),
                        "dyn_end": int(dyn_end),
                    }
                    _save_checkpoint(status="running", stage="segment_generated")
                else:
                    seg_log = dict(in_progress_segment.get("seg_log", {}) or {})
                    seg_log["frames"] = list(seg_log.get("frames", []) or [])
                    seg_id = max(int(seg_id), int(in_progress_segment.get("segment_id", seg_id)))
                    f = int(in_progress_segment.get("next_frame", seg_log.get("start_frame", cursor)))
                    dyn_end = int(in_progress_segment.get("dyn_end", seg_log.get("end_frame", f)))
                    self._dbg_full4d(
                        "resume_from_segment_frame",
                        {
                            "segment_id": int(seg_id),
                            "next_frame": int(f),
                            "dyn_end": int(dyn_end),
                            "cursor": int(cursor),
                        },
                    )

                while f <= dyn_end and f < total_frames:
                    self._set_current_frame_for_client(client, f, mirror_client=ui_client)
                    if not bool(validator_enabled):
                        frame_log = {
                            "frame_idx": int(f),
                            "validator_ok": True,
                            "validator_skipped": True,
                            "validator": {
                                "ok": True,
                                "skipped": True,
                                "reason": "agent4_validator_disabled",
                            },
                        }
                        if bool(memory_enabled):
                            upd = self._maybe_update_memory_bank_from_good_frame(client, frame_idx=int(f))
                        else:
                            upd = {
                                "ok": True,
                                "updated": False,
                                "reason": "memory_disabled",
                                "memory_frame_indices": [],
                            }
                        frame_log["memory_bank_update"] = upd
                    else:
                        val = self.run_4d_validator_sync(client, window_size=int(history_window), frame_idx=int(f))
                        frame_log = {
                            "frame_idx": int(f),
                            "validator_ok": bool(val.get("ok", False)),
                            "validator": val,
                        }
                        if not bool(val.get("ok", False)):
                            diag = dict(val.get("issue_diagnosis", {}) or {})
                            action = str(diag.get("action_recommendation", "manual_review") or "manual_review")
                            frame_log["diagnosis_action"] = action
                            if action == "run_full_3d_gen_pipeline":
                                repair = self._repair_frame_with_full3d(
                                    client,
                                    frame_idx=int(f),
                                    object_text=object_text,
                                    relation_hint=relation_hint,
                                    mirror_client=ui_client,
                                )
                                frame_log["repair"] = repair
                            elif action == "regenerate_with_memory_frame":
                                if bool(memory_enabled):
                                    repair = self._repair_frame_with_memory(
                                        client,
                                        frame_idx=int(f),
                                        issue_diagnosis=diag,
                                        history_window=int(history_window),
                                        mirror_client=ui_client,
                                    )
                                else:
                                    gen_plain = self._run_l4gm_generate_segment(
                                        client,
                                        start_frame=int(f),
                                        num_frames=16,
                                        mirror_client=ui_client,
                                    )
                                    repair = {
                                        "ok": bool(gen_plain.get("ok", False)),
                                        "reason": "memory_disabled_plain_regenerate",
                                        "frame_idx": int(f),
                                        "generate": gen_plain,
                                        "end_frame": int(gen_plain.get("end_frame", int(f)))
                                        if bool(gen_plain.get("ok", False))
                                        else int(f),
                                    }
                                frame_log["repair"] = repair
                                if bool(repair.get("ok", False)):
                                    dyn_end = max(dyn_end, int(repair.get("end_frame", dyn_end)))
                            else:
                                frame_log["repair"] = {"ok": False, "reason": "manual_review"}
                            reval = self.run_4d_validator_sync(client, window_size=int(history_window), frame_idx=int(f))
                            frame_log["revalidate"] = reval
                            frame_log["revalidate_ok"] = bool(reval.get("ok", False))
                        else:
                            if bool(memory_enabled):
                                upd = self._maybe_update_memory_bank_from_good_frame(client, frame_idx=int(f))
                            else:
                                upd = {
                                    "ok": True,
                                    "updated": False,
                                    "reason": "memory_disabled",
                                    "memory_frame_indices": [],
                                }
                            frame_log["memory_bank_update"] = upd
                    seg_log["frames"].append(frame_log)
                    f += 1
                    in_progress_segment = {
                        "segment_id": int(seg_id),
                        "seg_log": dict(seg_log),
                        "next_frame": int(f),
                        "dyn_end": int(dyn_end),
                    }
                    _save_checkpoint(status="running", stage="segment_frame_validated")

                seg_log["final_end_frame"] = int(dyn_end)
                result["segments"].append(seg_log)
                cursor = int(max(cursor + 1, dyn_end))
                in_progress_segment = None
                _save_checkpoint(status="running", stage="segment_completed")
                if cursor >= total_frames - 1:
                    break

            if cursor < total_frames - 1:
                result["errors"].append(
                    {
                        "stage": "incomplete_generation",
                        "detail": {
                            "cursor": int(cursor),
                            "target_last_frame": int(max(0, total_frames - 1)),
                            "max_segments": int(max_segments),
                            "reason": "stopped_before_last_frame",
                        },
                    }
                )
                self._dbg_full4d(
                    "incomplete_generation",
                    {
                        "cursor": int(cursor),
                        "target_last_frame": int(max(0, total_frames - 1)),
                        "max_segments": int(max_segments),
                    },
                )
            result["completed_until_frame"] = int(max(0, cursor))
            result["target_last_frame"] = int(max(0, total_frames - 1))
            result["fully_covered"] = bool(cursor >= total_frames - 1)
            result["ok"] = bool(len(result["segments"]) > 0 and len(result["errors"]) == 0 and result["fully_covered"])
            if bool(result.get("ok", False)):
                _save_checkpoint(status="done", stage="completed")
                self._clear_full4d_checkpoint()
            else:
                _save_checkpoint(status="failed", stage="finished_with_errors")
            return result
        except Exception as exc:
            result["errors"].append({"stage": "runtime_exception", "detail": f"{type(exc).__name__}:{exc}"})
            self._dbg_full4d("runtime_exception", {"error": f"{type(exc).__name__}:{exc}"})
            _save_checkpoint(status="failed", stage="runtime_exception")
            return result
        finally:
            self._dbg_full4d(
                "final_result_summary",
                {
                    "ok": bool(result.get("ok", False)),
                    "segments": int(len(list(result.get("segments", []) or []))),
                    "errors": int(len(list(result.get("errors", []) or []))),
                },
            )
            self._set_current_frame_for_client(client, int(orig_frame), mirror_client=ui_client)
            self._dump_artifact("pipeline_full_4d", result)
            self._set_scene_graph_text()
            self._set_scene_graph_detail_text()
            self._set_logs_text()
            self._set_status("idle")

    def run_full_4d_pipeline(self, client, object_text: str, relation_hint: str) -> None:
        if not self._is_enabled(client):
            return
        runtime_client = self._make_detached_pipeline_client(client, tag="full4d")
        self._safe_notify(
            client,
            "Full 4D Pipeline",
            "Pipeline detached from frontend connection; it will keep running if Viser disconnects.",
            color="blue",
            sec=5.0,
        )

        def _task():
            def _do() -> dict[str, Any]:
                out = self._run_full_4d_pipeline_task(
                    runtime_client,
                    object_text=object_text,
                    relation_hint=relation_hint,
                    ui_client=client,
                )
                try:
                    if bool(out.get("ok", False)):
                        client.add_notification(
                            "Full 4D Pipeline",
                            "Completed full 4D generation + validation loop.",
                            color="green",
                            auto_close_seconds=8,
                        )
                    else:
                        client.add_notification(
                            "Full 4D Pipeline",
                            f"Finished with issues. errors={len(list(out.get('errors', []) or []))}",
                            color="orange",
                            auto_close_seconds=10,
                        )
                except Exception:
                    pass
                return out

            self._run_with_intermediate_capture(
                "pipeline_full_4d",
                {"object_text": str(object_text), "relation_hint": str(relation_hint)},
                _do,
            )

        self._run_async(_task)

    def run_full_4d_pipeline_sync(self, client, object_text: str, relation_hint: str) -> dict[str, Any]:
        if not self._is_enabled(client):
            return {"ok": False, "reason": "plugin_disabled"}
        runtime_client = self._make_detached_pipeline_client(client, tag="full4d_sync")

        def _do() -> dict[str, Any]:
            return self._run_full_4d_pipeline_task(
                runtime_client,
                object_text=object_text,
                relation_hint=relation_hint,
                ui_client=client,
            )

        return self._run_with_intermediate_capture(
            "pipeline_full_4d_sync",
            {"object_text": str(object_text), "relation_hint": str(relation_hint)},
            _do,
        )

    def _run_full_pipeline_task(self, client, object_text: str, relation_hint: str) -> dict[str, Any]:
        self._set_status("running: full_pipeline")
        out0 = self.video_agent.run(client, object_text=object_text, relation_hint=relation_hint)
        if not bool(out0.get("ok", False)):
            packed = {
                "ok": False,
                "reason": "agent0_failed",
                "agent0": out0,
                "targets": [],
                "items": [],
            }
            self._dump_artifact("pipeline_failed_agent0", {"agent0": out0})
            self._dump_artifact("pipeline_full_3d", packed)
            self._set_scene_graph_text()
            self._set_scene_graph_detail_text()
            self._set_logs_text()
            self._set_status("idle")
            return packed

        targets = self._init_agent1_targets_from_scene_graph(reset_cursor=True)
        if not targets:
            packed = {
                "ok": False,
                "reason": "no_targets_from_agent0",
                "agent0": out0,
                "targets": [],
                "items": [],
            }
            self._dump_artifact("pipeline_full_3d", packed)
            self._set_scene_graph_text()
            self._set_scene_graph_detail_text()
            self._set_logs_text()
            self._set_status("idle")
            return packed

        items: list[dict[str, Any]] = []
        agent1_ok_count = 0
        agent2_ok_count = 0
        agent2_mode = "v2_direct_shift_poseopt" if self._use_pose_adjust_v2_for_pipeline() else "v1_candidate_search"
        for i, target_name in enumerate(list(targets)):
            with self.state.lock:
                self.state.agent1_target_index = int(i)
            self._set_object_description(str(target_name))
            self._set_status(
                f"running: full_pipeline [{int(i + 1)}/{len(targets)}] "
                f"agent1+agent2 target={str(target_name)}"
            )

            out1 = self.click_agent.run(client, object_text=str(target_name))
            if bool(out1.get("ok", False)) and int(out1.get("generated_count", 0)) > 0:
                agent1_ok_count += 1

            out2: dict[str, Any] = {
                "ok": False,
                "skipped": True,
                "reason": "agent1_failed_or_not_generated",
            }
            used_agent2_mode = agent2_mode
            if bool(out1.get("ok", False)) and int(out1.get("generated_count", 0)) > 0:
                out2, used_agent2_mode = self._run_agent2_for_pipeline(client, object_name=str(target_name))
                if bool(out2.get("ok", False)):
                    agent2_ok_count += 1

            item = {
                "target_index": int(i),
                "target": str(target_name),
                "agent1": out1,
                "agent2": out2,
                "agent2_mode": str(used_agent2_mode),
            }
            items.append(item)
            safe_target = str(target_name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
            self._dump_artifact(f"pipeline_item_{int(i + 1):02d}_{safe_target}", item)

        packed = {
            "ok": bool(agent1_ok_count == len(targets) and agent2_ok_count == len(targets)),
            "agent0": out0,
            "agent2_mode": str(agent2_mode),
            "targets": [str(x) for x in list(targets)],
            "target_count": int(len(targets)),
            "agent1_ok_count": int(agent1_ok_count),
            "agent2_ok_count": int(agent2_ok_count),
            "items": items,
        }
        self._dump_artifact("pipeline_full_3d", packed)
        self._set_scene_graph_text()
        self._set_scene_graph_detail_text()
        self._set_logs_text()
        try:
            if bool(packed.get("ok", False)):
                client.add_notification(
                    "Full 3D Pipeline",
                    f"Completed all targets: {int(agent2_ok_count)}/{len(targets)}",
                    color="green",
                    auto_close_seconds=6,
                )
            else:
                client.add_notification(
                    "Full 3D Pipeline",
                    (
                        f"Finished with partial success. "
                        f"Agent1: {int(agent1_ok_count)}/{len(targets)}, "
                        f"Agent2: {int(agent2_ok_count)}/{len(targets)}"
                    ),
                    color="orange",
                    auto_close_seconds=8,
                )
        except Exception:
            pass
        self._set_status("idle")
        return packed

    def run_full_pipeline(self, client, object_text: str, relation_hint: str) -> None:
        if not self._is_enabled(client):
            return

        def _task():
            self._run_with_intermediate_capture(
                "pipeline_full_3d",
                {"object_text": str(object_text), "relation_hint": str(relation_hint)},
                lambda: self._run_full_pipeline_task(client, object_text=object_text, relation_hint=relation_hint),
            )

        self._run_async(_task)

    def run_full_pipeline_sync(self, client, object_text: str, relation_hint: str) -> dict[str, Any]:
        if not self._is_enabled(client):
            return {"ok": False, "reason": "plugin_disabled"}
        return self._run_with_intermediate_capture(
            "pipeline_full_3d_sync",
            {"object_text": str(object_text), "relation_hint": str(relation_hint)},
            lambda: self._run_full_pipeline_task(client, object_text=object_text, relation_hint=relation_hint),
        )

    def headless_3d_gen(
        self,
        client,
        object_text: str,
        relation_hint: str,
        *,
        run_async: bool = False,
    ) -> dict[str, Any]:
        if bool(run_async):
            self.run_full_pipeline(client, object_text=object_text, relation_hint=relation_hint)
            return {"ok": True, "submitted": True, "mode": "async"}
        return self.run_full_pipeline_sync(client, object_text=object_text, relation_hint=relation_hint)

    def sync_ui(self) -> None:
        with self.state.lock:
            has_targets = bool(getattr(self.state, "agent1_targets", []))
            has_graph = bool(self.state.scene_graph)
            cur_desc = str(getattr(self.state, "object_description", "") or "").strip()
        if (not has_targets) and has_graph:
            names = self._init_agent1_targets_from_scene_graph(reset_cursor=True)
            if names and not cur_desc:
                self._set_object_description(names[0])
        self._set_scene_graph_text()
        self._set_scene_graph_detail_text()
        self._set_agent4_result_text()
        self._set_logs_text()
        self._set_status("idle")
