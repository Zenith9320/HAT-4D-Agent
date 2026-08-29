from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageDraw

from ..schemas import normalize_temporal_scene_graph
from ..skills import (
    Agent1SkillBundle,
    SKILL_SAM3D_ENSURE_SESSION,
    SKILL_SAM3D_GENERATE_AND_WAIT,
    SKILL_SAM3D_NEW_OBJECT,
    SKILL_SAM_EVALUATE_MASK,
    SKILL_SAM_SEGMENT_WITH_POINTS,
    SKILL_VLM_MASK_EVALUATOR,
    SKILL_VLM_POINT_PROMPT_PLANNER,
)
from ..state import AgentLogEntry

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class _AgentBase:
    def __init__(self, ctx, state, bridge, vlm_client, *, agent_name: str) -> None:
        self.ctx = ctx
        self.state = state
        self.bridge = bridge
        self.vlm_client = vlm_client
        self.agent_name = str(agent_name)

    def log(self, status: str, message: str) -> None:
        entry = AgentLogEntry(
            ts=datetime.now().strftime("%H:%M:%S"),
            agent=self.agent_name,
            status=str(status),
            message=str(message),
        )
        with self.state.lock:
            self.state.logs.append(entry)
            self.state.logs = self.state.logs[-int(self.state.max_logs) :]

    @staticmethod
    def _notify(client, title: str, body: str, color: str = "blue", sec: float = 4.0) -> None:
        try:
            client.add_notification(
                str(title),
                str(body),
                color=str(color),
                auto_close_seconds=float(sec),
            )
        except Exception:
            pass


class VideoUnderstandingAgent(_AgentBase):
    def __init__(self, ctx, state, bridge, vlm_client) -> None:
        super().__init__(ctx, state, bridge, vlm_client, agent_name="agent0_video")

    @staticmethod
    def _resize_long_side(rgb_u8: np.ndarray, max_side: int) -> np.ndarray:
        if int(max_side) <= 0:
            return rgb_u8
        h, w = rgb_u8.shape[:2]
        side = max(h, w)
        if side <= int(max_side):
            return rgb_u8
        scale = float(max_side) / float(side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pil = Image.fromarray(rgb_u8)
        pil = pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(white_bg, rgba)
            return np.asarray(composited.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _sort_paths(paths: list[Path]) -> list[Path]:
        digit_named = [p for p in paths if p.stem.isdigit()]
        if digit_named:
            return sorted(digit_named, key=lambda p: int(p.stem))
        return sorted(paths, key=lambda p: p.name)

    @staticmethod
    def _uniform_sample_indices(total: int, max_frames: int) -> list[int]:
        if total <= 0:
            return []
        if max_frames <= 0 or total <= max_frames:
            return list(range(total))
        xs = np.linspace(0, total - 1, num=max_frames, dtype=np.int64)
        return sorted({int(x) for x in xs})

    @staticmethod
    def _stride_sample_indices(total: int, stride: int) -> list[int]:
        if total <= 0:
            return []
        s = max(1, int(stride))
        return list(range(0, int(total), s))

    def _resolve_local_frame_dir_override(self) -> Path | None:
        with self.state.lock:
            backend_mode = str(getattr(self.state, "agent0_backend_mode", "official_api") or "official_api")
        if backend_mode != "local_openai_compatible":
            return None
        raw = str(getattr(self.ctx.app, "vlm_local_video_url", "") or "").strip()
        if not raw:
            return None
        candidate: Path | None = None
        low = raw.lower()
        if low.startswith("file://"):
            candidate = Path(unquote(raw[len("file://") :])).expanduser()
        elif raw.startswith("/") or raw.startswith("./") or raw.startswith("../") or (len(raw) > 1 and raw[1] == ":"):
            candidate = Path(raw).expanduser()
        if candidate is None:
            return None
        try:
            candidate = candidate.resolve()
        except Exception:
            pass
        if candidate.exists() and candidate.is_dir():
            return candidate
        return None

    def _resolve_video_paths(self) -> list[Path]:
        app = self.ctx.app
        input_dir = self._resolve_local_frame_dir_override() or Path(getattr(app, "input_dir", ""))
        if not input_dir.exists():
            return []

        frame_tokens = list(getattr(getattr(app, "sequence", None), "frame_tokens", []) or [])
        resolved: list[Path] = []

        for tok in frame_tokens:
            p = None
            for ext in _IMAGE_EXTS:
                cand = input_dir / f"{tok}{ext}"
                if cand.exists():
                    p = cand
                    break
            if p is not None:
                resolved.append(p)

        if resolved:
            return resolved

        generic = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        return self._sort_paths(generic)

    def _collect_frame_items(self) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        frame_paths = self._resolve_video_paths()
        with self.state.lock:
            render_size = int(getattr(self.state, "agent0_render_size", 384))
            max_frames = int(getattr(self.state, "agent0_max_frames", 0))

        items: list[dict[str, Any]] = []
        tokens: list[str] = []
        path_strs: list[str] = []
        for p in frame_paths:
            try:
                rgb = self._load_rgb(p)
                rgb = self._resize_long_side(rgb, render_size)
            except Exception:
                continue
            tok = str(p.stem)
            items.append({"frame_token": tok, "rgb_u8": rgb})
            tokens.append(tok)
            path_strs.append(str(p.resolve()))

        # Auto mode (max_frames<=0): sample every 4 frames (~25% of total frames).
        if int(max_frames) <= 0:
            keep = self._stride_sample_indices(len(items), stride=4)
        else:
            keep = self._uniform_sample_indices(len(items), max_frames=max_frames)
        items = [items[i] for i in keep]
        tokens = [tokens[i] for i in keep]
        path_strs = [path_strs[i] for i in keep]
        return items, tokens, path_strs

    def _save_sampled_frames_debug(
        self,
        *,
        frame_items: list[dict[str, Any]],
        frame_tokens: list[str],
        frame_paths: list[str],
    ) -> None:
        with self.state.lock:
            detailed_enabled = bool(getattr(self.state, "save_intermediate_results", False))
            root = getattr(self.state, "artifact_root", None)
        if (not detailed_enabled) or root is None:
            return
        try:
            out_dir = Path(root) / "agent0_video_understanding"
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        rows: list[dict[str, Any]] = []
        for i, (item, tok, src_path) in enumerate(zip(frame_items, frame_tokens, frame_paths), start=1):
            rgb = item.get("rgb_u8", None)
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or int(rgb.shape[2]) < 3:
                continue
            safe_tok = str(tok).replace("/", "_").replace("\\", "_").replace(" ", "_")
            file_name = f"{int(i):04d}_{safe_tok}.png"
            out_path = out_dir / file_name
            try:
                Image.fromarray(rgb[:, :, :3].astype(np.uint8, copy=False)).save(out_path)
            except Exception:
                continue
            rows.append(
                {
                    "index": int(i),
                    "frame_token": str(tok),
                    "source_path": str(src_path),
                    "saved_path": str(out_path),
                    "shape": [int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])],
                }
            )
        try:
            (out_dir / "sampled_frames_manifest.json").write_text(
                json.dumps(
                    {
                        "saved_count": int(len(rows)),
                        "frame_tokens": [str(x) for x in frame_tokens],
                        "rows": rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _update_state_success(self, packed: dict[str, Any], scene_graph: dict[str, Any], frame_count: int) -> None:
        with self.state.lock:
            self.state.scene_graph = dict(scene_graph)
            self.state.last_video_understanding = dict(packed)
            self.state.agent0_last_source = str(packed.get("provider", packed.get("source", "unknown")))
            self.state.agent0_last_frame_count = int(frame_count)

    def run(self, client, object_text: str, relation_hint: str) -> dict[str, Any]:
        with self.state.lock:
            use_gpt4o = bool(getattr(self.state, "agent0_use_gpt4o", True))
            backend_mode = str(getattr(self.state, "agent0_backend_mode", "official_api") or "official_api")
            ikg_enabled = bool(getattr(self.state, "agent0_ikg_enabled", True))
        if backend_mode not in {"official_api", "local_openai_compatible"}:
            backend_mode = "official_api"

        preflight = self.vlm_client.preflight_video_understanding(
            backend_mode=backend_mode,
            use_gpt4o=use_gpt4o,
        )
        for line in preflight.get("checks", []):
            self.log("check", str(line))
        for line in preflight.get("warnings", []):
            self.log("warn", str(line))
        if not bool(preflight.get("ok", False)):
            reason_lines = list(preflight.get("errors", [])) + list(preflight.get("hints", []))
            text = "Agent0 preflight failed: " + (" | ".join(reason_lines[:2]) if reason_lines else "unknown")
            self.log("fail", text)
            self._notify(client, "Agent0 preflight", text, color="red", sec=8.0)
            return {
                "ok": False,
                "reason": "preflight_failed",
                "preflight": preflight,
            }

        self.log("start", "collect_frames")
        frame_items, frame_tokens, frame_paths = self._collect_frame_items()
        if not frame_items:
            msg = "no_input_frames_found"
            self.log("fail", msg)
            self._notify(client, "Agent0", "No input frames found for video understanding.", color="red")
            return {"ok": False, "reason": msg}
        self._save_sampled_frames_debug(
            frame_items=frame_items,
            frame_tokens=frame_tokens,
            frame_paths=frame_paths,
        )

        self.log("info", f"frames_ready:{len(frame_items)}")
        self.log("info", f"backend_mode:{backend_mode}")
        self.log("info", f"ikg_enabled:{ikg_enabled}")

        attempts: list[dict[str, Any]] = []

        out = None
        if backend_mode == "official_api":
            # Preferred path: VLM_toolkit_cross_gpfs (Machine-B -> GPFS -> Machine-A).
            if use_gpt4o:
                out = self.vlm_client.run_video_understanding_vlm_toolkit(
                    object_text=str(object_text),
                    relation_hint=str(relation_hint),
                    frame_items=frame_items,
                    frame_tokens=frame_tokens,
                    timeout=300.0,
                    ikg_enabled=bool(ikg_enabled),
                )
                attempts.append({"backend": "vlm_toolkit_cross_gpfs", "result": out})

            # Fallback path 1: direct official OpenAI API.
            if use_gpt4o and not (isinstance(out, dict) and bool(out.get("ok"))):
                out = self.vlm_client.run_video_understanding_gpt4o(
                    object_text=str(object_text),
                    relation_hint=str(relation_hint),
                    frame_items=frame_items,
                    frame_tokens=frame_tokens,
                    timeout=300.0,
                    ikg_enabled=bool(ikg_enabled),
                )
                attempts.append({"backend": "gpt4o_direct", "result": out})
        else:
            # Local OpenAI-compatible endpoint path.
            out = self.vlm_client.run_video_understanding_local_openai(
                object_text=str(object_text),
                relation_hint=str(relation_hint),
                frame_items=frame_items,
                frame_tokens=frame_tokens,
                timeout=300.0,
                ikg_enabled=bool(ikg_enabled),
            )
            attempts.append({"backend": "local_openai_compatible", "result": out})

        # Fallback path 2: external HTTP VLM backend if configured.
        if not (isinstance(out, dict) and bool(out.get("ok"))):
            out = self.vlm_client.run_agent(
                "video_understanding",
                {
                    "object_text": str(object_text),
                    "relation_hint": str(relation_hint),
                    "frame_paths": frame_paths,
                    "frame_tokens": frame_tokens,
                    "max_frames": 0,  # Use full video frames.
                    "ikg_enabled": bool(ikg_enabled),
                },
                timeout=300.0,
            )
            attempts.append({"backend": "http_agent_backend", "result": out})

        if not (isinstance(out, dict) and bool(out.get("ok"))):
            msg = "video_understanding_backend_failed"
            attempt_msgs: list[str] = []
            for item in attempts:
                b = str(item.get("backend", "unknown"))
                r = item.get("result")
                if isinstance(r, dict):
                    attempt_msgs.append(f"{b}:{r.get('error', r.get('reason', 'fail'))}")
                else:
                    attempt_msgs.append(f"{b}:no_response")
            if attempt_msgs:
                self.log("fail", "backend_attempts=" + ";".join(attempt_msgs))
            self.log("fail", msg)
            hint = "; ".join(attempt_msgs[:2]) if attempt_msgs else "Check logs and toolkit server."
            self._notify(client, "Agent0", f"VLM backend failed: {hint}", color="red", sec=8.0)
            return {
                "ok": False,
                "reason": msg,
                "raw": out,
                "preflight": preflight,
                "attempts": attempts,
            }

        raw_graph = out.get("scene_graph", {})
        scene_graph = normalize_temporal_scene_graph(raw_graph, frame_tokens=frame_tokens)
        if not bool(ikg_enabled):
            strip_fn = getattr(self.vlm_client, "strip_interaction_knowledge_graph", None)
            if callable(strip_fn):
                scene_graph = strip_fn(scene_graph)
        packed = dict(out)
        packed["scene_graph"] = scene_graph
        packed["frame_tokens"] = frame_tokens
        packed["frames_total"] = len(frame_tokens)
        packed["ikg_enabled"] = bool(ikg_enabled)
        if not bool(ikg_enabled):
            packed["ablation_mode"] = "wo_IKG"
        self._update_state_success(packed, scene_graph, len(frame_tokens))

        provider = str(packed.get("provider", packed.get("source", "unknown")))
        self.log("ok", f"provider={provider},frames={len(frame_tokens)}")
        self._notify(client, "Agent0", f"Video understanding done ({provider}, {len(frame_tokens)} frames).")
        return packed


class ClickAgent(_AgentBase):
    def __init__(self, ctx, state, bridge, vlm_client) -> None:
        super().__init__(ctx, state, bridge, vlm_client, agent_name="agent1_click")
        self.skill_bundle = Agent1SkillBundle(bridge=bridge, vlm_client=vlm_client)

    @staticmethod
    def _split_object_text(text: str) -> list[str]:
        s = str(text or "")
        for sep in ["，", ";", "；", "|", "\n", "\t"]:
            s = s.replace(sep, ",")
        out: list[str] = []
        seen: set[str] = set()
        for part in s.split(","):
            name = str(part).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
        return out

    def _resolve_target_object(self, object_text: str) -> str:
        def _to_english_name(raw_name: str) -> str:
            s = str(raw_name).strip()
            if not s:
                return s
            fn = getattr(self.vlm_client, "to_english_object_name", None)
            if not callable(fn):
                return s
            try:
                out = str(fn(s)).strip()
                return out if out else s
            except Exception:
                return s

        names = self._split_object_text(object_text)
        if names:
            return _to_english_name(names[0])
        with self.state.lock:
            graph = dict(getattr(self.state, "scene_graph", {}) or {})
            fallback_desc = str(getattr(self.state, "object_description", "") or "")
            queue = list(getattr(self.state, "agent1_targets", []) or [])
            qidx = int(getattr(self.state, "agent1_target_index", 0))

        names = []
        for x in graph.get("objects", []):
            s = str(x).strip()
            if s:
                names.append(s)
        for x in graph.get("object_meta", []):
            if isinstance(x, dict):
                s = str(x.get("name", x.get("id", ""))).strip()
                if s:
                    names.append(s)

        if not names and queue:
            qidx = max(0, min(len(queue) - 1, qidx))
            return _to_english_name(str(queue[qidx]).strip())
        if not names and fallback_desc:
            names.extend(self._split_object_text(fallback_desc))

        return _to_english_name(str(names[0]).strip()) if names else ""

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(white_bg, rgba)
            return np.asarray(composited.convert("RGB"), dtype=np.uint8)

    def _current_frame_rgb(self) -> np.ndarray | None:
        app = self.ctx.app
        try:
            frame_token = str(app.sequence.frame_tokens[app.current_frame])
            img_path = Path(app.workspace.image_path_for_view(frame_token, "input"))
        except Exception:
            return None
        if not img_path.exists():
            return None
        try:
            return self._load_rgb(img_path)
        except Exception:
            return None

    def _collect_non_target_names(self, target_name: str) -> list[str]:
        non_target_names: list[str] = []
        with self.state.lock:
            all_targets = list(getattr(self.state, "agent1_targets", []) or [])
            if not all_targets:
                g = dict(getattr(self.state, "scene_graph", {}) or {})
                all_targets = [str(x).strip() for x in list(g.get("objects", []) or []) if str(x).strip()]
        target_low = str(target_name).strip().lower()
        to_en = getattr(self.vlm_client, "to_english_object_name", None)
        for n in all_targets:
            raw = str(n).strip()
            if not raw:
                continue
            cand = raw
            if callable(to_en):
                try:
                    en = str(to_en(raw)).strip()
                    if en:
                        cand = en
                except Exception:
                    pass
            if str(cand).strip().lower() != target_low:
                non_target_names.append(str(cand).strip())
        return non_target_names

    def _current_object_mask_and_points(self) -> tuple[np.ndarray | None, list[list[int]], list[list[int]], dict[str, Any]]:
        sam3d_state = getattr(self.ctx, "sam3d_state", None)
        if sam3d_state is None:
            return None, [], [], {"ok": False, "reason": "sam3d_state_unavailable"}
        try:
            with sam3d_state.lock:
                idx = int(getattr(sam3d_state, "current_object_idx", 0))
                objects = list(getattr(sam3d_state, "objects", []) or [])
                if not (0 <= idx < len(objects)):
                    return None, [], [], {
                        "ok": False,
                        "reason": "current_object_out_of_range",
                        "current_object_idx": idx,
                        "n_objects": len(objects),
                    }
                obj = objects[idx]
                mask = getattr(obj, "mask", None)
                pts_raw = list(getattr(obj, "points_xy", []) or [])
                labels_raw = list(getattr(obj, "point_labels", []) or [])
        except Exception as exc:
            return None, [], [], {"ok": False, "reason": f"sam3d_state_read_failed:{type(exc).__name__}:{exc}"}

        positive: list[list[int]] = []
        negative: list[list[int]] = []
        n = min(len(pts_raw), len(labels_raw))
        for i in range(n):
            p = pts_raw[i]
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
                lab = int(labels_raw[i])
            except Exception:
                continue
            if lab == 1:
                positive.append([x, y])
            else:
                negative.append([x, y])
        meta = {
            "ok": True,
            "points_total": int(n),
            "points_positive": int(len(positive)),
            "points_negative": int(len(negative)),
        }
        return (np.asarray(mask).copy() if mask is not None else None), positive, negative, meta

    def _evaluate_mask_once(
        self,
        *,
        object_name: str,
        frame_rgb_u8: np.ndarray,
        mask_bool: np.ndarray | None,
        positive_points: list[Any],
        negative_points: list[Any],
        backend_mode: str,
        non_target_names: list[str] | None = None,
        filtered_prompt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        eva_heuristic = self.skill_bundle.registry.call(
            SKILL_SAM_EVALUATE_MASK,
            positive_points=positive_points,
            negative_points=negative_points,
            min_area_ratio=0.0,
            max_area_ratio=1.0,
            min_positive_coverage=0.0,
            max_negative_coverage=1.0,
        )

        vlm_eval: dict[str, Any] = {"ok": False, "evaluated": False, "reason": "not_run"}
        if mask_bool is not None:
            vlm_eval = self.skill_bundle.registry.call(
                SKILL_VLM_MASK_EVALUATOR,
                object_name=str(object_name),
                image_rgb_u8=frame_rgb_u8,
                mask_bool=np.asarray(mask_bool),
                positive_points=positive_points,
                negative_points=negative_points,
                non_target_objects=list(non_target_names or []),
                backend_mode=backend_mode,
                timeout=60.0,
            )
        else:
            vlm_eval = {"ok": False, "evaluated": False, "reason": "mask_missing_after_seg"}

        eva = dict(eva_heuristic or {})
        pos_cov = float(eva_heuristic.get("positive_coverage", 0.0))
        neg_cov = float(eva_heuristic.get("negative_coverage", 1.0))
        point_rule = {
            "positive_in_mask": True,
            "negative_outside_mask": True,
            "positive_coverage_threshold": 0.999,
            "negative_coverage_threshold": 0.001,
        }
        point_ok = (pos_cov >= float(point_rule["positive_coverage_threshold"])) and (
            neg_cov <= float(point_rule["negative_coverage_threshold"])
        )
        qwen_ok = bool(vlm_eval.get("evaluated", False)) and bool(vlm_eval.get("ok", False))
        final_ok = bool(qwen_ok)

        eva["ok_heuristic"] = bool(eva_heuristic.get("ok", False))
        eva["vlm_eval"] = dict(vlm_eval or {})
        eva["point_rule"] = dict(point_rule)
        eva["ok_point_rule"] = bool(point_ok)
        eva["ok_qwen"] = bool(qwen_ok)
        eva["mode"] = "vlm_only_gate"
        eva["ok_vlm"] = bool(vlm_eval.get("ok", False)) if bool(vlm_eval.get("evaluated", False)) else None
        eva["ok"] = bool(final_ok)
        eva["filtered_prompt"] = dict(filtered_prompt or {})
        return {
            "ok": bool(final_ok),
            "evaluate": eva,
            "vlm_eval": dict(vlm_eval or {}),
            "point_ok": bool(point_ok),
            "pos_cov": float(pos_cov),
            "neg_cov": float(neg_cov),
        }

    @staticmethod
    def _safe_name_for_path(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return "object"
        out = []
        for ch in s:
            if ch.isalnum() or ch in {"-", "_"}:
                out.append(ch)
            else:
                out.append("_")
        joined = "".join(out).strip("_")
        return joined[:64] if joined else "object"

    @staticmethod
    def _point_to_rel_xy(point: Any, *, width: int, height: int) -> tuple[float, float] | None:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            return None
        w = max(1, int(width))
        h = max(1, int(height))
        xn = max(0.0, min(1.0, x / float(max(1, w - 1))))
        yn = max(0.0, min(1.0, y / float(max(1, h - 1))))
        return float(xn), float(yn)

    @staticmethod
    def _extract_rel_xy_points(raw: Any) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        if not isinstance(raw, list):
            return out
        for it in raw:
            if not isinstance(it, dict):
                continue
            x = it.get("x", None)
            y = it.get("y", None)
            if x is None or y is None:
                continue
            try:
                xf = float(x)
                yf = float(y)
            except Exception:
                continue
            xf = max(0.0, min(1.0, xf))
            yf = max(0.0, min(1.0, yf))
            out.append((float(xf), float(yf)))
        return out

    @staticmethod
    def _build_point_planner_retry_analysis(
        *,
        vlm_eval: dict[str, Any],
        point_ok: bool,
        pos_cov: float,
        neg_cov: float,
        filtered_positive_points: list[Any],
        filtered_negative_points: list[Any],
        dropped_positive_points: list[Any],
        dropped_negative_points: list[Any],
        non_target_names: list[str] | None,
    ) -> dict[str, Any]:
        vv = dict(vlm_eval or {})
        pd = dict(vv.get("point_diagnostics", {}) or {})
        out = {
            "stage": "mask_eval_failed",
            "summary": str(vv.get("summary", "")).strip(),
            "reason_codes": [str(x) for x in list(vv.get("reason_codes", []) or [])[:16]],
            "point_rule_passed": bool(point_ok),
            "positive_coverage": float(pos_cov),
            "negative_coverage": float(neg_cov),
            "non_target_names": [str(x) for x in list(non_target_names or [])[:8]],
            "missing_parts": list(vv.get("missing_parts", []) or [])[:8],
            "extra_objects": list(vv.get("extra_objects", []) or [])[:8],
            "wrong_positive_points": list(pd.get("wrong_positive_points", []) or [])[:8],
            "wrong_negative_points": list(pd.get("wrong_negative_points", []) or [])[:8],
            "inside_mask_objects": list(vv.get("inside_mask_objects", []) or [])[:8],
            "outside_mask_objects": list(vv.get("outside_mask_objects", []) or [])[:8],
            "target_complete_in_mask": bool(vv.get("target_complete_in_mask", False)),
            "non_target_outside_mask": bool(vv.get("non_target_outside_mask", False)),
            "filtered_positive_points": list(filtered_positive_points or [])[:12],
            "filtered_negative_points": list(filtered_negative_points or [])[:12],
            "dropped_positive_points": list(dropped_positive_points or [])[:12],
            "dropped_negative_points": list(dropped_negative_points or [])[:12],
            "retry_guidance": [
                "Move positive points away from non-target regions and onto missing target parts.",
                "Place negative points at centers of wrongly-included non-target regions.",
                "Avoid placing positive/negative points near contact boundaries.",
            ],
        }
        return out

    @staticmethod
    def _append_unique_rel_points(
        dst: list[tuple[float, float]],
        src: list[tuple[float, float]],
        *,
        min_dist: float = 1e-3,
    ) -> None:
        r2 = float(min_dist) * float(min_dist)
        for x, y in src:
            keep = True
            for xx, yy in dst:
                dx = float(x - xx)
                dy = float(y - yy)
                if dx * dx + dy * dy <= r2:
                    keep = False
                    break
            if keep:
                dst.append((float(x), float(y)))

    @staticmethod
    def _filter_points_by_blacklist(
        points: list[Any],
        *,
        blocked_rel: list[tuple[float, float]],
        width: int,
        height: int,
        radius_rel: float = 0.035,
    ) -> tuple[list[list[int]], list[list[int]]]:
        kept: list[list[int]] = []
        dropped: list[list[int]] = []
        if not points:
            return kept, dropped

        w = max(1, int(width))
        h = max(1, int(height))
        rr = float(radius_rel) * float(radius_rel)
        for p in list(points):
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
            except Exception:
                continue
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            rel = ClickAgent._point_to_rel_xy([x, y], width=w, height=h)
            if rel is None:
                continue
            xr, yr = rel
            blocked = False
            for bx, by in blocked_rel:
                dx = float(xr - bx)
                dy = float(yr - by)
                if dx * dx + dy * dy <= rr:
                    blocked = True
                    break
            if blocked:
                dropped.append([int(x), int(y)])
            else:
                kept.append([int(x), int(y)])
        return kept, dropped

    def _agent1_debug_enabled(self) -> bool:
        with self.state.lock:
            flag = bool(getattr(self.state, "agent1_debug_save_eval", False))
        if flag:
            return True
        app = getattr(self.ctx, "app", None)
        if app is None:
            return False
        if bool(getattr(app, "debug_mode", False)):
            return True
        if bool(getattr(app, "debug", False)):
            return True
        return False

    def _agent1_debug_dir(self) -> Path | None:
        with self.state.lock:
            root = getattr(self.state, "artifact_root", None)
        if root is None:
            return None
        try:
            d = Path(root) / "debug_agent1_eval"
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            return None

    def _dump_eval_debug_artifacts(
        self,
        *,
        object_name: str,
        attempt_index: int,
        planning_image_rgb_u8: np.ndarray | None,
        positive_points: list[Any],
        negative_points: list[Any],
        point_plan: dict[str, Any],
        segment: dict[str, Any],
        evaluate: dict[str, Any],
    ) -> None:
        if not self._agent1_debug_enabled():
            return
        out_dir = self._agent1_debug_dir()
        if out_dir is None:
            return

        sam3d_state = getattr(self.ctx, "sam3d_state", None)
        if sam3d_state is None:
            return

        img: np.ndarray | None = None
        mask: np.ndarray | None = None
        sam_points_xy: list[tuple[int, int]] = []
        sam_point_labels: list[int] = []
        try:
            with sam3d_state.lock:
                cur_img = getattr(sam3d_state, "image_rgb_u8", None)
                if cur_img is not None:
                    img = np.asarray(cur_img).copy()
                objects = list(getattr(sam3d_state, "objects", []) or [])
                idx = int(getattr(sam3d_state, "current_object_idx", 0))
                if 0 <= idx < len(objects):
                    cur_obj = objects[idx]
                    cur_mask = getattr(cur_obj, "mask", None)
                    if cur_mask is not None:
                        mask = np.asarray(cur_mask).copy()
                    sam_points_xy = [
                        (int(p[0]), int(p[1]))
                        for p in list(getattr(cur_obj, "points_xy", []) or [])
                        if isinstance(p, (list, tuple)) and len(p) >= 2
                    ]
                    sam_point_labels = [int(x) for x in list(getattr(cur_obj, "point_labels", []) or [])]
        except Exception:
            return
        if img is None or mask is None:
            return

        try:
            if img.ndim != 3 or img.shape[2] < 3:
                return
            mask_bool = (mask > 0).astype(np.uint8)
            h = min(int(img.shape[0]), int(mask_bool.shape[0]))
            w = min(int(img.shape[1]), int(mask_bool.shape[1]))
            if h <= 0 or w <= 0:
                return
            img = img[:h, :w, :3].astype(np.uint8, copy=False)
            mask_bool = mask_bool[:h, :w]

            color = np.zeros_like(img, dtype=np.uint8)
            color[:, :, 1] = 255
            overlay = img.astype(np.float32)
            m3 = mask_bool[..., None].astype(np.float32)
            overlay = overlay * (1.0 - 0.45 * m3) + color.astype(np.float32) * (0.45 * m3)
            overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)

            # VLM point-only overlay (raw image).
            vlm_points_pil = Image.fromarray(img.copy())
            vlm_draw = ImageDraw.Draw(vlm_points_pil)
            for i, p in enumerate(list(positive_points or []), start=1):
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue
                r = 6
                vlm_draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0), outline=(255, 255, 255), width=2)
                vlm_draw.text((x + 8, y - 8), f"P{i}", fill=(0, 255, 0))
            for i, p in enumerate(list(negative_points or []), start=1):
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue
                r = 6
                vlm_draw.rectangle((x - r, y - r, x + r, y + r), fill=(255, 215, 0), outline=(255, 255, 255), width=2)
                vlm_draw.text((x + 8, y - 8), f"N{i}", fill=(255, 215, 0))

            # SAM effective point overlay (may include auto-added negatives).
            sam_points_pil = Image.fromarray(img.copy())
            sam_draw = ImageDraw.Draw(sam_points_pil)
            for i, (xy, lab) in enumerate(zip(sam_points_xy, sam_point_labels), start=1):
                x, y = int(xy[0]), int(xy[1])
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue
                r = 6
                if int(lab) == 1:
                    sam_draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0), outline=(255, 255, 255), width=2)
                    sam_draw.text((x + 8, y - 8), f"SP{i}", fill=(0, 255, 0))
                else:
                    sam_draw.rectangle((x - r, y - r, x + r, y + r), fill=(255, 80, 80), outline=(255, 255, 255), width=2)
                    sam_draw.text((x + 8, y - 8), f"SN{i}", fill=(255, 80, 80))

            # Mask + VLM points overlay.
            mask_overlay_pil = Image.fromarray(overlay_u8)
            mask_draw = ImageDraw.Draw(mask_overlay_pil)
            for p in list(positive_points or []):
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue
                r = 6
                mask_draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0), outline=(255, 255, 255), width=2)
            for p in list(negative_points or []):
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue
                r = 6
                mask_draw.rectangle((x - r, y - r, x + r, y + r), fill=(255, 215, 0), outline=(255, 255, 255), width=2)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = self._safe_name_for_path(object_name)
            stem = f"{ts}_{name}_attempt_{int(attempt_index):02d}"

            # Save the exact image used for VLM point planning.
            if planning_image_rgb_u8 is not None:
                try:
                    plan_img = np.asarray(planning_image_rgb_u8)
                    if plan_img.ndim == 3 and plan_img.shape[2] >= 3:
                        ph = int(plan_img.shape[0])
                        pw = int(plan_img.shape[1])
                        plan_rgb = plan_img[:, :, :3].astype(np.uint8, copy=False)
                        plan_pil = Image.fromarray(plan_rgb)
                        plan_pil.save(out_dir / f"{stem}_planning_source.png")
                        plan_pts_pil = Image.fromarray(plan_rgb.copy())
                        plan_draw = ImageDraw.Draw(plan_pts_pil)
                        for i, p in enumerate(list(positive_points or []), start=1):
                            if not isinstance(p, (list, tuple)) or len(p) < 2:
                                continue
                            x = int(round(float(p[0])))
                            y = int(round(float(p[1])))
                            if x < 0 or x >= pw or y < 0 or y >= ph:
                                continue
                            r = 6
                            plan_draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0), outline=(255, 255, 255), width=2)
                            plan_draw.text((x + 8, y - 8), f"P{i}", fill=(0, 255, 0))
                        for i, p in enumerate(list(negative_points or []), start=1):
                            if not isinstance(p, (list, tuple)) or len(p) < 2:
                                continue
                            x = int(round(float(p[0])))
                            y = int(round(float(p[1])))
                            if x < 0 or x >= pw or y < 0 or y >= ph:
                                continue
                            r = 6
                            plan_draw.rectangle((x - r, y - r, x + r, y + r), fill=(255, 215, 0), outline=(255, 255, 255), width=2)
                            plan_draw.text((x + 8, y - 8), f"N{i}", fill=(255, 215, 0))
                        plan_pts_pil.save(out_dir / f"{stem}_planning_source_with_vlm_points.png")
                except Exception:
                    pass

            Image.fromarray((mask_bool * 255).astype(np.uint8)).save(out_dir / f"{stem}_mask.png")
            mask_overlay_pil.save(out_dir / f"{stem}_overlay.png")
            vlm_points_pil.save(out_dir / f"{stem}_vlm_points.png")
            sam_points_pil.save(out_dir / f"{stem}_sam_points.png")

            payload = {
                "object_name": str(object_name),
                "attempt": int(attempt_index),
                "positive_points": list(positive_points or []),
                "negative_points": list(negative_points or []),
                "sam_points_xy": [[int(x), int(y)] for (x, y) in sam_points_xy],
                "sam_point_labels": [int(x) for x in sam_point_labels],
                "planning_image_shape": (
                    [int(planning_image_rgb_u8.shape[0]), int(planning_image_rgb_u8.shape[1])]
                    if isinstance(planning_image_rgb_u8, np.ndarray) and planning_image_rgb_u8.ndim >= 2
                    else None
                ),
                "point_plan": dict(point_plan or {}),
                "segment": dict(segment or {}),
                "evaluate": dict(evaluate or {}),
                "image_shape": [int(h), int(w)],
            }
            (out_dir / f"{stem}_eval.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _segment_object_with_retries(
        self,
        client,
        *,
        object_name: str,
        frame_rgb_u8: np.ndarray,
        points_per_object: int,
        max_retries: int,
        min_area_ratio: float,
        backend_mode: str,
        non_target_names: list[str] | None = None,
    ) -> dict[str, Any]:
        attempt_logs: list[dict[str, Any]] = []
        prior_fail_reason = ""
        prior_fail_analysis: dict[str, Any] = {}
        blocked_positive_rel: list[tuple[float, float]] = []
        blocked_negative_rel: list[tuple[float, float]] = []
        ih = int(frame_rgb_u8.shape[0]) if hasattr(frame_rgb_u8, "shape") and len(frame_rgb_u8.shape) >= 2 else 1
        iw = int(frame_rgb_u8.shape[1]) if hasattr(frame_rgb_u8, "shape") and len(frame_rgb_u8.shape) >= 2 else 1

        for attempt in range(max(1, int(max_retries))):
            extra_pos = 0 if attempt == 0 else min(2, attempt)
            n_pos = int(points_per_object) + int(extra_pos)
            n_neg = 0 if attempt == 0 else 1
            pts = self.skill_bundle.registry.call(
                SKILL_VLM_POINT_PROMPT_PLANNER,
                object_name=str(object_name),
                image_rgb_u8=frame_rgb_u8,
                non_target_objects=list(non_target_names or []),
                points_per_object=n_pos,
                negative_points=n_neg,
                prior_fail_reason=prior_fail_reason,
                prior_fail_analysis=dict(prior_fail_analysis or {}),
                backend_mode=backend_mode,
                timeout=60.0,
            )


            if not bool(pts.get("ok", False)):
                prior_fail_reason = str(pts.get("reason", "point_planning_failed"))
                attempt_logs.append(
                    {
                        "attempt": int(attempt + 1),
                        "ok": False,
                        "stage": "point_planning",
                        "n_pos": int(n_pos),
                        "n_neg": int(n_neg),
                        "reason": prior_fail_reason,
                        "raw": pts,
                    }
                )
                continue

            positive_raw = list(pts.get("positive_points", []) or [])
            negative_raw = list(pts.get("negative_points", []) or [])
            positive, dropped_pos = self._filter_points_by_blacklist(
                positive_raw,
                blocked_rel=blocked_positive_rel,
                width=iw,
                height=ih,
            )
            negative, dropped_neg = self._filter_points_by_blacklist(
                negative_raw,
                blocked_rel=blocked_negative_rel,
                width=iw,
                height=ih,
            )
            if not positive:
                prior_fail_reason = (
                    f"all_positive_points_filtered:blocked_pos={len(blocked_positive_rel)},"
                    f"dropped_pos={len(dropped_pos)}"
                )
                prior_fail_analysis = {
                    "stage": "point_filter_failed",
                    "reason": str(prior_fail_reason),
                    "blocked_positive_rel": [[float(x), float(y)] for (x, y) in blocked_positive_rel],
                    "blocked_negative_rel": [[float(x), float(y)] for (x, y) in blocked_negative_rel],
                    "dropped_positive_points": list(dropped_pos or [])[:12],
                    "dropped_negative_points": list(dropped_neg or [])[:12],
                    "retry_guidance": [
                        "Re-sample positive points away from blocked regions.",
                        "Keep points on target interior and away from contact boundaries.",
                    ],
                }
                attempt_logs.append(
                    {
                        "attempt": int(attempt + 1),
                        "ok": False,
                        "stage": "point_filter",
                        "n_pos": int(n_pos),
                        "n_neg": int(n_neg),
                        "reason": prior_fail_reason,
                        "point_plan": pts,
                        "filtered_positive_points": positive,
                        "filtered_negative_points": negative,
                        "dropped_positive_points": dropped_pos,
                        "dropped_negative_points": dropped_neg,
                    }
                )
                continue

            seg = self.skill_bundle.registry.call(
                SKILL_SAM_SEGMENT_WITH_POINTS,
                client=client,
                points=positive,
                negative_points=negative,
                clear_existing=True,
            )
            if bool(seg.get("ok", False)):
                cur_mask = None
                sam3d_state = getattr(self.ctx, "sam3d_state", None)
                if sam3d_state is not None:
                    try:
                        with sam3d_state.lock:
                            idx = int(getattr(sam3d_state, "current_object_idx", 0))
                            objects = list(getattr(sam3d_state, "objects", []) or [])
                            if 0 <= idx < len(objects):
                                cur_mask = getattr(objects[idx], "mask", None)
                    except Exception:
                        cur_mask = None
                eval_pack = self._evaluate_mask_once(
                    object_name=str(object_name),
                    frame_rgb_u8=frame_rgb_u8,
                    mask_bool=(np.asarray(cur_mask) if cur_mask is not None else None),
                    positive_points=positive,
                    negative_points=negative,
                    backend_mode=backend_mode,
                    non_target_names=list(non_target_names or []),
                    filtered_prompt={
                        "positive_points": list(positive),
                        "negative_points": list(negative),
                        "dropped_positive_points": list(dropped_pos),
                        "dropped_negative_points": list(dropped_neg),
                        "blocked_positive_rel": [[float(x), float(y)] for (x, y) in blocked_positive_rel],
                        "blocked_negative_rel": [[float(x), float(y)] for (x, y) in blocked_negative_rel],
                    },
                )
            else:
                eval_pack = self._evaluate_mask_once(
                    object_name=str(object_name),
                    frame_rgb_u8=frame_rgb_u8,
                    mask_bool=None,
                    positive_points=positive,
                    negative_points=negative,
                    backend_mode=backend_mode,
                    non_target_names=list(non_target_names or []),
                    filtered_prompt={
                        "positive_points": list(positive),
                        "negative_points": list(negative),
                        "dropped_positive_points": list(dropped_pos),
                        "dropped_negative_points": list(dropped_neg),
                        "blocked_positive_rel": [[float(x), float(y)] for (x, y) in blocked_positive_rel],
                        "blocked_negative_rel": [[float(x), float(y)] for (x, y) in blocked_negative_rel],
                    },
                )

            vlm_eval = dict(eval_pack.get("vlm_eval", {}) or {})
            if bool(vlm_eval.get("evaluated", False)):
                pd = dict(vlm_eval.get("point_diagnostics", {}) or {})
                wrong_pos_rel = self._extract_rel_xy_points(pd.get("wrong_positive_points", []))
                wrong_neg_rel = self._extract_rel_xy_points(pd.get("wrong_negative_points", []))
                self._append_unique_rel_points(blocked_positive_rel, wrong_pos_rel)
                self._append_unique_rel_points(blocked_negative_rel, wrong_neg_rel)
            eva = dict(eval_pack.get("evaluate", {}) or {})
            pos_cov = float(eval_pack.get("pos_cov", 0.0))
            neg_cov = float(eval_pack.get("neg_cov", 1.0))
            point_ok = bool(eval_pack.get("point_ok", False))

            print("seg:", seg)
            print("eva:", eva)
            self._dump_eval_debug_artifacts(
                object_name=str(object_name),
                attempt_index=int(attempt + 1),
                planning_image_rgb_u8=frame_rgb_u8,
                positive_points=positive,
                negative_points=negative,
                point_plan=pts,
                segment=seg,
                evaluate=eva,
            )
            ok = bool(seg.get("ok", False)) and bool(eval_pack.get("ok", False))
            attempt_logs.append(
                {
                    "attempt": int(attempt + 1),
                    "ok": bool(ok),
                    "n_pos": int(n_pos),
                    "n_neg": int(n_neg),
                    "point_plan": pts,
                    "segment": seg,
                    "evaluate": eva,
                    "filtered_positive_points": positive,
                    "filtered_negative_points": negative,
                    "dropped_positive_points": dropped_pos,
                    "dropped_negative_points": dropped_neg,
                }
            )
            if ok:
                return {
                    "ok": True,
                    "positive_points": positive,
                    "negative_points": negative,
                    "attempts": attempt_logs,
                    "attempt_count": int(attempt + 1),
                }
            if bool(vlm_eval.get("evaluated", False)):
                extra_n = len(list(vlm_eval.get("extra_objects", []) or []))
                missing_n = len(list(vlm_eval.get("missing_parts", []) or []))
                pd = dict(vlm_eval.get("point_diagnostics", {}) or {})
                wrong_pos_n = len(list(pd.get("wrong_positive_points", []) or []))
                wrong_neg_n = len(list(pd.get("wrong_negative_points", []) or []))
                summary = str(vlm_eval.get("summary", "")).strip()
                non_targets_brief = ",".join([str(x) for x in list(non_target_names or [])[:4]])
                prior_fail_reason = (
                    f"mask_eval_vlm_failed:accurate={bool(vlm_eval.get('ok', False))},"
                    f"extra_n={extra_n},missing_n={missing_n},wrong_pos_n={wrong_pos_n},"
                    f"wrong_neg_n={wrong_neg_n},point_ok={bool(point_ok)},"
                    f"pos_cov={pos_cov:.4f},neg_cov={neg_cov:.4f},"
                    f"non_targets=[{non_targets_brief}],summary={summary[:120]}"
                )
            else:
                prior_fail_reason = (
                    f"mask_eval_failed:qwen_not_evaluated,"
                    f"point_ok={bool(point_ok)},pos_cov={pos_cov:.4f},neg_cov={neg_cov:.4f}"
                )
                prior_fail_analysis = {
                    "stage": "mask_eval_not_evaluated",
                    "reason": str(prior_fail_reason),
                    "point_rule_passed": bool(point_ok),
                    "positive_coverage": float(pos_cov),
                    "negative_coverage": float(neg_cov),
                    "filtered_positive_points": list(positive or [])[:12],
                    "filtered_negative_points": list(negative or [])[:12],
                    "dropped_positive_points": list(dropped_pos or [])[:12],
                    "dropped_negative_points": list(dropped_neg or [])[:12],
                    "retry_guidance": [
                        "Keep positive points on target core and avoid boundaries.",
                        "Use negative points near possible non-target leakage areas.",
                    ],
                }
                continue

            prior_fail_analysis = self._build_point_planner_retry_analysis(
                vlm_eval=dict(vlm_eval or {}),
                point_ok=bool(point_ok),
                pos_cov=float(pos_cov),
                neg_cov=float(neg_cov),
                filtered_positive_points=list(positive or []),
                filtered_negative_points=list(negative or []),
                dropped_positive_points=list(dropped_pos or []),
                dropped_negative_points=list(dropped_neg or []),
                non_target_names=list(non_target_names or []),
            )

        return {
            "ok": False,
            "reason": "segment_retries_exhausted",
            "attempts": attempt_logs,
        }

    def run(self, client, object_text: str) -> dict[str, Any]:
        with self.state.lock:
            points_per_object = int(getattr(self.state, "agent1_points_per_object", 3))
            max_retries = int(getattr(self.state, "agent1_max_retries", 5))
            min_area_ratio = float(getattr(self.state, "agent1_mask_min_area_ratio", 0.001))
            backend_mode = str(getattr(self.state, "agent0_backend_mode", "local_openai_compatible"))
            targets = list(getattr(self.state, "agent1_targets", []) or [])
            target_index = int(getattr(self.state, "agent1_target_index", 0))

        target_name = self._resolve_target_object(object_text)
        if not target_name:
            msg = "no_current_target_object"
            self.log("fail", msg)
            self._notify(client, "Agent1", "No current target object. Run Agent0 first or set Object description.", color="red")
            return {"ok": False, "reason": msg}

        # Keep current SAM3D state if already active, so Agent1 can generate objects continuously.
        begin = self.skill_bundle.registry.call(SKILL_SAM3D_ENSURE_SESSION, client=client)
        if not bool(begin.get("ok", False)):
            msg = str(begin.get("reason", "sam3d_begin_failed"))
            self.log("fail", msg)
            self._notify(client, "Agent1", f"Failed to enter SAM3D mode: {msg}", color="red")
            return {"ok": False, "reason": msg, "begin": begin}

        frame_rgb_u8: np.ndarray | None = None
        sam3d_state = getattr(self.ctx, "sam3d_state", None)
        if sam3d_state is not None:
            try:
                with sam3d_state.lock:
                    img = getattr(sam3d_state, "image_rgb_u8", None)
                    if img is not None:
                        frame_rgb_u8 = img.copy()
            except Exception:
                frame_rgb_u8 = None
        if frame_rgb_u8 is None:
            frame_rgb_u8 = self._current_frame_rgb()
        if frame_rgb_u8 is None:
            msg = "current_frame_image_missing"
            self.log("fail", msg)
            self._notify(client, "Agent1", "Cannot read current frame input image.", color="red")
            return {"ok": False, "reason": msg}

        self.log("start", f"target={target_name}")
        non_target_names = self._collect_non_target_names(target_name)
        seg = self._segment_object_with_retries(
            client,
            object_name=target_name,
            frame_rgb_u8=frame_rgb_u8,

            points_per_object=points_per_object,
            max_retries=max_retries,
            min_area_ratio=min_area_ratio,
            backend_mode=backend_mode,
            non_target_names=non_target_names,
        )
        if not bool(seg.get("ok", False)):
            self.log("fail", f"object={target_name},stage=segment")
            self._notify(client, "Agent1", f"Mask generation failed for {target_name}.", color="red", sec=8.0)
            return {
                "ok": False,
                "target": target_name,
                "stage": "segment",
                "reason": str(seg.get("reason", "segment_failed")),
                "segment": seg,
                "points_per_object": int(points_per_object),
                "max_retries": int(max_retries),
                "mask_min_area_ratio": float(min_area_ratio),
                "backend_mode": backend_mode,
            }

        self.log("info", f"object={target_name},stage=generate")
        gen = self.skill_bundle.registry.call(
            SKILL_SAM3D_GENERATE_AND_WAIT,
            client=client,
            timeout=2400.0,
            poll_interval=0.5,
        )
        if not bool(gen.get("ok", False)):
            self.log("fail", f"object={target_name},stage=generate")
            self._notify(client, "Agent1", f"3D generation failed for {target_name}.", color="red", sec=8.0)
            return {
                "ok": False,
                "target": target_name,
                "stage": "generate",
                "reason": str(gen.get("reason", "generate_failed")),
                "segment": seg,
                "generate": gen,
                "points_per_object": int(points_per_object),
                "max_retries": int(max_retries),
                "mask_min_area_ratio": float(min_area_ratio),
                "backend_mode": backend_mode,
            }

        # Do not auto "Add All Objects & Exit" here; keep session alive for multi-object generation.
        next_object_out = self.skill_bundle.registry.call(SKILL_SAM3D_NEW_OBJECT, client=client)
        if not bool(next_object_out.get("ok", False)):
            self.log("warn", f"object={target_name},stage=next_object,reason={next_object_out.get('reason')}")

        ok = True
        print(f"[VLM Assistor][Agent1] generated object: {target_name}")
        gen_obj_idx = int(gen.get("object_idx", -1))
        with self.state.lock:
            done = list(getattr(self.state, "agent1_generated_objects", []) or [])
            keyset = {str(x).strip().lower() for x in done}
            if str(target_name).strip().lower() not in keyset:
                done.append(str(target_name))
            self.state.agent1_generated_objects = done
            idx_map = dict(getattr(self.state, "agent1_generated_object_indices", {}) or {})
            if gen_obj_idx >= 0:
                idx_map[str(target_name).strip().lower()] = int(gen_obj_idx)
            self.state.agent1_generated_object_indices = idx_map
        packed = {
            "ok": bool(ok),
            "target": target_name,
            "target_index": int(target_index),
            "target_count": int(len(targets)),
            "generated_count": 1 if ok else 0,
            "generated_object_idx": int(gen_obj_idx),
            "segment": seg,
            "generate": gen,
            "add_to_session": {
                "ok": False,
                "skipped": True,
                "reason": "agent1_no_auto_add_all_objects",
            },
            "next_object": next_object_out,
            "points_per_object": int(points_per_object),
            "max_retries": int(max_retries),
            "mask_min_area_ratio": float(min_area_ratio),
            "backend_mode": backend_mode,
        }
        with self.state.lock:
            self.state.last_click_result = dict(packed)

        if ok:
            self._notify(
                client,
                "Agent1",
                f"Generated object: {target_name}. SAM3D kept active for next object.",
                color="green",
                sec=6.0,
            )
        else:
            self._notify(client, "Agent1", f"Failed to generate object: {target_name}", color="red", sec=8.0)
        return packed

    def evaluate_current_mask(self, client, object_text: str) -> dict[str, Any]:
        with self.state.lock:
            backend_mode = str(getattr(self.state, "agent0_backend_mode", "local_openai_compatible"))

        target_name = self._resolve_target_object(object_text)
        if not target_name:
            msg = "no_current_target_object"
            self.log("fail", msg)
            self._notify(client, "Agent1 Eval", "No current target object.", color="red")
            return {"ok": False, "reason": msg}

        frame_rgb_u8: np.ndarray | None = None
        sam3d_state = getattr(self.ctx, "sam3d_state", None)
        if sam3d_state is not None:
            try:
                with sam3d_state.lock:
                    img = getattr(sam3d_state, "image_rgb_u8", None)
                    if img is not None:
                        frame_rgb_u8 = img.copy()
            except Exception:
                frame_rgb_u8 = None
        if frame_rgb_u8 is None:
            frame_rgb_u8 = self._current_frame_rgb()
        if frame_rgb_u8 is None:
            msg = "current_frame_image_missing"
            self.log("fail", msg)
            self._notify(client, "Agent1 Eval", "Cannot read current frame input image.", color="red")
            return {"ok": False, "reason": msg}

        mask_bool, positive, negative, state_meta = self._current_object_mask_and_points()
        if not bool(state_meta.get("ok", False)):
            msg = str(state_meta.get("reason", "sam3d_state_invalid"))
            self.log("fail", f"manual_eval:{msg}")
            self._notify(client, "Agent1 Eval", f"Cannot access current object/mask: {msg}", color="red", sec=8.0)
            return {"ok": False, "reason": msg, "state_meta": state_meta}
        if mask_bool is None:
            msg = "mask_missing_for_manual_eval"
            self.log("fail", msg)
            self._notify(client, "Agent1 Eval", "Current object has no mask to evaluate.", color="red", sec=8.0)
            return {"ok": False, "reason": msg, "state_meta": state_meta}

        non_target_names = self._collect_non_target_names(target_name)
        eval_pack = self._evaluate_mask_once(
            object_name=target_name,
            frame_rgb_u8=frame_rgb_u8,
            mask_bool=np.asarray(mask_bool),
            positive_points=positive,
            negative_points=negative,
            backend_mode=backend_mode,
            non_target_names=non_target_names,
            filtered_prompt={
                "positive_points": list(positive),
                "negative_points": list(negative),
                "manual_eval": True,
            },
        )
        self._dump_eval_debug_artifacts(
            object_name=str(target_name),
            attempt_index=1,
            planning_image_rgb_u8=frame_rgb_u8,
            positive_points=positive,
            negative_points=negative,
            point_plan={"ok": True, "manual_eval": True, "target": str(target_name)},
            segment={"ok": True, "stage": "manual_current_mask_eval"},
            evaluate=dict(eval_pack.get("evaluate", {}) or {}),
        )

        ok = bool(eval_pack.get("ok", False))
        eva = dict(eval_pack.get("evaluate", {}) or {})
        packed = {
            "ok": bool(ok),
            "stage": "manual_evaluate_current_mask",
            "target": target_name,
            "backend_mode": backend_mode,
            "state_meta": state_meta,
            "positive_points": list(positive),
            "negative_points": list(negative),
            "evaluate": eva,
        }
        with self.state.lock:
            self.state.last_click_result = dict(packed)

        if ok:
            self.log("ok", f"manual_eval_pass:target={target_name}")
            self._notify(client, "Agent1 Eval", f"Mask evaluation passed: {target_name}", color="green", sec=6.0)
        else:
            self.log("fail", f"manual_eval_fail:target={target_name}")
            self._notify(client, "Agent1 Eval", f"Mask evaluation failed: {target_name}", color="orange", sec=8.0)
        return packed


class Gen3DValidatorAgent(_AgentBase):
    def __init__(self, ctx, state, bridge, vlm_client) -> None:
        super().__init__(ctx, state, bridge, vlm_client, agent_name="agent2_gen3d_validator")

    def run(self, client) -> dict[str, Any]:
        self.log("skip", "not_implemented_yet")
        self._notify(client, "Agent2", "Gen3D validator is not implemented yet.", color="orange")
        return {"ok": False, "reason": "not_implemented"}


class PoseAdjusterAgent(_AgentBase):
    def __init__(self, ctx, state, bridge, vlm_client) -> None:
        super().__init__(ctx, state, bridge, vlm_client, agent_name="agent2_xpose")

    @staticmethod
    def _normalize_direction(raw: str) -> str:
        s = str(raw or "").strip().lower()
        if s in {"+x", "positive", "forward", "front"}:
            return "positive"
        if s in {"-x", "negative", "backward", "back"}:
            return "negative"
        return "none"

    @staticmethod
    def _candidate_brief(c: dict[str, Any]) -> dict[str, Any]:
        rule = dict(c.get("rule_diagnostics", {}) or {})
        lock_report = dict(c.get("pose_opt_lock_report", {}) or {})
        return {
            "candidate_id": int(c.get("candidate_id", 0)),
            "delta_x": float(c.get("delta_x", 0.0)),
            "collision_count": int(rule.get("collision_count", 0)),
            "x_out_of_bounds": bool(rule.get("x_out_of_bounds", False)),
            "target_center_x": float((rule.get("target_center_xyz", [0.0, 0.0, 0.0]) or [0.0])[0]),
            "lock_target_world_x": lock_report.get("lock_target_world_x", None),
            "lock_drift_world_x_before_post_correct": lock_report.get("lock_drift_world_x_before_post_correct", None),
            "lock_post_correct_applied": bool(lock_report.get("lock_post_correct_applied", False)),
            "lock_drift_world_x_after_post_correct": lock_report.get("lock_drift_world_x_after_post_correct", None),
            "view_paths": dict(c.get("view_paths", {}) or {}),
            "overlay_path": str(c.get("overlay_path", "") or ""),
        }

    def _resolve_target_idx(self, object_name: str) -> int:
        name = str(object_name or "").strip()
        key = str(self.vlm_client.to_english_object_name(name)).strip().lower() if name else ""
        with self.state.lock:
            idx_map = dict(getattr(self.state, "agent1_generated_object_indices", {}) or {})
        if key and key in idx_map:
            return int(idx_map[key])
        lst = self.bridge.registry.call("object.list")
        if bool(lst.get("ok", False)):
            objs = list(lst.get("objects", []) or [])
            for it in reversed(objs):
                if bool(it.get("has_splat", False)):
                    return int(it.get("object_idx", -1))
        return -1

    def _artifact_round_dir(self, round_idx: int, run_token: str | None = None) -> Path | None:
        with self.state.lock:
            root = getattr(self.state, "artifact_root", None)
        if root is None:
            return None
        try:
            token = str(run_token or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()
            d = Path(root) / "agent2_xpose" / token / f"round_{int(round_idx):02d}"
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            return None

    def _collect_object_relation_context(self, *, target_idx: int, target_name: str) -> dict[str, Any]:
        with self.state.lock:
            idx_map_raw = dict(getattr(self.state, "agent1_generated_object_indices", {}) or {})

        idx_to_name: dict[int, str] = {}
        for k, v in idx_map_raw.items():
            try:
                idx_to_name[int(v)] = str(k)
            except Exception:
                continue
        if int(target_idx) >= 0:
            idx_to_name[int(target_idx)] = str(target_name)

        obj_list_pack = self.bridge.registry.call("object.list")
        generated: list[dict[str, Any]] = []
        if bool(obj_list_pack.get("ok", False)):
            for it in list(obj_list_pack.get("objects", []) or []):
                if not bool(it.get("has_splat", False)):
                    continue
                idx = int(it.get("object_idx", -1))
                if idx < 0:
                    continue
                generated.append(
                    {
                        "object_idx": int(idx),
                        "object_name": str(idx_to_name.get(idx, f"obj_{idx}")),
                        "center_xyz": it.get("center_xyz", None),
                        "is_target": bool(idx == int(target_idx)),
                    }
                )
        by_idx = {int(x.get("object_idx", -1)): x for x in generated}
        if int(target_idx) >= 0 and int(target_idx) not in by_idx:
            generated.append(
                {
                    "object_idx": int(target_idx),
                    "object_name": str(target_name),
                    "center_xyz": None,
                    "is_target": True,
                }
            )
            by_idx[int(target_idx)] = generated[-1]

        generated.sort(key=lambda x: int(x.get("object_idx", -1)))
        current_obj = dict(by_idx.get(int(target_idx), {"object_idx": int(target_idx), "object_name": str(target_name)}))
        already = [dict(x) for x in generated if int(x.get("object_idx", -1)) != int(target_idx)]
        return {
            "current_generated_object": current_obj,
            "already_generated_objects": already,
            "all_generated_objects": [dict(x) for x in generated],
            "generated_name_map": {str(int(k)): str(v) for k, v in idx_to_name.items()},
        }

    def _build_target_specific_relation_hint(
        self,
        *,
        target_name: str,
        base_relation_hint: str,
        scene_graph: dict[str, Any],
        object_context: dict[str, Any],
        max_items: int = 8,
    ) -> str:
        """Build a target-focused relation hint for Agent2 pose validation.

        Rules:
        - Prioritize only relations directly involving target object.
        - If no direct relation exists, fall back to distance/depth/collision constraints
          against already generated objects.
        """
        target_raw = str(target_name or "").strip()
        target_key = str(self.vlm_client.to_english_object_name(target_raw)).strip().lower()

        generated_names: list[str] = []
        cur_obj = dict((object_context or {}).get("current_generated_object", {}) or {})
        cur_name = str(cur_obj.get("object_name", "")).strip()
        if cur_name:
            generated_names.append(cur_name)
        for it in list((object_context or {}).get("already_generated_objects", []) or []):
            if not isinstance(it, dict):
                continue
            nm = str(it.get("object_name", "")).strip()
            if nm:
                generated_names.append(nm)
        uniq_gen: list[str] = []
        seen_gen: set[str] = set()
        for nm in generated_names:
            key = str(self.vlm_client.to_english_object_name(nm)).strip().lower() or nm.lower()
            if key in seen_gen:
                continue
            seen_gen.add(key)
            uniq_gen.append(nm)

        def _name_key(x: str) -> str:
            s = str(x or "").strip()
            if not s:
                return ""
            k = str(self.vlm_client.to_english_object_name(s)).strip().lower()
            return k or s.lower()

        target_rel_rows: list[str] = []
        for rel in list((scene_graph or {}).get("relations", []) or []):
            if not isinstance(rel, dict):
                continue
            sub = str(rel.get("subject", "")).strip()
            obj = str(rel.get("obj", rel.get("object", ""))).strip()
            pred = str(rel.get("predicate", "")).strip()
            if not (sub and obj and pred):
                continue
            sub_k = _name_key(sub)
            obj_k = _name_key(obj)
            if not target_key or (target_key not in {sub_k, obj_k}):
                continue
            other = obj if sub_k == target_key else sub
            other_k = _name_key(other)
            if seen_gen and other_k and other_k not in seen_gen:
                continue
            dist_hint = str(rel.get("distance_hint", "unknown")).strip() or "unknown"
            target_rel_rows.append(f"{sub} --{pred}--> {obj} (distance_hint={dist_hint})")
            if len(target_rel_rows) >= int(max_items):
                break

        target_spatial_rows: list[str] = []
        for seg in list(
            (scene_graph or {}).get(
                "relative_position_timeline",
                (scene_graph or {}).get("relative_spatial_timeline", []),
            )
            or []
        ):
            if not isinstance(seg, dict):
                continue
            frame_range = str(seg.get("frame_range", seg.get("range", "global"))).strip() or "global"
            pairs = list(seg.get("pairs", seg.get("relations", seg.get("relative_pairs", []))) or [])
            if not isinstance(pairs, list):
                continue
            for p in pairs:
                if not isinstance(p, dict):
                    continue
                sub = str(p.get("subject", p.get("src", p.get("a", "")))).strip()
                obj = str(p.get("obj", p.get("object", p.get("dst", p.get("b", ""))))).strip()
                if not (sub and obj):
                    continue
                sub_k = _name_key(sub)
                obj_k = _name_key(obj)
                if not target_key or (target_key not in {sub_k, obj_k}):
                    continue
                other = obj if sub_k == target_key else sub
                other_k = _name_key(other)
                if seen_gen and other_k and other_k not in seen_gen:
                    continue
                h = str(p.get("horizontal", p.get("left_right", p.get("lr", "unknown")))).strip() or "unknown"
                v = str(p.get("vertical", p.get("up_down", p.get("ud", "unknown")))).strip() or "unknown"
                d = str(p.get("depth", p.get("front_back", p.get("fb", "unknown")))).strip() or "unknown"
                target_spatial_rows.append(f"[{frame_range}] {sub} vs {obj}: horizontal={h}, vertical={v}, depth={d}")
                if len(target_spatial_rows) >= int(max_items):
                    break
            if len(target_spatial_rows) >= int(max_items):
                break

        base_hint = str(base_relation_hint or "").strip()
        if target_rel_rows or target_spatial_rows:
            pieces: list[str] = []
            if target_rel_rows:
                pieces.append("relations: " + " ; ".join(target_rel_rows))
            if target_spatial_rows:
                pieces.append("relative_positions: " + " ; ".join(target_spatial_rows))
            joined = " | ".join(pieces)
            return (
                f"Target-focused relation hint for `{target_raw}`: {joined}. "
                "Only treat relations involving current target as hard constraints; "
                "relations not involving this target are background context only."
                + (f" Original global hint: {base_hint}" if base_hint else "")
            )

        neighbor_text = ", ".join([str(x) for x in uniq_gen if str(x).strip() and _name_key(str(x)) != target_key][:6])
        return (
            f"No direct target relation found for `{target_raw}` in video understanding. "
            "For current pose validation, prioritize target-vs-neighbor physical constraints: "
            "distance/proximity consistency, relative depth ordering, non-penetration/non-collision."
            + (f" Generated neighbors: {neighbor_text}." if neighbor_text else "")
            + " Ignore unrelated relation chains that do not involve current target."
            + (f" Original global hint: {base_hint}" if base_hint else "")
        )

    @staticmethod
    def _rule_with_named_objects(rule_diag: dict[str, Any], object_context: dict[str, Any]) -> dict[str, Any]:
        rule = dict(rule_diag or {})
        name_map_raw = dict((object_context or {}).get("generated_name_map", {}) or {})
        name_map: dict[int, str] = {}
        for k, v in name_map_raw.items():
            try:
                name_map[int(k)] = str(v)
            except Exception:
                continue
        collisions = []
        for it in list(rule.get("collisions", []) or []):
            item = dict(it or {})
            try:
                other_idx = int(item.get("other_object_idx", -1))
            except Exception:
                other_idx = -1
            if other_idx >= 0:
                item["other_object_name"] = str(name_map.get(other_idx, f"obj_{other_idx}"))
            collisions.append(item)
        if collisions:
            rule["collisions"] = collisions
        cur_obj = dict((object_context or {}).get("current_generated_object", {}) or {})
        if cur_obj:
            rule["target_object_name"] = str(cur_obj.get("object_name", ""))
        return rule

    def _infer_direction_from_rule(self, rule_diag: dict[str, Any]) -> str:
        if bool(rule_diag.get("x_out_of_bounds", False)):
            x_bounds = list(rule_diag.get("x_bounds", [-1.0, 1.0]) or [-1.0, 1.0])
            center = list(rule_diag.get("target_center_xyz", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
            try:
                x = float(center[0])
                xmin = float(x_bounds[0])
                xmax = float(x_bounds[1])
                if x < xmin:
                    return "positive"
                if x > xmax:
                    return "negative"
            except Exception:
                pass
        collisions = list(rule_diag.get("collisions", []) or [])
        if not collisions:
            return "none"
        center_x = float((rule_diag.get("target_center_xyz", [0.0, 0.0, 0.0]) or [0.0])[0])
        signed = 0.0
        for it in collisions:
            other_x = float(it.get("other_center_x", center_x))
            w = float(it.get("overlap_volume", 1.0))
            signed += (center_x - other_x) * max(1e-6, w)
        if signed > 0:
            return "positive"
        if signed < 0:
            return "negative"
        return "none"

    @staticmethod
    def _collision_severity(vlm_eval: dict[str, Any]) -> str:
        s = str(
            (vlm_eval or {}).get(
                "collision_severity",
                (vlm_eval or {}).get("collision_level", "unknown"),
            )
        ).strip().lower()
        if s in {"none", "no", "no_collision", "clean"}:
            return "none"
        if s in {"slight", "minor", "small", "mild", "light", "tiny", "slight_penetration"}:
            return "slight"
        if s in {"severe", "major", "heavy", "deep", "hard", "large", "strong", "penetration"}:
            return "severe"
        return "unknown"

    @classmethod
    def _collision_blocked_by_contact_policy(
        cls,
        *,
        aabb_collision: bool,
        vlm_collision: bool,
        vlm_eval: dict[str, Any],
    ) -> bool:
        if not (bool(aabb_collision) and bool(vlm_collision)):
            return False
        allow_contact_slight = bool((vlm_eval or {}).get("contact_relation_allows_slight_collision", False))
        severity = cls._collision_severity(dict(vlm_eval or {}))
        allow_slight_collision = bool(allow_contact_slight and severity == "slight")
        return not bool(allow_slight_collision)

    def _render_candidate_images(
        self,
        *,
        client,
        target_idx: int,
        save_dir: Path | None,
        prefix: str,
    ) -> dict[str, Any]:
        mv = self.bridge.registry.call(
            "render.object_multiview",
            client=client,
            object_idx=int(target_idx),
            views=["front", "back", "left", "right", "up", "down"],
            image_size=512,
            save_dir=(str(save_dir) if save_dir is not None else None),
            prefix=str(prefix),
        )
        if not bool(mv.get("ok", False)):
            return {"ok": False, "reason": f"multiview_failed:{mv.get('reason', 'unknown')}", "multiview": mv}
        input_path = None
        if save_dir is not None:
            input_path = save_dir / f"{str(prefix)}_input.png"
        inp = self.bridge.registry.call(
            "render.input_image",
            save_path=(str(input_path) if input_path is not None else None),
        )
        overlay_path = None
        if save_dir is not None:
            overlay_path = save_dir / f"{str(prefix)}_overlay.png"
        ov = self.bridge.registry.call(
            "render.input_overlay",
            client=client,
            object_idx=int(target_idx),
            save_path=(str(overlay_path) if overlay_path is not None else None),
        )
        if not bool(ov.get("ok", False)):
            return {"ok": False, "reason": f"overlay_failed:{ov.get('reason', 'unknown')}", "multiview": mv, "overlay": ov}
        return {
            "ok": True,
            "view_images": dict(mv.get("view_images", {}) or {}),
            "view_paths": dict(mv.get("view_paths", {}) or {}),
            "input_rgb_u8": inp.get("input_rgb_u8", None) if bool(inp.get("ok", False)) else None,
            "input_path": str(input_path) if input_path is not None else "",
            "input_fetch": dict(inp),
            "overlay_rgb_u8": ov.get("overlay_rgb_u8", None),
            "overlay_path": str(overlay_path) if overlay_path is not None else "",
        }

    def _evaluate_pose_once(
        self,
        *,
        client,
        target_idx: int,
        object_name: str,
        relation_hint: str,
        scene_graph: dict[str, Any],
        object_context: dict[str, Any],
        x_min: float,
        x_max: float,
        save_dir: Path | None,
        prefix: str,
    ) -> dict[str, Any]:
        rule = self.bridge.registry.call(
            "physics.check_aabb",
            object_idx=int(target_idx),
            x_min=float(x_min),
            x_max=float(x_max),
        )
        if not bool(rule.get("ok", False)):
            return {"ok": False, "reason": f"rule_eval_failed:{rule.get('reason', 'unknown')}"}
        rule_named = self._rule_with_named_objects(dict(rule), object_context)
        render_pack = self._render_candidate_images(
            client=client,
            target_idx=int(target_idx),
            save_dir=save_dir,
            prefix=str(prefix),
        )
        if not bool(render_pack.get("ok", False)):
            return {"ok": False, "reason": render_pack.get("reason", "render_failed")}

        vlm_eval = self.vlm_client.evaluate_pose_plausibility_with_vlm(
            object_name=str(object_name),
            relation_hint=str(relation_hint),
            scene_graph=dict(scene_graph or {}),
            object_context=dict(object_context or {}),
            rule_diagnostics=dict(rule_named),
            view_images=dict(render_pack.get("view_images", {}) or {}),
            input_rgb_u8=render_pack.get("input_rgb_u8", None),
            overlay_rgb_u8=render_pack.get("overlay_rgb_u8", None),
            backend_mode=getattr(self.state, "agent0_backend_mode", "local_openai_compatible"),
            timeout=90.0,
        )
        vlm_ok = bool(vlm_eval.get("evaluated", False))
        vlm_collision = bool(vlm_eval.get("has_collision_visual", False)) if vlm_ok else False
        vlm_plausible = bool(vlm_eval.get("is_plausible", False)) if vlm_ok else False
        vlm_pixel_aligned = bool(vlm_eval.get("is_pixel_aligned", True)) if vlm_ok else True
        vlm_view_centered = bool(vlm_eval.get("is_view_centered", True)) if vlm_ok else True
        vlm_off_center_allowed = bool(vlm_eval.get("off_center_allowed_by_physics", False)) if vlm_ok else False
        x_bad = bool(rule_named.get("x_out_of_bounds", False))
        collision_count = int(rule_named.get("collision_count", 0))
        aabb_collision = collision_count > 0
        collision_block = self._collision_blocked_by_contact_policy(
            aabb_collision=bool(aabb_collision),
            vlm_collision=bool(vlm_collision),
            vlm_eval=dict(vlm_eval or {}),
        )
        center_ok = bool(vlm_view_centered or vlm_off_center_allowed)
        final_ok = bool(
            (not x_bad)
            and (not collision_block)
            and (vlm_plausible if vlm_ok else (not aabb_collision))
            and bool(vlm_pixel_aligned)
            and bool(center_ok)
        )

        dir_vlm = self._normalize_direction(vlm_eval.get("suggested_x_direction", "none"))
        dir_relation = self._normalize_direction(vlm_eval.get("suggested_x_direction_relation", "none"))
        relation_consistent = bool(vlm_eval.get("relation_consistent_with_video", True))
        dir_centering = self._normalize_direction(vlm_eval.get("suggested_x_direction_centering", "none"))
        depth_prior = dict(vlm_eval.get("depth_order_prior", {}) or {})
        dir_depth = self._normalize_direction(
            depth_prior.get(
                "suggested_x_direction_prior",
                depth_prior.get("suggested_x_direction", "none"),
            )
        )
        dir_rule = self._infer_direction_from_rule(rule_named)
        allow_centering_adjust = bool((not vlm_view_centered) and (not vlm_off_center_allowed))
        direction = (
            dir_vlm
            if dir_vlm != "none"
            else (
                dir_relation
                if dir_relation != "none"
                else (
                    dir_centering
                    if (allow_centering_adjust and dir_centering != "none" and relation_consistent)
                    else (dir_depth if dir_depth != "none" else dir_rule)
                )
            )
        )
        return {
            "ok": True,
            "final_ok": bool(final_ok),
            "rule_diagnostics": dict(rule_named),
            "vlm_eval": dict(vlm_eval),
            "suggested_direction": str(direction),
            "view_paths": dict(render_pack.get("view_paths", {}) or {}),
            "input_path": str(render_pack.get("input_path", "") or ""),
            "overlay_path": str(render_pack.get("overlay_path", "") or ""),
            "collision_block_by_vlm": bool(collision_block),
        }

    def _evaluate_candidate_with_vlm(
        self,
        *,
        object_name: str,
        relation_hint: str,
        scene_graph: dict[str, Any],
        object_context: dict[str, Any],
        rule_diagnostics: dict[str, Any],
        view_images: dict[str, Any],
        input_rgb_u8: Any,
        overlay_rgb_u8: Any,
    ) -> dict[str, Any]:
        vlm_eval = self.vlm_client.evaluate_pose_plausibility_with_vlm(
            object_name=str(object_name),
            relation_hint=str(relation_hint),
            scene_graph=dict(scene_graph or {}),
            object_context=dict(object_context or {}),
            rule_diagnostics=dict(rule_diagnostics or {}),
            view_images=dict(view_images or {}),
            input_rgb_u8=input_rgb_u8,
            overlay_rgb_u8=overlay_rgb_u8,
            backend_mode=getattr(self.state, "agent0_backend_mode", "local_openai_compatible"),
            timeout=90.0,
        )
        vlm_ok = bool(vlm_eval.get("evaluated", False))
        vlm_collision = bool(vlm_eval.get("has_collision_visual", False)) if vlm_ok else False
        vlm_plausible = bool(vlm_eval.get("is_plausible", False)) if vlm_ok else False
        vlm_pixel_aligned = bool(vlm_eval.get("is_pixel_aligned", True)) if vlm_ok else True
        vlm_view_centered = bool(vlm_eval.get("is_view_centered", True)) if vlm_ok else True
        vlm_off_center_allowed = bool(vlm_eval.get("off_center_allowed_by_physics", False)) if vlm_ok else False
        x_bad = bool(rule_diagnostics.get("x_out_of_bounds", False))
        collision_count = int(rule_diagnostics.get("collision_count", 0))
        aabb_collision = collision_count > 0
        collision_block = self._collision_blocked_by_contact_policy(
            aabb_collision=bool(aabb_collision),
            vlm_collision=bool(vlm_collision),
            vlm_eval=dict(vlm_eval or {}),
        )
        center_ok = bool(vlm_view_centered or vlm_off_center_allowed)
        final_ok = bool(
            (not x_bad)
            and (not collision_block)
            and (vlm_plausible if vlm_ok else (not aabb_collision))
            and bool(vlm_pixel_aligned)
            and bool(center_ok)
        )
        return {
            "final_ok": bool(final_ok),
            "vlm_eval": dict(vlm_eval or {}),
            "collision_block_by_vlm": bool(collision_block),
        }

    def run(self, client, object_name: str) -> dict[str, Any]:
        with self.state.lock:
            x_step = float(getattr(self.state, "agent2_x_step", 0.1))
            num_candidates = int(getattr(self.state, "agent2_num_candidates", 4))
            max_rounds = int(getattr(self.state, "agent2_max_rounds", 3))
            x_min = float(getattr(self.state, "agent2_x_min", -1.0))
            x_max = float(getattr(self.state, "agent2_x_max", 1.0))
            pose_steps = int(getattr(self.state, "agent2_pose_opt_steps", 160))
            pose_lr = float(getattr(self.state, "agent2_pose_opt_lr", 0.05))
            pose_min_side = int(getattr(self.state, "agent2_pose_opt_min_side", 512))
            pose_rgb_weight = float(getattr(self.state, "agent2_pose_opt_rgb_weight", 0.4))
            pose_timeout = float(getattr(self.state, "agent2_pose_opt_timeout_sec", 1500.0))
            debug_save_views = bool(getattr(self.state, "agent2_debug_save_views", False))
            relation_hint = str(getattr(self.state, "relation_hint", "") or "")
            scene_graph = dict(getattr(self.state, "scene_graph", {}) or {})

        target_idx = self._resolve_target_idx(object_name)
        if target_idx < 0:
            self.log("fail", "agent2_target_not_found")
            self._notify(client, "Agent2", "No generated object found for Agent2.", color="red", sec=6.0)
            return {"ok": False, "reason": "target_object_not_found"}
        prev_current_idx = -1
        cur_pack = self.bridge.registry.call("object.list")
        if bool(cur_pack.get("ok", False)):
            try:
                prev_current_idx = int(cur_pack.get("current_object_idx", -1))
            except Exception:
                prev_current_idx = -1

        sel = self.bridge.registry.call("object.select", object_idx=int(target_idx), client=client, ensure_bbox=True)
        if not bool(sel.get("ok", False)):
            return {"ok": False, "reason": f"select_target_failed:{sel.get('reason', 'unknown')}"}

        target_name = str(object_name or "").strip() or f"obj_{int(target_idx)}"
        object_context = self._collect_object_relation_context(target_idx=int(target_idx), target_name=str(target_name))
        relation_hint_target = self._build_target_specific_relation_hint(
            target_name=str(target_name),
            base_relation_hint=str(relation_hint),
            scene_graph=dict(scene_graph or {}),
            object_context=dict(object_context or {}),
        )
        self.log("info", f"agent2_target_relation_hint:{relation_hint_target}")
        self.log("start", f"agent2_target={target_name},idx={target_idx}")

        def _restore_previous_current_object() -> None:
            if int(prev_current_idx) < 0 or int(prev_current_idx) == int(target_idx):
                return
            _ = self.bridge.registry.call(
                "object.select",
                object_idx=int(prev_current_idx),
                client=client,
                ensure_bbox=False,
            )

        round_logs: list[dict[str, Any]] = []
        final_ok = False
        forced_direction = "none"
        debug_run_token = datetime.now().strftime("%Y%m%d_%H%M%S") if bool(debug_save_views) else ""
        debug_root = None
        if bool(debug_save_views):
            with self.state.lock:
                root = getattr(self.state, "artifact_root", None)
            if root is not None:
                debug_root = Path(root) / "agent2_xpose" / str(debug_run_token)

        for rd in range(max(1, int(max_rounds))):
            round_dir = self._artifact_round_dir(rd + 1, run_token=debug_run_token) if bool(debug_save_views) else None
            cur_eval = self._evaluate_pose_once(
                client=client,
                target_idx=int(target_idx),
                object_name=target_name,
                relation_hint=relation_hint_target,
                scene_graph=scene_graph,
                object_context=object_context,
                x_min=float(x_min),
                x_max=float(x_max),
                save_dir=round_dir,
                prefix=f"round{int(rd+1):02d}_current",
            )
            if not bool(cur_eval.get("ok", False)):
                self.log("fail", f"agent2_eval_failed_round{rd+1}:{cur_eval.get('reason')}")
                _restore_previous_current_object()
                return {"ok": False, "reason": cur_eval.get("reason", "evaluate_failed"), "round": int(rd + 1)}
            pre_adjust_summary = str(
                dict(cur_eval.get("vlm_eval", {}) or {}).get("pre_adjust_analysis_summary", "")
            ).strip()
            if pre_adjust_summary:
                self.log("info", f"pre_adjust_relative_analysis_round{rd+1}:{pre_adjust_summary}")
            if bool(cur_eval.get("final_ok", False)):
                final_ok = True
                round_logs.append(
                    {
                        "round": int(rd + 1),
                        "accepted_current_pose": True,
                        "rule_diagnostics": dict(cur_eval.get("rule_diagnostics", {}) or {}),
                        "vlm_eval": dict(cur_eval.get("vlm_eval", {}) or {}),
                        "view_paths": dict(cur_eval.get("view_paths", {}) or {}),
                        "input_path": str(cur_eval.get("input_path", "") or ""),
                        "overlay_path": str(cur_eval.get("overlay_path", "") or ""),
                    }
                )
                break

            direction = str(forced_direction if forced_direction != "none" else cur_eval.get("suggested_direction", "none"))
            direction = self._normalize_direction(direction)
            if direction == "none":
                direction = self._infer_direction_from_rule(dict(cur_eval.get("rule_diagnostics", {}) or {}))
            if direction == "none":
                round_logs.append(
                    {
                        "round": int(rd + 1),
                        "accepted_current_pose": False,
                        "reason": "no_adjust_direction",
                        "rule_diagnostics": dict(cur_eval.get("rule_diagnostics", {}) or {}),
                        "vlm_eval": dict(cur_eval.get("vlm_eval", {}) or {}),
                    }
                )
                break

            base_snapshot_pack = self.bridge.registry.call("object.snapshot_state", object_idx=int(target_idx))
            if not bool(base_snapshot_pack.get("ok", False)):
                _restore_previous_current_object()
                return {"ok": False, "reason": f"snapshot_failed:{base_snapshot_pack.get('reason', 'unknown')}"}
            base_snapshot = base_snapshot_pack.get("snapshot", None)
            if not isinstance(base_snapshot, dict):
                _restore_previous_current_object()
                return {"ok": False, "reason": "snapshot_invalid"}

            sign = 1.0 if direction == "positive" else -1.0
            candidates: list[dict[str, Any]] = []
            accepted_candidate: dict[str, Any] | None = None
            for ci in range(max(1, int(num_candidates))):
                delta = sign * float(x_step) * float(ci + 1)
                _ = self.bridge.registry.call(
                    "object.restore_state",
                    snapshot=base_snapshot,
                    client=client,
                    make_current=True,
                )
                tr = self.bridge.registry.call(
                    "transform.translate_x",
                    client=client,
                    object_idx=int(target_idx),
                    delta_x=float(delta),
                    lock_axis=True,
                    commit=True,
                )
                if not bool(tr.get("ok", False)):
                    continue
                po = self.bridge.registry.call(
                    "pose.pixel_align",
                    client=client,
                    object_idx=int(target_idx),
                    lock_x=True,
                    lock_y=False,
                    lock_z=False,
                    steps=int(pose_steps),
                    lr=float(pose_lr),
                    min_side=int(pose_min_side),
                    rgb_weight=float(pose_rgb_weight),
                    timeout_sec=float(pose_timeout),
                )
                if not bool(po.get("ok", False)):
                    continue
                rule = self.bridge.registry.call(
                    "physics.check_aabb",
                    object_idx=int(target_idx),
                    x_min=float(x_min),
                    x_max=float(x_max),
                )
                rule_named = self._rule_with_named_objects(dict(rule or {}), object_context)
                render_pack = self._render_candidate_images(
                    client=client,
                    target_idx=int(target_idx),
                    save_dir=round_dir,
                    prefix=f"round{int(rd+1):02d}_cand{int(ci+1):02d}",
                )
                snap_pack = self.bridge.registry.call("object.snapshot_state", object_idx=int(target_idx))
                if not bool(render_pack.get("ok", False)) or not bool(snap_pack.get("ok", False)):
                    continue
                candidate = {
                    "candidate_id": int(ci + 1),
                    "delta_x": float(delta),
                    "rule_diagnostics": dict(rule_named),
                    "pose_opt_lock_report": {
                        "lock_axis_frame": po.get("lock_axis_frame", "world"),
                        "lock_target_world_x": po.get("lock_target_world_x", None),
                        "lock_drift_world_x_before_post_correct": po.get("lock_drift_world_x_before_post_correct", None),
                        "lock_post_correct_applied": bool(po.get("lock_post_correct_applied", False)),
                        "lock_post_correct_ok": bool(po.get("lock_post_correct_ok", False)),
                        "lock_drift_world_x_after_post_correct": po.get("lock_drift_world_x_after_post_correct", None),
                        "center_world_after_pose_opt": po.get("center_world_after_pose_opt", None),
                    },
                    "view_images": dict(render_pack.get("view_images", {}) or {}),
                    "view_paths": dict(render_pack.get("view_paths", {}) or {}),
                    "overlay_rgb_u8": render_pack.get("overlay_rgb_u8", None),
                    "overlay_path": str(render_pack.get("overlay_path", "") or ""),
                    "input_rgb_u8": render_pack.get("input_rgb_u8", None),
                    "input_path": str(render_pack.get("input_path", "") or ""),
                    "snapshot": snap_pack.get("snapshot", None),
                }
                cand_eval = self._evaluate_candidate_with_vlm(
                    object_name=target_name,
                    relation_hint=relation_hint_target,
                    scene_graph=scene_graph,
                    object_context=object_context,
                    rule_diagnostics=dict(rule_named),
                    view_images=dict(candidate.get("view_images", {}) or {}),
                    input_rgb_u8=candidate.get("input_rgb_u8", None),
                    overlay_rgb_u8=candidate.get("overlay_rgb_u8", None),
                )
                candidate["vlm_eval"] = dict(cand_eval.get("vlm_eval", {}) or {})
                candidate["final_ok"] = bool(cand_eval.get("final_ok", False))
                candidates.append(candidate)
                if bool(candidate["final_ok"]):
                    accepted_candidate = candidate
                    break

            if isinstance(accepted_candidate, dict):
                best_snapshot = accepted_candidate.get("snapshot", None)
                if isinstance(best_snapshot, dict):
                    _ = self.bridge.registry.call(
                        "object.restore_state",
                        snapshot=best_snapshot,
                        client=client,
                        make_current=True,
                    )
                final_ok = True
                forced_direction = "none"
                round_logs.append(
                    {
                        "round": int(rd + 1),
                        "accepted_current_pose": True,
                        "accepted_candidate_pose": True,
                        "selected_candidate_id": int(accepted_candidate.get("candidate_id", 0)),
                        "suggested_direction": str(direction),
                        "current_eval": {
                            "rule_diagnostics": dict(cur_eval.get("rule_diagnostics", {}) or {}),
                            "vlm_eval": dict(cur_eval.get("vlm_eval", {}) or {}),
                            "view_paths": dict(cur_eval.get("view_paths", {}) or {}),
                            "input_path": str(cur_eval.get("input_path", "") or ""),
                            "overlay_path": str(cur_eval.get("overlay_path", "") or ""),
                        },
                        "accepted_candidate_eval": {
                            "rule_diagnostics": dict(accepted_candidate.get("rule_diagnostics", {}) or {}),
                            "vlm_eval": dict(accepted_candidate.get("vlm_eval", {}) or {}),
                            "view_paths": dict(accepted_candidate.get("view_paths", {}) or {}),
                            "input_path": str(accepted_candidate.get("input_path", "") or ""),
                            "overlay_path": str(accepted_candidate.get("overlay_path", "") or ""),
                        },
                        "candidates": [self._candidate_brief(c) for c in candidates],
                        "early_stop_on_first_pass": True,
                    }
                )
                break

            if not candidates:
                _ = self.bridge.registry.call("object.restore_state", snapshot=base_snapshot, client=client, make_current=True)
                round_logs.append(
                    {
                        "round": int(rd + 1),
                        "accepted_current_pose": False,
                        "reason": "no_valid_candidates",
                        "suggested_direction": str(direction),
                    }
                )
                break

            rank = self.vlm_client.rank_xpose_candidates_with_vlm(
                object_name=target_name,
                relation_hint=relation_hint_target,
                scene_graph=scene_graph,
                object_context=object_context,
                candidates=candidates,
                backend_mode=getattr(self.state, "agent0_backend_mode", "local_openai_compatible"),
                timeout=120.0,
            )
            best_id = int(rank.get("best_candidate_id", candidates[0].get("candidate_id", 1)))
            by_id = {int(c.get("candidate_id", 0)): c for c in candidates}
            best = by_id.get(best_id, candidates[0])
            best_snapshot = best.get("snapshot", None)
            if isinstance(best_snapshot, dict):
                _ = self.bridge.registry.call(
                    "object.restore_state",
                    snapshot=best_snapshot,
                    client=client,
                    make_current=True,
                )
            forced_direction = self._normalize_direction(rank.get("next_direction", "none"))

            round_logs.append(
                {
                    "round": int(rd + 1),
                    "accepted_current_pose": False,
                    "suggested_direction": str(direction),
                    "selected_candidate_id": int(best.get("candidate_id", 0)),
                    "rank_result": dict(rank),
                        "current_eval": {
                            "rule_diagnostics": dict(cur_eval.get("rule_diagnostics", {}) or {}),
                            "vlm_eval": dict(cur_eval.get("vlm_eval", {}) or {}),
                            "view_paths": dict(cur_eval.get("view_paths", {}) or {}),
                            "input_path": str(cur_eval.get("input_path", "") or ""),
                            "overlay_path": str(cur_eval.get("overlay_path", "") or ""),
                        },
                        "candidates": [self._candidate_brief(c) for c in candidates],
                    }
                )

            if not bool(rank.get("need_another_round", True)):
                forced_direction = "none"

        final_eval = self.bridge.registry.call(
            "physics.check_aabb",
            object_idx=int(target_idx),
            x_min=float(x_min),
            x_max=float(x_max),
        )
        final_eval_named = self._rule_with_named_objects(dict(final_eval or {}), object_context)
        final_vlm: dict[str, Any] = {"ok": False, "evaluated": False, "reason": "not_run"}
        if bool(final_eval_named.get("ok", False)):
            # Safety: require VLM-confirmed collision to hard-fail collision.
            final_render = self._render_candidate_images(
                client=client,
                target_idx=int(target_idx),
                save_dir=((debug_root / "final") if (bool(debug_save_views) and debug_root is not None) else None),
                prefix="final_check",
            )
            if bool(final_render.get("ok", False)):
                final_vlm = self.vlm_client.evaluate_pose_plausibility_with_vlm(
                    object_name=target_name,
                    relation_hint=relation_hint_target,
                    scene_graph=scene_graph,
                    object_context=object_context,
                    rule_diagnostics=dict(final_eval_named),
                    view_images=dict(final_render.get("view_images", {}) or {}),
                    input_rgb_u8=final_render.get("input_rgb_u8", None),
                    overlay_rgb_u8=final_render.get("overlay_rgb_u8", None),
                    backend_mode=getattr(self.state, "agent0_backend_mode", "local_openai_compatible"),
                    timeout=90.0,
                )
                x_bad = bool(final_eval_named.get("x_out_of_bounds", False))
                aabb_collision = int(final_eval_named.get("collision_count", 0)) > 0
                vlm_collision = bool(final_vlm.get("has_collision_visual", False)) if bool(final_vlm.get("evaluated", False)) else False
                vlm_pixel_aligned = bool(final_vlm.get("is_pixel_aligned", True)) if bool(final_vlm.get("evaluated", False)) else True
                vlm_view_centered = bool(final_vlm.get("is_view_centered", True)) if bool(final_vlm.get("evaluated", False)) else True
                vlm_off_center_allowed = bool(final_vlm.get("off_center_allowed_by_physics", False)) if bool(final_vlm.get("evaluated", False)) else False
                collision_block = self._collision_blocked_by_contact_policy(
                    aabb_collision=bool(aabb_collision),
                    vlm_collision=bool(vlm_collision),
                    vlm_eval=dict(final_vlm or {}),
                )
                center_ok = bool(vlm_view_centered or vlm_off_center_allowed)
                final_ok = bool(
                    (not x_bad)
                    and (not collision_block)
                    and bool(final_vlm.get("is_plausible", False))
                    and bool(vlm_pixel_aligned)
                    and bool(center_ok)
                )

        out = {
            "ok": bool(final_ok),
            "agent": "agent2_xpose_adjuster",
            "target": str(target_name),
            "target_object_idx": int(target_idx),
            "x_step": float(x_step),
            "num_candidates": int(num_candidates),
            "max_rounds": int(max_rounds),
            "x_bounds": [float(x_min), float(x_max)],
            "debug_save_views": bool(debug_save_views),
            "debug_root_dir": (str(debug_root) if debug_root is not None else ""),
            "object_relation_context": dict(object_context or {}),
            "target_relation_hint": str(relation_hint_target),
            "rounds": round_logs,
            "final_rule_diagnostics": dict(final_eval_named or {}),
            "final_vlm_eval": dict(final_vlm or {}),
        }
        try:
            print(
                "[VLM Assistor][Agent2] final_eval_result:",
                json.dumps(
                    {
                        "ok": bool(out.get("ok", False)),
                        "target": str(target_name),
                        "target_object_idx": int(target_idx),
                        "debug_save_views": bool(debug_save_views),
                        "debug_root_dir": (str(debug_root) if debug_root is not None else ""),
                        "object_relation_context": dict(object_context or {}),
                        "final_rule_diagnostics": dict(final_eval_named or {}),
                        "final_vlm_eval": dict(final_vlm or {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        except Exception:
            pass
        with self.state.lock:
            self.state.last_pose_adjust = dict(out)
        if bool(final_ok):
            self.log("ok", f"agent2_success:target={target_name},idx={target_idx}")
            self._notify(client, "Agent2", f"X-axis pose adjust done: {target_name}", color="green", sec=6.0)
        else:
            self.log("fail", f"agent2_failed:target={target_name},idx={target_idx}")
            self._notify(client, "Agent2", f"X-axis pose adjust incomplete: {target_name}", color="orange", sec=8.0)
        _restore_previous_current_object()
        return out


class PoseAdjusterAgentV2(PoseAdjusterAgent):
    """Direct-shift Agent2 variant (no candidate enumeration).

    Strategy:
    - First generated object: move world-x close to 0, then pose-opt.
    - Later objects: ask VLM for signed normalized image-space shift and world-x direction,
      convert normalized magnitude to world-x units using camera geometry, apply once, then pose-opt.
    - Final step always runs VLM plausibility verification.
    """

    def __init__(self, ctx, state, bridge, vlm_client) -> None:
        super().__init__(ctx, state, bridge, vlm_client)
        self.agent_name = "agent2_xpose_v2"

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return float(max(float(lo), min(float(hi), float(v))))

    @staticmethod
    def _alignment_tolerance(x_step: float) -> float:
        # Scale alignment tolerance with x_step, capped to avoid over-expansion.
        return float(max(0.03, min(0.35, max(0.05, abs(float(x_step))) * 2.0)))

    @staticmethod
    def _is_finite_number(v: Any) -> bool:
        try:
            return bool(np.isfinite(float(v)))
        except Exception:
            return False

    def _name_key(self, raw: str) -> str:
        s = str(raw or "").strip()
        if not s:
            return ""
        try:
            en = str(self.vlm_client.to_english_object_name(s)).strip().lower()
        except Exception:
            en = ""
        return en or s.lower()

    def _camera_input_pose_pack(self, client) -> dict[str, Any]:
        pack = self.bridge.registry.call("camera.input_pose", client=client)
        if bool(pack.get("ok", False)):
            return dict(pack)
        try:
            fov_deg = float(np.rad2deg(float(client.camera.fov)))
        except Exception:
            fov_deg = 49.1
        return {
            "ok": True,
            "source": "client_fallback",
            "position": np.asarray(getattr(client.camera, "position", [0.0, 0.0, 1.5]), dtype=np.float32).astype(float).tolist(),
            "wxyz": np.asarray(getattr(client.camera, "wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).astype(float).tolist(),
            "fov_deg": float(fov_deg),
            "image_width": 512,
            "image_height": 512,
        }

    def _project_world_to_norm_x(self, world_xyz: np.ndarray, cam_pack: dict[str, Any]) -> tuple[float | None, float | None]:
        try:
            p_w = np.asarray(world_xyz, dtype=np.float32).reshape(3)
            cam_pos = np.asarray(cam_pack.get("position", [0.0, 0.0, 1.5]), dtype=np.float32).reshape(3)
            cam_wxyz = np.asarray(cam_pack.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
            fov_deg = float(cam_pack.get("fov_deg", 49.1))
            width = max(64.0, float(cam_pack.get("image_width", 512)))
            R_cw = np.asarray(self.bridge._rotation_matrix_from_wxyz(cam_wxyz), dtype=np.float32)
            R_wc = R_cw.T
            p_c = R_wc @ (p_w - cam_pos)
            z = float(p_c[2])
            if abs(z) < 1e-6:
                return None, None
            tan_half = float(np.tan(0.5 * np.deg2rad(max(1e-3, fov_deg))))
            fx = float(width / (2.0 * max(1e-6, tan_half)))
            u = float(fx * (float(p_c[0]) / z) + width * 0.5)
            norm_x = float((u - width * 0.5) / (width * 0.5))
            return norm_x, z
        except Exception:
            return None, None

    def _project_world_to_image_uvz(self, world_xyz: np.ndarray, cam_pack: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
        """Project a world point to image pixel coordinates (u, v) and camera depth z."""
        try:
            p_w = np.asarray(world_xyz, dtype=np.float32).reshape(3)
            cam_pos = np.asarray(cam_pack.get("position", [0.0, 0.0, 1.5]), dtype=np.float32).reshape(3)
            cam_wxyz = np.asarray(cam_pack.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
            fov_deg = float(cam_pack.get("fov_deg", 49.1))
            width = max(64.0, float(cam_pack.get("image_width", 512)))
            height = max(64.0, float(cam_pack.get("image_height", 512)))
            R_cw = np.asarray(self.bridge._rotation_matrix_from_wxyz(cam_wxyz), dtype=np.float32)
            R_wc = R_cw.T
            p_c = R_wc @ (p_w - cam_pos)
            z = float(p_c[2])
            if abs(z) < 1e-6:
                return None, None, None
            tan_half = float(np.tan(0.5 * np.deg2rad(max(1e-3, fov_deg))))
            fy = float(height / (2.0 * max(1e-6, tan_half)))
            fx = float(width / (2.0 * max(1e-6, tan_half)))
            u = float(fx * (float(p_c[0]) / z) + width * 0.5)
            # Image row coordinates: 0 at top, height-1 at bottom.
            v = float(height * 0.5 - fy * (float(p_c[1]) / z))
            return u, v, z
        except Exception:
            return None, None, None

    def _world_x_axis_profile_at_y_up(
        self,
        *,
        target_center_world: np.ndarray,
        cam_pack: dict[str, Any],
        eps_world_x: float,
        y_up_norm_ref: float | None = None,
    ) -> dict[str, Any]:
        """Estimate projected world-x axis position at top/bottom and a y_up-normalized row."""
        u0, v0, z0 = self._project_world_to_image_uvz(target_center_world, cam_pack)
        if u0 is None or v0 is None:
            return {"ok": False, "reason": "base_projection_failed"}
        p_plus = np.asarray(target_center_world, dtype=np.float32) + np.asarray([float(eps_world_x), 0.0, 0.0], dtype=np.float32)
        u1, v1, _ = self._project_world_to_image_uvz(p_plus, cam_pack)
        if u1 is None or v1 is None:
            return {"ok": False, "reason": "plus_projection_failed"}

        width = max(64.0, float(cam_pack.get("image_width", 512)))
        height = max(64.0, float(cam_pack.get("image_height", 512)))
        du = float(u1 - u0)
        dv = float(v1 - v0)
        denom_h = max(1.0, height - 1.0)
        if y_up_norm_ref is None:
            y_up = float((height - 1.0 - float(v0)) / denom_h)
            y_up = self._clamp(y_up, 0.0, 1.0)
        else:
            y_up = self._clamp(float(y_up_norm_ref), 0.0, 1.0)
        v_ref = float((1.0 - y_up) * (height - 1.0))

        if abs(dv) > 1e-9:
            t_top = float((0.0 - v0) / dv)
            t_bottom = float(((height - 1.0) - v0) / dv)
            t_ref = float((v_ref - v0) / dv)
            x_top = float(u0 + t_top * du)
            x_bottom = float(u0 + t_bottom * du)
            x_ref = float(u0 + t_ref * du)
            slope_px_per_v = float(du / dv)
        else:
            # If projected world-x is nearly horizontal in image, top/bottom intersections are undefined.
            x_top = float(u0)
            x_bottom = float(u0)
            x_ref = float(u0)
            slope_px_per_v = None

        norm_x_ref = float((x_ref - width * 0.5) / (width * 0.5))
        return {
            "ok": True,
            "u_px": float(u0),
            "v_px": float(v0),
            "du_px": float(du),
            "dv_px": float(dv),
            "depth_cam_z": float(z0) if z0 is not None else None,
            "image_width": float(width),
            "image_height": float(height),
            "y_up_norm": float(y_up),
            "x_axis_top_px": float(x_top),
            "x_axis_bottom_px": float(x_bottom),
            "x_axis_at_y_up_px": float(x_ref),
            "norm_x_at_y_up": float(norm_x_ref),
            "x_axis_slope_px_per_v": (None if slope_px_per_v is None else float(slope_px_per_v)),
            "v_ref_px": float(v_ref),
        }

    def _camera_pose_pack_for_named_view(self, *, view_name: str, image_width: int, image_height: int) -> dict[str, Any]:
        app = getattr(self.ctx, "app", None)
        if app is None:
            return {"ok": False, "reason": "app_unavailable"}
        try:
            eye, wxyz = self.bridge._camera_pose_for_view_origin(app, str(view_name), 1.5)
            fov_deg = float(getattr(app, "fov_deg", 49.1))
            return {
                "ok": True,
                "source": f"named_view_{str(view_name).strip().lower()}",
                "position": np.asarray(eye, dtype=np.float32).astype(float).tolist(),
                "wxyz": np.asarray(wxyz, dtype=np.float32).astype(float).tolist(),
                "fov_deg": float(fov_deg),
                "image_width": int(max(64, int(image_width))),
                "image_height": int(max(64, int(image_height))),
            }
        except Exception as exc:
            return {"ok": False, "reason": f"named_view_camera_failed:{type(exc).__name__}:{exc}"}

    def _extract_up_view_target_point(self, delta_plan: dict[str, Any]) -> dict[str, Any]:
        raw = (delta_plan or {}).get("up_view_target_center_point", None)
        if not isinstance(raw, dict):
            return {"ok": False, "reason": "up_point_missing"}
        valid = bool(raw.get("valid", False))
        try:
            x_norm = float(raw.get("x_norm", None))
            y_up_norm = float(raw.get("y_up_norm", None))
        except Exception:
            return {"ok": False, "reason": "up_point_invalid_values"}
        if not (self._is_finite_number(x_norm) and self._is_finite_number(y_up_norm)):
            return {"ok": False, "reason": "up_point_non_finite"}
        if not bool(valid):
            return {
                "ok": False,
                "reason": "up_point_marked_invalid",
                "x_norm": float(self._clamp(x_norm, 0.0, 1.0)),
                "y_up_norm": float(self._clamp(y_up_norm, 0.0, 1.0)),
                "point_reason": str(raw.get("reason", "")).strip(),
            }
        return {
            "ok": True,
            "x_norm": float(self._clamp(x_norm, 0.0, 1.0)),
            "y_up_norm": float(self._clamp(y_up_norm, 0.0, 1.0)),
            "point_reason": str(raw.get("reason", "")).strip(),
            "image_width": int(max(1, int(raw.get("image_width", 512)))),
            "image_height": int(max(1, int(raw.get("image_height", 512)))),
        }

    def _solve_world_x_delta_from_up_point(
        self,
        *,
        target_center_world: np.ndarray,
        up_target_x_norm: float,
        up_target_y_up_norm: float,
        cam_pack: dict[str, Any],
    ) -> dict[str, Any]:
        eps = max(1e-4, 1e-3 * max(1.0, float(np.linalg.norm(target_center_world))))
        y_up = float(self._clamp(float(up_target_y_up_norm), 0.0, 1.0))
        x_norm = float(self._clamp(float(up_target_x_norm), 0.0, 1.0))
        p0 = np.asarray(target_center_world, dtype=np.float32).reshape(3)
        p1 = p0 + np.asarray([eps, 0.0, 0.0], dtype=np.float32)
        u0, v0, z0 = self._project_world_to_image_uvz(p0, dict(cam_pack or {}))
        u1, v1, _ = self._project_world_to_image_uvz(p1, dict(cam_pack or {}))
        if u0 is None or v0 is None:
            return {"ok": False, "reason": "up_projection_base_failed"}
        if u1 is None or v1 is None:
            return {"ok": False, "reason": "up_projection_plus_failed"}

        width = max(64.0, float(cam_pack.get("image_width", 512)))
        height = max(64.0, float(cam_pack.get("image_height", 512)))
        u_t = float(x_norm * max(1.0, width - 1.0))
        v_t = float((1.0 - y_up) * max(1.0, height - 1.0))

        du = float(u1 - u0)
        dv = float(v1 - v0)
        denom = float(du * du + dv * dv)
        if denom <= 1e-9:
            return {"ok": False, "reason": "up_axis_line_degenerate"}

        # Orthogonally project VLM point to the projected world-x line in up view.
        lam = float(((u_t - u0) * du + (v_t - v0) * dv) / denom)
        u_proj = float(u0 + lam * du)
        v_proj = float(v0 + lam * dv)
        world_dx = float(lam * eps)
        world_target_x = float(np.asarray(target_center_world, dtype=np.float32).reshape(3)[0] + world_dx)
        gain_norm = float(((du / max(1.0, width * 0.5))) / eps)
        if abs(dv) > 1e-9:
            t_top = float((0.0 - v0) / dv)
            t_bottom = float(((height - 1.0) - v0) / dv)
            x_top = float(u0 + t_top * du)
            x_bottom = float(u0 + t_bottom * du)
        else:
            x_top = float(u0)
            x_bottom = float(u0)

        norm_cur = float((u0 - width * 0.5) / (width * 0.5))
        norm_proj = float((u_proj - width * 0.5) / (width * 0.5))
        norm_delta = float(norm_proj - norm_cur)
        return {
            "ok": True,
            "source": "up_view_point_axis_projection",
            "world_delta_x": float(world_dx),
            "world_delta_magnitude": float(abs(world_dx)),
            "world_target_x": float(world_target_x),
            "gain_norm_per_world_x": float(gain_norm),
            "up_target_x_norm": float(x_norm),
            "up_target_y_up_norm": float(y_up),
            "target_norm_x": float((u_t - width * 0.5) / (width * 0.5)),
            "current_norm_x_at_target_y_up": float(norm_cur),
            "norm_delta": float(norm_delta),
            "x_axis_top_px": float(x_top),
            "x_axis_bottom_px": float(x_bottom),
            "x_axis_at_y_up_px": float(u_proj),
            "v_ref_px": float(v_proj),
            "base_depth_cam_z": (None if z0 is None else float(z0)),
            "line_base_u_px": float(u0),
            "line_base_v_px": float(v0),
            "line_dir_u_px": float(du),
            "line_dir_v_px": float(dv),
            "projection_lambda": float(lam),
            "up_target_u_px": float(u_t),
            "up_target_v_px": float(v_t),
            "up_projected_u_px": float(u_proj),
            "up_projected_v_px": float(v_proj),
        }

    def _save_up_point_debug_visualization(
        self,
        *,
        up_rgb_u8: np.ndarray | None,
        save_dir: Path | None,
        prefix: str,
        up_target_x_norm: float,
        up_target_y_up_norm: float,
        profile: dict[str, Any] | None = None,
    ) -> str:
        if save_dir is None or up_rgb_u8 is None:
            return ""
        try:
            arr = np.asarray(up_rgb_u8, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
                return ""
            h, w = int(arr.shape[0]), int(arr.shape[1])
            px = float(self._clamp(float(up_target_x_norm), 0.0, 1.0) * max(1, w - 1))
            py = float((1.0 - self._clamp(float(up_target_y_up_norm), 0.0, 1.0)) * max(1, h - 1))
            r = max(4, int(round(min(w, h) * 0.015)))
            pil = Image.fromarray(arr, mode="RGB")
            dr = ImageDraw.Draw(pil)
            dr.line([(0.0, py), (float(w - 1), py)], fill=(255, 80, 80), width=2)
            dr.line([(px, 0.0), (px, float(h - 1))], fill=(255, 80, 80), width=2)
            dr.ellipse([(px - r, py - r), (px + r, py + r)], outline=(255, 32, 32), width=3)
            if isinstance(profile, dict):
                try:
                    x_top = float(profile.get("x_axis_top_px", np.nan))
                    x_bottom = float(profile.get("x_axis_bottom_px", np.nan))
                    x_ref = float(profile.get("x_axis_at_y_up_px", np.nan))
                    if np.isfinite(x_top) and np.isfinite(x_bottom):
                        dr.line([(x_top, 0.0), (x_bottom, float(h - 1))], fill=(255, 220, 0), width=2)
                    if np.isfinite(x_ref):
                        dr.ellipse([(x_ref - r, py - r), (x_ref + r, py + r)], outline=(32, 180, 255), width=3)
                except Exception:
                    pass
            out_path = Path(save_dir) / f"{str(prefix)}_up_point_debug.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pil.save(out_path)
            return str(out_path)
        except Exception:
            return ""

    def _normalized_shift_to_world_x(
        self,
        *,
        target_center_world: np.ndarray,
        normalized_magnitude: float,
        cam_pack: dict[str, Any],
    ) -> dict[str, Any]:
        mag = float(max(0.0, min(1.0, abs(normalized_magnitude))))
        if mag <= 1e-9:
            return {"ok": True, "world_delta_magnitude": 0.0, "gain_norm_per_world_x": None, "source": "zero_shift"}

        eps = max(1e-4, 1e-3 * max(1.0, float(np.linalg.norm(target_center_world))))
        base_profile = self._world_x_axis_profile_at_y_up(
            target_center_world=np.asarray(target_center_world, dtype=np.float32),
            cam_pack=cam_pack,
            eps_world_x=float(eps),
            y_up_norm_ref=None,
        )
        gain = None
        base_z = None
        if bool(base_profile.get("ok", False)):
            y_up_ref = float(base_profile.get("y_up_norm", 0.5))
            plus_profile = self._world_x_axis_profile_at_y_up(
                target_center_world=np.asarray(target_center_world, dtype=np.float32) + np.asarray([eps, 0.0, 0.0], dtype=np.float32),
                cam_pack=cam_pack,
                eps_world_x=float(eps),
                y_up_norm_ref=float(y_up_ref),
            )
            if bool(plus_profile.get("ok", False)):
                try:
                    width = max(64.0, float(base_profile.get("image_width", cam_pack.get("image_width", 512))))
                    x0 = float(base_profile.get("x_axis_at_y_up_px", np.nan))
                    x1 = float(plus_profile.get("x_axis_at_y_up_px", np.nan))
                    if np.isfinite(x0) and np.isfinite(x1):
                        gain = float(((x1 - x0) / (width * 0.5)) / eps)
                        base_z = base_profile.get("depth_cam_z", None)
                except Exception:
                    gain = None
            if gain is not None and abs(gain) > 1e-6:
                return {
                    "ok": True,
                    "world_delta_magnitude": float(mag / abs(gain)),
                    "gain_norm_per_world_x": float(gain),
                    "source": "numeric_projection_gain_y_up_axis",
                    "y_up_norm": float(base_profile.get("y_up_norm", 0.5)),
                    "x_axis_top_px": float(base_profile.get("x_axis_top_px", 0.0)),
                    "x_axis_bottom_px": float(base_profile.get("x_axis_bottom_px", 0.0)),
                    "base_norm_x": float(base_profile.get("norm_x_at_y_up", 0.0)),
                    "base_depth_cam_z": (None if base_z is None else float(base_z)),
                }

        # Backward-compatible projection gain fallback.
        base_norm, base_z = self._project_world_to_norm_x(target_center_world, cam_pack)
        plus_norm, _ = self._project_world_to_norm_x(
            np.asarray(target_center_world, dtype=np.float32) + np.asarray([eps, 0.0, 0.0], dtype=np.float32),
            cam_pack,
        )
        gain_legacy = None
        if base_norm is not None and plus_norm is not None:
            gain_legacy = float((plus_norm - base_norm) / eps)
        if gain_legacy is not None and abs(gain_legacy) > 1e-6:
            return {
                "ok": True,
                "world_delta_magnitude": float(mag / abs(gain_legacy)),
                "gain_norm_per_world_x": float(gain_legacy),
                "source": "numeric_projection_gain",
                "base_norm_x": float(base_norm),
                "base_depth_cam_z": (None if base_z is None else float(base_z)),
            }

        # Fallback geometric approximation.
        try:
            cam_wxyz = np.asarray(cam_pack.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
            fov_deg = float(cam_pack.get("fov_deg", 49.1))
            R_cw = np.asarray(self.bridge._rotation_matrix_from_wxyz(cam_wxyz), dtype=np.float32)
            cam_x_axis_world = np.asarray(R_cw[:, 0], dtype=np.float32).reshape(3)
            align = float(abs(cam_x_axis_world[0]))
            depth = max(1e-3, abs(float(base_z) if base_z is not None else 1.0))
            tan_half = float(np.tan(0.5 * np.deg2rad(max(1e-3, fov_deg))))
            cam_dx = float(mag * depth * tan_half)
            world_mag = float(cam_dx / max(1e-3, align))
            return {
                "ok": True,
                "world_delta_magnitude": float(world_mag),
                "gain_norm_per_world_x": None,
                "source": "fallback_camera_axis_alignment",
                "camera_x_world_align": float(align),
                "base_depth_cam_z": (None if base_z is None else float(base_z)),
            }
        except Exception as exc:
            return {"ok": False, "reason": f"shift_conversion_failed:{type(exc).__name__}:{exc}"}

    def _object_center_x_map(self, object_context: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        rows = list((object_context or {}).get("all_generated_objects", []) or [])
        for it in rows:
            if not isinstance(it, dict):
                continue
            name = str(it.get("object_name", "")).strip()
            center = it.get("center_xyz", None)
            if not name or not isinstance(center, (list, tuple)) or len(center) < 1:
                continue
            try:
                cx = float(center[0])
            except Exception:
                continue
            if not self._is_finite_number(cx):
                continue
            out[self._name_key(name)] = float(cx)
            out[str(name).strip().lower()] = float(cx)
        return out

    def _resolve_reference_alignment_plan(
        self,
        *,
        delta_plan: dict[str, Any],
        object_context: dict[str, Any],
        cur_x: float,
        fallback_offset_world_x: float,
        x_min: float,
        x_max: float,
    ) -> dict[str, Any]:
        center_map = self._object_center_x_map(object_context)
        explicit_mode = str((delta_plan or {}).get("target_alignment_mode", "")).strip().lower()
        explicit_ref = str((delta_plan or {}).get("reference_object", "")).strip()
        if explicit_mode in {"fallback_zero_x", "x_zero", "center_to_zero_x"}:
            desired = self._clamp(0.0, float(x_min), float(x_max))
            return {
                "has_reference": False,
                "mode": "fallback_zero_x",
                "desired_world_x": float(desired),
                "desired_world_x_unclamped": 0.0,
                "used_relative_analysis": False,
                "source": "vlm_explicit_mode",
            }
        if explicit_ref:
            ref_k = self._name_key(explicit_ref)
            if ref_k in center_map:
                ref_x = float(center_map[ref_k])
                if explicit_mode in {"align_reference_center", "align_with_reference_center", "align_center"}:
                    desired_raw = float(ref_x)
                    return {
                        "has_reference": True,
                        "mode": "align_reference_center",
                        "reference_object_name": str(explicit_ref),
                        "reference_world_x": float(ref_x),
                        "desired_world_x": float(self._clamp(desired_raw, float(x_min), float(x_max))),
                        "desired_world_x_unclamped": float(desired_raw),
                        "used_relative_analysis": False,
                        "source": "vlm_explicit_mode",
                    }
                if explicit_mode in {"offset_reference_center", "offset_from_reference_center", "offset_center"}:
                    desired_raw = float(ref_x + float(fallback_offset_world_x))
                    return {
                        "has_reference": True,
                        "mode": "offset_reference_center",
                        "reference_object_name": str(explicit_ref),
                        "reference_world_x": float(ref_x),
                        "desired_world_x": float(self._clamp(desired_raw, float(x_min), float(x_max))),
                        "desired_world_x_unclamped": float(desired_raw),
                        "used_relative_analysis": False,
                        "source": "vlm_explicit_mode",
                    }
        rel_rows = list((delta_plan or {}).get("relative_analysis", []) or [])
        best: dict[str, Any] | None = None
        best_score = 10**9
        for row in rel_rows:
            if not isinstance(row, dict):
                continue
            other = str(row.get("other_object", "")).strip()
            if not other:
                continue
            k = self._name_key(other)
            if not k or k not in center_map:
                continue
            eh = str(row.get("expected_horizontal", "unknown")).strip().lower()
            ev = str(row.get("expected_vertical", "unknown")).strip().lower()
            ed = str(row.get("expected_depth", "unknown")).strip().lower()
            consistent = bool(row.get("consistent", True))
            if eh == "overlap" or ev in {"above", "below"}:
                mode = "align_reference_center"
                pri = 0
            elif eh in {"left", "right"} or ed in {"front", "behind", "same_depth"}:
                mode = "offset_reference_center"
                pri = 2
            else:
                continue
            score = int(pri + (5 if consistent else 0))
            if score < best_score:
                best_score = score
                best = {
                    "reference_object_name": other,
                    "reference_key": k,
                    "reference_world_x": float(center_map[k]),
                    "mode": str(mode),
                    "expected_horizontal": eh,
                    "expected_vertical": ev,
                    "expected_depth": ed,
                    "consistent": bool(consistent),
                    "evidence": str(row.get("evidence", "")).strip(),
                }

        if isinstance(best, dict):
            ref_x = float(best["reference_world_x"])
            mode = str(best.get("mode", "offset_reference_center"))
            desired_raw = float(ref_x + float(fallback_offset_world_x))
            if mode == "align_reference_center":
                # For above/below or overlap relations, prefer direct x-center alignment.
                desired_raw = float(ref_x)
            desired = self._clamp(desired_raw, float(x_min), float(x_max))
            out = {
                "has_reference": True,
                "mode": str(mode),
                "reference_object_name": str(best.get("reference_object_name", "")),
                "reference_world_x": float(ref_x),
                "desired_world_x": float(desired),
                "desired_world_x_unclamped": float(desired_raw),
                "used_relative_analysis": True,
                "expected_horizontal": str(best.get("expected_horizontal", "unknown")),
                "expected_vertical": str(best.get("expected_vertical", "unknown")),
                "expected_depth": str(best.get("expected_depth", "unknown")),
                "evidence": str(best.get("evidence", "")),
            }
            return out

        fallback_zero = bool(abs(float(cur_x)) > 0.12 and abs(float(fallback_offset_world_x)) < 1e-6)
        if fallback_zero:
            desired = self._clamp(0.0, float(x_min), float(x_max))
            return {
                "has_reference": False,
                "mode": "fallback_zero_x",
                "desired_world_x": float(desired),
                "desired_world_x_unclamped": 0.0,
                "used_relative_analysis": False,
            }
        desired_raw = float(cur_x + float(fallback_offset_world_x))
        desired = self._clamp(desired_raw, float(x_min), float(x_max))
        return {
            "has_reference": False,
            "mode": "delta_only",
            "desired_world_x": float(desired),
            "desired_world_x_unclamped": float(desired_raw),
            "used_relative_analysis": False,
        }

    def _translate_world_x(self, *, client, target_idx: int, delta_x: float) -> dict[str, Any]:
        if abs(float(delta_x)) <= 1e-9:
            return {"ok": True, "delta_x": 0.0, "reason": "zero_delta"}
        tr = self.bridge.registry.call(
            "transform.translate_x",
            client=client,
            object_idx=int(target_idx),
            delta_x=float(delta_x),
            lock_axis=True,
            commit=True,
        )
        return dict(tr or {})

    def run(self, client, object_name: str) -> dict[str, Any]:
        with self.state.lock:
            x_step = float(getattr(self.state, "agent2_x_step", 0.1))
            x_min = float(getattr(self.state, "agent2_x_min", -1.0))
            x_max = float(getattr(self.state, "agent2_x_max", 1.0))
            pose_steps = int(getattr(self.state, "agent2_pose_opt_steps", 160))
            pose_lr = float(getattr(self.state, "agent2_pose_opt_lr", 0.05))
            pose_min_side = int(getattr(self.state, "agent2_pose_opt_min_side", 512))
            pose_rgb_weight = float(getattr(self.state, "agent2_pose_opt_rgb_weight", 0.4))
            pose_timeout = float(getattr(self.state, "agent2_pose_opt_timeout_sec", 1500.0))
            debug_save_views = bool(getattr(self.state, "agent2_debug_save_views", False))
            relation_hint = str(getattr(self.state, "relation_hint", "") or "")
            scene_graph = dict(getattr(self.state, "scene_graph", {}) or {})

        target_idx = self._resolve_target_idx(object_name)
        if target_idx < 0:
            self.log("fail", "agent2_v2_target_not_found")
            self._notify(client, "Agent2 V2", "No generated object found for Agent2 V2.", color="red", sec=6.0)
            return {"ok": False, "reason": "target_object_not_found"}

        prev_current_idx = -1
        cur_pack = self.bridge.registry.call("object.list")
        if bool(cur_pack.get("ok", False)):
            try:
                prev_current_idx = int(cur_pack.get("current_object_idx", -1))
            except Exception:
                prev_current_idx = -1

        sel = self.bridge.registry.call("object.select", object_idx=int(target_idx), client=client, ensure_bbox=True)
        if not bool(sel.get("ok", False)):
            return {"ok": False, "reason": f"select_target_failed:{sel.get('reason', 'unknown')}"}

        target_name = str(object_name or "").strip() or f"obj_{int(target_idx)}"
        object_context = self._collect_object_relation_context(target_idx=int(target_idx), target_name=str(target_name))
        relation_hint_target = self._build_target_specific_relation_hint(
            target_name=str(target_name),
            base_relation_hint=str(relation_hint),
            scene_graph=dict(scene_graph or {}),
            object_context=dict(object_context or {}),
        )
        self.log("info", f"agent2_v2_target_relation_hint:{relation_hint_target}")
        self.log("start", f"agent2_v2_target={target_name},idx={target_idx}")

        def _restore_previous_current_object() -> None:
            if int(prev_current_idx) < 0 or int(prev_current_idx) == int(target_idx):
                return
            _ = self.bridge.registry.call(
                "object.select",
                object_idx=int(prev_current_idx),
                client=client,
                ensure_bbox=False,
            )

        debug_run_token = datetime.now().strftime("%Y%m%d_%H%M%S") if bool(debug_save_views) else ""
        debug_root = None
        if bool(debug_save_views):
            with self.state.lock:
                root = getattr(self.state, "artifact_root", None)
            if root is not None:
                debug_root = Path(root) / "agent2_xpose_v2" / str(debug_run_token)
        plan_dir = None
        if bool(debug_save_views) and debug_root is not None:
            try:
                plan_dir = debug_root / "round_01"
                plan_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                plan_dir = None

        try:
            init_rule = self.bridge.registry.call(
                "physics.check_aabb",
                object_idx=int(target_idx),
                x_min=float(x_min),
                x_max=float(x_max),
            )
            if not bool(init_rule.get("ok", False)):
                return {"ok": False, "reason": f"rule_eval_failed:{init_rule.get('reason', 'unknown')}"}
            init_rule_named = self._rule_with_named_objects(dict(init_rule), object_context)
            target_center_world = np.asarray(init_rule_named.get("target_center_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
            cur_x = float(target_center_world[0])
            align_tol = self._alignment_tolerance(float(x_step))

            already_objects = list((object_context or {}).get("already_generated_objects", []) or [])
            is_first_generated = len(already_objects) == 0

            delta_plan: dict[str, Any] = {}
            conversion: dict[str, Any] = {}
            anchor_plan: dict[str, Any] = {}
            up_point_plan: dict[str, Any] = {}
            up_point_vis_path = ""
            move_world_dx = 0.0
            desired_world_x = None
            if is_first_generated:
                desired_x = self._clamp(0.0, float(x_min), float(x_max))
                desired_world_x = float(desired_x)
                move_world_dx = float(desired_x - cur_x)
                delta_plan = {
                    "strategy": "first_object_center_to_zero_x",
                    "suggested_world_x_direction": ("positive" if move_world_dx > 0 else ("negative" if move_world_dx < 0 else "none")),
                    "normalized_shift_signed": 0.0,
                    "normalized_shift_magnitude": 0.0,
                    "analysis_summary": "First generated object: move world-x close to 0 before pose optimization.",
                    "evaluated": True,
                }
                conversion = {
                    "ok": True,
                    "source": "direct_world_target_x0",
                    "current_world_x": float(cur_x),
                    "desired_world_x": float(desired_x),
                    "world_delta_magnitude": float(abs(move_world_dx)),
                }
                anchor_plan = {
                    "has_reference": False,
                    "mode": "x_zero",
                    "desired_world_x": float(desired_x),
                    "used_relative_analysis": False,
                }
            else:
                render_pack = self._render_candidate_images(
                    client=client,
                    target_idx=int(target_idx),
                    save_dir=plan_dir,
                    prefix="v2_plan",
                )
                if not bool(render_pack.get("ok", False)):
                    return {"ok": False, "reason": render_pack.get("reason", "render_failed")}

                delta_plan = self.vlm_client.suggest_xpose_delta_with_vlm(
                    object_name=str(target_name),
                    relation_hint=str(relation_hint_target),
                    scene_graph=dict(scene_graph or {}),
                    object_context=dict(object_context or {}),
                    rule_diagnostics=dict(init_rule_named or {}),
                    view_images=dict(render_pack.get("view_images", {}) or {}),
                    input_rgb_u8=render_pack.get("input_rgb_u8", None),
                    overlay_rgb_u8=render_pack.get("overlay_rgb_u8", None),
                    backend_mode=getattr(self.state, "agent0_backend_mode", "local_openai_compatible"),
                    timeout=120.0,
                )
                if not bool(delta_plan.get("evaluated", False)):
                    fallback_eval = self._evaluate_pose_once(
                        client=client,
                        target_idx=int(target_idx),
                        object_name=target_name,
                        relation_hint=relation_hint_target,
                        scene_graph=scene_graph,
                        object_context=object_context,
                        x_min=float(x_min),
                        x_max=float(x_max),
                        save_dir=plan_dir,
                        prefix="v2_fallback_eval",
                    )
                    fallback_dir = self._normalize_direction(str(fallback_eval.get("suggested_direction", "none")))
                    if fallback_dir == "none":
                        fallback_dir = self._infer_direction_from_rule(dict(init_rule_named or {}))
                    signed = 0.0
                    if fallback_dir == "positive":
                        signed = 0.08
                    elif fallback_dir == "negative":
                        signed = -0.08
                    delta_plan = {
                        "ok": True,
                        "evaluated": True,
                        "provider": "fallback_rule",
                        "suggested_world_x_direction": str(fallback_dir),
                        "normalized_shift_signed": float(signed),
                        "normalized_shift_magnitude": float(abs(signed)),
                        "analysis_summary": "Fallback from rule/VLM direction without explicit delta response.",
                        "confidence": 0.3,
                    }

                up_point_plan = self._extract_up_view_target_point(dict(delta_plan or {}))
                up_views = dict(render_pack.get("view_images", {}) or {})
                up_rgb = up_views.get("up", None)
                use_up_point = False
                has_up_coords = bool(
                    self._is_finite_number((up_point_plan or {}).get("x_norm", None))
                    and self._is_finite_number((up_point_plan or {}).get("y_up_norm", None))
                )
                if bool(has_up_coords):
                    up_h = int(up_point_plan.get("image_height", 512))
                    up_w = int(up_point_plan.get("image_width", 512))
                    try:
                        if isinstance(up_rgb, np.ndarray) and up_rgb.ndim >= 2:
                            up_h = int(max(2, up_rgb.shape[0]))
                            up_w = int(max(2, up_rgb.shape[1]))
                    except Exception:
                        pass
                    if bool(up_point_plan.get("ok", False)):
                        cam_pack_up = self._camera_pose_pack_for_named_view(view_name="up", image_width=up_w, image_height=up_h)
                        up_point_plan["camera_pack"] = dict(cam_pack_up or {})
                        point_conversion: dict[str, Any] = {}
                        if bool(cam_pack_up.get("ok", False)):
                            point_conversion = self._solve_world_x_delta_from_up_point(
                                target_center_world=np.asarray(target_center_world, dtype=np.float32),
                                up_target_x_norm=float(up_point_plan.get("x_norm", 0.5)),
                                up_target_y_up_norm=float(up_point_plan.get("y_up_norm", 0.5)),
                                cam_pack=dict(cam_pack_up or {}),
                            )
                            up_point_plan["conversion"] = dict(point_conversion or {})
                            if bool(point_conversion.get("ok", False)):
                                use_up_point = True
                                desired_raw = float(
                                    point_conversion.get(
                                        "world_target_x",
                                        cur_x + float(point_conversion.get("world_delta_x", 0.0)),
                                    )
                                )
                                desired_world_x = float(self._clamp(desired_raw, float(x_min), float(x_max)))
                                move_world_dx = float(desired_world_x - cur_x)
                                conversion = dict(point_conversion or {})
                                conversion["desired_world_x_unclamped"] = float(desired_raw)
                                conversion["desired_world_x"] = float(desired_world_x)
                                ref_obj = str((delta_plan or {}).get("reference_object", "")).strip()
                                anchor_plan = {
                                    "has_reference": bool(ref_obj),
                                    "mode": "up_view_point_target",
                                    "reference_object_name": str(ref_obj),
                                    "desired_world_x": float(desired_world_x),
                                    "desired_world_x_unclamped": float(desired_raw),
                                    "used_relative_analysis": bool((delta_plan or {}).get("relative_analysis", [])),
                                    "point_reason": str(up_point_plan.get("point_reason", "")),
                                }
                                delta_plan["execution_mode"] = "up_view_point_target"
                                delta_plan["up_view_target_center_point_used"] = True
                    up_point_vis_path = self._save_up_point_debug_visualization(
                        up_rgb_u8=(None if up_rgb is None else np.asarray(up_rgb)),
                        save_dir=plan_dir,
                        prefix="v2_plan",
                        up_target_x_norm=float(up_point_plan.get("x_norm", 0.5)),
                        up_target_y_up_norm=float(up_point_plan.get("y_up_norm", 0.5)),
                        profile=(dict(up_point_plan.get("conversion", {}) or {})),
                    )
                    if up_point_vis_path:
                        up_point_plan["visualization_path"] = str(up_point_vis_path)

                if not bool(use_up_point):
                    dir_world = self._normalize_direction(delta_plan.get("suggested_world_x_direction", "none"))
                    signed_norm = float(delta_plan.get("normalized_shift_signed", 0.0))
                    mode_hint = str(delta_plan.get("target_alignment_mode", "")).strip().lower()
                    if mode_hint in {"offset_reference_center", "offset_from_reference_center", "offset_center"}:
                        try:
                            signed_norm = float(delta_plan.get("reference_offset_normalized_signed", signed_norm))
                        except Exception:
                            pass
                    if dir_world == "none":
                        if signed_norm > 0:
                            dir_world = "positive"
                        elif signed_norm < 0:
                            dir_world = "negative"
                    norm_mag = float(abs(signed_norm))
                    cam_pack = self._camera_input_pose_pack(client)
                    conversion = self._normalized_shift_to_world_x(
                        target_center_world=np.asarray(target_center_world, dtype=np.float32),
                        normalized_magnitude=float(norm_mag),
                        cam_pack=dict(cam_pack or {}),
                    )
                    if not bool(conversion.get("ok", False)):
                        conversion = {
                            "ok": True,
                            "source": "fallback_x_step",
                            "world_delta_magnitude": float(max(0.02, norm_mag) * max(1e-3, float(x_step))),
                        }
                    world_mag = float(max(0.0, conversion.get("world_delta_magnitude", 0.0)))
                    min_effective_shift = float(max(0.03, min(0.25, max(abs(float(x_step)), 0.05))))
                    if dir_world != "none" and world_mag < min_effective_shift:
                        conversion["world_delta_magnitude_before_min"] = float(world_mag)
                        conversion["world_delta_magnitude"] = float(min_effective_shift)
                        conversion["min_effective_shift_applied"] = True
                        world_mag = float(min_effective_shift)
                    sign = 1.0 if dir_world == "positive" else (-1.0 if dir_world == "negative" else 0.0)
                    offset_world_x = float(sign * world_mag)
                    anchor_plan = self._resolve_reference_alignment_plan(
                        delta_plan=dict(delta_plan or {}),
                        object_context=dict(object_context or {}),
                        cur_x=float(cur_x),
                        fallback_offset_world_x=float(offset_world_x),
                        x_min=float(x_min),
                        x_max=float(x_max),
                    )
                    desired_world_x = float(anchor_plan.get("desired_world_x", cur_x + offset_world_x))
                    desired_world_x = float(self._clamp(desired_world_x, float(x_min), float(x_max)))
                    move_world_dx = float(desired_world_x - cur_x)

            tr = self._translate_world_x(client=client, target_idx=int(target_idx), delta_x=float(move_world_dx))
            if not bool(tr.get("ok", False)):
                return {"ok": False, "reason": f"translate_failed:{tr.get('reason', 'unknown')}", "delta_x": float(move_world_dx)}

            pose_out = self.bridge.registry.call(
                "pose.pixel_align",
                client=client,
                object_idx=int(target_idx),
                lock_x=True,
                lock_y=False,
                lock_z=False,
                steps=int(pose_steps),
                lr=float(pose_lr),
                min_side=int(pose_min_side),
                rgb_weight=float(pose_rgb_weight),
                timeout_sec=float(pose_timeout),
            )
            if not bool(pose_out.get("ok", False)):
                return {"ok": False, "reason": f"pose_opt_failed:{pose_out.get('reason', 'unknown')}"}

            post_correct: dict[str, Any] = {"applied": False}
            post_rule = self.bridge.registry.call(
                "physics.check_aabb",
                object_idx=int(target_idx),
                x_min=float(x_min),
                x_max=float(x_max),
            )
            if bool(post_rule.get("ok", False)) and desired_world_x is not None:
                post_rule_named = self._rule_with_named_objects(dict(post_rule), object_context)
                post_center = None
                try:
                    post_center = float((post_rule_named.get("target_center_xyz", [None]) or [None])[0])
                except Exception:
                    post_center = None
                if self._is_finite_number(post_center):
                    drift = float(post_center - float(desired_world_x))
                    post_correct = {
                        "applied": False,
                        "center_world_x_before_correction": float(post_center),
                        "desired_world_x": float(desired_world_x),
                        "drift_before_correction": float(drift),
                        "alignment_tolerance": float(align_tol),
                    }
                    if abs(drift) > float(align_tol):
                        corr_dx = float(float(desired_world_x) - float(post_center))
                        tr2 = self._translate_world_x(client=client, target_idx=int(target_idx), delta_x=float(corr_dx))
                        post_correct["translate_correction"] = dict(tr2 or {})
                        if bool(tr2.get("ok", False)):
                            pose_out_2 = self.bridge.registry.call(
                                "pose.pixel_align",
                                client=client,
                                object_idx=int(target_idx),
                                lock_x=True,
                                lock_y=False,
                                lock_z=False,
                                steps=int(max(24, min(200, int(pose_steps) // 3))),
                                lr=float(pose_lr),
                                min_side=int(pose_min_side),
                                rgb_weight=float(pose_rgb_weight),
                                timeout_sec=float(max(120.0, min(float(pose_timeout), 600.0))),
                            )
                            post_correct["pose_opt_after_correction"] = dict(pose_out_2 or {})
                            post_correct["applied"] = True

            final_eval = self._evaluate_pose_once(
                client=client,
                target_idx=int(target_idx),
                object_name=target_name,
                relation_hint=relation_hint_target,
                scene_graph=scene_graph,
                object_context=object_context,
                x_min=float(x_min),
                x_max=float(x_max),
                save_dir=((debug_root / "final") if (bool(debug_save_views) and debug_root is not None) else None),
                prefix="v2_final",
            )
            if not bool(final_eval.get("ok", False)):
                return {"ok": False, "reason": final_eval.get("reason", "final_eval_failed")}

            final_rule = dict(final_eval.get("rule_diagnostics", {}) or {})
            final_vlm = dict(final_eval.get("vlm_eval", {}) or {})
            final_center_x = None
            try:
                final_center_x = float((final_rule.get("target_center_xyz", [None]) or [None])[0])
            except Exception:
                final_center_x = None
            x_target_alignment_ok = True
            x_target_alignment_error = None
            if desired_world_x is not None:
                if self._is_finite_number(final_center_x):
                    x_target_alignment_error = float(abs(float(final_center_x) - float(desired_world_x)))
                    x_target_alignment_ok = bool(float(x_target_alignment_error) <= float(align_tol))
                else:
                    x_target_alignment_ok = False
            relation_consistent = bool(final_vlm.get("relation_consistent_with_video", True))
            require_relation_consistency = bool((not is_first_generated) and bool(anchor_plan.get("has_reference", False)))
            final_ok = bool(
                bool(final_eval.get("final_ok", False))
                and bool(x_target_alignment_ok)
                and (bool(relation_consistent) if bool(require_relation_consistency) else True)
            )
            out = {
                "ok": bool(final_ok),
                "agent": "agent2_xpose_adjuster_v2",
                "target": str(target_name),
                "target_object_idx": int(target_idx),
                "is_first_generated_object": bool(is_first_generated),
                "x_bounds": [float(x_min), float(x_max)],
                "debug_save_views": bool(debug_save_views),
                "debug_root_dir": (str(debug_root) if debug_root is not None else ""),
                "object_relation_context": dict(object_context or {}),
                "target_relation_hint": str(relation_hint_target),
                "delta_plan": dict(delta_plan or {}),
                "anchor_plan": dict(anchor_plan or {}),
                "up_view_point_plan": dict(up_point_plan or {}),
                "up_view_point_vis_path": str(up_point_vis_path or ""),
                "shift_conversion": dict(conversion or {}),
                "desired_world_x": (None if desired_world_x is None else float(desired_world_x)),
                "alignment_tolerance": float(align_tol),
                "x_target_alignment_ok": bool(x_target_alignment_ok),
                "x_target_alignment_error": (None if x_target_alignment_error is None else float(x_target_alignment_error)),
                "relation_consistent_with_video": bool(relation_consistent),
                "require_relation_consistency": bool(require_relation_consistency),
                "applied_world_delta_x": float(move_world_dx),
                "translate_result": dict(tr or {}),
                "pose_opt_result": dict(pose_out or {}),
                "post_pose_x_correction": dict(post_correct or {}),
                "final_rule_diagnostics": dict(final_rule or {}),
                "final_vlm_eval": dict(final_vlm or {}),
                "view_paths": dict(final_eval.get("view_paths", {}) or {}),
                "input_path": str(final_eval.get("input_path", "") or ""),
                "overlay_path": str(final_eval.get("overlay_path", "") or ""),
            }
            with self.state.lock:
                self.state.last_pose_adjust = dict(out)
            if bool(final_ok):
                self.log("ok", f"agent2_v2_success:target={target_name},idx={target_idx},dx={move_world_dx:.5f}")
                self._notify(client, "Agent2 V2", f"Direct shift + pose-opt done: {target_name}", color="green", sec=6.0)
            else:
                self.log("fail", f"agent2_v2_failed:target={target_name},idx={target_idx},dx={move_world_dx:.5f}")
                self._notify(client, "Agent2 V2", f"Direct shift incomplete: {target_name}", color="orange", sec=8.0)
            return out
        finally:
            _restore_previous_current_object()


class FourDValidatorAgent(_AgentBase):
    _VIEW_ORDER = ("down", "up", "left", "front")

    def __init__(self, ctx, state, bridge, vlm_client) -> None:
        super().__init__(ctx, state, bridge, vlm_client, agent_name="agent-4_4d_validator")

    @staticmethod
    def _load_rgb_over_white(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(white_bg, rgba)
            return np.asarray(composited.convert("RGB"), dtype=np.uint8)

    def _artifact_run_dir(self) -> Path | None:
        with self.state.lock:
            root = getattr(self.state, "artifact_root", None)
        if root is None:
            return None
        try:
            out = Path(root) / "agent4_validator" / datetime.now().strftime("%Y%m%d_%H%M%S")
            out.mkdir(parents=True, exist_ok=True)
            return out
        except Exception:
            return None

    def _collect_window_frames(
        self,
        window_size: int,
        *,
        current_idx_override: int | None = None,
    ) -> tuple[list[int], list[str], int]:
        app = self.ctx.app
        seq = getattr(app, "sequence", None)
        frame_tokens = list(getattr(seq, "frame_tokens", []) or [])
        if not frame_tokens:
            return [], [], -1
        cur = int(getattr(app, "current_frame", 0) if current_idx_override is None else current_idx_override)
        cur = max(0, min(cur, len(frame_tokens) - 1))
        st = max(0, cur - max(0, int(window_size)))
        idxs = list(range(st, cur + 1))
        toks = [str(frame_tokens[i]) for i in idxs]
        return idxs, toks, cur

    def _resolve_input_image_path(self, frame_token: str) -> Path | None:
        app = self.ctx.app
        tok = str(frame_token).strip()
        if not tok:
            return None
        try:
            p = Path(app.workspace.image_path_for_view(tok, "input"))
            if p.exists():
                return p
        except Exception:
            pass
        input_dir = Path(getattr(app, "input_dir", ""))
        if not input_dir.exists():
            return None
        for ext in _IMAGE_EXTS:
            cand = input_dir / f"{tok}{ext}"
            if cand.exists():
                return cand
        return None

    def _restore_frame_state(self, client, prev_frame_idx: int) -> None:
        app = self.ctx.app
        try:
            app.current_frame = int(prev_frame_idx)
        except Exception:
            pass
        try:
            preview_map = getattr(app, "preview_handles", None)
            preview_handle = preview_map.get(client) if isinstance(preview_map, dict) else None
            apply_frame = getattr(app, "_apply_frame", None)
            if callable(apply_frame) and preview_handle is not None:
                apply_frame(int(prev_frame_idx), preview_handle)
                return
        except Exception:
            pass
        try:
            seq = getattr(app, "sequence", None)
            load_fn = getattr(seq, "load", None)
            upd_fn = getattr(app, "_update_gaussian_handle", None)
            if callable(load_fn) and callable(upd_fn):
                data = load_fn(int(prev_frame_idx))
                upd_fn(data)
        except Exception:
            pass

    def _render_multiview_window(
        self,
        *,
        client,
        frame_indices: list[int],
        frame_tokens: list[str],
        save_root: Path | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        packs: list[dict[str, Any]] = []
        fails: list[dict[str, Any]] = []
        app = self.ctx.app
        prev_frame = int(getattr(app, "current_frame", 0))
        try:
            for idx, tok in zip(frame_indices, frame_tokens):
                safe_tok = str(tok).replace("/", "_").replace("\\", "_").replace(" ", "_")
                mv = self.bridge.registry.call(
                    "render.frame_multiview",
                    client=client,
                    frame_idx=int(idx),
                    views=list(self._VIEW_ORDER),
                    image_size=768,
                    save_dir=(str(save_root) if save_root is not None else None),
                    prefix=f"f{int(idx):04d}_{safe_tok}",
                )
                if not bool(mv.get("ok", False)):
                    fails.append(
                        {
                            "frame_idx": int(idx),
                            "frame_token": str(tok),
                            "reason": str(mv.get("reason", "render_failed")),
                            "gs_path": str(mv.get("gs_path", "") or ""),
                        }
                    )
                    continue
                view_images_raw = dict(mv.get("view_images", {}) or {})
                ordered_view_images: dict[str, np.ndarray] = {}
                for vn in self._VIEW_ORDER:
                    arr = view_images_raw.get(vn, None)
                    if arr is not None:
                        ordered_view_images[str(vn)] = arr
                if not ordered_view_images:
                    fails.append(
                        {
                            "frame_idx": int(idx),
                            "frame_token": str(tok),
                            "reason": "empty_view_images",
                            "gs_path": str(mv.get("gs_path", "") or ""),
                        }
                    )
                    continue
                packs.append(
                    {
                        "frame_idx": int(idx),
                        "frame_token": str(tok),
                        "view_images": ordered_view_images,
                        "view_paths": dict(mv.get("view_paths", {}) or {}),
                        "gs_path": str(mv.get("gs_path", "") or ""),
                        "gs_source": str(mv.get("source", "sequence_ply") or "sequence_ply"),
                    }
                )
        finally:
            self._restore_frame_state(client, prev_frame)
        return packs, fails

    def run(self, client, window_size: int, frame_idx: int | None = None) -> dict[str, Any]:
        with self.state.lock:
            relation_hint = str(getattr(self.state, "relation_hint", "") or "")
            object_text = str(getattr(self.state, "object_description", "") or "")
            scene_graph = dict(getattr(self.state, "scene_graph", {}) or {})
            backend_mode = str(getattr(self.state, "agent0_backend_mode", "local_openai_compatible") or "local_openai_compatible")

        frame_indices, frame_tokens, current_idx = self._collect_window_frames(
            int(window_size),
            current_idx_override=frame_idx,
        )
        if not frame_indices:
            msg = "frame_tokens_empty"
            self.log("fail", msg)
            self._notify(client, "Agent-4", "No sequence frames found.", color="red")
            return {"ok": False, "reason": msg}

        save_root = self._artifact_run_dir()
        render_packs, render_failures = self._render_multiview_window(
            client=client,
            frame_indices=frame_indices,
            frame_tokens=frame_tokens,
            save_root=save_root,
        )
        if not render_packs:
            msg = "multiview_render_failed"
            self.log("fail", msg)
            self._notify(client, "Agent-4", "4D validator failed: cannot render multiview frames.", color="red", sec=8.0)
            return {
                "ok": False,
                "reason": msg,
                "window_size": int(window_size),
                "render_failures": render_failures,
            }

        render_packs.sort(key=lambda x: int(x.get("frame_idx", -1)))
        target_pack = None
        for pack in reversed(render_packs):
            if int(pack.get("frame_idx", -1)) == int(current_idx):
                target_pack = pack
                break
        if target_pack is None:
            msg = "target_frame_render_missing"
            self.log("fail", msg)
            self._notify(client, "Agent-4", "4D validator failed: current frame render is missing.", color="red", sec=8.0)
            return {
                "ok": False,
                "reason": msg,
                "window_size": int(window_size),
                "current_frame_idx": int(current_idx),
                "rendered_frame_indices": [int(x.get("frame_idx", -1)) for x in render_packs],
                "render_failures": render_failures,
            }

        target_token = str(target_pack.get("frame_token", "") or "")
        input_path = self._resolve_input_image_path(target_token)
        if input_path is None:
            msg = "target_input_image_missing"
            self.log("fail", msg)
            self._notify(client, "Agent-4", "4D validator failed: target input image missing.", color="red", sec=8.0)
            return {
                "ok": False,
                "reason": msg,
                "target_frame_token": target_token,
                "window_size": int(window_size),
            }
        try:
            input_rgb_u8 = self._load_rgb_over_white(input_path)
        except Exception as exc:
            msg = f"target_input_image_load_failed:{type(exc).__name__}:{exc}"
            self.log("fail", msg)
            self._notify(client, "Agent-4", "4D validator failed: cannot read target input image.", color="red", sec=8.0)
            return {
                "ok": False,
                "reason": msg,
                "target_frame_token": target_token,
                "input_path": str(input_path),
            }

        eval_frames = [
            {
                "frame_idx": int(x.get("frame_idx", -1)),
                "frame_token": str(x.get("frame_token", "")),
                "view_images": dict(x.get("view_images", {}) or {}),
            }
            for x in render_packs
        ]
        max_images = min(49, max(5, 1 + len(eval_frames) * len(self._VIEW_ORDER)))
        vlm_eval = self.vlm_client.evaluate_4d_frame_validator_with_vlm(
            object_text=str(object_text),
            relation_hint=str(relation_hint),
            scene_graph=dict(scene_graph or {}),
            rendered_frames=eval_frames,
            reference_input_rgb_u8=input_rgb_u8,
            backend_mode=backend_mode,
            timeout=150.0,
            max_images=int(max_images),
        )
        issue_diagnosis: dict[str, Any] = {}
        if not bool(vlm_eval.get("ok", False)):
            issue_diagnosis = self.vlm_client.diagnose_4d_validator_issue_with_vlm(
                object_text=str(object_text),
                relation_hint=str(relation_hint),
                scene_graph=dict(scene_graph or {}),
                rendered_frames=eval_frames,
                reference_input_rgb_u8=input_rgb_u8,
                validator_eval=dict(vlm_eval or {}),
                backend_mode=backend_mode,
                timeout=120.0,
                max_images=20,
            )
            if isinstance(issue_diagnosis, dict):
                cause = str(issue_diagnosis.get("root_cause", "unknown")).strip()
                action = str(issue_diagnosis.get("action_recommendation", "manual_review")).strip()
                feasible = bool(issue_diagnosis.get("sam3d_repair_feasible", False))
                phenomena = [
                    str(x).strip()
                    for x in list(issue_diagnosis.get("failure_phenomena", []) or [])
                    if str(x).strip()
                ]
                phenomenon_label = ", ".join(phenomena) if phenomena else "unspecified"
                self.log(
                    "info",
                    f"diagnosis:cause={cause},action={action},sam3d_feasible={feasible},phenomena={phenomenon_label}",
                )
                if action == "run_full_3d_gen_pipeline":
                    self._notify(
                        client,
                        "Agent-4 Repair Advice",
                        "Detected issue phenomena: "
                        f"{phenomenon_label}. Likely current-frame 3D quality issue and SAM3D repair is feasible. "
                        "Suggest clicking `Run Full 3D Gen Pipeline` on current frame.",
                        color="blue",
                        sec=10.0,
                    )
                elif action == "regenerate_with_memory_frame":
                    mem_rows = list(issue_diagnosis.get("memory_frame_candidates", []) or [])
                    mem_traits = [str(x).strip() for x in list(issue_diagnosis.get("memory_frame_characteristics", []) or []) if str(x).strip()]
                    mem_hint = ""
                    if mem_rows:
                        top = mem_rows[0]
                        mem_hint = (
                            f" Suggested memory frame: idx={int(top.get('frame_idx', -1))},"
                            f" token={str(top.get('frame_token', '')).strip()}."
                        )
                    trait_hint = ""
                    if mem_traits:
                        trait_hint = f" Memory frame traits: {'; '.join(mem_traits[:3])}."
                    sam3d_note = str(issue_diagnosis.get("sam3d_repair_rationale", "")).strip()
                    if sam3d_note:
                        sam3d_note = f" SAM3D note: {sam3d_note}"
                    self._notify(
                        client,
                        "Agent-4 Repair Advice",
                        "Detected issue phenomena: "
                        f"{phenomenon_label}. Likely memory-loss / insufficient current-frame observability."
                        " Suggest selecting a memory frame, then running model4dgen `Generate Next Segment with memory`"
                        f" to regenerate current frame.{mem_hint}{trait_hint}{sam3d_note}",
                        color="blue",
                        sec=12.0,
                    )

        rendered_meta = [
            {
                "frame_idx": int(x.get("frame_idx", -1)),
                "frame_token": str(x.get("frame_token", "")),
                "view_keys": [k for k in self._VIEW_ORDER if k in dict(x.get("view_images", {}) or {})],
                "view_paths": dict(x.get("view_paths", {}) or {}),
                "gs_path": str(x.get("gs_path", "") or ""),
                "gs_source": str(x.get("gs_source", "sequence_ply") or "sequence_ply"),
            }
            for x in render_packs
        ]
        out = {
            "ok": bool(vlm_eval.get("ok", False)),
            "agent": "agent-4_4d_validator",
            "window_size": int(window_size),
            "frame_window": {
                "current_frame_idx": int(current_idx),
                "requested_frame_indices": [int(x) for x in frame_indices],
                "requested_frame_tokens": [str(x) for x in frame_tokens],
                "rendered_frame_indices": [int(x.get("frame_idx", -1)) for x in render_packs],
                "rendered_gs_paths": [str(x.get("gs_path", "") or "") for x in render_packs],
            },
            "rendered_frames": rendered_meta,
            "render_failures": render_failures,
            "target_frame_token": target_token,
            "input_reference_path": str(input_path),
            "scene_graph_summary": {
                "objects": len(list(scene_graph.get("objects", []) or [])),
                "relations": len(list(scene_graph.get("relations", []) or [])),
                "events": len(list(scene_graph.get("events", []) or [])),
                "constraints": len(list(scene_graph.get("constraints", []) or [])),
            },
            "evaluation": dict(vlm_eval or {}),
            "issue_diagnosis": dict(issue_diagnosis or {}),
        }
        with self.state.lock:
            self.state.last_4d_validation = dict(out)

        score = int(dict(vlm_eval or {}).get("overall_score", 0))
        if bool(out.get("ok", False)):
            self.log("ok", f"score={score},frames={len(render_packs)},target={target_token}")
            self._notify(client, "Agent-4", f"4D validator done. score={score}", color="green", sec=7.0)
        else:
            self.log("fail", f"score={score},frames={len(render_packs)},target={target_token}")
            self._notify(client, "Agent-4", f"4D validator flagged issues. score={score}", color="orange", sec=8.0)
        return out
