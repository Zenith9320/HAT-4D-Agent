#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from http_security import (
    DEFAULT_MAX_REQUEST_BYTES,
    RequestError,
    parse_allowed_roots,
    read_json_body,
    request_is_authorized,
    require_secure_bind,
    validate_payload_paths,
)

TOOL_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_AGENT_ALIASES = {
    "video_understanding",
    "agent0",
    "agent_0",
    "agent0_video_understanding",
    "agent_0_video_understanding",
}


def _ensure_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"unsupported image shape: {arr.shape}")
    if arr.shape[2] == 4:
        arrf = arr.astype(np.float32)
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arrf)) <= 1.0:
            alpha = np.clip(arrf[..., 3:4], 0.0, 1.0)
            arrf = np.clip(arrf[..., :3], 0.0, 1.0) * alpha + (1.0 - alpha)
            arr = np.clip(arrf * 255.0, 0.0, 255.0).astype(np.uint8)
            return arr
        alpha = np.clip(arrf[..., 3:4] / 255.0, 0.0, 1.0)
        arrf = np.clip(arrf[..., :3], 0.0, 255.0) * alpha + (1.0 - alpha) * 255.0
        arr = np.clip(arrf, 0.0, 255.0).astype(np.uint8)
        return arr
    if arr.shape[2] != 3:
        raise ValueError(f"unsupported channel count: {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _sort_frame_paths(frame_dir: Path) -> list[Path]:
    paths = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    digit_paths = [p for p in paths if p.stem.isdigit()]
    if digit_paths:
        return sorted(digit_paths, key=lambda p: int(p.stem))
    return sorted(paths, key=lambda p: p.name)


def _uniform_sample_indices(total: int, max_frames: int) -> list[int]:
    if total <= 0:
        return []
    if max_frames <= 0 or total <= max_frames:
        return list(range(total))
    xs = np.linspace(0, total - 1, num=max_frames, dtype=np.int64)
    return sorted({int(x) for x in xs})


def _image_to_data_url(rgb_u8: np.ndarray, jpeg_quality: int = 70) -> str:
    img = Image.fromarray(rgb_u8)
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=int(jpeg_quality), optimize=True)
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + b64


def _decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("invalid data URL")
    _, b64 = data_url.split(",", 1)
    return base64.b64decode(b64)


def _decode_image_b64_to_rgb(image_b64: str) -> np.ndarray:
    raw = _decode_data_url(image_b64) if image_b64.startswith("data:") else base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as img:
        rgba = img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white_bg, rgba)
        return np.asarray(composited.convert("RGB"), dtype=np.uint8)


def _load_image_path_to_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white_bg, rgba)
        return np.asarray(composited.convert("RGB"), dtype=np.uint8)


def _resize_long_side(rgb_u8: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return rgb_u8
    h, w = rgb_u8.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return rgb_u8
    scale = float(max_side) / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(rgb_u8)
    pil = pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _strip_json_fence(text: str) -> str:
    s = str(text).strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s


def _try_parse_int(raw: Any) -> int | None:
    try:
        if raw is None:
            return None
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            return int(raw)
        s = str(raw).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _parse_range_text(range_text: str) -> tuple[int | None, int | None]:
    s = str(range_text or "").strip()
    if not s:
        return None, None
    m = re.search(r"(-?\d+)\s*[-_~至到]\s*(-?\d+)", s)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


def _normalize_order_list(raw: Any) -> list[str]:
    out: list[str] = []
    if isinstance(raw, list):
        seq = raw
    elif isinstance(raw, str):
        seq = re.split(r"\s*(?:->|→|>|,|，)\s*", raw.strip())
    else:
        seq = []
    for x in seq:
        s = str(x).strip()
        if not s:
            continue
        if s not in out:
            out.append(s)
    return out


def _normalize_x_depth_order_timeline(payload: dict[str, Any], *, frame_count: int) -> list[dict[str, Any]]:
    raw = payload.get(
        "x_depth_order_timeline",
        payload.get("depth_order_timeline", payload.get("x_axis_depth_order_timeline", [])),
    )
    rows: list[tuple[str, Any]] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            rows.append((str(k).strip(), v))
    elif isinstance(raw, list):
        for v in raw:
            rows.append(("", v))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, (fallback_range, item) in enumerate(rows):
        frame_range = ""
        start_idx: int | None = None
        end_idx: int | None = None
        order_raw: Any = []
        notes = ""
        conf: float | None = None

        if isinstance(item, dict):
            frame_range = str(
                item.get("frame_range", item.get("range", item.get("time_range", fallback_range)))
            ).strip()
            start_idx = _try_parse_int(item.get("start_frame_index", item.get("start_idx", item.get("start_frame"))))
            end_idx = _try_parse_int(item.get("end_frame_index", item.get("end_idx", item.get("end_frame"))))
            order_raw = item.get(
                "order_positive_to_negative",
                item.get(
                    "x_axis_positive_to_negative",
                    item.get("x_axis_order_pos_to_neg", item.get("x轴方向由正到负", item.get("order", []))),
                ),
            )
            notes = str(item.get("notes", item.get("summary", item.get("description", "")))).strip()
            try:
                if "confidence" in item:
                    conf = float(item.get("confidence"))
            except Exception:
                conf = None
        elif isinstance(item, list):
            frame_range = fallback_range
            order_raw = item
        elif isinstance(item, str):
            frame_range = fallback_range
            order_raw = item
        else:
            continue

        if (start_idx is None or end_idx is None) and frame_range:
            st2, ed2 = _parse_range_text(frame_range)
            if start_idx is None:
                start_idx = st2
            if end_idx is None:
                end_idx = ed2
        if start_idx is None and end_idx is not None:
            start_idx = int(end_idx)
        if end_idx is None and start_idx is not None:
            end_idx = int(start_idx)
        if start_idx is not None and end_idx is not None and start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        if frame_count > 0:
            hi = int(frame_count - 1)
            if start_idx is not None:
                start_idx = max(0, min(hi, int(start_idx)))
            if end_idx is not None:
                end_idx = max(0, min(hi, int(end_idx)))
            if start_idx is not None and end_idx is not None and start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

        order_list = _normalize_order_list(order_raw)
        if not order_list:
            continue

        if not frame_range:
            if start_idx is not None and end_idx is not None:
                frame_range = f"{int(start_idx)}-{int(end_idx)}"
            else:
                frame_range = f"range_{i}"
        key = f"{frame_range}|{'>'.join(order_list)}"
        if key in seen:
            continue
        seen.add(key)

        rec: dict[str, Any] = {
            "frame_range": str(frame_range),
            "order_positive_to_negative": list(order_list),
        }
        if start_idx is not None:
            rec["start_frame_index"] = int(start_idx)
        if end_idx is not None:
            rec["end_frame_index"] = int(end_idx)
        if conf is not None:
            rec["confidence"] = float(max(0.0, min(1.0, conf)))
        if notes:
            rec["notes"] = str(notes)
        out.append(rec)
        if len(out) >= 64:
            break
    return out


def _normalize_temporal_scene_graph(
    payload: dict[str, Any] | None,
    *,
    frame_tokens: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}

    known_tokens = set(frame_tokens or [])
    distance_allowed = {"near_zero", "near", "mid", "far", "unknown"}

    objects_raw = payload.get("objects", [])
    objects = [str(x).strip() for x in objects_raw if str(x).strip()]

    relations: list[dict[str, Any]] = []
    for item in payload.get("relations", []):
        if not isinstance(item, dict):
            continue
        sub = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        obj = str(item.get("obj", item.get("object", ""))).strip()
        if not (sub and pred and obj):
            continue
        distance_hint = str(item.get("distance_hint", "unknown")).strip()
        if distance_hint not in distance_allowed:
            distance_hint = "unknown"
        relations.append(
            {
                "subject": sub,
                "predicate": pred,
                "obj": obj,
                "phase": str(item.get("phase", "global")).strip() or "global",
                "confidence": float(item.get("confidence", 0.5)),
                "distance_hint": distance_hint,
            }
        )

    object_meta: list[dict[str, Any]] = []
    object_meta_raw = payload.get("object_meta", payload.get("objects_detail", []))
    if isinstance(object_meta_raw, list):
        for i, item in enumerate(object_meta_raw):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", item.get("id", f"object_{i}"))).strip() or f"object_{i}"
            object_meta.append(
                {
                    "id": str(item.get("id", name)).strip() or name,
                    "name": name,
                    "category": str(item.get("category", item.get("type", "object"))).strip() or "object",
                    "attributes": list(item.get("attributes", []))
                    if isinstance(item.get("attributes", []), list)
                    else [],
                }
            )

    phases: list[dict[str, Any]] = []
    for i, item in enumerate(payload.get("phases", [])):
        if not isinstance(item, dict):
            continue
        st = str(item.get("start_frame_token", item.get("start_frame", ""))).strip()
        ed = str(item.get("end_frame_token", item.get("end_frame", ""))).strip()
        if known_tokens and st and st not in known_tokens:
            st = ""
        if known_tokens and ed and ed not in known_tokens:
            ed = ""
        phases.append(
            {
                "name": str(item.get("name", f"phase_{i}")).strip() or f"phase_{i}",
                "start_frame_token": st,
                "end_frame_token": ed,
                "description": str(item.get("description", "")).strip(),
            }
        )

    events: list[dict[str, Any]] = []
    for item in payload.get("events", []):
        if not isinstance(item, dict):
            continue
        ev = {
            "type": str(item.get("type", "")).strip(),
            "subject": str(item.get("subject", "")).strip(),
            "obj": str(item.get("obj", item.get("object", ""))).strip(),
            "frame_token": str(item.get("frame_token", item.get("frame", ""))).strip(),
            "phase": str(item.get("phase", "global")).strip() or "global",
            "confidence": float(item.get("confidence", 0.5)),
        }
        if not ev["type"]:
            continue
        if known_tokens and ev["frame_token"] and ev["frame_token"] not in known_tokens:
            ev["frame_token"] = ""
        events.append(ev)

    constraints: list[dict[str, Any]] = []
    for item in payload.get("constraints", []):
        if isinstance(item, str):
            rule = item.strip()
            if rule:
                constraints.append({"rule": rule})
            continue
        if not isinstance(item, dict):
            continue
        rule = str(item.get("rule", "")).strip()
        if not rule:
            continue
        constraints.append(
            {
                "rule": rule,
                "target_pair": list(item.get("target_pair", []))
                if isinstance(item.get("target_pair", []), list)
                else [],
                "phase": str(item.get("phase", "global")).strip() or "global",
            }
        )

    x_depth_order_timeline = _normalize_x_depth_order_timeline(
        payload,
        frame_count=len(frame_tokens or []),
    )

    return {
        "objects": objects,
        "relations": relations,
        "notes": str(payload.get("notes", "")).strip(),
        "object_meta": object_meta,
        "phases": phases,
        "events": events,
        "constraints": constraints,
        "x_depth_order_timeline": x_depth_order_timeline,
    }


@dataclass
class BackendConfig:
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    default_max_frames: int = 0
    image_max_side: int = 512
    timeout_seconds: float = 240.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BackendConfig":
        key = str(os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is missing. Export it before starting the backend.")
        base_url = str(args.openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        model = str(args.openai_model or os.getenv("OPENAI_MODEL") or "gpt-4o").strip()
        return cls(
            openai_api_key=key,
            openai_base_url=base_url,
            openai_model=model,
            default_max_frames=int(args.default_max_frames),
            image_max_side=int(args.image_max_side),
            timeout_seconds=float(args.timeout),
        )


class GPT4oClient:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @staticmethod
    def _language_for_model(model_name: str | None) -> str:
        return "zh" if "qwen" in str(model_name or "").strip().lower() else "en"

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.openai_base_url + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.openai_api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.config.timeout_seconds)) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8")
            except Exception:
                pass
            msg = f"openai HTTPError {getattr(exc, 'code', '?')} {getattr(exc, 'reason', '')}"
            if raw:
                msg += f" | body={raw[:1000]}"
            raise RuntimeError(msg) from exc
        except Exception as exc:
            raise RuntimeError(f"openai request failed: {exc}") from exc

    @staticmethod
    def _extract_text(resp: dict[str, Any]) -> str:
        choices = resp.get("choices", [])
        if not choices:
            raise ValueError("no choices in openai response")
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
            return "\n".join(chunks)
        return str(content)

    def run_video_understanding(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_items: list[dict[str, Any]],
        frame_tokens: list[str],
    ) -> dict[str, Any]:
        language = self._language_for_model(self.config.openai_model)
        visual: list[dict[str, Any]] = []
        token_lines: list[str] = []
        for idx, item in enumerate(frame_items):
            rgb = item.get("rgb_u8")
            tok = str(item.get("frame_token", f"f{idx:03d}"))
            if rgb is None:
                continue
            token_lines.append(f"{idx}: frame_token={tok}")
            visual.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_url(rgb, jpeg_quality=70),
                        "detail": "low",
                    },
                }
            )

        if not visual:
            raise ValueError("no valid frames for model input")

        if language == "zh":
            system_prompt = (
                "你是 4D 场景编辑的视频理解专家代理。"
                "只输出 JSON，禁止输出数值化 6DoF 位姿。"
            )
            user_prompt = (
                "请分析给定所有视频帧并输出时序场景图（TSG）。\n"
                "重点关注：物体、相对位置、交互关系、时序阶段、接触事件。\n"
                "约束：\n"
                "1) 禁止输出数值位姿。\n"
                "2) distance_hint 只能取：near_zero | near | mid | far | unknown。\n"
                "3) 若发生接触，推断大致接触开始 frame_token。\n"
                "4) 保持跨帧物体命名一致。\n\n"
                "5) 输出分时段 x 轴深度排序：按世界坐标系 x 从正到负列出物体顺序（越负越远）。\n\n"
                "6) 必须输出分时段 relative_position_timeline：给出关键物体对在每个时段的左右/上下/前后关系。"
                "例如 0-10 帧 A 相对 B 可为 horizontal=left, vertical=below。\n\n"
                f"物体提示：{object_text}\n"
                f"关系提示：{relation_hint}\n"
                f"已知 frame token：{frame_tokens}\n"
                "帧映射：\n"
                + "\n".join(token_lines)
                + "\n\n"
                "返回 JSON（键名如下）：\n"
                "{\n"
                '  "objects": [string],\n'
                '  "object_meta": [{"id":string,"name":string,"category":string,"attributes":[string]}],\n'
                '  "phases": [{"name":string,"start_frame_token":string,"end_frame_token":string,"description":string}],\n'
                '  "relations": [{"subject":string,"predicate":string,"obj":string,"phase":string,"confidence":number,"distance_hint":string}],\n'
                '  "events": [{"type":string,"subject":string,"obj":string,"frame_token":string,"phase":string,"confidence":number}],\n'
                '  "constraints": [{"rule":string,"target_pair":[string,string],"phase":string}],\n'
                '  "x_depth_order_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"order_positive_to_negative":[string],"confidence":number,"notes":string}],\n'
                '  "relative_position_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"pairs":[{"subject":string,"obj":string,"horizontal":"left|right|overlap|unknown","vertical":"above|below|overlap|unknown","depth":"front|behind|same_depth|unknown","confidence":number,"notes":string}],"notes":string}],\n'
                '  "notes": string\n'
                "}\n"
                "请使用中文填写自由文本字段。"
            )
        else:
            system_prompt = (
                "You are an expert video-understanding agent for 4D scene editing. "
                "Output JSON only. Never output numeric 6DoF poses."
            )
            user_prompt = (
                "Analyze all provided video frames and output a Temporal Scene Graph (TSG).\n"
                "You must focus on: objects, relative positions, interaction relations, temporal phases, and contact events.\n"
                "Constraints:\n"
                "1) Never output numeric pose values.\n"
                "2) distance_hint must be one of: near_zero | near | mid | far | unknown.\n"
                "3) If contact happens, infer approximate contact start frame_token.\n"
                "4) Keep object naming consistent across frames.\n\n"
                "5) Output time-sliced x-axis depth ordering: list objects from world +X to -X (more negative means farther).\n\n"
                "6) You must output time-sliced relative_position_timeline with key pairwise left/right, above/below, and front/behind relations.\n\n"
                f"Object hint: {object_text}\n"
                f"Relation hint: {relation_hint}\n"
                f"Known frame tokens: {frame_tokens}\n"
                "Frame mapping:\n"
                + "\n".join(token_lines)
                + "\n\n"
                "Return JSON with keys:\n"
                "{\n"
                '  "objects": [string],\n'
                '  "object_meta": [{"id":string,"name":string,"category":string,"attributes":[string]}],\n'
                '  "phases": [{"name":string,"start_frame_token":string,"end_frame_token":string,"description":string}],\n'
                '  "relations": [{"subject":string,"predicate":string,"obj":string,"phase":string,"confidence":number,"distance_hint":string}],\n'
                '  "events": [{"type":string,"subject":string,"obj":string,"frame_token":string,"phase":string,"confidence":number}],\n'
                '  "constraints": [{"rule":string,"target_pair":[string,string],"phase":string}],\n'
                '  "x_depth_order_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"order_positive_to_negative":[string],"confidence":number,"notes":string}],\n'
                '  "relative_position_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"pairs":[{"subject":string,"obj":string,"horizontal":"left|right|overlap|unknown","vertical":"above|below|overlap|unknown","depth":"front|behind|same_depth|unknown","confidence":number,"notes":string}],"notes":string}],\n'
                '  "notes": string\n'
                "}"
            )

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(visual)
        payload = {
            "model": self.config.openai_model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        resp = self._post_chat_completions(payload)
        raw_text = self._extract_text(resp)
        parsed = json.loads(_strip_json_fence(raw_text))
        normalized = _normalize_temporal_scene_graph(parsed, frame_tokens=frame_tokens)
        return {
            "ok": True,
            "provider": "gpt4o",
            "model": self.config.openai_model,
            "scene_graph": normalized,
            "raw_scene_graph": parsed,
            "frames_used": len(visual),
        }


class VLMAgentService:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.gpt4o = GPT4oClient(config)

    def run_agent(self, agent: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent_name = str(agent or "").strip().lower()
        if agent_name in VIDEO_AGENT_ALIASES:
            return self._run_video_understanding(payload)
        return {
            "ok": False,
            "error": f"unsupported agent: {agent}",
            "supported_agents": sorted(VIDEO_AGENT_ALIASES),
        }

    def _build_frame_items(self, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        max_frames = int(payload.get("max_frames", self.config.default_max_frames))
        image_max_side = int(payload.get("image_max_side", self.config.image_max_side))
        out: list[dict[str, Any]] = []

        if isinstance(payload.get("frame_items"), list):
            source = "frame_items"
            frame_items = payload["frame_items"]
            for i, item in enumerate(frame_items):
                if not isinstance(item, dict):
                    continue
                tok = str(item.get("frame_token", f"f{i:03d}"))
                rgb: np.ndarray | None = None
                if item.get("image_path"):
                    path = Path(str(item["image_path"])).expanduser().resolve()
                    rgb = _ensure_rgb_u8(_load_image_path_to_rgb(path))
                elif item.get("image_b64"):
                    rgb = _decode_image_b64_to_rgb(str(item["image_b64"]))
                elif item.get("rgb_u8") is not None:
                    rgb = _ensure_rgb_u8(np.asarray(item.get("rgb_u8")))
                if rgb is None:
                    continue
                rgb = _resize_long_side(rgb, image_max_side)
                out.append({"frame_token": tok, "rgb_u8": rgb})
        elif isinstance(payload.get("frame_paths"), list):
            source = "frame_paths"
            for i, raw in enumerate(payload["frame_paths"]):
                path = Path(str(raw)).expanduser().resolve()
                rgb = _ensure_rgb_u8(_load_image_path_to_rgb(path))
                rgb = _resize_long_side(rgb, image_max_side)
                tok = str(payload.get("frame_tokens", [])[i]) if isinstance(payload.get("frame_tokens"), list) and i < len(payload.get("frame_tokens")) else path.stem
                out.append({"frame_token": tok, "rgb_u8": rgb, "path": str(path)})
        else:
            source = "video_dir"
            frame_dir = Path(str(payload.get("video_dir", ""))).expanduser().resolve()
            if not frame_dir.exists():
                raise FileNotFoundError(f"video_dir not found: {frame_dir}")
            paths = _sort_frame_paths(frame_dir)
            for path in paths:
                rgb = _ensure_rgb_u8(_load_image_path_to_rgb(path))
                rgb = _resize_long_side(rgb, image_max_side)
                out.append({"frame_token": path.stem, "rgb_u8": rgb, "path": str(path)})

        if not out:
            raise ValueError("no frames found; provide one of video_dir / frame_paths / frame_items")

        keep = _uniform_sample_indices(len(out), max_frames)
        sampled = [out[i] for i in keep]
        return sampled, source

    def _run_video_understanding(self, payload: dict[str, Any]) -> dict[str, Any]:
        frames, source = self._build_frame_items(payload)
        object_text = str(payload.get("object_text", "")).strip()
        relation_hint = str(payload.get("relation_hint", "")).strip()
        frame_tokens = [str(x.get("frame_token", "")) for x in frames]
        result = self.gpt4o.run_video_understanding(
            object_text=object_text,
            relation_hint=relation_hint,
            frame_items=frames,
            frame_tokens=frame_tokens,
        )
        result["agent"] = "video_understanding"
        result["source"] = source
        result["frame_tokens"] = frame_tokens
        result["frames_total_after_sampling"] = len(frames)
        return result


def health_payload() -> dict[str, str]:
    return {"status": "ok", "service": "vlm_backend"}


class VLMBackendRequestHandler(BaseHTTPRequestHandler):
    service: VLMAgentService | None = None
    api_token = ""
    max_request_bytes = DEFAULT_MAX_REQUEST_BYTES
    allowed_roots = parse_allowed_roots(TOOL_ROOT.parent)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[vlm-backend] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path == "/health":
            self._send_json(health_payload())
            return
        self._send_json({"ok": False, "error": "unsupported endpoint"}, code=404)

    def do_POST(self) -> None:  # noqa: N802
        if not request_is_authorized(self.headers, self.api_token):
            self._send_json({"ok": False, "error": "unauthorized"}, code=401)
            return
        try:
            payload = read_json_body(self, self.max_request_bytes)
            validate_payload_paths(payload, self.allowed_roots)
        except RequestError as exc:
            self._send_json({"ok": False, "error": str(exc)}, code=exc.status)
            return
        except Exception as exc:
            self._send_json({"ok": False, "error": f"invalid json: {exc}"}, code=400)
            return

        path = self.path.rstrip("/")
        if path not in {"/agent/run", "/agent/video_understanding"}:
            self._send_json({"ok": False, "error": "unsupported endpoint"}, code=404)
            return
        try:
            if self.service is None:
                raise RuntimeError("VLM agent service has not been initialized")
            if path == "/agent/run":
                agent = payload.get("agent", "")
                agent_input = payload.get("input", {})
                if not isinstance(agent_input, dict):
                    raise ValueError("payload.input must be a JSON object")
                result = self.service.run_agent(str(agent), agent_input)
            elif path == "/agent/video_understanding":
                result = self.service.run_agent("video_understanding", payload)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: request failed; see server logs",
                },
                code=500,
            )
            return

        code = 200 if bool(result.get("ok", False)) else 400
        self._send_json(result, code=code)

    def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(code))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM Assistor backend for GPT-4o video understanding")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--api-token", default=os.environ.get("HAT4D_API_TOKEN", ""))
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument("--openai-base-url", type=str, default=None)
    parser.add_argument("--openai-model", type=str, default=None)
    parser.add_argument(
        "--default-max-frames",
        type=int,
        default=0,
        help="0 means use all frames; >0 means uniform sampling before VLM call.",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=512,
        help="Resize each frame so long side <= this value. 0 disables resize.",
    )
    parser.add_argument("--timeout", type=float, default=240.0, help="OpenAI request timeout (seconds).")
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Serve health checks without configuring a VLM client.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    require_secure_bind(args.host, args.api_token)
    VLMBackendRequestHandler.service = None
    VLMBackendRequestHandler.api_token = str(args.api_token or "")
    VLMBackendRequestHandler.max_request_bytes = int(args.max_request_bytes)
    VLMBackendRequestHandler.allowed_roots = parse_allowed_roots(TOOL_ROOT.parent)
    if not args.health_only:
        config = BackendConfig.from_args(args)
        VLMBackendRequestHandler.service = VLMAgentService(config)
    server = ThreadingHTTPServer((args.host, int(args.port)), VLMBackendRequestHandler)
    if args.health_only:
        print(f"[vlm-backend] health endpoint listening on http://{args.host}:{int(args.port)}")
    else:
        print(
            "[vlm-backend] listening on "
            f"http://{args.host}:{int(args.port)} "
            f"(model={config.openai_model}, base={config.openai_base_url}, "
            f"default_max_frames={config.default_max_frames})"
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
