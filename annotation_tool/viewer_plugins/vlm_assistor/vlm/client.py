from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_LOCAL_VLM_BASE_URL, DEFAULT_LOCAL_VLM_MODEL


class VLMBackendClient:
    """VLM bridge.

    Supports four backends:
    1) Cross-machine GPFS toolkit (`VLM_toolkit_cross_gpfs`)
    2) Direct OpenAI API (fallback)
    3) Local OpenAI-compatible VLM endpoint
    4) Optional HTTP backend (`/agent/run`)
    """

    def __init__(self, app):
        self.app = app

    _BILINGUAL_NAME_ALIASES: dict[str, tuple[str, ...]] = {
        "苹果": ("apple", "apples"),
        "apple": ("苹果", "apples"),
        "apples": ("苹果", "apple"),
        "香蕉": ("banana", "bananas"),
        "banana": ("香蕉", "bananas"),
        "bananas": ("香蕉", "banana"),
        "刀": ("knife", "blade", "knives"),
        "knife": ("刀", "blade", "knives"),
        "blade": ("刀", "knife", "knives"),
        "knives": ("刀", "knife", "blade"),
        "削皮机": ("peeler", "peeling machine", "peeler machine"),
        "peeler": ("削皮机", "peeling machine", "peeler machine"),
        "peeling machine": ("削皮机", "peeler", "peeler machine"),
        "peeler machine": ("削皮机", "peeler", "peeling machine"),
    }
    _ENGLISH_CANONICAL_NAMES: dict[str, str] = {
        "苹果": "apple",
        "apple": "apple",
        "apples": "apple",
        "香蕉": "banana",
        "banana": "banana",
        "bananas": "banana",
        "刀": "knife",
        "knife": "knife",
        "knives": "knife",
        "blade": "blade",
        "削皮机": "peeler",
        "peeler": "peeler",
        "peeling machine": "peeler",
        "peeler machine": "peeler",
    }
    _CHINESE_CANONICAL_NAMES: dict[str, str] = {
        "苹果": "苹果",
        "apple": "苹果",
        "apples": "苹果",
        "香蕉": "香蕉",
        "banana": "香蕉",
        "bananas": "香蕉",
        "刀": "刀",
        "knife": "刀",
        "knives": "刀",
        "blade": "刀片",
        "削皮机": "削皮机",
        "peeler": "削皮机",
        "peeling machine": "削皮机",
        "peeler machine": "削皮机",
    }

    def _base_url(self) -> str | None:
        url = getattr(self.app, "vlm_agent_url", None)
        if not url:
            return None
        return str(url).rstrip("/")

    def available(self) -> bool:
        return self._base_url() is not None

    @staticmethod
    def _clean_opt_str(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def _agent0_backend_mode(self) -> str:
        mode = (
            getattr(self.app, "vlm_agent0_backend_mode", None)
            or os.getenv("VLM_AGENT0_BACKEND_MODE")
            or "official_api"
        )
        s = str(mode).strip()
        return s if s in {"official_api", "local_openai_compatible"} else "official_api"

    def _openai_api_key(self) -> str | None:
        key = getattr(self.app, "vlm_openai_api_key", None) or os.getenv("OPENAI_API_KEY")
        return self._clean_opt_str(key)

    def _openai_base_url(self) -> str:
        base = (
            getattr(self.app, "vlm_openai_base_url", None)
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        return str(base).rstrip("/")

    def _openai_model(self) -> str:
        return str(getattr(self.app, "vlm_openai_model", None) or os.getenv("OPENAI_MODEL") or "gpt-4o")

    def _local_vlm_base_url(self) -> str | None:
        base = (
            getattr(self.app, "vlm_local_base_url", None)
            or os.getenv("VLM_LOCAL_BASE_URL")
            or DEFAULT_LOCAL_VLM_BASE_URL
        )
        s = self._clean_opt_str(base)
        return s.rstrip("/") if s else None

    def _local_vlm_model(self) -> str:
        return str(
            getattr(self.app, "vlm_local_model", None)
            or os.getenv("VLM_LOCAL_MODEL")
            or DEFAULT_LOCAL_VLM_MODEL
        )

    def _local_vlm_api_key(self) -> str | None:
        key = getattr(self.app, "vlm_local_api_key", None) or os.getenv("VLM_LOCAL_API_KEY")
        return self._clean_opt_str(key)

    def _local_vlm_video_url(self) -> str | None:
        u = getattr(self.app, "vlm_local_video_url", None) or os.getenv("VLM_LOCAL_VIDEO_URL")
        return self._clean_opt_str(u)

    @staticmethod
    def _normalize_local_video_source(raw: str | None) -> tuple[str, str | None]:
        """
        Return (mode, value):
        - ("none", None)
        - ("frame_dir", "/abs/path/to/dir")
        - ("video_url", "file:///.../x.mp4" or "http(s)://...")
        - ("invalid", raw)
        """
        s = (str(raw).strip() if raw is not None else "")
        if not s:
            return "none", None

        lower = s.lower()
        if lower.startswith("http://") or lower.startswith("https://"):
            return "video_url", s

        if lower.startswith("file://"):
            path_raw = urllib.parse.unquote(s[len("file://") :])
            p = Path(path_raw).expanduser()
            try:
                p = p.resolve()
            except Exception:
                pass
            if p.exists() and p.is_dir():
                return "frame_dir", str(p)
            if p.exists() and p.is_file():
                return "video_url", f"file://{p}"
            return "invalid", s

        # Plain local filesystem path.
        if s.startswith("/") or s.startswith("./") or s.startswith("../") or (len(s) > 1 and s[1] == ":"):
            p = Path(s).expanduser()
            try:
                p = p.resolve()
            except Exception:
                pass
            if p.exists() and p.is_dir():
                return "frame_dir", str(p)
            if p.exists() and p.is_file():
                return "video_url", f"file://{p}"
            return "invalid", s

        # Treat all other strings as opaque URL accepted by local endpoint.
        return "video_url", s

    @staticmethod
    def _to_data_url(rgb_u8, *, jpeg_quality: int = 70) -> str | None:
        try:
            from PIL import Image
        except Exception:
            return None
        try:
            img = Image.fromarray(rgb_u8)
            bio = io.BytesIO()
            img.save(bio, format="JPEG", quality=int(jpeg_quality), optimize=True)
            b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
            return "data:image/jpeg;base64," + b64
        except Exception:
            return None

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        s = str(text).strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:].strip()
        return s

    def _post_chat_completions(
        self,
        payload: dict[str, Any],
        *,
        base_url: str,
        api_key: str | None,
        timeout: float = 120.0,
        require_api_key: bool = True,
    ) -> dict[str, Any] | None:
        if require_api_key and not api_key:
            return None
        url = str(base_url).rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except Exception:
            return None

    def _post_openai(self, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any] | None:
        return self._post_chat_completions(
            payload,
            base_url=self._openai_base_url(),
            api_key=self._openai_api_key(),
            timeout=timeout,
            require_api_key=True,
        )

    def infer_physical_properties(
        self,
        *,
        object_name: str,
        scene_graph: dict[str, Any],
        backend_mode: str | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        """Infer continuous physical attributes without category-to-mode rules."""
        mode = str(backend_mode or self._agent0_backend_mode())
        schema = {
            "rigidity": "0..1; resistance to shape change",
            "elasticity": "0..1; reversible deformation tendency",
            "articulatedness": "0..1; motion through persistent joints",
            "sheet_likeness": "0..1; thin surface dominated structure",
            "topology_stability": "0..1; likelihood topology stays fixed",
            "deformation_observed": "0..1; visible non-rigid deformation in the video",
            "confidence": "0..1",
            "evidence": "short explanation based on observations and common physical knowledge",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a physical property estimator for 4D reconstruction. Infer continuous "
                    "properties, not an object-class-specific implementation. Return one JSON object "
                    "only. Do not choose a solver or motion representation. Use conservative scores "
                    "when evidence is weak."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target object: {object_name}\n"
                    f"Temporal scene graph: {json.dumps(scene_graph, ensure_ascii=False)}\n"
                    f"Required fields: {json.dumps(schema, ensure_ascii=False)}"
                ),
            },
        ]
        if mode == "official_api":
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
            payload = {
                "model": self._openai_model(),
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": messages,
            }
            response = self._post_chat_completions(
                payload, base_url=base_url, api_key=api_key, timeout=timeout, require_api_key=True
            )
            provider = "official_api"
        else:
            base_url = self._local_vlm_base_url()
            api_key = self._local_vlm_api_key()
            if not base_url:
                return {"ok": False, "reason": "local_vlm_base_url_missing"}
            payload = {
                "model": self._local_vlm_model(),
                "temperature": 0.0,
                "messages": self._convert_system_to_user_prefix(messages),
            }
            response = self._post_chat_completions(
                payload, base_url=base_url, api_key=api_key, timeout=timeout, require_api_key=False
            )
            provider = "local_openai_compatible"
        if not isinstance(response, dict):
            return {"ok": False, "reason": f"{provider}_http_failed"}
        try:
            text = self._extract_message_text(response["choices"][0]["message"]["content"])
            parsed = self._try_parse_json_obj(text)
            if not isinstance(parsed, dict):
                raise ValueError("json_object_required")
            return {"ok": True, "provider": provider, "physical_properties": parsed}
        except Exception as exc:
            return {"ok": False, "reason": f"invalid_physics_json:{type(exc).__name__}:{exc}"}

    @staticmethod
    def _normalize_click_points(
        raw: Any,
        *,
        width: int,
        height: int,
        limit: int,
    ) -> list[tuple[int, int]]:
        if int(limit) <= 0:
            return []
        items: list[Any] = []
        if isinstance(raw, list):
            items = list(raw)
        elif isinstance(raw, dict):
            cand = raw.get("points", [])
            if isinstance(cand, list):
                items = list(cand)

        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        w = max(1, int(width))
        h = max(1, int(height))
        for it in items:
            x = None
            y = None
            if isinstance(it, dict):
                x = it.get("x", it.get("u", it.get("col")))
                y = it.get("y", it.get("v", it.get("row")))
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                x = it[0]
                y = it[1]
            if x is None or y is None:
                continue
            try:
                xf = float(x)
                yf = float(y)
            except Exception:
                continue
            # Accept normalized [0,1] coordinates too.
            if 0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0 and (w > 2 and h > 2):
                xf = xf * float(w - 1)
                yf = yf * float(h - 1)
            xi = int(round(xf))
            yi = int(round(yf))
            xi = max(0, min(w - 1, xi))
            yi = max(0, min(h - 1, yi))
            key = (xi, yi)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= int(limit):
                break
        return out

    @staticmethod
    def _pad_click_points(
        points: list[tuple[int, int]],
        *,
        width: int,
        height: int,
        target_count: int,
    ) -> list[tuple[int, int]]:
        w = max(1, int(width))
        h = max(1, int(height))
        target = max(1, int(target_count))
        out = list(points)
        if not out:
            out = [(w // 2, h // 2)]
        seed_x, seed_y = out[0]
        step = max(8, int(round(min(w, h) * 0.03)))
        offsets = [
            (0, 0),
            (step, 0),
            (0, step),
            (-step, 0),
            (0, -step),
            (step, step),
            (-step, step),
            (step, -step),
        ]
        seen = set(out)
        for dx, dy in offsets:
            if len(out) >= target:
                break
            x = max(0, min(w - 1, int(seed_x + dx)))
            y = max(0, min(h - 1, int(seed_y + dy)))
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out[:target]

    @staticmethod
    def _center_band_refine_points(
        points: list[tuple[int, int]],
        *,
        width: int,
        height: int,
        shrink: float = 0.68,
        max_move_ratio: float = 0.08,
    ) -> list[tuple[int, int]]:
        """Pull positive points toward interior centers while preserving multi-part coverage."""
        if len(points) <= 1:
            return list(points)
        w = max(1, int(width))
        h = max(1, int(height))
        s = float(max(0.45, min(0.92, float(shrink))))
        max_move_px = float(max(10.0, min(float(min(w, h)) * 0.18, float(min(w, h)) * max(0.02, float(max_move_ratio)))))
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        n = int(arr.shape[0])
        diag = float(np.hypot(float(w), float(h)))
        cluster_dist_px = float(max(18.0, min(diag * 0.26, float(min(w, h)) * 0.34)))
        comps: list[list[int]] = []
        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = [i]
            while stack:
                cur = int(stack.pop())
                for j in range(n):
                    if visited[j]:
                        continue
                    d = float(np.hypot(float(arr[cur, 0] - arr[j, 0]), float(arr[cur, 1] - arr[j, 1])))
                    if d <= cluster_dist_px:
                        visited[j] = True
                        stack.append(j)
                        comp.append(j)
            comps.append(comp)
        multipart_hypothesis = len(comps) >= 2
        local_shrink = float(max(s, 0.78)) if multipart_hypothesis else float(s)
        local_max_move_px = float(min(max_move_px, max(8.0, float(min(w, h)) * 0.06))) if multipart_hypothesis else float(max_move_px)
        comp_of_idx = {}
        comp_centers: dict[int, tuple[float, float]] = {}
        for ci, comp in enumerate(comps):
            for idx in comp:
                comp_of_idx[int(idx)] = int(ci)
            sub = arr[np.asarray(comp, dtype=np.int32)]
            comp_centers[int(ci)] = (float(np.mean(sub[:, 0])), float(np.mean(sub[:, 1])))
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for i, (x, y) in enumerate(points):
            xf = float(x)
            yf = float(y)
            ci = int(comp_of_idx.get(int(i), 0))
            cx, cy = comp_centers.get(ci, (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))))
            tx = cx + (xf - cx) * local_shrink
            ty = cy + (yf - cy) * local_shrink
            mvx = tx - xf
            mvy = ty - yf
            mv = float(np.hypot(mvx, mvy))
            if mv > local_max_move_px and mv > 1e-6:
                scale = float(local_max_move_px / mv)
                tx = xf + mvx * scale
                ty = yf + mvy * scale
            xi = int(round(max(0.0, min(float(w - 1), tx))))
            yi = int(round(max(0.0, min(float(h - 1), ty))))
            key = (xi, yi)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        if not out:
            return list(points)
        if multipart_hypothesis and len(out) >= 2:
            orig_major = float(
                max(
                    float(np.max(arr[:, 0]) - np.min(arr[:, 0])),
                    float(np.max(arr[:, 1]) - np.min(arr[:, 1])),
                )
            )
            out_arr = np.asarray(out, dtype=np.float32).reshape(-1, 2)
            out_major = float(
                max(
                    float(np.max(out_arr[:, 0]) - np.min(out_arr[:, 0])),
                    float(np.max(out_arr[:, 1]) - np.min(out_arr[:, 1])),
                )
            )
            if orig_major > 1.0 and (out_major / orig_major) < 0.72:
                return list(points)
        return out

    @staticmethod
    def _normalize_name_alias_text(text: str) -> str:
        s = str(text or "").strip().lower()
        if not s:
            return ""
        # Keep alnum/underscore/CJK, normalize separators to spaces.
        s = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _expand_name_aliases(cls, names: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        queue: list[str] = []
        for n in list(names or []):
            norm = cls._normalize_name_alias_text(str(n))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
            queue.append(norm)

        i = 0
        while i < len(queue):
            cur = queue[i]
            i += 1
            for alias in cls._BILINGUAL_NAME_ALIASES.get(cur, ()):
                a = cls._normalize_name_alias_text(alias)
                if not a or a in seen:
                    continue
                seen.add(a)
                out.append(a)
                queue.append(a)
        return out

    @staticmethod
    def _text_contains_alias(text: str, alias: str) -> bool:
        t = str(text or "").lower()
        a = str(alias or "").lower()
        if not t or not a:
            return False
        if re.fullmatch(r"[0-9a-z_ ]+", a):
            pat = r"\b" + re.escape(a) + r"\b"
            return re.search(pat, t) is not None
        return a in t

    @classmethod
    def _missing_part_detail_is_specific(
        cls,
        item: dict[str, Any],
        *,
        target_name_norm_en: str,
    ) -> bool:
        if not isinstance(item, dict):
            return False
        part_name = str(
            item.get(
                "missing_part_name",
                item.get("part_name", item.get("target_part", item.get("part", ""))),
            )
        ).strip()
        reason = str(item.get("reason", "")).strip()
        detail_text = part_name if part_name else reason
        norm = cls._normalize_name_alias_text(detail_text)
        if not norm:
            return False

        generic_tokens = {
            "missing",
            "part",
            "missing part",
            "target part",
            "target missing part",
            "target is not fully contained",
            "not fully contained",
            "outside mask",
            "which target part is missing",
            "target_missing_part",
            "target_missing",
            "missing_parts",
            "missing part of target",
            "目标缺失",
            "缺失部分",
            "目标缺失部分",
            "目标未完整包含",
            "未完整包含",
            "漏分",
            "目标漏分",
            "目标不完整",
            "不完整",
            "目标部分缺失",
        }
        if norm in generic_tokens:
            return False

        target_aliases = set(cls._expand_name_aliases([target_name_norm_en]))
        detail_aliases = set(cls._expand_name_aliases([detail_text]))
        if detail_aliases and detail_aliases.issubset(target_aliases):
            return False
        return True

    @classmethod
    def _is_occlusion_text(cls, text: str) -> bool:
        s = cls._normalize_name_alias_text(str(text or ""))
        if not s:
            return False
        keywords = (
            "occlud",
            "遮挡",
            "被挡住",
            "被遮住",
            "挡住",
            "挡住了",
            "遮住",
            "不可见",
            "看不见",
            "hidden behind",
            "blocked by",
            "covered by",
        )
        return any(k in s for k in keywords)

    @staticmethod
    def _is_near_rel_point(a: dict[str, Any], b: dict[str, Any], tol: float = 0.08) -> bool:
        try:
            ax = float(a.get("x", -1.0))
            ay = float(a.get("y", -1.0))
            bx = float(b.get("x", -1.0))
            by = float(b.get("y", -1.0))
        except Exception:
            return False
        if not (0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0 and 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0):
            return False
        return (abs(ax - bx) <= float(tol)) and (abs(ay - by) <= float(tol))

    @classmethod
    def to_english_object_name(cls, name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""
        norm = cls._normalize_name_alias_text(raw)
        if not norm:
            return raw
        mapped = cls._ENGLISH_CANONICAL_NAMES.get(norm)
        if mapped:
            return str(mapped)
        # Keep original if we do not know a safe translation.
        return raw

    @classmethod
    def to_english_object_names(cls, names: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for n in list(names or []):
            en = cls.to_english_object_name(str(n))
            s = str(en).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    @classmethod
    def to_chinese_object_name(cls, name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""
        norm = cls._normalize_name_alias_text(raw)
        if not norm:
            return raw
        mapped = cls._CHINESE_CANONICAL_NAMES.get(norm)
        if mapped:
            return str(mapped)
        return raw

    @classmethod
    def to_chinese_object_names(cls, names: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for n in list(names or []):
            zh = cls.to_chinese_object_name(str(n))
            s = str(zh).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    @staticmethod
    def _language_for_model(model_name: str | None) -> str:
        s = str(model_name or "").strip().lower()
        return "zh" if ("qwen" in s) else "en"

    def _resolve_agent1_mask_eval_language(self, model_name: str | None) -> tuple[str, str]:
        raw = (
            getattr(self.app, "vlm_agent1_mask_eval_language", None)
            or os.getenv("VLM_AGENT1_MASK_EVAL_LANGUAGE")
            or "auto"
        )
        s = str(raw).strip().lower()
        if s in {"zh", "cn", "chinese"}:
            return "zh", "forced"
        if s in {"en", "english"}:
            return "en", "forced"
        return self._language_for_model(model_name), "model_auto"

    @classmethod
    def _to_language_object_name(cls, name: str, language: str) -> str:
        if str(language) == "zh":
            return cls.to_chinese_object_name(name)
        return cls.to_english_object_name(name)

    @classmethod
    def _to_language_object_names(cls, names: list[str], language: str) -> list[str]:
        if str(language) == "zh":
            return cls.to_chinese_object_names(names)
        return cls.to_english_object_names(names)

    @classmethod
    def _to_language_object_text(cls, text: str, language: str) -> str:
        s = str(text or "")
        for sep in ["，", ";", "；", "|", "\n", "\t"]:
            s = s.replace(sep, ",")
        names = [str(x).strip() for x in s.split(",") if str(x).strip()]
        if not names:
            return str(text or "")
        mapped = cls._to_language_object_names(names, language)
        return ", ".join(mapped) if mapped else str(text or "")

    @staticmethod
    def strip_interaction_knowledge_graph(scene_graph: dict[str, Any] | None) -> dict[str, Any]:
        """Keep object discovery while removing detailed interaction knowledge."""
        graph = dict(scene_graph or {})
        pseudo_names = {"无", "none", "null", "n/a", "na", "nothing", "no object", "no objects"}
        objects = []
        for x in list(graph.get("objects", []) or []):
            name = str(x).strip()
            if not name or name.lower() in pseudo_names:
                continue
            objects.append(name)
        object_meta = []
        for item in list(graph.get("object_meta", []) or []):
            if isinstance(item, dict):
                name = str(item.get("name", item.get("id", ""))).strip()
                if name and name.lower() in pseudo_names:
                    continue
                object_meta.append(dict(item))
        return {
            "objects": objects,
            "object_meta": object_meta,
            "relations": [],
            "events": [],
            "constraints": [],
            "x_depth_order_timeline": [],
            "relative_position_timeline": [],
            "notes": "IKG disabled: object-instance discovery only.",
            "ikg_enabled": False,
            "ablation_mode": "wo_IKG",
        }

    @classmethod
    def _normalize_object_name_list(cls, raw: Any, limit: int = 32, language: str = "en") -> list[str]:
        if int(limit) <= 0:
            return []
        items: list[Any]
        if isinstance(raw, list):
            items = list(raw)
        elif isinstance(raw, dict):
            cand = raw.get("objects", raw.get("names", []))
            items = list(cand) if isinstance(cand, list) else []
        else:
            items = []
        out: list[str] = []
        seen: set[str] = set()
        for it in items:
            s = str(it).strip()
            if not s:
                continue
            name_lang = cls._to_language_object_name(s, language)
            key = str(cls.to_english_object_name(s)).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(name_lang).strip())
            if len(out) >= int(limit):
                break
        return out

    @classmethod
    def _name_alias_set(cls, name: str) -> set[str]:
        s = str(name or "").strip()
        if not s:
            return set()
        aliases = cls._expand_name_aliases(
            [
                s,
                cls.to_english_object_name(s),
                cls.to_chinese_object_name(s),
            ]
        )
        return {str(a).strip().lower() for a in aliases if str(a).strip()}

    @classmethod
    def _name_matches_alias_set(cls, name: str, alias_set: set[str]) -> bool:
        if not alias_set:
            return False
        cur = cls._name_alias_set(str(name or ""))
        return len(cur.intersection(set(alias_set))) > 0

    def _build_target_relation_context(
        self,
        *,
        object_name: str,
        scene_graph: dict[str, Any] | None,
        object_context: dict[str, Any] | None,
        max_items: int = 24,
    ) -> dict[str, Any]:
        graph = dict(scene_graph or {})
        ctx = dict(object_context or {})
        target = str(object_name or "").strip()
        target_aliases = self._name_alias_set(target)

        generated_names: list[str] = []
        seen_gen: set[str] = set()
        cur_obj = dict(ctx.get("current_generated_object", {}) or {})
        cur_name = str(cur_obj.get("object_name", "")).strip()
        if cur_name:
            k = str(self.to_english_object_name(cur_name)).strip().lower() or cur_name.lower()
            if k not in seen_gen:
                seen_gen.add(k)
                generated_names.append(cur_name)
        for it in list(ctx.get("already_generated_objects", []) or []):
            if not isinstance(it, dict):
                continue
            n = str(it.get("object_name", "")).strip()
            if not n:
                continue
            k = str(self.to_english_object_name(n)).strip().lower() or n.lower()
            if k in seen_gen:
                continue
            seen_gen.add(k)
            generated_names.append(n)
            if len(generated_names) >= 32:
                break

        generated_alias_map: dict[str, set[str]] = {}
        for n in generated_names:
            generated_alias_map[str(n)] = self._name_alias_set(str(n))

        def _is_generated_name(name: str) -> bool:
            for aliases in generated_alias_map.values():
                if self._name_matches_alias_set(name, aliases):
                    return True
            return False

        relations_hit: list[dict[str, Any]] = []
        for rel in list(graph.get("relations", []) or []):
            if not isinstance(rel, dict):
                continue
            sub = str(rel.get("subject", "")).strip()
            obj = str(rel.get("obj", rel.get("object", ""))).strip()
            pred = str(rel.get("predicate", "")).strip()
            if not (sub and obj and pred):
                continue
            sub_is_target = self._name_matches_alias_set(sub, target_aliases)
            obj_is_target = self._name_matches_alias_set(obj, target_aliases)
            if not (sub_is_target or obj_is_target):
                continue
            other = obj if sub_is_target else sub
            if generated_names and (not _is_generated_name(other)):
                continue
            relations_hit.append(
                {
                    "subject": sub,
                    "predicate": pred,
                    "obj": obj,
                    "phase": str(rel.get("phase", "global")).strip() or "global",
                    "distance_hint": str(rel.get("distance_hint", "unknown")).strip() or "unknown",
                    "confidence": float(rel.get("confidence", 0.5)),
                }
            )
            if len(relations_hit) >= int(max_items):
                break

        events_hit: list[dict[str, Any]] = []
        for ev in list(graph.get("events", []) or []):
            if not isinstance(ev, dict):
                continue
            sub = str(ev.get("subject", "")).strip()
            obj = str(ev.get("obj", ev.get("object", ""))).strip()
            tp = str(ev.get("type", "")).strip()
            if not tp:
                continue
            sub_is_target = self._name_matches_alias_set(sub, target_aliases) if sub else False
            obj_is_target = self._name_matches_alias_set(obj, target_aliases) if obj else False
            if not (sub_is_target or obj_is_target):
                continue
            other = obj if sub_is_target else sub
            if generated_names and other and (not _is_generated_name(other)):
                continue
            events_hit.append(
                {
                    "type": tp,
                    "subject": sub,
                    "obj": obj,
                    "frame_token": str(ev.get("frame_token", ev.get("frame", ""))).strip(),
                    "phase": str(ev.get("phase", "global")).strip() or "global",
                    "confidence": float(ev.get("confidence", 0.5)),
                }
            )
            if len(events_hit) >= int(max_items):
                break

        constraints_hit: list[dict[str, Any]] = []
        for cs in list(graph.get("constraints", []) or []):
            if isinstance(cs, str):
                rule = str(cs).strip()
                if not rule:
                    continue
                if not any(self._text_contains_alias(rule, a) for a in target_aliases):
                    continue
                constraints_hit.append({"rule": rule, "target_pair": [], "phase": "global"})
                if len(constraints_hit) >= int(max_items):
                    break
                continue
            if not isinstance(cs, dict):
                continue
            rule = str(cs.get("rule", "")).strip()
            pair = list(cs.get("target_pair", [])) if isinstance(cs.get("target_pair", []), list) else []
            pair_has_target = any(self._name_matches_alias_set(str(x), target_aliases) for x in pair)
            rule_has_target = any(self._text_contains_alias(rule, a) for a in target_aliases)
            if not (pair_has_target or rule_has_target):
                continue
            if generated_names and pair:
                other_in_generated = any((not self._name_matches_alias_set(str(x), target_aliases)) and _is_generated_name(str(x)) for x in pair)
                if not other_in_generated and (not pair_has_target):
                    continue
            constraints_hit.append(
                {
                    "rule": rule,
                    "target_pair": [str(x).strip() for x in pair if str(x).strip()][:4],
                    "phase": str(cs.get("phase", "global")).strip() or "global",
                }
            )
            if len(constraints_hit) >= int(max_items):
                break

        depth_timeline_hit: list[dict[str, Any]] = []
        for dt in list(graph.get("x_depth_order_timeline", graph.get("depth_order_timeline", [])) or []):
            if not isinstance(dt, dict):
                continue
            order = [str(x).strip() for x in list(dt.get("order_positive_to_negative", dt.get("x_axis_positive_to_negative", [])) or []) if str(x).strip()]
            if not order:
                continue
            has_target = any(self._name_matches_alias_set(x, target_aliases) for x in order)
            if not has_target:
                continue
            if generated_names:
                has_generated_other = any((not self._name_matches_alias_set(x, target_aliases)) and _is_generated_name(x) for x in order)
                if not has_generated_other:
                    continue
            depth_timeline_hit.append(
                {
                    "frame_range": str(dt.get("frame_range", dt.get("range", ""))).strip(),
                    "start_frame_index": dt.get("start_frame_index", None),
                    "end_frame_index": dt.get("end_frame_index", None),
                    "order_positive_to_negative": order[:12],
                    "confidence": dt.get("confidence", None),
                    "notes": str(dt.get("notes", "")).strip(),
                }
            )
            if len(depth_timeline_hit) >= int(max_items):
                break

        relative_timeline_hit: list[dict[str, Any]] = []
        for dt in list(
            graph.get(
                "relative_position_timeline",
                graph.get("relative_spatial_timeline", graph.get("spatial_relation_timeline", [])),
            )
            or []
        ):
            if not isinstance(dt, dict):
                continue
            pairs = list(dt.get("pairs", dt.get("relations", dt.get("relative_pairs", []))) or [])
            if not isinstance(pairs, list):
                continue
            selected_pairs: list[dict[str, Any]] = []
            for p in pairs:
                if not isinstance(p, dict):
                    continue
                sub = str(p.get("subject", p.get("src", p.get("a", "")))).strip()
                obj = str(p.get("obj", p.get("object", p.get("dst", p.get("b", ""))))).strip()
                if not (sub and obj):
                    continue
                sub_is_target = self._name_matches_alias_set(sub, target_aliases)
                obj_is_target = self._name_matches_alias_set(obj, target_aliases)
                if not (sub_is_target or obj_is_target):
                    continue
                other = obj if sub_is_target else sub
                if generated_names and (not _is_generated_name(other)):
                    continue
                try:
                    p_conf = float(p.get("confidence", 0.5))
                except Exception:
                    p_conf = 0.5
                selected_pairs.append(
                    {
                        "subject": sub,
                        "obj": obj,
                        "horizontal": str(
                            p.get("horizontal", p.get("left_right", p.get("lr", p.get("x_relation", "unknown"))))
                        ).strip()
                        or "unknown",
                        "vertical": str(
                            p.get("vertical", p.get("up_down", p.get("ud", p.get("y_relation", "unknown"))))
                        ).strip()
                        or "unknown",
                        "depth": str(
                            p.get("depth", p.get("front_back", p.get("fb", p.get("z_relation", "unknown"))))
                        ).strip()
                        or "unknown",
                        "confidence": float(max(0.0, min(1.0, p_conf))),
                        "notes": str(p.get("notes", p.get("evidence", ""))).strip(),
                    }
                )
                if len(selected_pairs) >= 24:
                    break
            if not selected_pairs:
                continue
            relative_timeline_hit.append(
                {
                    "frame_range": str(dt.get("frame_range", dt.get("range", ""))).strip(),
                    "start_frame_index": dt.get("start_frame_index", None),
                    "end_frame_index": dt.get("end_frame_index", None),
                    "pairs": selected_pairs,
                    "confidence": dt.get("confidence", None),
                    "notes": str(dt.get("notes", "")).strip(),
                }
            )
            if len(relative_timeline_hit) >= int(max_items):
                break

        return {
            "target_object": str(target),
            "generated_objects_for_relation_check": [str(x) for x in generated_names[:20]],
            "target_relations_from_video": relations_hit,
            "target_events_from_video": events_hit,
            "target_constraints_from_video": constraints_hit,
            "target_x_depth_timeline_from_video": depth_timeline_hit,
            "target_relative_position_timeline_from_video": relative_timeline_hit,
        }

    @staticmethod
    def _debug_sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in list(messages or []):
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", ""))
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for it in content:
                    if not isinstance(it, dict):
                        parts.append(it)
                        continue
                    kind = str(it.get("type", ""))
                    if kind == "image_url":
                        image_url = it.get("image_url", {})
                        if isinstance(image_url, dict):
                            raw_url = str(image_url.get("url", ""))
                            if raw_url.startswith("data:image/"):
                                image_url = dict(image_url)
                                image_url["url"] = f"<data_url:{len(raw_url)} chars>"
                                parts.append({"type": "image_url", "image_url": image_url})
                                continue
                    parts.append(it)
                out.append({"role": role, "content": parts})
                continue
            out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _debug_print_json(prefix: str, payload: Any) -> None:
        try:
            print(prefix + json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            print(prefix + str(payload))

    @staticmethod
    def _normalize_x_direction(raw: Any) -> str:
        s = str(raw or "").strip().lower()
        dir_map = {
            "positive": "positive",
            "+x": "positive",
            "forward": "positive",
            "front": "positive",
            "negative": "negative",
            "-x": "negative",
            "backward": "negative",
            "back": "negative",
            "none": "none",
            "stay": "none",
        }
        return dir_map.get(s, "none")

    @staticmethod
    def _extract_int(d: Any, keys: list[str]) -> int | None:
        if not isinstance(d, dict):
            return None
        for k in keys:
            if k in d:
                try:
                    return int(d.get(k))
                except Exception:
                    continue
        return None

    def _fetch_local_model_info(self, base_url: str | None, timeout: float = 1.2) -> dict[str, Any] | None:
        if not base_url:
            return None
        url = str(base_url).rstrip("/") + "/model_info"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _estimate_resize_by_pixels(
        *,
        width: int,
        height: int,
        min_pixels: int | None,
        max_pixels: int | None,
        factor: int = 28,
    ) -> dict[str, Any] | None:
        w = max(1, int(width))
        h = max(1, int(height))
        if min_pixels is None and max_pixels is None:
            return None
        pixels = int(w * h)
        scale = 1.0
        if max_pixels is not None and pixels > int(max_pixels):
            scale = min(scale, math.sqrt(float(max_pixels) / float(max(1, pixels))))
        if min_pixels is not None and pixels < int(min_pixels):
            scale = max(scale, math.sqrt(float(min_pixels) / float(max(1, pixels))))

        def _round_to_factor(x: float, fac: int) -> int:
            if fac <= 1:
                return max(1, int(round(x)))
            return max(fac, int(round(x / float(fac))) * fac)

        ew = _round_to_factor(float(w) * float(scale), int(factor))
        eh = _round_to_factor(float(h) * float(scale), int(factor))
        return {
            "input_size": [int(w), int(h)],
            "estimated_size": [int(ew), int(eh)],
            "estimated_pixels": int(ew * eh),
            "scale": float(scale),
            "factor": int(factor),
            "min_pixels": int(min_pixels) if min_pixels is not None else None,
            "max_pixels": int(max_pixels) if max_pixels is not None else None,
        }

    @staticmethod
    def _normalize_relative_points(
        raw: Any,
        *,
        width: int,
        height: int,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        if int(limit) <= 0:
            return []
        items: list[Any]
        if isinstance(raw, list):
            items = list(raw)
        elif isinstance(raw, dict):
            cand = raw.get("points", [])
            items = list(cand) if isinstance(cand, list) else []
        else:
            items = []

        out: list[dict[str, Any]] = []
        seen: set[tuple[float, float]] = set()
        w = max(1, int(width))
        h = max(1, int(height))
        for it in items:
            x = None
            y = None
            label = ""
            object_name = ""
            part_name = ""
            idx: int | None = None
            if isinstance(it, dict):
                x = it.get("x", it.get("u", it.get("col")))
                y = it.get("y", it.get("v", it.get("row")))
                label = str(it.get("reason", it.get("label", ""))).strip()
                object_name = str(
                    it.get(
                        "object_name",
                        it.get("object", it.get("name", it.get("occluder_name", it.get("occluder", "")))),
                    )
                ).strip()
                part_name = str(
                    it.get(
                        "missing_part_name",
                        it.get("part_name", it.get("target_part", it.get("part", ""))),
                    )
                ).strip()
                raw_idx = it.get("index", it.get("id", None))
                if raw_idx is not None:
                    try:
                        idx = int(raw_idx)
                    except Exception:
                        idx = None
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                x = it[0]
                y = it[1]
                if len(it) >= 3:
                    label = str(it[2]).strip()
            if x is None or y is None:
                continue
            try:
                xf = float(x)
                yf = float(y)
            except Exception:
                continue
            if not (0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0):
                xf = xf / float(max(1, w - 1))
                yf = yf / float(max(1, h - 1))
            xf = max(0.0, min(1.0, xf))
            yf = max(0.0, min(1.0, yf))
            kx = round(xf, 5)
            ky = round(yf, 5)
            key = (kx, ky)
            if key in seen:
                continue
            seen.add(key)
            item = {"x": float(kx), "y": float(ky)}
            if idx is not None:
                item["index"] = int(idx)
            if object_name:
                item["object_name"] = object_name
            if part_name:
                item["missing_part_name"] = part_name
            if label:
                item["reason"] = label
            out.append(item)
            if len(out) >= int(limit):
                break
        return out

    @staticmethod
    def _normalize_prompt_points_for_eval(
        points: list[Any] | None,
        *,
        width: int,
        height: int,
        limit: int = 32,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not isinstance(points, list):
            return out
        w = max(1, int(width))
        h = max(1, int(height))
        for i, p in enumerate(points):
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x = float(p[0])
                y = float(p[1])
            except Exception:
                continue
            xn = max(0.0, min(1.0, x / float(max(1, w - 1))))
            yn = max(0.0, min(1.0, y / float(max(1, h - 1))))
            out.append({"index": int(i), "x": float(round(xn, 5)), "y": float(round(yn, 5))})
            if len(out) >= int(limit):
                break
        return out

    @staticmethod
    def _build_mask_overlay_rgb(image_rgb_u8: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        img = np.asarray(image_rgb_u8, dtype=np.uint8)
        mask = (np.asarray(mask_bool) > 0).astype(np.uint8)
        h = min(int(img.shape[0]), int(mask.shape[0]))
        w = min(int(img.shape[1]), int(mask.shape[1]))
        if h <= 0 or w <= 0:
            return img[:, :, :3].astype(np.uint8, copy=False)
        img = img[:h, :w, :3].astype(np.uint8, copy=False)
        mask = mask[:h, :w]
        mask_bool2 = mask.astype(bool)
        out = img.astype(np.float32)

        # Make outside region slightly dim so masked region is visually dominant.
        out = out * 0.72

        # Use high-saturation magenta for inside-mask area.
        mask_color = np.array([255.0, 0.0, 255.0], dtype=np.float32)
        m3 = mask_bool2[..., None]
        out = np.where(m3, out * 0.20 + mask_color[None, None, :] * 0.80, out)

        # Highlight mask boundary with bright yellow for easier boundary reading.
        edge = np.zeros_like(mask_bool2, dtype=bool)
        edge[1:, :] |= (mask_bool2[1:, :] != mask_bool2[:-1, :])
        edge[:-1, :] |= (mask_bool2[:-1, :] != mask_bool2[1:, :])
        edge[:, 1:] |= (mask_bool2[:, 1:] != mask_bool2[:, :-1])
        edge[:, :-1] |= (mask_bool2[:, :-1] != mask_bool2[:, 1:])
        edge &= mask_bool2
        edge_color = np.array([255.0, 255.0, 0.0], dtype=np.float32)
        out[edge] = out[edge] * 0.10 + edge_color[None, :] * 0.90
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _build_mask_binary_rgb(mask_bool: np.ndarray) -> np.ndarray:
        m = (np.asarray(mask_bool) > 0).astype(np.uint8) * 255
        return np.stack([m, m, m], axis=-1)

    @staticmethod
    def _build_mask_foreground_cutout_rgb(image_rgb_u8: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        img = np.asarray(image_rgb_u8, dtype=np.uint8)
        mask = (np.asarray(mask_bool) > 0)
        h = min(int(img.shape[0]), int(mask.shape[0]))
        w = min(int(img.shape[1]), int(mask.shape[1]))
        if h <= 0 or w <= 0:
            return img[:, :, :3].astype(np.uint8, copy=False)
        img = img[:h, :w, :3].astype(np.uint8, copy=False)
        mask = mask[:h, :w]
        out = img.copy()
        out[~mask] = 255
        return out

    @staticmethod
    def _build_mask_background_cutout_rgb(image_rgb_u8: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        img = np.asarray(image_rgb_u8, dtype=np.uint8)
        mask = (np.asarray(mask_bool) > 0)
        h = min(int(img.shape[0]), int(mask.shape[0]))
        w = min(int(img.shape[1]), int(mask.shape[1]))
        if h <= 0 or w <= 0:
            return img[:, :, :3].astype(np.uint8, copy=False)
        img = img[:h, :w, :3].astype(np.uint8, copy=False)
        mask = mask[:h, :w]
        out = img.copy()
        out[mask] = 255
        return out

    def evaluate_mask_with_vlm(
        self,
        *,
        object_name: str,
        image_rgb_u8: np.ndarray,
        mask_bool: np.ndarray,
        positive_points: list[Any] | None = None,
        negative_points: list[Any] | None = None,
        non_target_objects: list[str] | None = None,
        backend_mode: str | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        if image_rgb_u8 is None or mask_bool is None:
            return {"ok": False, "evaluated": False, "reason": "missing_image_or_mask"}
        try:
            ih = int(image_rgb_u8.shape[0])
            iw = int(image_rgb_u8.shape[1])
            mh = int(mask_bool.shape[0])
            mw = int(mask_bool.shape[1])
        except Exception:
            return {"ok": False, "evaluated": False, "reason": "invalid_image_or_mask_shape"}

        h = min(ih, mh)
        w = min(iw, mw)
        if h <= 0 or w <= 0:
            return {"ok": False, "evaluated": False, "reason": "invalid_clipped_shape"}
        img = np.asarray(image_rgb_u8)[:h, :w, :3]
        mask = np.asarray(mask_bool)[:h, :w]
        overlay = self._build_mask_overlay_rgb(img, mask)
        mask_rgb = self._build_mask_binary_rgb(mask)
        fg_cutout = self._build_mask_foreground_cutout_rgb(img, mask)
        bg_cutout = self._build_mask_background_cutout_rgb(img, mask)
        positive_points_norm = self._normalize_prompt_points_for_eval(
            positive_points, width=w, height=h, limit=64
        )
        negative_points_norm = self._normalize_prompt_points_for_eval(
            negative_points, width=w, height=h, limit=64
        )
        non_targets = [
            str(x).strip()
            for x in list(non_target_objects or [])
            if str(x).strip() and str(x).strip().lower() != str(object_name).strip().lower()
        ]
        target_name_norm_en = self.to_english_object_name(str(object_name))
        non_targets_norm_en = self.to_english_object_names(non_targets)

        image_url = self._to_data_url(img, jpeg_quality=85)
        overlay_url = self._to_data_url(overlay, jpeg_quality=85)
        mask_url = self._to_data_url(mask_rgb, jpeg_quality=85)
        fg_cutout_url = self._to_data_url(fg_cutout, jpeg_quality=85)
        bg_cutout_url = self._to_data_url(bg_cutout, jpeg_quality=85)
        if not image_url or not overlay_url or not mask_url or not fg_cutout_url or not bg_cutout_url:
            return {"ok": False, "evaluated": False, "reason": "mask_eval_image_encode_failed"}

        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"
        model_for_lang = self._openai_model() if mode == "official_api" else self._local_vlm_model()
        language, language_source = self._resolve_agent1_mask_eval_language(model_for_lang)
        target_name_lang = self._to_language_object_name(str(object_name), language)
        non_targets_lang = self._to_language_object_names(non_targets, language)

        if language == "zh":
            prompt = (
                "你是一个严格的分割质量评测器。\n"
                "正确的 mask 定义：完整包含目标物体，并排除所有非目标物体。\n"
                "非目标物体位于 mask 外是正确行为，不是错误。\n"
                "最终判定规则（唯一标准）：当且仅当【目标的可见部分完整在mask内】且【所有非目标都在mask外】且 uncertain=false 时 accurate=true；否则 accurate=false。\n"
                f"目标物体名称（中文）：{target_name_lang}。\n"
                "你将按顺序收到5张图：\n"
                "1）原图，\n"
                "2）原图+mask叠加图（高饱和亮洋红色区域=mask内，黄色细边=mask边界），\n"
                "3）二值mask图（白色=mask），\n"
                "4）前景抠图：仅保留mask内区域（mask外为白色），\n"
                "5）背景抠图：仅保留mask外区域（mask内为白色）。\n"
                "颜色解释（非常重要）：图(2)中亮洋红色区域表示 mask 内；非亮洋红色区域表示 mask 外；黄色细边仅用于显示边界。\n"
                "请判断该mask是否准确覆盖目标并排除非目标。\n"
                "请使用图(4)识别mask内对象，使用图(5)识别mask外对象。\n"
                "必做流程：\n"
                "Step A: 识别图(4)中的mask内对象。\n"
                "Step B: 识别图(5)中目标是否有漏分部分。\n"
                "Step C: 先区分“真实漏分”与“被其他物体遮挡导致不可见”，再决定 extra_objects / missing_parts / occluded_target_parts。\n"
                f"非目标物体列表（中文，必须排除）：{json.dumps(non_targets_lang, ensure_ascii=False)}\n"
                f"当前正点（归一化）：{json.dumps(positive_points_norm, ensure_ascii=False)}\n"
                f"当前负点（归一化）：{json.dumps(negative_points_norm, ensure_ascii=False)}\n"
                "仅返回JSON：\n"
                "{\n"
                '  "accurate": boolean,\n'
                '  "inside_mask_objects": [string],\n'
                '  "outside_mask_objects": [string],\n'
                '  "target_fully_contained": boolean,\n'
                '  "extra_objects": [{"object_name": string, "x": float_0_to_1, "y": float_0_to_1, "reason": "被错误包含在mask内的非目标"}],\n'
                '  "missing_parts": [{"object_name": string, "missing_part_name": string, "x": float_0_to_1, "y": float_0_to_1, "reason": "目标缺失的具体部位"}],\n'
                '  "occluded_target_parts": [{"occluder_name": string, "missing_part_name": string, "x": float_0_to_1, "y": float_0_to_1, "reason": "因遮挡不可见的目标部位"}],\n'
                '  "wrong_positive_points": [{"index": int, "x": float_0_to_1, "y": float_0_to_1, "reason": "该正点落在非目标"}],\n'
                '  "wrong_negative_points": [{"index": int, "x": float_0_to_1, "y": float_0_to_1, "reason": "该负点落在目标"}],\n'
                '  "uncertain": boolean,\n'
                '  "uncertainty_reason": "简短说明",\n'
                '  "violates_instance_policy": boolean,\n'
                '  "consistency_note": "简短说明",\n'
                '  "summary": "结果：<通过/失败>。原因：<目标状态>；<非目标状态>。"\n'
                "}\n"
                "规则：\n"
                "1）坐标x,y必须在[0,1]。\n"
                "2）若mask内有非目标，accurate=false，且extra_objects至少给一个坐标。\n"
                "3）若目标有漏分，accurate=false，且missing_parts至少给一个坐标。\n"
                "4）可同时存在extra与missing。\n"
                "5）必须分析所有给定点prompt的对错。\n"
                "6）严格实例边界：目标与非目标互斥，不能因上下文/功能关系合并。\n"
                "7）禁止把非目标写成目标missing_parts。\n"
                "8）若推理依赖上下文而非实例边界，violates_instance_policy=true。\n"
                "9）summary保持简洁。\n"
                "10）非目标出现在图(4)时，写入extra_objects（不要写missing_parts）。\n"
                "11）目标在图(5)出现且属于真实漏分时，写入missing_parts；若该部分是被其他物体遮挡导致不可见，则写入occluded_target_parts。\n"
                "12）自由文本字段（reason/consistency_note/summary）请使用中文。\n"
                "13）missing_parts只能描述目标。\n"
                "14）非目标在mask外是正确行为，不得作为missing_parts上报。\n"
                "15）若目标完整包含，则missing_parts必须为空。\n"
                "16）当苹果是非目标时，禁止报告“苹果缺失于mask”。\n"
                "17）若target_fully_contained=true，则missing_parts必须为空。\n"
                "18）若missing_parts非空，则target_fully_contained必须为false。\n"
                "19）若uncertain=true，则accurate=false，且给出uncertainty_reason。\n"
                "20）禁止将“非目标被排除在mask外”描述为错误；例如“非目标苹果被错误排除在mask外”是错误表述。\n"
                "21）若非目标在mask外，这只能作为正确证据，不能作为 fail 原因。\n"
                "22）若目标未完整包含（target_fully_contained=false 或 missing_parts 非空），missing_parts 每一项必须给出具体缺失部位名 missing_part_name（如“刀尖”“苹果左上边缘”），禁止使用“目标缺失部分”“漏分”等泛化描述。\n"
                "23）若目标某部位只是被其他物体遮挡导致不可见，不得计入 missing_parts，不应判错；请写入 occluded_target_parts。\n"
                "24）occluded_target_parts 仅用于记录遮挡，不作为 fail 依据。\n"
                "25）计算 accurate 时，只将真实漏分计入 missing_parts；occluded_target_parts 不能导致 fail。\n"
                "26）关键规则：若图(5)（非mask区域）中未发现任何目标部位，则 missing_parts 必须为空；"
                "此时若你仍感到目标看起来不完整，应归因为遮挡并写入 occluded_target_parts，而不是判失败。\n"
                "一致性契约（必须遵守）：\n"
                "1）accurate必须按规则计算，不可猜测：\n"
                "   accurate = (target_fully_contained == true)\n"
                "              AND (len(extra_objects) == 0)\n"
                "              AND (len(missing_parts) == 0)\n"
                "              AND (uncertain == false)。\n"
                "2）summary必须与accurate一致：\n"
                "   - accurate=true 时，summary必须以“结果：通过”开头。\n"
                "   - accurate=false 时，summary必须以“结果：失败”开头。\n"
                "3）若字段冲突，必须设置：uncertain=true，uncertainty_reason=\"field inconsistency\"，accurate=false。\n"
                "4）missing_parts只能描述目标，不能写非目标。\n"
                "5）非目标在mask外是正确行为，不是错误。\n"
                "6）通过态硬规则：若 target_fully_contained=true 且 extra_objects为空 且 missing_parts为空 且 uncertain=false，"
                "则 accurate 必须为 true，且 summary 必须以“结果：通过”开头。\n"
                "7）在满足通过态硬规则时，严禁输出 accurate=false 或“结果：失败”。\n"
                "8）最终输出前请执行一次自检：按一致性契约重算 accurate，并让 summary 与 accurate 严格一致。\n"
                "9）严禁把“非目标被排除在mask外”当作错误理由写入 summary 或 consistency_note。\n"
            )
            system_content = "仅返回合法JSON。严格遵循schema与一致性契约。"
        else:
            prompt = (
                "You are a strict segmentation quality evaluator.\n"
                "A correct mask contains the full target object and excludes all non-target objects.\n"
                "A non-target being outside the mask is correct, not an error.\n"
                "Final decision rule (the only pass criterion): accurate=true iff the full VISIBLE part of target is inside mask AND every non-target is outside mask AND uncertain=false; otherwise accurate=false.\n"
                f"Target object name (English): {target_name_lang}.\n"
                "You receive five images in order:\n"
                "1) original image,\n"
                "2) original image with mask overlay (high-saturation bright magenta = inside mask, thin yellow contour = mask boundary),\n"
                "3) binary mask image (white=mask),\n"
                "4) foreground cutout: original image masked by current mask (outside-mask region is white),\n"
                "5) background cutout: original image masked by inverse mask (inside-mask region is white).\n"
                "Color legend (critical): in image (2), bright magenta means INSIDE mask; non-magenta means OUTSIDE mask; yellow contour is boundary only.\n"
                "Judge whether the mask accurately contains the full target object and excludes non-target objects.\n"
                "Use image (4) to identify what is INSIDE the mask, and image (5) to identify what is OUTSIDE the mask.\n"
                "Required workflow:\n"
                "Step A: identify what is inside mask (image 4).\n"
                "Step B: identify target parts outside mask (image 5).\n"
                "Step C: first separate true under-segmentation vs occlusion-caused invisibility, then decide extra_objects / missing_parts / occluded_target_parts.\n"
                f"Non-target object names (English, must be excluded from target mask): {json.dumps(non_targets_lang, ensure_ascii=False)}\n"
                f"Current positive point prompts (normalized): {json.dumps(positive_points_norm, ensure_ascii=False)}\n"
                f"Current negative point prompts (normalized): {json.dumps(negative_points_norm, ensure_ascii=False)}\n"
                "Return JSON only:\n"
                "{\n"
                '  "accurate": boolean,\n'
                '  "inside_mask_objects": [string],\n'
                '  "outside_mask_objects": [string],\n'
                '  "target_fully_contained": boolean,\n'
                '  "extra_objects": [{"object_name": string, "x": float_0_to_1, "y": float_0_to_1, "reason": "which non-target is wrongly inside mask"}],\n'
                '  "missing_parts": [{"object_name": string, "missing_part_name": string, "x": float_0_to_1, "y": float_0_to_1, "reason": "which target part is missing"}],\n'
                '  "occluded_target_parts": [{"occluder_name": string, "missing_part_name": string, "x": float_0_to_1, "y": float_0_to_1, "reason": "target part is occluded by another object"}],\n'
                '  "wrong_positive_points": [{"index": int, "x": float_0_to_1, "y": float_0_to_1, "reason": "this positive point falls on non-target"}],\n'
                '  "wrong_negative_points": [{"index": int, "x": float_0_to_1, "y": float_0_to_1, "reason": "this negative point falls on target"}],\n'
                '  "uncertain": boolean,\n'
                '  "uncertainty_reason": "short text",\n'
                '  "violates_instance_policy": boolean,\n'
                '  "consistency_note": "short text",\n'
                '  "summary": "Result: <pass/fail>. Because: <target status>; <non-target status>."\n'
                "}\n"
                "Rules:\n"
                "1) x,y must be normalized coordinates in [0,1].\n"
                "2) If there are extra non-target regions in mask, set accurate=false and put at least one coordinate in extra_objects.\n"
                "3) If target object has missing parts, set accurate=false and put at least one coordinate in missing_parts.\n"
                "4) If both issues exist, include both lists.\n"
                "5) Analyze every given point prompt: identify wrong positive/negative points when applicable.\n"
                "6) STRICT INSTANCE POLICY: target object and non-target objects are mutually exclusive instances.\n"
                "7) Never claim a non-target object is a missing part of the target because of function/context.\n"
                "8) If your reasoning depends on context/function instead of instance boundaries, set violates_instance_policy=true.\n"
                "9) Keep summary concise.\n"
                "10) If a non-target appears in image (4), report it in extra_objects (not missing_parts).\n"
                "11) If target appears in image (5), report it in missing_parts only for true under-segmentation; if it is absent only due to occlusion by another object, report it in occluded_target_parts.\n"
                "12) Use ENGLISH ONLY in all free-text fields: reason, consistency_note, summary.\n"
                "13) missing_parts must refer to TARGET object only.\n"
                "14) A non-target outside mask is CORRECT and must NOT be reported as missing_parts.\n"
                "15) If target is fully contained, missing_parts must be empty.\n"
                "16) Do NOT report 'apple is missing from mask' when apple is a non-target.\n"
                "17) If target_fully_contained=true, then missing_parts must be empty.\n"
                "18) If missing_parts is non-empty, then target_fully_contained must be false.\n"
                "19) If uncertain=true, set accurate=false and provide uncertainty_reason.\n"
                "20) Do NOT describe 'non-target excluded from mask' as an error.\n"
                "21) If a non-target is outside mask, it is positive evidence, not a fail reason.\n"
                "22) If target is not fully contained (target_fully_contained=false or missing_parts non-empty), every missing_parts entry MUST provide a specific missing_part_name (e.g., tip, handle, left edge), not a generic phrase like 'missing part'.\n"
                "23) If a target part is absent only because it is occluded by another object, do NOT put it in missing_parts and do NOT fail for that reason; report it in occluded_target_parts.\n"
                "24) occluded_target_parts is for recording occlusion only and must not be treated as a fail reason.\n"
                "25) When computing accurate, count only true under-segmentation in missing_parts; occluded_target_parts must not cause fail.\n"
                "26) Critical rule: if image (5) (outside-mask region) contains NO visible target part, missing_parts must be empty."
                " If target still looks incomplete, treat it as occlusion and report in occluded_target_parts instead of failing.\n"
                "Consistency contract (MUST follow):\n"
                "1) Compute accurate by rule, do not guess:\n"
                "   accurate = (target_fully_contained == true)\n"
                "              AND (len(extra_objects) == 0)\n"
                "              AND (len(missing_parts) == 0)\n"
                "              AND (uncertain == false).\n"
                "2) summary MUST match accurate:\n"
                '   - if accurate == true, summary must start with "Result: pass."\n'
                '   - if accurate == false, summary must start with "Result: fail."\n'
                "3) Contradiction handling:\n"
                "   If any conflict is detected among fields, set:\n"
                "   uncertain = true,\n"
                '   uncertainty_reason = "field inconsistency",\n'
                "   accurate = false.\n"
                "4) missing_parts can only describe target object.\n"
                "   Never put any non-target object in missing_parts.\n"
                "5) A non-target outside mask is correct behavior, not an error.\n"
                "6) PASS hard rule: if target_fully_contained=true AND extra_objects is empty AND missing_parts is empty "
                "AND uncertain=false, then accurate MUST be true and summary MUST start with \"Result: pass.\".\n"
                "7) Under PASS hard rule condition, NEVER output accurate=false or \"Result: fail.\".\n"
                "8) Final self-check before output: recompute accurate from the consistency contract, then make summary strictly match accurate.\n"
                "9) Never use 'non-target excluded from mask' as an error phrase in summary or consistency_note.\n"
            )
            system_content = "Return valid JSON only. Follow schema and consistency contract strictly."
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": overlay_url, "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": mask_url, "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": fg_cutout_url, "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": bg_cutout_url, "detail": "high"}},
                ],
            },
        ]

        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
            payload["model"] = model
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = messages
            safe_payload = dict(payload)
            safe_payload["messages"] = self._debug_sanitize_messages(messages)
            self._debug_print_json(
                "[VLM Assistor][Agent1][MaskEval] request=",
                {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "has_api_key": bool(api_key),
                    "endpoint": str(base_url).rstrip("/") + "/chat/completions",
                    "payload": safe_payload,
                    "language": language,
                    "language_source": language_source,
                    "image_shape": [int(h), int(w)],
                },
            )
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            local_messages = self._convert_system_to_user_prefix(messages)
            payload["messages"] = local_messages
            if not base_url:
                return {"ok": False, "evaluated": False, "reason": "local_openai_base_url_missing"}
            safe_payload = dict(payload)
            safe_payload["messages"] = self._debug_sanitize_messages(local_messages)
            self._debug_print_json(
                "[VLM Assistor][Agent1][MaskEval] request=",
                {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "has_api_key": bool(api_key),
                    "endpoint": str(base_url).rstrip("/") + "/chat/completions",
                    "payload": safe_payload,
                    "language": language,
                    "language_source": language_source,
                    "image_shape": [int(h), int(w)],
                },
            )
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )

        if not isinstance(resp, dict):
            return {
                "ok": False,
                "evaluated": False,
                "reason": f"{provider}_http_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
        except Exception as exc:
            return {
                "ok": False,
                "evaluated": False,
                "reason": f"invalid_response:{type(exc).__name__}:{exc}",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }
        print("[VLM Assistor][Agent1][MaskEval] raw_response_text:", str(text)[:1200])

        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            return {
                "ok": False,
                "evaluated": False,
                "reason": "mask_eval_json_parse_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "raw_text": str(text)[:400],
            }

        model_accurate = bool(parsed.get("accurate", parsed.get("ok", parsed.get("is_accurate", False))))
        extra = self._normalize_relative_points(parsed.get("extra_objects", []), width=w, height=h, limit=12)
        missing = self._normalize_relative_points(parsed.get("missing_parts", []), width=w, height=h, limit=12)
        occluded_target_parts = self._normalize_relative_points(
            parsed.get("occluded_target_parts", parsed.get("occlusion_parts", [])),
            width=w,
            height=h,
            limit=12,
        )
        wrong_pos = self._normalize_relative_points(parsed.get("wrong_positive_points", []), width=w, height=h, limit=32)
        wrong_neg = self._normalize_relative_points(parsed.get("wrong_negative_points", []), width=w, height=h, limit=32)
        inside_mask_objects = self._normalize_object_name_list(
            parsed.get("inside_mask_objects", []), limit=32, language=language
        )
        outside_mask_objects = self._normalize_object_name_list(
            parsed.get("outside_mask_objects", []), limit=32, language=language
        )
        target_fully_contained = bool(parsed.get("target_fully_contained", False))
        uncertain = bool(parsed.get("uncertain", False))
        uncertainty_reason = str(parsed.get("uncertainty_reason", "")).strip()
        summary = str(parsed.get("summary", parsed.get("reason", ""))).strip()
        consistency_note = str(parsed.get("consistency_note", "")).strip()
        violates_instance_policy = bool(parsed.get("violates_instance_policy", False))

        # Ensure coordinate reasons bind to an object name for downstream analysis.
        for it in extra:
            obj = self._to_language_object_name(str(it.get("object_name", "")).strip(), language)
            if not obj:
                obj = "未知非目标" if language == "zh" else "unknown_non_target"
            it["object_name"] = obj
            it["object_name_normalized_en"] = self.to_english_object_name(str(obj))
        for it in missing:
            obj = self._to_language_object_name(str(it.get("object_name", "")).strip(), language)
            if not obj:
                obj = target_name_lang
            it["object_name"] = obj
            it["object_name_normalized_en"] = self.to_english_object_name(str(obj))
            part_name = str(
                it.get(
                    "missing_part_name",
                    it.get("part_name", it.get("target_part", it.get("part", ""))),
                )
            ).strip()
            if part_name:
                it["missing_part_name"] = part_name
        for it in occluded_target_parts:
            occ = self._to_language_object_name(str(it.get("object_name", "")).strip(), language)
            if occ:
                it["occluder_name"] = occ
            elif str(it.get("occluder_name", "")).strip():
                it["occluder_name"] = str(it.get("occluder_name", "")).strip()
            part_name = str(
                it.get(
                    "missing_part_name",
                    it.get("part_name", it.get("target_part", it.get("part", ""))),
                )
            ).strip()
            if part_name:
                it["missing_part_name"] = part_name

        # Hard guardrail: if evaluator says a known non-target is "part of target",
        # treat this VLM evaluation as inconsistent and ignore it (fallback to heuristic).
        non_target_hits = []
        non_target_in_missing: list[dict[str, Any]] = []
        reason_texts: list[str] = []
        for it in list(missing) + list(wrong_neg):
            if isinstance(it, dict):
                s = str(it.get("reason", "")).strip()
                if s:
                    reason_texts.append(s)
        non_target_aliases = self._expand_name_aliases(non_targets_norm_en)
        for it in list(missing):
            obj = str(it.get("object_name", "")).strip()
            obj_aliases = self._expand_name_aliases([obj]) if obj else []
            mention_non_target = any(a in non_target_aliases for a in obj_aliases)
            if (not mention_non_target) and str(it.get("reason", "")).strip():
                rl = str(it.get("reason", "")).lower()
                mention_non_target = any(
                    alias and self._text_contains_alias(rl, alias) for alias in non_target_aliases
                )
            if mention_non_target:
                non_target_in_missing.append(dict(it))
        missing_ignored_as_occlusion: list[dict[str, Any]] = []
        if missing:
            kept_missing: list[dict[str, Any]] = []
            for it in missing:
                txt = " ".join(
                    [
                        str(it.get("reason", "")).strip(),
                        str(it.get("missing_part_name", "")).strip(),
                    ]
                ).strip()
                occluded = self._is_occlusion_text(txt)
                if (not occluded) and occluded_target_parts:
                    for occ in occluded_target_parts:
                        if self._is_near_rel_point(it, occ, tol=0.08):
                            occluded = True
                            break
                if (not occluded) and wrong_pos and (not extra):
                    near_wrong_pos = any(self._is_near_rel_point(it, wp, tol=0.06) for wp in wrong_pos)
                    if near_wrong_pos:
                        occluded = True
                        derived = dict(it)
                        if not str(derived.get("occluder_name", "")).strip():
                            derived["occluder_name"] = "未知遮挡物" if language == "zh" else "unknown_occluder"
                        base_reason = str(derived.get("reason", "")).strip()
                        extra_reason = (
                            "由点诊断推断：该部位与非目标重叠，按遮挡豁免处理"
                            if language == "zh"
                            else "derived from point diagnostics: overlaps with non-target, treated as occlusion exemption"
                        )
                        derived["reason"] = f"{base_reason}; {extra_reason}" if base_reason else extra_reason
                        occluded_target_parts.append(derived)
                if occluded:
                    missing_ignored_as_occlusion.append(dict(it))
                else:
                    kept_missing.append(it)
            missing = kept_missing

        # Recompute non-target-in-missing on effective (non-occlusion) missing list.
        non_target_in_missing = []
        for it in list(missing):
            obj = str(it.get("object_name", "")).strip()
            obj_aliases = self._expand_name_aliases([obj]) if obj else []
            mention_non_target = any(a in non_target_aliases for a in obj_aliases)
            if (not mention_non_target) and str(it.get("reason", "")).strip():
                rl = str(it.get("reason", "")).lower()
                mention_non_target = any(
                    alias and self._text_contains_alias(rl, alias) for alias in non_target_aliases
                )
            if mention_non_target:
                non_target_in_missing.append(dict(it))
        missing_part_detail_issues: list[dict[str, Any]] = []
        if missing:
            for it in missing:
                if not self._missing_part_detail_is_specific(
                    it,
                    target_name_norm_en=target_name_norm_en,
                ):
                    missing_part_detail_issues.append(dict(it))
        low_non_targets = self._expand_name_aliases(non_targets_norm_en)
        policy_keywords = ["part of", "integral", "context", "属于", "一部分", "上下文", "功能", "operation"]
        for r in reason_texts:
            rl = r.lower()
            for name in low_non_targets:
                if name and self._text_contains_alias(rl, name) and any(k in rl for k in policy_keywords):
                    non_target_hits.append({"name": name, "reason": r})
        if non_target_hits:
            violates_instance_policy = True

        inside_non_target_objects: list[str] = []
        inside_target_objects: list[str] = []
        outside_target_objects: list[str] = []
        seen_inside_non_target: set[str] = set()
        seen_inside_target: set[str] = set()
        seen_outside_target: set[str] = set()
        target_aliases = set(self._expand_name_aliases([target_name_norm_en]))
        for obj_name in inside_mask_objects:
            obj_aliases = self._expand_name_aliases([str(obj_name)])
            if any(alias in target_aliases for alias in obj_aliases):
                k = str(self.to_english_object_name(str(obj_name))).strip().lower()
                if k and k not in seen_inside_target:
                    seen_inside_target.add(k)
                    inside_target_objects.append(str(obj_name))
            if any(alias in non_target_aliases for alias in obj_aliases):
                k = str(self.to_english_object_name(str(obj_name))).strip().lower()
                if k and k not in seen_inside_non_target:
                    seen_inside_non_target.add(k)
                    inside_non_target_objects.append(str(obj_name))
        for obj_name in outside_mask_objects:
            obj_aliases = self._expand_name_aliases([str(obj_name)])
            if any(alias in target_aliases for alias in obj_aliases):
                k = str(self.to_english_object_name(str(obj_name))).strip().lower()
                if k and k not in seen_outside_target:
                    seen_outside_target.add(k)
                    outside_target_objects.append(str(obj_name))

        # Hard fallback for prompt rule:
        # If outside-mask region does NOT contain target parts, any "missing_parts"
        # is reclassified as occlusion instead of true under-segmentation.
        missing_reclassified_no_outside_target = False
        if missing and (not outside_target_objects):
            missing_reclassified_no_outside_target = True
            for it in list(missing):
                moved = dict(it)
                if not str(moved.get("occluder_name", "")).strip():
                    moved["occluder_name"] = "未知遮挡物" if language == "zh" else "unknown_occluder"
                base_reason = str(moved.get("reason", "")).strip()
                extra_reason = (
                    "图(5)非mask区未发现目标部位，按遮挡豁免重分类"
                    if language == "zh"
                    else "reclassified as occlusion because no target part is visible in outside-mask region (image 5)"
                )
                moved["reason"] = f"{base_reason}; {extra_reason}" if base_reason else extra_reason
                occluded_target_parts.append(moved)
                missing_ignored_as_occlusion.append(dict(it))
            missing = []
            non_target_in_missing = []
            missing_part_detail_issues = []
            target_fully_contained = True

        target_fully_contained_overridden_by_occlusion = False
        if (not bool(target_fully_contained)) and (not missing) and (
            bool(missing_ignored_as_occlusion) or bool(occluded_target_parts)
        ):
            target_fully_contained = True
            target_fully_contained_overridden_by_occlusion = True

        reason_codes: list[str] = []
        if extra:
            reason_codes.append("extra_objects")
        if missing:
            reason_codes.append("missing_parts")
        if missing_ignored_as_occlusion:
            reason_codes.append("missing_parts_ignored_as_occlusion")
        if occluded_target_parts:
            reason_codes.append("occluded_target_parts_reported")
        if inside_non_target_objects:
            reason_codes.append("inside_mask_contains_non_target")
        if outside_target_objects:
            reason_codes.append("outside_mask_contains_target")
        if non_target_in_missing:
            reason_codes.append("missing_parts_contains_non_target")
        if missing_part_detail_issues:
            reason_codes.append("missing_parts_detail_not_specific")
        if target_fully_contained_overridden_by_occlusion:
            reason_codes.append("target_fully_contained_overridden_by_occlusion_policy")
        if missing_reclassified_no_outside_target:
            reason_codes.append("missing_parts_reclassified_no_target_in_outside_region")
        if wrong_pos:
            reason_codes.append("wrong_positive_points")
        if wrong_neg:
            reason_codes.append("wrong_negative_points")
        if violates_instance_policy:
            reason_codes.append("instance_policy_violated")
        if uncertain:
            reason_codes.append("evaluator_uncertain")
        consistency_violations: list[str] = []
        if target_fully_contained and missing:
            consistency_violations.append("target_fully_contained_but_missing_parts_non_empty")
            target_fully_contained = False
        if inside_non_target_objects and (not extra):
            consistency_violations.append("inside_mask_contains_non_target_but_extra_objects_empty")
        if outside_target_objects and (not missing):
            consistency_violations.append("outside_mask_contains_target_but_missing_parts_empty")
        if (not target_fully_contained) and (not missing):
            # keep as-is: this can be legitimate if target status is unclear.
            pass
        if missing_part_detail_issues:
            consistency_violations.append("missing_parts_detail_not_specific")
        if consistency_violations:
            reason_codes.append("schema_consistency_violation")

        # The evaluator sometimes returns target_fully_contained=false while its
        # own structured inventory says the target is inside, every non-target is
        # outside, and no target part is missing. Prefer that internally
        # consistent inventory over the contradictory boolean.
        target_containment_overridden_by_inventory = False
        known_non_targets_all_outside = True
        outside_aliases: set[str] = set()
        for obj_name in outside_mask_objects:
            outside_aliases.update(self._expand_name_aliases([str(obj_name)]))
        for non_target_name in non_targets_norm_en:
            aliases = set(self._expand_name_aliases([str(non_target_name)]))
            if aliases and not any(alias in outside_aliases for alias in aliases):
                known_non_targets_all_outside = False
                break
        inventory_proves_complete = bool(
            inside_target_objects
            and known_non_targets_all_outside
            and (not missing)
            and (not extra)
            and (not inside_non_target_objects)
            and (not outside_target_objects)
            and (not uncertain)
            and (not violates_instance_policy)
        )
        if (not target_fully_contained) and inventory_proves_complete:
            target_fully_contained = True
            target_containment_overridden_by_inventory = True
            reason_codes.append("target_fully_contained_overridden_by_inventory")

        # Final mask verdict is always computed with hard boolean rules.
        target_complete_in_mask = bool(target_fully_contained) and (len(missing) == 0) and (len(outside_target_objects) == 0)
        non_target_outside_mask = (len(extra) == 0) and (len(inside_non_target_objects) == 0)
        accurate = bool(target_complete_in_mask and non_target_outside_mask and (not uncertain) and (not violates_instance_policy))
        if bool(model_accurate) != bool(accurate):
            reason_codes.append("model_accurate_overridden_by_contract")
            summary = (
                "结果：通过。目标完整位于mask内，所有非目标均位于mask外。"
                if accurate and language == "zh"
                else "Result: pass. The target is complete in the mask and all non-targets are outside."
                if accurate
                else "结果：失败。最终一致性检查未通过。"
                if language == "zh"
                else "Result: fail. The final consistency check did not pass."
            )
        if not target_complete_in_mask:
            reason_codes.append("target_not_complete_in_mask")
        if not non_target_outside_mask:
            reason_codes.append("non_target_inside_mask")
        if (not accurate) and (not reason_codes):
            reason_codes.append("low_confidence_or_unclear")

        evaluated_ok = True
        eval_reason = ""
        if violates_instance_policy:
            evaluated_ok = False
            eval_reason = "instance_policy_violated"
        elif non_target_in_missing:
            evaluated_ok = False
            eval_reason = "invalid_missing_parts_contains_non_target"
        elif missing_part_detail_issues:
            evaluated_ok = False
            eval_reason = "missing_part_detail_missing"
        elif consistency_violations:
            evaluated_ok = False
            eval_reason = "schema_consistency_violation"
        elif uncertain:
            evaluated_ok = False
            eval_reason = "uncertain_eval"
        reason_codes = list(dict.fromkeys(reason_codes))

        out = {
            "ok": bool(accurate) if evaluated_ok else False,
            "evaluated": bool(evaluated_ok),
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "accurate": bool(accurate),
            "model_accurate_raw": bool(model_accurate),
            "extra_objects": extra,
            "missing_parts": missing,
            "missing_parts_ignored_as_occlusion": missing_ignored_as_occlusion,
            "occluded_target_parts": occluded_target_parts,
            "point_diagnostics": {
                "wrong_positive_points": wrong_pos,
                "wrong_negative_points": wrong_neg,
            },
            "inside_mask_objects": inside_mask_objects,
            "outside_mask_objects": outside_mask_objects,
            "target_fully_contained": bool(target_fully_contained),
            "target_complete_in_mask": bool(target_complete_in_mask),
            "non_target_outside_mask": bool(non_target_outside_mask),
            "uncertain": bool(uncertain),
            "uncertainty_reason": uncertainty_reason,
            "language": language,
            "language_source": language_source,
            "prompt_points_input": {
                "positive_points": positive_points_norm,
                "negative_points": negative_points_norm,
            },
            "instance_policy": {
                "target_object": target_name_lang,
                "non_target_objects": non_targets_lang,
                "target_object_normalized_en": target_name_norm_en,
                "non_target_objects_normalized_en": non_targets_norm_en,
                "violated": bool(violates_instance_policy),
                "consistency_note": consistency_note,
                "violation_hits": non_target_hits,
                "non_target_in_missing_parts": non_target_in_missing,
                "missing_parts_detail_issues": missing_part_detail_issues,
                "inside_mask_contains_non_targets": inside_non_target_objects,
                "inside_mask_contains_target": inside_target_objects,
                "outside_mask_contains_target": outside_target_objects,
            },
            "occlusion_policy": {
                "target_fully_contained_overridden": bool(target_fully_contained_overridden_by_occlusion),
                "missing_reclassified_no_outside_target": bool(missing_reclassified_no_outside_target),
            },
            "inventory_policy": {
                "target_fully_contained_overridden": bool(target_containment_overridden_by_inventory),
                "known_non_targets_all_outside": bool(known_non_targets_all_outside),
            },
            "consistency_violations": consistency_violations,
            "reason_codes": reason_codes,
            "summary": summary,
            "image_shape": [int(h), int(w)],
            "reason": eval_reason if eval_reason else "",
            "raw": parsed,
        }
        self._debug_print_json("[VLM Assistor][Agent1][MaskEval] parsed=", out)
        return out

    def suggest_click_points_for_object(
        self,
        *,
        object_name: str,
        image_rgb_u8,
        non_target_objects: list[str] | None = None,
        points_per_object: int = 3,
        negative_points: int = 1,
        prior_fail_reason: str | None = None,
        prior_fail_analysis: dict[str, Any] | None = None,
        backend_mode: str | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        if image_rgb_u8 is None:
            return {"ok": False, "reason": "missing_image"}

        try:
            height = int(image_rgb_u8.shape[0])
            width = int(image_rgb_u8.shape[1])
        except Exception:
            return {"ok": False, "reason": "invalid_image_shape"}

        data_url = self._to_data_url(image_rgb_u8, jpeg_quality=80)
        if not data_url:
            return {"ok": False, "reason": "image_encode_failed"}

        n_pos = max(1, min(8, int(points_per_object)))
        n_neg = max(0, min(4, int(negative_points)))
        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"
        model_for_lang = self._openai_model() if mode == "official_api" else self._local_vlm_model()
        language = self._language_for_model(model_for_lang)
        non_targets = [
            str(x).strip()
            for x in list(non_target_objects or [])
            if str(x).strip() and str(x).strip().lower() != str(object_name).strip().lower()
        ]
        target_name_lang = self._to_language_object_name(str(object_name), language)
        non_targets_lang = self._to_language_object_names(non_targets, language)

        base_url_probe = self._local_vlm_base_url() if mode == "local_openai_compatible" else self._openai_base_url()
        model_info = self._fetch_local_model_info(base_url_probe, timeout=min(1.5, max(0.5, float(timeout) * 0.2)))
        min_pixels = self._extract_int(model_info, ["min_pixels", "image_min_pixels"]) if model_info else None
        max_pixels = self._extract_int(model_info, ["max_pixels", "image_max_pixels"]) if model_info else None
        factor = self._extract_int(model_info, ["image_factor", "factor"]) if model_info else None
        if factor is None:
            factor = 28
        resize_est = self._estimate_resize_by_pixels(
            width=width,
            height=height,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            factor=int(factor),
        )

        self._debug_print_json(
            "[VLM Assistor][Agent1][PointPlan] input=",
            {
                "object_name": str(object_name),
                "object_name_prompt": target_name_lang,
                "language": language,
                "image_width": int(width),
                "image_height": int(height),
                "points_per_object": int(points_per_object),
                "negative_points": int(negative_points),
                "backend_mode": str(mode),
                "timeout": float(timeout),
                "prior_fail_reason": str(prior_fail_reason or ""),
                "prior_fail_analysis": dict(prior_fail_analysis or {}),
                "non_target_objects_prompt": non_targets_lang,
                "qwen_model_info": model_info,
                "qwen_resize_estimate": resize_est,
            },
        )

        fail_note = str(prior_fail_reason or "").strip()
        fail_analysis = dict(prior_fail_analysis or {})
        fail_analysis_text = ""
        if fail_analysis:
            try:
                fail_analysis_text = json.dumps(fail_analysis, ensure_ascii=False)
            except Exception:
                fail_analysis_text = str(fail_analysis)
            if len(fail_analysis_text) > 2200:
                fail_analysis_text = fail_analysis_text[:2200] + "...(truncated)"
        if language == "zh":
            fail_text = f"上一次尝试失败原因：{fail_note}\n" if fail_note else ""
            fail_analysis_text_block = (
                f"上一次失败结构化分析（用于纠正点位）: {fail_analysis_text}\n"
                if fail_analysis_text
                else ""
            )
            prompt = (
                "你是用于 SAM 分割的点击点规划器。\n"
                f"图像大小：width={width}, height={height}。\n"
                f"目标物体（中文）：{target_name_lang}。\n"
                f"非目标物体（中文，绝对不能被分到目标mask）：{json.dumps(non_targets_lang, ensure_ascii=False)}。\n"
                f"{fail_text}"
                f"{fail_analysis_text_block}"
                "仅返回 JSON，坐标必须是 [0,1] 的归一化坐标（不是像素坐标）：\n"
                "{\n"
                '  "part_decomposition": [{"part_name": string, "is_visible": bool, "importance": "high|medium|low"}],\n'
                '  "positive_points": [{"x": float_0_to_1, "y": float_0_to_1, "part_name": string}],\n'
                '  "negative_points": [{"x": float_0_to_1, "y": float_0_to_1}],\n'
                '  "positive_point_part_assignments": [{"point_index": int, "part_name": string}],\n'
                '  "reason": "简短说明"\n'
                "}\n"
                "规则：\n"
                f"1) 必须给出恰好 {n_pos} 个正点，全部落在目标物体上。\n"
                "2) 必须先结合“目标文本语义 + 输入图像可见结构”做部位拆分，并在 part_decomposition 中输出可见主要部位。\n"
                "3) 若目标有多个明显部位，正点尽量覆盖每个主要部位；若点数不足，优先关键部位（例如刀身+刀把、杯身+杯把）。\n"
                f"4) 必须给出恰好 {n_neg} 个负点，优先放在非目标物体中心；若无明显非目标中心，再放纯白背景中心。\n"
                "5) 所有坐标必须是 [0,1] 归一化坐标（不要像素坐标）。\n"
                "6) 正点必须避开所有列出的非目标物体，并尽量不要落在背景上。\n"
                "7) 严格实例策略：接触/上下文/功能关系不代表同一实例。\n"
                "8) 若提供了“上一次失败结构化分析”，必须优先修正 wrong_positive_points / wrong_negative_points / missing_parts / extra_objects。\n"
                "9) reason 字段必须使用中文。\n"
            )
            system_content = "仅返回合法 JSON。自由文本字段使用中文。"
        else:
            fail_text = f"Previous attempt failed because: {fail_note}\n" if fail_note else ""
            fail_analysis_text_block = (
                f"Structured failure analysis from previous attempt (for point correction): {fail_analysis_text}\n"
                if fail_analysis_text
                else ""
            )
            prompt = (
                "You are an object click-point planner for SAM segmentation.\n"
                f"Image size: width={width}, height={height}.\n"
                f"Target object (English): {target_name_lang}.\n"
                f"Non-target object names (English, must NOT be segmented into target): {json.dumps(non_targets_lang, ensure_ascii=False)}.\n"
                f"{fail_text}"
                f"{fail_analysis_text_block}"
                "Return JSON only with normalized coordinates in [0,1] (NOT absolute pixels):\n"
                "{\n"
                '  "part_decomposition": [{"part_name": string, "is_visible": bool, "importance": "high|medium|low"}],\n'
                '  "positive_points": [{"x": float_0_to_1, "y": float_0_to_1, "part_name": string}],\n'
                '  "negative_points": [{"x": float_0_to_1, "y": float_0_to_1}],\n'
                '  "positive_point_part_assignments": [{"point_index": int, "part_name": string}],\n'
                '  "reason": "short explanation"\n'
                "}\n"
                "Rules:\n"
                f"1) Provide exactly {n_pos} positive_points on the target object.\n"
                "2) First decompose the target into visible major parts using BOTH target text semantics and image structure; output them in part_decomposition.\n"
                "3) If multiple parts are visible, distribute positive points across major parts; when points are limited, prioritize key parts (e.g., knife blade+handle, cup body+handle).\n"
                f"4) Provide exactly {n_neg} negative_points: prefer non-target centers; if unavailable, use clean white-background centers.\n"
                "5) All coordinates must be normalized to [0,1] (not pixel coordinates).\n"
                "6) Positive points must avoid listed non-target objects and should avoid background whenever possible.\n"
                "7) Strict instance policy: contact/context/function relation does NOT merge objects.\n"
                "8) If structured failure analysis is provided, prioritize correcting wrong_positive_points / wrong_negative_points / missing_parts / extra_objects.\n"
                "9) Use ENGLISH ONLY in the reason field.\n"
            )
            system_content = "Output JSON only. Use English only in free-text fields."

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ]

        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
            payload["model"] = model
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = messages
            safe_payload = dict(payload)
            safe_payload["messages"] = self._debug_sanitize_messages(messages)
            self._debug_print_json(
                "[VLM Assistor][Agent1][PointPlan] request=",
                {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "has_api_key": bool(api_key),
                    "endpoint": str(base_url).rstrip("/") + "/chat/completions",
                    "payload": safe_payload,
                    "prompt": prompt,
                },
            )
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            local_messages = self._convert_system_to_user_prefix(messages)
            payload["messages"] = local_messages
            if not base_url:
                return {"ok": False, "reason": "local_openai_base_url_missing"}
            safe_payload = dict(payload)
            safe_payload["messages"] = self._debug_sanitize_messages(local_messages)
            self._debug_print_json(
                "[VLM Assistor][Agent1][PointPlan] request=",
                {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "has_api_key": bool(api_key),
                    "endpoint": str(base_url).rstrip("/") + "/chat/completions",
                    "payload": safe_payload,
                    "prompt": prompt,
                },
            )
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )

        if not isinstance(resp, dict):
            self._debug_print_json(
                "[VLM Assistor][Agent1][PointPlan] response_error=",
                {"provider": provider, "model": model, "base_url": base_url, "resp_type": str(type(resp))},
            )
            return {
                "ok": False,
                "reason": f"{provider}_http_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }

        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
        except Exception as exc:
            self._debug_print_json(
                "[VLM Assistor][Agent1][PointPlan] response_parse_error=",
                {"provider": provider, "model": model, "base_url": base_url, "error": f"{type(exc).__name__}:{exc}"},
            )
            return {
                "ok": False,
                "reason": f"invalid_response:{type(exc).__name__}:{exc}",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }
        print("[VLM Assistor][Agent1][PointPlan] raw_response_text:", str(text)[:1200])

        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            self._debug_print_json(
                "[VLM Assistor][Agent1][PointPlan] response_json_parse_failed=",
                {"provider": provider, "model": model, "base_url": base_url, "raw_text_head": str(text)[:400]},
            )
            return {
                "ok": False,
                "reason": "points_json_parse_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "raw_text": str(text)[:400],
            }

        part_raw = parsed.get("part_decomposition", parsed.get("parts", parsed.get("part_plan", [])))
        part_decomposition: list[dict[str, Any]] = []
        if isinstance(part_raw, list):
            for it in part_raw:
                if not isinstance(it, dict):
                    continue
                pn = str(it.get("part_name", it.get("name", ""))).strip()
                if not pn:
                    continue
                part_decomposition.append(
                    {
                        "part_name": pn,
                        "is_visible": bool(it.get("is_visible", True)),
                        "importance": str(it.get("importance", "medium")).strip().lower() or "medium",
                    }
                )
                if len(part_decomposition) >= 16:
                    break
        pos_raw = parsed.get("positive_points", parsed.get("points", parsed.get("positive", [])))
        neg_raw = parsed.get("negative_points", parsed.get("negative", []))
        pos_assign_raw = parsed.get("positive_point_part_assignments", parsed.get("point_part_assignments", []))
        pos_part_assignments: list[dict[str, Any]] = []
        if isinstance(pos_assign_raw, list):
            for it in pos_assign_raw:
                if not isinstance(it, dict):
                    continue
                part_name = str(it.get("part_name", it.get("name", ""))).strip()
                if not part_name:
                    continue
                try:
                    idx = int(it.get("point_index", -1))
                except Exception:
                    idx = -1
                pos_part_assignments.append({"point_index": int(idx), "part_name": str(part_name)})
                if len(pos_part_assignments) >= 32:
                    break
        positive = self._normalize_click_points(pos_raw, width=width, height=height, limit=n_pos)
        positive_before_refine = list(positive)
        positive = self._center_band_refine_points(
            list(positive),
            width=int(width),
            height=int(height),
            shrink=0.68,
            max_move_ratio=0.08,
        )
        negative = self._normalize_click_points(neg_raw, width=width, height=height, limit=n_neg)
        positive = self._pad_click_points(positive, width=width, height=height, target_count=n_pos)
        self._debug_print_json(
            "[VLM Assistor][Agent1][PointPlan] parsed_points=",
            {
                "reason": str(parsed.get("reason", "")).strip(),
                "part_decomposition": part_decomposition,
                "raw_positive_points": pos_raw,
                "raw_negative_points": neg_raw,
                "positive_point_part_assignments": pos_part_assignments,
                "positive_points_before_center_band_refine": positive_before_refine,
                "positive_points": positive,
                "negative_points": negative,
                "convert_rule": "if 0<=x<=1 and 0<=y<=1 then x*=width-1,y*=height-1 else treat as pixels",
                "center_band_refine": {"enabled": True, "shrink": 0.68, "max_move_ratio": 0.08},
            },
        )

        return {
            "ok": len(positive) > 0,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "part_decomposition": part_decomposition,
            "positive_point_part_assignments": pos_part_assignments,
            "positive_points": positive,
            "negative_points": negative,
            "reason": str(parsed.get("reason", "")).strip(),
        }

    def _infer_x_depth_order_prior_with_vlm(
        self,
        *,
        object_name: str,
        relation_hint: str,
        scene_graph: dict[str, Any] | None,
        object_context: dict[str, Any] | None,
        input_rgb_u8: np.ndarray | None,
        overlay_rgb_u8: np.ndarray | None,
        view_images: dict[str, np.ndarray],
        mode: str,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        language = self._language_for_model(model)
        target_name = self._to_language_object_name(str(object_name), language)
        scene_graph_compact = dict(scene_graph or {})
        if len(json.dumps(scene_graph_compact, ensure_ascii=False)) > 3500:
            scene_graph_compact = {
                "objects": list(scene_graph_compact.get("objects", []) or [])[:16],
                "relations": list(scene_graph_compact.get("relations", []) or [])[:20],
                "events": list(scene_graph_compact.get("events", []) or [])[:12],
                "x_depth_order_timeline": list(scene_graph_compact.get("x_depth_order_timeline", []) or [])[:8],
                "relative_position_timeline": list(scene_graph_compact.get("relative_position_timeline", []) or [])[:8],
            }
        object_context_compact = dict(object_context or {})
        try:
            if len(json.dumps(object_context_compact, ensure_ascii=False)) > 3000:
                object_context_compact = {
                    "current_generated_object": dict(object_context_compact.get("current_generated_object", {}) or {}),
                    "already_generated_objects": list(object_context_compact.get("already_generated_objects", []) or [])[:16],
                    "all_generated_objects": list(object_context_compact.get("all_generated_objects", []) or [])[:20],
                    "generated_name_map": dict(object_context_compact.get("generated_name_map", {}) or {}),
                }
        except Exception:
            object_context_compact = {
                "current_generated_object": dict((object_context or {}).get("current_generated_object", {}) or {}),
                "already_generated_objects": list((object_context or {}).get("already_generated_objects", []) or [])[:16],
            }
        target_relation_context = self._build_target_relation_context(
            object_name=str(object_name),
            scene_graph=dict(scene_graph_compact or {}),
            object_context=dict(object_context_compact or {}),
            max_items=16,
        )

        if language == "zh":
            prompt = (
                "你是3D深度排序分析器。任务：先根据视频理解描述和输入图像，分析当前目标物体与已生成物体在x轴(相机深度)上的前后排序。"
                "请显式利用目标相关的视频关系上下文(关系/事件/约束)，例如“夹子夹住杯子”。"
                "注意：输出方向时必须使用世界坐标系方向：positive=+X, negative=-X。"
                "只返回JSON："
                '{'
                '"target_depth_order":[{"other_object":string,"relation":"target_in_front|target_behind|same_depth|unknown","confidence":0-1}],'
                '"suggested_x_direction_prior":"positive|negative|none",'
                '"is_depth_order_consistent_with_video":bool,'
                '"confidence":0-1,'
                '"summary":"一句中文结论"'
                '}'
                f"\n目标物体: {target_name}"
                f"\n关系提示: {str(relation_hint or '').strip()}"
                f"\n时序图摘要: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\n物体上下文(当前刚生成目标 + 已生成物体): {json.dumps(object_context_compact, ensure_ascii=False)}"
                f"\n目标相关视频关系上下文: {json.dumps(target_relation_context, ensure_ascii=False)}"
            )
            system_content = "只返回合法JSON。不要输出额外文字。"
        else:
            prompt = (
                "You are a 3D depth-order analyzer. First infer x-axis (camera-depth-like) ordering between current target and already generated objects from video understanding and input image."
                "You must explicitly use target-related video relation context (relations/events/constraints), e.g., 'gripper holds cup'."
                "Direction convention must be world-axis: positive=+X, negative=-X."
                "Return JSON only: "
                '{'
                '"target_depth_order":[{"other_object":string,"relation":"target_in_front|target_behind|same_depth|unknown","confidence":0-1}],'
                '"suggested_x_direction_prior":"positive|negative|none",'
                '"is_depth_order_consistent_with_video":bool,'
                '"confidence":0-1,'
                '"summary":"one-sentence conclusion"'
                '}'
                f"\nTarget object: {target_name}"
                f"\nRelation hint: {str(relation_hint or '').strip()}"
                f"\nTSG summary: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\nObject context (current generated target + already generated objects): {json.dumps(object_context_compact, ensure_ascii=False)}"
                f"\nTarget-related video relation context: {json.dumps(target_relation_context, ensure_ascii=False)}"
            )
            system_content = "Return valid JSON only."

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if input_rgb_u8 is not None:
            input_url = self._to_data_url(input_rgb_u8, jpeg_quality=85)
            if input_url:
                content.append({"type": "text", "text": "[input_image]"})
                content.append({"type": "image_url", "image_url": {"url": input_url, "detail": "high"}})
        if overlay_rgb_u8 is not None:
            overlay_url = self._to_data_url(overlay_rgb_u8, jpeg_quality=82)
            if overlay_url:
                content.append({"type": "text", "text": "[input_overlay]"})
                content.append({"type": "image_url", "image_url": {"url": overlay_url, "detail": "high"}})
        for k in ["front", "back", "left", "right", "up", "down"]:
            arr = view_images.get(k, None)
            if arr is None:
                continue
            u = self._to_data_url(arr, jpeg_quality=72)
            if not u:
                continue
            content.append({"type": "text", "text": f"[view:{k}]"})
            content.append({"type": "image_url", "image_url": {"url": u, "detail": "low"}})

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": content}]
        self._debug_print_json(
            "[VLM Assistor][Agent2][DepthOrder] request=",
            {
                "mode": mode,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "target_object": target_name,
                "relation_hint": str(relation_hint or "").strip(),
                "object_context": dict(object_context_compact or {}),
                "target_relation_context": dict(target_relation_context or {}),
                "prompt": prompt,
                "has_input": bool(input_rgb_u8 is not None),
                "has_overlay": bool(overlay_rgb_u8 is not None),
                "view_keys": sorted([str(k) for k in list(view_images.keys())]),
                "messages_preview": self._debug_sanitize_messages(messages),
            },
        )
        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            payload["messages"] = messages
            payload["response_format"] = {"type": "json_object"}
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            payload["messages"] = self._convert_system_to_user_prefix(messages)
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )
        if not isinstance(resp, dict):
            return {"ok": False, "evaluated": False, "reason": "depth_order_http_failed"}
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
            print("[VLM Assistor][Agent2][DepthOrder] raw_response_text:", str(text)[:2000])
        except Exception as exc:
            return {"ok": False, "evaluated": False, "reason": f"depth_order_parse_failed:{type(exc).__name__}:{exc}"}
        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            return {"ok": False, "evaluated": False, "reason": "depth_order_json_parse_failed", "raw_text": str(text)[:300]}
        self._debug_print_json(
            "[VLM Assistor][Agent2][DepthOrder] parsed_json=",
            parsed,
        )
        rel_map = {
            "target_in_front": "target_in_front",
            "in_front": "target_in_front",
            "front": "target_in_front",
            "target_behind": "target_behind",
            "behind": "target_behind",
            "back": "target_behind",
            "same_depth": "same_depth",
            "same": "same_depth",
            "unknown": "unknown",
        }
        order_items: list[dict[str, Any]] = []
        raw_items = list(parsed.get("target_depth_order", parsed.get("depth_order", [])) or [])
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            other = str(it.get("other_object", it.get("object", it.get("name", "")))).strip()
            rel_raw = str(it.get("relation", it.get("relative_depth", "unknown"))).strip().lower()
            rel = rel_map.get(rel_raw, "unknown")
            try:
                conf = float(it.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            order_items.append(
                {
                    "other_object": other,
                    "relation": rel,
                    "confidence": float(max(0.0, min(1.0, conf))),
                }
            )
            if len(order_items) >= 24:
                break
        out = {
            "ok": True,
            "evaluated": True,
            "suggested_x_direction_prior": self._normalize_x_direction(
                parsed.get("suggested_x_direction_prior", parsed.get("suggested_x_direction", "none"))
            ),
            "is_depth_order_consistent_with_video": bool(
                parsed.get("is_depth_order_consistent_with_video", parsed.get("consistent_with_video", True))
            ),
            "confidence": float(max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))),
            "target_depth_order": order_items,
            "summary": str(parsed.get("summary", parsed.get("depth_order_summary", ""))).strip(),
            "raw": parsed,
        }
        self._debug_print_json(
            "[VLM Assistor][Agent2][DepthOrder] normalized_result=",
            out,
        )
        return out

    def suggest_xpose_delta_with_vlm(
        self,
        *,
        object_name: str,
        relation_hint: str,
        scene_graph: dict[str, Any] | None,
        rule_diagnostics: dict[str, Any],
        view_images: dict[str, np.ndarray],
        input_rgb_u8: np.ndarray | None = None,
        overlay_rgb_u8: np.ndarray | None = None,
        object_context: dict[str, Any] | None = None,
        backend_mode: str | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        if not isinstance(view_images, dict) or len(view_images) <= 0:
            return {"ok": False, "evaluated": False, "reason": "missing_view_images"}
        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"

        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
        if not base_url:
            return {"ok": False, "evaluated": False, "reason": "vlm_base_url_missing"}

        language = self._language_for_model(model)
        target_name = self._to_language_object_name(str(object_name), language)
        scene_graph_compact = dict(scene_graph or {})
        if len(json.dumps(scene_graph_compact, ensure_ascii=False)) > 4500:
            scene_graph_compact = {
                "objects": list(scene_graph_compact.get("objects", []) or [])[:18],
                "relations": list(scene_graph_compact.get("relations", []) or [])[:24],
                "events": list(scene_graph_compact.get("events", []) or [])[:16],
                "constraints": list(scene_graph_compact.get("constraints", []) or [])[:16],
                "x_depth_order_timeline": list(scene_graph_compact.get("x_depth_order_timeline", []) or [])[:10],
                "relative_position_timeline": list(scene_graph_compact.get("relative_position_timeline", []) or [])[:10],
            }
        object_context_compact = dict(object_context or {})
        try:
            if len(json.dumps(object_context_compact, ensure_ascii=False)) > 3000:
                object_context_compact = {
                    "current_generated_object": dict(object_context_compact.get("current_generated_object", {}) or {}),
                    "already_generated_objects": list(object_context_compact.get("already_generated_objects", []) or [])[:16],
                    "all_generated_objects": list(object_context_compact.get("all_generated_objects", []) or [])[:20],
                    "generated_name_map": dict(object_context_compact.get("generated_name_map", {}) or {}),
                }
        except Exception:
            object_context_compact = {
                "current_generated_object": dict((object_context or {}).get("current_generated_object", {}) or {}),
                "already_generated_objects": list((object_context or {}).get("already_generated_objects", []) or [])[:16],
            }

        depth_order_prior = self._infer_x_depth_order_prior_with_vlm(
            object_name=str(object_name),
            relation_hint=str(relation_hint),
            scene_graph=dict(scene_graph_compact or {}),
            object_context=dict(object_context_compact or {}),
            input_rgb_u8=input_rgb_u8,
            overlay_rgb_u8=overlay_rgb_u8,
            view_images=dict(view_images or {}),
            mode=mode,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=min(60.0, max(15.0, float(timeout) * 0.65)),
        )
        target_relation_context = self._build_target_relation_context(
            object_name=str(object_name),
            scene_graph=dict(scene_graph_compact or {}),
            object_context=dict(object_context_compact or {}),
            max_items=20,
        )

        if language == "zh":
            prompt = (
                "你是 Agent2 的位移规划器。目标：直接给出当前目标物体沿世界坐标系 X 轴的一次性位移方向和位移量，"
                "不做多候选穷举。"
                "请结合输入图、overlay、六视角渲染、video understanding 的时序关系和相对位置关系来判断。"
                "特别注意：输入图中的“正上方/正下方/正左方/正右方”关系常隐含前后深度约束，必须显式分析。"
                "请先分析目标物体与已生成物体的相对关系，再给位移建议。"
                "请输出 JSON："
                "{"
                '"suggested_world_x_direction":"positive|negative|none",'
                '"normalized_shift_magnitude":0-1,'
                '"normalized_shift_signed":-1~1,'
                '"up_view_target_center_point":{"x_norm":0-1,"y_up_norm":0-1,"valid":bool,"reason":string},'
                '"target_alignment_mode":"align_reference_center|offset_reference_center|delta_only|fallback_zero_x|none",'
                '"reference_object":string,'
                '"reference_offset_normalized_signed":-1~1,'
                '"relative_analysis":[{"other_object":string,"expected_horizontal":"left|right|overlap|unknown","expected_vertical":"above|below|overlap|unknown","expected_depth":"front|behind|same_depth|unknown","current_horizontal":"left|right|overlap|unknown","current_vertical":"above|below|overlap|unknown","current_depth":"front|behind|same_depth|unknown","consistent":bool,"evidence":string}],'
                '"analysis_summary":string,'
                '"confidence":0-1,'
                '"reason_codes":[string]'
                "}"
                f"\n目标物体: {target_name}"
                f"\n关系提示: {str(relation_hint or '').strip()}"
                f"\n时序图摘要: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\n目标相关关系上下文: {json.dumps(target_relation_context, ensure_ascii=False)}"
                f"\n前置x轴深度排序先验: {json.dumps(depth_order_prior, ensure_ascii=False)}"
                f"\n规则诊断(AABB/x范围): {json.dumps(rule_diagnostics, ensure_ascii=False)}"
                f"\n物体上下文(当前目标 + 已生成物体): {json.dumps(object_context_compact, ensure_ascii=False)}"
                "\n方向定义：positive=世界 +X，negative=世界 -X。"
                "\nnormalized_shift_magnitude 定义：相对于输入图半宽度归一化（0.1 表示约 10% 半宽度）。"
                "\n必须在 [view:up] 中给出“目标中心点” up_view_target_center_point。"
                "\nup_view_target_center_point 约定：x_norm 左=0 右=1；y_up_norm 上=1 下=0。"
                "\n该点必须满足物理合理性，并与已生成物体相对关系尽量一致。若确实无法判断，设置 valid=false 并给 reason。"
                "\n若目标应在另一物体正上/正下或同列，优先给 target_alignment_mode=align_reference_center，并给 reference_object。"
                "\n若目标应相对另一物体偏移 n 单位，给 target_alignment_mode=offset_reference_center，reference_object 和 reference_offset_normalized_signed。"
                "\n若无需移动，请输出 direction=none 且 shift=0。"
            )
            system_content = "只返回合法JSON，不要输出额外解释。"
        else:
            prompt = (
                "You are the Agent2 displacement planner. Goal: directly output ONE world-x translation direction and amount for current target object,"
                " without candidate enumeration."
                " Use input image, overlay, six-view renders, and video-understanding temporal + relative-position relations."
                " Important: top/bottom/left/right relations in the input image can imply depth-order constraints; reason about this explicitly."
                " First analyze target-vs-generated relative relations, then output the displacement suggestion."
                " Return JSON only: "
                "{"
                '"suggested_world_x_direction":"positive|negative|none",'
                '"normalized_shift_magnitude":0-1,'
                '"normalized_shift_signed":-1~1,'
                '"up_view_target_center_point":{"x_norm":0-1,"y_up_norm":0-1,"valid":bool,"reason":string},'
                '"target_alignment_mode":"align_reference_center|offset_reference_center|delta_only|fallback_zero_x|none",'
                '"reference_object":string,'
                '"reference_offset_normalized_signed":-1~1,'
                '"relative_analysis":[{"other_object":string,"expected_horizontal":"left|right|overlap|unknown","expected_vertical":"above|below|overlap|unknown","expected_depth":"front|behind|same_depth|unknown","current_horizontal":"left|right|overlap|unknown","current_vertical":"above|below|overlap|unknown","current_depth":"front|behind|same_depth|unknown","consistent":bool,"evidence":string}],'
                '"analysis_summary":string,'
                '"confidence":0-1,'
                '"reason_codes":[string]'
                "}"
                f"\nTarget object: {target_name}"
                f"\nRelation hint: {str(relation_hint or '').strip()}"
                f"\nTSG summary: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\nTarget-related relation context: {json.dumps(target_relation_context, ensure_ascii=False)}"
                f"\nPre-step x-depth prior: {json.dumps(depth_order_prior, ensure_ascii=False)}"
                f"\nRule diagnostics (AABB/x-range): {json.dumps(rule_diagnostics, ensure_ascii=False)}"
                f"\nObject context (current target + generated objects): {json.dumps(object_context_compact, ensure_ascii=False)}"
                "\nDirection convention: positive=world +X, negative=world -X."
                "\nnormalized_shift_magnitude is normalized by half image width (e.g., 0.1 means ~10% of half width)."
                "\nYou must provide up_view_target_center_point in [view:up]."
                "\nup_view_target_center_point convention: x_norm left=0 right=1; y_up_norm top=1 bottom=0."
                "\nThe point must satisfy physical plausibility and be consistent with generated-object relative relations as much as possible."
                "\nIf the point is not inferable, set valid=false and provide reason."
                "\nIf target should be directly above/below or column-aligned with another object, prefer target_alignment_mode=align_reference_center and provide reference_object."
                "\nIf target should be offset by n units from another object center, use target_alignment_mode=offset_reference_center and provide reference_object + reference_offset_normalized_signed."
                "\nIf no move is needed, output direction=none and shift=0."
            )
            system_content = "Return valid JSON only."

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if input_rgb_u8 is not None:
            input_url = self._to_data_url(input_rgb_u8, jpeg_quality=85)
            if input_url:
                content.append({"type": "text", "text": "[input_image]"})
                content.append({"type": "image_url", "image_url": {"url": input_url, "detail": "high"}})
        if overlay_rgb_u8 is not None:
            overlay_url = self._to_data_url(overlay_rgb_u8, jpeg_quality=80)
            if overlay_url:
                content.append({"type": "text", "text": "[input_overlay]"})
                content.append({"type": "image_url", "image_url": {"url": overlay_url, "detail": "high"}})
        for k in sorted(view_images.keys()):
            arr = view_images.get(k)
            if arr is None:
                continue
            u = self._to_data_url(arr, jpeg_quality=75)
            if not u:
                continue
            content.append({"type": "text", "text": f"[view:{k}]"})
            content.append({"type": "image_url", "image_url": {"url": u, "detail": "low"}})

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": content}]
        self._debug_print_json(
            "[VLM Assistor][Agent2][DeltaPlan] request=",
            {
                "mode": mode,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "target_object": target_name,
                "relation_hint": str(relation_hint or "").strip(),
                "rule_diagnostics": dict(rule_diagnostics or {}),
                "target_relation_context": dict(target_relation_context or {}),
                "depth_order_prior": dict(depth_order_prior or {}),
                "messages_preview": self._debug_sanitize_messages(messages),
            },
        )
        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            payload["messages"] = messages
            payload["response_format"] = {"type": "json_object"}
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            payload["messages"] = self._convert_system_to_user_prefix(messages)
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )
        if not isinstance(resp, dict):
            return {
                "ok": False,
                "evaluated": False,
                "reason": f"{provider}_http_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
            print("[VLM Assistor][Agent2][DeltaPlan] raw_response_text:", str(text)[:2000])
        except Exception as exc:
            return {
                "ok": False,
                "evaluated": False,
                "reason": f"invalid_response:{type(exc).__name__}:{exc}",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }

        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            return {
                "ok": False,
                "evaluated": False,
                "reason": "delta_plan_json_parse_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "raw_text": str(text)[:300],
            }

        dir_world = self._normalize_x_direction(
            parsed.get(
                "suggested_world_x_direction",
                parsed.get("suggested_x_direction", parsed.get("direction", "none")),
            )
        )
        try:
            signed_shift = float(
                parsed.get(
                    "normalized_shift_signed",
                    parsed.get("normalized_dx_signed", parsed.get("normalized_delta_x", 0.0)),
                )
            )
        except Exception:
            signed_shift = 0.0
        try:
            mag_shift = float(
                parsed.get(
                    "normalized_shift_magnitude",
                    parsed.get("normalized_dx_magnitude", abs(signed_shift)),
                )
            )
        except Exception:
            mag_shift = abs(signed_shift)

        mag_shift = float(max(0.0, min(1.0, abs(mag_shift))))
        if abs(signed_shift) > 1e-6:
            signed_shift = float(max(-1.5, min(1.5, signed_shift)))
            if dir_world == "none":
                dir_world = "positive" if signed_shift > 0 else "negative"
        else:
            if dir_world == "positive":
                signed_shift = float(mag_shift)
            elif dir_world == "negative":
                signed_shift = float(-mag_shift)
            else:
                signed_shift = 0.0
        if dir_world != "none" and abs(signed_shift) < 1e-6:
            signed_shift = float(0.08 if dir_world == "positive" else -0.08)

        up_img = view_images.get("up", None)
        up_h = up_w = 512
        try:
            if isinstance(up_img, np.ndarray) and up_img.ndim >= 2:
                up_h = max(1, int(up_img.shape[0]))
                up_w = max(1, int(up_img.shape[1]))
        except Exception:
            up_h = up_w = 512

        def _coerce_norm_01(raw_v: Any, *, span: int) -> float | None:
            try:
                fv = float(raw_v)
            except Exception:
                return None
            if not np.isfinite(fv):
                return None
            if 0.0 <= fv <= 1.0:
                return float(fv)
            den = max(1.0, float(int(span) - 1))
            return float(max(0.0, min(1.0, fv / den)))

        up_point_raw = parsed.get(
            "up_view_target_center_point",
            parsed.get("up_view_target_point", parsed.get("target_center_point_up_view", None)),
        )
        up_x = None
        up_y_up = None
        up_valid = False
        up_reason = ""
        if isinstance(up_point_raw, dict):
            up_x = up_point_raw.get(
                "x_norm",
                up_point_raw.get("x_norm_01", up_point_raw.get("x", up_point_raw.get("u", up_point_raw.get("u_norm", None)))),
            )
            up_y_up = up_point_raw.get(
                "y_up_norm",
                up_point_raw.get(
                    "y_up_norm_01",
                    up_point_raw.get("y_up", up_point_raw.get("v_up", up_point_raw.get("y", up_point_raw.get("v", None)))),
                ),
            )
            up_valid = bool(up_point_raw.get("valid", True))
            up_reason = str(up_point_raw.get("reason", up_point_raw.get("evidence", ""))).strip()
        elif isinstance(up_point_raw, (list, tuple)) and len(up_point_raw) >= 2:
            up_x = up_point_raw[0]
            up_y_up = up_point_raw[1]
            up_valid = True
        else:
            up_x = parsed.get("up_view_target_x_norm", parsed.get("up_target_x_norm", None))
            up_y_up = parsed.get("up_view_target_y_up_norm", parsed.get("up_target_y_up_norm", None))
            up_valid = up_x is not None and up_y_up is not None
        up_x01 = _coerce_norm_01(up_x, span=up_w)
        up_y01 = _coerce_norm_01(up_y_up, span=up_h)
        if up_x01 is None or up_y01 is None:
            up_valid = False
        else:
            up_valid = bool(up_valid)

        align_mode_raw = str(
            parsed.get(
                "target_alignment_mode",
                parsed.get("alignment_mode", parsed.get("target_align_mode", "none")),
            )
        ).strip().lower()
        align_mode_map = {
            "align_reference_center": "align_reference_center",
            "align_with_reference_center": "align_reference_center",
            "align_center": "align_reference_center",
            "offset_reference_center": "offset_reference_center",
            "offset_from_reference_center": "offset_reference_center",
            "offset_center": "offset_reference_center",
            "delta_only": "delta_only",
            "fallback_zero_x": "fallback_zero_x",
            "none": "none",
        }
        align_mode = align_mode_map.get(align_mode_raw, "none")
        reference_object = str(
            parsed.get("reference_object", parsed.get("anchor_object", parsed.get("reference", "")))
        ).strip()
        try:
            reference_offset_norm_signed = float(
                parsed.get(
                    "reference_offset_normalized_signed",
                    parsed.get("anchor_offset_normalized_signed", signed_shift),
                )
            )
        except Exception:
            reference_offset_norm_signed = float(signed_shift)
        reference_offset_norm_signed = float(max(-1.5, min(1.5, reference_offset_norm_signed)))

        rel_raw = parsed.get("relative_analysis", parsed.get("pre_adjust_relative_analysis", []))
        relative_analysis: list[dict[str, Any]] = []
        if isinstance(rel_raw, list):
            for it in rel_raw:
                if not isinstance(it, dict):
                    continue
                other = str(it.get("other_object", it.get("object", it.get("name", "")))).strip()
                if not other:
                    continue
                relative_analysis.append(
                    {
                        "other_object": other,
                        "expected_horizontal": self._coerce_relative_axis(
                            it.get("expected_horizontal", it.get("input_horizontal", "unknown")),
                            axis="horizontal",
                        ),
                        "expected_vertical": self._coerce_relative_axis(
                            it.get("expected_vertical", it.get("input_vertical", "unknown")),
                            axis="vertical",
                        ),
                        "expected_depth": self._coerce_relative_axis(
                            it.get("expected_depth", it.get("input_depth", "unknown")),
                            axis="depth",
                        ),
                        "current_horizontal": self._coerce_relative_axis(
                            it.get("current_horizontal", it.get("candidate_horizontal", "unknown")),
                            axis="horizontal",
                        ),
                        "current_vertical": self._coerce_relative_axis(
                            it.get("current_vertical", it.get("candidate_vertical", "unknown")),
                            axis="vertical",
                        ),
                        "current_depth": self._coerce_relative_axis(
                            it.get("current_depth", it.get("candidate_depth", "unknown")),
                            axis="depth",
                        ),
                        "consistent": bool(it.get("consistent", it.get("is_consistent", True))),
                        "evidence": str(it.get("evidence", it.get("notes", ""))).strip(),
                    }
                )
                if len(relative_analysis) >= 16:
                    break

        out = {
            "ok": True,
            "evaluated": True,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "suggested_world_x_direction": str(dir_world),
            "normalized_shift_signed": float(signed_shift),
            "normalized_shift_magnitude": float(abs(signed_shift)),
            "up_view_target_center_point": {
                "valid": bool(up_valid),
                "x_norm": (None if up_x01 is None else float(up_x01)),
                "y_up_norm": (None if up_y01 is None else float(up_y01)),
                "reason": str(up_reason),
                "image_width": int(up_w),
                "image_height": int(up_h),
            },
            "target_alignment_mode": str(align_mode),
            "reference_object": str(reference_object),
            "reference_offset_normalized_signed": float(reference_offset_norm_signed),
            "relative_analysis": relative_analysis,
            "analysis_summary": str(
                parsed.get("analysis_summary", parsed.get("pre_adjust_analysis_summary", parsed.get("summary", "")))
            ).strip(),
            "confidence": float(max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))),
            "reason_codes": [str(x) for x in list(parsed.get("reason_codes", [])) if str(x).strip()],
            "target_relation_context": dict(target_relation_context or {}),
            "depth_order_prior": dict(depth_order_prior or {}),
            "raw": parsed,
        }
        self._debug_print_json(
            "[VLM Assistor][Agent2][DeltaPlan] normalized_result=",
            out,
        )
        return out

    def evaluate_pose_plausibility_with_vlm(
        self,
        *,
        object_name: str,
        relation_hint: str,
        scene_graph: dict[str, Any] | None,
        rule_diagnostics: dict[str, Any],
        view_images: dict[str, np.ndarray],
        input_rgb_u8: np.ndarray | None = None,
        overlay_rgb_u8: np.ndarray | None = None,
        object_context: dict[str, Any] | None = None,
        backend_mode: str | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        if not isinstance(view_images, dict) or len(view_images) <= 0:
            return {"ok": False, "evaluated": False, "reason": "missing_view_images"}
        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"

        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
        if not base_url:
            return {"ok": False, "evaluated": False, "reason": "vlm_base_url_missing"}

        language = self._language_for_model(model)
        target_name = self._to_language_object_name(str(object_name), language)
        scene_graph_compact = dict(scene_graph or {})
        if len(json.dumps(scene_graph_compact, ensure_ascii=False)) > 4000:
            scene_graph_compact = {
                "objects": list(scene_graph_compact.get("objects", []) or [])[:16],
                "relations": list(scene_graph_compact.get("relations", []) or [])[:20],
                "events": list(scene_graph_compact.get("events", []) or [])[:12],
                "constraints": list(scene_graph_compact.get("constraints", []) or [])[:12],
                "x_depth_order_timeline": list(scene_graph_compact.get("x_depth_order_timeline", []) or [])[:8],
                "relative_position_timeline": list(scene_graph_compact.get("relative_position_timeline", []) or [])[:8],
            }
        object_context_compact = dict(object_context or {})
        try:
            if len(json.dumps(object_context_compact, ensure_ascii=False)) > 3000:
                object_context_compact = {
                    "current_generated_object": dict(object_context_compact.get("current_generated_object", {}) or {}),
                    "already_generated_objects": list(object_context_compact.get("already_generated_objects", []) or [])[:16],
                    "all_generated_objects": list(object_context_compact.get("all_generated_objects", []) or [])[:20],
                    "generated_name_map": dict(object_context_compact.get("generated_name_map", {}) or {}),
                }
        except Exception:
            object_context_compact = {
                "current_generated_object": dict((object_context or {}).get("current_generated_object", {}) or {}),
                "already_generated_objects": list((object_context or {}).get("already_generated_objects", []) or [])[:16],
            }
        depth_order_prior = self._infer_x_depth_order_prior_with_vlm(
            object_name=str(object_name),
            relation_hint=str(relation_hint),
            scene_graph=dict(scene_graph_compact or {}),
            object_context=dict(object_context_compact or {}),
            input_rgb_u8=input_rgb_u8,
            overlay_rgb_u8=overlay_rgb_u8,
            view_images=dict(view_images or {}),
            mode=mode,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=min(60.0, max(15.0, float(timeout) * 0.7)),
        )
        target_relation_context = self._build_target_relation_context(
            object_name=str(object_name),
            scene_graph=dict(scene_graph_compact or {}),
            object_context=dict(object_context_compact or {}),
            max_items=20,
        )

        if language == "zh":
            prompt = (
                "你是3D位姿合理性判别器。请根据多视角渲染图 + 输入图overlay + 关系描述 + 规则诊断，判断目标位姿是否物理合理。"
                "请先执行“x轴深度排序分析”，再据此判断物理合理性、pixel-align和移动方向。"
                "你必须显式利用 video understanding 中与目标物体相关的关系/事件/约束/远近排序(例如“夹子夹住杯子”)来判断当前位置关系是否合理。"
                "注意：与当前目标物体无直接关系的关系描述，只能作为背景，不得作为硬约束。"
                "你还必须判断目标物体在视角中是否居中：重点看 left/right/up/down 视角，结合规则诊断中的世界坐标x，判断是否接近 x=0（可视作更接近图像中心）。"
                "注意：AABB碰撞只作为粗规则，必须结合视觉内容再做碰撞判断。"
                "在给出是否通过和移动方向前，你必须先输出“输入图目标 vs 已生成物体”的相对位置分析结果（pre_adjust_relative_analysis）。"
                "请输出JSON："
                '{'
                '"is_plausible":bool,'
                '"is_pixel_aligned":bool,'
                '"is_view_centered":bool,'
                '"off_center_allowed_by_physics":bool,'
                '"relation_consistent_with_video":bool,'
                '"relation_violation_reasons":[string],'
                '"pre_adjust_relative_analysis":[{"other_object":string,"input_horizontal":"left|right|overlap|unknown","input_vertical":"above|below|overlap|unknown","input_depth":"front|behind|same_depth|unknown","candidate_horizontal":"left|right|overlap|unknown","candidate_vertical":"above|below|overlap|unknown","candidate_depth":"front|behind|same_depth|unknown","consistent":bool,"suggested_x_direction":"positive|negative|none","evidence":string}],'
                '"pre_adjust_analysis_summary":string,'
                '"suggested_x_direction_pre_adjust":"positive|negative|none",'
                '"suggested_x_direction_relation":"positive|negative|none",'
                '"centering_score":0-1,'
                '"has_collision_visual":bool,'
                '"collision_severity":"none|slight|severe|unknown",'
                '"contact_relation_allows_slight_collision":bool,'
                '"collision_with":[string],'
                '"suggested_x_direction":"positive|negative|none",'
                '"suggested_x_direction_centering":"positive|negative|none",'
                '"confidence":0-1,'
                '"reason_codes":[string],'
                '"summary":"一句中文结论"'
                '}'
                f"\n目标物体: {target_name}"
                f"\n关系提示: {str(relation_hint or '').strip()}"
                f"\n时序图摘要: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\n物体上下文(当前刚生成目标 + 已生成物体): {json.dumps(object_context_compact, ensure_ascii=False)}"
                f"\n目标相关视频关系上下文(用于关系一致性判断): {json.dumps(target_relation_context, ensure_ascii=False)}"
                f"\n前置x轴深度排序结果(先验): {json.dumps(depth_order_prior, ensure_ascii=False)}"
                f"\n规则诊断(AABB/x范围): {json.dumps(rule_diagnostics, ensure_ascii=False)}"
                "\n方向定义：positive 表示世界坐标系 +X，negative 表示世界坐标系 -X。"
                "\n请重点判断：当前目标物体与“已生成物体”之间的相对关系是否合理。"
                "\n如果 video understanding 指示当前关系被破坏（如应接触但分离、应前后顺序但被反转），请设置 relation_consistent_with_video=false，并给出 suggested_x_direction_relation。"
                "\n若 video understanding 显示目标与某物体存在接触关系（如夹持/切割/接触），允许轻微碰撞：设置 contact_relation_allows_slight_collision=true 且 collision_severity=slight。"
                "\n但若为明显穿透/深度重叠，应设置 collision_severity=severe，并不得按轻微碰撞放行。"
                "\n如果目标在 video understanding 中没有直接关系，请不要套用其他物体的关系链（例如“夹子-白杯”）来约束当前目标。"
                "\n这种情况下，应重点检查当前目标与已生成邻近物体的远近关系、深度顺序、碰撞/穿透。"
                "\n如果主要问题是碰撞且目标应远离他物，请给出需要移动的方向。"
                "\n如果物理基本合理但目标在多视角中明显贴边，请优先给出让目标更靠近图像中心的x方向建议。"
                "\n如果为了维持物理合理性(如避免碰撞/满足关系)必须偏离中心，请设置 off_center_allowed_by_physics=true；此时即便不居中也可判通过。"
            )
            system_content = "只返回合法JSON。不要输出额外文字。"
        else:
            prompt = (
                "You are a 3D pose plausibility evaluator. Use multiview renders + input overlay + relation hint + rule diagnostics to judge physical plausibility."
                "First perform x-axis depth-order reasoning, then use it to judge physical plausibility, pixel alignment, and movement direction."
                "You must explicitly use video-understanding relations/events/constraints/depth-order tied to the target object (e.g., 'gripper holds cup') to judge relational consistency."
                "Important: relations not directly involving the current target are background context only, not hard constraints."
                "You must also judge centering with emphasis on left/right/up/down views, and use world-axis x from rule diagnostics: x near 0 means closer to image center."
                "Important: AABB collision is coarse and must be validated by visual evidence."
                "Before pass/fail and movement direction, you must first output a pre-adjust relative-position analysis between target-in-input-image and already-generated objects (pre_adjust_relative_analysis)."
                "Return JSON only: "
                '{'
                '"is_plausible":bool,'
                '"is_pixel_aligned":bool,'
                '"is_view_centered":bool,'
                '"off_center_allowed_by_physics":bool,'
                '"relation_consistent_with_video":bool,'
                '"relation_violation_reasons":[string],'
                '"pre_adjust_relative_analysis":[{"other_object":string,"input_horizontal":"left|right|overlap|unknown","input_vertical":"above|below|overlap|unknown","input_depth":"front|behind|same_depth|unknown","candidate_horizontal":"left|right|overlap|unknown","candidate_vertical":"above|below|overlap|unknown","candidate_depth":"front|behind|same_depth|unknown","consistent":bool,"suggested_x_direction":"positive|negative|none","evidence":string}],'
                '"pre_adjust_analysis_summary":string,'
                '"suggested_x_direction_pre_adjust":"positive|negative|none",'
                '"suggested_x_direction_relation":"positive|negative|none",'
                '"centering_score":0-1,'
                '"has_collision_visual":bool,'
                '"collision_severity":"none|slight|severe|unknown",'
                '"contact_relation_allows_slight_collision":bool,'
                '"collision_with":[string],'
                '"suggested_x_direction":"positive|negative|none",'
                '"suggested_x_direction_centering":"positive|negative|none",'
                '"confidence":0-1,'
                '"reason_codes":[string],'
                '"summary":"one-sentence conclusion"'
                '}'
                f"\nTarget object: {target_name}"
                f"\nRelation hint: {str(relation_hint or '').strip()}"
                f"\nTSG summary: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\nObject context (current generated target + already generated objects): {json.dumps(object_context_compact, ensure_ascii=False)}"
                f"\nTarget-related video-relation context (must be used): {json.dumps(target_relation_context, ensure_ascii=False)}"
                f"\nPre-step x-axis depth-order prior: {json.dumps(depth_order_prior, ensure_ascii=False)}"
                f"\nRule diagnostics (AABB/x-range): {json.dumps(rule_diagnostics, ensure_ascii=False)}"
                "\nDirection convention: positive means world +X, negative means world -X."
                "\nFocus on whether the current target object relation to already generated objects is physically plausible."
                "\nIf video-understanding relations are violated (e.g., should-contact but separated, expected depth order inverted), set relation_consistent_with_video=false and provide suggested_x_direction_relation."
                "\nIf video-understanding indicates contact relations (e.g., grasping/cutting/contact), slight collision can be allowed: set contact_relation_allows_slight_collision=true and collision_severity=slight."
                "\nBut obvious penetration/deep overlap must be treated as collision_severity=severe and must not be waived as slight contact."
                "\nIf the target has no direct relation in video understanding, do NOT enforce relation chains of other objects (e.g., 'gripper-white-cup') on this target."
                "\nIn that case, prioritize target-vs-neighbor distance/proximity consistency, depth ordering, and collision/penetration checks."
                "\nIf physics is mostly plausible but the target is near view borders, suggest x-direction to move it toward image center."
                "\nIf staying near image center conflicts with physical plausibility (e.g., collision avoidance/relation constraints), set off_center_allowed_by_physics=true; then off-center can still pass."
            )
            system_content = "Return valid JSON only."

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if input_rgb_u8 is not None:
            input_url = self._to_data_url(input_rgb_u8, jpeg_quality=85)
            if input_url:
                content.append({"type": "text", "text": "[input_image]"})
                content.append({"type": "image_url", "image_url": {"url": input_url, "detail": "high"}})
        if overlay_rgb_u8 is not None:
            overlay_url = self._to_data_url(overlay_rgb_u8, jpeg_quality=80)
            if overlay_url:
                content.append({"type": "text", "text": "[input_overlay]"})
                content.append({"type": "image_url", "image_url": {"url": overlay_url, "detail": "high"}})
        for k in sorted(view_images.keys()):
            arr = view_images.get(k)
            if arr is None:
                continue
            u = self._to_data_url(arr, jpeg_quality=75)
            if not u:
                continue
            content.append({"type": "text", "text": f"[view:{k}]"})
            content.append({"type": "image_url", "image_url": {"url": u, "detail": "low"}})

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": content}]
        self._debug_print_json(
            "[VLM Assistor][Agent2][PoseEval] request=",
            {
                "mode": mode,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "target_object": target_name,
                "relation_hint": str(relation_hint or "").strip(),
                "object_context": dict(object_context_compact or {}),
                "target_relation_context": dict(target_relation_context or {}),
                "depth_order_prior": dict(depth_order_prior or {}),
                "prompt": prompt,
                "rule_diagnostics": dict(rule_diagnostics or {}),
                "view_keys": sorted([str(k) for k in list(view_images.keys())]),
                "has_input": bool(input_rgb_u8 is not None),
                "has_overlay": bool(overlay_rgb_u8 is not None),
                "messages_preview": self._debug_sanitize_messages(messages),
            },
        )
        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            payload["messages"] = messages
            payload["response_format"] = {"type": "json_object"}
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            payload["messages"] = self._convert_system_to_user_prefix(messages)
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )

        if not isinstance(resp, dict):
            self._debug_print_json(
                "[VLM Assistor][Agent2][PoseEval] response_error=",
                {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "resp_type": str(type(resp)),
                },
            )
            return {
                "ok": False,
                "evaluated": False,
                "reason": f"{provider}_http_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
            print("[VLM Assistor][Agent2][PoseEval] raw_response_text:", str(text)[:2000])
        except Exception as exc:
            self._debug_print_json(
                "[VLM Assistor][Agent2][PoseEval] response_parse_error=",
                {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "error": f"{type(exc).__name__}:{exc}",
                },
            )
            return {
                "ok": False,
                "evaluated": False,
                "reason": f"invalid_response:{type(exc).__name__}:{exc}",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }
        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            self._debug_print_json(
                "[VLM Assistor][Agent2][PoseEval] response_json_parse_failed=",
                {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "raw_text_head": str(text)[:400],
                },
            )
            return {
                "ok": False,
                "evaluated": False,
                "reason": "pose_eval_json_parse_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "raw_text": str(text)[:300],
            }
        self._debug_print_json(
            "[VLM Assistor][Agent2][PoseEval] parsed_json=",
            parsed,
        )
        direction = self._normalize_x_direction(
            parsed.get(
                "suggested_x_direction",
                parsed.get("direction", parsed.get("x_direction", "none")),
            )
        )
        has_collision_visual = bool(
            parsed.get("has_collision_visual", parsed.get("has_collision", parsed.get("collision", False)))
        )
        collision_severity_raw = str(
            parsed.get("collision_severity", parsed.get("collision_level", parsed.get("collision_magnitude", "unknown")))
        ).strip().lower()
        if collision_severity_raw in {"none", "no", "no_collision", "clean"}:
            collision_severity = "none"
        elif collision_severity_raw in {"slight", "minor", "small", "mild", "light", "tiny", "slight_penetration"}:
            collision_severity = "slight"
        elif collision_severity_raw in {"severe", "major", "heavy", "deep", "hard", "large", "strong", "penetration"}:
            collision_severity = "severe"
        else:
            collision_severity = "unknown"
        if not bool(has_collision_visual):
            collision_severity = "none"
        contact_relation_allows_slight_collision = bool(
            parsed.get(
                "contact_relation_allows_slight_collision",
                parsed.get(
                    "allow_slight_collision_under_contact",
                    parsed.get("contact_collision_allowed", False),
                ),
            )
        )
        is_pixel_aligned = bool(
            parsed.get(
                "is_pixel_aligned",
                parsed.get("pixel_aligned", parsed.get("pixel_align_ok", True)),
            )
        )
        is_view_centered = bool(
            parsed.get(
                "is_view_centered",
                parsed.get("view_centered", parsed.get("centered", True)),
            )
        )
        off_center_allowed_by_physics = bool(
            parsed.get(
                "off_center_allowed_by_physics",
                parsed.get("centering_relaxed_by_physics", parsed.get("off_center_but_physical", False)),
            )
        )
        dir_centering = self._normalize_x_direction(
            parsed.get(
                "suggested_x_direction_centering",
                parsed.get("centering_direction", "none"),
            )
        )
        dir_relation = self._normalize_x_direction(
            parsed.get(
                "suggested_x_direction_relation",
                parsed.get("relation_based_x_direction", parsed.get("relation_direction", "none")),
            )
        )
        dir_pre_adjust = self._normalize_x_direction(
            parsed.get(
                "suggested_x_direction_pre_adjust",
                parsed.get("suggested_x_direction_from_relative_analysis", "none"),
            )
        )
        relation_consistent = bool(
            parsed.get(
                "relation_consistent_with_video",
                parsed.get("relation_consistent", parsed.get("is_relation_consistent", True)),
            )
        )
        relation_violation_reasons = [
            str(x).strip()
            for x in list(parsed.get("relation_violation_reasons", parsed.get("relation_violations", [])) or [])
            if str(x).strip()
        ]
        pre_adjust_raw = parsed.get("pre_adjust_relative_analysis", parsed.get("relative_position_analysis", []))
        pre_adjust_analysis: list[dict[str, Any]] = []
        if isinstance(pre_adjust_raw, list):
            for it in pre_adjust_raw:
                if not isinstance(it, dict):
                    continue
                other = str(it.get("other_object", it.get("object", it.get("name", "")))).strip()
                if not other:
                    continue
                sug_dir = self._normalize_x_direction(
                    it.get(
                        "suggested_x_direction",
                        it.get("suggested_x_direction_pre_adjust", it.get("suggested_direction", "none")),
                    )
                )
                pre_adjust_analysis.append(
                    {
                        "other_object": other,
                        "input_horizontal": self._coerce_relative_axis(
                            it.get("input_horizontal", it.get("expected_horizontal", "unknown")),
                            axis="horizontal",
                        ),
                        "input_vertical": self._coerce_relative_axis(
                            it.get("input_vertical", it.get("expected_vertical", "unknown")),
                            axis="vertical",
                        ),
                        "input_depth": self._coerce_relative_axis(
                            it.get("input_depth", it.get("expected_depth", "unknown")),
                            axis="depth",
                        ),
                        "candidate_horizontal": self._coerce_relative_axis(
                            it.get("candidate_horizontal", it.get("current_horizontal", "unknown")),
                            axis="horizontal",
                        ),
                        "candidate_vertical": self._coerce_relative_axis(
                            it.get("candidate_vertical", it.get("current_vertical", "unknown")),
                            axis="vertical",
                        ),
                        "candidate_depth": self._coerce_relative_axis(
                            it.get("candidate_depth", it.get("current_depth", "unknown")),
                            axis="depth",
                        ),
                        "consistent": bool(it.get("consistent", it.get("is_consistent", True))),
                        "suggested_x_direction": sug_dir,
                        "evidence": str(it.get("evidence", it.get("notes", ""))).strip(),
                    }
                )
                if len(pre_adjust_analysis) >= 16:
                    break
        pre_adjust_analysis_summary = str(
            parsed.get("pre_adjust_analysis_summary", parsed.get("relative_analysis_summary", ""))
        ).strip()
        if direction == "none" and dir_relation != "none":
            direction = dir_relation
        if direction == "none" and dir_pre_adjust != "none":
            direction = dir_pre_adjust
        if direction == "none":
            for it in pre_adjust_analysis:
                sug = self._normalize_x_direction(it.get("suggested_x_direction", "none"))
                if sug != "none":
                    direction = sug
                    break
        if direction == "none" and (not is_view_centered) and (not off_center_allowed_by_physics) and dir_centering != "none":
            direction = dir_centering
        out = {
            "ok": bool(parsed.get("is_plausible", parsed.get("ok", False))),
            "evaluated": True,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "is_plausible": bool(parsed.get("is_plausible", parsed.get("ok", False))),
            "is_pixel_aligned": bool(is_pixel_aligned),
            "is_view_centered": bool(is_view_centered),
            "off_center_allowed_by_physics": bool(off_center_allowed_by_physics),
            "centering_score": float(max(0.0, min(1.0, float(parsed.get("centering_score", 0.5))))),
            "has_collision_visual": bool(has_collision_visual),
            "collision_severity": str(collision_severity),
            "contact_relation_allows_slight_collision": bool(contact_relation_allows_slight_collision),
            "collision_with": [str(x) for x in list(parsed.get("collision_with", [])) if str(x).strip()],
            "suggested_x_direction": direction,
            "suggested_x_direction_relation": dir_relation,
            "suggested_x_direction_pre_adjust": dir_pre_adjust,
            "suggested_x_direction_centering": dir_centering,
            "relation_consistent_with_video": bool(relation_consistent),
            "relation_violation_reasons": relation_violation_reasons,
            "pre_adjust_relative_analysis": pre_adjust_analysis,
            "pre_adjust_analysis_summary": pre_adjust_analysis_summary,
            "target_relation_context": dict(target_relation_context or {}),
            "depth_order_prior": dict(depth_order_prior or {}),
            "confidence": float(max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))),
            "reason_codes": [str(x) for x in list(parsed.get("reason_codes", [])) if str(x).strip()],
            "summary": str(parsed.get("summary", "")).strip(),
            "raw": parsed,
        }
        self._debug_print_json(
            "[VLM Assistor][Agent2][PoseEval] normalized_result=",
            out,
        )
        return out

    def evaluate_4d_frame_validator_with_vlm(
        self,
        *,
        object_text: str,
        relation_hint: str,
        scene_graph: dict[str, Any] | None,
        rendered_frames: list[dict[str, Any]],
        reference_input_rgb_u8: np.ndarray | None,
        backend_mode: str | None = None,
        timeout: float = 150.0,
        max_images: int = 49,
    ) -> dict[str, Any]:
        if not isinstance(rendered_frames, list) or len(rendered_frames) <= 0:
            return {"ok": False, "evaluated": False, "reason": "missing_rendered_frames"}
        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"

        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
        if not base_url:
            return {"ok": False, "evaluated": False, "reason": "vlm_base_url_missing"}

        frame_packs = [dict(x) for x in rendered_frames if isinstance(x, dict)]
        if not frame_packs:
            return {"ok": False, "evaluated": False, "reason": "empty_rendered_frames"}
        frame_packs.sort(key=lambda x: int(x.get("frame_idx", -1)))

        scene_graph_compact = dict(scene_graph or {})
        try:
            if len(json.dumps(scene_graph_compact, ensure_ascii=False)) > 5000:
                scene_graph_compact = {
                    "objects": list(scene_graph_compact.get("objects", []) or [])[:20],
                    "relations": list(scene_graph_compact.get("relations", []) or [])[:28],
                    "events": list(scene_graph_compact.get("events", []) or [])[:20],
                    "constraints": list(scene_graph_compact.get("constraints", []) or [])[:18],
                    "x_depth_order_timeline": list(scene_graph_compact.get("x_depth_order_timeline", []) or [])[:10],
                    "relative_position_timeline": list(scene_graph_compact.get("relative_position_timeline", []) or [])[:10],
                }
        except Exception:
            scene_graph_compact = {
                "objects": list((scene_graph or {}).get("objects", []) or [])[:20],
                "relations": list((scene_graph or {}).get("relations", []) or [])[:28],
                "events": list((scene_graph or {}).get("events", []) or [])[:20],
                "constraints": list((scene_graph or {}).get("constraints", []) or [])[:18],
                "x_depth_order_timeline": list((scene_graph or {}).get("x_depth_order_timeline", []) or [])[:10],
                "relative_position_timeline": list((scene_graph or {}).get("relative_position_timeline", []) or [])[:10],
            }

        view_order = ["down", "up", "left", "front"]
        budget = max(5, min(49, int(max_images)))
        ref_cost = 1 if reference_input_rgb_u8 is not None else 0
        per_frame_cost = len(view_order)

        target_pack = dict(frame_packs[-1])
        history = [dict(x) for x in frame_packs[:-1]]
        history_budget = max(0, budget - ref_cost - per_frame_cost)
        max_hist_frames = history_budget // per_frame_cost
        if len(history) > max_hist_frames:
            if max_hist_frames <= 0:
                history = []
            else:
                pick = np.linspace(0, len(history) - 1, num=max_hist_frames, dtype=np.int64).tolist()
                keep: list[int] = []
                seen: set[int] = set()
                for i in pick:
                    ii = int(i)
                    if ii in seen:
                        continue
                    seen.add(ii)
                    keep.append(ii)
                history = [history[i] for i in keep]
        used_frames = history + [target_pack]

        frame_meta: list[dict[str, Any]] = []
        for i, pack in enumerate(used_frames):
            frame_meta.append(
                {
                    "frame_idx": int(pack.get("frame_idx", -1)),
                    "frame_token": str(pack.get("frame_token", "")),
                    "role": ("target" if i == len(used_frames) - 1 else "history"),
                    "views": list(view_order),
                }
            )

        language = self._language_for_model(model)
        target_name = self._to_language_object_name(str(object_text or "").strip(), language)
        if language == "zh":
            prompt = (
                "你是4D生成结果评测器。下面给你的图片中："
                "前面若干帧是历史生成结果，最后一帧是待评测生成帧(target)。"
                "另外给你一张 target 帧对应的原始输入图(reference_input，已做白底合成)。"
                "你必须结合 video understanding 的时序交互语义(scene graph)进行评测。"
                "评测策略："
                "1) 若当前帧主要问题是几何形变/穿透/分离/相对位置错误，且目标物体在输入图中大部分可见，"
                "应优先视为“当前帧3D质量问题（可通过 Run Full 3D Gen Pipeline 修复）”，不要默认归因到记忆问题。"
                "2) 只有在存在明确历史帧证据且表现为时序漂移/历史一致性断裂，或关键目标大范围不可见时，"
                "才倾向记忆相关问题。"
                "3) 如果 frame_meta 只有 target（没有 history），不要把“缺少时序上下文”直接等价为记忆丢失。"
                "请从以下维度打分(0-100)："
                "input_alignment_score（与输入图对齐程度）、"
                "interaction_semantic_alignment_score（与交互语义对齐程度）、"
                "physical_plausibility_score（物理合理性）、"
                "deformation_reasonableness_score（形变合理性）、"
                "temporal_consistency_4d_score（4D时间一致性）、"
                "clarity_3d_score（3D清晰程度）。"
                "再给 overall_score(0-100) 与总结。"
                "仅输出 JSON："
                "{"
                '"overall_score":int,'
                '"input_alignment_score":int,'
                '"interaction_semantic_alignment_score":int,'
                '"physical_plausibility_score":int,'
                '"deformation_reasonableness_score":int,'
                '"temporal_consistency_4d_score":int,'
                '"clarity_3d_score":int,'
                '"key_issues":[string],'
                '"strengths":[string],'
                '"summary":"一句中文总结",'
                '"confidence":0-1'
                "}"
                f"\n目标物体文本: {target_name}"
                f"\n关系提示: {str(relation_hint or '').strip()}"
                f"\nscene_graph(来自video understanding): {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\n输入帧顺序说明: {json.dumps(frame_meta, ensure_ascii=False)}"
                "\n注意：最后一帧(role=target)是必须重点评测对象。"
            )
            system_content = "只返回合法JSON。不要输出额外解释。"
        else:
            prompt = (
                "You are a 4D generation quality evaluator."
                " The provided generated images are ordered in time: history frames first, and the last frame is the target frame to evaluate."
                " You are also given one reference_input image for the target frame (RGBA composited over white)."
                " You must use video-understanding scene-graph semantics to evaluate."
                " Evaluation policy:"
                " 1) If the main errors are geometry deformation / penetration / separation / wrong relative pose in the target frame,"
                " and target objects are mostly visible in input, treat this primarily as current-frame 3D quality issue"
                " (repairable by Run Full 3D Gen Pipeline), not memory-loss by default."
                " 2) Prefer memory-related interpretation only when there is explicit history-frame evidence of temporal drift/history-break,"
                " or key targets are mostly occluded."
                " 3) If frame_meta is target-only (no history), do not equate missing temporal context with memory loss."
                " Score each dimension (0-100): "
                "input_alignment_score, interaction_semantic_alignment_score, physical_plausibility_score, "
                "deformation_reasonableness_score, temporal_consistency_4d_score, clarity_3d_score."
                " Then provide overall_score (0-100) and short summary."
                " Return JSON only: "
                "{"
                '"overall_score":int,'
                '"input_alignment_score":int,'
                '"interaction_semantic_alignment_score":int,'
                '"physical_plausibility_score":int,'
                '"deformation_reasonableness_score":int,'
                '"temporal_consistency_4d_score":int,'
                '"clarity_3d_score":int,'
                '"key_issues":[string],'
                '"strengths":[string],'
                '"summary":"one-line summary",'
                '"confidence":0-1'
                "}"
                f"\nTarget object text: {target_name}"
                f"\nRelation hint: {str(relation_hint or '').strip()}"
                f"\nscene_graph(from video understanding): {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\nFrame order meta: {json.dumps(frame_meta, ensure_ascii=False)}"
                "\nImportant: the last frame (role=target) is the primary evaluation target."
            )
            system_content = "Return valid JSON only. No extra text."

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if reference_input_rgb_u8 is not None:
            ref_url = self._to_data_url(reference_input_rgb_u8, jpeg_quality=85)
            if ref_url:
                content.append({"type": "text", "text": "[reference_input_target_frame]"})
                content.append({"type": "image_url", "image_url": {"url": ref_url, "detail": "high"}})

        for i, pack in enumerate(used_frames):
            role = "target" if i == len(used_frames) - 1 else "history"
            tok = str(pack.get("frame_token", ""))
            idx = int(pack.get("frame_idx", -1))
            content.append({"type": "text", "text": f"[generated_frame role={role} idx={idx} token={tok}]"})
            views = dict(pack.get("view_images", {}) or {})
            for vn in view_order:
                arr = views.get(vn, None)
                if arr is None:
                    continue
                u = self._to_data_url(arr, jpeg_quality=74)
                if not u:
                    continue
                content.append({"type": "text", "text": f"[view:{vn}]"})
                content.append({"type": "image_url", "image_url": {"url": u, "detail": "low"}})

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": content}]
        self._debug_print_json(
            "[VLM Assistor][Agent-4][Validator] request=",
            {
                "mode": mode,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "target_object": target_name,
                "relation_hint": str(relation_hint or "").strip(),
                "scene_graph_summary": {
                    "objects": len(list(scene_graph_compact.get("objects", []) or [])),
                    "relations": len(list(scene_graph_compact.get("relations", []) or [])),
                    "events": len(list(scene_graph_compact.get("events", []) or [])),
                    "constraints": len(list(scene_graph_compact.get("constraints", []) or [])),
                },
                "used_frames": frame_meta,
                "reference_input_used": bool(reference_input_rgb_u8 is not None),
                "messages_preview": self._debug_sanitize_messages(messages),
            },
        )

        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            payload["messages"] = messages
            payload["response_format"] = {"type": "json_object"}
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            payload["messages"] = self._convert_system_to_user_prefix(messages)
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )

        if not isinstance(resp, dict):
            return {
                "ok": False,
                "evaluated": False,
                "reason": f"{provider}_http_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
            print("[VLM Assistor][Agent-4][Validator] raw_response_text:", str(text)[:2000])
        except Exception as exc:
            return {
                "ok": False,
                "evaluated": False,
                "reason": f"invalid_response:{type(exc).__name__}:{exc}",
                "provider": provider,
                "model": model,
                "base_url": base_url,
            }
        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            return {
                "ok": False,
                "evaluated": False,
                "reason": "validator_json_parse_failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "raw_text": str(text)[:500],
            }

        def _score(*keys: str, default: int = 0) -> int:
            for k in keys:
                if k not in parsed:
                    continue
                try:
                    v = int(round(float(parsed.get(k))))
                    return max(0, min(100, v))
                except Exception:
                    continue
            return max(0, min(100, int(default)))

        dim_scores = {
            "input_alignment_score": _score(
                "input_alignment_score",
                "alignment_with_input_score",
                "input_fidelity_score",
                default=0,
            ),
            "interaction_semantic_alignment_score": _score(
                "interaction_semantic_alignment_score",
                "semantic_alignment_score",
                "interaction_alignment_score",
                default=0,
            ),
            "physical_plausibility_score": _score(
                "physical_plausibility_score",
                "physics_score",
                default=0,
            ),
            "deformation_reasonableness_score": _score(
                "deformation_reasonableness_score",
                "deformation_score",
                "shape_deformation_score",
                default=0,
            ),
            "temporal_consistency_4d_score": _score(
                "temporal_consistency_4d_score",
                "temporal_consistency_score",
                "consistency_4d_score",
                default=0,
            ),
            "clarity_3d_score": _score(
                "clarity_3d_score",
                "3d_clarity_score",
                "clarity_score",
                default=0,
            ),
        }
        avg_score = int(round(float(np.mean(list(dim_scores.values()) or [0.0]))))
        overall_score = _score(
            "overall_score",
            "final_score",
            "score",
            default=avg_score,
        )
        raw_conf = parsed.get("confidence", 0.5)
        try:
            confidence = float(raw_conf)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        def _norm_str_list(raw: Any) -> list[str]:
            if isinstance(raw, list):
                out = [str(x).strip() for x in raw if str(x).strip()]
                return out[:16]
            s = str(raw or "").strip()
            return [s] if s else []

        out = {
            "ok": bool(overall_score >= 60),
            "evaluated": True,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "overall_score": int(overall_score),
            "input_alignment_score": int(dim_scores["input_alignment_score"]),
            "interaction_semantic_alignment_score": int(dim_scores["interaction_semantic_alignment_score"]),
            "physical_plausibility_score": int(dim_scores["physical_plausibility_score"]),
            "deformation_reasonableness_score": int(dim_scores["deformation_reasonableness_score"]),
            "temporal_consistency_4d_score": int(dim_scores["temporal_consistency_4d_score"]),
            "clarity_3d_score": int(dim_scores["clarity_3d_score"]),
            "key_issues": _norm_str_list(parsed.get("key_issues", parsed.get("issues", []))),
            "strengths": _norm_str_list(parsed.get("strengths", [])),
            "summary": str(parsed.get("summary", parsed.get("conclusion", ""))).strip(),
            "confidence": float(confidence),
            "used_frames": frame_meta,
            "reference_input_used": bool(reference_input_rgb_u8 is not None),
            "raw": parsed,
        }
        self._debug_print_json(
            "[VLM Assistor][Agent-4][Validator] normalized_result=",
            out,
        )
        return out

    @staticmethod
    def _extract_failure_phenomena_from_text(text: Any) -> list[str]:
        s = str(text or "").strip().lower()
        if not s:
            return []
        tags: list[str] = []
        drift_keys = (
            "position drift",
            "drift",
            "misalignment",
            "misaligned",
            "offset",
            "shifted",
            "漂移",
            "错位",
            "偏移",
            "偏离",
            "跑偏",
            "位置不一致",
        )
        penetration_keys = (
            "penetration",
            "interpenetration",
            "deep overlap",
            "hard overlap",
            "穿透",
            "穿模",
            "严重重叠",
            "深度重叠",
        )
        separation_keys = (
            "separation",
            "detached",
            "disconnected",
            "too far apart",
            "gap too large",
            "分离",
            "分开",
            "脱离",
            "远离",
            "间隙过大",
            "距离过大",
        )
        if any(k in s for k in drift_keys):
            tags.append("position_drift")
        if any(k in s for k in penetration_keys):
            tags.append("penetration")
        if any(k in s for k in separation_keys):
            tags.append("separation")
        return tags

    @classmethod
    def _normalize_failure_phenomena(cls, raw: Any) -> list[str]:
        texts: list[str] = []
        if isinstance(raw, list):
            texts = [str(x).strip() for x in raw if str(x).strip()]
        elif isinstance(raw, str):
            s = str(raw).strip()
            if s:
                texts = [x.strip() for x in re.split(r"[,\n;/|]+", s) if x.strip()]
        else:
            s = str(raw or "").strip()
            if s:
                texts = [s]

        out: list[str] = []
        for t in texts:
            for tag in cls._extract_failure_phenomena_from_text(t):
                if tag not in out:
                    out.append(tag)
        if not out and texts:
            out.append("other")
        return out[:4]

    @classmethod
    def _collect_failure_phenomena_from_eval(cls, eval_pack: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for key in ("summary", "reason", "conclusion"):
            v = str(eval_pack.get(key, "")).strip()
            if v:
                texts.append(v)
        for key in ("key_issues", "issues", "evidence"):
            rows = list(eval_pack.get(key, []) or [])
            texts.extend([str(x).strip() for x in rows if str(x).strip()])
        for key in ("collision_severity", "collision_level"):
            v = str(eval_pack.get(key, "")).strip()
            if v:
                texts.append(v)

        out: list[str] = []
        for t in texts:
            for tag in cls._extract_failure_phenomena_from_text(t):
                if tag not in out:
                    out.append(tag)
        if not out and texts:
            out.append("other")
        return out[:4]

    @staticmethod
    def _heuristic_4d_issue_diagnosis(
        *,
        validator_eval: dict[str, Any] | None,
        rendered_frames: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        eval_pack = dict(validator_eval or {})

        def _score(key: str, default: int = 0) -> int:
            try:
                return max(0, min(100, int(round(float(eval_pack.get(key, default))))))
            except Exception:
                return int(default)

        s_input = _score("input_alignment_score")
        s_inter = _score("interaction_semantic_alignment_score")
        s_phys = _score("physical_plausibility_score")
        s_def = _score("deformation_reasonableness_score")
        s_temp = _score("temporal_consistency_4d_score")
        s_clarity = _score("clarity_3d_score")
        s_overall = _score("overall_score")
        current_quality = int(round(float(np.mean([s_input, s_phys, s_def, s_clarity]))))
        failure_phenomena = VLMBackendClient._collect_failure_phenomena_from_eval(eval_pack)
        phenomenon_set = set(failure_phenomena)
        has_drift = bool("position_drift" in phenomenon_set)
        has_structural = bool(phenomenon_set.intersection({"position_drift", "penetration", "separation"}))

        root_cause = "mixed"
        if s_temp <= max(0, current_quality - 12):
            root_cause = "past_memory_loss"
        elif current_quality <= max(0, s_temp - 12):
            root_cause = "current_frame_3d_quality"
        elif s_overall >= 60:
            root_cause = "uncertain"
        if root_cause == "mixed" and has_drift and s_temp <= max(0, current_quality - 8):
            root_cause = "past_memory_loss"
        elif root_cause == "mixed" and (not has_drift) and has_structural and s_temp >= max(35, current_quality - 4):
            root_cause = "current_frame_3d_quality"

        issue_text = " ".join([str(x).strip().lower() for x in list(eval_pack.get("key_issues", []) or []) if str(x).strip()])
        occ_high_keys = ("heavy occlusion", "severe occlusion", "大遮挡", "严重遮挡")
        occ_mid_keys = ("occlusion", "遮挡")
        occlusion_level = "unknown"
        if any(k in issue_text for k in occ_high_keys):
            occlusion_level = "high"
        elif any(k in issue_text for k in occ_mid_keys):
            occlusion_level = "medium"
        elif issue_text:
            occlusion_level = "low"

        sam3d_repair_feasible = bool(
            root_cause == "current_frame_3d_quality"
            and occlusion_level in {"low", "medium"}
            and current_quality >= 35
        )
        if root_cause == "past_memory_loss" or occlusion_level == "high":
            sam3d_repair_feasible = False
        if has_drift and s_temp < 45:
            sam3d_repair_feasible = False

        sam3d_rationale = (
            "Target object appears mostly visible and failure is current-frame geometry/pose issue."
            if sam3d_repair_feasible
            else (
                "Likely memory loss or heavy/uncertain occlusion; current-frame SAM3D object-wise refresh is insufficient."
                if (root_cause in {"past_memory_loss", "mixed"} or occlusion_level in {"high", "unknown"})
                else "Current-frame information is insufficient for reliable SAM3D-only repair."
            )
        )

        action = "manual_review"
        if sam3d_repair_feasible:
            action = "run_full_3d_gen_pipeline"
        elif root_cause in {"past_memory_loss", "mixed"} or occlusion_level == "high":
            action = "regenerate_with_memory_frame"
        elif root_cause == "current_frame_3d_quality" and not sam3d_repair_feasible:
            action = "regenerate_with_memory_frame"

        frames = [dict(x) for x in list(rendered_frames or []) if isinstance(x, dict)]
        frames.sort(key=lambda x: int(x.get("frame_idx", -1)))
        target_idx = int(frames[-1].get("frame_idx", -1)) if frames else -1
        memory_candidates: list[dict[str, Any]] = []
        for pack in reversed(frames[:-1]):
            idx = int(pack.get("frame_idx", -1))
            tok = str(pack.get("frame_token", "")).strip()
            if idx < 0:
                continue
            memory_candidates.append(
                {
                    "frame_idx": int(idx),
                    "frame_token": str(tok),
                    "priority": len(memory_candidates) + 1,
                    "reason": (
                        "recent_history_frame_before_target"
                        if idx < target_idx
                        else "history_frame_candidate"
                    ),
                }
            )
            if len(memory_candidates) >= 3:
                break
        memory_frame_characteristics: list[str] = []
        if action == "regenerate_with_memory_frame":
            memory_frame_characteristics = [
                "Choose frames before visible drift/penetration starts.",
                "Prefer frames where target objects are mostly visible with low occlusion.",
                "Prefer frames where key contact/separation relations are physically plausible and stable.",
            ]

        reason = (
            "Temporal score is much lower than current-frame quality; likely memory drift."
            if root_cause == "past_memory_loss"
            else (
                "Current-frame geometry/appearance scores are lower than temporal score."
                if root_cause == "current_frame_3d_quality"
                else "Signals are mixed; need memory-aware regeneration or manual confirmation."
            )
        )
        if has_structural:
            reason = f"{reason} Detected phenomena={','.join(failure_phenomena)}."
        evidence = [
            f"overall={s_overall}",
            f"temporal={s_temp}",
            f"current_quality={current_quality}",
            f"occlusion={occlusion_level}",
            f"phenomena={','.join(failure_phenomena) if failure_phenomena else 'none'}",
        ]
        return {
            "ok": True,
            "evaluated": False,
            "source": "heuristic",
            "root_cause": root_cause,
            "confidence": 0.45,
            "failure_phenomena": list(failure_phenomena),
            "sam3d_repair_feasible": bool(sam3d_repair_feasible),
            "sam3d_repair_rationale": sam3d_rationale,
            "occlusion_level": occlusion_level,
            "action_recommendation": action,
            "reason": reason,
            "evidence": evidence,
            "object_visibility": [],
            "memory_frame_candidates": memory_candidates,
            "memory_frame_characteristics": memory_frame_characteristics,
            "pipeline_advice": (
                "Trigger Run Full 3D Gen Pipeline on current frame."
                if action == "run_full_3d_gen_pipeline"
                else ""
            ),
            "memory_advice": (
                "Select a memory frame and run model4dgen Generate Next Segment with memory for current frame."
                if action == "regenerate_with_memory_frame"
                else ""
            ),
        }

    def diagnose_4d_validator_issue_with_vlm(
        self,
        *,
        object_text: str,
        relation_hint: str,
        scene_graph: dict[str, Any] | None,
        rendered_frames: list[dict[str, Any]],
        reference_input_rgb_u8: np.ndarray | None,
        validator_eval: dict[str, Any] | None,
        backend_mode: str | None = None,
        timeout: float = 120.0,
        max_images: int = 20,
    ) -> dict[str, Any]:
        score_fallback = self._heuristic_4d_issue_diagnosis(
            validator_eval=validator_eval,
            rendered_frames=rendered_frames,
        )
        fallback = dict(score_fallback)
        if not isinstance(rendered_frames, list) or len(rendered_frames) <= 0:
            out = dict(fallback)
            out.update({"ok": False, "reason": "missing_rendered_frames"})
            return out

        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"
        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
        if not base_url:
            out = dict(fallback)
            out.update(
                {
                    "ok": False,
                    "reason": "vlm_base_url_missing",
                    "provider": provider,
                    "model": model,
                }
            )
            return out

        frame_packs = [dict(x) for x in rendered_frames if isinstance(x, dict)]
        frame_packs.sort(key=lambda x: int(x.get("frame_idx", -1)))
        if not frame_packs:
            out = dict(fallback)
            out.update({"ok": False, "reason": "empty_rendered_frames"})
            return out

        target_pack = dict(frame_packs[-1])
        hist = [dict(x) for x in frame_packs[:-1]]
        hist_keep = hist[-2:] if len(hist) > 2 else hist
        used_frames = hist_keep + [target_pack]
        has_history_frames = len(hist_keep) > 0
        view_order = ["down", "up", "left", "front"]
        frame_meta = [
            {
                "frame_idx": int(x.get("frame_idx", -1)),
                "frame_token": str(x.get("frame_token", "")),
                "role": ("target" if i == len(used_frames) - 1 else "history"),
            }
            for i, x in enumerate(used_frames)
        ]

        eval_compact = dict(validator_eval or {})
        eval_compact.pop("raw", None)
        scene_graph_compact = dict(scene_graph or {})
        try:
            if len(json.dumps(scene_graph_compact, ensure_ascii=False)) > 3000:
                scene_graph_compact = {
                    "objects": list(scene_graph_compact.get("objects", []) or [])[:20],
                    "relations": list(scene_graph_compact.get("relations", []) or [])[:24],
                    "events": list(scene_graph_compact.get("events", []) or [])[:16],
                    "constraints": list(scene_graph_compact.get("constraints", []) or [])[:12],
                    "x_depth_order_timeline": list(scene_graph_compact.get("x_depth_order_timeline", []) or [])[:8],
                    "relative_position_timeline": list(scene_graph_compact.get("relative_position_timeline", []) or [])[:8],
                }
        except Exception:
            scene_graph_compact = {}

        language = self._language_for_model(model)
        target_name = self._to_language_object_name(str(object_text or "").strip(), language)
        if language == "zh":
            prompt = (
                "你是4D结果修复诊断器。请判断当前不合理是来自："
                "A) 当前帧3D生成质量差（几何/形变/对齐问题）"
                "B) 过去记忆丢失（时序漂移/历史一致性断裂）"
                "C) 混合原因。\n"
                "请重点识别这三类现象：位置漂移(position_drift)、不合理穿透(penetration)、不合理分离(separation)。\n"
                "必须结合 scene_graph（来自video understanding agent）、当前输入帧(reference_input_target_frame)和目标帧多视角渲染来判断根因。\n"
                "决策策略："
                "1) 若主要问题是当前帧几何结构错误（形变、穿透、分离、相对位姿错误），"
                "且关键目标物体大部分可见，则优先建议 run_full_3d_gen_pipeline。"
                "这类错误通常可通过“基于输入图重新SAM3D生成 + Agent2相对位置调整”修复。\n"
                "2) 仅当存在明确时序证据（历史帧对比显示漂移/一致性断裂），或关键目标大部分不可见时，"
                "才建议 regenerate_with_memory_frame。\n"
                "请特别考虑：sam3d 只能在目标物体绝大部分可见（允许少量遮挡）时有效重建；"
                "若目标被大范围遮挡，sam3d重生通常无效，应改用 memory-frame + Generate Next Segment with memory。\n"
                "硬约束：如果 frame_meta 中没有 history 帧（只有 target），则 memory 路线不可执行，"
                "禁止推荐 regenerate_with_memory_frame，应推荐 run_full_3d_gen_pipeline（或 manual_review）。\n"
                "如果建议记忆修复，请给出“理想记忆帧应具备的特征”，用于选择memory frame。\n"
                "只输出JSON："
                "{"
                '"root_cause":"current_frame_3d_quality|past_memory_loss|mixed|uncertain",'
                '"confidence":0-1,'
                '"failure_phenomena":["position_drift|penetration|separation|other"],'
                '"sam3d_repair_feasible":true/false,'
                '"sam3d_repair_rationale":"一句话说明当前帧做sam3d逐物体重建+位置调整是否足够",'
                '"occlusion_level":"low|medium|high|unknown",'
                '"object_visibility":[{"object_name":string,"visibility":"mostly_visible|partially_visible|mostly_occluded|unknown","reason":string}],'
                '"action_recommendation":"run_full_3d_gen_pipeline|regenerate_with_memory_frame|manual_review",'
                '"reason":"一句话",'
                '"evidence":[string],'
                '"pipeline_advice":"若建议pipeline则写明触发 Run Full 3D Gen Pipeline",'
                '"memory_advice":"若建议记忆修复则写明选memory frame并执行Generate Next Segment with memory",'
                '"memory_frame_candidates":[{"frame_idx":int,"frame_token":string,"priority":int,"reason":string}],'
                '"memory_frame_characteristics":[string]'
                "}"
                f"\n目标物体: {target_name}"
                f"\n关系提示: {str(relation_hint or '').strip()}"
                f"\nscene_graph: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\nvalidator_eval: {json.dumps(eval_compact, ensure_ascii=False)}"
                f"\nframe_meta: {json.dumps(frame_meta, ensure_ascii=False)}"
            )
            system_content = "只返回合法JSON，不要解释。"
        else:
            prompt = (
                "You are a 4D repair diagnostician. Decide whether the current failure is caused by: "
                "A) current-frame 3D quality issue, "
                "B) past-memory loss / temporal drift, or "
                "C) mixed.\n"
                "Focus on these phenomena: position drift, unreasonable penetration, and unreasonable separation.\n"
                "You must use scene_graph (from the video understanding agent), the current input frame, and target-frame multiview renders as evidence.\n"
                "Decision policy:"
                " 1) If primary failures are current-frame geometry errors (deformation/penetration/separation/wrong relative pose)"
                " and key targets are mostly visible, prefer run_full_3d_gen_pipeline."
                " Such errors are often repairable by re-running SAM3D from input plus Agent2 relative-pose adjustment.\n"
                " 2) Recommend regenerate_with_memory_frame only when there is explicit temporal-drift/history-break evidence from history frames,"
                " or key targets are mostly occluded in the target frame.\n"
                "Important constraint: SAM3D is effective only when most of the target object is visible "
                "(small occlusion is acceptable). If the target is heavily occluded, SAM3D re-generation is usually ineffective; "
                "prefer memory-frame + Generate Next Segment with memory.\n"
                "Hard constraint: if frame_meta has no history frame (target-only), memory-based regeneration is not executable; "
                "do NOT recommend regenerate_with_memory_frame, recommend run_full_3d_gen_pipeline (or manual_review) instead.\n"
                "If memory-based repair is recommended, provide characteristics of the ideal memory frame(s).\n"
                "Return JSON only: "
                "{"
                '"root_cause":"current_frame_3d_quality|past_memory_loss|mixed|uncertain",'
                '"confidence":0-1,'
                '"failure_phenomena":["position_drift|penetration|separation|other"],'
                '"sam3d_repair_feasible":true/false,'
                '"sam3d_repair_rationale":"one line on whether SAM3D object-wise re-generation + pose adjustment can sufficiently fix the target frame",'
                '"occlusion_level":"low|medium|high|unknown",'
                '"object_visibility":[{"object_name":string,"visibility":"mostly_visible|partially_visible|mostly_occluded|unknown","reason":string}],'
                '"action_recommendation":"run_full_3d_gen_pipeline|regenerate_with_memory_frame|manual_review",'
                '"reason":"one line",'
                '"evidence":[string],'
                '"pipeline_advice":"if pipeline is recommended, mention Run Full 3D Gen Pipeline",'
                '"memory_advice":"if memory route is recommended, mention selecting memory frame + Generate Next Segment with memory",'
                '"memory_frame_candidates":[{"frame_idx":int,"frame_token":string,"priority":int,"reason":string}],'
                '"memory_frame_characteristics":[string]'
                "}"
                f"\nTarget object: {target_name}"
                f"\nRelation hint: {str(relation_hint or '').strip()}"
                f"\nscene_graph: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\nvalidator_eval: {json.dumps(eval_compact, ensure_ascii=False)}"
                f"\nframe_meta: {json.dumps(frame_meta, ensure_ascii=False)}"
            )
            system_content = "Return valid JSON only."

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_budget = max(5, min(32, int(max_images)))
        image_used = 0

        def _append_image(label: str, arr: Any, *, quality: int, detail: str) -> None:
            nonlocal image_used
            if image_used >= image_budget:
                return
            if arr is None:
                return
            u = self._to_data_url(arr, jpeg_quality=int(quality))
            if not u:
                return
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": u, "detail": detail}})
            image_used += 1

        _append_image("[reference_input_target_frame]", reference_input_rgb_u8, quality=85, detail="high")
        for i, pack in enumerate(used_frames):
            role = "target" if i == len(used_frames) - 1 else "history"
            tok = str(pack.get("frame_token", ""))
            idx = int(pack.get("frame_idx", -1))
            views = dict(pack.get("view_images", {}) or {})
            keep_views = list(view_order) if role == "target" else ["front", "down"]
            for vn in keep_views:
                _append_image(
                    f"[frame role={role} idx={idx} token={tok} view={vn}]",
                    views.get(vn, None),
                    quality=76,
                    detail=("high" if role == "target" else "low"),
                )

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": content}]
        self._debug_print_json(
            "[VLM Assistor][Agent-4][Diagnosis] request=",
            {
                "mode": mode,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "target_object": target_name,
                "frame_meta": frame_meta,
                "images_used": int(image_used),
                "messages_preview": self._debug_sanitize_messages(messages),
            },
        )

        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            payload["messages"] = messages
            payload["response_format"] = {"type": "json_object"}
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            payload["messages"] = self._convert_system_to_user_prefix(messages)
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )
        if not isinstance(resp, dict):
            out = dict(fallback)
            out.update(
                {
                    "ok": False,
                    "reason": f"{provider}_http_failed",
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                }
            )
            return out
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
            print("[VLM Assistor][Agent-4][Diagnosis] raw_response_text:", str(text)[:2000])
        except Exception as exc:
            out = dict(fallback)
            out.update(
                {
                    "ok": False,
                    "reason": f"invalid_response:{type(exc).__name__}:{exc}",
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                }
            )
            return out

        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            out = dict(fallback)
            out.update(
                {
                    "ok": False,
                    "reason": "diagnosis_json_parse_failed",
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "raw_text": str(text)[:500],
                }
            )
            return out

        def _norm_root(raw: Any) -> str:
            s = str(raw or "").strip().lower()
            if s in {"current_frame_3d_quality", "current_frame_quality", "current"}:
                return "current_frame_3d_quality"
            if s in {"past_memory_loss", "memory_loss", "temporal_memory_loss", "history_drift"}:
                return "past_memory_loss"
            if s in {"mixed", "both"}:
                return "mixed"
            return "uncertain"

        def _norm_occ(raw: Any) -> str:
            s = str(raw or "").strip().lower()
            if s in {"low", "small"}:
                return "low"
            if s in {"medium", "mid", "moderate"}:
                return "medium"
            if s in {"high", "heavy", "severe"}:
                return "high"
            return "unknown"

        def _norm_action(raw: Any) -> str:
            s = str(raw or "").strip().lower()
            if s in {"run_full_3d_gen_pipeline", "pipeline", "run_pipeline"}:
                return "run_full_3d_gen_pipeline"
            if s in {"regenerate_with_memory_frame", "memory", "memory_frame", "generate_next_segment_with_memory"}:
                return "regenerate_with_memory_frame"
            return "manual_review"

        def _to_bool(raw: Any, default: bool) -> bool:
            if isinstance(raw, bool):
                return bool(raw)
            s = str(raw or "").strip().lower()
            if s in {"1", "true", "yes", "y"}:
                return True
            if s in {"0", "false", "no", "n"}:
                return False
            return bool(default)

        def _norm_str_list(raw: Any, limit: int = 12) -> list[str]:
            if isinstance(raw, list):
                out = [str(x).strip() for x in raw if str(x).strip()]
                return out[:limit]
            s = str(raw or "").strip()
            return [s] if s else []

        def _norm_candidates(raw: Any) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            if not isinstance(raw, list):
                return out
            for i, item in enumerate(raw):
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("frame_idx", -1))
                except Exception:
                    idx = -1
                tok = str(item.get("frame_token", "")).strip()
                try:
                    pr = int(item.get("priority", i + 1))
                except Exception:
                    pr = i + 1
                reason = str(item.get("reason", "")).strip()
                if idx < 0 and not tok:
                    continue
                out.append(
                    {
                        "frame_idx": int(idx),
                        "frame_token": str(tok),
                        "priority": int(max(1, pr)),
                        "reason": reason,
                    }
                )
                if len(out) >= 5:
                    break
            return out

        def _norm_object_visibility(raw: Any) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            if not isinstance(raw, list):
                return out
            for i, item in enumerate(raw):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("object_name", item.get("name", ""))).strip()
                vis_raw = str(item.get("visibility", item.get("level", "unknown"))).strip().lower()
                if vis_raw in {"mostly_visible", "visible", "high", "good"}:
                    vis = "mostly_visible"
                elif vis_raw in {"partially_visible", "partial", "medium"}:
                    vis = "partially_visible"
                elif vis_raw in {"mostly_occluded", "occluded", "low", "hidden"}:
                    vis = "mostly_occluded"
                else:
                    vis = "unknown"
                reason = str(item.get("reason", item.get("evidence", ""))).strip()
                if not name and not reason:
                    continue
                out.append(
                    {
                        "object_name": name or f"obj_{int(i + 1)}",
                        "visibility": vis,
                        "reason": reason,
                    }
                )
                if len(out) >= 24:
                    break
            return out

        root_cause = _norm_root(parsed.get("root_cause", fallback.get("root_cause", "uncertain")))
        occlusion_level = _norm_occ(parsed.get("occlusion_level", fallback.get("occlusion_level", "unknown")))
        failure_phenomena = self._normalize_failure_phenomena(
            parsed.get("failure_phenomena", fallback.get("failure_phenomena", []))
        )
        if not failure_phenomena:
            failure_phenomena = self._collect_failure_phenomena_from_eval(eval_compact)
        action = _norm_action(parsed.get("action_recommendation", fallback.get("action_recommendation", "manual_review")))
        object_visibility = _norm_object_visibility(parsed.get("object_visibility", fallback.get("object_visibility", [])))
        memory_frame_candidates = _norm_candidates(parsed.get("memory_frame_candidates", []))
        memory_frame_characteristics = _norm_str_list(
            parsed.get("memory_frame_characteristics", fallback.get("memory_frame_characteristics", [])),
            limit=16,
        )
        pipeline_advice = str(parsed.get("pipeline_advice", "")).strip()
        memory_advice = str(parsed.get("memory_advice", "")).strip()
        sam3d_repair_feasible = _to_bool(
            parsed.get("sam3d_repair_feasible", fallback.get("sam3d_repair_feasible", False)),
            default=bool(fallback.get("sam3d_repair_feasible", False)),
        )
        try:
            confidence = float(parsed.get("confidence", fallback.get("confidence", 0.5)))
        except Exception:
            confidence = float(fallback.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        has_mostly_occluded = any(str(x.get("visibility", "")) == "mostly_occluded" for x in object_visibility)
        if has_mostly_occluded:
            sam3d_repair_feasible = False
        if occlusion_level == "high":
            sam3d_repair_feasible = False
        if root_cause == "past_memory_loss":
            sam3d_repair_feasible = False
        if "position_drift" in set(failure_phenomena) and root_cause == "past_memory_loss":
            sam3d_repair_feasible = False
        if sam3d_repair_feasible and root_cause in {"current_frame_3d_quality", "mixed", "uncertain"} and (not has_mostly_occluded):
            action = "run_full_3d_gen_pipeline"
        elif (not sam3d_repair_feasible) and root_cause in {"current_frame_3d_quality", "past_memory_loss", "mixed"}:
            action = "regenerate_with_memory_frame"

        if (not bool(has_history_frames)) and action == "regenerate_with_memory_frame":
            action = "run_full_3d_gen_pipeline"
            sam3d_repair_feasible = True
            memory_frame_candidates = []
            memory_frame_characteristics = []
            if not pipeline_advice:
                pipeline_advice = "No history frame available; run Run Full 3D Gen Pipeline on current frame."
            memory_advice = ""

        if action == "regenerate_with_memory_frame" and not memory_frame_characteristics:
            memory_frame_characteristics = [
                "Select frames before visible drift/penetration starts.",
                "Target objects should be mostly visible with low occlusion.",
                "Choose frames with stable and physically plausible contact/separation relations.",
            ]

        if action == "run_full_3d_gen_pipeline":
            fallback = {
                "ok": True,
                "evaluated": True,
                "source": "vlm_guided_fallback",
                "root_cause": root_cause,
                "confidence": float(confidence),
                "failure_phenomena": list(failure_phenomena),
                "sam3d_repair_feasible": True,
                "sam3d_repair_rationale": str(parsed.get("sam3d_repair_rationale", "")).strip(),
                "occlusion_level": occlusion_level,
                "object_visibility": list(object_visibility),
                "action_recommendation": "run_full_3d_gen_pipeline",
                "reason": str(parsed.get("reason", "")).strip(),
                "evidence": _norm_str_list(parsed.get("evidence", []), limit=16),
                "memory_frame_candidates": list(memory_frame_candidates),
                "memory_frame_characteristics": list(memory_frame_characteristics),
                "pipeline_advice": pipeline_advice,
                "memory_advice": memory_advice,
            }
        else:
            fallback = {
                "ok": True,
                "evaluated": True,
                "source": "vlm_guided_fallback",
                "root_cause": root_cause,
                "confidence": float(confidence),
                "failure_phenomena": list(failure_phenomena),
                "sam3d_repair_feasible": bool(sam3d_repair_feasible),
                "sam3d_repair_rationale": str(parsed.get("sam3d_repair_rationale", "")).strip(),
                "occlusion_level": occlusion_level,
                "object_visibility": list(object_visibility),
                "action_recommendation": "regenerate_with_memory_frame",
                "reason": str(parsed.get("reason", "")).strip(),
                "evidence": _norm_str_list(parsed.get("evidence", []), limit=16),
                "memory_frame_candidates": list(memory_frame_candidates),
                "memory_frame_characteristics": list(memory_frame_characteristics),
                "pipeline_advice": pipeline_advice,
                "memory_advice": memory_advice,
            }

        out = {
            "ok": True,
            "evaluated": True,
            "source": "vlm",
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "root_cause": root_cause,
            "confidence": float(confidence),
            "failure_phenomena": list(failure_phenomena),
            "sam3d_repair_feasible": bool(sam3d_repair_feasible),
            "sam3d_repair_rationale": str(
                parsed.get(
                    "sam3d_repair_rationale",
                    fallback.get("sam3d_repair_rationale", ""),
                )
            ).strip(),
            "occlusion_level": occlusion_level,
            "object_visibility": list(object_visibility),
            "action_recommendation": action,
            "reason": str(parsed.get("reason", "")).strip(),
            "evidence": _norm_str_list(parsed.get("evidence", []), limit=16),
            "pipeline_advice": pipeline_advice,
            "memory_advice": memory_advice,
            "memory_frame_candidates": list(memory_frame_candidates),
            "memory_frame_characteristics": list(memory_frame_characteristics),
            "validator_scores": {
                "overall_score": int(float(eval_compact.get("overall_score", 0) or 0)),
                "input_alignment_score": int(float(eval_compact.get("input_alignment_score", 0) or 0)),
                "interaction_semantic_alignment_score": int(float(eval_compact.get("interaction_semantic_alignment_score", 0) or 0)),
                "physical_plausibility_score": int(float(eval_compact.get("physical_plausibility_score", 0) or 0)),
                "deformation_reasonableness_score": int(float(eval_compact.get("deformation_reasonableness_score", 0) or 0)),
                "temporal_consistency_4d_score": int(float(eval_compact.get("temporal_consistency_4d_score", 0) or 0)),
                "clarity_3d_score": int(float(eval_compact.get("clarity_3d_score", 0) or 0)),
            },
            "raw": parsed,
            "fallback": fallback,
            "heuristic_backup": score_fallback,
        }
        self._debug_print_json(
            "[VLM Assistor][Agent-4][Diagnosis] normalized_result=",
            out,
        )
        return out

    def evaluate_memory_frame_candidate_with_vlm(
        self,
        *,
        object_text: str,
        relation_hint: str,
        scene_graph: dict[str, Any] | None,
        desired_traits: list[str] | None,
        target_frame: dict[str, Any],
        candidate_frame: dict[str, Any],
        backend_mode: str | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        def _fallback(reason: str) -> dict[str, Any]:
            views = dict(candidate_frame.get("view_images", {}) or {})
            vis = int(sum(1 for k in ["down", "up", "left", "front"] if views.get(k) is not None))
            score = max(0, min(100, 45 + 12 * vis))
            return {
                "ok": True,
                "evaluated": False,
                "source": "heuristic",
                "reason": str(reason),
                "score": int(score),
                "recommended": bool(score >= 70),
                "visibility_score": int(min(100, vis * 25)),
                "geometry_consistency_score": 60,
                "interaction_consistency_score": 60,
                "temporal_anchor_score": 60,
                "summary": "heuristic_fallback",
                "evidence": [],
            }

        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"
        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
        if not base_url:
            return _fallback("vlm_base_url_missing")

        traits = [str(x).strip() for x in list(desired_traits or []) if str(x).strip()][:8]
        scene_graph_compact = dict(scene_graph or {})
        try:
            if len(json.dumps(scene_graph_compact, ensure_ascii=False)) > 3000:
                scene_graph_compact = {
                    "objects": list(scene_graph_compact.get("objects", []) or [])[:20],
                    "relations": list(scene_graph_compact.get("relations", []) or [])[:20],
                    "events": list(scene_graph_compact.get("events", []) or [])[:12],
                }
        except Exception:
            scene_graph_compact = {}

        language = self._language_for_model(model)
        target_name = self._to_language_object_name(str(object_text or "").strip(), language)
        prompt_zh = (
            "你是记忆帧选择评估器。任务：评估一个候选历史帧是否适合作为当前目标帧修复的 memory frame。"
            "结合目标帧(target)与候选帧(candidate)的输入图+四视图、scene_graph与期望特征(desired_traits)打分。"
            "只输出JSON：{"
            '"score":0-100整数,'
            '"recommended":true/false,'
            '"visibility_score":0-100整数,'
            '"geometry_consistency_score":0-100整数,'
            '"interaction_consistency_score":0-100整数,'
            '"temporal_anchor_score":0-100整数,'
            '"summary":"一句中文结论",'
            '"evidence":[string]'
            "}"
            f"\n目标物体: {target_name}"
            f"\n关系提示: {str(relation_hint or '').strip()}"
            f"\ndesired_traits: {json.dumps(traits, ensure_ascii=False)}"
            f"\nscene_graph: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
            f"\n目标帧: idx={int(target_frame.get('frame_idx', -1))}, token={str(target_frame.get('frame_token', ''))}"
            f"\n候选帧: idx={int(candidate_frame.get('frame_idx', -1))}, token={str(candidate_frame.get('frame_token', ''))}"
        )
        prompt_en = (
            "You are a memory-frame selector."
            " Evaluate whether ONE candidate history frame is suitable as memory for repairing the target frame."
            " Use target/candidate input images + four views, scene graph, and desired traits."
            " Return JSON only: {"
            '"score":int(0-100),'
            '"recommended":bool,'
            '"visibility_score":int,'
            '"geometry_consistency_score":int,'
            '"interaction_consistency_score":int,'
            '"temporal_anchor_score":int,'
            '"summary":"one-line conclusion",'
            '"evidence":[string]'
            "}"
            f"\nTarget object: {target_name}"
            f"\nRelation hint: {str(relation_hint or '').strip()}"
            f"\ndesired_traits: {json.dumps(traits, ensure_ascii=False)}"
            f"\nscene_graph: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
            f"\nTarget frame: idx={int(target_frame.get('frame_idx', -1))}, token={str(target_frame.get('frame_token', ''))}"
            f"\nCandidate frame: idx={int(candidate_frame.get('frame_idx', -1))}, token={str(candidate_frame.get('frame_token', ''))}"
        )
        prompt = prompt_zh if language == "zh" else prompt_en
        system_content = "只返回合法JSON。不要输出解释。" if language == "zh" else "Return valid JSON only."
        view_order = ["down", "up", "left", "front"]
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        def _append_img(label: str, arr: Any, *, quality: int, detail: str) -> None:
            if arr is None:
                return
            u = self._to_data_url(arr, jpeg_quality=int(quality))
            if not u:
                return
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": u, "detail": detail}})

        _append_img("[target_input]", target_frame.get("input_rgb_u8", None), quality=84, detail="high")
        for vn in view_order:
            _append_img(f"[target_view:{vn}]", dict(target_frame.get("view_images", {}) or {}).get(vn, None), quality=75, detail="low")
        _append_img("[candidate_input]", candidate_frame.get("input_rgb_u8", None), quality=84, detail="high")
        for vn in view_order:
            _append_img(
                f"[candidate_view:{vn}]",
                dict(candidate_frame.get("view_images", {}) or {}).get(vn, None),
                quality=75,
                detail="low",
            )

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": content}]
        self._debug_print_json(
            "[VLM Assistor][MemorySelector] candidate_request=",
            {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "target": {
                    "frame_idx": int(target_frame.get("frame_idx", -1)),
                    "frame_token": str(target_frame.get("frame_token", "")),
                },
                "candidate": {
                    "frame_idx": int(candidate_frame.get("frame_idx", -1)),
                    "frame_token": str(candidate_frame.get("frame_token", "")),
                },
                "desired_traits": traits,
                "messages_preview": self._debug_sanitize_messages(messages),
            },
        )

        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            payload["messages"] = messages
            payload["response_format"] = {"type": "json_object"}
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            payload["messages"] = self._convert_system_to_user_prefix(messages)
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )
        if not isinstance(resp, dict):
            return _fallback(f"{provider}_http_failed")
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
            print("[VLM Assistor][MemorySelector] candidate_raw_response:", str(text)[:1200])
        except Exception:
            return _fallback("invalid_response")
        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            return _fallback("json_parse_failed")

        def _score(name: str, default: int = 0) -> int:
            try:
                return max(0, min(100, int(round(float(parsed.get(name, default))))))
            except Exception:
                return int(default)

        score = _score("score", 0)
        out = {
            "ok": True,
            "evaluated": True,
            "source": "vlm",
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "score": int(score),
            "recommended": bool(parsed.get("recommended", score >= 70)),
            "visibility_score": _score("visibility_score", score),
            "geometry_consistency_score": _score("geometry_consistency_score", score),
            "interaction_consistency_score": _score("interaction_consistency_score", score),
            "temporal_anchor_score": _score("temporal_anchor_score", score),
            "summary": str(parsed.get("summary", "")).strip(),
            "evidence": [str(x).strip() for x in list(parsed.get("evidence", []) or []) if str(x).strip()][:8],
            "raw": parsed,
        }
        self._debug_print_json("[VLM Assistor][MemorySelector] candidate_normalized=", out)
        return out

    def decide_memory_bank_update_with_vlm(
        self,
        *,
        object_text: str,
        relation_hint: str,
        scene_graph: dict[str, Any] | None,
        current_frame: dict[str, Any],
        memory_frames: list[dict[str, Any]],
        backend_mode: str | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"
        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()

        def _fallback(reason: str) -> dict[str, Any]:
            return {
                "ok": True,
                "evaluated": False,
                "source": "heuristic",
                "reason": str(reason),
                "should_add": bool(len(memory_frames) < 3),
                "replace_frame_idx": -1,
                "score": 60 if len(memory_frames) < 3 else 45,
                "summary": "heuristic_fallback",
            }

        if not base_url:
            return _fallback("vlm_base_url_missing")

        scene_graph_compact = {
            "objects": list((scene_graph or {}).get("objects", []) or [])[:20],
            "relations": list((scene_graph or {}).get("relations", []) or [])[:20],
            "events": list((scene_graph or {}).get("events", []) or [])[:12],
        }
        language = self._language_for_model(model)
        target_name = self._to_language_object_name(str(object_text or "").strip(), language)
        memory_rows = [
            {
                "frame_idx": int(x.get("frame_idx", -1)),
                "frame_token": str(x.get("frame_token", "")),
            }
            for x in list(memory_frames or [])[:6]
        ]
        if language == "zh":
            prompt = (
                "你是memory bank管理器。判断是否应把当前帧加入memory bank，或替换已有记忆帧。"
                "只输出JSON：{"
                '"should_add":true/false,'
                '"replace_frame_idx":int(-1表示不替换),'
                '"score":0-100整数,'
                '"reason":"一句中文理由"'
                "}"
                f"\n目标物体: {target_name}"
                f"\n关系提示: {str(relation_hint or '').strip()}"
                f"\nscene_graph: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\ncurrent_frame: idx={int(current_frame.get('frame_idx', -1))}, token={str(current_frame.get('frame_token', ''))}"
                f"\nmemory_frames: {json.dumps(memory_rows, ensure_ascii=False)}"
            )
            system_content = "只返回合法JSON。"
        else:
            prompt = (
                "You are a memory-bank manager."
                " Decide whether current frame should be added to memory bank or replace an existing memory frame."
                " Return JSON only: {"
                '"should_add":bool,'
                '"replace_frame_idx":int(-1 means no replacement),'
                '"score":int(0-100),'
                '"reason":"one-line reason"'
                "}"
                f"\nTarget object: {target_name}"
                f"\nRelation hint: {str(relation_hint or '').strip()}"
                f"\nscene_graph: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\ncurrent_frame: idx={int(current_frame.get('frame_idx', -1))}, token={str(current_frame.get('frame_token', ''))}"
                f"\nmemory_frames: {json.dumps(memory_rows, ensure_ascii=False)}"
            )
            system_content = "Return valid JSON only."

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        def _append_img(label: str, arr: Any, *, quality: int, detail: str) -> None:
            if arr is None:
                return
            u = self._to_data_url(arr, jpeg_quality=int(quality))
            if not u:
                return
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": u, "detail": detail}})

        _append_img("[current_input]", current_frame.get("input_rgb_u8", None), quality=84, detail="high")
        for vn in ["down", "up", "left", "front"]:
            _append_img(f"[current_view:{vn}]", dict(current_frame.get("view_images", {}) or {}).get(vn, None), quality=75, detail="low")
        for i, m in enumerate(list(memory_frames or [])[:4]):
            _append_img(f"[memory_{i}_input]", m.get("input_rgb_u8", None), quality=80, detail="low")
            _append_img(
                f"[memory_{i}_front]",
                dict(m.get("view_images", {}) or {}).get("front", None),
                quality=74,
                detail="low",
            )

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": content}]
        self._debug_print_json(
            "[VLM Assistor][MemorySelector] memory_bank_update_request=",
            {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "current_frame": {
                    "frame_idx": int(current_frame.get("frame_idx", -1)),
                    "frame_token": str(current_frame.get("frame_token", "")),
                },
                "memory_rows": memory_rows,
                "messages_preview": self._debug_sanitize_messages(messages),
            },
        )

        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            payload["messages"] = messages
            payload["response_format"] = {"type": "json_object"}
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            payload["messages"] = self._convert_system_to_user_prefix(messages)
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )
        if not isinstance(resp, dict):
            return _fallback(f"{provider}_http_failed")
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
            print("[VLM Assistor][MemorySelector] memory_bank_update_raw_response:", str(text)[:1200])
        except Exception:
            return _fallback("invalid_response")
        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            return _fallback("json_parse_failed")
        try:
            score = max(0, min(100, int(round(float(parsed.get("score", 0))))))
        except Exception:
            score = 0
        try:
            replace_idx = int(parsed.get("replace_frame_idx", -1))
        except Exception:
            replace_idx = -1
        out = {
            "ok": True,
            "evaluated": True,
            "source": "vlm",
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "should_add": bool(parsed.get("should_add", score >= 70)),
            "replace_frame_idx": int(replace_idx),
            "score": int(score),
            "reason": str(parsed.get("reason", "")).strip(),
            "raw": parsed,
        }
        self._debug_print_json("[VLM Assistor][MemorySelector] memory_bank_update_normalized=", out)
        return out

    def rank_xpose_candidates_with_vlm(
        self,
        *,
        object_name: str,
        relation_hint: str,
        scene_graph: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        object_context: dict[str, Any] | None = None,
        backend_mode: str | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        if not isinstance(candidates, list) or len(candidates) <= 0:
            return {"ok": False, "evaluated": False, "reason": "empty_candidates"}

        def _heuristic_rank() -> dict[str, Any]:
            best = None
            best_key = None
            for item in candidates:
                c_rule = dict(item.get("rule_diagnostics", {}) or {})
                collision_count = int(c_rule.get("collision_count", 0))
                out_of_bounds = bool(c_rule.get("x_out_of_bounds", False))
                center_x = float(c_rule.get("target_center_xyz", [0.0, 0.0, 0.0])[0])
                score_key = (int(out_of_bounds), int(collision_count), abs(center_x))
                if best_key is None or score_key < best_key:
                    best_key = score_key
                    best = item
            if best is None:
                self._debug_print_json(
                    "[VLM Assistor][Agent2][Rank] heuristic_failed=",
                    {"reason": "heuristic_rank_failed"},
                )
                return {"ok": False, "evaluated": False, "reason": "heuristic_rank_failed"}
            out = {
                "ok": True,
                "evaluated": False,
                "best_candidate_id": int(best.get("candidate_id", 0)),
                "need_another_round": bool(best.get("rule_diagnostics", {}).get("collision_count", 0) > 0),
                "next_direction": "none",
                "summary": "heuristic fallback ranking",
                "source": "heuristic",
            }
            self._debug_print_json(
                "[VLM Assistor][Agent2][Rank] heuristic_result=",
                out,
            )
            return out

        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "local_openai_compatible"
        provider = "local_openai_compatible"
        model = self._local_vlm_model()
        base_url = self._local_vlm_base_url()
        api_key = self._local_vlm_api_key()
        if mode == "official_api":
            provider = "gpt4o"
            model = self._openai_model()
            base_url = self._openai_base_url()
            api_key = self._openai_api_key()
        if not base_url:
            return _heuristic_rank()

        language = self._language_for_model(model)
        target_name = self._to_language_object_name(str(object_name), language)
        scene_graph_compact = dict(scene_graph or {})
        if len(json.dumps(scene_graph_compact, ensure_ascii=False)) > 3000:
            scene_graph_compact = {
                "objects": list(scene_graph_compact.get("objects", []) or [])[:16],
                "relations": list(scene_graph_compact.get("relations", []) or [])[:16],
                "events": list(scene_graph_compact.get("events", []) or [])[:10],
                "x_depth_order_timeline": list(scene_graph_compact.get("x_depth_order_timeline", []) or [])[:6],
                "relative_position_timeline": list(scene_graph_compact.get("relative_position_timeline", []) or [])[:6],
            }
        object_context_compact = dict(object_context or {})
        try:
            if len(json.dumps(object_context_compact, ensure_ascii=False)) > 3000:
                object_context_compact = {
                    "current_generated_object": dict(object_context_compact.get("current_generated_object", {}) or {}),
                    "already_generated_objects": list(object_context_compact.get("already_generated_objects", []) or [])[:16],
                    "all_generated_objects": list(object_context_compact.get("all_generated_objects", []) or [])[:20],
                    "generated_name_map": dict(object_context_compact.get("generated_name_map", {}) or {}),
                }
        except Exception:
            object_context_compact = {
                "current_generated_object": dict((object_context or {}).get("current_generated_object", {}) or {}),
                "already_generated_objects": list((object_context or {}).get("already_generated_objects", []) or [])[:16],
            }

        content: list[dict[str, Any]] = []
        if language == "zh":
            intro = (
                "你是候选位姿排序器。请比较所有candidate的多视角图和overlay，选择最优位姿。"
                "你必须先分析每个candidate中“目标物体与已生成物体”的相对位置（左右/上下/前后）再排序。"
                "在满足物理合理性和pixel-align的前提下，优先选择目标在六视角中更靠近图像中心、不贴边的candidate。"
                "关系约束只应使用与当前目标直接相关的关系；无关关系仅作背景。"
                "返回JSON："
                '{"best_candidate_id":int,"need_another_round":bool,"next_direction":"positive|negative|none","pre_adjust_analysis_summary":string,"summary":"一句中文说明"}'
                f"\n目标物体: {target_name}"
                f"\n关系提示: {str(relation_hint or '').strip()}"
                f"\n时序图摘要: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\n物体上下文(当前刚生成目标 + 已生成物体): {json.dumps(object_context_compact, ensure_ascii=False)}"
                "\n若目标缺少直接关系，请优先比较与已生成邻近物体的远近关系、深度顺序、碰撞/穿透风险。"
                "\n方向定义：positive=世界坐标系 +X，negative=世界坐标系 -X。"
            )
        else:
            intro = (
                "You are a candidate pose ranker. Compare all candidates using multiview images and overlays, then choose the best pose."
                " You must first analyze target-vs-generated relative positions (left/right, above/below, front/behind) for each candidate before ranking."
                " Under physical plausibility and pixel alignment, prefer candidates where the target is more centered and less border-touching in six views."
                " Use relation constraints only when they directly involve the current target; unrelated relations are background context."
                ' Return JSON: {"best_candidate_id":int,"need_another_round":bool,"next_direction":"positive|negative|none","pre_adjust_analysis_summary":string,"summary":"short reason"}'
                f"\nTarget: {target_name}"
                f"\nRelation hint: {str(relation_hint or '').strip()}"
                f"\nTSG summary: {json.dumps(scene_graph_compact, ensure_ascii=False)}"
                f"\nObject context (current generated target + already generated objects): {json.dumps(object_context_compact, ensure_ascii=False)}"
                "\nIf target-specific relations are absent, prioritize distance/proximity, depth order, and collision/penetration against generated neighbors."
                "\nDirection convention: positive=world +X, negative=world -X."
            )
        content.append({"type": "text", "text": intro})

        max_images = 42
        img_count = 0
        for cand in candidates:
            cid = int(cand.get("candidate_id", 0))
            dx = float(cand.get("delta_x", 0.0))
            rule_diag = dict(cand.get("rule_diagnostics", {}) or {})
            content.append(
                {
                    "type": "text",
                    "text": f"[candidate {cid}] delta_x={dx:.4f} rule={json.dumps(rule_diag, ensure_ascii=False)}",
                }
            )
            overlay = cand.get("overlay_rgb_u8", None)
            if overlay is not None and img_count < max_images:
                u = self._to_data_url(overlay, jpeg_quality=80)
                if u:
                    content.append({"type": "text", "text": f"[candidate {cid} overlay]"})
                    content.append({"type": "image_url", "image_url": {"url": u, "detail": "high"}})
                    img_count += 1
            views = dict(cand.get("view_images", {}) or {})
            for k in ["front", "back", "left", "right", "up", "down"]:
                if img_count >= max_images:
                    break
                arr = views.get(k, None)
                if arr is None:
                    continue
                u = self._to_data_url(arr, jpeg_quality=75)
                if not u:
                    continue
                content.append({"type": "text", "text": f"[candidate {cid} view:{k}]"})
                content.append({"type": "image_url", "image_url": {"url": u, "detail": "low"}})
                img_count += 1

        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": content},
        ]
        self._debug_print_json(
            "[VLM Assistor][Agent2][Rank] request=",
            {
                "mode": mode,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "target_object": target_name,
                "relation_hint": str(relation_hint or "").strip(),
                "object_context": dict(object_context_compact or {}),
                "intro_prompt": intro,
                "candidate_count": int(len(candidates)),
                "candidates_meta": [
                    {
                        "candidate_id": int(c.get("candidate_id", 0)),
                        "delta_x": float(c.get("delta_x", 0.0)),
                        "rule_diagnostics": dict(c.get("rule_diagnostics", {}) or {}),
                    }
                    for c in list(candidates)
                ],
                "messages_preview": self._debug_sanitize_messages(messages),
            },
        )
        payload: dict[str, Any] = {"model": model, "temperature": 0.0}
        if mode == "official_api":
            payload["messages"] = messages
            payload["response_format"] = {"type": "json_object"}
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=True,
            )
        else:
            payload["messages"] = self._convert_system_to_user_prefix(messages)
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                require_api_key=False,
            )
        if not isinstance(resp, dict):
            self._debug_print_json(
                "[VLM Assistor][Agent2][Rank] response_error=",
                {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "resp_type": str(type(resp)),
                    "fallback": "heuristic",
                },
            )
            return _heuristic_rank()
        try:
            text = self._extract_message_text(resp["choices"][0]["message"]["content"])
            print("[VLM Assistor][Agent2][Rank] raw_response_text:", str(text)[:2000])
        except Exception:
            self._debug_print_json(
                "[VLM Assistor][Agent2][Rank] response_parse_error=",
                {"fallback": "heuristic"},
            )
            return _heuristic_rank()
        parsed = self._try_parse_json_obj(text)
        if not isinstance(parsed, dict):
            self._debug_print_json(
                "[VLM Assistor][Agent2][Rank] response_json_parse_failed=",
                {
                    "raw_text_head": str(text)[:400],
                    "fallback": "heuristic",
                },
            )
            return _heuristic_rank()
        self._debug_print_json(
            "[VLM Assistor][Agent2][Rank] parsed_json=",
            parsed,
        )
        cand_ids = {int(x.get("candidate_id", -1)) for x in candidates}
        best_id = int(parsed.get("best_candidate_id", -1))
        if best_id not in cand_ids:
            self._debug_print_json(
                "[VLM Assistor][Agent2][Rank] invalid_best_candidate_id=",
                {
                    "best_candidate_id": int(best_id),
                    "candidate_ids": sorted([int(x) for x in cand_ids]),
                    "fallback": "heuristic",
                },
            )
            return _heuristic_rank()
        raw_dir = str(parsed.get("next_direction", parsed.get("suggested_x_direction", "none"))).strip().lower()
        if raw_dir in {"+x", "positive", "forward", "front"}:
            next_dir = "positive"
        elif raw_dir in {"-x", "negative", "backward", "back"}:
            next_dir = "negative"
        else:
            next_dir = "none"
        out = {
            "ok": True,
            "evaluated": True,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "best_candidate_id": int(best_id),
            "need_another_round": bool(parsed.get("need_another_round", False)),
            "next_direction": str(next_dir),
            "pre_adjust_analysis_summary": str(parsed.get("pre_adjust_analysis_summary", "")).strip(),
            "summary": str(parsed.get("summary", "")).strip(),
            "raw": parsed,
        }
        self._debug_print_json(
            "[VLM Assistor][Agent2][Rank] normalized_result=",
            out,
        )
        return out

    @staticmethod
    def _try_parse_json_obj(text: Any) -> dict[str, Any] | None:
        s = str(text or "").strip()
        if not s:
            return None
        # direct JSON
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        # fenced JSON
        if s.startswith("```"):
            sf = s.strip("`")
            if sf.lower().startswith("json"):
                sf = sf[4:].strip()
            try:
                parsed = json.loads(sf)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                pass
        # loose text: take first {...}
        st = s.find("{")
        ed = s.rfind("}")
        if st >= 0 and ed > st:
            cand = s[st : ed + 1]
            try:
                parsed = json.loads(cand)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    @staticmethod
    def _convert_system_to_user_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages for stricter local OpenAI-compatible servers."""
        sys_lines: list[str] = []
        out: list[dict[str, Any]] = []
        prefixed = False
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).strip()
            content = m.get("content")
            if role in {"system", "developer"}:
                if isinstance(content, str):
                    sys_lines.append(content)
                else:
                    sys_lines.append(str(content))
                continue
            if role != "user":
                continue
            prefix = "\n\n".join([x for x in sys_lines if x]).strip()
            if (not prefixed) and prefix:
                if isinstance(content, str):
                    out.append({"role": "user", "content": prefix + "\n\n" + content})
                elif isinstance(content, list):
                    out.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prefix}] + list(content),
                        }
                    )
                else:
                    out.append({"role": "user", "content": prefix + "\n\n" + str(content)})
                prefixed = True
            else:
                out.append({"role": "user", "content": content})
        if out:
            return out
        if sys_lines:
            return [{"role": "user", "content": "\n\n".join([x for x in sys_lines if x])}]
        return messages

    def run_agent(self, agent: str, payload: dict[str, Any], timeout: float = 45.0) -> dict[str, Any] | None:
        base = self._base_url()
        if not base:
            return None
        req = {"agent": str(agent), "input": payload}
        try:
            return self.app._post_json(base + "/agent/run", req, timeout=timeout)
        except Exception:
            return None

    def _build_agent0_prompts(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_tokens: list[str],
        token_lines: list[str],
        language: str = "en",
        ikg_enabled: bool = True,
    ) -> tuple[str, str]:
        if not bool(ikg_enabled):
            if str(language) == "zh":
                system_prompt = (
                    "你是 4D 编辑中的 Agent-0，用于轻量物体实例发现。"
                    "只输出 JSON，不要分析交互关系、事件、物理约束、深度排序或相对方位时间线。"
                )
                user_prompt = (
                    "请只识别给定视频中的真实可分割物体实例，生成 object-only scene_graph。\n"
                    "要求：\n"
                    "1) objects 只包含真实物体实例名，不要包含背景、区域、动作、状态或“无”。\n"
                    "2) 同类物体优先按外观细节命名（颜色、纹理、形状、大小、标记等），并保持全片一致。\n"
                    "3) 不要输出交互关系、接触事件、约束、x轴深度排序、相对位置时间线。\n\n"
                    f"物体提示：{object_text}\n"
                    f"已知 frame token：{frame_tokens}\n"
                    f"帧映射：\n{chr(10).join(token_lines)}\n\n"
                    "返回 JSON（键名如下）：\n"
                    "{\n"
                    '  "objects": [string],\n'
                    '  "object_meta": [{"id":string,"name":string,"category":string,"attributes":[string]}],\n'
                    '  "relations": [],\n'
                    '  "events": [],\n'
                    '  "constraints": [],\n'
                    '  "x_depth_order_timeline": [],\n'
                    '  "relative_position_timeline": [],\n'
                    '  "notes": string\n'
                    "}\n"
                    "请使用中文填写自由文本字段。"
                )
            else:
                system_prompt = (
                    "You are Agent-0 for lightweight object-instance discovery in 4D editing. "
                    "Output JSON only. Do not analyze interactions, events, physical constraints, depth order, or relative-position timelines."
                )
                user_prompt = (
                    "Identify only real segmentable object instances in the provided video and return an object-only scene_graph.\n"
                    "Requirements:\n"
                    "1) `objects` must contain real object instances only; exclude background, regions, actions, states, and `none`.\n"
                    "2) For same-category objects, name them by visible appearance details and keep names consistent across the video.\n"
                    "3) Do not output interaction relations, contact events, constraints, x-depth order, or relative-position timelines.\n\n"
                    f"Object text hint: {object_text}\n"
                    f"Known frame tokens: {frame_tokens}\n"
                    f"Frame mapping:\n{chr(10).join(token_lines)}\n\n"
                    "Return JSON with keys:\n"
                    "{\n"
                    '  "objects": [string],\n'
                    '  "object_meta": [{"id":string,"name":string,"category":string,"attributes":[string]}],\n'
                    '  "relations": [],\n'
                    '  "events": [],\n'
                    '  "constraints": [],\n'
                    '  "x_depth_order_timeline": [],\n'
                    '  "relative_position_timeline": [],\n'
                    '  "notes": string\n'
                    "}"
                )
            return system_prompt, user_prompt
        if str(language) == "zh":
            system_prompt = (
                "你是 4D 编辑中的 Agent-0，用于时序场景理解。"
                "只输出 JSON，禁止输出数值化 6DoF 位姿。"
                "重点关注对象身份、交互关系、时序阶段、接触事件和可执行约束。"
            )
            user_prompt = (
                "请分析给定视频帧并生成时序场景图（TSG）。\n"
                "要求：\n"
                "1) 不要输出数值化的物体位姿。\n"
                "2) 只使用定性距离：near_zero | near | mid | far | unknown。\n"
                "3) 输出要能支持后续分割与位姿调整代理。\n"
                "4) 若发生接触，请用 frame_token 粗略标注接触开始时刻。\n\n"
                "4.1) 同类物体优先按外观细节区分并命名（颜色、纹理、形状、大小、破损/污渍/标记等）。\n"
                "4.2) objects 里默认只写真实物体实例名，不把方位/区域当物体名。\n"
                "4.3) 仅当同类物体在外观上确实无法区分时，允许使用方位限定名（如“左侧的杯子/右侧的杯子”），并在全片保持一致命名。\n"
                "4.4) 相对方位关系仍应主要写入 relative_position_timeline 的 horizontal/vertical/depth 字段。\n\n"
                "5) 额外输出分时段的 x 轴深度排序：按世界坐标系 x 从正到负列出物体顺序（越负越远）。\n\n"
                "6) 必须输出分时段的相对位置关系(relative_position_timeline)：每个时段给出关键物体对的左右/上下/前后关系。"
                "例如 0-10 帧可写成 A 相对 B: horizontal=left, vertical=below。\n\n"
                f"物体提示：{object_text}\n"
                f"关系提示：{relation_hint}\n"
                f"已知 frame token：{frame_tokens}\n"
                f"帧映射：\n{chr(10).join(token_lines)}\n\n"
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
                "You are Agent-0 for temporal scene understanding in 4D editing. "
                "Output JSON only. Never output numeric 6DoF pose. "
                "Focus on object identity, interaction relations, temporal phases, contact events, and actionable constraints."
            )
            user_prompt = (
                "Analyze the provided video frames and produce a Temporal Scene Graph (TSG).\n"
                "Requirements:\n"
                "1) Do NOT output numeric object pose.\n"
                "2) Use qualitative distance hints only: near_zero | near | mid | far | unknown.\n"
                "3) Output should support downstream agents for segmentation and pose adjustment.\n"
                "4) If contact happens, infer roughly when contact starts using frame_token.\n\n"
                "4.1) For same-category objects, first distinguish and name them by visible appearance details "
                "(color, texture, shape, size, damage/stain/markings, etc.).\n"
                "4.2) In `objects`, prefer real object instance names; do NOT use pure spatial/region phrases as names by default.\n"
                "4.3) Exception: if objects are truly visually indistinguishable, you may use spatial qualifiers in names "
                "(e.g., `left cup`, `right cup`), but keep this naming consistent across the whole video.\n"
                "4.4) Relative location semantics should still be represented in relative_position_timeline "
                "(horizontal/vertical/depth).\n\n"
                "5) Also output time-sliced x-axis depth order: list object order from world +X to -X (more negative means farther).\n\n"
                "6) You must output time-sliced relative positions (relative_position_timeline): for each range, provide key object-pair left/right, above/below, and front/behind relations."
                " Example: in frames 0-10, A is left and below B.\n\n"
                f"Object text hint: {object_text}\n"
                f"Relation hint: {relation_hint}\n"
                f"Known frame tokens: {frame_tokens}\n"
                f"Frame mapping:\n{chr(10).join(token_lines)}\n\n"
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
        return system_prompt, user_prompt

    def _prepare_visual_content(
        self, frame_items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        vis_content: list[dict[str, Any]] = []
        token_lines: list[str] = []
        for i, item in enumerate(frame_items):
            tok = str(item.get("frame_token", f"f{i:03d}"))
            rgb = item.get("rgb_u8")
            if rgb is None:
                continue
            data_url = self._to_data_url(rgb, jpeg_quality=70)
            if not data_url:
                continue
            token_lines.append(f"{i}: frame_token={tok}")
            vis_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "low"},
                }
            )
        return vis_content, token_lines

    @staticmethod
    def _extract_message_text(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            chunks: list[str] = []
            for item in message_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
            return "\n".join(chunks)
        return str(message_content)

    def _segment_config(self) -> tuple[int, int]:
        # GPT-4o image limit is 50 per request. Use hard safety margin (49).
        env_chunk = os.getenv("VLM_AGENT0_CHUNK_SIZE")
        env_overlap = os.getenv("VLM_AGENT0_CHUNK_OVERLAP")
        chunk_size = int(
            getattr(self.app, "vlm_agent0_chunk_size", None)
            or (int(env_chunk) if env_chunk else 12)
        )
        overlap = int(
            getattr(self.app, "vlm_agent0_chunk_overlap", None)
            or (int(env_overlap) if env_overlap else 3)
        )
        chunk_size = max(1, min(49, int(chunk_size)))
        overlap = max(0, min(int(overlap), chunk_size - 1))
        return chunk_size, overlap

    @staticmethod
    def _segment_ranges(total: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
        if total <= 0:
            return []
        if total <= chunk_size:
            return [(0, total)]
        step = max(1, chunk_size - overlap)
        out: list[tuple[int, int]] = []
        st = 0
        while st < total:
            ed = min(total, st + chunk_size)
            out.append((st, ed))
            if ed >= total:
                break
            st += step
        return out

    def _build_agent0_segment_prompt(
        self,
        *,
        object_text: str,
        relation_hint: str,
        segment_id: str,
        segment_tokens: list[str],
        memory_context: list[dict[str, Any]],
        token_lines: list[str],
        language: str = "en",
        ikg_enabled: bool = True,
    ) -> tuple[str, str]:
        mem_lines: list[str] = []
        for i, m in enumerate(memory_context[-6:]):
            mem_lines.append(f"[mem{i}] {json.dumps(m, ensure_ascii=False)}")
        memory_text = "\n".join(mem_lines) if mem_lines else "(none)"
        if not bool(ikg_enabled):
            if str(language) == "zh":
                system_prompt = (
                    "你是 4D 编辑中的 Agent-0，负责长视频的轻量物体实例发现。"
                    "你当前只分析一个短片段。只输出 JSON，不要输出交互关系或时序约束。"
                )
                user_prompt = (
                    "只分析当前片段，并返回 object-only 片段结果。\n"
                    "规则：\n"
                    "1) 只识别真实可分割物体实例。\n"
                    "2) 不要把背景、纯方位区域、动作、状态或“无”写入 objects。\n"
                    "3) 若可能，物体命名要与历史记忆保持一致。\n"
                    "4) 不要输出 relations/events/constraints/x_depth_order_timeline/relative_position_timeline。\n\n"
                    f"segment_id: {segment_id}\n"
                    f"object_text_hint: {object_text}\n"
                    f"segment_frame_tokens: {segment_tokens}\n"
                    f"frame_mapping:\n{chr(10).join(token_lines)}\n\n"
                    f"previous_segment_memory:\n{memory_text}\n\n"
                    "返回 JSON（键名如下）：\n"
                    "{\n"
                    '  "segment_id": string,\n'
                    '  "objects": [string],\n'
                    '  "object_meta": [{"id":string,"name":string,"category":string,"attributes":[string]}],\n'
                    '  "relations": [],\n'
                    '  "events": [],\n'
                    '  "constraints": [],\n'
                    '  "x_depth_order_timeline": [],\n'
                    '  "relative_position_timeline": [],\n'
                    '  "segment_memory": {"summary":string,"object_names":[string]},\n'
                    '  "notes": string\n'
                    "}\n"
                    "请使用中文填写自由文本字段。"
                )
            else:
                system_prompt = (
                    "You are Agent-0 for lightweight object-instance discovery in long-video 4D editing. "
                    "You are analyzing one short segment. Output JSON only. Do not output interaction relations or temporal constraints."
                )
                user_prompt = (
                    "Analyze only this segment and return an object-only fragment.\n"
                    "Rules:\n"
                    "1) Identify real segmentable object instances only.\n"
                    "2) Exclude background, pure spatial regions, actions, states, and `none` from `objects`.\n"
                    "3) Keep object naming consistent with previous memory if possible.\n"
                    "4) Do not output relations/events/constraints/x_depth_order_timeline/relative_position_timeline.\n\n"
                    f"segment_id: {segment_id}\n"
                    f"object_text_hint: {object_text}\n"
                    f"segment_frame_tokens: {segment_tokens}\n"
                    f"frame_mapping:\n{chr(10).join(token_lines)}\n\n"
                    f"previous_segment_memory:\n{memory_text}\n\n"
                    "Return JSON with keys:\n"
                    "{\n"
                    '  "segment_id": string,\n'
                    '  "objects": [string],\n'
                    '  "object_meta": [{"id":string,"name":string,"category":string,"attributes":[string]}],\n'
                    '  "relations": [],\n'
                    '  "events": [],\n'
                    '  "constraints": [],\n'
                    '  "x_depth_order_timeline": [],\n'
                    '  "relative_position_timeline": [],\n'
                    '  "segment_memory": {"summary":string,"object_names":[string]},\n'
                    '  "notes": string\n'
                    "}"
                )
            return system_prompt, user_prompt
        if str(language) == "zh":
            system_prompt = (
                "你是 4D 编辑中的 Agent-0，负责长视频时序理解。"
                "你当前只分析一个短片段。只输出 JSON，禁止输出数值化 6DoF 位姿。"
            )
            user_prompt = (
                "只分析当前片段，并返回片段级 TSG 片段。\n"
                "规则：\n"
                "1) 不要输出数值化位姿/距离。\n"
                "2) distance_hint 只能取 {near_zero, near, mid, far, unknown}。\n"
                "3) 重点描述在该片段内开始/结束/变化的事件。\n"
                "4) 若可能，物体命名要与历史记忆保持一致。\n\n"
                "4.0) 必须依次比较片段首帧、中间帧、末帧中每个物体的轮廓、位置、接触和形变。"
                "只要任一物体发生可见位移、按压形变、接触开始/结束或遮挡变化，就不得将整个片段描述为静止，"
                "必须拆分阶段并记录对应事件。相机不动不代表物体不动。\n\n"
                "4.1) 同类物体优先按外观细节区分并命名（颜色、纹理、形状、大小、破损/污渍/标记等）。\n"
                "4.2) objects 默认只包含真实物体实例名，不用纯方位/区域短语当物体名。\n"
                "4.3) 仅当外观确实无法区分时，允许使用“左侧的杯子/右侧的杯子”等方位限定名，并保持全片一致。\n"
                "4.4) 方位语义仍应写入 relative_position_timeline 的 horizontal/vertical/depth 字段。\n\n"
                "5) 输出该片段内 x 轴深度排序：按世界坐标系 x 从正到负排列物体（越负越远）。\n\n"
                "6) 必须输出该片段内关键物体对的相对方位关系(relative_position_timeline)，"
                "至少包含 horizontal(left/right/overlap) 和 vertical(above/below/overlap)，可附加 depth(front/behind/same_depth)。\n\n"
                f"segment_id: {segment_id}\n"
                f"object_text_hint: {object_text}\n"
                f"relation_hint: {relation_hint}\n"
                f"segment_frame_tokens: {segment_tokens}\n"
                f"frame_mapping:\n{chr(10).join(token_lines)}\n\n"
                f"previous_segment_memory:\n{memory_text}\n\n"
                "返回 JSON（键名如下）：\n"
                "{\n"
                '  "segment_id": string,\n'
                '  "objects": [string],\n'
                '  "relations": [{"subject":string,"predicate":string,"obj":string,"phase":string,"confidence":number,"distance_hint":string}],\n'
                '  "events": [{"type":string,"subject":string,"obj":string,"frame_token":string,"phase":string,"confidence":number}],\n'
                '  "x_depth_order_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"order_positive_to_negative":[string],"confidence":number,"notes":string}],\n'
                '  "relative_position_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"pairs":[{"subject":string,"obj":string,"horizontal":"left|right|overlap|unknown","vertical":"above|below|overlap|unknown","depth":"front|behind|same_depth|unknown","confidence":number,"notes":string}],"notes":string}],\n'
                '  "segment_memory": {"summary":string,"active_relations":[string],"constraints":[string]},\n'
                '  "notes": string\n'
                "}\n"
                "请使用中文填写自由文本字段。"
            )
        else:
            system_prompt = (
                "You are Agent-0 for long-video temporal scene understanding in 4D editing. "
                "You are analyzing ONE short segment from a longer video. "
                "Output JSON only. Never output numeric 6DoF pose."
            )
            user_prompt = (
                "Analyze ONLY this segment and return a segment-level TSG fragment.\n"
                "Rules:\n"
                "1) No numeric pose/distance.\n"
                "2) distance_hint in {near_zero, near, mid, far, unknown}.\n"
                "3) Focus on events that start/end/change in this segment.\n"
                "4) Keep object naming consistent with previous memory if possible.\n\n"
                "4.0) Explicitly compare each object's contour, position, contact, and deformation in the first, middle, and last frame. "
                "If any object visibly translates, deforms under pressure, starts/ends contact, or changes occlusion, do not label the whole segment static; "
                "split phases and record the event. A static camera does not imply static objects.\n\n"
                "4.1) For same-category objects, first distinguish and name by visible appearance details "
                "(color, texture, shape, size, damage/stain/markings, etc.).\n"
                "4.2) `objects` should primarily contain real object instance names, not pure spatial/region phrases.\n"
                "4.3) Exception: if objects are truly visually indistinguishable, you may use spatial qualifiers "
                "(e.g., `left cup`, `right cup`) and keep them consistent across the whole video.\n"
                "4.4) Location semantics should still be represented in relative_position_timeline "
                "(horizontal/vertical/depth).\n\n"
                "5) Output x-axis depth ordering in this segment: object order from world +X to -X (more negative is farther).\n\n"
                "6) You must output key pairwise relative positions in this segment (relative_position_timeline),"
                " including horizontal(left/right/overlap) and vertical(above/below/overlap), optionally depth(front/behind/same_depth).\n\n"
                f"segment_id: {segment_id}\n"
                f"object_text_hint: {object_text}\n"
                f"relation_hint: {relation_hint}\n"
                f"segment_frame_tokens: {segment_tokens}\n"
                f"frame_mapping:\n{chr(10).join(token_lines)}\n\n"
                f"previous_segment_memory:\n{memory_text}\n\n"
                "Return JSON with keys:\n"
                "{\n"
                '  "segment_id": string,\n'
                '  "objects": [string],\n'
                '  "relations": [{"subject":string,"predicate":string,"obj":string,"phase":string,"confidence":number,"distance_hint":string}],\n'
                '  "events": [{"type":string,"subject":string,"obj":string,"frame_token":string,"phase":string,"confidence":number}],\n'
                '  "x_depth_order_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"order_positive_to_negative":[string],"confidence":number,"notes":string}],\n'
                '  "relative_position_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"pairs":[{"subject":string,"obj":string,"horizontal":"left|right|overlap|unknown","vertical":"above|below|overlap|unknown","depth":"front|behind|same_depth|unknown","confidence":number,"notes":string}],"notes":string}],\n'
                '  "segment_memory": {"summary":string,"active_relations":[string],"constraints":[string]},\n'
                '  "notes": string\n'
                "}"
            )
        return system_prompt, user_prompt

    def _build_agent0_aggregate_prompt(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_tokens: list[str],
        segment_fragments: list[dict[str, Any]],
        segment_memories: list[dict[str, Any]],
        language: str = "en",
        ikg_enabled: bool = True,
    ) -> tuple[str, str]:
        frag_text = json.dumps(segment_fragments, ensure_ascii=False)
        mem_text = json.dumps(segment_memories, ensure_ascii=False)
        if not bool(ikg_enabled):
            if str(language) == "zh":
                system_prompt = (
                    "你是 Agent-0 的 object-only 全局聚合器。"
                    "请只合并物体实例名称，不要生成交互关系、事件、约束或时间线。只输出 JSON。"
                )
                user_prompt = (
                    "给定同一长视频的片段级 object-only 结果与记忆，请完成全局物体列表合并。\n"
                    "要求：\n"
                    "1) 去重物体名称。\n"
                    "2) 保留按外观细节区分的真实物体实例名。\n"
                    "3) 删除背景、纯方位区域、动作、状态或“无”等伪物体。\n"
                    "4) 不要输出 relations/events/constraints/x_depth_order_timeline/relative_position_timeline。\n\n"
                    f"object_text_hint: {object_text}\n"
                    f"all_frame_tokens: {frame_tokens}\n"
                    f"segment_fragments_json: {frag_text}\n"
                    f"segment_memories_json: {mem_text}\n\n"
                    "返回 JSON（键名如下）：\n"
                    "{\n"
                    '  "objects": [string],\n'
                    '  "object_meta": [{"id":string,"name":string,"category":string,"attributes":[string]}],\n'
                    '  "relations": [],\n'
                    '  "events": [],\n'
                    '  "constraints": [],\n'
                    '  "x_depth_order_timeline": [],\n'
                    '  "relative_position_timeline": [],\n'
                    '  "notes": string\n'
                    "}\n"
                    "请使用中文填写自由文本字段。"
                )
            else:
                system_prompt = (
                    "You are Agent-0 object-only global aggregator. "
                    "Merge object instance names only. Do not generate interactions, events, constraints, or timelines. Output JSON only."
                )
                user_prompt = (
                    "Given object-only segment fragments and memories from one long video, merge the global object list.\n"
                    "Requirements:\n"
                    "1) Deduplicate object names.\n"
                    "2) Keep real object instance names distinguished by appearance details.\n"
                    "3) Remove background, pure spatial regions, actions, states, and `none` pseudo-objects.\n"
                    "4) Do not output relations/events/constraints/x_depth_order_timeline/relative_position_timeline.\n\n"
                    f"object_text_hint: {object_text}\n"
                    f"all_frame_tokens: {frame_tokens}\n"
                    f"segment_fragments_json: {frag_text}\n"
                    f"segment_memories_json: {mem_text}\n\n"
                    "Return JSON with keys:\n"
                    "{\n"
                    '  "objects": [string],\n'
                    '  "object_meta": [{"id":string,"name":string,"category":string,"attributes":[string]}],\n'
                    '  "relations": [],\n'
                    '  "events": [],\n'
                    '  "constraints": [],\n'
                    '  "x_depth_order_timeline": [],\n'
                    '  "relative_position_timeline": [],\n'
                    '  "notes": string\n'
                    "}"
                )
            return system_prompt, user_prompt
        if str(language) == "zh":
            system_prompt = (
                "你是 Agent-0 的全局聚合器，用于长视频理解。"
                "请将多个片段结果合并成一个时序场景图，只输出 JSON。"
            )
            user_prompt = (
                "给定同一长视频的片段级结果与记忆，请完成全局合并。\n"
                "要求：\n"
                "1) 去重物体名称。\n"
                "2) 按时间一致性解决重叠冲突。\n"
                "3) 仅保留符号化空间/交互约束（不输出数值位姿）。\n\n"
                "3.0) 不得因多数片段外观相似而抹掉短暂动作。若任一片段报告位移、按压形变、接触变化或遮挡变化，"
                "全局结果必须保留相应阶段和事件；只有所有片段都明确无变化时，才能输出全程静止。\n\n"
                "3.1) 合并时优先保留按外观细节区分的实例名（颜色、纹理、形状、大小、标记等）。\n"
                "3.2) 若同类物体外观确实无法区分，可保留“左侧的杯子/右侧的杯子”等方位限定名，但必须时序一致、语义稳定。\n"
                "3.3) 纯方位区域伪物体（如“杯子右侧区域”）应删除；方位语义保留在 relative_position_timeline 中。\n\n"
                "4) 合并并输出分时段 x 轴深度排序，按世界坐标系 x 从正到负（越负越远）。\n\n"
                "5) 合并并输出分时段 relative_position_timeline，明确关键物体对的左右/上下/前后关系。\n\n"
                f"object_text_hint: {object_text}\n"
                f"relation_hint: {relation_hint}\n"
                f"all_frame_tokens: {frame_tokens}\n"
                f"segment_fragments_json: {frag_text}\n"
                f"segment_memories_json: {mem_text}\n\n"
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
                "You are Agent-0 global aggregator for long-video understanding. "
                "Merge segment fragments into a single Temporal Scene Graph. Output JSON only."
            )
            user_prompt = (
                "Given segment-level fragments and memories from one long video, merge them.\n"
                "Requirements:\n"
                "1) Deduplicate object names.\n"
                "2) Resolve overlap conflicts by temporal consistency.\n"
                "3) Keep symbolic spatial/interaction constraints only (no numeric poses).\n\n"
                "3.0) Do not erase a brief action because most segments look similar. If any segment reports translation, pressure deformation, "
                "contact change, or occlusion change, preserve the corresponding phase and event globally. Mark the whole video static only when every segment explicitly reports no change.\n\n"
                "3.1) During merge, prefer instance names distinguished by appearance details "
                "(color, texture, shape, size, markings, etc.).\n"
                "3.2) If same-category objects are truly visually indistinguishable, spatially-qualified names "
                "(`left cup`, `right cup`) are allowed, but must stay temporally consistent.\n"
                "3.3) Remove pure spatial-region pseudo-objects (e.g., `right-side area of cup`); "
                "keep spatial semantics in relative_position_timeline.\n\n"
                "4) Merge and output time-sliced x-axis depth order from world +X to -X (more negative means farther).\n\n"
                "5) Merge and output time-sliced relative_position_timeline with explicit pairwise left/right, above/below, and front/behind relations.\n\n"
                f"object_text_hint: {object_text}\n"
                f"relation_hint: {relation_hint}\n"
                f"all_frame_tokens: {frame_tokens}\n"
                f"segment_fragments_json: {frag_text}\n"
                f"segment_memories_json: {mem_text}\n\n"
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
        return system_prompt, user_prompt

    def _build_agent0_depth_timeline_fill_prompt(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_tokens: list[str],
        current_scene_graph: dict[str, Any],
        segment_memories: list[dict[str, Any]],
        language: str = "en",
    ) -> tuple[str, str]:
        g_compact = dict(current_scene_graph or {})
        if len(json.dumps(g_compact, ensure_ascii=False)) > 3500:
            g_compact = {
                "objects": list(g_compact.get("objects", []) or [])[:20],
                "relations": list(g_compact.get("relations", []) or [])[:24],
                "events": list(g_compact.get("events", []) or [])[:16],
                "phases": list(g_compact.get("phases", []) or [])[:8],
                "constraints": list(g_compact.get("constraints", []) or [])[:16],
            }
        mem_compact = list(segment_memories or [])[:16]
        if str(language) == "zh":
            system_prompt = (
                "你是 Agent-0 的深度排序补全器。"
                "目标：补全分时段 x 轴深度排序。只输出 JSON。"
            )
            user_prompt = (
                "请根据当前 scene_graph 与片段记忆，补全 x 轴深度排序。\n"
                "要求：\n"
                "1) 世界坐标系方向：按 +X 到 -X 排序（越负越远）。\n"
                "2) 给出分时段结果，frame_range 尽量覆盖整个视频。\n"
                "3) 若信息不足，可给单个全局时段，但不要返回空列表。\n\n"
                f"object_text_hint: {object_text}\n"
                f"relation_hint: {relation_hint}\n"
                f"frame_tokens: {frame_tokens}\n"
                f"current_scene_graph_json: {json.dumps(g_compact, ensure_ascii=False)}\n"
                f"segment_memories_json: {json.dumps(mem_compact, ensure_ascii=False)}\n\n"
                "返回 JSON：\n"
                "{\n"
                '  "x_depth_order_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"order_positive_to_negative":[string],"confidence":number,"notes":string}],\n'
                '  "notes": string\n'
                "}"
            )
        else:
            system_prompt = (
                "You are Agent-0 depth-order completion helper."
                "Goal: complete time-sliced x-axis depth ordering. Return JSON only."
            )
            user_prompt = (
                "Fill missing x-axis depth ordering from current scene_graph and segment memories.\n"
                "Requirements:\n"
                "1) World-axis convention: order from +X to -X (more negative is farther).\n"
                "2) Return time-sliced results with frame_range covering the video timeline.\n"
                "3) If uncertain, provide one global range instead of empty list.\n\n"
                f"object_text_hint: {object_text}\n"
                f"relation_hint: {relation_hint}\n"
                f"frame_tokens: {frame_tokens}\n"
                f"current_scene_graph_json: {json.dumps(g_compact, ensure_ascii=False)}\n"
                f"segment_memories_json: {json.dumps(mem_compact, ensure_ascii=False)}\n\n"
                "Return JSON:\n"
                "{\n"
                '  "x_depth_order_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"order_positive_to_negative":[string],"confidence":number,"notes":string}],\n'
                '  "notes": string\n'
                "}"
            )
        return system_prompt, user_prompt

    def _build_agent0_relative_timeline_fill_prompt(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_tokens: list[str],
        current_scene_graph: dict[str, Any],
        segment_memories: list[dict[str, Any]],
        language: str = "en",
    ) -> tuple[str, str]:
        g_compact = dict(current_scene_graph or {})
        if len(json.dumps(g_compact, ensure_ascii=False)) > 3500:
            g_compact = {
                "objects": list(g_compact.get("objects", []) or [])[:20],
                "relations": list(g_compact.get("relations", []) or [])[:24],
                "events": list(g_compact.get("events", []) or [])[:16],
                "phases": list(g_compact.get("phases", []) or [])[:8],
                "constraints": list(g_compact.get("constraints", []) or [])[:16],
                "x_depth_order_timeline": list(g_compact.get("x_depth_order_timeline", []) or [])[:12],
            }
        mem_compact = list(segment_memories or [])[:16]
        if str(language) == "zh":
            system_prompt = (
                "你是 Agent-0 的相对位置时间线补全器。"
                "目标：补全分时段的关键物体对相对位置关系。只输出 JSON。"
            )
            user_prompt = (
                "请根据当前 scene_graph 与片段记忆，补全 relative_position_timeline。\n"
                "要求：\n"
                "1) 每个时段输出关键物体对的 horizontal(left/right/overlap) 与 vertical(above/below/overlap)。\n"
                "2) 若能判断，再补 depth(front/behind/same_depth)。\n"
                "3) frame_range 尽量覆盖整个视频；不确定时可给一个全局时段，但不要空列表。\n\n"
                f"object_text_hint: {object_text}\n"
                f"relation_hint: {relation_hint}\n"
                f"frame_tokens: {frame_tokens}\n"
                f"current_scene_graph_json: {json.dumps(g_compact, ensure_ascii=False)}\n"
                f"segment_memories_json: {json.dumps(mem_compact, ensure_ascii=False)}\n\n"
                "返回 JSON：\n"
                "{\n"
                '  "relative_position_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"pairs":[{"subject":string,"obj":string,"horizontal":"left|right|overlap|unknown","vertical":"above|below|overlap|unknown","depth":"front|behind|same_depth|unknown","confidence":number,"notes":string}],"notes":string}],\n'
                '  "notes": string\n'
                "}"
            )
        else:
            system_prompt = (
                "You are Agent-0 relative-position timeline completion helper."
                "Goal: complete time-sliced pairwise relative positions. Return JSON only."
            )
            user_prompt = (
                "Fill missing relative_position_timeline from current scene_graph and segment memories.\n"
                "Requirements:\n"
                "1) For each range, output key object-pair horizontal(left/right/overlap) and vertical(above/below/overlap).\n"
                "2) Add depth(front/behind/same_depth) when inferable.\n"
                "3) Frame ranges should cover the video timeline; if uncertain, provide one global range instead of empty list.\n\n"
                f"object_text_hint: {object_text}\n"
                f"relation_hint: {relation_hint}\n"
                f"frame_tokens: {frame_tokens}\n"
                f"current_scene_graph_json: {json.dumps(g_compact, ensure_ascii=False)}\n"
                f"segment_memories_json: {json.dumps(mem_compact, ensure_ascii=False)}\n\n"
                "Return JSON:\n"
                "{\n"
                '  "relative_position_timeline": [{"frame_range":string,"start_frame_index":int,"end_frame_index":int,"pairs":[{"subject":string,"obj":string,"horizontal":"left|right|overlap|unknown","vertical":"above|below|overlap|unknown","depth":"front|behind|same_depth|unknown","confidence":number,"notes":string}],"notes":string}],\n'
                '  "notes": string\n'
                "}"
            )
        return system_prompt, user_prompt

    @staticmethod
    def _coerce_x_depth_timeline(raw: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    item = dict(v)
                    item.setdefault("frame_range", str(k).strip())
                    rows.append(item)
                elif isinstance(v, list):
                    rows.append({"frame_range": str(k).strip(), "order_positive_to_negative": list(v)})
        elif isinstance(raw, list):
            for it in raw:
                if isinstance(it, dict):
                    rows.append(dict(it))
        out: list[dict[str, Any]] = []
        for i, item in enumerate(rows):
            frame_range = str(item.get("frame_range", item.get("range", f"range_{i}"))).strip() or f"range_{i}"
            order_raw = item.get(
                "order_positive_to_negative",
                item.get("x_axis_positive_to_negative", item.get("x_axis_order_pos_to_neg", item.get("x轴方向由正到负", []))),
            )
            if isinstance(order_raw, str):
                order = [x.strip() for x in re.split(r"\s*(?:->|→|>|,|，)\s*", order_raw) if str(x).strip()]
            elif isinstance(order_raw, list):
                order = [str(x).strip() for x in order_raw if str(x).strip()]
            else:
                order = []
            if not order:
                continue
            rec: dict[str, Any] = {"frame_range": frame_range, "order_positive_to_negative": order}
            for k in ["start_frame_index", "end_frame_index"]:
                try:
                    if k in item:
                        rec[k] = int(item.get(k))
                except Exception:
                    pass
            try:
                if "confidence" in item:
                    rec["confidence"] = float(max(0.0, min(1.0, float(item.get("confidence")))))
            except Exception:
                pass
            notes = str(item.get("notes", item.get("summary", ""))).strip()
            if notes:
                rec["notes"] = notes
            out.append(rec)
            if len(out) >= 24:
                break
        return out

    @staticmethod
    def _coerce_relative_axis(raw: Any, *, axis: str) -> str:
        s = str(raw or "").strip().lower()
        if axis == "horizontal":
            if s in {"left", "left_of", "on_left", "subject_left_of_obj", "target_left_of_other"}:
                return "left"
            if s in {"right", "right_of", "on_right", "subject_right_of_obj", "target_right_of_other"}:
                return "right"
            if s in {"overlap", "same_x", "aligned_x", "aligned", "centered", "intersect"}:
                return "overlap"
            return "unknown"
        if axis == "vertical":
            if s in {"above", "up", "upper", "on_top", "on_top_of", "subject_above_obj", "target_above_other"}:
                return "above"
            if s in {"below", "down", "under", "lower", "underneath", "subject_below_obj", "target_below_other"}:
                return "below"
            if s in {"overlap", "same_y", "aligned_y", "aligned", "intersect"}:
                return "overlap"
            return "unknown"
        if s in {"front", "in_front", "in_front_of", "target_in_front", "closer"}:
            return "front"
        if s in {"behind", "back", "back_of", "target_behind", "farther"}:
            return "behind"
        if s in {"same_depth", "equal_depth", "aligned_depth", "overlap"}:
            return "same_depth"
        return "unknown"

    @classmethod
    def _coerce_relative_position_timeline(cls, raw: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    item = dict(v)
                    item.setdefault("frame_range", str(k).strip())
                    rows.append(item)
                elif isinstance(v, list):
                    rows.append({"frame_range": str(k).strip(), "pairs": list(v)})
        elif isinstance(raw, list):
            for it in raw:
                if isinstance(it, dict):
                    rows.append(dict(it))

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for i, item in enumerate(rows):
            frame_range = str(item.get("frame_range", item.get("range", f"range_{i}"))).strip() or f"range_{i}"
            pairs_raw = item.get("pairs", item.get("relations", item.get("relative_pairs", [])))
            if not isinstance(pairs_raw, list):
                pairs_raw = []
            pairs: list[dict[str, Any]] = []
            pair_seen: set[str] = set()
            for p in pairs_raw:
                if not isinstance(p, dict):
                    continue
                sub = str(p.get("subject", p.get("src", p.get("a", "")))).strip()
                obj = str(p.get("obj", p.get("object", p.get("dst", p.get("b", ""))))).strip()
                if not (sub and obj):
                    continue
                h = cls._coerce_relative_axis(
                    p.get("horizontal", p.get("left_right", p.get("lr", p.get("x_relation", "unknown")))),
                    axis="horizontal",
                )
                v = cls._coerce_relative_axis(
                    p.get("vertical", p.get("up_down", p.get("ud", p.get("y_relation", "unknown")))),
                    axis="vertical",
                )
                d = cls._coerce_relative_axis(
                    p.get("depth", p.get("front_back", p.get("fb", p.get("z_relation", "unknown")))),
                    axis="depth",
                )
                pred = str(p.get("predicate", "")).strip()
                notes = str(p.get("notes", p.get("evidence", ""))).strip()
                try:
                    conf = float(p.get("confidence", 0.5))
                except Exception:
                    conf = 0.5
                p_key = f"{sub}|{obj}|{h}|{v}|{d}|{pred}"
                if p_key in pair_seen:
                    continue
                pair_seen.add(p_key)
                rec_p: dict[str, Any] = {
                    "subject": sub,
                    "obj": obj,
                    "horizontal": h,
                    "vertical": v,
                    "depth": d,
                    "confidence": float(max(0.0, min(1.0, conf))),
                }
                if pred:
                    rec_p["predicate"] = pred
                if notes:
                    rec_p["notes"] = notes
                pairs.append(rec_p)
                if len(pairs) >= 32:
                    break

            if not pairs:
                continue
            rec: dict[str, Any] = {"frame_range": frame_range, "pairs": pairs}
            for k in ["start_frame_index", "end_frame_index"]:
                try:
                    if k in item:
                        rec[k] = int(item.get(k))
                except Exception:
                    pass
            try:
                if "confidence" in item:
                    rec["confidence"] = float(max(0.0, min(1.0, float(item.get("confidence")))))
            except Exception:
                pass
            notes = str(item.get("notes", item.get("summary", ""))).strip()
            if notes:
                rec["notes"] = notes
            key = json.dumps({"frame_range": frame_range, "pairs": pairs}, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)
            if len(out) >= 24:
                break
        return out

    @staticmethod
    def _parse_relation_triplet_text(text: str) -> tuple[str, str, str] | None:
        s = str(text or "").strip()
        if not s:
            return None
        for pat in [r"\s*-\s*", r"\s*—\s*", r"\s*->\s*", r"\s*→\s*", r"\s*>\s*"]:
            parts = [x.strip() for x in re.split(pat, s) if str(x).strip()]
            if len(parts) >= 3:
                sub = parts[0]
                pred = parts[1]
                obj = parts[2] if len(parts) == 3 else "".join(parts[2:]).strip()
                if sub and pred and obj:
                    return sub, pred, obj
        return None

    def _enrich_segment_fragment_from_memory(self, seg: dict[str, Any]) -> dict[str, Any]:
        out = dict(seg or {})
        out.setdefault("objects", [])
        out.setdefault("relations", [])
        out.setdefault("events", [])
        out.setdefault("constraints", [])
        out.setdefault("x_depth_order_timeline", [])
        out.setdefault("relative_position_timeline", [])
        out.setdefault("notes", "")

        objects: list[str] = []
        seen_obj: set[str] = set()
        for o in list(out.get("objects", []) or []):
            s = str(o).strip()
            if not s or s in seen_obj:
                continue
            seen_obj.add(s)
            objects.append(s)

        relations: list[dict[str, Any]] = []
        rel_seen: set[str] = set()
        for r in list(out.get("relations", []) or []):
            if not isinstance(r, dict):
                continue
            sub = str(r.get("subject", "")).strip()
            pred = str(r.get("predicate", "")).strip()
            obj = str(r.get("obj", r.get("object", ""))).strip()
            if not (sub and pred and obj):
                continue
            key = json.dumps({"s": sub, "p": pred, "o": obj, "ph": str(r.get("phase", "segment"))}, ensure_ascii=False, sort_keys=True)
            if key in rel_seen:
                continue
            rel_seen.add(key)
            rel = dict(r)
            rel.setdefault("phase", "segment")
            rel.setdefault("confidence", 0.6)
            rel.setdefault("distance_hint", "unknown")
            relations.append(rel)
            for n in [sub, obj]:
                if n and n not in seen_obj:
                    seen_obj.add(n)
                    objects.append(n)

        constraints: list[dict[str, Any]] = []
        cst_seen: set[str] = set()
        for c in list(out.get("constraints", []) or []):
            if isinstance(c, str):
                c = {"rule": c}
            if not isinstance(c, dict):
                continue
            rule = str(c.get("rule", "")).strip()
            if not rule:
                continue
            target_pair = list(c.get("target_pair", [])) if isinstance(c.get("target_pair", []), list) else []
            key = json.dumps({"rule": rule, "target_pair": target_pair, "phase": str(c.get("phase", "segment"))}, ensure_ascii=False, sort_keys=True)
            if key in cst_seen:
                continue
            cst_seen.add(key)
            constraints.append(
                {
                    "rule": rule,
                    "target_pair": target_pair,
                    "phase": str(c.get("phase", "segment")).strip() or "segment",
                }
            )

        seg_mem = out.get("segment_memory", {})
        if isinstance(seg_mem, dict):
            active_rel = list(seg_mem.get("active_relations", []) or [])
            for x in active_rel:
                tri = self._parse_relation_triplet_text(str(x))
                if tri is None:
                    continue
                sub, pred, obj = tri
                key = json.dumps({"s": sub, "p": pred, "o": obj, "ph": "segment"}, ensure_ascii=False, sort_keys=True)
                if key in rel_seen:
                    continue
                rel_seen.add(key)
                relations.append(
                    {
                        "subject": sub,
                        "predicate": pred,
                        "obj": obj,
                        "phase": "segment",
                        "confidence": 0.6,
                        "distance_hint": "unknown",
                    }
                )
                for n in [sub, obj]:
                    if n and n not in seen_obj:
                        seen_obj.add(n)
                        objects.append(n)

            for x in list(seg_mem.get("constraints", []) or []):
                rule = str(x).strip()
                if not rule:
                    continue
                key = json.dumps({"rule": rule, "target_pair": [], "phase": "segment"}, ensure_ascii=False, sort_keys=True)
                if key in cst_seen:
                    continue
                cst_seen.add(key)
                constraints.append({"rule": rule, "target_pair": [], "phase": "segment"})

            if not str(out.get("notes", "")).strip():
                out["notes"] = str(seg_mem.get("summary", "")).strip()

        out["objects"] = objects
        out["relations"] = relations
        out["constraints"] = constraints
        out["x_depth_order_timeline"] = self._coerce_x_depth_timeline(out.get("x_depth_order_timeline", []))
        out["relative_position_timeline"] = self._coerce_relative_position_timeline(
            out.get("relative_position_timeline", out.get("relative_spatial_timeline", []))
        )
        return out

    @staticmethod
    def _scene_graph_structural_score(graph: dict[str, Any]) -> int:
        if not isinstance(graph, dict):
            return 0
        def _n(key: str) -> int:
            v = graph.get(key, [])
            return len(v) if isinstance(v, list) else 0
        return int(
            _n("objects")
            + 2 * _n("relations")
            + _n("events")
            + _n("constraints")
            + _n("x_depth_order_timeline")
            + _n("relative_position_timeline")
        )

    @classmethod
    def _fallback_merge_segments(cls, segment_fragments: list[dict[str, Any]]) -> dict[str, Any]:
        obj_set: set[str] = set()
        relations: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        constraints: list[dict[str, Any]] = []
        x_timeline: list[dict[str, Any]] = []
        relative_timeline: list[dict[str, Any]] = []
        notes: list[str] = []
        rel_seen: set[str] = set()
        evt_seen: set[str] = set()
        cst_seen: set[str] = set()
        depth_seen: set[str] = set()
        relpos_seen: set[str] = set()
        for seg in segment_fragments:
            seg_id = str(seg.get("segment_id", "")).strip()
            for o in seg.get("objects", []):
                so = str(o).strip()
                if so:
                    obj_set.add(so)
            for r in seg.get("relations", []):
                if not isinstance(r, dict):
                    continue
                key = json.dumps(
                    {
                        "s": r.get("subject"),
                        "p": r.get("predicate"),
                        "o": r.get("obj"),
                        "ph": r.get("phase"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if key in rel_seen:
                    continue
                rel_seen.add(key)
                relations.append(r)
                try:
                    sub = str(r.get("subject", "")).strip()
                    obj = str(r.get("obj", r.get("object", ""))).strip()
                    if sub:
                        obj_set.add(sub)
                    if obj:
                        obj_set.add(obj)
                except Exception:
                    pass
            for e in seg.get("events", []):
                if not isinstance(e, dict):
                    continue
                key = json.dumps(
                    {
                        "t": e.get("type"),
                        "s": e.get("subject"),
                        "o": e.get("obj"),
                        "f": e.get("frame_token"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if key in evt_seen:
                    continue
                evt_seen.add(key)
                events.append(e)
                try:
                    sub = str(e.get("subject", "")).strip()
                    obj = str(e.get("obj", e.get("object", ""))).strip()
                    if sub:
                        obj_set.add(sub)
                    if obj:
                        obj_set.add(obj)
                except Exception:
                    pass
            for c in seg.get("constraints", []):
                if isinstance(c, str):
                    c = {"rule": c}
                if not isinstance(c, dict):
                    continue
                rule = str(c.get("rule", "")).strip()
                if not rule:
                    continue
                pair = list(c.get("target_pair", [])) if isinstance(c.get("target_pair", []), list) else []
                ph = str(c.get("phase", "global")).strip() or "global"
                key = json.dumps({"rule": rule, "pair": pair, "phase": ph}, ensure_ascii=False, sort_keys=True)
                if key in cst_seen:
                    continue
                cst_seen.add(key)
                constraints.append({"rule": rule, "target_pair": pair, "phase": ph})
                for n in pair:
                    sn = str(n).strip()
                    if sn:
                        obj_set.add(sn)
            n = str(seg.get("notes", "")).strip()
            if n:
                notes.append(n)
            raw_depth = seg.get(
                "x_depth_order_timeline",
                seg.get("depth_order_timeline", seg.get("x_axis_depth_order_timeline", [])),
            )
            for item in cls._coerce_x_depth_timeline(raw_depth):
                fr = str(item.get("frame_range", seg_id)).strip() or seg_id
                order = [str(x).strip() for x in list(item.get("order_positive_to_negative", [])) if str(x).strip()]
                if not order:
                    continue
                key = json.dumps({"fr": fr, "order": order}, ensure_ascii=False, sort_keys=True)
                if key in depth_seen:
                    continue
                depth_seen.add(key)
                rec = {"frame_range": fr, "order_positive_to_negative": order}
                if "start_frame_index" in item:
                    try:
                        rec["start_frame_index"] = int(item.get("start_frame_index"))
                    except Exception:
                        pass
                if "end_frame_index" in item:
                    try:
                        rec["end_frame_index"] = int(item.get("end_frame_index"))
                    except Exception:
                        pass
                if "confidence" in item:
                    try:
                        rec["confidence"] = float(item.get("confidence"))
                    except Exception:
                        pass
                if str(item.get("notes", "")).strip():
                    rec["notes"] = str(item.get("notes", "")).strip()
                x_timeline.append(rec)

            raw_rel_pos = seg.get(
                "relative_position_timeline",
                seg.get("relative_spatial_timeline", seg.get("spatial_relation_timeline", [])),
            )
            for item in cls._coerce_relative_position_timeline(raw_rel_pos):
                fr = str(item.get("frame_range", seg_id)).strip() or seg_id
                pairs = list(item.get("pairs", []) or [])
                if not pairs:
                    continue
                key = json.dumps({"fr": fr, "pairs": pairs}, ensure_ascii=False, sort_keys=True)
                if key in relpos_seen:
                    continue
                relpos_seen.add(key)
                rec: dict[str, Any] = {"frame_range": fr, "pairs": pairs}
                if "start_frame_index" in item:
                    try:
                        rec["start_frame_index"] = int(item.get("start_frame_index"))
                    except Exception:
                        pass
                if "end_frame_index" in item:
                    try:
                        rec["end_frame_index"] = int(item.get("end_frame_index"))
                    except Exception:
                        pass
                if "confidence" in item:
                    try:
                        rec["confidence"] = float(item.get("confidence"))
                    except Exception:
                        pass
                if str(item.get("notes", "")).strip():
                    rec["notes"] = str(item.get("notes", "")).strip()
                relative_timeline.append(rec)
        return {
            "objects": sorted(obj_set),
            "relations": relations,
            "events": events,
            "object_meta": [],
            "phases": [],
            "constraints": constraints,
            "x_depth_order_timeline": x_timeline,
            "relative_position_timeline": relative_timeline,
            "notes": " | ".join(notes[:8]),
        }

    def _run_hierarchical_video_understanding(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_items: list[dict[str, Any]],
        frame_tokens: list[str],
        timeout: float,
        provider: str,
        chat_json_fn,
        language: str = "en",
        ikg_enabled: bool = True,
    ) -> dict[str, Any]:
        object_text_prompt = self._to_language_object_text(object_text, language)
        chunk_size, overlap = self._segment_config()
        ranges = self._segment_ranges(len(frame_items), chunk_size, overlap)
        if not ranges:
            return {"ok": False, "provider": provider, "error": "no_segments"}

        seg_fragments: list[dict[str, Any]] = []
        seg_memories: list[dict[str, Any]] = []

        for seg_i, (st, ed) in enumerate(ranges):
            seg_items = frame_items[st:ed]
            seg_tokens = frame_tokens[st:ed]
            vis_content, token_lines = self._prepare_visual_content(seg_items)
            # Hard cap to guarantee we never hit provider image-limit edge cases.
            if len(vis_content) > 49:
                vis_content = vis_content[:49]
                token_lines = token_lines[:49]
                seg_tokens = seg_tokens[:49]
            seg_id = f"{st}_{ed - 1}"
            sp, up = self._build_agent0_segment_prompt(
                object_text=object_text_prompt,
                relation_hint=relation_hint,
                segment_id=seg_id,
                segment_tokens=seg_tokens,
                memory_context=seg_memories,
                token_lines=token_lines,
                language=language,
                ikg_enabled=bool(ikg_enabled),
            )
            if not bool(ikg_enabled):
                if str(language).strip().lower() == "zh":
                    post_visual_prompt = (
                        up
                        + "\n\n【要求重申】请只输出真实物体实例列表；不要分析或输出交互关系、事件、约束、深度排序或相对位置时间线。"
                    )
                else:
                    post_visual_prompt = (
                        up
                        + "\n\n[Requirement Restatement] Output only real object instances. "
                        + "Do not analyze or output interactions, events, constraints, depth order, or relative-position timelines."
                    )
            elif str(language).strip().lower() == "zh":
                post_visual_prompt = (
                    up
                    + "\n\n【要求重申】请在看完以上全部视频帧后，再严格按上述要求输出 JSON。"
                    + "同类物体优先按外观细节区分命名（颜色/纹理/形状/大小/标记等）；"
                    + "仅当外观确实无法区分时，才允许“左侧的杯子/右侧的杯子”等方位限定名，并保持全片一致；"
                    + "相对方位语义仍需写入 relative_position_timeline。"
                )
            else:
                post_visual_prompt = (
                    up
                    + "\n\n[Requirement Restatement] After reviewing all frames above, strictly follow the requirements and output JSON only. "
                    + "For same-category objects, first distinguish by appearance details (color/texture/shape/size/markings). "
                    + "Only if they are truly visually indistinguishable, allow spatially-qualified names "
                    + "(`left cup` / `right cup`) and keep them consistent across the video. "
                    + "Relative spatial semantics must still be represented in relative_position_timeline."
                )
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": vis_content + [{"type": "text", "text": post_visual_prompt}]},
            ]
            ok, parsed_or_error = chat_json_fn(messages, timeout)
            if not ok:
                return {
                    "ok": False,
                    "provider": provider,
                    "error": str(parsed_or_error),
                    "segment_index": seg_i,
                    "segment_id": seg_id,
                }
            seg_out = parsed_or_error if isinstance(parsed_or_error, dict) else {}
            seg_out.setdefault("segment_id", seg_id)
            seg_out = self._enrich_segment_fragment_from_memory(seg_out)
            seg_fragments.append(seg_out)
            seg_mem = seg_out.get("segment_memory", {})
            if isinstance(seg_mem, dict):
                seg_memories.append({"segment_id": seg_id, **seg_mem})
            else:
                seg_memories.append({"segment_id": seg_id, "summary": str(seg_mem)})

        # Global aggregation (text-only) on segment memories/fragments.
        asp, aup = self._build_agent0_aggregate_prompt(
            object_text=object_text_prompt,
            relation_hint=relation_hint,
            frame_tokens=frame_tokens,
            segment_fragments=seg_fragments,
            segment_memories=seg_memories,
            language=language,
            ikg_enabled=bool(ikg_enabled),
        )
        agg_messages = [
            {"role": "system", "content": asp},
            {"role": "user", "content": aup},
        ]
        fallback_graph = self._fallback_merge_segments(seg_fragments)
        ok, parsed_or_error = chat_json_fn(agg_messages, timeout)
        if ok and isinstance(parsed_or_error, dict):
            candidate = dict(parsed_or_error)
            if self._scene_graph_structural_score(fallback_graph) > self._scene_graph_structural_score(candidate):
                final_graph = fallback_graph
            else:
                final_graph = candidate
        else:
            final_graph = fallback_graph
        if isinstance(final_graph, dict) and not bool(ikg_enabled):
            final_graph = self.strip_interaction_knowledge_graph(final_graph)
        if isinstance(final_graph, dict) and bool(ikg_enabled):
            cur_timeline = final_graph.get("x_depth_order_timeline", [])
            if (not isinstance(cur_timeline, list) or len(cur_timeline) <= 0) and self._scene_graph_structural_score(final_graph) > 0:
                dsp, dup = self._build_agent0_depth_timeline_fill_prompt(
                    object_text=object_text_prompt,
                    relation_hint=relation_hint,
                    frame_tokens=frame_tokens,
                    current_scene_graph=final_graph,
                    segment_memories=seg_memories,
                    language=language,
                )
                d_messages = [{"role": "system", "content": dsp}, {"role": "user", "content": dup}]
                ok_d, parsed_d = chat_json_fn(d_messages, min(float(timeout), 120.0))
                if ok_d and isinstance(parsed_d, dict):
                    xt_raw = parsed_d.get(
                        "x_depth_order_timeline",
                        parsed_d.get("depth_order_timeline", parsed_d.get("x_axis_depth_order_timeline", [])),
                    )
                    xt = self._coerce_x_depth_timeline(xt_raw)
                    if xt:
                        final_graph["x_depth_order_timeline"] = xt
                    if (not str(final_graph.get("notes", "")).strip()) and str(parsed_d.get("notes", "")).strip():
                        final_graph["notes"] = str(parsed_d.get("notes", "")).strip()

            cur_rel_timeline = final_graph.get("relative_position_timeline", [])
            if (not isinstance(cur_rel_timeline, list) or len(cur_rel_timeline) <= 0) and self._scene_graph_structural_score(final_graph) > 0:
                rsp, rup = self._build_agent0_relative_timeline_fill_prompt(
                    object_text=object_text_prompt,
                    relation_hint=relation_hint,
                    frame_tokens=frame_tokens,
                    current_scene_graph=final_graph,
                    segment_memories=seg_memories,
                    language=language,
                )
                r_messages = [{"role": "system", "content": rsp}, {"role": "user", "content": rup}]
                ok_r, parsed_r = chat_json_fn(r_messages, min(float(timeout), 120.0))
                if ok_r and isinstance(parsed_r, dict):
                    rt_raw = parsed_r.get(
                        "relative_position_timeline",
                        parsed_r.get("relative_spatial_timeline", parsed_r.get("spatial_relation_timeline", [])),
                    )
                    rt = self._coerce_relative_position_timeline(rt_raw)
                    if rt:
                        final_graph["relative_position_timeline"] = rt
                    if (not str(final_graph.get("notes", "")).strip()) and str(parsed_r.get("notes", "")).strip():
                        final_graph["notes"] = str(parsed_r.get("notes", "")).strip()

        return {
            "ok": True,
            "provider": provider,
            "scene_graph": final_graph,
            "frames_used": len(frame_items),
            "segment_count": len(ranges),
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "ikg_enabled": bool(ikg_enabled),
            "ablation_mode": "" if bool(ikg_enabled) else "wo_IKG",
            "segments": [{"segment_id": f"{st}_{ed-1}", "start": st, "end": ed - 1} for st, ed in ranges],
            "segment_memories": seg_memories,
        }

    def _toolkit_shared_dir(self) -> str | None:
        shared_dir = getattr(self.app, "vlm_shared_dir", None) or os.getenv("VLM_SHARED_DIR")
        if not shared_dir:
            return None
        s = str(shared_dir).strip()
        if not s or s == "/path/to/shared/gpfs/vlm_requests":
            return None
        return s

    @staticmethod
    def _toolkit_module_root() -> Path:
        # viewer_plugins/vlm_assistor/vlm/client.py -> .../vlm_assistor/thirdparty/VLM_toolkit_cross_gpfs
        return Path(__file__).resolve().parents[1] / "thirdparty" / "VLM_toolkit_cross_gpfs"

    def _load_toolkit_client_cls(self) -> tuple[type[Any] | None, str | None]:
        root = self._toolkit_module_root()
        if not root.exists():
            return None, f"toolkit_root_missing:{root}"
        root_s = str(root.resolve())
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
        try:
            # Toolkit files use top-level imports: from config import ..., from protocol import ...
            # So we load client.py from toolkit root context instead of package import.
            mod_name = "_vlm_toolkit_cross_gpfs_client"
            spec = importlib.util.spec_from_file_location(mod_name, root / "client.py")
            if spec is None or spec.loader is None:
                return None, "toolkit_client_spec_unavailable"
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, "LLMClient", None)
            if cls is None:
                return None, "toolkit_LLMClient_missing"
            return cls, None
        except Exception as exc:
            return None, f"{type(exc).__name__}:{exc}"

    @staticmethod
    def _safe_read_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _probe_http_health(self, timeout: float = 2.0) -> tuple[bool, str]:
        base = self._base_url()
        if not base:
            return False, "vlm_agent_url_not_set"
        url = base + "/health"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                body = resp.read().decode("utf-8")
            return True, f"http_backend_ok:{url}:{body[:120]}"
        except urllib.error.URLError as exc:
            return False, f"http_backend_unreachable:{url}:{exc}"
        except Exception as exc:
            return False, f"http_backend_probe_failed:{url}:{type(exc).__name__}:{exc}"

    def _probe_openai_compatible_health(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        model: str,
        timeout: float = 3.0,
    ) -> tuple[bool, str]:
        if not base_url:
            return False, "local_openai_base_url_missing"
        base = str(base_url).rstrip("/")

        # Probe /models first.
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(base + "/models", method="GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                body = resp.read().decode("utf-8")
            return True, f"local_openai_models_ok:{base}/models:{body[:120]}"
        except Exception:
            pass

        # Some OpenAI-compatible servers (e.g. sglang) expose /model_info.
        req = urllib.request.Request(base + "/model_info", method="GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                body = resp.read().decode("utf-8")
            return True, f"local_openai_model_info_ok:{base}/model_info:{body[:120]}"
        except Exception:
            pass

        # Fallback: tiny /chat/completions probe.
        payload = {
            "model": str(model),
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 4,
            "temperature": 0.0,
        }
        resp = self._post_chat_completions(
            payload,
            base_url=base,
            api_key=api_key,
            timeout=timeout,
            require_api_key=False,
        )
        if isinstance(resp, dict):
            return True, f"local_openai_chat_ok:{base}/chat/completions"
        return False, f"local_openai_probe_failed:{base}"

    def preflight_video_understanding(self, *, backend_mode: str, use_gpt4o: bool) -> dict[str, Any]:
        """Pre-check available backend paths for Agent0 and return actionable diagnostics."""
        errors: list[str] = []
        warnings: list[str] = []
        hints: list[str] = []
        checks: list[str] = []

        shared_dir = self._toolkit_shared_dir()
        llm_cls, import_err = self._load_toolkit_client_cls()
        toolkit_path = self._toolkit_module_root()
        lock_exists = False
        chunk_size, overlap = self._segment_config()
        checks.append(f"segment_strategy:chunk_size={chunk_size},overlap={overlap},max_images=49")

        mode = str(backend_mode or self._agent0_backend_mode())
        if mode not in {"official_api", "local_openai_compatible"}:
            mode = "official_api"

        toolkit_ready = bool(shared_dir and llm_cls)
        local_base_url = self._local_vlm_base_url()
        local_model = self._local_vlm_model()
        local_api_key = self._local_vlm_api_key()
        local_ok = False

        if mode == "official_api":
            if not toolkit_path.exists():
                warnings.append(f"toolkit_missing:{toolkit_path}")
            if not shared_dir:
                warnings.append("missing_VLM_SHARED_DIR")
                hints.append("Set VLM_SHARED_DIR to shared GPFS directory used by toolkit server/client.")
            else:
                p = Path(shared_dir).expanduser()
                checks.append(f"shared_dir:{p}")
                lock_path = p / "server.lock"
                if lock_path.exists():
                    lock_exists = True
                    info = self._safe_read_json(lock_path) or {}
                    holder = f"{info.get('hostname', 'unknown')}:{info.get('pid', 'unknown')}"
                    checks.append(f"toolkit_server_lock_found:{holder}")
                else:
                    warnings.append(f"toolkit_server_lock_missing:{lock_path}")
                    hints.append("Start toolkit server.py on the internet machine with the same --shared-dir.")
            if llm_cls is None:
                warnings.append("toolkit_client_import_failed")
                if import_err:
                    checks.append(f"toolkit_import_error:{import_err}")
                hints.append("Check viewer_plugins/vlm_assistor/thirdparty/VLM_toolkit_cross_gpfs is available.")
            if toolkit_ready:
                checks.append("toolkit_ready:true")

            # Direct OpenAI path is optional fallback. If no API key, mark as warning only.
            if not self._openai_api_key():
                warnings.append("OPENAI_API_KEY_missing_for_direct_fallback")
        else:
            if not local_base_url:
                warnings.append("local_openai_base_url_missing")
                hints.append("Set Local VLM base URL (e.g. http://127.0.0.1:22002/v1) in VLM Assistor panel.")
            checks.append(f"local_model:{local_model}")
            src_mode, src_value = self._normalize_local_video_source(self._local_vlm_video_url())
            checks.append(f"local_video_source_mode:{src_mode}")
            if src_mode == "invalid":
                warnings.append("local_video_source_invalid")
                hints.append("Set Local video URL to a valid file://video, http(s)://video, or file://<input_dir>.")
            if src_mode == "frame_dir" and src_value:
                checks.append(f"local_frame_source_dir:{src_value}")
            local_ok, local_msg = self._probe_openai_compatible_health(
                base_url=local_base_url,
                api_key=local_api_key,
                model=local_model,
                timeout=3.0,
            )
            checks.append(local_msg)
            if not local_ok:
                warnings.append("local_openai_unreachable_or_invalid")
                hints.append("Check local VLM service is running and model name matches deployed endpoint.")

        http_ok, http_msg = self._probe_http_health(timeout=2.0)
        checks.append(http_msg)

        has_any_path = False
        if mode == "official_api":
            if use_gpt4o and toolkit_ready:
                has_any_path = True
            if use_gpt4o and self._openai_api_key():
                has_any_path = True
        else:
            if local_ok:
                has_any_path = True
        if http_ok:
            has_any_path = True

        if mode == "official_api" and use_gpt4o and toolkit_ready and (not lock_exists) and (not self._openai_api_key()) and (not http_ok):
            errors.append("toolkit_server_not_running_or_lock_missing")
            hints.append("Start VLM_toolkit_cross_gpfs/server.py on the internet machine and ensure it uses the same --shared-dir.")

        if not has_any_path:
            errors.append("no_available_backend_path")
            if mode == "official_api":
                hints.append(
                    "Enable at least one path: toolkit(shared_dir + server), direct OpenAI(OPENAI_API_KEY), or HTTP backend(--vlm-agent-url)."
                )
            else:
                hints.append(
                    "Enable at least one path: local OpenAI-compatible endpoint(base_url+model), or HTTP backend(--vlm-agent-url)."
                )

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "hints": hints,
            "checks": checks,
            "toolkit_ready": toolkit_ready,
            "local_openai_ready": local_ok,
            "http_ready": http_ok,
            "backend_mode": mode,
            "use_gpt4o": bool(use_gpt4o),
        }

    def run_video_understanding_vlm_toolkit(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_items: list[dict[str, Any]],
        frame_tokens: list[str],
        timeout: float = 300.0,
        ikg_enabled: bool = True,
    ) -> dict[str, Any] | None:
        if not frame_items:
            return {"ok": False, "provider": "vlm_toolkit_cross_gpfs", "error": "no_frame_items"}
        shared_dir = self._toolkit_shared_dir()
        if not shared_dir:
            return {"ok": False, "provider": "vlm_toolkit_cross_gpfs", "error": "missing_VLM_SHARED_DIR"}
        llm_cls, import_err = self._load_toolkit_client_cls()
        if llm_cls is None:
            return {
                "ok": False,
                "provider": "vlm_toolkit_cross_gpfs",
                "error": "toolkit_client_import_failed",
                "detail": import_err,
            }
        try:
            poll_interval = float(getattr(self.app, "vlm_client_poll_interval", None) or 0.2)
            llm_client = llm_cls(shared_dir=shared_dir, timeout=float(timeout), poll_interval=poll_interval)
        except Exception as exc:
            return {
                "ok": False,
                "provider": "vlm_toolkit_cross_gpfs",
                "error": "toolkit_chat_exception",
                "detail": f"{type(exc).__name__}:{exc}",
            }

        openai_model = self._openai_model()
        language = self._language_for_model(openai_model)

        def _toolkit_chat_json(messages: list[dict[str, Any]], local_timeout: float) -> tuple[bool, Any]:
            try:
                response = llm_client.chat(
                    messages=messages,
                    model=openai_model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                return False, f"toolkit_chat_exception:{type(exc).__name__}:{exc}"
            if response is None or not hasattr(response, "is_success"):
                return False, "invalid_toolkit_response"
            if not response.is_success():
                return False, str(getattr(response, "error_message", "unknown_error"))
            try:
                content = str(getattr(response, "content", "") or "")
                cleaned = self._strip_json_fence(content)
                parsed = json.loads(cleaned)
                return True, parsed
            except Exception as exc:
                return False, f"invalid_json_content:{type(exc).__name__}:{exc}"

        out = self._run_hierarchical_video_understanding(
            object_text=object_text,
            relation_hint=relation_hint,
            frame_items=frame_items,
            frame_tokens=frame_tokens,
            timeout=timeout,
            provider="vlm_toolkit_cross_gpfs",
            chat_json_fn=_toolkit_chat_json,
            language=language,
            ikg_enabled=bool(ikg_enabled),
        )
        if isinstance(out, dict):
            out["model"] = openai_model
            out["shared_dir"] = shared_dir
            out["strategy"] = "hierarchical_segment_memory"
            out["ikg_enabled"] = bool(ikg_enabled)
        return out

    def run_video_understanding_gpt4o(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_items: list[dict[str, Any]],
        frame_tokens: list[str],
        timeout: float = 150.0,
        ikg_enabled: bool = True,
    ) -> dict[str, Any] | None:
        if not self._openai_api_key():
            return {"ok": False, "provider": "gpt4o", "error": "missing_OPENAI_API_KEY"}
        if not frame_items:
            return {"ok": False, "provider": "gpt4o", "error": "no_frame_items"}
        openai_model = self._openai_model()
        language = self._language_for_model(openai_model)

        def _openai_chat_json(messages: list[dict[str, Any]], local_timeout: float) -> tuple[bool, Any]:
            payload = {
                "model": openai_model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": messages,
            }
            resp = self._post_openai(payload, timeout=local_timeout)
            if not isinstance(resp, dict):
                return False, "openai_http_failed"
            try:
                text = self._extract_message_text(resp["choices"][0]["message"]["content"])
                parsed = self._try_parse_json_obj(text)
                if parsed is None:
                    raise ValueError("json_parse_failed")
                return True, parsed
            except Exception as exc:
                return False, f"invalid_json_content:{type(exc).__name__}:{exc}"

        out = self._run_hierarchical_video_understanding(
            object_text=object_text,
            relation_hint=relation_hint,
            frame_items=frame_items,
            frame_tokens=frame_tokens,
            timeout=timeout,
            provider="gpt4o",
            chat_json_fn=_openai_chat_json,
            language=language,
            ikg_enabled=bool(ikg_enabled),
        )
        if isinstance(out, dict):
            out["model"] = openai_model
            out["strategy"] = "hierarchical_segment_memory"
            out["ikg_enabled"] = bool(ikg_enabled)
        return out

    def run_video_understanding_local_openai(
        self,
        *,
        object_text: str,
        relation_hint: str,
        frame_items: list[dict[str, Any]],
        frame_tokens: list[str],
        timeout: float = 300.0,
        ikg_enabled: bool = True,
    ) -> dict[str, Any] | None:
        base_url = self._local_vlm_base_url()
        model = self._local_vlm_model()
        language = self._language_for_model(model)
        api_key = self._local_vlm_api_key()
        raw_video_source = self._local_vlm_video_url()
        source_mode, source_value = self._normalize_local_video_source(raw_video_source)

        if not base_url:
            return {"ok": False, "provider": "local_openai_compatible", "error": "missing_local_base_url"}
        if not model:
            return {"ok": False, "provider": "local_openai_compatible", "error": "missing_local_model"}
        if source_mode == "invalid":
            return {
                "ok": False,
                "provider": "local_openai_compatible",
                "error": "invalid_local_video_source",
                "detail": str(raw_video_source),
            }
        if (not frame_items) and source_mode in {"none", "frame_dir"}:
            return {
                "ok": False,
                "provider": "local_openai_compatible",
                "error": "no_frame_items_or_video_url",
            }

        def _local_chat_json(messages: list[dict[str, Any]], local_timeout: float) -> tuple[bool, Any]:
            local_messages = self._convert_system_to_user_prefix(messages)
            payload = {
                "model": model,
                "temperature": 0.0,
                "messages": local_messages,
            }
            resp = self._post_chat_completions(
                payload,
                base_url=base_url,
                api_key=api_key,
                timeout=local_timeout,
                require_api_key=False,
            )
            if not isinstance(resp, dict):
                return False, "local_openai_http_failed"
            try:
                text = self._extract_message_text(resp["choices"][0]["message"]["content"])
                parsed = self._try_parse_json_obj(text)
                if parsed is None:
                    raise ValueError("json_parse_failed")
                return True, parsed
            except Exception as exc:
                return False, f"invalid_json_content:{type(exc).__name__}:{exc}"

        if source_mode == "video_url" and source_value:
            token_lines = [f"{i}: frame_token={tok}" for i, tok in enumerate(frame_tokens)]
            system_prompt, user_prompt = self._build_agent0_prompts(
                object_text=self._to_language_object_text(object_text, language),
                relation_hint=relation_hint,
                frame_tokens=frame_tokens,
                token_lines=token_lines,
                language=language,
                ikg_enabled=bool(ikg_enabled),
            )
            if not bool(ikg_enabled):
                restate_prompt = (
                    "【要求重申】请只输出真实物体实例列表；不要分析或输出交互关系、事件、约束、深度排序或相对位置时间线。"
                    if str(language).strip().lower() == "zh"
                    else "[Requirement Restatement] Output only real object instances. Do not analyze or output interactions, events, constraints, depth order, or relative-position timelines."
                )
            elif str(language).strip().lower() == "zh":
                restate_prompt = (
                    "【要求重申】请在看完视频后严格按上述要求输出 JSON。"
                    "同类物体优先按外观细节区分命名（颜色/纹理/形状/大小/标记等）；"
                    "仅当外观确实无法区分时，才允许“左侧的杯子/右侧的杯子”等方位限定名，并保持全片一致；"
                    "相对方位语义仍需写入 relative_position_timeline。"
                )
            else:
                restate_prompt = (
                    "[Requirement Restatement] After reviewing the video, strictly follow the requirements and output JSON only. "
                    "For same-category objects, first distinguish by appearance details (color/texture/shape/size/markings). "
                    "Only if truly visually indistinguishable, allow spatial qualifiers (`left cup` / `right cup`) "
                    "and keep naming consistent across the video. "
                    "Spatial semantics still belongs to relative_position_timeline."
                )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": str(source_value)}},
                        {"type": "text", "text": system_prompt + "\n\n" + user_prompt},
                        {"type": "text", "text": restate_prompt},
                    ],
                }
            ]
            ok, parsed_or_error = _local_chat_json(messages, timeout)
            if not ok or not isinstance(parsed_or_error, dict):
                return {
                    "ok": False,
                    "provider": "local_openai_compatible",
                    "error": str(parsed_or_error),
                }
            scene_graph = parsed_or_error
            if not bool(ikg_enabled):
                scene_graph = self.strip_interaction_knowledge_graph(scene_graph)
            return {
                "ok": True,
                "provider": "local_openai_compatible",
                "scene_graph": scene_graph,
                "frames_used": len(frame_items),
                "segment_count": 1,
                "strategy": "single_pass_video_url",
                "model": model,
                "base_url": base_url,
                "video_url": str(source_value),
                "source_mode": source_mode,
                "ikg_enabled": bool(ikg_enabled),
                "ablation_mode": "" if bool(ikg_enabled) else "wo_IKG",
            }

        out = self._run_hierarchical_video_understanding(
            object_text=object_text,
            relation_hint=relation_hint,
            frame_items=frame_items,
            frame_tokens=frame_tokens,
            timeout=timeout,
            provider="local_openai_compatible",
            chat_json_fn=_local_chat_json,
            language=language,
            ikg_enabled=bool(ikg_enabled),
        )
        if isinstance(out, dict):
            out["model"] = model
            out["base_url"] = base_url
            out["strategy"] = "hierarchical_segment_memory"
            out["source_mode"] = source_mode
            out["ikg_enabled"] = bool(ikg_enabled)
            if not bool(ikg_enabled):
                out["ablation_mode"] = "wo_IKG"
            if source_mode == "frame_dir" and source_value:
                out["frame_source_dir"] = str(source_value)
        return out
