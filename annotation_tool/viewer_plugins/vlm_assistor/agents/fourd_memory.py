from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import _AgentBase

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class _HistoryStoreLayer:
    def __init__(self, state, log_fn) -> None:
        self.state = state
        self._log = log_fn

    def snapshot(self) -> tuple[list[dict[str, Any]], str]:
        with self.state.lock:
            rows = list(getattr(self.state, "agent4_history_records", []) or [])
            memo = str(getattr(self.state, "agent4_history_memory", "") or "")
        return rows, memo

    def append(self, record: dict[str, Any], memory_text: str, *, capacity: int) -> None:
        cap = max(1, int(capacity))
        with self.state.lock:
            rows = list(getattr(self.state, "agent4_history_records", []) or [])
            rows.append(dict(record))
            self.state.agent4_history_records = rows[-cap:]
            self.state.agent4_history_memory = str(memory_text)


class _HistoryCompressorLayer:
    @staticmethod
    def _trend(scores: list[int]) -> str:
        if len(scores) < 2:
            return "flat"
        last = float(scores[-1])
        prev = float(scores[-2])
        delta = last - prev
        if delta >= 4:
            return "up"
        if delta <= -4:
            return "down"
        return "flat"

    def compress(
        self,
        *,
        previous_memory: str,
        history_records: list[dict[str, Any]],
        latest_eval: dict[str, Any],
    ) -> str:
        scores = [int(x.get("quality_score", 0)) for x in history_records[-6:]]
        if "quality_score" in latest_eval:
            scores.append(int(latest_eval.get("quality_score", 0)))
        avg = float(np.mean(scores)) if scores else 0.0
        trend = self._trend(scores) if scores else "flat"

        issue_counter: Counter[str] = Counter()
        for row in history_records[-8:]:
            for issue in row.get("issues", []) or []:
                s = str(issue).strip()
                if s:
                    issue_counter[s] += 1
        for issue in latest_eval.get("issues", []) or []:
            s = str(issue).strip()
            if s:
                issue_counter[s] += 1
        top_issues = [k for k, _ in issue_counter.most_common(3)]

        lines: list[str] = []
        if previous_memory:
            lines.append(f"prev={previous_memory[:220]}")
        lines.append(f"score_avg={avg:.1f}")
        lines.append(f"score_trend={trend}")
        if top_issues:
            lines.append("issues=" + ",".join(top_issues))
        summary = str(latest_eval.get("summary", "")).strip()
        if summary:
            lines.append(f"latest={summary[:240]}")
        return " | ".join(lines)[:1400]


class _CurrentHistoryDiscriminatorLayer:
    def __init__(self, log_fn) -> None:
        self._log = log_fn

    @staticmethod
    def _safe_int(v: Any, fallback: int) -> int:
        try:
            return int(v)
        except Exception:
            return int(fallback)

    @staticmethod
    def _image_path_to_data_url(path: Path, *, max_side: int = 448, quality: int = 70) -> str | None:
        try:
            with Image.open(path) as img:
                rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
        h, w = rgb.shape[:2]
        side = max(h, w)
        if side > int(max_side) > 0:
            scale = float(max_side) / float(side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            rgb = np.asarray(
                Image.fromarray(rgb).resize((new_w, new_h), resample=Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, format="JPEG", quality=int(quality), optimize=True)
        b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
        return "data:image/jpeg;base64," + b64

    @staticmethod
    def _extract_text(resp: dict[str, Any]) -> str:
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    out.append(str(item.get("text", "")))
            return "\n".join(out)
        return str(content)

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        s = str(text or "").strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
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

    def _post_chat(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        messages: list[dict[str, Any]],
        timeout: float,
    ) -> dict[str, Any] | None:
        payload = {
            "model": str(model),
            "temperature": 0.0,
            "messages": messages,
            "max_tokens": 512,
        }
        url = str(base_url).rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            url,
            method="POST",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            self._log("warn", f"agent4_vlm_http_failed:{type(exc).__name__}:{exc}")
            return None

    def _heuristic_eval(
        self,
        *,
        current_pack: dict[str, Any],
        history_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        current_views = list(current_pack.get("view_paths", []) or [])
        issues: list[str] = []
        if not current_views:
            return {
                "quality_score": 0,
                "confidence": 0.2,
                "issues": ["missing_current_multiview"],
                "summary": "No current multi-view images are available for 4D quality evaluation.",
                "source": "heuristic",
            }

        score = 72
        if len(current_views) >= 3:
            score += 6
        elif len(current_views) == 1:
            score -= 8
            issues.append("single_view_only")

        hist_scores = [self._safe_int(x.get("quality_score", 0), 0) for x in history_records[-6:]]
        if hist_scores:
            hist_mean = int(round(float(np.mean(hist_scores))))
            score = int(round(0.6 * score + 0.4 * hist_mean))
            if len(hist_scores) >= 2 and hist_scores[-1] - hist_scores[-2] <= -8:
                issues.append("recent_quality_drop")

        score = max(0, min(100, int(score)))
        if score < 45:
            issues.append("likely_unstable_4d")
        summary = "Heuristic fallback found no issue." if not issues else "Heuristic fallback found potential issues."
        return {
            "quality_score": score,
            "confidence": 0.45,
            "issues": sorted(set(issues)),
            "summary": summary,
            "source": "heuristic",
        }

    def evaluate(
        self,
        *,
        current_pack: dict[str, Any],
        history_packs: list[dict[str, Any]],
        previous_memory: str,
        history_records: list[dict[str, Any]],
        base_url: str,
        model: str,
        api_key: str | None,
        max_images: int,
    ) -> dict[str, Any]:
        # Always keep a valid fallback result.
        fallback = self._heuristic_eval(
            current_pack=current_pack,
            history_records=history_records,
        )
        if not base_url or not model:
            return fallback

        content: list[dict[str, Any]] = []
        brief = {
            "task": "4d_quality_eval",
            "frame_token": str(current_pack.get("frame_token", "")),
            "history_frame_tokens": [str(x.get("frame_token", "")) for x in history_packs],
            "history_memory": str(previous_memory or ""),
            "recent_scores": [int(x.get("quality_score", 0)) for x in history_records[-6:]],
        }
        content.append(
            {
                "type": "text",
                "text": (
                    "你是4D生成质量判别器。请结合当前帧多视角图、过去帧多视角图和历史记忆，"
                    "输出严格 JSON："
                    '{"quality_score":0-100整数,"confidence":0-1浮点,'
                    '"issues":[字符串],'
                    '"summary":"一句话结论"}。'
                    f"\n上下文:{json.dumps(brief, ensure_ascii=False)}"
                ),
            }
        )

        image_count = 0

        def _append_image(label: str, path: Path) -> None:
            nonlocal image_count
            if image_count >= int(max_images):
                return
            data_url = self._image_path_to_data_url(path)
            if not data_url:
                return
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "low"}})
            image_count += 1

        for idx, path_str in enumerate(current_pack.get("view_paths", []) or []):
            _append_image(f"[current_view_{idx}]", Path(path_str))
        for hist_idx, pack in enumerate(history_packs):
            first = list(pack.get("view_paths", []) or [])[:1]
            for path_str in first:
                _append_image(f"[history_{hist_idx}]", Path(path_str))

        messages = [{"role": "user", "content": content}]
        resp = self._post_chat(
            base_url=base_url,
            model=model,
            api_key=api_key,
            messages=messages,
            timeout=120.0,
        )
        if not isinstance(resp, dict):
            return fallback
        parsed = self._try_parse_json(self._extract_text(resp))
        if not isinstance(parsed, dict):
            return fallback

        score = self._safe_int(parsed.get("quality_score", fallback["quality_score"]), fallback["quality_score"])
        confidence = float(parsed.get("confidence", fallback.get("confidence", 0.5)))
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        summary = str(parsed.get("summary", fallback["summary"])).strip() or fallback["summary"]
        return {
            "quality_score": max(0, min(100, int(score))),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "issues": [str(x).strip() for x in issues if str(x).strip()],
            "summary": summary[:400],
            "source": "vlm",
            "raw": parsed,
        }


class _SmolAgentsAppLayer:
    def __init__(self, log_fn) -> None:
        self._log = log_fn

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        s = str(text or "").strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        st = s.find("{")
        ed = s.rfind("}")
        if st >= 0 and ed > st:
            try:
                obj = json.loads(s[st : ed + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    def maybe_refine(
        self,
        *,
        enabled: bool,
        draft_result: dict[str, Any],
        history_memory: str,
        base_url: str,
        api_key: str | None,
    ) -> dict[str, Any] | None:
        if not bool(enabled):
            return None
        try:
            from smolagents import CodeAgent, LiteLLMModel, tool  # type: ignore
        except Exception as exc:
            self._log("warn", f"smolagents_import_failed:{type(exc).__name__}:{exc}")
            return None

        model_id = str(os.getenv("SMOLAGENTS_MODEL_ID", "")).strip()
        if not model_id:
            self._log("warn", "smolagents_model_id_missing: set SMOLAGENTS_MODEL_ID")
            return None
        if not base_url:
            self._log("warn", "smolagents_base_url_missing")
            return None

        @tool
        def get_draft_result() -> str:
            """Return current 4D quality draft result as JSON string."""
            return json.dumps(draft_result, ensure_ascii=False)

        @tool
        def get_history_memory() -> str:
            """Return compressed history memory text."""
            return str(history_memory or "")

        try:
            model = LiteLLMModel(
                model_id=model_id,
                api_base=base_url,
                api_key=(api_key or "EMPTY"),
            )
            agent = CodeAgent(
                model=model,
                tools=[get_draft_result, get_history_memory],
                max_steps=2,
            )
            out = agent.run(
                "Read tools and return JSON only: "
                '{"risk_level":"low|mid|high","action":"一句建议","memory_hint":"短句"}'
            )
            parsed = self._try_parse_json(str(out))
            if isinstance(parsed, dict):
                parsed["enabled"] = True
                return parsed
        except Exception as exc:
            self._log("warn", f"smolagents_run_failed:{type(exc).__name__}:{exc}")
        return None


class FourDMemoryJudgeAgent(_AgentBase):
    """Agent4: history-aware 4D quality scoring for current frame."""

    def __init__(self, ctx, state, bridge, vlm_client) -> None:
        super().__init__(ctx, state, bridge, vlm_client, agent_name="agent4_4d_memory_judge")
        self.store = _HistoryStoreLayer(state, self.log)
        self.compressor = _HistoryCompressorLayer()
        self.discriminator = _CurrentHistoryDiscriminatorLayer(self.log)
        self.smol_layer = _SmolAgentsAppLayer(self.log)

    @staticmethod
    def _find_input_image(input_dir: Path, token: str) -> Path | None:
        for ext in _IMAGE_EXTS:
            p = input_dir / f"{token}{ext}"
            if p.exists():
                return p
        return None

    @staticmethod
    def _sort_multiview_paths(paths: list[Path]) -> list[Path]:
        def _key(p: Path) -> tuple[int, str]:
            name = p.stem
            if "_az" in name:
                tail = name.split("_az", 1)[1]
                try:
                    return int(tail), name
                except Exception:
                    pass
            return 10_000, name

        return sorted(paths, key=_key)

    def _collect_window_packs(
        self,
        *,
        window_size: int,
        views_per_frame: int,
        current_idx_override: int | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        app = self.ctx.app
        seq = getattr(app, "sequence", None)
        frame_tokens = list(getattr(seq, "frame_tokens", []) or [])
        if not frame_tokens:
            return None, []

        if current_idx_override is None:
            current_idx = int(getattr(app, "current_frame", 0))
        else:
            current_idx = int(current_idx_override)
        current_idx = max(0, min(current_idx, len(frame_tokens) - 1))
        start = max(0, current_idx - max(0, int(window_size)))

        session_root = getattr(app, "session_root", None)
        mv_dir = (Path(session_root) / "multi_view") if session_root else None
        input_dir = Path(getattr(app, "input_dir", ""))

        packs: list[dict[str, Any]] = []
        for idx in range(start, current_idx + 1):
            tok = str(frame_tokens[idx])
            paths: list[Path] = []
            if mv_dir and mv_dir.exists():
                paths = self._sort_multiview_paths(list(mv_dir.glob(f"{tok}_az*.png")))
            if not paths and input_dir.exists():
                fallback = self._find_input_image(input_dir, tok)
                if fallback is not None:
                    paths = [fallback]
            selected = [str(p.resolve()) for p in paths[: max(1, int(views_per_frame))]]
            packs.append(
                {
                    "frame_idx": idx,
                    "frame_token": tok,
                    "view_paths": selected,
                }
            )

        current_pack = packs[-1] if packs else None
        history_packs = packs[:-1] if len(packs) > 1 else []
        return current_pack, history_packs

    def run(
        self,
        client,
        *,
        window_size: int | None = None,
        frame_idx: int | None = None,
    ) -> dict[str, Any]:
        with self.state.lock:
            cfg_window = int(getattr(self.state, "agent4_history_window", 6))
            views_per_frame = int(getattr(self.state, "agent4_views_per_frame", 3))
            max_images = int(getattr(self.state, "agent4_max_images", 12))
            use_smol = bool(getattr(self.state, "agent4_use_smolagents", False))
            history_capacity = int(getattr(self.state, "agent4_history_capacity", 64))
            base_url = str(getattr(self.state, "agent0_local_base_url", "") or "")
            model = str(getattr(self.state, "agent0_local_model", "") or "")
            api_key = str(getattr(self.state, "agent0_local_api_key", "") or "").strip() or None

        effective_window = cfg_window if window_size is None else int(window_size)
        current_pack, history_packs = self._collect_window_packs(
            window_size=max(0, effective_window),
            views_per_frame=max(1, views_per_frame),
            current_idx_override=frame_idx,
        )
        if current_pack is None:
            msg = "frame_tokens_empty"
            self.log("fail", msg)
            self._notify(client, "Agent4", "No sequence frames found.", color="red")
            return {"ok": False, "reason": msg}

        history_rows, previous_memory = self.store.snapshot()
        eval_out = self.discriminator.evaluate(
            current_pack=current_pack,
            history_packs=history_packs,
            previous_memory=previous_memory,
            history_records=history_rows,
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_images=max(1, int(max_images)),
        )
        new_memory = self.compressor.compress(
            previous_memory=previous_memory,
            history_records=history_rows,
            latest_eval=eval_out,
        )

        record = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_token": str(current_pack.get("frame_token", "")),
            "frame_idx": int(current_pack.get("frame_idx", 0)),
            "quality_score": int(eval_out.get("quality_score", 0)),
            "confidence": float(eval_out.get("confidence", 0.0)),
            "issues": list(eval_out.get("issues", []) or []),
            "summary": str(eval_out.get("summary", "")),
            "source": str(eval_out.get("source", "unknown")),
        }
        self.store.append(record, new_memory, capacity=max(1, history_capacity))

        smol_extra = self.smol_layer.maybe_refine(
            enabled=bool(use_smol),
            draft_result=eval_out,
            history_memory=new_memory,
            base_url=base_url,
            api_key=api_key,
        )

        out = {
            "ok": True,
            "agent": "agent4_4d_memory_judge",
            "quality_score": int(eval_out.get("quality_score", 0)),
            "confidence": float(eval_out.get("confidence", 0.0)),
            "issues": list(eval_out.get("issues", []) or []),
            "summary": str(eval_out.get("summary", "")),
            "history_memory": new_memory,
            "current_frame": current_pack,
            "history_frames": history_packs,
            "history_records_count": len(history_rows) + 1,
            "source": str(eval_out.get("source", "unknown")),
        }
        if isinstance(smol_extra, dict):
            out["smolagents"] = smol_extra

        with self.state.lock:
            self.state.last_4d_memory_result = dict(out)
            self.state.last_4d_validation = dict(out)

        score = int(out.get("quality_score", 0))
        self.log("ok", f"score={score},source={out.get('source', 'unknown')}")
        self._notify(client, "Agent4", f"4D quality score={score}", color="green")
        return out
