from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_VIEW_ORDER = ("down", "up", "left", "front")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class MemoryFrameSelector:
    """Select/update memory frames for 4D repair with VLM scoring."""

    def __init__(self, ctx, state, bridge, vlm_client) -> None:
        self.ctx = ctx
        self.state = state
        self.bridge = bridge
        self.vlm_client = vlm_client

    @staticmethod
    def _load_rgb_over_white(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(white_bg, rgba)
            return np.asarray(composited.convert("RGB"), dtype=np.uint8)

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

    def _render_frame_pack(self, client, frame_idx: int, *, save_root: Path | None) -> dict[str, Any]:
        app = self.ctx.app
        seq = getattr(app, "sequence", None)
        frame_tokens = list(getattr(seq, "frame_tokens", []) or [])
        if not frame_tokens:
            return {"ok": False, "reason": "frame_tokens_empty", "frame_idx": int(frame_idx)}
        idx = max(0, min(int(frame_idx), len(frame_tokens) - 1))
        tok = str(frame_tokens[idx])
        safe_tok = tok.replace("/", "_").replace("\\", "_").replace(" ", "_")
        mv = self.bridge.registry.call(
            "render.frame_multiview",
            client=client,
            frame_idx=int(idx),
            views=list(_VIEW_ORDER),
            image_size=768,
            save_dir=(str(save_root) if save_root is not None else None),
            prefix=f"memsel_f{int(idx):04d}_{safe_tok}",
        )
        if not bool(mv.get("ok", False)):
            return {
                "ok": False,
                "reason": str(mv.get("reason", "render_failed")),
                "frame_idx": int(idx),
                "frame_token": tok,
            }
        input_path = self._resolve_input_image_path(tok)
        input_rgb = None
        if input_path is not None:
            try:
                input_rgb = self._load_rgb_over_white(input_path)
            except Exception:
                input_rgb = None
        view_images = dict(mv.get("view_images", {}) or {})
        ordered = {k: view_images.get(k, None) for k in _VIEW_ORDER}
        return {
            "ok": True,
            "frame_idx": int(idx),
            "frame_token": tok,
            "input_rgb_u8": input_rgb,
            "view_images": ordered,
            "view_paths": dict(mv.get("view_paths", {}) or {}),
        }

    def select_memory_frames(
        self,
        client,
        *,
        target_frame_idx: int,
        candidate_frame_indices: list[int],
        desired_traits: list[str] | None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        app = self.ctx.app
        with self.state.lock:
            object_text = str(getattr(self.state, "object_description", "") or "")
            relation_hint = str(getattr(self.state, "relation_hint", "") or "")
            scene_graph = dict(getattr(self.state, "scene_graph", {}) or {})
            backend_mode = str(getattr(self.state, "agent0_backend_mode", "local_openai_compatible") or "local_openai_compatible")

        save_root = None
        try:
            with self.state.lock:
                root = getattr(self.state, "artifact_root", None)
            if root is not None:
                save_root = Path(root) / "memory_selector"
                save_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            save_root = None

        print(
            "[VLM Assistor][MemorySelector] select_memory_frames start:",
            json.dumps(
                {
                    "target_frame_idx": int(target_frame_idx),
                    "candidate_count": int(len(candidate_frame_indices)),
                    "top_k": int(top_k),
                    "desired_traits": [str(x) for x in list(desired_traits or [])],
                },
                ensure_ascii=False,
            ),
        )

        target_pack = self._render_frame_pack(client, int(target_frame_idx), save_root=save_root)
        if not bool(target_pack.get("ok", False)):
            print("[VLM Assistor][MemorySelector] target render failed:", target_pack)
            return {"ok": False, "reason": "target_render_failed", "target": target_pack}

        rows: list[dict[str, Any]] = []
        uniq_candidates = sorted({int(x) for x in list(candidate_frame_indices or []) if int(x) >= 0})
        for rank, idx in enumerate(uniq_candidates):
            cand_pack = self._render_frame_pack(client, int(idx), save_root=save_root)
            if not bool(cand_pack.get("ok", False)):
                row = {
                    "frame_idx": int(idx),
                    "ok": False,
                    "reason": str(cand_pack.get("reason", "candidate_render_failed")),
                    "score": 0,
                    "recommended": False,
                }
                print("[VLM Assistor][MemorySelector] candidate skipped:", json.dumps(row, ensure_ascii=False))
                rows.append(row)
                continue
            vlm_out = self.vlm_client.evaluate_memory_frame_candidate_with_vlm(
                object_text=object_text,
                relation_hint=relation_hint,
                scene_graph=scene_graph,
                desired_traits=list(desired_traits or []),
                target_frame=target_pack,
                candidate_frame=cand_pack,
                backend_mode=backend_mode,
                timeout=90.0,
            )
            score = int(vlm_out.get("score", 0))
            # Slight recency preference among similarly good candidates.
            recency_bonus = max(0, int(3 - min(3, abs(int(target_frame_idx) - int(idx)) // 4)))
            final_score = max(0, min(100, int(score + recency_bonus)))
            row = {
                "frame_idx": int(idx),
                "frame_token": str(cand_pack.get("frame_token", "")),
                "ok": bool(vlm_out.get("ok", False)),
                "evaluated": bool(vlm_out.get("evaluated", False)),
                "score": int(score),
                "final_score": int(final_score),
                "recommended": bool(vlm_out.get("recommended", False)),
                "recency_bonus": int(recency_bonus),
                "summary": str(vlm_out.get("summary", "")),
                "reason": str(vlm_out.get("reason", "")),
                "evidence": list(vlm_out.get("evidence", []) or []),
                "rank_order": int(rank + 1),
            }
            print("[VLM Assistor][MemorySelector] candidate eval:", json.dumps(row, ensure_ascii=False))
            rows.append(row)

        rows_sorted = sorted(rows, key=lambda x: (int(x.get("final_score", 0)), bool(x.get("recommended", False))), reverse=True)
        selected: list[int] = []
        for row in rows_sorted:
            if len(selected) >= max(1, int(top_k)):
                break
            idx = int(row.get("frame_idx", -1))
            if idx < 0:
                continue
            if bool(row.get("recommended", False)) or int(row.get("final_score", 0)) >= 70:
                selected.append(idx)
        if not selected:
            selected = [int(x.get("frame_idx", -1)) for x in rows_sorted[: max(1, int(top_k))] if int(x.get("frame_idx", -1)) >= 0]
        out = {
            "ok": True,
            "target_frame_idx": int(target_frame_idx),
            "selected_frame_indices": [int(x) for x in selected],
            "rows": rows_sorted,
        }
        print(
            "[VLM Assistor][MemorySelector] select_memory_frames result:",
            json.dumps(
                {
                    "target_frame_idx": int(target_frame_idx),
                    "selected_frame_indices": [int(x) for x in selected],
                    "top_rows": rows_sorted[:3],
                },
                ensure_ascii=False,
            ),
        )
        return out

    def decide_memory_bank_update(
        self,
        client,
        *,
        frame_idx: int,
        existing_memory_indices: list[int],
    ) -> dict[str, Any]:
        with self.state.lock:
            object_text = str(getattr(self.state, "object_description", "") or "")
            relation_hint = str(getattr(self.state, "relation_hint", "") or "")
            scene_graph = dict(getattr(self.state, "scene_graph", {}) or {})
            backend_mode = str(getattr(self.state, "agent0_backend_mode", "local_openai_compatible") or "local_openai_compatible")
        current_pack = self._render_frame_pack(client, int(frame_idx), save_root=None)
        if not bool(current_pack.get("ok", False)):
            return {"ok": False, "reason": "current_render_failed", "current": current_pack}
        memory_frames: list[dict[str, Any]] = []
        for idx in sorted({int(x) for x in list(existing_memory_indices or []) if int(x) >= 0})[:6]:
            pack = self._render_frame_pack(client, int(idx), save_root=None)
            if bool(pack.get("ok", False)):
                memory_frames.append(pack)
        out = self.vlm_client.decide_memory_bank_update_with_vlm(
            object_text=object_text,
            relation_hint=relation_hint,
            scene_graph=scene_graph,
            current_frame=current_pack,
            memory_frames=memory_frames,
            backend_mode=backend_mode,
            timeout=90.0,
        )
        print(
            "[VLM Assistor][MemorySelector] memory_bank_update decision:",
            json.dumps(
                {
                    "frame_idx": int(frame_idx),
                    "existing_memory_indices": [int(x) for x in list(existing_memory_indices or [])],
                    "decision": out,
                },
                ensure_ascii=False,
            ),
        )
        return out
