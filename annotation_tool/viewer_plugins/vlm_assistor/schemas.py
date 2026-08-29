from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any


@dataclass
class RelationEdge:
    subject: str
    predicate: str
    obj: str
    phase: str = "global"
    confidence: float = 0.5
    distance_hint: str = "unknown"  # near_zero | near | mid | far | unknown


@dataclass
class TemporalSceneGraph:
    objects: list[str] = field(default_factory=list)
    relations: list[RelationEdge] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["relations"] = [asdict(r) for r in self.relations]
        return data


def normalize_scene_graph(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return TemporalSceneGraph().to_dict()
    objects_raw = payload.get("objects", [])
    objects = [str(x).strip() for x in objects_raw if str(x).strip()]

    rels: list[RelationEdge] = []
    for item in payload.get("relations", []):
        if not isinstance(item, dict):
            continue
        sub = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        obj = str(item.get("obj", item.get("object", ""))).strip()
        if not (sub and pred and obj):
            continue
        rels.append(
            RelationEdge(
                subject=sub,
                predicate=pred,
                obj=obj,
                phase=str(item.get("phase", "global")),
                confidence=float(item.get("confidence", 0.5)),
                distance_hint=str(item.get("distance_hint", "unknown")),
            )
        )

    notes = str(payload.get("notes", "")).strip()
    return TemporalSceneGraph(objects=objects, relations=rels, notes=notes).to_dict()


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


def _normalize_relative_axis(raw: Any, *, axis: str) -> str:
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


def _normalize_relative_pair_list(raw: Any) -> list[dict[str, Any]]:
    rows = raw if isinstance(raw, list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for it in rows:
        if not isinstance(it, dict):
            continue
        subject = str(it.get("subject", it.get("src", it.get("a", "")))).strip()
        obj = str(it.get("obj", it.get("object", it.get("dst", it.get("b", ""))))).strip()
        if not (subject and obj):
            continue
        horizontal = _normalize_relative_axis(
            it.get("horizontal", it.get("left_right", it.get("lr", it.get("x_relation", "unknown")))),
            axis="horizontal",
        )
        vertical = _normalize_relative_axis(
            it.get("vertical", it.get("up_down", it.get("ud", it.get("y_relation", "unknown")))),
            axis="vertical",
        )
        depth = _normalize_relative_axis(
            it.get("depth", it.get("front_back", it.get("fb", it.get("z_relation", "unknown")))),
            axis="depth",
        )
        pred = str(it.get("predicate", "")).strip()
        notes = str(it.get("notes", it.get("evidence", ""))).strip()
        conf: float | None = None
        try:
            if "confidence" in it:
                conf = float(it.get("confidence"))
        except Exception:
            conf = None
        key = f"{subject}|{obj}|{horizontal}|{vertical}|{depth}|{pred}"
        if key in seen:
            continue
        seen.add(key)
        rec: dict[str, Any] = {
            "subject": subject,
            "obj": obj,
            "horizontal": horizontal,
            "vertical": vertical,
            "depth": depth,
        }
        if pred:
            rec["predicate"] = pred
        if conf is not None:
            rec["confidence"] = float(max(0.0, min(1.0, conf)))
        if notes:
            rec["notes"] = notes
        out.append(rec)
        if len(out) >= 96:
            break
    return out


def _normalize_relative_position_timeline(payload: dict[str, Any], *, frame_count: int) -> list[dict[str, Any]]:
    raw = payload.get(
        "relative_position_timeline",
        payload.get("relative_spatial_timeline", payload.get("spatial_relation_timeline", [])),
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
        pairs_raw: Any = []
        notes = ""
        conf: float | None = None

        if isinstance(item, dict):
            frame_range = str(
                item.get("frame_range", item.get("range", item.get("time_range", fallback_range)))
            ).strip()
            start_idx = _try_parse_int(item.get("start_frame_index", item.get("start_idx", item.get("start_frame"))))
            end_idx = _try_parse_int(item.get("end_frame_index", item.get("end_idx", item.get("end_frame"))))
            pairs_raw = item.get(
                "pairs",
                item.get("relations", item.get("relative_pairs", item.get("relative_relations", []))),
            )
            if not isinstance(pairs_raw, list):
                pairs_raw = [item]
            notes = str(item.get("notes", item.get("summary", item.get("description", "")))).strip()
            try:
                if "confidence" in item:
                    conf = float(item.get("confidence"))
            except Exception:
                conf = None
        elif isinstance(item, list):
            frame_range = fallback_range
            pairs_raw = item
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

        pairs = _normalize_relative_pair_list(pairs_raw)
        if not pairs:
            continue

        if not frame_range:
            if start_idx is not None and end_idx is not None:
                frame_range = f"{int(start_idx)}-{int(end_idx)}"
            else:
                frame_range = f"range_{i}"
        key = json.dumps({"frame_range": frame_range, "pairs": pairs}, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)

        rec: dict[str, Any] = {
            "frame_range": str(frame_range),
            "pairs": pairs,
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


def normalize_temporal_scene_graph(
    payload: dict[str, Any] | None,
    *,
    frame_tokens: list[str] | None = None,
) -> dict[str, Any]:
    """Normalize richer TSG while keeping compatibility with downstream agents."""
    base = normalize_scene_graph(payload)
    if not isinstance(payload, dict):
        payload = {}

    known_tokens = set(frame_tokens or [])
    distance_allowed = {"near_zero", "near", "mid", "far", "unknown"}

    object_meta_raw = payload.get("object_meta", payload.get("objects_detail", []))
    object_meta: list[dict[str, Any]] = []
    if isinstance(object_meta_raw, list):
        for i, x in enumerate(object_meta_raw):
            if isinstance(x, dict):
                name = str(x.get("name", x.get("id", f"object_{i}"))).strip()
                if not name:
                    name = f"object_{i}"
                object_meta.append(
                    {
                        "id": str(x.get("id", name)).strip() or name,
                        "name": name,
                        "category": str(x.get("category", x.get("type", "object"))).strip() or "object",
                        "attributes": list(x.get("attributes", [])) if isinstance(x.get("attributes", []), list) else [],
                    }
                )

    phases: list[dict[str, Any]] = []
    for i, x in enumerate(payload.get("phases", [])):
        if not isinstance(x, dict):
            continue
        name = str(x.get("name", f"phase_{i}")).strip() or f"phase_{i}"
        st = str(x.get("start_frame_token", x.get("start_frame", ""))).strip()
        ed = str(x.get("end_frame_token", x.get("end_frame", ""))).strip()
        phases.append(
            {
                "name": name,
                "start_frame_token": st if (not known_tokens or st in known_tokens or not st) else "",
                "end_frame_token": ed if (not known_tokens or ed in known_tokens or not ed) else "",
                "description": str(x.get("description", "")).strip(),
            }
        )

    events: list[dict[str, Any]] = []
    for x in payload.get("events", []):
        if not isinstance(x, dict):
            continue
        ev = {
            "type": str(x.get("type", "")).strip(),
            "subject": str(x.get("subject", "")).strip(),
            "obj": str(x.get("obj", x.get("object", ""))).strip(),
            "frame_token": str(x.get("frame_token", x.get("frame", ""))).strip(),
            "phase": str(x.get("phase", "global")).strip() or "global",
            "confidence": float(x.get("confidence", 0.5)),
        }
        if not ev["type"]:
            continue
        if known_tokens and ev["frame_token"] and ev["frame_token"] not in known_tokens:
            ev["frame_token"] = ""
        events.append(ev)

    constraints: list[dict[str, Any]] = []
    for x in payload.get("constraints", []):
        if isinstance(x, str):
            rule = x.strip()
            if rule:
                constraints.append({"rule": rule})
            continue
        if isinstance(x, dict):
            rule = str(x.get("rule", "")).strip()
            if rule:
                constraints.append(
                    {
                        "rule": rule,
                        "target_pair": list(x.get("target_pair", []))
                        if isinstance(x.get("target_pair", []), list)
                        else [],
                        "phase": str(x.get("phase", "global")).strip() or "global",
                    }
                )

    relations = base.get("relations", [])
    for rel in relations:
        d = str(rel.get("distance_hint", "unknown")).strip()
        if d not in distance_allowed:
            rel["distance_hint"] = "unknown"

    x_depth_order_timeline = _normalize_x_depth_order_timeline(
        payload,
        frame_count=len(frame_tokens or []),
    )
    relative_position_timeline = _normalize_relative_position_timeline(
        payload,
        frame_count=len(frame_tokens or []),
    )

    return {
        "objects": base.get("objects", []),
        "relations": relations,
        "notes": str(payload.get("notes", base.get("notes", ""))).strip(),
        "object_meta": object_meta,
        "phases": phases,
        "events": events,
        "constraints": constraints,
        "x_depth_order_timeline": x_depth_order_timeline,
        "relative_position_timeline": relative_position_timeline,
    }
