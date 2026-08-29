"""Shared security checks for local HAT-4D HTTP services."""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PATH_FIELD_NAMES = {
    "anchor_image",
    "gs_ply",
    "gt_image",
    "input_dir",
    "eval_gt_dir",
    "workspace_root",
    "output_dir",
    "image_path",
    "mask_path",
    "gs_path",
    "ply_path",
    "render_path",
}
PATH_COLLECTION_NAMES = {
    "frame_paths",
    "ply_paths",
    "rendered_gs_paths",
    "view_paths",
}
IDENTIFIER_FIELD_NAMES = {"exp_id", "experiment_id", "session_id"}


class RequestError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


def is_loopback_host(host: str) -> bool:
    value = str(host).strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def require_secure_bind(host: str, api_token: str | None) -> None:
    if not is_loopback_host(host) and not str(api_token or "").strip():
        raise ValueError(
            "HAT4D_API_TOKEN is required when binding a backend to a non-loopback address"
        )


def request_is_authorized(headers: Any, api_token: str | None) -> bool:
    expected = str(api_token or "").strip()
    if not expected:
        return True
    supplied = str(headers.get("X-HAT4D-Token", "")).strip()
    authorization = str(headers.get("Authorization", "")).strip()
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    return bool(supplied) and hmac.compare_digest(supplied, expected)


def read_json_body(handler: Any, max_request_bytes: int) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        raise RequestError(411, "Content-Length is required")
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise RequestError(400, "Content-Length must be an integer") from exc
    if content_length < 0:
        raise RequestError(400, "Content-Length must not be negative")
    if content_length > int(max_request_bytes):
        raise RequestError(413, "request body is too large")
    raw = handler.rfile.read(content_length)
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestError(400, "request body must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise RequestError(400, "request body must be a JSON object")
    return payload


def parse_allowed_roots(default_root: Path) -> tuple[Path, ...]:
    roots = [default_root.expanduser().resolve()]
    configured = os.environ.get("HAT4D_ALLOWED_ROOTS", "")
    for value in configured.split(os.pathsep):
        if value.strip():
            roots.append(Path(value).expanduser().resolve())
    return tuple(dict.fromkeys(roots))


def ensure_path_allowed(value: str | Path, allowed_roots: Iterable[Path]) -> Path:
    path = Path(value).expanduser().resolve()
    for root in allowed_roots:
        try:
            path.relative_to(root)
            return path
        except ValueError:
            continue
    raise RequestError(403, "requested path is outside HAT4D_ALLOWED_ROOTS")


def validate_payload_paths(payload: dict[str, Any], allowed_roots: Iterable[Path]) -> None:
    def visit(value: Any, key: str = "") -> None:
        if key in IDENTIFIER_FIELD_NAMES and value is not None:
            validate_identifier(str(value), key)
        if key in PATH_COLLECTION_NAMES:
            if isinstance(value, dict):
                for child in value.values():
                    visit(child, key)
                return
            if isinstance(value, list):
                for child in value:
                    visit(child, key)
                return
            if isinstance(value, str):
                ensure_path_allowed(value, allowed_roots)
                return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (
            key in PATH_FIELD_NAMES or key.endswith(("_dir", "_file", "_path", "_root"))
        ):
            ensure_path_allowed(value, allowed_roots)

    visit(payload)


def validate_identifier(value: str, name: str = "identifier") -> str:
    candidate = str(value)
    if IDENTIFIER_RE.fullmatch(candidate) is None or candidate in {".", ".."}:
        raise RequestError(400, f"invalid {name}")
    return candidate


def client_headers(*, include_json: bool = False) -> dict[str, str]:
    headers = {"Content-Type": "application/json"} if include_json else {}
    token = os.environ.get("HAT4D_API_TOKEN", "").strip()
    if token:
        headers["X-HAT4D-Token"] = token
    return headers
