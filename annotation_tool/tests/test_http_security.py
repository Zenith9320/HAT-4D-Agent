import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from http_security import (  # noqa: E402
    RequestError,
    client_headers,
    ensure_path_allowed,
    read_json_body,
    request_is_authorized,
    require_secure_bind,
    validate_identifier,
    validate_payload_paths,
)


def test_non_loopback_bind_requires_token() -> None:
    with pytest.raises(ValueError, match="HAT4D_API_TOKEN"):
        require_secure_bind("0.0.0.0", "")
    require_secure_bind("0.0.0.0", "secret")
    require_secure_bind("127.0.0.1", "")


def test_bearer_and_service_token_authentication() -> None:
    assert request_is_authorized({"Authorization": "Bearer secret"}, "secret")
    assert request_is_authorized({"X-HAT4D-Token": "secret"}, "secret")
    assert not request_is_authorized({}, "secret")


def test_client_headers_use_environment_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAT4D_API_TOKEN", "secret")
    assert client_headers(include_json=True) == {
        "Content-Type": "application/json",
        "X-HAT4D-Token": "secret",
    }


def test_request_body_limit_is_enforced() -> None:
    raw = json.dumps({"value": "large"}).encode("utf-8")
    handler = SimpleNamespace(
        headers={"Content-Length": str(len(raw))},
        rfile=BytesIO(raw),
    )
    with pytest.raises(RequestError) as caught:
        read_json_body(handler, max_request_bytes=4)
    assert caught.value.status == 413


def test_paths_and_identifiers_cannot_escape_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    assert ensure_path_allowed(allowed / "output", [allowed]) == allowed / "output"
    with pytest.raises(RequestError) as caught:
        ensure_path_allowed(tmp_path / "outside", [allowed])
    assert caught.value.status == 403
    with pytest.raises(RequestError):
        validate_identifier("../../session", "session_id")


@pytest.mark.parametrize(
    "payload",
    [
        {"paths": {"gs_ply": "/private/model.ply"}},
        {"paths": {"anchor_image": "/private/frame.png"}},
        {"frame_paths": ["/private/frame.png"]},
        {"view_paths": {"input": "/private/frame.png"}},
        {"session_id": "../../session"},
        {"experiment_id": "../experiment"},
    ],
)
def test_payload_validation_covers_path_aliases_and_ids(
    tmp_path: Path,
    payload: dict,
) -> None:
    with pytest.raises(RequestError):
        validate_payload_paths(payload, [tmp_path])
