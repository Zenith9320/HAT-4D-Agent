import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENV = {**os.environ, "PYTHONPATH": str(ROOT)}
PYTHON = shutil.which("python")
assert PYTHON is not None
BACKENDS = {
    "roi_backend": "roi_backend",
    "mvtap_backend": "mvtap_backend",
    "viewer_plugins.sam3d_gen.backend.server": "sam3d_backend",
    "viewer_plugins.vlm_assistor.backend_server": "vlm_backend",
}
ERROR_PROBES = {
    "roi_backend": ("/session/start", {"input_dir": "/private/hat4d-secret/frames"}),
    "mvtap_backend": ("/track_roi", {"paths": {"workspace_root": "/private/hat4d-secret"}}),
    "viewer_plugins.sam3d_gen.backend.server": (
        "/sam/set_image",
        {"image_path": "/private/hat4d-secret/frame.png"},
    ),
    "viewer_plugins.vlm_assistor.backend_server": (
        "/agent/video_understanding",
        {"video_dir": "/private/hat4d-secret/frames"},
    ),
}


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, "-c", code],
        cwd=ROOT,
        env=PYTHON_ENV,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_json(url: str, payload: dict) -> tuple[int, str, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return (
            int(response.status),
            str(response.headers.get("Content-Type", "")),
            json.load(response),
        )


@pytest.mark.parametrize("module_name", [*BACKENDS, "viser_frontend"])
def test_runtime_module_imports_without_external_checkouts(module_name: str) -> None:
    result = _run_python(f"import {module_name}; print('ok')")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


@pytest.mark.parametrize("module_name", BACKENDS)
def test_backend_default_host_is_loopback(module_name: str) -> None:
    result = _run_python(f"import {module_name} as module; print(module.parse_args([]).host)")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("127.0.0.1")


@pytest.mark.parametrize(("module_name", "service_name"), BACKENDS.items())
def test_backend_health_payload_is_model_independent(
    module_name: str,
    service_name: str,
) -> None:
    result = _run_python(
        f"import json, {module_name} as module; print(json.dumps(module.health_payload(), sort_keys=True))"
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["service"] == service_name


@pytest.mark.parametrize(("module_name", "service_name"), BACKENDS.items())
def test_backend_health_only_server(
    module_name: str,
    service_name: str,
) -> None:
    port = _free_port()
    process = subprocess.Popen(
        [PYTHON, "-m", module_name, "--health-only", "--port", str(port)],
        cwd=ROOT,
        env=PYTHON_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"backend exited early\nstdout:\n{stdout}\nstderr:\n{stderr}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    payload = json.load(response)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.1)
        else:
            pytest.fail(f"backend did not become healthy: {last_error}")

        assert payload == {"status": "ok", "service": service_name}

        status, content_type, error_payload = _post_json(
            f"http://127.0.0.1:{port}/unknown",
            {},
        )
        assert status == 404
        assert content_type.startswith("application/json")
        assert isinstance(error_payload, dict)

        path, probe_payload = ERROR_PROBES[module_name]
        status, content_type, error_payload = _post_json(
            f"http://127.0.0.1:{port}{path}",
            probe_payload,
        )
        serialized = json.dumps(error_payload).lower()
        assert status == 500
        assert content_type.startswith("application/json")
        assert "traceback" not in serialized
        assert "/private/hat4d-secret" not in serialized
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_gaussian_renderer_reports_missing_gsplat() -> None:
    result = _run_python(
        "import core.gs as module; "
        "module.rasterization = None; "
        "\ntry:\n module._require_rasterization()"
        "\nexcept RuntimeError as exc:\n print(exc)"
        "\nelse:\n raise SystemExit('expected a missing-dependency error')"
    )

    assert result.returncode == 0, result.stderr
    assert "gsplat" in result.stdout.lower()


def test_roi_import_does_not_initialize_optimizer_models() -> None:
    result = _run_python("import roi_backend; print(roi_backend.ROIRequestHandler.fixer)")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("None")
    assert "torch.load" not in result.stderr


def test_frontend_interrupt_stops_server_cleanly() -> None:
    result = _run_python(
        "from types import SimpleNamespace; import viser_frontend as module; "
        "app = module.ViewerApp.__new__(module.ViewerApp); "
        "app.workspace = SimpleNamespace(root='demo-workspace'); "
        "app.server_port = 8897; "
        "app.server = SimpleNamespace(stop=lambda: print('server-stopped')); "
        "module.time.sleep = lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()); "
        "app.run(); print('clean-exit')"
    )

    assert result.returncode == 0, result.stderr
    assert "server-stopped" in result.stdout
    assert result.stdout.strip().endswith("clean-exit")
    assert "Traceback" not in result.stderr


def test_frontend_default_host_is_loopback() -> None:
    result = _run_python(
        "import viser_frontend as module; print(module.parse_args([]).host)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("127.0.0.1")


def test_workspace_stages_gaussians_without_mutating_source(tmp_path: Path) -> None:
    source_gs = tmp_path / "source-gs"
    input_dir = tmp_path / "input"
    eval_dir = tmp_path / "eval"
    workspace_root = tmp_path / "workspaces"
    for directory in (source_gs, input_dir, eval_dir, workspace_root):
        directory.mkdir()
    source_ply = source_gs / "000000.ply"
    source_ply.write_bytes(b"original-ply")

    result = _run_python(
        "import json; from pathlib import Path; from viser_frontend import WorkspaceManager; "
        f"manager = WorkspaceManager(Path({str(workspace_root)!r}), Path({str(source_gs)!r}), "
        f"Path({str(input_dir)!r}), Path({str(eval_dir)!r})); "
        "target = manager.gs_dir / '000000.ply'; target.write_bytes(b'workspace-edit'); "
        "print(json.dumps({'root': str(manager.root), 'is_symlink': manager.gs_dir.is_symlink()}))"
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload["is_symlink"] is False
    assert (Path(payload["root"]) / "gs_sequence" / "000000.ply").read_bytes() == b"workspace-edit"
    assert source_ply.read_bytes() == b"original-ply"
