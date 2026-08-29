import importlib.util
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_test_services.py"


def _load_module():
    assert SCRIPT.is_file(), "service smoke runner is missing"
    spec = importlib.util.spec_from_file_location("smoke_test_services", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _health_server(payload: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_runner_reports_ready_and_skipped_services() -> None:
    module = _load_module()
    with _health_server(b'{"status":"ok","service":"roi_backend"}') as url:
        report = module.check_services(
            {"roi_backend": url, "sam3d_backend": None},
            timeout=1,
        )

    assert report["roi_backend"]["status"] == "ready"
    assert report["sam3d_backend"]["status"] == "skipped"


def test_smoke_runner_fails_on_wrong_service_identity() -> None:
    module = _load_module()
    with _health_server(b'{"status":"ok","service":"wrong_backend"}') as url:
        report = module.check_services({"roi_backend": url}, timeout=1)

    assert report["roi_backend"]["status"] == "failed"
    assert "wrong_backend" in report["roi_backend"]["error"]
