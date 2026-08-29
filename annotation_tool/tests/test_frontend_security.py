from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VLM_UI = ROOT / "viewer_plugins" / "vlm_assistor" / "ui.py"
HEADLESS_3D = ROOT / "headless_3d_gen.py"
HEADLESS_4D = ROOT / "headless_4dgen.py"
VLM_BACKEND = ROOT / "viewer_plugins" / "vlm_assistor" / "backend_server.py"


def test_vlm_api_key_is_not_rendered_in_browser_ui() -> None:
    text = VLM_UI.read_text(encoding="utf-8")

    assert "Local VLM API key" not in text
    assert "self.agent0_local_api_key" not in text
    assert 'os.getenv("VLM_LOCAL_API_KEY")' not in text


def test_headless_entry_points_do_not_accept_keys_on_command_line() -> None:
    for path in (HEADLESS_3D, HEADLESS_4D):
        text = path.read_text(encoding="utf-8")
        assert "--vlm-local-api-key" not in text


def test_vlm_backend_reads_api_key_from_environment_only() -> None:
    text = VLM_BACKEND.read_text(encoding="utf-8")

    assert "--openai-api-key" not in text
    assert 'os.getenv("OPENAI_API_KEY")' in text


def test_headless_4d_contains_only_single_run_orchestration() -> None:
    text = HEADLESS_4D.read_text(encoding="utf-8")

    for prohibited in (
        "--batch-json",
        "--batch-file",
        "--max-workers",
        "ThreadPoolExecutor",
        "subprocess.run",
        "_run_batch",
    ):
        assert prohibited not in text
