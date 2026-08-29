import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "debug_utils.py"


def _load_debug_utils():
    assert MODULE_PATH.is_file(), "debug configuration helper is missing"
    spec = importlib.util.spec_from_file_location("debug_utils", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_debug_output_is_disabled_by_default(monkeypatch) -> None:
    debug_utils = _load_debug_utils()
    monkeypatch.delenv("HAT4D_DEBUG_DIR", raising=False)

    assert debug_utils.configured_debug_dir("optimizer") is None


def test_debug_output_uses_component_subdirectory(tmp_path: Path, monkeypatch) -> None:
    debug_utils = _load_debug_utils()
    monkeypatch.setenv("HAT4D_DEBUG_DIR", str(tmp_path))

    assert debug_utils.configured_debug_dir("optimizer") == tmp_path / "optimizer"
    assert not (tmp_path / "optimizer").exists()
    assert debug_utils.configured_debug_dir("optimizer", create=True) == tmp_path / "optimizer"
    assert (tmp_path / "optimizer").is_dir()


def test_runtime_output_uses_configurable_root(tmp_path: Path, monkeypatch) -> None:
    debug_utils = _load_debug_utils()
    default_root = tmp_path / "default"
    configured_root = tmp_path / "configured"
    monkeypatch.delenv("HAT4D_RUNTIME_DIR", raising=False)
    assert debug_utils.configured_runtime_dir("sv4d", default_root=default_root) == default_root / "sv4d"

    monkeypatch.setenv("HAT4D_RUNTIME_DIR", str(configured_root))
    path = debug_utils.configured_runtime_dir("sv4d", default_root=default_root, create=True)
    assert path == configured_root / "sv4d"
    assert path.is_dir()
