import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "viser_compat.py"


def _load_module():
    assert MODULE_PATH.is_file(), "Viser compatibility helper is missing"
    spec = importlib.util.spec_from_file_location("viser_compat", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_key_handler_is_optional_for_public_viser_clients() -> None:
    module = _load_module()

    assert module.register_key_handler(SimpleNamespace(), lambda _event: None) is False


def test_key_handler_registers_with_custom_viser_clients() -> None:
    module = _load_module()
    callbacks = []
    client = SimpleNamespace(on_key=callbacks.append)

    def callback(_event) -> None:  # noqa: ANN001
        return None

    assert module.register_key_handler(client, callback) is True
    assert callbacks == [callback]
