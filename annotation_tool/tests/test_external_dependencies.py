import json
import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "external_dependencies.json"
UNAVAILABLE_PATH = ROOT / "external_dependencies.unavailable.json"
INSTALLER_PATH = ROOT / "scripts" / "install_external_dependencies.py"


def _load_installer():
    assert INSTALLER_PATH.is_file(), "external dependency installer is missing"
    spec = importlib.util.spec_from_file_location("install_external_dependencies", INSTALLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_dependencies_are_pinned_and_relative() -> None:
    assert LOCK_PATH.is_file(), "external dependency lock is missing"
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert data
    for item in data:
        assert item["url"].startswith("https://")
        assert len(item["revision"]) == 40
        assert all(char in "0123456789abcdef" for char in item["revision"])
        path = Path(item["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts


def test_installable_lock_contains_only_publicly_reproducible_sources() -> None:
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked = {item["name"]: (item["url"], item["revision"]) for item in data}

    assert locked == {
        "sam-3d-objects": (
            "https://github.com/facebookresearch/sam-3d-objects.git",
            "f91db411c50efee93d8db7aeb323885650f6f722",
        ),
        "segment-anything": (
            "https://github.com/facebookresearch/segment-anything.git",
            "6fdee8f2727f4506cfbbe553e23b895e27956588",
        ),
        "Qwen3-VL": (
            "https://github.com/QwenLM/Qwen3-VL.git",
            "96588727e44c78b25ba03ea03b8e12f7e64fd0da",
        ),
    }


def test_unavailable_dependencies_are_separate_from_installer_lock() -> None:
    installable_names = {
        item["name"] for item in json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    }
    unavailable = json.loads(UNAVAILABLE_PATH.read_text(encoding="utf-8"))

    assert {item["name"] for item in unavailable} == {"VLM_toolkit_cross_gpfs"}
    assert installable_names.isdisjoint(item["name"] for item in unavailable)
    for item in unavailable:
        assert item["availability"] == "not-publicly-installable"
        assert item["reason"]
        assert len(item["source_revision"]) == 40


def test_installer_rejects_paths_outside_tool_root(tmp_path: Path) -> None:
    installer = _load_installer()
    lock_path = tmp_path / "external_dependencies.json"
    lock_path.write_text(
        json.dumps(
            [
                {
                    "name": "unsafe",
                    "path": "../outside",
                    "url": "https://example.com/repo.git",
                    "revision": "a" * 40,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative"):
        installer.load_dependencies(lock_path, tmp_path)


def test_dependency_status_distinguishes_missing_and_ready(tmp_path: Path) -> None:
    installer = _load_installer()
    destination = tmp_path / "dependency"
    dependency = installer.Dependency(
        name="demo",
        path=Path("dependency"),
        url="https://example.com/repo.git",
        revision="0" * 40,
        destination=destination,
    )
    assert installer.dependency_status(dependency) == "missing"

    destination.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=destination, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=destination, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=destination, check=True)
    (destination / "README.md").write_text("demo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=destination, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=destination, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ready = installer.Dependency(
        name="demo",
        path=Path("dependency"),
        url="https://example.com/repo.git",
        revision=revision,
        destination=destination,
    )
    assert installer.dependency_status(ready) == "ready"

    (destination / "README.md").unlink()
    assert installer.dependency_status(ready) == "worktree-dirty"

    dirty_mismatch = installer.Dependency(
        name="demo",
        path=Path("dependency"),
        url="https://example.com/repo.git",
        revision="f" * 40,
        destination=destination,
    )
    assert installer.dependency_status(dirty_mismatch) == "worktree-dirty"
    with pytest.raises(ValueError, match="modified or incomplete"):
        installer.install_dependency(dirty_mismatch)
