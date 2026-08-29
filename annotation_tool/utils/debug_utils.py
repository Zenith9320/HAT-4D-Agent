"""Opt-in debug artifact configuration."""

import os
from pathlib import Path


def _component_path(root: Path, component: str, *, create: bool) -> Path:
    component_path = Path(component)
    if component_path.is_absolute() or ".." in component_path.parts:
        raise ValueError(f"invalid output component path: {component}")
    path = root.expanduser().resolve() / component_path
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configured_debug_dir(component: str, *, create: bool = False) -> Path | None:
    """Return a component debug directory when ``HAT4D_DEBUG_DIR`` is set."""
    root = os.environ.get("HAT4D_DEBUG_DIR", "").strip()
    if not root:
        return None
    return _component_path(Path(root), component, create=create)


def configured_runtime_dir(
    component: str,
    *,
    default_root: Path,
    create: bool = False,
) -> Path:
    """Return a component runtime directory below the configured output root."""
    root = Path(os.environ.get("HAT4D_RUNTIME_DIR", "").strip() or default_root)
    return _component_path(root, component, create=create)
