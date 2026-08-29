"""Compatibility helpers for public and locally extended Viser clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def register_key_handler(client: Any, callback: Callable[[Any], None]) -> bool:
    """Register a custom key callback when the Viser client exposes one."""
    on_key = getattr(client, "on_key", None)
    if not callable(on_key):
        return False
    on_key(callback)
    return True
