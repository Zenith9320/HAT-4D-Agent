#!/usr/bin/env python3
"""Backward-compatible entry point for SAM3D backend server.

This file allows running the backend directly:
    python viewer_plugins/sam3d_gen/backend.py --port 8015

All implementation has been refactored into the backend/ package.
"""
from __future__ import annotations

import sys
from pathlib import Path

# When run directly (python backend.py), ensure package imports work.
if __name__ == "__main__":
    # Add parent directories to sys.path for direct execution.
    _this_file = Path(__file__).resolve()
    _plugin_root = _this_file.parent
    _viewer_plugins = _plugin_root.parent
    _repo_root = _viewer_plugins.parent
    for p in [str(_repo_root), str(_viewer_plugins), str(_plugin_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Import from the backend subpackage directly.
    from backend import main
    main()
else:
    # Re-export for backward compatibility when imported as a module.
    from .backend import BackendConfig, Sam3DBackend, main, parse_args

    __all__ = ["BackendConfig", "Sam3DBackend", "main", "parse_args"]
