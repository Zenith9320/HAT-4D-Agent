"""SAM3D backend module.

This package provides the Sam3DBackend class for handling GPU-intensive
SAM and SAM3D inference, plus the HTTP server for remote access.
"""
from __future__ import annotations

from .core import BackendConfig, Sam3DBackend


def parse_args(argv=None):  # noqa: ANN001, ANN201
    from .server import parse_args as _parse_args

    return _parse_args(argv)


def main() -> None:
    from .server import main as _main

    _main()

__all__ = ["BackendConfig", "Sam3DBackend", "main", "parse_args"]
