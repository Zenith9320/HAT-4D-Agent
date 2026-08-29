#!/usr/bin/env python3
"""Scan annotation-tool files for common publication blockers."""

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = TOOL_ROOT / "publication_allowlist.json"
MAX_FILE_SIZE = 5 * 1024 * 1024
TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
GENERATED_NAMES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "analyze_OIK",
    "dataset",
    "outputs",
    "render_res",
    "results_workspace",
    "tmp",
    "workspace",
}
VENDORED_SOURCE_PREFIXES = (
    "model4dgen/L4GM_MoGA/",
    "model4dgen/generative-models/",
)
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
HOME_RELATIVE_PREFIX = re.escape("~" + "/")
INTERNAL_PATH_RE = re.compile(
    rf"(?:/(?:inspire|home|root)/|{HOME_RELATIVE_PREFIX})[^\s\"'`]+"
)
INTERNAL_PATH_BYTES_RE = re.compile(
    rb"/(?:inspire|home|root)/[A-Za-z0-9._/-]{3,}"
)
CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\b\s*=\s*[\"'][^\"']{8,}[\"']"
)
COMMENTED_CODE_RE = re.compile(
    r"^(?:async\s+def\s|def\s|class\s|from\s|import\s|if\s|elif\s|else:|for\s|while\s|"
    r"try:|except\s|with\s|return\b|raise\s|[A-Za-z_]\w*(?:\.\w+)*\s*=|"
    r"[A-Za-z_]\w*(?:\.\w+)*\()"
)


@dataclass(frozen=True, order=True)
class Finding:
    kind: str
    path: Path
    line: int
    message: str


def _git_ignored(path: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path)],
            cwd=path.parent if path.parent.exists() else Path.cwd(),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _load_multilingual_allowlist(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("multilingual_string_files", [])
    if not isinstance(entries, list):
        raise ValueError("multilingual_string_files must be a list")
    allowlist: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("multilingual allowlist entries must be objects")
        item_path = str(entry.get("path", ""))
        reason = str(entry.get("reason", "")).strip()
        if not item_path or not reason:
            raise ValueError("multilingual allowlist entries require path and reason")
        allowlist[item_path] = reason
    return allowlist


def _docstring_lines(source: str) -> set[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    nodes = [tree, *ast.walk(tree)]
    for node in nodes:
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        start = int(getattr(first, "lineno", 0))
        end = int(getattr(first, "end_lineno", start))
        lines.update(range(start, end + 1))
    return lines


def _scan_python_text(
    path: Path,
    relative_path: str,
    source: str,
    multilingual_allowlist: dict[str, str],
) -> list[Finding]:
    findings: list[Finding] = []
    doc_lines = _docstring_lines(source)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if HAN_RE.search(token.string) is None:
                continue
            if token.type == tokenize.COMMENT:
                findings.append(
                    Finding("chinese-comment", path, token.start[0], "Chinese text in a Python comment")
                )
            elif token.type == tokenize.STRING and token.start[0] in doc_lines:
                findings.append(
                    Finding("chinese-docstring", path, token.start[0], "Chinese text in a Python docstring")
                )
            elif token.type == tokenize.STRING and relative_path not in multilingual_allowlist:
                findings.append(
                    Finding(
                        "chinese-string",
                        path,
                        token.start[0],
                        "Chinese text outside the documented multilingual prompt allowlist",
                    )
                )
    except (IndentationError, tokenize.TokenError):
        pass
    block_start = 0
    code_line_count = 0
    for line_number, line in enumerate([*source.splitlines(), ""], start=1):
        match = re.match(r"^\s*#(.*)$", line)
        if match:
            body = match.group(1).lstrip("#").strip()
            if COMMENTED_CODE_RE.match(body):
                block_start = block_start or line_number
                code_line_count += 1
            continue
        if code_line_count >= 5:
            findings.append(
                Finding(
                    "commented-code",
                    path,
                    block_start,
                    f"comment block contains {code_line_count} code-like lines",
                )
            )
        block_start = 0
        code_line_count = 0
    return findings


def _scan_text_file(
    path: Path,
    relative_path: str,
    multilingual_allowlist: dict[str, str],
) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[Finding] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        match = INTERNAL_PATH_RE.search(line)
        if match:
            findings.append(
                Finding("internal-path", path, line_number, f"internal absolute path: {match.group(0)}")
            )
        if CREDENTIAL_RE.search(line):
            findings.append(
                Finding("credential", path, line_number, "possible hard-coded credential")
            )
    is_vendored_source = relative_path.startswith(VENDORED_SOURCE_PREFIXES)
    if path.suffix.lower() == ".py" and not is_vendored_source:
        findings.extend(_scan_python_text(path, relative_path, source, multilingual_allowlist))
    return findings


def _scan_binary_file(path: Path) -> list[Finding]:
    """Detect machine-specific paths embedded in binary metadata."""
    overlap = b""
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                data = overlap + chunk
                match = INTERNAL_PATH_BYTES_RE.search(data)
                if match:
                    value = match.group(0).decode("utf-8", errors="replace")
                    return [
                        Finding(
                            "internal-path",
                            path,
                            0,
                            f"internal absolute path in binary metadata: {value}",
                        )
                    ]
                overlap = data[-4096:]
    except OSError:
        return []
    return []


def _iter_paths(root: Path) -> Iterable[tuple[Path, str]]:
    if root.is_file() or root.is_symlink():
        yield root, root.name
        return
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                yield path, path.relative_to(root).as_posix()
            elif _git_ignored(path):
                continue
            else:
                yield path, path.relative_to(root).as_posix()
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in file_names:
            path = current_path / name
            if not _git_ignored(path):
                yield path, path.relative_to(root).as_posix()


def scan_paths(
    paths: Sequence[Path],
    *,
    allowlist_path: Path | None = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> list[Finding]:
    """Return publication findings for files below ``paths``."""
    allowlist = _load_multilingual_allowlist(allowlist_path)
    findings: list[Finding] = []
    seen: set[Path] = set()
    for supplied_root in paths:
        root = supplied_root.expanduser().resolve()
        for path, relative_path in _iter_paths(root):
            if path in seen:
                continue
            seen.add(path)
            if path.is_symlink():
                findings.append(Finding("symlink", path, 0, "symbolic link is not publishable"))
                continue
            if path.name in GENERATED_NAMES:
                findings.append(Finding("generated-path", path, 0, "generated or local-only path"))
                if path.is_dir():
                    continue
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_file_size:
                findings.append(
                    Finding("large-file", path, 0, f"file is {size} bytes; limit is {max_file_size}")
                )
            if path.suffix.lower() in TEXT_SUFFIXES:
                findings.extend(_scan_text_file(path, relative_path, allowlist))
            else:
                findings.extend(_scan_binary_file(path))
    return sorted(findings)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to scan.")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="JSON allowlist for intentional multilingual prompt strings.",
    )
    parser.add_argument("--max-file-size", type=int, default=MAX_FILE_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    findings = scan_paths(
        args.paths,
        allowlist_path=args.allowlist,
        max_file_size=args.max_file_size,
    )
    for finding in findings:
        location = f"{finding.path}:{finding.line}" if finding.line else str(finding.path)
        print(f"{location}: {finding.kind}: {finding.message}")
    print(f"publication scan: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"publication scan failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
