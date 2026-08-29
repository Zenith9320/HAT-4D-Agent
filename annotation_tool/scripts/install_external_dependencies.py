#!/usr/bin/env python3
"""Install external annotation-tool repositories at locked revisions."""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = TOOL_ROOT / "external_dependencies.json"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Dependency:
    name: str
    path: Path
    url: str
    revision: str
    destination: Path


def load_dependencies(lock_path: Path, root: Path) -> list[Dependency]:
    """Load and validate dependency metadata below ``root``."""
    lock_path = lock_path.expanduser().resolve()
    root = root.expanduser().resolve()
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("dependency lock must contain a non-empty JSON list")

    dependencies: list[Dependency] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"dependency entry {index} must be an object")
        try:
            name = str(item["name"])
            path = Path(str(item["path"]))
            url = str(item["url"])
            revision = str(item["revision"])
        except KeyError as exc:
            raise ValueError(f"dependency entry {index} is missing {exc.args[0]}") from exc

        if not name or name in names:
            raise ValueError(f"dependency name is empty or duplicated: {name!r}")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"dependency path must be a safe relative path: {path}")
        if not url.startswith("https://"):
            raise ValueError(f"dependency URL must use HTTPS: {name}")
        if REVISION_RE.fullmatch(revision) is None:
            raise ValueError(f"dependency revision must be a 40-character lowercase SHA: {name}")

        destination = (root / path).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"dependency path resolves outside the tool root: {path}") from exc

        names.add(name)
        dependencies.append(Dependency(name, path, url, revision, destination))
    return dependencies


def _git_output(arguments: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def dependency_status(dependency: Dependency) -> str:
    """Return a stable status without modifying the dependency checkout."""
    if not dependency.destination.exists():
        return "missing"
    if not dependency.destination.is_dir():
        return "not-a-directory"
    try:
        revision = _git_output(["rev-parse", "HEAD"], dependency.destination)
    except (OSError, subprocess.CalledProcessError):
        return "not-a-git-repository"
    try:
        worktree = _git_output(
            ["status", "--porcelain", "--untracked-files=no"],
            dependency.destination,
        )
    except (OSError, subprocess.CalledProcessError):
        return "worktree-unreadable"
    if worktree:
        return "worktree-dirty"
    if revision != dependency.revision:
        return f"revision-mismatch:{revision}"
    return "ready"


def install_dependency(dependency: Dependency) -> None:
    """Clone or update one dependency and detach it at the locked revision."""
    destination = dependency.destination
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"refusing to replace non-directory path: {dependency.path}")
    if destination.exists() and any(destination.iterdir()):
        status = dependency_status(dependency)
        if status == "not-a-git-repository":
            raise ValueError(f"refusing to overwrite non-Git directory: {dependency.path}")
        if status in {"worktree-dirty", "worktree-unreadable"}:
            raise ValueError(
                f"refusing to overwrite a modified or incomplete checkout: {dependency.path}"
            )

    if not destination.exists() or not any(destination.iterdir()):
        if destination.exists():
            destination.rmdir()
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--no-checkout",
                dependency.url,
                str(destination),
            ],
            check=True,
        )

    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", dependency.revision],
        cwd=destination,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", dependency.revision],
        cwd=destination,
        check=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="Dependency lock JSON file.")
    parser.add_argument("--check", action="store_true", help="Check installed revisions without writing.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="NAME",
        help="Install or check only this dependency; repeat for multiple names.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dependencies = load_dependencies(args.lock, args.lock.expanduser().resolve().parent)
    requested = set(args.only)
    if requested:
        known = {dependency.name for dependency in dependencies}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown dependency names: {', '.join(unknown)}")
        dependencies = [dependency for dependency in dependencies if dependency.name in requested]

    failed = False
    for dependency in dependencies:
        status = dependency_status(dependency)
        if args.check:
            print(f"[{status}] {dependency.name}: {dependency.path}")
            failed = failed or status != "ready"
            continue
        if status != "ready":
            print(f"[install] {dependency.name}: {dependency.path}")
            install_dependency(dependency)
        final_status = dependency_status(dependency)
        print(f"[{final_status}] {dependency.name}: {dependency.path}")
        failed = failed or final_status != "ready"
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"dependency setup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
