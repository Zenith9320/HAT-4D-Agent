import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "scripts" / "check_publication.py"


def _load_checker():
    assert CHECKER_PATH.is_file(), "publication checker is missing"
    spec = importlib.util.spec_from_file_location("check_publication", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_detects_prohibited_publication_content(tmp_path: Path) -> None:
    checker = _load_checker()
    (tmp_path / "target.txt").write_text("target\n", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(tmp_path / "target.txt")
    (tmp_path / "large.bin").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"cache")
    internal_path = "/" + "inspire/private/data"
    (tmp_path / "path.py").write_text(f'ROOT = "{internal_path}"\n', encoding="utf-8")
    credential = "API_" + 'KEY = "sk-test-secret-value"\n'
    (tmp_path / "secret.py").write_text(credential, encoding="utf-8")
    chinese_comment = "# \u8fd9\u662f\u4e2d\u6587\u6ce8\u91ca\nVALUE = 1\n"
    (tmp_path / "comment.py").write_text(chinese_comment, encoding="utf-8")

    findings = checker.scan_paths([tmp_path])
    kinds = {finding.kind for finding in findings}

    assert {
        "symlink",
        "large-file",
        "generated-path",
        "internal-path",
        "credential",
        "chinese-comment",
    } <= kinds


def test_checker_accepts_clean_english_source(tmp_path: Path) -> None:
    checker = _load_checker()
    (tmp_path / "module.py").write_text(
        '"""Small public module."""\n\nVALUE = "safe"\n',
        encoding="utf-8",
    )

    assert checker.scan_paths([tmp_path]) == []


def test_checker_still_scans_vendored_source_for_internal_paths(tmp_path: Path) -> None:
    checker = _load_checker()
    vendored = tmp_path / "model4dgen" / "L4GM_MoGA"
    vendored.mkdir(parents=True)
    multilingual_comment = "# Original multilingual comment: " + "\u4fdd\u7559"
    internal_path = "/" + "inspire/private/data"
    (vendored / "module.py").write_text(
        f'{multilingual_comment}\nROOT = "{internal_path}"\n',
        encoding="utf-8",
    )

    findings = checker.scan_paths([tmp_path])

    assert {finding.kind for finding in findings} == {"internal-path"}


def test_checker_detects_home_relative_internal_paths(tmp_path: Path) -> None:
    checker = _load_checker()
    internal_path = "~" + "/private-project/cache"
    (tmp_path / "config.sh").write_text(
        f"CACHE_ROOT={internal_path}\n",
        encoding="utf-8",
    )

    findings = checker.scan_paths([tmp_path])

    assert any(
        finding.kind == "internal-path" and internal_path in finding.message
        for finding in findings
    )


def test_checker_detects_internal_paths_in_binary_metadata(tmp_path: Path) -> None:
    checker = _load_checker()
    internal_path = b"/" + b"inspire/private/render/source.blend"
    (tmp_path / "frame.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"tEXtFile\x00" + internal_path + b"\x00"
    )

    findings = checker.scan_paths([tmp_path])

    assert any(
        finding.kind == "internal-path"
        and internal_path.decode() in finding.message
        for finding in findings
    )


def test_checker_detects_large_commented_out_implementation(tmp_path: Path) -> None:
    checker = _load_checker()
    (tmp_path / "stale.py").write_text(
        "# def stale():\n"
        "#     value = 1\n"
        "#     if value:\n"
        "#         print(value)\n"
        "#     return value\n",
        encoding="utf-8",
    )

    findings = checker.scan_paths([tmp_path])

    assert "commented-code" in {finding.kind for finding in findings}


def test_checker_skips_gitignored_generated_paths(tmp_path: Path, monkeypatch) -> None:
    checker = _load_checker()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    source_root = tmp_path / "source"
    cache = source_root / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.pyc").write_bytes(b"cache")
    monkeypatch.chdir(tmp_path)

    assert checker.scan_paths([Path("source")]) == []
