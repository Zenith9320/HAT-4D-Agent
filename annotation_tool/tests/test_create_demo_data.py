import os
import shutil
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "create_demo_data.py"
PYTHON = shutil.which("python")
assert PYTHON is not None


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_create_demo_data_is_small_deterministic_and_loadable(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(output), "--frame-count", "3"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    expected_stems = ["000000", "000001", "000002"]
    images = sorted((output / "input").glob("*.png"))
    plys = sorted((output / "gs").glob("*.ply"))
    assert [path.stem for path in images] == expected_stems
    assert [path.stem for path in plys] == expected_stems
    assert all(_png_size(path) == (64, 64) for path in images)
    assert (output / "eval_gt").is_dir()
    assert all(path.stat().st_size < 100 * 1024 for path in output.rglob("*") if path.is_file())
    assert all(b"format ascii 1.0" in path.read_bytes()[:100] for path in plys)

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    load_result = subprocess.run(
        [
            PYTHON,
            "-c",
            (
                "from pathlib import Path; "
                "from utils.ply_utils import load_splat_from_ply; "
                f"data = load_splat_from_ply(Path({str(plys[0])!r})); "
                "print(data.centers.shape)"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert load_result.returncode == 0, load_result.stderr
    assert "(48, 3)" in load_result.stdout


def test_create_demo_data_refuses_implicit_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    output.mkdir()
    (output / "keep.txt").write_text("user data\n", encoding="utf-8")

    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data\n"
    assert "--force" in result.stderr
