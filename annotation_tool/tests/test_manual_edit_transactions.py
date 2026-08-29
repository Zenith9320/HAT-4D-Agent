import sys
import types
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

injected_viser_stub = "viser.transforms" not in sys.modules
if injected_viser_stub:
    viser_module = types.ModuleType("viser")
    transforms_module = types.ModuleType("viser.transforms")

    class _SO3:
        @staticmethod
        def from_matrix(_matrix):
            return SimpleNamespace(
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )

    transforms_module.SO3 = _SO3
    viser_module.transforms = transforms_module
    sys.modules["viser"] = viser_module
    sys.modules["viser.transforms"] = transforms_module

from utils import ply_utils  # noqa: E402
from utils.ply_utils import SplatData, write_ply_vertices  # noqa: E402
from viewer_plugins.manual_edit.ops import ManualEditOps  # noqa: E402
from viewer_plugins.manual_edit.state import ManualEditState  # noqa: E402

if injected_viser_stub:
    sys.modules.pop("viser.transforms", None)
    sys.modules.pop("viser", None)


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
        ("rot_0", "<f4"),
        ("rot_1", "<f4"),
        ("rot_2", "<f4"),
        ("rot_3", "<f4"),
    ]
)


def _make_data(count: int, offset: float = 0.0) -> SplatData:
    centers = np.arange(count * 3, dtype=np.float32).reshape(count, 3) + offset
    return SplatData(
        centers=centers,
        rgbs=np.full((count, 3), 0.5, dtype=np.float32),
        opacities=np.full((count, 1), 0.5, dtype=np.float32),
        covariances=np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0),
        scales=np.ones((count, 3), dtype=np.float32),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1)),
        sh=np.zeros((count, 3), dtype=np.float32),
        opacity_raw=np.zeros(count, dtype=np.float32),
        centers_raw=centers.copy(),
    )


def _write_test_ply(
    path: Path,
    count: int = 3,
    ply_format: str = "binary_little_endian",
) -> None:
    vertices = np.zeros(count, dtype=PLY_DTYPE)
    vertices["rot_0"] = 1.0
    header = [
        b"ply\n",
        f"format {ply_format} 1.0\n".encode("ascii"),
        f"element vertex {count}\n".encode("ascii"),
        *[f"property float {name}\n".encode("ascii") for name in PLY_DTYPE.names],
        b"end_header\n",
    ]
    write_ply_vertices(header, vertices, path)


class _Sequence:
    def __init__(self, data: SplatData):
        self.cache = {0: data}

    def load(self, frame: int) -> SplatData:
        return self.cache[frame]

    def gs_path(self, _frame: int) -> Path:
        return Path("000000.ply")


class _Client:
    def __init__(self):
        self.notifications = []

    def add_notification(self, *args, **kwargs):
        self.notifications.append((args, kwargs))


def _make_ops(tmp_path: Path, data: SplatData) -> tuple[ManualEditOps, SimpleNamespace, _Client, Path]:
    ply_path = tmp_path / "000000.ply"
    _write_test_ply(ply_path, len(data.centers))
    sequence = _Sequence(data)
    app = SimpleNamespace(
        current_frame=0,
        current_view="input",
        sequence=sequence,
        workspace=SimpleNamespace(gs_path_for_frame=lambda _path: ply_path),
        selection_overlay_enabled=False,
        _update_gaussian_handle=lambda _data: None,
        _render_selection_overlay=lambda _enabled: None,
        _set_selection_info=lambda *_args: None,
    )
    state = ManualEditState()
    state.selected_indices = [0]
    state.selection_frame = 0
    state.selection_view = "input"
    return ManualEditOps(SimpleNamespace(app=app), state), app, _Client(), ply_path


def test_atomic_ply_write_preserves_original_when_replace_fails(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "scene.ply"
    _write_test_ply(path, 2)
    original = path.read_bytes()
    _, vertices, header = ply_utils.read_ply_vertices(path, return_header=True)
    vertices["x"] = 42.0

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(ply_utils.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        write_ply_vertices(header, vertices, path)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".scene.ply.*.tmp")) == []


def test_atomic_ply_write_preserves_ascii_format(tmp_path: Path) -> None:
    path = tmp_path / "scene_ascii.ply"
    vertices = np.zeros(2, dtype=PLY_DTYPE)
    vertices["x"] = [1.25, -2.5]
    header = [
        b"ply\n",
        b"format ascii 1.0\n",
        b"element vertex 2\n",
        *[f"property float {name}\n".encode("ascii") for name in PLY_DTYPE.names],
        b"end_header\n",
    ]

    write_ply_vertices(header, vertices, path)

    assert b"format ascii 1.0" in path.read_bytes()[:100]
    _, loaded = ply_utils.read_ply_vertices(path)
    np.testing.assert_array_equal(loaded["x"], vertices["x"])


@pytest.mark.parametrize("operation", ["delete_selected", "apply_attributes"])
def test_edit_failure_does_not_change_memory_or_history(tmp_path: Path, monkeypatch, operation: str) -> None:
    original = _make_data(3)
    ops, app, client, _ply_path = _make_ops(tmp_path, original)
    sentinel = object()
    ops.state.redo_history.append(sentinel)
    if operation == "delete_selected":
        monkeypatch.setattr(ops, "_write_filtered_ply", lambda *_args: (_ for _ in ()).throw(OSError("disk")))
        ops.delete_selected(client)
    else:
        monkeypatch.setattr(ops, "_update_ply_from_splats", lambda *_args: (_ for _ in ()).throw(OSError("disk")))
        ops.apply_attributes(client, (1.0, 0.0, 0.0), 0.2, (1, 1, 1), 1, (0, 0, 0), 1)

    assert app.sequence.cache[0] is original
    np.testing.assert_array_equal(app.sequence.cache[0].centers, _make_data(3).centers)
    np.testing.assert_array_equal(app.sequence.cache[0].rgbs, _make_data(3).rgbs)
    assert ops.state.edit_history == []
    assert ops.state.redo_history == [sentinel]
    assert client.notifications[-1][1]["color"] == "red"


@pytest.mark.parametrize("operation", ["undo", "redo"])
def test_history_failure_keeps_both_stacks_and_current_data(
    tmp_path: Path,
    monkeypatch,
    operation: str,
) -> None:
    current = _make_data(3, offset=10.0)
    target = _make_data(3, offset=20.0)
    ops, app, client, ply_path = _make_ops(tmp_path, current)
    snapshot = ops._capture_snapshot(target, ply_path)
    if operation == "undo":
        ops.state.edit_history.append(snapshot)
    else:
        ops.state.redo_history.append(snapshot)
    before_edit = list(ops.state.edit_history)
    before_redo = list(ops.state.redo_history)
    monkeypatch.setattr(ops, "_restore_snapshot", lambda *_args: (_ for _ in ()).throw(OSError("disk")))

    getattr(ops, operation)(client)

    assert app.sequence.cache[0] is current
    assert ops.state.edit_history == before_edit
    assert ops.state.redo_history == before_redo
    assert client.notifications[-1][1]["color"] == "red"


def test_transform_failure_rolls_back_live_preview(tmp_path: Path, monkeypatch) -> None:
    baseline = _make_data(3)
    preview = _make_data(3, offset=10.0)
    ops, app, client, _ply_path = _make_ops(tmp_path, preview)
    ops.state.bbox_frame_handle = SimpleNamespace(
        position=np.array([1.0, 0.0, 0.0]),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    ops.state.bbox_base_data = baseline
    ops.state.bbox_base_center = np.zeros(3, dtype=np.float32)
    ops.state.bbox_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ops.state.bbox_base_dims = np.ones(3, dtype=np.float32)
    ops.state.bbox_dimensions = np.ones(3, dtype=np.float32)
    monkeypatch.setattr(ops, "_update_ply_from_splats", lambda *_args: (_ for _ in ()).throw(OSError("disk")))

    ops.apply_transform(client)

    np.testing.assert_array_equal(app.sequence.cache[0].centers, baseline.centers)
    assert ops.state.edit_history == []
    assert ops.state.redo_history == []
    assert client.notifications[-1][1]["color"] == "red"


def test_delete_undo_redo_keeps_memory_and_full_ply_in_sync(tmp_path: Path) -> None:
    ops, app, client, ply_path = _make_ops(tmp_path, _make_data(3))
    _write_test_ply(ply_path, 3, ply_format="ascii")

    ops.delete_selected(client)
    _, deleted_vertices = ply_utils.read_ply_vertices(ply_path)
    assert b"format ascii 1.0" in ply_path.read_bytes()[:100]
    assert len(app.sequence.cache[0].centers) == 2
    assert len(deleted_vertices) == 2

    ops.undo(client)
    _, restored_vertices = ply_utils.read_ply_vertices(ply_path)
    assert len(app.sequence.cache[0].centers) == 3
    assert len(restored_vertices) == 3

    ops.redo(client)
    _, redone_vertices = ply_utils.read_ply_vertices(ply_path)
    assert len(app.sequence.cache[0].centers) == 2
    assert len(redone_vertices) == 2


def test_attributes_then_transform_undo_keeps_memory_and_ply_in_sync(tmp_path: Path) -> None:
    original = _make_data(3)
    ops, app, client, ply_path = _make_ops(tmp_path, original)
    ops._update_ply_from_splats(ply_path, original, None)
    ops.state.bbox_base_data = ops._clone_data(original)
    ops.state.bbox_base_center = np.zeros(3, dtype=np.float32)
    ops.state.bbox_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ops.state.bbox_base_dims = np.ones(3, dtype=np.float32)
    ops.state.bbox_dimensions = np.ones(3, dtype=np.float32)
    ops.state.bbox_frame_handle = SimpleNamespace(
        position=np.array([1.0, 0.0, 0.0]),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    ops.apply_attributes(client, (1.0, 0.0, 0.0), 0.2, (2, 1, 1), 1, (0, 0, 0), 1)
    attributes_data = app.sequence.cache[0]
    baseline_centers = attributes_data.centers.copy()
    baseline_covariances = attributes_data.covariances.copy()
    attributes_data.centers[0] += 10.0
    ops.apply_transform(client)
    transformed_centers = app.sequence.cache[0].centers.copy()
    ops.undo(client)

    restored_memory = app.sequence.cache[0]
    restored_ply = ply_utils.load_splat_from_ply(ply_path)
    np.testing.assert_allclose(restored_memory.centers, baseline_centers)
    np.testing.assert_allclose(restored_memory.covariances, baseline_covariances)
    np.testing.assert_allclose(restored_memory.rgbs[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(restored_ply.centers, restored_memory.centers, atol=1e-5)
    np.testing.assert_allclose(restored_ply.rgbs, restored_memory.rgbs, atol=1e-5)
    np.testing.assert_allclose(
        restored_ply.covariances,
        restored_memory.covariances,
        atol=1e-5,
    )
    assert ops.state.bbox_frame_handle is None
    assert ops.state.bbox_base_data is None

    ops.apply_transform(client)
    np.testing.assert_allclose(app.sequence.cache[0].centers, baseline_centers)

    ops.redo(client)
    redone_memory = app.sequence.cache[0]
    redone_ply = ply_utils.load_splat_from_ply(ply_path)
    np.testing.assert_allclose(redone_memory.centers, transformed_centers)
    np.testing.assert_allclose(redone_ply.centers, redone_memory.centers, atol=1e-5)
    np.testing.assert_allclose(redone_ply.rgbs, redone_memory.rgbs, atol=1e-5)
    np.testing.assert_allclose(
        redone_ply.covariances,
        redone_memory.covariances,
        atol=1e-5,
    )
