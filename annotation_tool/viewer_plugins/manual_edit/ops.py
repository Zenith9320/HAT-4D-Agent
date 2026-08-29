# plugins/manual_edit/ops.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import viser.transforms as tf
from utils.constants import COORD_ALIGN_MAT, SH_C0
from utils.ply_utils import read_ply_vertices, write_ply_vertices, SplatData


@dataclass
class EditSnapshot:
    data: Any
    header_lines: list[bytes]
    vertices: np.ndarray


class ManualEditOps:
    def __init__(self, ctx, state):
        self.ctx = ctx
        self.state = state

    def _rotation_matrix_from_wxyz(self, wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in wxyz]
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float32,
        )

    def _cov_to_log_scales_and_wxyz(self, cov_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eigvals, eigvecs = np.linalg.eigh(cov_raw.astype(np.float32))
        eigvals = np.clip(eigvals, 1e-12, None)
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 0] *= -1.0
        scales = np.sqrt(eigvals)
        log_scales = np.log(np.maximum(scales, 1e-8)).astype(np.float32)
        wxyz = tf.SO3.from_matrix(eigvecs.astype(np.float32)).wxyz.astype(np.float32)
        return log_scales, wxyz

    def _update_ply_from_splats(
        self, ply_path: Path, data, indices: np.ndarray | None = None
    ) -> None:
        if not ply_path.exists():
            raise FileNotFoundError(f"Workspace PLY is missing: {ply_path}")
        _, vertices, header_lines = read_ply_vertices(ply_path, return_header=True)
        n = vertices.shape[0]

        if indices is None:
            indices = np.arange(n, dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)
            indices = indices[(indices >= 0) & (indices < n)]

        if indices.size == 0:
            return

        centers_aligned = data.centers[indices]
        cov_aligned = data.covariances[indices]
        A = COORD_ALIGN_MAT.astype(np.float32)
        centers_raw = centers_aligned @ A
        cov_raw = np.einsum("ij,njk,kl->nil", A.T, cov_aligned, A)

        vertices["x"][indices] = centers_raw[:, 0]
        vertices["y"][indices] = centers_raw[:, 1]
        vertices["z"][indices] = centers_raw[:, 2]

        log_scales = np.zeros((indices.size, 3), dtype=np.float32)
        wxyz = np.zeros((indices.size, 4), dtype=np.float32)
        for i, cov in enumerate(cov_raw):
            log_scales[i], wxyz[i] = self._cov_to_log_scales_and_wxyz(cov)

        vertices["scale_0"][indices] = log_scales[:, 0]
        vertices["scale_1"][indices] = log_scales[:, 1]
        vertices["scale_2"][indices] = log_scales[:, 2]
        vertices["rot_0"][indices] = wxyz[:, 0]
        vertices["rot_1"][indices] = wxyz[:, 1]
        vertices["rot_2"][indices] = wxyz[:, 2]
        vertices["rot_3"][indices] = wxyz[:, 3]

        rgbs = data.rgbs[indices]
        sh = (rgbs - 0.5) / float(SH_C0)
        vertices["f_dc_0"][indices] = sh[:, 0]
        vertices["f_dc_1"][indices] = sh[:, 1]
        vertices["f_dc_2"][indices] = sh[:, 2]

        opacities = np.clip(data.opacities[indices].reshape(-1), 1e-6, 1 - 1e-6)
        logits = np.log(opacities / (1.0 - opacities)).astype(np.float32)
        vertices["opacity"][indices] = logits

        write_ply_vertices(header_lines, vertices, ply_path)

    def _write_filtered_ply(self, ply_path: Path, mask: np.ndarray) -> None:
        if not ply_path.exists():
            raise FileNotFoundError(f"Workspace PLY is missing: {ply_path}")
        _, vertices, header_lines = read_ply_vertices(ply_path, return_header=True)
        if mask.shape[0] != vertices.shape[0]:
            min_len = min(mask.shape[0], vertices.shape[0])
            mask = mask[:min_len]
            vertices = vertices[:min_len]
        vertices = vertices[mask]
        new_header = []
        for line in header_lines:
            if line.startswith(b"element vertex"):
                new_header.append(
                    f"element vertex {vertices.shape[0]}\n".encode("ascii")
                )
            else:
                new_header.append(line)
        write_ply_vertices(new_header, vertices, ply_path)

    def _notify_persistence_failure(self, client, action: str, exc: Exception) -> None:
        client.add_notification(
            f"{action} failed",
            f"Could not write the workspace PLY: {exc}",
            color="red",
            with_close_button=True,
            auto_close_seconds=10,
        )

    def _face_handle_position(self, key: str) -> np.ndarray:
        local_dir, axis = self.state.bbox_face_defs[key]
        dims = self.state.bbox_dimensions
        return local_dir * (dims[axis] * 0.5)

    def _update_face_handles(self) -> None:
        if not self.state.bbox_face_handles or self.state.bbox_dimensions is None:
            return
        self.state.bbox_updating = True
        try:
            for key, handle in self.state.bbox_face_handles.items():
                handle.position = tuple(self._face_handle_position(key).tolist())
        finally:
            self.state.bbox_updating = False

    def _apply_bbox_transform_live(self, client) -> None:
        if self.state.bbox_updating:
            return
        if self.state.bbox_frame_handle is None:
            return
        if self.state.bbox_base_data is None:
            return

        indices = self._current_selection_set()
        if not indices:
            return

        data = self._get_current_data()
        base = self.state.bbox_base_data
        idx = np.array(sorted(indices), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < base.centers.shape[0])]
        if idx.size == 0:
            return

        mode = self.state.transform_mode
        pos = np.array(self.state.bbox_frame_handle.position, dtype=np.float32)
        wxyz = np.array(self.state.bbox_frame_handle.wxyz, dtype=np.float32)
        c0 = self.state.bbox_base_center
        wxyz0 = self.state.bbox_base_wxyz
        dims0 = self.state.bbox_base_dims
        dims = self.state.bbox_dimensions
        if c0 is None or wxyz0 is None or dims0 is None or dims is None:
            return

        if mode == "Scale":
            pos = c0
            if wxyz is not None:
                self.state.bbox_base_wxyz = wxyz
                wxyz0 = wxyz
            wxyz = wxyz0

        R0 = self._rotation_matrix_from_wxyz(wxyz0)
        R = self._rotation_matrix_from_wxyz(wxyz)
        S = np.diag((dims / np.maximum(dims0, 1e-6)).astype(np.float32))

        centers0 = base.centers[idx]
        local0 = (centers0 - c0) @ R0
        local1 = local0 @ S
        centers_new = local1 @ R.T + pos

        cov0 = base.covariances[idx]
        cov_local = R0.T @ cov0 @ R0
        cov_scaled = S @ cov_local @ S
        cov_new = R @ cov_scaled @ R.T

        data.centers[idx] = centers_new
        data.covariances[idx] = cov_new
        self._set_current_data(data)

    def _update_bbox_from_face(self, key: str) -> None:
        dims = self.state.bbox_dimensions
        if dims is None:
            return
        handle = self.state.bbox_face_handles.get(key)
        if handle is None:
            return
        pos = np.array(handle.position, dtype=np.float32)
        local_dir, axis = self.state.bbox_face_defs[key]
        val = float(pos[axis])
        if key[0] == "p":
            dims[axis] = max(val * 2.0, 1e-4)
        else:
            dims[axis] = max(-val * 2.0, 1e-4)
        self.state.bbox_dimensions = dims
        if self.state.bbox_box_handle is not None:
            self.state.bbox_box_handle.dimensions = tuple(dims.tolist())

    def _attach_bbox_callbacks(self, client) -> None:
        frame = self.state.bbox_frame_handle
        if frame is None:
            return

        @frame.on_update
        def _(_event):
            self._apply_bbox_transform_live(client)

        for key, handle in self.state.bbox_face_handles.items():
            @handle.on_update
            def _evt(_event, key=key):
                if self.state.bbox_updating:
                    return
                self._update_bbox_from_face(key)
                self._update_face_handles()
                self._apply_bbox_transform_live(client)

    def _apply_transform_mode_controls(self) -> None:
        frame = self.state.bbox_frame_handle
        if frame is None:
            return

        mode = self.state.transform_mode
        if mode == "Translate":
            frame.disable_axes = False
            frame.disable_sliders = False
            frame.disable_rotations = True
        elif mode == "Rotate":
            frame.disable_axes = True
            frame.disable_sliders = True
            frame.disable_rotations = False
        elif mode == "Scale":
            frame.disable_axes = True
            frame.disable_sliders = True
            frame.disable_rotations = False

        x_on, y_on, z_on = self.state.axis_constraint
        use_constraint = any(self.state.axis_constraint)
        frame.active_axes = (
            (x_on if use_constraint else True),
            (y_on if use_constraint else True),
            (z_on if use_constraint else True),
        )

        show_faces = mode == "Scale"
        for key, handle in self.state.bbox_face_handles.items():
            axis = self.state.bbox_face_defs[key][1]
            if not use_constraint:
                handle.visible = show_faces
            else:
                handle.visible = show_faces and [x_on, y_on, z_on][axis]

    def _app(self):
        return self.ctx.app

    def _selection_matches_frame(self) -> bool:
        app = self._app()
        return (
            self.state.selection_frame == app.current_frame
            and self.state.selection_view == app.current_view
        )

    def _current_selection_set(self) -> set[int]:
        if not self._selection_matches_frame():
            return set()
        return set(int(i) for i in self.state.selected_indices)

    def _set_selection(self, indices: list[int]) -> None:
        app = self._app()
        self.state.selected_indices = list(indices)
        self.state.selection_frame = int(app.current_frame)
        self.state.selection_view = str(app.current_view)
        if self.state.selection_status_handle is not None:
            self.state.selection_status_handle.value = f"{len(indices)} selected"
        app._set_selection_info(indices, app.current_frame, app.current_view)

    def _workspace_ply_path(self) -> Path:
        app = self._app()
        return app.workspace.gs_path_for_frame(
            app.sequence.gs_path(app.current_frame)
        )

    def _capture_snapshot(
        self,
        data,
        ply_path: Path | None = None,
    ) -> EditSnapshot:
        path = ply_path or self._workspace_ply_path()
        if not path.exists():
            raise FileNotFoundError(f"Workspace PLY is missing: {path}")
        _, vertices, header_lines = read_ply_vertices(path, return_header=True)
        return EditSnapshot(
            data=self._clone_data(data),
            header_lines=list(header_lines),
            vertices=vertices.copy(),
        )

    def _restore_snapshot(
        self,
        snapshot: EditSnapshot,
        ply_path: Path | None = None,
    ) -> None:
        path = ply_path or self._workspace_ply_path()
        write_ply_vertices(snapshot.header_lines, snapshot.vertices, path)

    def _push_history(self, snapshot: EditSnapshot) -> None:
        self.state.edit_history.append(snapshot)
        self.state.redo_history.clear()

    def _clone_data(self, data):
        if isinstance(data, SplatData):
            return SplatData(
                centers=data.centers.copy(),
                rgbs=data.rgbs.copy(),
                opacities=data.opacities.copy(),
                covariances=data.covariances.copy(),
                scales=data.scales.copy(),
                rotations=data.rotations.copy(),
                sh=data.sh.copy(),
                opacity_raw=data.opacity_raw.copy(),
                centers_raw=data.centers_raw.copy(),
            )
        return type(data)(
            centers=data.centers.copy(),
            rgbs=data.rgbs.copy(),
            opacities=data.opacities.copy(),
            covariances=data.covariances.copy(),
        )

    def _get_current_data(self):
        app = self._app()
        return app.sequence.load(app.current_frame)

    def _set_current_data(self, data) -> None:
        app = self._app()
        app.sequence.cache[app.current_frame] = data
        app._update_gaussian_handle(data)
        if app.selection_overlay_enabled:
            app._render_selection_overlay(True)

    def _begin_rect_select(self, client, op: str) -> None:
        app = self._app()
        if not app.backend_url:
            client.add_notification(
                "Selection failed",
                "backend_url is not set.",
                color="red",
                auto_close_seconds=6,
            )
            return

        @client.scene.on_pointer_event(event_type="rect-select")
        def _(event):
            client.scene.remove_pointer_callback()
            self._handle_rect_select(event, client, op)

        @client.scene.on_pointer_callback_removed
        def _() -> None:
            pass

    def _handle_rect_select(self, event, client, op: str) -> None:
        app = self._app()
        if event.screen_pos is None or len(event.screen_pos) < 2:
            client.add_notification(
                "Selection failed",
                "No selection received.",
                color="red",
                auto_close_seconds=6,
            )
            return

        (x0n, y0n), (x1n, y1n) = event.screen_pos[:2]
        render_w, render_h = app._choose_capture_size(client)
        x0 = int(max(0, min(x0n, x1n) * render_w))
        x1 = int(min(1, max(x0n, x1n)) * render_w)
        y0 = int(max(0, min(y0n, y1n) * render_h))
        y1 = int(min(1, max(y0n, y1n)) * render_h)
        if x1 <= x0 or y1 <= y0:
            client.add_notification(
                "Selection skipped",
                "Selection is empty, try again.",
                color="orange",
                auto_close_seconds=6,
            )
            return

        render = app._capture_clean_render(client, render_w, render_h)
        roi = render[y0:y1, x0:x1]

        frame_token = app.sequence.frame_tokens[app.current_frame]
        gs_path = app.workspace.gs_path_for_frame(app.sequence.gs_path(app.current_frame))
        gt_path = app.workspace.image_path_for_view(frame_token, app.current_view)
        output_dir = app.workspace.output_dir_for(frame_token, app.current_view)

        camera = client.camera
        sel_dir = app.workspace.selections_dir / f"{frame_token}_{app.current_view}"
        payload = {
            "experiment_id": f"{app.workspace.root.name}_{frame_token}_{app.current_view}",
            "frame_id": int(app.current_frame),
            "camera_id": app.current_view,
            "frame_token": frame_token,
            "image_size": [render_w, render_h],
            "bbox_2d": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "paths": {
                "gs_ply": str(gs_path),
                "gt_image": str(gt_path),
                "output_dir": str(output_dir),
                "workspace_root": str(app.workspace.root),
                "capture_image": str(sel_dir / "render.png"),
                "roi_image": str(sel_dir / "roi.png"),
                "eval_gt_dir": str(app.workspace.images_root / "eval_gt"),
            },
            "camera": {
                "wxyz": camera.wxyz.tolist(),
                "position": camera.position.tolist(),
                "look_at": camera.look_at.tolist(),
                "up": camera.up_direction.tolist(),
                "fov_deg": app.fov_deg,
                "near": float(camera.near),
                "far": float(camera.far),
            },
            "depth_margin_ratio": app.depth_margin_ratio,
            "min_depth_margin": app.min_depth_margin,
        }

        app.workspace.save_selection(frame_token, app.current_view, render, roi, payload)

        try:
            base = app._backend_base_url()
            resp = app._post_json(base + "/select_roi", payload, timeout=30)
        except Exception as exc:
            client.add_notification(
                "Selection failed",
                str(exc),
                color="red",
                auto_close_seconds=8,
            )
            return

        if resp.get("status") != "success":
            client.add_notification(
                "Selection failed",
                resp.get("message", "Backend error"),
                color="red",
                auto_close_seconds=8,
            )
            return

        selection = resp.get("selection", {})
        new_indices = selection.get("gs_indices") or selection.get("indices") or []
        new_indices = [int(i) for i in new_indices]

        current = self._current_selection_set()
        new_set = set(new_indices)
        if op == "union":
            result = current | new_set
        elif op == "intersect":
            result = current & new_set
        elif op == "subtract":
            result = current - new_set
        else:
            result = new_set

        result_list = sorted(result)
        self._set_selection(result_list)
        client.add_notification(
            "Selection updated",
            f"Selected {len(result_list)} splats.",
            color="green",
            auto_close_seconds=6,
        )

    def start_box_select(self, client):
        self._begin_rect_select(client, op="replace")

    def add_selection(self, client):
        self._begin_rect_select(client, op="union")

    def intersect_selection(self, client):
        self._begin_rect_select(client, op="intersect")

    def subtract_selection(self, client):
        self._begin_rect_select(client, op="subtract")

    def clear_selection(self, client):
        self._set_selection([])
        client.add_notification(
            "Selection cleared",
            "No splats selected.",
            color="blue",
            auto_close_seconds=4,
        )

    def toggle_selection_overlay(self, client, enabled: bool):
        app = self._app()
        app.selection_overlay_enabled = bool(enabled)
        app._render_selection_overlay(app.selection_overlay_enabled)

    def delete_selected(self, client):
        indices = self._current_selection_set()
        if not indices:
            client.add_notification(
                "Delete skipped",
                "No selection to delete.",
                color="orange",
                auto_close_seconds=4,
            )
            return
        data = self._get_current_data()
        mask = np.ones((data.centers.shape[0],), dtype=bool)
        idx = np.array(sorted(indices), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < mask.shape[0])]
        mask[idx] = False
        if isinstance(data, SplatData):
            new_data = SplatData(
                centers=data.centers[mask],
                rgbs=data.rgbs[mask],
                opacities=data.opacities[mask],
                covariances=data.covariances[mask],
                scales=data.scales[mask],
                rotations=data.rotations[mask],
                sh=data.sh[mask],
                opacity_raw=data.opacity_raw[mask],
                centers_raw=data.centers_raw[mask],
            )
        else:
            new_data = type(data)(
                centers=data.centers[mask],
                rgbs=data.rgbs[mask],
                opacities=data.opacities[mask],
                covariances=data.covariances[mask],
            )
        try:
            ply_path = self._workspace_ply_path()
            snapshot = self._capture_snapshot(data, ply_path)
            self._write_filtered_ply(ply_path, mask)
        except Exception as exc:
            self._notify_persistence_failure(client, "Delete", exc)
            return
        self._push_history(snapshot)
        self._set_current_data(new_data)
        self._discard_bbox_controls()
        self._set_selection([])
        client.add_notification(
            "Delete",
            f"Deleted {len(idx)} splats.",
            color="green",
            auto_close_seconds=6,
        )

    def delete_unselected(self, client):
        data = self._get_current_data()
        current = self._current_selection_set()
        if not current:
            client.add_notification(
                "Delete skipped",
                "No selection; cannot invert delete.",
                color="orange",
                auto_close_seconds=4,
            )
            return
        mask = np.zeros((data.centers.shape[0],), dtype=bool)
        idx = np.array(sorted(current), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < mask.shape[0])]
        mask[idx] = True
        if isinstance(data, SplatData):
            new_data = SplatData(
                centers=data.centers[mask],
                rgbs=data.rgbs[mask],
                opacities=data.opacities[mask],
                covariances=data.covariances[mask],
                scales=data.scales[mask],
                rotations=data.rotations[mask],
                sh=data.sh[mask],
                opacity_raw=data.opacity_raw[mask],
                centers_raw=data.centers_raw[mask],
            )
        else:
            new_data = type(data)(
                centers=data.centers[mask],
                rgbs=data.rgbs[mask],
                opacities=data.opacities[mask],
                covariances=data.covariances[mask],
            )
        try:
            ply_path = self._workspace_ply_path()
            snapshot = self._capture_snapshot(data, ply_path)
            self._write_filtered_ply(ply_path, mask)
        except Exception as exc:
            self._notify_persistence_failure(client, "Delete", exc)
            return
        self._push_history(snapshot)
        self._set_current_data(new_data)
        self._discard_bbox_controls()
        self._set_selection([])
        client.add_notification(
            "Delete",
            f"Kept {len(idx)} splats (deleted unselected).",
            color="green",
            auto_close_seconds=6,
        )

    def apply_attributes(
        self,
        client,
        color_rgb,
        opacity,
        scale_xyz,
        uniform_scale,
        rot_xyz,
        sh_intensity,
    ):
        indices = self._current_selection_set()
        if not indices:
            client.add_notification(
                "Edit skipped",
                "No selection to edit.",
                color="orange",
                auto_close_seconds=4,
            )
            return
        data = self._get_current_data()
        idx = np.array(sorted(indices), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < data.centers.shape[0])]
        if idx.size == 0:
            return
        new_data = self._clone_data(data)

        color = np.array(color_rgb, dtype=np.float32)
        color = np.clip(color * float(sh_intensity), 0.0, 1.0)
        new_data.rgbs[idx] = color[None, :]
        new_data.opacities[idx] = float(opacity)

        sx, sy, sz = scale_xyz
        scale = np.array([sx, sy, sz], dtype=np.float32) * float(uniform_scale)
        if not np.allclose(scale, 1.0):
            S = np.diag(scale.astype(np.float32))
            cov = new_data.covariances[idx]
            new_data.covariances[idx] = S[None, :, :] @ cov @ S[None, :, :]

        rx, ry, rz = [np.deg2rad(float(v)) for v in rot_xyz]
        if not np.allclose([rx, ry, rz], 0.0):
            cx, sx_ = np.cos(rx), np.sin(rx)
            cy, sy_ = np.cos(ry), np.sin(ry)
            cz, sz_ = np.cos(rz), np.sin(rz)
            Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]], dtype=np.float32)
            Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]], dtype=np.float32)
            Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]], dtype=np.float32)
            R = Rz @ Ry @ Rx
            centers = new_data.centers[idx]
            center_mean = centers.mean(axis=0, keepdims=True)
            new_data.centers[idx] = (centers - center_mean) @ R.T + center_mean
            cov = new_data.covariances[idx]
            new_data.covariances[idx] = R[None, :, :] @ cov @ R.T[None, :, :]

        try:
            ply_path = self._workspace_ply_path()
            snapshot = self._capture_snapshot(data, ply_path)
            self._update_ply_from_splats(ply_path, new_data, idx)
        except Exception as exc:
            self._notify_persistence_failure(client, "Edit", exc)
            return
        self._push_history(snapshot)
        self._set_current_data(new_data)
        self._commit_bbox_baseline(new_data)
        client.add_notification(
            "Edit applied",
            f"Updated {len(idx)} splats.",
            color="green",
            auto_close_seconds=6,
        )

    def generate_bbox_from_selection(self, client):
        indices = self._current_selection_set()
        if not indices:
            client.add_notification(
                "BBox skipped",
                "No selection to build BBox.",
                color="orange",
                auto_close_seconds=4,
            )
            return
        data = self._get_current_data()
        idx = np.array(sorted(indices), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < data.centers.shape[0])]
        if idx.size == 0:
            return
        centers = data.centers[idx]
        mins = centers.min(axis=0)
        maxs = centers.max(axis=0)
        dims = np.maximum(maxs - mins, 1e-4)
        center = (mins + maxs) * 0.5

        if self.state.bbox_frame_handle is not None:
            self.state.bbox_frame_handle.remove()
            self.state.bbox_frame_handle = None
        if self.state.bbox_box_handle is not None:
            self.state.bbox_box_handle.remove()
            self.state.bbox_box_handle = None
        for handle in self.state.bbox_face_handles.values():
            try:
                handle.remove()
            except Exception:
                pass
        self.state.bbox_face_handles = {}
        self.state.bbox_face_defs = {}

        frame = client.scene.add_transform_controls(
            "/manual_bbox", position=tuple(center.tolist())
        )
        box = client.scene.add_box(
            "/manual_bbox/box",
            dimensions=tuple(dims.tolist()),
            wireframe=True,
            color=(255, 220, 60),
            opacity=0.9,
            position=(0.0, 0.0, 0.0),
        )
        face_defs = {
            "px": (np.array([+1, 0, 0], np.float32), 0),
            "nx": (np.array([-1, 0, 0], np.float32), 0),
            "py": (np.array([0, +1, 0], np.float32), 1),
            "ny": (np.array([0, -1, 0], np.float32), 1),
            "pz": (np.array([0, 0, +1], np.float32), 2),
            "nz": (np.array([0, 0, -1], np.float32), 2),
        }

        def _limits(axis: int):
            big = (-1000.0, 1000.0)
            lock = (0.0, 0.0)
            limits = [lock, lock, lock]
            limits[axis] = big
            return (tuple(limits[0]), tuple(limits[1]), tuple(limits[2]))

        face_handles = {}
        for key, (local_dir, axis) in face_defs.items():
            handle = client.scene.add_transform_controls(
                f"/manual_bbox/face_{key}",
                scale=0.35,
                disable_rotations=True,
                disable_sliders=True,
                active_axes=(axis == 0, axis == 1, axis == 2),
                translation_limits=_limits(axis),
                position=tuple((local_dir * (dims[axis] * 0.5)).tolist()),
            )
            face_handles[key] = handle

        self.state.bbox_frame_handle = frame
        self.state.bbox_box_handle = box
        self.state.bbox_face_handles = face_handles
        self.state.bbox_face_defs = face_defs
        self.state.bbox_initial_center = np.array(center, dtype=np.float32)
        self.state.bbox_initial_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.state.bbox_dimensions = np.array(dims, dtype=np.float32)
        self.state.bbox_base_data = self._clone_data(data)
        self.state.bbox_base_center = np.array(center, dtype=np.float32)
        self.state.bbox_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.state.bbox_base_dims = np.array(dims, dtype=np.float32)

        self._update_face_handles()
        self._attach_bbox_callbacks(client)
        self._apply_transform_mode_controls()
        self.state.bbox_visible = True
        client.add_notification(
            "BBox",
            "BBox gizmo created.",
            color="green",
            auto_close_seconds=4,
        )

    def set_bbox_visible(self, client, visible: bool):
        self.state.bbox_visible = bool(visible)
        if self.state.bbox_frame_handle is not None:
            self.state.bbox_frame_handle.visible = bool(visible)
        if self.state.bbox_box_handle is not None:
            self.state.bbox_box_handle.visible = bool(visible)
        if not visible:
            for handle in self.state.bbox_face_handles.values():
                handle.visible = False
        else:
            self._apply_transform_mode_controls()

    def apply_transform(self, client):
        indices = self._current_selection_set()
        if not indices:
            client.add_notification(
                "Transform skipped",
                "No selection to transform.",
                color="orange",
                auto_close_seconds=4,
            )
            return
        if self.state.bbox_frame_handle is None:
            client.add_notification(
                "Transform skipped",
                "BBox gizmo not created.",
                color="orange",
                auto_close_seconds=4,
            )
            return
        data = self._get_current_data()
        rollback_data = self._clone_data(data)
        geometry_baseline = self.state.bbox_base_data
        if geometry_baseline is not None:
            rollback_data.centers = geometry_baseline.centers.copy()
            rollback_data.covariances = geometry_baseline.covariances.copy()

        try:
            idx = np.array(sorted(indices), dtype=np.int64)
            ply_path = self._workspace_ply_path()
            snapshot = self._capture_snapshot(rollback_data, ply_path)
            self._update_ply_from_splats(ply_path, data, idx)
        except Exception as exc:
            self._set_current_data(rollback_data)
            self._restore_bbox_controls_to_baseline()
            self._notify_persistence_failure(client, "Transform", exc)
            return
        self._push_history(snapshot)
        self._commit_bbox_baseline(data)
        client.add_notification(
            "Transform applied",
            "Transform committed.",
            color="green",
            auto_close_seconds=6,
        )

    def update_transform_mode(self, mode: str) -> None:
        self.state.transform_mode = mode
        self._apply_transform_mode_controls()

    def update_axis_mode(self, mode: str) -> None:
        self.state.axis_mode = mode

    def update_axis_constraint(self, x: bool, y: bool, z: bool) -> None:
        self.state.axis_constraint = (bool(x), bool(y), bool(z))
        self._apply_transform_mode_controls()

    def _commit_bbox_baseline(self, data) -> None:
        frame = self.state.bbox_frame_handle
        if frame is None:
            return
        position = np.asarray(frame.position, dtype=np.float32).copy()
        wxyz = np.asarray(frame.wxyz, dtype=np.float32).copy()
        self.state.bbox_base_data = self._clone_data(data)
        self.state.bbox_base_center = position
        self.state.bbox_base_wxyz = wxyz
        if self.state.bbox_dimensions is not None:
            self.state.bbox_base_dims = np.asarray(
                self.state.bbox_dimensions,
                dtype=np.float32,
            ).copy()
        self.state.bbox_initial_center = position.copy()
        self.state.bbox_initial_wxyz = wxyz.copy()

    def _discard_bbox_controls(self) -> None:
        handles = [
            self.state.bbox_frame_handle,
            self.state.bbox_box_handle,
            *self.state.bbox_face_handles.values(),
        ]
        self.state.bbox_updating = True
        try:
            for handle in handles:
                remove = getattr(handle, "remove", None)
                if callable(remove):
                    try:
                        remove()
                    except Exception:
                        pass
            self.state.bbox_frame_handle = None
            self.state.bbox_box_handle = None
            self.state.bbox_face_handles = {}
            self.state.bbox_face_defs = {}
            self.state.bbox_initial_center = None
            self.state.bbox_initial_wxyz = None
            self.state.bbox_dimensions = None
            self.state.bbox_base_data = None
            self.state.bbox_base_center = None
            self.state.bbox_base_wxyz = None
            self.state.bbox_base_dims = None
            self.state.bbox_visible = False
        finally:
            self.state.bbox_updating = False

    def _restore_bbox_controls_to_baseline(self) -> None:
        frame = self.state.bbox_frame_handle
        center = self.state.bbox_base_center
        wxyz = self.state.bbox_base_wxyz
        dims = self.state.bbox_base_dims
        self.state.bbox_updating = True
        try:
            if frame is not None and center is not None:
                frame.position = tuple(np.asarray(center).tolist())
            if frame is not None and wxyz is not None:
                frame.wxyz = tuple(np.asarray(wxyz).tolist())
            if dims is not None:
                self.state.bbox_dimensions = np.asarray(dims, dtype=np.float32).copy()
                if self.state.bbox_box_handle is not None:
                    self.state.bbox_box_handle.dimensions = tuple(
                        self.state.bbox_dimensions.tolist()
                    )
            self.state.bbox_initial_center = (
                None if center is None else np.asarray(center, dtype=np.float32).copy()
            )
            self.state.bbox_initial_wxyz = (
                None if wxyz is None else np.asarray(wxyz, dtype=np.float32).copy()
            )
        finally:
            self.state.bbox_updating = False
        self._update_face_handles()

    def undo(self, client):
        if not self.state.edit_history:
            client.add_notification(
                "Undo",
                "Nothing to undo.",
                color="orange",
                auto_close_seconds=4,
            )
            return
        data = self._get_current_data()
        last = self.state.edit_history[-1]
        try:
            ply_path = self._workspace_ply_path()
            current = self._capture_snapshot(data, ply_path)
            self._restore_snapshot(last, ply_path)
        except Exception as exc:
            self._notify_persistence_failure(client, "Undo", exc)
            return
        self.state.redo_history.append(current)
        self.state.edit_history.pop()
        self._set_current_data(self._clone_data(last.data))
        self._discard_bbox_controls()
        client.add_notification(
            "Undo",
            "Reverted last edit.",
            color="green",
            auto_close_seconds=4,
        )

    def redo(self, client):
        if not self.state.redo_history:
            client.add_notification(
                "Redo",
                "Nothing to redo.",
                color="orange",
                auto_close_seconds=4,
            )
            return
        data = self._get_current_data()
        nxt = self.state.redo_history[-1]
        try:
            ply_path = self._workspace_ply_path()
            current = self._capture_snapshot(data, ply_path)
            self._restore_snapshot(nxt, ply_path)
        except Exception as exc:
            self._notify_persistence_failure(client, "Redo", exc)
            return
        self.state.edit_history.append(current)
        self.state.redo_history.pop()
        self._set_current_data(self._clone_data(nxt.data))
        self._discard_bbox_controls()
        client.add_notification(
            "Redo",
            "Re-applied edit.",
            color="green",
            auto_close_seconds=4,
        )

    def save_snapshot(self, client):
        data = self._get_current_data()
        try:
            snapshot = self._capture_snapshot(data)
        except Exception as exc:
            self._notify_persistence_failure(client, "Snapshot", exc)
            return
        self._push_history(snapshot)
        client.add_notification(
            "Snapshot",
            "Snapshot saved.",
            color="blue",
            auto_close_seconds=4,
        )
