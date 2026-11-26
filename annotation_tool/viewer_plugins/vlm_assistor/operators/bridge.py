from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from utils.camera_utils import eye_wxyz_for_view, eye_wxyz_from_angles
from utils.geometry_utils import transform_gaussians

CAMERA_ZNEAR = 0.05
CAMERA_ZFAR = 150.0


class _OperatorRegistry:
    def __init__(self, bridge: "Sam3DOperatorBridge") -> None:
        self._bridge = bridge

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        fn = getattr(self._bridge, f"_op_{str(name).replace('.', '_')}", None)
        if callable(fn):
            try:
                out = fn(**kwargs)
                if isinstance(out, dict):
                    return out
                return {"ok": True, "result": out}
            except Exception as exc:
                return {"ok": False, "reason": f"operator_exception:{type(exc).__name__}:{exc}"}
        return {"ok": False, "reason": f"operator_not_implemented:{name}"}


class Sam3DOperatorBridge:
    """Bridge from VLM agents to sam3d plugin ops.

    This keeps a stable operator calling surface even if sam3d ops evolve.
    """

    def __init__(self, ctx, state) -> None:
        self.ctx = ctx
        self.state = state
        self.registry = _OperatorRegistry(self)

    @staticmethod
    def _normalize_points(raw: Any) -> list[tuple[int, int]]:
        if not isinstance(raw, list):
            return []
        out: list[tuple[int, int]] = []
        for it in raw:
            x = None
            y = None
            if isinstance(it, dict):
                x = it.get("x", it.get("u"))
                y = it.get("y", it.get("v"))
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                x, y = it[0], it[1]
            if x is None or y is None:
                continue
            try:
                out.append((int(round(float(x))), int(round(float(y)))))
            except Exception:
                continue
        return out

    def _sam3d_ops(self):
        return getattr(self.ctx, "sam3d_ops", None)

    def _sam3d_state(self):
        return getattr(self.ctx, "sam3d_state", None)

    def _resolve_object_idx(self, object_idx: int | None) -> int:
        st = self._sam3d_state()
        if st is None:
            return -1
        with st.lock:
            n = len(getattr(st, "objects", []))
            cur = int(getattr(st, "current_object_idx", 0))
        idx = cur if object_idx is None else int(object_idx)
        if idx < 0 or idx >= n:
            return -1
        return idx

    @staticmethod
    def _clone_splat_data(splat: Any, clone_fn: Any = None) -> Any:
        if splat is None:
            return None
        if callable(clone_fn):
            try:
                return clone_fn(splat)
            except Exception:
                pass
        cls = splat.__class__
        try:
            return cls(
                centers=np.array(getattr(splat, "centers"), copy=True),
                rgbs=np.array(getattr(splat, "rgbs"), copy=True),
                opacities=np.array(getattr(splat, "opacities"), copy=True),
                covariances=np.array(getattr(splat, "covariances"), copy=True),
                scales=np.array(getattr(splat, "scales"), copy=True),
                rotations=np.array(getattr(splat, "rotations"), copy=True),
                sh=np.array(getattr(splat, "sh"), copy=True),
                opacity_raw=np.array(getattr(splat, "opacity_raw"), copy=True),
                centers_raw=np.array(getattr(splat, "centers_raw"), copy=True),
            )
        except Exception:
            return splat

    @staticmethod
    def _aabb_of_centers(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(centers, dtype=np.float32)
        return c.min(axis=0), c.max(axis=0)

    @staticmethod
    def _rotation_matrix_from_wxyz(wxyz: np.ndarray) -> np.ndarray:
        q = np.asarray(wxyz, dtype=np.float32).reshape(-1)
        if q.size < 4:
            return np.eye(3, dtype=np.float32)
        w, x, y, z = [float(v) for v in q[:4]]
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _aabb_overlap(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))

    @staticmethod
    def _ensure_rgb_u8(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        if a.ndim != 3:
            raise ValueError(f"invalid image shape: {a.shape}")
        if a.shape[2] >= 4:
            a = a[:, :, :3]
        if a.dtype != np.uint8:
            if np.issubdtype(a.dtype, np.floating):
                a = np.clip(a, 0.0, 1.0) * 255.0
            a = np.clip(a, 0, 255).astype(np.uint8)
        return a

    @staticmethod
    def _camera_target_origin() -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _camera_planes_from_app(app: Any) -> tuple[float, float]:
        try:
            near = float(getattr(app, "camera_znear", CAMERA_ZNEAR))
        except Exception:
            near = float(CAMERA_ZNEAR)
        try:
            far = float(getattr(app, "camera_zfar", CAMERA_ZFAR))
        except Exception:
            far = float(CAMERA_ZFAR)
        near = float(max(1e-4, near))
        far = float(max(near + 1e-3, far))
        return near, far

    def _camera_pose_for_view_origin(self, app: Any, view_name: str, distance: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        view = str(view_name or "").strip().lower()
        up_axis = str(getattr(app, "up_axis", "y") or "y")
        presets = dict(getattr(app, "view_presets", {}) or {})
        origin = self._camera_target_origin()
        if view in presets:
            return eye_wxyz_for_view(view, float(distance), origin, presets, up_axis)
        az_map = {
            "front": (0.0, 0.0),
            "back": (180.0, 0.0),
            "left": (90.0, 0.0),
            "right": (-90.0, 0.0),
            "up": (0.0, 70.0),
            "down": (0.0, -70.0),
            "input": (0.0, 0.0),
        }
        az, el = az_map.get(view, (0.0, 0.0))
        return eye_wxyz_from_angles(float(az), float(el), float(distance), origin, up_axis)

    def _set_camera_pose(
        self,
        app: Any,
        client: Any,
        eye: np.ndarray,
        wxyz: np.ndarray,
        *,
        look_at: np.ndarray | None = None,
        up: np.ndarray | None = None,
    ) -> None:
        set_pose = getattr(app, "_set_client_camera_pose", None)
        eye_f = np.asarray(eye, dtype=np.float32)
        wxyz_f = np.asarray(wxyz, dtype=np.float32)
        look_at_f = None if look_at is None else np.asarray(look_at, dtype=np.float32)
        up_f = None if up is None else np.asarray(up, dtype=np.float32)
        znear, zfar = self._camera_planes_from_app(app)
        if callable(set_pose):
            try:
                set_pose(client, eye_f, wxyz_f, look_at=look_at_f, up=up_f)
                try:
                    with client.atomic():
                        client.camera.near = float(znear)
                        client.camera.far = float(zfar)
                except Exception:
                    pass
                return
            except Exception:
                pass
        with client.atomic():
            client.camera.position = eye_f
            client.camera.wxyz = wxyz_f
            if look_at_f is not None:
                client.camera.look_at = look_at_f
            if up_f is not None:
                client.camera.up_direction = up_f
            try:
                client.camera.fov = np.deg2rad(float(getattr(app, "fov_deg", 60.0)))
            except Exception:
                pass
            try:
                client.camera.near = float(znear)
                client.camera.far = float(zfar)
            except Exception:
                pass

    @staticmethod
    def _collect_visible_handles(*groups: Any) -> list[tuple[Any, bool]]:
        out: list[tuple[Any, bool]] = []
        seen: set[int] = set()
        for group in groups:
            if group is None:
                continue
            if isinstance(group, (list, tuple)):
                items = list(group)
            else:
                items = [group]
            for h in items:
                if h is None:
                    continue
                hid = id(h)
                if hid in seen:
                    continue
                seen.add(hid)
                try:
                    vis = bool(getattr(h, "visible", True))
                except Exception:
                    vis = True
                out.append((h, vis))
        return out

    @staticmethod
    def _set_handles_visible(handles: list[tuple[Any, bool]], visible: bool) -> None:
        for h, _ in handles:
            try:
                h.visible = bool(visible)
            except Exception:
                pass

    @staticmethod
    def _restore_handles_visible(handles: list[tuple[Any, bool]]) -> None:
        for h, vis in handles:
            try:
                h.visible = bool(vis)
            except Exception:
                pass

    def _set_sam3d_frustum_opacity(self, opacity: float) -> bool:
        op = float(np.clip(float(opacity), 0.0, 1.0))
        ops = self._sam3d_ops()
        set_fn = getattr(ops, "set_frustum_opacity", None)
        if callable(set_fn):
            try:
                set_fn(op)
                return True
            except Exception:
                pass
        st = self._sam3d_state()
        if st is None:
            return False
        try:
            with st.lock:
                setattr(st, "frustum_opacity", float(op))
            refresh_fn = getattr(ops, "_refresh_frustum_image", None)
            if callable(refresh_fn):
                try:
                    refresh_fn()
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def _refresh_main_scene_splats(self, app: Any, *, use_full: bool) -> None:
        load_fn = getattr(getattr(app, "sequence", None), "load", None)
        upd_fn = getattr(app, "_update_gaussian_handle", None)
        if not callable(load_fn) or not callable(upd_fn):
            return
        frame_idx = int(getattr(app, "current_frame", 0))
        try:
            data_full = load_fn(frame_idx)
        except Exception:
            return
        if not bool(use_full):
            get_disp_fn = getattr(app, "_get_display_splats", None)
            if callable(get_disp_fn):
                try:
                    data_show = get_disp_fn(frame_idx, data_full)
                except Exception:
                    data_show = data_full
            else:
                data_show = data_full
        else:
            data_show = data_full
        try:
            upd_fn(data_show)
        except Exception:
            pass

    def _refresh_sam3d_object_handles(self, st: Any, *, max_points: int) -> None:
        ops = self._sam3d_ops()
        ensure_obj_fn = getattr(ops, "_ensure_object_handle", None)
        if st is None or not callable(ensure_obj_fn):
            return
        try:
            with st.lock:
                st.display_max_points = int(max_points)
                objs = list(getattr(st, "objects", []) or [])
            for i, obj in enumerate(objs):
                if getattr(obj, "splat", None) is None:
                    continue
                try:
                    ensure_obj_fn(int(i))
                except Exception:
                    pass
        except Exception:
            return

    def _collect_headless_render_components(
        self,
        app: Any,
        st: Any,
        *,
        include_background: bool,
        include_objects: bool,
    ) -> dict[str, np.ndarray] | None:
        centers_list: list[np.ndarray] = []
        rotations_list: list[np.ndarray] = []
        scales_list: list[np.ndarray] = []
        opacities_list: list[np.ndarray] = []
        rgbs_list: list[np.ndarray] = []

        if bool(include_background):
            load_fn = getattr(getattr(app, "sequence", None), "load", None)
            if callable(load_fn):
                try:
                    frame_idx = int(getattr(app, "current_frame", 0))
                    data_full = load_fn(frame_idx)
                    centers = np.asarray(getattr(data_full, "centers", None), dtype=np.float32)
                    rotations = np.asarray(getattr(data_full, "rotations", None), dtype=np.float32)
                    scales = np.asarray(getattr(data_full, "scales", None), dtype=np.float32)
                    opacities = np.asarray(getattr(data_full, "opacities", None), dtype=np.float32).reshape(-1)
                    rgbs = np.asarray(getattr(data_full, "rgbs", None), dtype=np.float32)
                    if (
                        centers.ndim == 2
                        and centers.shape[0] > 0
                        and rotations.ndim == 2
                        and scales.ndim == 2
                        and rgbs.ndim == 2
                    ):
                        n = int(centers.shape[0])
                        centers_list.append(centers[:n])
                        rotations_list.append(rotations[:n])
                        scales_list.append(scales[:n])
                        opacities_list.append(opacities[:n])
                        rgbs_list.append(rgbs[:n])
                except Exception:
                    pass

        if bool(include_objects) and st is not None:
            with st.lock:
                fr_wxyz = (
                    None
                    if getattr(st, "frustum_wxyz", None) is None
                    else np.asarray(st.frustum_wxyz, dtype=np.float32).copy()
                )
                fr_pos = (
                    None
                    if getattr(st, "frustum_position", None) is None
                    else np.asarray(st.frustum_position, dtype=np.float32).copy()
                )
                objs = list(getattr(st, "objects", []) or [])
            R_wc = None
            if fr_wxyz is not None and fr_pos is not None:
                R_wc = self._rotation_matrix_from_wxyz(np.asarray(fr_wxyz, dtype=np.float32))

            for obj in objs:
                splat = getattr(obj, "splat", None)
                if splat is None:
                    continue
                centers = np.asarray(getattr(splat, "centers", None), dtype=np.float32)
                rotations = np.asarray(getattr(splat, "rotations", None), dtype=np.float32)
                scales = np.asarray(getattr(splat, "scales", None), dtype=np.float32)
                opacities = np.asarray(getattr(splat, "opacities", None), dtype=np.float32).reshape(-1)
                rgbs = np.asarray(getattr(splat, "rgbs", None), dtype=np.float32)
                if (
                    centers.ndim != 2
                    or centers.shape[0] <= 0
                    or rotations.ndim != 2
                    or scales.ndim != 2
                    or rgbs.ndim != 2
                ):
                    continue

                if R_wc is not None and fr_pos is not None:
                    try:
                        centers_w, rotations_w, scales_w = transform_gaussians(
                            centers=centers,
                            rotations=rotations,
                            scales=scales,
                            A=np.asarray(R_wc, dtype=np.float32),
                            return_cov=False,
                        )
                        centers = np.asarray(centers_w, dtype=np.float32) + np.asarray(fr_pos, dtype=np.float32).reshape(1, 3)
                        rotations = np.asarray(rotations_w, dtype=np.float32)
                        scales = np.asarray(scales_w, dtype=np.float32)
                    except Exception:
                        pass

                n = int(centers.shape[0])
                centers_list.append(centers[:n])
                rotations_list.append(rotations[:n])
                scales_list.append(scales[:n])
                opacities_list.append(opacities[:n])
                rgbs_list.append(rgbs[:n])

        if not centers_list:
            return None

        return {
            "centers": np.concatenate(centers_list, axis=0).astype(np.float32, copy=False),
            "rotations": np.concatenate(rotations_list, axis=0).astype(np.float32, copy=False),
            "scales": np.concatenate(scales_list, axis=0).astype(np.float32, copy=False),
            "opacities": np.concatenate(opacities_list, axis=0).astype(np.float32, copy=False),
            "rgbs": np.concatenate(rgbs_list, axis=0).astype(np.float32, copy=False),
        }

    def _headless_render_scene(
        self,
        app: Any,
        client: Any,
        *,
        width: int,
        height: int,
        include_background: bool,
        include_objects: bool,
        return_alpha: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        st = self._sam3d_state()
        comp = self._collect_headless_render_components(
            app,
            st,
            include_background=bool(include_background),
            include_objects=bool(include_objects),
        )

        h = int(max(64, int(height)))
        w = int(max(64, int(width)))
        if comp is None:
            blank = np.full((h, w, 3), 255, dtype=np.uint8)
            if bool(return_alpha):
                return blank, np.zeros((h, w), dtype=np.float32)
            return blank

        try:
            import torch
            from gsplat.rendering import rasterization
        except Exception:
            return None

        try:
            cam_pos = np.asarray(client.camera.position, dtype=np.float32).reshape(3)
            cam_wxyz = np.asarray(client.camera.wxyz, dtype=np.float32).reshape(4)
            cam_fov = float(getattr(client.camera, "fov", np.deg2rad(float(getattr(app, "fov_deg", 49.1)))))
        except Exception:
            return None
        cam_near, cam_far = self._camera_planes_from_app(app)
        if not np.isfinite(cam_fov) or cam_fov <= 0.0:
            cam_fov = float(np.deg2rad(float(getattr(app, "fov_deg", 49.1))))

        R_cw = self._rotation_matrix_from_wxyz(np.asarray(cam_wxyz, dtype=np.float32))
        T_cw = np.eye(4, dtype=np.float32)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = np.asarray(cam_pos, dtype=np.float32)
        T_wc = np.linalg.inv(T_cw).astype(np.float32)

        tan_half_fov = float(np.tan(float(cam_fov) * 0.5))
        if not np.isfinite(tan_half_fov) or tan_half_fov <= 1e-6:
            tan_half_fov = float(np.tan(np.deg2rad(float(getattr(app, "fov_deg", 49.1))) * 0.5))
        focal = float(w) / (2.0 * tan_half_fov)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        means = torch.from_numpy(comp["centers"]).to(device)
        quats = torch.from_numpy(comp["rotations"]).to(device)
        scales = torch.from_numpy(comp["scales"]).to(device)
        opacities = torch.from_numpy(comp["opacities"]).to(device)
        colors = torch.from_numpy(comp["rgbs"]).to(device)
        viewmats = torch.from_numpy(T_wc[None]).float().to(device)
        Ks = torch.tensor(
            [[[focal, 0.0, float(w) * 0.5], [0.0, focal, float(h) * 0.5], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=device,
        )
        bg = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
        rendered_rgb, rendered_alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=int(w),
            height=int(h),
            near_plane=float(max(1e-4, cam_near)),
            far_plane=float(max(cam_near + 1e-3, cam_far)),
            packed=False,
            backgrounds=bg,
            render_mode="RGB",
        )
        if rendered_rgb is None or len(rendered_rgb) <= 0:
            return None
        rgb = rendered_rgb[0].detach().float().clamp(0.0, 1.0).cpu().numpy()
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        if not bool(return_alpha):
            return rgb_u8
        alpha = np.zeros((h, w), dtype=np.float32)
        if rendered_alpha is not None and len(rendered_alpha) > 0:
            try:
                alpha = rendered_alpha[0].detach().float().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            except Exception:
                alpha = np.zeros((h, w), dtype=np.float32)
        return rgb_u8, alpha

    def _capture_rgb(
        self,
        app: Any,
        client: Any,
        *,
        width: int,
        height: int,
        include_background: bool,
        include_objects: bool,
    ) -> np.ndarray:
        # Prefer deterministic gsplat headless rendering for VLM-facing captures.
        headless = self._headless_render_scene(
            app,
            client,
            width=int(width),
            height=int(height),
            include_background=bool(include_background),
            include_objects=bool(include_objects),
            return_alpha=False,
        )
        if isinstance(headless, np.ndarray):
            return self._ensure_rgb_u8(headless)

        if bool(getattr(client, "is_headless", False)):
            raise RuntimeError(
                "headless_render_unavailable: install gsplat in the frontend environment "
                "and ensure its CUDA extension can load"
            )

        last_exc: Exception | None = None
        capture_fn = getattr(app, "_capture_clean_render", None)
        if callable(capture_fn):
            try:
                arr = capture_fn(client, int(width), int(height))
                return self._ensure_rgb_u8(np.asarray(arr))
            except Exception as exc:
                last_exc = exc

        get_render_fn = getattr(client, "get_render", None)
        if callable(get_render_fn):
            try:
                arr = get_render_fn(height=int(height), width=int(width))
                return self._ensure_rgb_u8(np.asarray(arr))
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("render_unavailable")

    def _capture_input_overlay_rgb(
        self,
        app: Any,
        client: Any,
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        try:
            return self._capture_rgb(
                app,
                client,
                width=int(width),
                height=int(height),
                include_background=True,
                include_objects=True,
            )
        except Exception:
            pass

        try:
            frame_token = str(app.sequence.frame_tokens[app.current_frame])
            input_path = Path(app.workspace.image_path_for_view(frame_token, "input"))
            input_rgb = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
            if input_rgb.shape[0] != int(height) or input_rgb.shape[1] != int(width):
                input_rgb = np.asarray(
                    Image.fromarray(input_rgb).resize((int(width), int(height)), resample=Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
        except Exception:
            input_rgb = np.full((int(height), int(width), 3), 255, dtype=np.uint8)

        rendered = self._headless_render_scene(
            app,
            client,
            width=int(width),
            height=int(height),
            include_background=False,
            include_objects=True,
            return_alpha=True,
        )
        if not isinstance(rendered, tuple):
            return self._ensure_rgb_u8(input_rgb)
        obj_rgb, alpha = rendered
        a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
        if a.ndim == 2:
            a = a[..., None]
        out = (a * np.asarray(obj_rgb, dtype=np.float32) + (1.0 - a) * np.asarray(input_rgb, dtype=np.float32)).astype(np.uint8)
        return self._ensure_rgb_u8(out)

    def _object_center_world(self, object_idx: int) -> dict[str, Any] | None:
        st = self._sam3d_state()
        if st is None:
            return None
        idx = self._resolve_object_idx(int(object_idx))
        if idx < 0:
            return None
        with st.lock:
            obj = st.objects[idx]
            splat = getattr(obj, "splat", None)
            fr_wxyz = None if getattr(st, "frustum_wxyz", None) is None else np.asarray(st.frustum_wxyz, dtype=np.float32).copy()
            fr_pos = None if getattr(st, "frustum_position", None) is None else np.asarray(st.frustum_position, dtype=np.float32).copy()
        if splat is None:
            return None
        centers_local = np.asarray(getattr(splat, "centers", None), dtype=np.float32)
        if centers_local.ndim != 2 or centers_local.shape[0] <= 0:
            return None
        center_local = centers_local.mean(axis=0).astype(np.float32)
        if fr_wxyz is not None and fr_pos is not None:
            R_wc = self._rotation_matrix_from_wxyz(np.asarray(fr_wxyz, dtype=np.float32))
            center_world = (center_local @ R_wc.T + fr_pos).astype(np.float32)
        else:
            center_world = center_local.copy()
        return {
            "center_local_xyz": center_local.astype(float).tolist(),
            "center_world_xyz": center_world.astype(float).tolist(),
        }

    def _op_sam3d_generate(self, client=None, **_kwargs: Any) -> dict[str, Any]:
        ops = self._sam3d_ops()
        if ops is None:
            return {"ok": False, "reason": "sam3d_ops_unavailable"}
        fn = getattr(ops, "generate_current_object_async", None)
        if not callable(fn):
            return {"ok": False, "reason": "sam3d_generate_entry_missing"}
        fn(client)
        return {"ok": True, "mode": "async_triggered"}

    def _op_session_ensure_sam3d(self, client=None, **_kwargs: Any) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        active = bool(getattr(st, "gen_active", False))
        if not active:
            fn = getattr(ops, "begin", None)
            if not callable(fn):
                return {"ok": False, "reason": "sam3d_begin_missing"}
            fn(client)
        return {
            "ok": True,
            "gen_active": bool(getattr(st, "gen_active", False)),
            "current_object_idx": int(getattr(st, "current_object_idx", 0)),
            "n_objects": int(len(getattr(st, "objects", []))),
        }

    def _op_session_end(self, client=None, **_kwargs: Any) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        fn = getattr(ops, "end", None)
        if not callable(fn):
            return {"ok": False, "reason": "sam3d_end_missing"}
        fn(client)
        return {"ok": True, "gen_active": bool(getattr(st, "gen_active", False))}

    def _op_object_new(self, client=None, **_kwargs: Any) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        fn = getattr(ops, "new_object", None)
        if not callable(fn):
            return {"ok": False, "reason": "new_object_missing"}
        fn(client)
        return {
            "ok": True,
            "current_object_idx": int(getattr(st, "current_object_idx", 0)),
            "n_objects": int(len(getattr(st, "objects", []))),
        }

    def _op_sam_segment_with_points(
        self,
        client=None,
        points: list[Any] | None = None,
        negative_points: list[Any] | None = None,
        clear_existing: bool = True,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        run_sam = getattr(ops, "_run_sam_and_update_mask", None)
        cur_obj_fn = getattr(ops, "_cur_object", None)
        if not callable(run_sam) or not callable(cur_obj_fn):
            return {"ok": False, "reason": "sam_internal_methods_missing"}

        pos = self._normalize_points(points or [])
        neg = self._normalize_points(negative_points or [])
        if not pos:
            return {"ok": False, "reason": "empty_positive_points"}

        with st.lock:
            obj = cur_obj_fn()
            if bool(clear_existing):
                obj.points_xy = []
                obj.point_labels = []
                obj.logits = None
                obj.logits_key = None
                obj.mask = None
                obj.mask_path = None
            for x, y in pos:
                obj.points_xy.append((int(x), int(y)))
                obj.point_labels.append(1)
            for x, y in neg:
                obj.points_xy.append((int(x), int(y)))
                obj.point_labels.append(0)

        run_sam(client)

        with st.lock:
            obj = cur_obj_fn()
            mask = obj.mask
            n_total = len(obj.points_xy)
            n_pos = int(sum(1 for x in obj.point_labels if int(x) == 1))
            n_neg = int(sum(1 for x in obj.point_labels if int(x) == 0))
        if mask is None:
            return {
                "ok": False,
                "reason": "sam_mask_missing",
                "points_total": n_total,
                "points_positive": n_pos,
                "points_negative": n_neg,
            }
        area = int(mask.sum())
        total = int(mask.shape[0] * mask.shape[1]) if hasattr(mask, "shape") else 0
        ratio = float(area / max(1, total))

        # Auto-refine when mask is abnormally large (common failure mode in backend SAM path).
        refined = False
        if hasattr(mask, "shape") and ratio > 0.88:
            h = int(mask.shape[0])
            w = int(mask.shape[1])
            border = max(2, int(round(min(h, w) * 0.02)))
            corrective = [
                (border, border),
                (w - 1 - border, border),
                (border, h - 1 - border),
                (w - 1 - border, h - 1 - border),
                (w // 2, border),
                (w // 2, h - 1 - border),
                (border, h // 2),
                (w - 1 - border, h // 2),
            ]
            neg2 = list(neg)
            seen = set(neg2)
            for p in corrective:
                if p in seen:
                    continue
                seen.add(p)
                neg2.append(p)
            with st.lock:
                obj = cur_obj_fn()
                obj.points_xy = []
                obj.point_labels = []
                obj.logits = None
                obj.logits_key = None
                obj.mask = None
                obj.mask_path = None
                for x, y in pos:
                    obj.points_xy.append((int(x), int(y)))
                    obj.point_labels.append(1)
                for x, y in neg2:
                    obj.points_xy.append((int(x), int(y)))
                    obj.point_labels.append(0)
            run_sam(client)
            refined = True
            with st.lock:
                obj = cur_obj_fn()
                mask = obj.mask
                n_total = len(obj.points_xy)
                n_pos = int(sum(1 for x in obj.point_labels if int(x) == 1))
                n_neg = int(sum(1 for x in obj.point_labels if int(x) == 0))
            if mask is None:
                return {
                    "ok": False,
                    "reason": "sam_mask_missing_after_refine",
                    "points_total": n_total,
                    "points_positive": n_pos,
                    "points_negative": n_neg,
                }
            area = int(mask.sum())
            total = int(mask.shape[0] * mask.shape[1]) if hasattr(mask, "shape") else 0
            ratio = float(area / max(1, total))

        return {
            "ok": True,
            "points_total": n_total,
            "points_positive": n_pos,
            "points_negative": n_neg,
            "mask_area": area,
            "mask_ratio": ratio,
            "auto_refined": bool(refined),
        }

    def _op_sam_evaluate_mask(
        self,
        client=None,
        positive_points: list[Any] | None = None,
        negative_points: list[Any] | None = None,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.85,
        min_positive_coverage: float = 0.67,
        max_negative_coverage: float = 0.25,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        _ = client
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        cur_obj_fn = getattr(ops, "_cur_object", None)
        if not callable(cur_obj_fn):
            return {"ok": False, "reason": "sam_internal_methods_missing"}

        pos = self._normalize_points(positive_points or [])
        neg = self._normalize_points(negative_points or [])
        with st.lock:
            obj = cur_obj_fn()
            mask = obj.mask
        if mask is None or not hasattr(mask, "shape"):
            return {"ok": False, "reason": "mask_missing"}

        h = int(mask.shape[0])
        w = int(mask.shape[1])
        area = int(mask.sum())
        total = max(1, int(h * w))
        area_ratio = float(area / total)

        def _inside(pt_list: list[tuple[int, int]], *, default_if_empty: float) -> float:
            if not pt_list:
                return float(default_if_empty)
            hit = 0
            valid = 0
            for x, y in pt_list:
                xi = max(0, min(w - 1, int(x)))
                yi = max(0, min(h - 1, int(y)))
                valid += 1
                try:
                    if bool(mask[yi, xi]):
                        hit += 1
                except Exception:
                    continue
            if valid <= 0:
                return 0.0
            return float(hit) / float(valid)

        pos_cov = _inside(pos, default_if_empty=1.0)
        neg_cov = _inside(neg, default_if_empty=0.0)
        ok = (
            area_ratio >= float(min_area_ratio)
            and area_ratio <= float(max_area_ratio)
            and pos_cov >= float(min_positive_coverage)
            and neg_cov <= float(max_negative_coverage)
        )
        return {
            "ok": bool(ok),
            "mask_area": area,
            "mask_ratio": area_ratio,
            "positive_coverage": pos_cov,
            "negative_coverage": neg_cov,
            "thresholds": {
                "min_area_ratio": float(min_area_ratio),
                "max_area_ratio": float(max_area_ratio),
                "min_positive_coverage": float(min_positive_coverage),
                "max_negative_coverage": float(max_negative_coverage),
            },
        }

    def _op_sam3d_generate_and_wait(
        self,
        client=None,
        timeout: float = 1800.0,
        poll_interval: float = 0.5,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        fn = getattr(ops, "generate_current_object_async", None)
        if not callable(fn):
            return {"ok": False, "reason": "sam3d_generate_entry_missing"}

        with st.lock:
            idx = int(getattr(st, "current_object_idx", 0))
            has_mask = False
            if 0 <= idx < len(st.objects):
                has_mask = st.objects[idx].mask is not None
        if not has_mask:
            return {"ok": False, "reason": "mask_missing_before_generate", "object_idx": idx}

        fn(client)
        t0 = time.time()
        deadline = t0 + max(1.0, float(timeout))
        poll_s = max(0.05, float(poll_interval))

        while time.time() <= deadline:
            with st.lock:
                obj_ok = 0 <= idx < len(st.objects)
                if obj_ok and st.objects[idx].splat is not None:
                    return {
                        "ok": True,
                        "object_idx": idx,
                        "wait_sec": float(time.time() - t0),
                    }
                gen_status = ""
                try:
                    handle = getattr(st, "gen_status_handle", None)
                    if handle is not None:
                        gen_status = str(getattr(handle, "value", "")).strip().lower()
                except Exception:
                    gen_status = ""
            if gen_status.startswith("failed"):
                return {
                    "ok": False,
                    "reason": "sam3d_generate_failed",
                    "object_idx": idx,
                    "wait_sec": float(time.time() - t0),
                }
            if gen_status.startswith("done"):
                return {
                    "ok": False,
                    "reason": "sam3d_generate_done_without_splat",
                    "object_idx": idx,
                    "wait_sec": float(time.time() - t0),
                }
            time.sleep(poll_s)

        return {
            "ok": False,
            "reason": "sam3d_generate_timeout",
            "object_idx": idx,
            "timeout": float(timeout),
        }

    def _op_session_add_all_objects_and_exit(self, client=None, **_kwargs: Any) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        fn = getattr(ops, "add_object_to_session_ply", None)
        if not callable(fn):
            return {"ok": False, "reason": "add_object_to_session_missing"}

        with st.lock:
            n_with_splat = int(sum(1 for x in st.objects if getattr(x, "splat", None) is not None))
        if n_with_splat <= 0:
            return {"ok": False, "reason": "no_generated_objects"}

        fn(client)
        return {
            "ok": True,
            "objects_written": n_with_splat,
            "gen_active": bool(getattr(st, "gen_active", False)),
        }

    def _op_object_list(self, **_kwargs: Any) -> dict[str, Any]:
        st = self._sam3d_state()
        if st is None:
            return {"ok": False, "reason": "sam3d_state_unavailable"}
        out: list[dict[str, Any]] = []
        with st.lock:
            objs = list(getattr(st, "objects", []) or [])
            cur = int(getattr(st, "current_object_idx", 0))
        for i, obj in enumerate(objs):
            splat = getattr(obj, "splat", None)
            has_splat = splat is not None
            center = None
            if has_splat:
                try:
                    center = np.asarray(splat.centers, dtype=np.float32).mean(axis=0).reshape(-1).astype(float).tolist()
                except Exception:
                    center = None
            out.append(
                {
                    "object_idx": int(i),
                    "has_splat": bool(has_splat),
                    "has_mask": getattr(obj, "mask", None) is not None,
                    "center_xyz": center,
                    "is_current": bool(i == cur),
                }
            )
        return {"ok": True, "current_object_idx": int(cur), "objects": out}

    def _op_object_select(self, object_idx: int, client=None, ensure_bbox: bool = True, **_kwargs: Any) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        idx = self._resolve_object_idx(int(object_idx))
        if idx < 0:
            return {"ok": False, "reason": "invalid_object_idx", "object_idx": int(object_idx)}
        with st.lock:
            prev_idx = int(getattr(st, "current_object_idx", 0))
            st.current_object_idx = int(idx)
        if int(prev_idx) != int(idx):
            hide_bbox_fn = getattr(ops, "_hide_bbox_for_object", None)
            if callable(hide_bbox_fn):
                try:
                    hide_bbox_fn(int(prev_idx))
                except Exception:
                    pass
        refresh_fn = getattr(ops, "_refresh_frustum_image", None)
        if callable(refresh_fn):
            try:
                refresh_fn()
            except Exception:
                pass
        update_status_fn = getattr(ops, "_update_status_handles", None)
        if callable(update_status_fn):
            try:
                update_status_fn()
            except Exception:
                pass
        if client is not None and bool(ensure_bbox):
            ensure_bbox_fn = getattr(ops, "_ensure_bbox_for_object", None)
            if callable(ensure_bbox_fn):
                try:
                    ensure_bbox_fn(client, int(idx))
                except Exception:
                    pass
        return {"ok": True, "object_idx": int(idx)}

    def _op_object_snapshot_state(self, object_idx: int | None = None, **_kwargs: Any) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        idx = self._resolve_object_idx(object_idx)
        if idx < 0:
            return {"ok": False, "reason": "invalid_object_idx", "object_idx": object_idx}
        clone_fn = getattr(ops, "_clone_data", None)
        with st.lock:
            obj = st.objects[idx]
            frame = getattr(obj, "bbox_frame_handle", None)
            box = getattr(obj, "bbox_box_handle", None)
            snap = {
                "object_idx": int(idx),
                "splat": self._clone_splat_data(getattr(obj, "splat", None), clone_fn=clone_fn),
                "manual_scale": float(getattr(obj, "manual_scale", 1.0)),
                "manual_scale_base_splat": self._clone_splat_data(
                    getattr(obj, "manual_scale_base_splat", None),
                    clone_fn=clone_fn,
                ),
                "bbox_dimensions": None
                if getattr(obj, "bbox_dimensions", None) is None
                else np.array(getattr(obj, "bbox_dimensions"), copy=True),
                "bbox_base_splat": self._clone_splat_data(getattr(obj, "bbox_base_splat", None), clone_fn=clone_fn),
                "bbox_base_center": None
                if getattr(obj, "bbox_base_center", None) is None
                else np.array(getattr(obj, "bbox_base_center"), copy=True),
                "bbox_base_wxyz": None
                if getattr(obj, "bbox_base_wxyz", None) is None
                else np.array(getattr(obj, "bbox_base_wxyz"), copy=True),
                "bbox_base_dims": None
                if getattr(obj, "bbox_base_dims", None) is None
                else np.array(getattr(obj, "bbox_base_dims"), copy=True),
                "bbox_frame_position": None if frame is None else np.array(getattr(frame, "position", (0, 0, 0)), dtype=np.float32),
                "bbox_frame_wxyz": None if frame is None else np.array(getattr(frame, "wxyz", (1, 0, 0, 0)), dtype=np.float32),
                "bbox_box_dimensions": None if box is None else np.array(getattr(box, "dimensions", (0, 0, 0)), dtype=np.float32),
            }
        return {"ok": True, "object_idx": int(idx), "snapshot": snap}

    def _op_object_restore_state(
        self,
        snapshot: dict[str, Any],
        client=None,
        make_current: bool = True,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        if not isinstance(snapshot, dict):
            return {"ok": False, "reason": "invalid_snapshot"}
        idx = self._resolve_object_idx(snapshot.get("object_idx", None))
        if idx < 0:
            return {"ok": False, "reason": "invalid_object_idx_in_snapshot"}
        if client is not None:
            ensure_bbox_fn = getattr(ops, "_ensure_bbox_for_object", None)
            if callable(ensure_bbox_fn):
                try:
                    ensure_bbox_fn(client, int(idx))
                except Exception:
                    pass

        with st.lock:
            obj = st.objects[idx]
            obj.splat = self._clone_splat_data(snapshot.get("splat", None), clone_fn=getattr(ops, "_clone_data", None))
            obj.manual_scale = float(snapshot.get("manual_scale", 1.0))
            obj.manual_scale_base_splat = self._clone_splat_data(
                snapshot.get("manual_scale_base_splat", None),
                clone_fn=getattr(ops, "_clone_data", None),
            )
            obj.bbox_dimensions = (
                None
                if snapshot.get("bbox_dimensions", None) is None
                else np.array(snapshot.get("bbox_dimensions"), dtype=np.float32, copy=True)
            )
            obj.bbox_base_splat = self._clone_splat_data(
                snapshot.get("bbox_base_splat", None),
                clone_fn=getattr(ops, "_clone_data", None),
            )
            obj.bbox_base_center = (
                None
                if snapshot.get("bbox_base_center", None) is None
                else np.array(snapshot.get("bbox_base_center"), dtype=np.float32, copy=True)
            )
            obj.bbox_base_wxyz = (
                None
                if snapshot.get("bbox_base_wxyz", None) is None
                else np.array(snapshot.get("bbox_base_wxyz"), dtype=np.float32, copy=True)
            )
            obj.bbox_base_dims = (
                None
                if snapshot.get("bbox_base_dims", None) is None
                else np.array(snapshot.get("bbox_base_dims"), dtype=np.float32, copy=True)
            )
            if bool(make_current):
                st.current_object_idx = int(idx)
            frame = getattr(obj, "bbox_frame_handle", None)
            box = getattr(obj, "bbox_box_handle", None)
            frame_pos = snapshot.get("bbox_frame_position", None)
            frame_wxyz = snapshot.get("bbox_frame_wxyz", None)
            box_dims = snapshot.get("bbox_box_dimensions", None)
            if frame is not None and frame_pos is not None:
                try:
                    frame.position = tuple(np.asarray(frame_pos, dtype=np.float32).reshape(-1).tolist())
                except Exception:
                    pass
            if frame is not None and frame_wxyz is not None:
                try:
                    frame.wxyz = tuple(np.asarray(frame_wxyz, dtype=np.float32).reshape(-1).tolist())
                except Exception:
                    pass
            if box is not None and box_dims is not None:
                try:
                    box.dimensions = tuple(np.asarray(box_dims, dtype=np.float32).reshape(-1).tolist())
                except Exception:
                    pass

        ensure_obj_fn = getattr(ops, "_ensure_object_handle", None)
        if callable(ensure_obj_fn):
            try:
                ensure_obj_fn(int(idx))
            except Exception:
                pass
        apply_live_fn = getattr(ops, "_apply_bbox_transform_live", None)
        if callable(apply_live_fn):
            try:
                apply_live_fn(int(idx))
            except Exception:
                pass
        return {"ok": True, "object_idx": int(idx)}

    def _op_transform_translate_x(
        self,
        client=None,
        object_idx: int | None = None,
        delta_x: float = 0.0,
        lock_axis: bool = True,
        commit: bool = True,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        idx = self._resolve_object_idx(object_idx)
        if idx < 0:
            return {"ok": False, "reason": "invalid_object_idx", "object_idx": object_idx}
        if client is None:
            return {"ok": False, "reason": "missing_client"}

        with st.lock:
            prev_idx = int(getattr(st, "current_object_idx", 0))
            st.current_object_idx = int(idx)
        try:
            ensure_bbox_fn = getattr(ops, "_ensure_bbox_for_object", None)
            if callable(ensure_bbox_fn):
                ensure_bbox_fn(client, int(idx))
            upd_mode = getattr(ops, "update_transform_mode", None)
            upd_axis = getattr(ops, "update_axis_constraint", None)
            if callable(upd_mode):
                upd_mode("Translate")
            if bool(lock_axis) and callable(upd_axis):
                upd_axis(True, False, False)

            with st.lock:
                obj = st.objects[idx]
                frame = getattr(obj, "bbox_frame_handle", None)
                fr_wxyz = None if getattr(st, "frustum_wxyz", None) is None else np.asarray(st.frustum_wxyz, dtype=np.float32).copy()
                fr_pos = None if getattr(st, "frustum_position", None) is None else np.asarray(st.frustum_position, dtype=np.float32).copy()
            if frame is None:
                return {"ok": False, "reason": "bbox_frame_missing"}
            pos_local = np.asarray(frame.position, dtype=np.float32).copy()
            pos_world = None
            if fr_wxyz is not None and fr_pos is not None:
                rot_fn = getattr(ops, "_rotation_matrix_from_wxyz", None)
                if callable(rot_fn):
                    try:
                        R_wc = np.asarray(rot_fn(np.asarray(fr_wxyz, dtype=np.float32)), dtype=np.float32)
                    except Exception:
                        R_wc = self._rotation_matrix_from_wxyz(np.asarray(fr_wxyz, dtype=np.float32))
                else:
                    R_wc = self._rotation_matrix_from_wxyz(np.asarray(fr_wxyz, dtype=np.float32))
                pos_world = pos_local @ R_wc.T + fr_pos
                pos_world[0] = float(pos_world[0] + float(delta_x))
                pos_local = (pos_world - fr_pos) @ R_wc
            else:
                # Fallback when frustum pose is unavailable.
                pos_local[0] = float(pos_local[0] + float(delta_x))
                pos_world = np.asarray(pos_local, dtype=np.float32)
            frame.position = tuple(np.asarray(pos_local, dtype=np.float32).tolist())
            apply_live_fn = getattr(ops, "_apply_bbox_transform_live", None)
            if callable(apply_live_fn):
                apply_live_fn(int(idx))
            if bool(commit):
                apply_fn = getattr(ops, "apply_transform", None)
                if callable(apply_fn):
                    apply_fn(client)
            return {
                "ok": True,
                "object_idx": int(idx),
                "delta_x": float(delta_x),
                "axis_frame": "world",
                "new_x": float(np.asarray(pos_world, dtype=np.float32)[0]),
                "new_center_local_xyz": np.asarray(pos_local, dtype=np.float32).astype(float).tolist(),
                "new_center_world_xyz": np.asarray(pos_world, dtype=np.float32).astype(float).tolist(),
            }
        finally:
            with st.lock:
                st.current_object_idx = int(prev_idx)

    def _op_pose_pixel_align(
        self,
        client=None,
        object_idx: int | None = None,
        lock_x: bool = True,
        lock_y: bool = False,
        lock_z: bool = False,
        steps: int = 160,
        lr: float = 0.05,
        min_side: int = 512,
        rgb_weight: float = 0.4,
        timeout_sec: float = 1500.0,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        ops = self._sam3d_ops()
        st = self._sam3d_state()
        if ops is None or st is None:
            return {"ok": False, "reason": "sam3d_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        idx = self._resolve_object_idx(object_idx)
        if idx < 0:
            return {"ok": False, "reason": "invalid_object_idx", "object_idx": object_idx}

        with st.lock:
            prev_idx = int(getattr(st, "current_object_idx", 0))
            st.current_object_idx = int(idx)
            center_reg_weight = float(getattr(st, "pose_opt_center_reg_weight", 1e-2))
            pose_reg_weight = float(getattr(st, "pose_opt_pose_reg_weight", 1e-3))
            scale_reg_weight = float(getattr(st, "pose_opt_scale_reg_weight", 1e-5))
            box_reg_weight = float(getattr(st, "pose_opt_box_reg_weight", 1e-1))
            debug_mode = bool(getattr(st, "pose_opt_debug_mode", False))

        try:
            run_fn = getattr(ops, "optimize_pose_current_object_async", None)
            if not callable(run_fn):
                return {"ok": False, "reason": "pose_opt_entry_missing"}
            lock_target_world_x = None
            lock_center_before = self._object_center_world(int(idx))
            if bool(lock_x) and isinstance(lock_center_before, dict):
                try:
                    lock_target_world_x = float((lock_center_before.get("center_world_xyz", [None]) or [None])[0])
                except Exception:
                    lock_target_world_x = None
            t0 = time.time()
            run_fn(
                client,
                steps=int(steps),
                lr=float(lr),
                min_side=int(min_side),
                rgb_weight=float(rgb_weight),
                center_reg_weight=float(center_reg_weight),
                pose_reg_weight=float(pose_reg_weight),
                scale_reg_weight=float(scale_reg_weight),
                box_reg_weight=float(box_reg_weight),
                debug_mode=bool(debug_mode),
                lock_translation_x=bool(lock_x),
                lock_translation_y=bool(lock_y),
                lock_translation_z=bool(lock_z),
            )
            deadline = t0 + max(1.0, float(timeout_sec))
            while time.time() < deadline:
                with st.lock:
                    running = bool(getattr(st, "pose_opt_running", False))
                if not running:
                    break
                time.sleep(0.2)
            with st.lock:
                running = bool(getattr(st, "pose_opt_running", False))
            if running:
                return {
                    "ok": False,
                    "reason": "pose_opt_timeout",
                    "object_idx": int(idx),
                    "timeout_sec": float(timeout_sec),
                }
            lock_drift_before = None
            lock_drift_after = None
            lock_post_correct_applied = False
            lock_post_correct_ok = False
            lock_center_after = self._object_center_world(int(idx))
            if bool(lock_x) and lock_target_world_x is not None and isinstance(lock_center_after, dict):
                try:
                    cur_world_x = float((lock_center_after.get("center_world_xyz", [None]) or [None])[0])
                    lock_drift_before = float(cur_world_x - float(lock_target_world_x))
                except Exception:
                    lock_drift_before = None
                lock_drift_after = lock_drift_before
            return {
                "ok": True,
                "object_idx": int(idx),
                "wait_sec": float(time.time() - t0),
                "lock_axis_frame": "world",
                "lock_x": bool(lock_x),
                "lock_y": bool(lock_y),
                "lock_z": bool(lock_z),
                "lock_target_world_x": (None if lock_target_world_x is None else float(lock_target_world_x)),
                "lock_drift_world_x_before_post_correct": (None if lock_drift_before is None else float(lock_drift_before)),
                "lock_post_correct_applied": bool(lock_post_correct_applied),
                "lock_post_correct_ok": bool(lock_post_correct_ok),
                "lock_drift_world_x_after_post_correct": (None if lock_drift_after is None else float(lock_drift_after)),
                "center_world_after_pose_opt": (
                    None
                    if not isinstance(lock_center_after, dict)
                    else list(lock_center_after.get("center_world_xyz", []))
                ),
            }
        finally:
            with st.lock:
                st.current_object_idx = int(prev_idx)

    def _op_render_object_multiview(
        self,
        client=None,
        object_idx: int | None = None,
        views: list[str] | None = None,
        image_size: int = 512,
        save_dir: str | None = None,
        prefix: str = "candidate",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        app = getattr(self.ctx, "app", None)
        st = self._sam3d_state()
        if app is None or st is None:
            return {"ok": False, "reason": "app_or_state_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        idx = self._resolve_object_idx(object_idx)
        if idx < 0:
            return {"ok": False, "reason": "invalid_object_idx", "object_idx": object_idx}

        with st.lock:
            obj = st.objects[idx]
            splat = getattr(obj, "splat", None)
        if splat is None:
            return {"ok": False, "reason": "object_has_no_splat", "object_idx": int(idx)}

        view_names = list(views or ["front", "back", "left", "right", "up", "down"])
        h = w = max(64, int(image_size))
        out_dir = Path(save_dir) if str(save_dir or "").strip() else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        fixed_near, fixed_far = self._camera_planes_from_app(app)
        prev_cam = {
            "position": np.asarray(client.camera.position, dtype=np.float32).copy(),
            "wxyz": np.asarray(client.camera.wxyz, dtype=np.float32).copy(),
            "look_at": np.asarray(client.camera.look_at, dtype=np.float32).copy(),
            "up": np.asarray(client.camera.up_direction, dtype=np.float32).copy(),
            "fov": float(client.camera.fov),
            "near": float(getattr(client.camera, "near", fixed_near)),
            "far": float(getattr(client.camera, "far", fixed_far)),
        }
        render_near = float(fixed_near)
        render_far = float(fixed_far)
        view_images: dict[str, np.ndarray] = {}
        view_paths: dict[str, str] = {}
        with st.lock:
            sam3d_frustum = getattr(st, "frustum_handle", None)
            prev_sam3d_max_pts = int(getattr(st, "display_max_points", 150000))
        prev_app_max_pts = int(getattr(app, "gs_display_max_points", 0)) if hasattr(app, "gs_display_max_points") else None
        app_frustums = list(getattr(app, "frustum_handles", []) or [])
        frustum_vis = self._collect_visible_handles(app_frustums, sam3d_frustum)
        try:
            if prev_app_max_pts is not None:
                try:
                    app.gs_display_max_points = 0
                    cache = getattr(app, "_display_cache", None)
                    if isinstance(cache, dict):
                        cache.clear()
                except Exception:
                    pass
            self._refresh_main_scene_splats(app, use_full=True)
            self._refresh_sam3d_object_handles(st, max_points=0)
            self._set_handles_visible(frustum_vis, False)
            for name in view_names:
                name_s = str(name).strip().lower()
                eye, wxyz = self._camera_pose_for_view_origin(app, name_s, 1.5)
                self._set_camera_pose(
                    app,
                    client,
                    eye,
                    wxyz,
                    look_at=self._camera_target_origin(),
                    up=np.asarray(getattr(app, "up_vec", [0.0, -1.0, 0.0]), dtype=np.float32),
                )
                try:
                    client.camera.near = float(render_near)
                    client.camera.far = float(render_far)
                except Exception:
                    pass
                client.flush()
                rgb = self._capture_rgb(
                    app,
                    client,
                    width=int(w),
                    height=int(h),
                    include_background=True,
                    include_objects=True,
                )
                view_images[name_s] = rgb
                if out_dir is not None:
                    p = out_dir / f"{str(prefix)}_{name_s}.png"
                    try:
                        Image.fromarray(rgb).save(p)
                        view_paths[name_s] = str(p)
                    except Exception:
                        pass
        finally:
            self._restore_handles_visible(frustum_vis)
            if prev_app_max_pts is not None:
                try:
                    app.gs_display_max_points = int(prev_app_max_pts)
                    cache = getattr(app, "_display_cache", None)
                    if isinstance(cache, dict):
                        cache.clear()
                except Exception:
                    pass
            self._refresh_main_scene_splats(app, use_full=False)
            self._refresh_sam3d_object_handles(st, max_points=int(prev_sam3d_max_pts))
            try:
                self._set_camera_pose(
                    app,
                    client,
                    prev_cam["position"],
                    prev_cam["wxyz"],
                    look_at=prev_cam["look_at"],
                    up=prev_cam["up"],
                )
                client.camera.fov = float(prev_cam["fov"])
                client.camera.near = float(fixed_near)
                client.camera.far = float(fixed_far)
                client.flush()
            except Exception:
                pass
        return {
            "ok": True,
            "object_idx": int(idx),
            "views": list(view_images.keys()),
            "view_images": view_images,
            "view_paths": view_paths,
            "image_size": int(w),
        }

    def _op_render_frame_multiview(
        self,
        client=None,
        frame_idx: int | None = None,
        views: list[str] | None = None,
        image_size: int = 512,
        save_dir: str | None = None,
        prefix: str = "frame",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Render multi-view images from persisted frame GS PLY (sequence.load)."""
        app = getattr(self.ctx, "app", None)
        if app is None:
            return {"ok": False, "reason": "app_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}

        seq = getattr(app, "sequence", None)
        load_fn = getattr(seq, "load", None)
        if seq is None or not callable(load_fn):
            return {"ok": False, "reason": "sequence_load_unavailable"}
        frame_tokens = list(getattr(seq, "frame_tokens", []) or [])
        if not frame_tokens:
            return {"ok": False, "reason": "frame_tokens_empty"}
        total = len(frame_tokens)

        idx = int(getattr(app, "current_frame", 0) if frame_idx is None else frame_idx)
        idx = max(0, min(idx, total - 1))
        tok = str(frame_tokens[idx])

        gs_path_str = ""
        gs_path_fn = getattr(seq, "gs_path", None)
        if callable(gs_path_fn):
            try:
                gs_path = Path(gs_path_fn(int(idx)))
                gs_path_str = str(gs_path)
                if not gs_path.exists():
                    return {
                        "ok": False,
                        "reason": "gs_ply_missing",
                        "frame_idx": int(idx),
                        "frame_token": tok,
                        "gs_path": gs_path_str,
                    }
            except Exception:
                gs_path_str = ""

        try:
            data_full = load_fn(int(idx))
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"gs_load_failed:{type(exc).__name__}:{exc}",
                "frame_idx": int(idx),
                "frame_token": tok,
                "gs_path": gs_path_str,
            }

        upd_fn = getattr(app, "_update_gaussian_handle", None)
        if not callable(upd_fn):
            return {
                "ok": False,
                "reason": "update_gaussian_handle_unavailable",
                "frame_idx": int(idx),
                "frame_token": tok,
                "gs_path": gs_path_str,
            }

        view_names = list(views or ["front", "back", "left", "right", "up", "down"])
        h = w = max(64, int(image_size))
        out_dir = Path(save_dir) if str(save_dir or "").strip() else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        fixed_near, fixed_far = self._camera_planes_from_app(app)
        prev_cam = {
            "position": np.asarray(client.camera.position, dtype=np.float32).copy(),
            "wxyz": np.asarray(client.camera.wxyz, dtype=np.float32).copy(),
            "look_at": np.asarray(client.camera.look_at, dtype=np.float32).copy(),
            "up": np.asarray(client.camera.up_direction, dtype=np.float32).copy(),
            "fov": float(client.camera.fov),
            "near": float(getattr(client.camera, "near", fixed_near)),
            "far": float(getattr(client.camera, "far", fixed_far)),
        }

        st = self._sam3d_state()
        prev_sam3d_max_pts = None
        sam3d_frustum = None
        if st is not None:
            try:
                with st.lock:
                    sam3d_frustum = getattr(st, "frustum_handle", None)
                    prev_sam3d_max_pts = int(getattr(st, "display_max_points", 150000))
            except Exception:
                prev_sam3d_max_pts = None
                sam3d_frustum = None

        prev_app_max_pts = int(getattr(app, "gs_display_max_points", 0)) if hasattr(app, "gs_display_max_points") else None
        prev_app_fixed_pts = int(getattr(app, "gs_display_fixed_points", 0)) if hasattr(app, "gs_display_fixed_points") else None
        app_frustums = list(getattr(app, "frustum_handles", []) or [])
        frustum_vis = self._collect_visible_handles(app_frustums, sam3d_frustum)
        view_images: dict[str, np.ndarray] = {}
        view_paths: dict[str, str] = {}

        try:
            try:
                app.current_frame = int(idx)
            except Exception:
                pass
            if prev_app_max_pts is not None:
                try:
                    app.gs_display_max_points = 0
                    if prev_app_fixed_pts is not None:
                        app.gs_display_fixed_points = 0
                    cache = getattr(app, "_display_cache", None)
                    if isinstance(cache, dict):
                        cache.clear()
                except Exception:
                    pass
            try:
                upd_fn(data_full)
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": f"apply_frame_gs_failed:{type(exc).__name__}:{exc}",
                    "frame_idx": int(idx),
                    "frame_token": tok,
                    "gs_path": gs_path_str,
                }
            if st is not None:
                try:
                    self._refresh_sam3d_object_handles(st, max_points=0)
                except Exception:
                    pass
            self._set_handles_visible(frustum_vis, False)
            for name in view_names:
                name_s = str(name).strip().lower()
                eye, wxyz = self._camera_pose_for_view_origin(app, name_s, 1.5)
                self._set_camera_pose(
                    app,
                    client,
                    eye,
                    wxyz,
                    look_at=self._camera_target_origin(),
                    up=np.asarray(getattr(app, "up_vec", [0.0, -1.0, 0.0]), dtype=np.float32),
                )
                try:
                    client.camera.near = float(fixed_near)
                    client.camera.far = float(fixed_far)
                except Exception:
                    pass
                client.flush()
                rgb = self._capture_rgb(
                    app,
                    client,
                    width=int(w),
                    height=int(h),
                    include_background=True,
                    include_objects=True,
                )
                view_images[name_s] = rgb
                if out_dir is not None:
                    p = out_dir / f"{str(prefix)}_{name_s}.png"
                    try:
                        Image.fromarray(rgb).save(p)
                        view_paths[name_s] = str(p)
                    except Exception:
                        pass
        finally:
            self._restore_handles_visible(frustum_vis)
            if prev_app_max_pts is not None:
                try:
                    app.gs_display_max_points = int(prev_app_max_pts)
                    if prev_app_fixed_pts is not None:
                        app.gs_display_fixed_points = int(prev_app_fixed_pts)
                    cache = getattr(app, "_display_cache", None)
                    if isinstance(cache, dict):
                        cache.clear()
                except Exception:
                    pass
            if st is not None and prev_sam3d_max_pts is not None:
                try:
                    self._refresh_sam3d_object_handles(st, max_points=int(prev_sam3d_max_pts))
                except Exception:
                    pass
            try:
                self._set_camera_pose(
                    app,
                    client,
                    prev_cam["position"],
                    prev_cam["wxyz"],
                    look_at=prev_cam["look_at"],
                    up=prev_cam["up"],
                )
                client.camera.fov = float(prev_cam["fov"])
                client.camera.near = float(fixed_near)
                client.camera.far = float(fixed_far)
                client.flush()
            except Exception:
                pass
        return {
            "ok": True,
            "frame_idx": int(idx),
            "frame_token": tok,
            "gs_path": gs_path_str,
            "views": list(view_images.keys()),
            "view_images": view_images,
            "view_paths": view_paths,
            "image_size": int(w),
            "source": "sequence_ply",
        }

    def _op_render_input_overlay(
        self,
        client=None,
        object_idx: int | None = None,
        save_path: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        app = getattr(self.ctx, "app", None)
        st = self._sam3d_state()
        if app is None or st is None:
            return {"ok": False, "reason": "app_or_state_unavailable"}
        if client is None:
            return {"ok": False, "reason": "missing_client"}
        idx = self._resolve_object_idx(object_idx)
        if idx < 0:
            return {"ok": False, "reason": "invalid_object_idx", "object_idx": object_idx}

        w = h = 512
        try:
            frame_token = str(app.sequence.frame_tokens[app.current_frame])
            img_path = Path(app.workspace.image_path_for_view(frame_token, "input"))
            if img_path.exists():
                with Image.open(img_path) as im:
                    w = max(64, int(im.width))
                    h = max(64, int(im.height))
        except Exception:
            pass

        with st.lock:
            obj = st.objects[idx]
            splat = getattr(obj, "splat", None)
            fr_pos = None if getattr(st, "frustum_position", None) is None else np.asarray(st.frustum_position, dtype=np.float32).copy()
            fr_wxyz = None if getattr(st, "frustum_wxyz", None) is None else np.asarray(st.frustum_wxyz, dtype=np.float32).copy()
            sam3d_frustum = getattr(st, "frustum_handle", None)
            prev_fr_opacity = float(getattr(st, "frustum_opacity", 1.0))
        if splat is None:
            return {"ok": False, "reason": "object_has_no_splat", "object_idx": int(idx)}

        if fr_pos is None or fr_wxyz is None:
            fr_pos, fr_wxyz = self._camera_pose_for_view_origin(app, "input", 1.5)

        fixed_near, fixed_far = self._camera_planes_from_app(app)
        prev_cam = {
            "position": np.asarray(client.camera.position, dtype=np.float32).copy(),
            "wxyz": np.asarray(client.camera.wxyz, dtype=np.float32).copy(),
            "look_at": np.asarray(client.camera.look_at, dtype=np.float32).copy(),
            "up": np.asarray(client.camera.up_direction, dtype=np.float32).copy(),
            "fov": float(client.camera.fov),
            "near": float(getattr(client.camera, "near", fixed_near)),
            "far": float(getattr(client.camera, "far", fixed_far)),
        }
        render_near = float(fixed_near)
        render_far = float(fixed_far)
        app_frustums = list(getattr(app, "frustum_handles", []) or [])
        all_frustums = self._collect_visible_handles(app_frustums, sam3d_frustum)
        out: np.ndarray | None = None
        try:
            # Keep only SAM3D input frustum visible in overlay render.
            self._set_handles_visible(all_frustums, False)
            if sam3d_frustum is not None:
                try:
                    sam3d_frustum.visible = True
                except Exception:
                    pass
            _ = self._set_sam3d_frustum_opacity(0.5)
            forward = self._rotation_matrix_from_wxyz(fr_wxyz)[:, 2].astype(np.float32)
            self._set_camera_pose(
                app,
                client,
                fr_pos,
                fr_wxyz,
                look_at=(np.asarray(fr_pos, dtype=np.float32) + forward),
                up=np.asarray(getattr(app, "up_vec", [0.0, -1.0, 0.0]), dtype=np.float32),
            )
            try:
                client.camera.near = float(render_near)
                client.camera.far = float(render_far)
            except Exception:
                pass
            client.flush()
            out = self._capture_input_overlay_rgb(
                app,
                client,
                width=int(w),
                height=int(h),
            )
        finally:
            _ = self._set_sam3d_frustum_opacity(float(prev_fr_opacity))
            self._restore_handles_visible(all_frustums)
            try:
                self._set_camera_pose(
                    app,
                    client,
                    prev_cam["position"],
                    prev_cam["wxyz"],
                    look_at=prev_cam["look_at"],
                    up=prev_cam["up"],
                )
                client.camera.fov = float(prev_cam["fov"])
                client.camera.near = float(fixed_near)
                client.camera.far = float(fixed_far)
                client.flush()
            except Exception:
                pass
        if out is None:
            return {"ok": False, "reason": "overlay_render_failed"}

        p = str(save_path or "").strip()
        if p:
            try:
                pp = Path(p)
                pp.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(out).save(pp)
            except Exception:
                pass
        return {
            "ok": True,
            "object_idx": int(idx),
            "overlay_rgb_u8": out,
            "mask_ratio": None,
            "image_shape": [int(h), int(w)],
        }

    def _op_render_input_image(
        self,
        save_path: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        app = getattr(self.ctx, "app", None)
        if app is None:
            return {"ok": False, "reason": "app_unavailable"}
        try:
            frame_token = str(app.sequence.frame_tokens[app.current_frame])
            img_path = Path(app.workspace.image_path_for_view(frame_token, "input"))
        except Exception:
            return {"ok": False, "reason": "input_frame_resolve_failed"}
        if not img_path.exists():
            return {"ok": False, "reason": "input_image_missing", "path": str(img_path)}
        try:
            img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            return {"ok": False, "reason": f"input_image_load_failed:{type(exc).__name__}:{exc}"}
        h, w = int(img.shape[0]), int(img.shape[1])
        p = str(save_path or "").strip()
        if p:
            try:
                pp = Path(p)
                pp.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(pp)
            except Exception:
                pass
        return {
            "ok": True,
            "input_rgb_u8": img,
            "input_image_path": str(img_path),
            "image_shape": [int(h), int(w)],
        }

    def _op_camera_input_pose(
        self,
        client=None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        app = getattr(self.ctx, "app", None)
        st = self._sam3d_state()
        if app is None:
            return {"ok": False, "reason": "app_unavailable"}

        w = h = 512
        try:
            frame_token = str(app.sequence.frame_tokens[app.current_frame])
            img_path = Path(app.workspace.image_path_for_view(frame_token, "input"))
            if img_path.exists():
                with Image.open(img_path) as im:
                    w = max(64, int(im.width))
                    h = max(64, int(im.height))
        except Exception:
            pass

        pos = None
        wxyz = None
        source = "fallback_input_view"
        if st is not None:
            try:
                with st.lock:
                    if getattr(st, "frustum_position", None) is not None:
                        pos = np.asarray(st.frustum_position, dtype=np.float32).copy()
                    if getattr(st, "frustum_wxyz", None) is not None:
                        wxyz = np.asarray(st.frustum_wxyz, dtype=np.float32).copy()
            except Exception:
                pos = None
                wxyz = None
        if pos is None or wxyz is None:
            try:
                pos, wxyz = self._camera_pose_for_view_origin(app, "input", 1.5)
            except Exception:
                pos = np.asarray([0.0, 0.0, 1.5], dtype=np.float32)
                wxyz = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            source = "view_preset_input"
        else:
            source = "sam3d_frustum"

        up = np.asarray(getattr(app, "up_vec", [0.0, -1.0, 0.0]), dtype=np.float32)
        look_at = self._camera_target_origin().astype(np.float32)
        fov_deg = float(getattr(app, "fov_deg", 49.1))
        near, far = self._camera_planes_from_app(app)
        if client is not None:
            try:
                fov_deg = float(np.rad2deg(float(client.camera.fov)))
            except Exception:
                pass

        return {
            "ok": True,
            "source": str(source),
            "position": np.asarray(pos, dtype=np.float32).astype(float).tolist(),
            "wxyz": np.asarray(wxyz, dtype=np.float32).astype(float).tolist(),
            "look_at": np.asarray(look_at, dtype=np.float32).astype(float).tolist(),
            "up": np.asarray(up, dtype=np.float32).astype(float).tolist(),
            "fov_deg": float(fov_deg),
            "near": float(near),
            "far": float(far),
            "image_width": int(w),
            "image_height": int(h),
        }

    def _op_physics_check_aabb(
        self,
        object_idx: int | None = None,
        x_min: float = -1.0,
        x_max: float = 1.0,
        overlap_eps: float = 1e-6,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        st = self._sam3d_state()
        if st is None:
            return {"ok": False, "reason": "sam3d_state_unavailable"}
        idx = self._resolve_object_idx(object_idx)
        if idx < 0:
            return {"ok": False, "reason": "invalid_object_idx", "object_idx": object_idx}

        with st.lock:
            objects = list(getattr(st, "objects", []) or [])
            fr_wxyz = None if getattr(st, "frustum_wxyz", None) is None else np.asarray(st.frustum_wxyz, dtype=np.float32).copy()
            fr_pos = None if getattr(st, "frustum_position", None) is None else np.asarray(st.frustum_position, dtype=np.float32).copy()

        if fr_wxyz is not None and fr_pos is not None:
            R_wc = self._rotation_matrix_from_wxyz(np.asarray(fr_wxyz, dtype=np.float32))
            t_wc = np.asarray(fr_pos, dtype=np.float32).reshape(1, 3)
        else:
            R_wc = np.eye(3, dtype=np.float32)
            t_wc = np.zeros((1, 3), dtype=np.float32)

        aabbs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for i, obj in enumerate(objects):
            splat = getattr(obj, "splat", None)
            if splat is None:
                continue
            centers_local = np.asarray(getattr(splat, "centers", None), dtype=np.float32)
            if centers_local.ndim != 2 or centers_local.shape[0] <= 0:
                continue
            centers_world = centers_local @ R_wc.T + t_wc
            mn, mx = self._aabb_of_centers(centers_world)
            center = centers_world.mean(axis=0)
            aabbs[i] = (mn, mx, center)

        if idx not in aabbs:
            return {"ok": False, "reason": "target_has_no_aabb", "object_idx": int(idx)}
        t_min, t_max, t_center = aabbs[idx]

        collisions: list[dict[str, Any]] = []
        for j, (o_min, o_max, o_center) in aabbs.items():
            if j == idx:
                continue
            ov = self._aabb_overlap(t_min, t_max, o_min, o_max)
            if float(ov[0]) > float(overlap_eps) and float(ov[1]) > float(overlap_eps) and float(ov[2]) > float(overlap_eps):
                collisions.append(
                    {
                        "other_object_idx": int(j),
                        "overlap_xyz": ov.astype(float).tolist(),
                        "overlap_volume": float(np.prod(ov)),
                        "other_center_x": float(o_center[0]),
                    }
                )
        collisions.sort(key=lambda x: float(x.get("overlap_volume", 0.0)), reverse=True)
        out_of_bounds = bool(float(t_center[0]) < float(x_min) or float(t_center[0]) > float(x_max))
        return {
            "ok": True,
            "object_idx": int(idx),
            "axis_frame": "world",
            "target_center_xyz": np.asarray(t_center, dtype=np.float32).astype(float).tolist(),
            "target_aabb_min": np.asarray(t_min, dtype=np.float32).astype(float).tolist(),
            "target_aabb_max": np.asarray(t_max, dtype=np.float32).astype(float).tolist(),
            "x_bounds": [float(x_min), float(x_max)],
            "x_out_of_bounds": bool(out_of_bounds),
            "collisions": collisions,
            "collision_count": int(len(collisions)),
        }
