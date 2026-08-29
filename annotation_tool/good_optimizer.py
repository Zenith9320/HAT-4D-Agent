from __future__ import annotations

import json
import math
import time
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import viser.transforms as tf
from tqdm import tqdm
from metric.evaluator import MetricEvaluator

try:
    from gsplat.rendering import rasterization as gsplat_rasterization
except Exception:  # noqa: BLE001
    gsplat_rasterization = None

from utils.constants import SH_C0, COORD_ALIGN_MAT
from utils.debug_utils import configured_debug_dir
from utils.geometry_utils import compute_center_radius, sanitize_points
from utils.image_utils import ensure_rgb, load_image
from utils.ply_utils import (
    read_ply_vertices,
    write_ply_vertices,
    load_gaussian_tensors_transformed,
    write_ply_subset_aligned,
)
from utils.camera_utils import (
    direction_from_angles,
    eye_wxyz_from_angles,
    interp_angle,
    look_at_rotation,
    spherical_to_vec,
    vec_to_spherical,
)

ROOT = Path(__file__).resolve().parent


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Simple SSIM implementation for 3-channel images.

    Expects shape (B, C, H, W). Returns scalar tensor.
    """

    channel = img1.size(1)
    # Build Gaussian window.
    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype)
    gauss = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * 1.5 ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(1)
    window_2d = (gauss @ gauss.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def draw_bbox_outline(
    image: np.ndarray, bbox: Dict[str, int], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3
) -> np.ndarray:
    img = ensure_rgb(image)
    h, w = img.shape[:2]
    x0, x1 = int(bbox["x0"]), int(bbox["x1"])
    y0, y1 = int(bbox["y0"]), int(bbox["y1"])
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    img = img.copy()
    img[y0 : min(y0 + thickness, y1), x0:x1] = color
    img[max(y1 - thickness, 0):y1, x0:x1] = color
    img[y0:y1, x0 : min(x0 + thickness, x1)] = color
    img[y0:y1, max(x1 - thickness, 0):x1] = color
    return img


class FixOptimizer:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.progress: Dict[str, Dict] = {}
        self.metric_eval = MetricEvaluator(self.device)

    # --- Camera helpers -------------------------------------------------
    def _view_direction(self, view: str, up_axis: str = "y", yaw_off: float = 0.0, pitch_off: float = 0.0) -> np.ndarray:
        presets = {
            "input": (0.0, 0.0),
            "eval_0": (-75.0, 0.0),
            "eval_1": (15.0, 0.0),
            "eval_2": (105.0, 0.0),
            "eval_3": (195.0, 0.0),
        }
        az_deg, el_deg = presets.get(view, (0.0, 0.0))
        return direction_from_angles(az_deg + yaw_off, el_deg + pitch_off, up_axis)

    def _camera_pose_for_view(
        self, view: str, center: np.ndarray, depth: float, up_axis: str = "y",
        yaw_off: float = 0.0, pitch_off: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        presets = {
            "input": (0.0, 0.0),
            "eval_0": (-75.0, 0.0),
            "eval_1": (15.0, 0.0),
            "eval_2": (105.0, 0.0),
            "eval_3": (195.0, 0.0),
        }
        az_deg, el_deg = presets.get(view, (0.0, 0.0))
        return eye_wxyz_from_angles(az_deg + yaw_off, el_deg + pitch_off, depth, center, up_axis)

    def _camera_to_world_rotation(self, wxyz: List[float]) -> np.ndarray:
        return tf.SO3(np.asarray(wxyz, dtype=np.float32)).as_matrix()

    # --- Camera math helpers ----------------------------------------
    @staticmethod
    def _vec_to_spherical(vec: np.ndarray) -> Tuple[float, float, float]:
        return vec_to_spherical(vec)

    @staticmethod
    def _spherical_to_vec(radius: float, yaw: float, pitch: float) -> np.ndarray:
        return spherical_to_vec(radius, yaw, pitch)

    @staticmethod
    def _interp_angle(a0: float, a1: float, t: float) -> float:
        return interp_angle(a0, a1, t)

    @staticmethod
    def _look_at_rotation(eye_vec: np.ndarray, center_target: np.ndarray, up_axis: str) -> Tuple[np.ndarray, np.ndarray]:
        return look_at_rotation(eye_vec, center_target, up_axis)

    # --- PLY helpers ----------------------------------------------------
    def _load_ply_tensors(self, gs_path: Path):
        prop_names, vertices, header_lines, data = load_gaussian_tensors_transformed(gs_path, return_header=True)
        centers = data["means"]
        scales = data["scales"]
        rotations = data["rotations"]
        opacity = data["opacity"]
        sh_dc = data["sh_dc"]
        tensors = {
            "means": torch.from_numpy(centers).to(self.device),
            "scales": torch.from_numpy(scales).to(self.device),
            "rotations": torch.from_numpy(rotations).to(self.device),
            "opacity": torch.from_numpy(opacity).to(self.device),
            "sh_dc": torch.from_numpy(sh_dc).to(self.device),
        }
        return prop_names, vertices, header_lines, tensors

    def _combine_params(self, base: torch.Tensor, updates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype == torch.bool:
            expected = int(indices.sum().item())
        else:
            expected = int(indices.numel())
        if int(updates.shape[0]) != expected:
            raise RuntimeError(
                f"_combine_params shape mismatch: updates has N={int(updates.shape[0])}, "
                f"but indices selects N={expected}. "
                "If you are using GS densification (split), ensure `sel/indices_arr` and the full tensors "
                "(means/scales/rotations/opacity/sh_dc) are expanded consistently."
            )
        out = base.clone()
        out[indices] = updates
        return out

    # --- Gsplat helpers -------------------------------------------------
    def _prepare_gsplat_inputs_tensor(
        self,
        means: torch.Tensor | np.ndarray,
        scales: torch.Tensor | np.ndarray,
        rotations: torch.Tensor | np.ndarray | None,
        opacity: torch.Tensor | np.ndarray,
        sh_dc: torch.Tensor | np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize GS tensors for gsplat.

        Rotations are accepted in quaternion (N, 4), axis-angle (N, 3), or rotation
        matrix (N, 3, 3) forms. Everything is moved to the optimizer device and
        cast to float32.
        """

        device = self.device
        means_t = torch.as_tensor(means, device=device, dtype=torch.float32)
        scales_t = torch.as_tensor(scales, device=device, dtype=torch.float32)
        rotations_t = torch.as_tensor(rotations if rotations is not None else None, device=device) if rotations is not None else None
        opacity_t = torch.as_tensor(opacity, device=device, dtype=torch.float32)
        sh_t = torch.as_tensor(sh_dc, device=device, dtype=torch.float32)

        if rotations_t is None:
            rotations_t = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).expand(means_t.shape[0], 4)
        if rotations_t.ndim == 1:
            rotations_t = rotations_t.expand(means_t.shape[0], -1)
        if rotations_t.ndim == 3 and rotations_t.shape[1:] == (3, 3):
            rotations_t = tf.SO3(rotations_t).wxyz.to(device=device, dtype=torch.float32)
        if rotations_t.ndim == 2 and rotations_t.shape[1] == 3:
            ones = torch.ones((rotations_t.shape[0], 1), device=device, dtype=torch.float32)
            rotations_t = torch.cat([ones, rotations_t], dim=1)
        if rotations_t.shape[-1] != 4:
            rotations_t = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).expand(means_t.shape[0], 4)
        rotations_t = rotations_t / (rotations_t.norm(dim=1, keepdim=True) + 1e-8)

        scales_t = torch.exp(scales_t)
        opacity_t = torch.sigmoid(opacity_t)
        colors_full = torch.clamp(0.5 + SH_C0 * sh_t, 0.0, 1.0)

        return means_t, scales_t, rotations_t, opacity_t, colors_full

    def _build_camera_tensors(
        self, rotation_wc: np.ndarray, position_w: np.ndarray, width: int, height: int, fov_deg: float
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        device = self.device
        fx = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        fy = height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        cx = width * 0.5
        cy = height * 0.5
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        Ks = K.unsqueeze(0)

        R_cw = torch.as_tensor(rotation_wc.T, device=device, dtype=torch.float32)
        t_cw = (-R_cw @ torch.as_tensor(position_w, device=device, dtype=torch.float32)).view(3)
        viewmat = torch.eye(4, device=device, dtype=torch.float32)
        viewmat[:3, :3] = R_cw
        viewmat[:3, 3] = t_cw
        viewmats = viewmat.unsqueeze(0)

        near, far = 0.01, 100.0
        return viewmats, Ks, near, far

    def _gsplat_render(
        self,
        means_t: torch.Tensor,
        scales_t: torch.Tensor,
        rotations_t: torch.Tensor,
        opacity_t: torch.Tensor,
        colors_in: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        near: float,
        far: float,
        detach: bool = True,
        to_numpy: bool = True,
        backgrounds: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor | np.ndarray:
        if backgrounds is None:
            backgrounds = torch.ones((1, 3), device=means_t.device, dtype=colors_in.dtype)
        else:
            backgrounds = torch.as_tensor(backgrounds, device=means_t.device, dtype=colors_in.dtype)
        out = gsplat_rasterization(
            means=means_t,
            quats=rotations_t,
            scales=scales_t,
            opacities=opacity_t,
            colors=colors_in,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=near,
            far_plane=far,
            packed=False,
            backgrounds=backgrounds,
            render_mode="RGB",
        )
        img = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(img, dict):
            img = img.get("rgb") or img.get("color") or img.get("image")
        if img is None:
            raise RuntimeError("gsplat rasterization returned None")
        if img.dim() == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        if detach:
            img = img.detach()
        img = img.clamp(0.0, 1.0)
        if to_numpy:
            return img.cpu().numpy().astype(np.float32)
        return img

    # --- Rendering utilities -------------------------------------------
    def _render_full_simple(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        opacity: np.ndarray,
        sh_dc: np.ndarray,
        rotation_wc: np.ndarray,
        position_w: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        colors_override: np.ndarray | None = None,
    ) -> np.ndarray:
        if gsplat_rasterization is None:
            pts_cam = (sanitize_points(means) - sanitize_points(position_w[None, :])) @ rotation_wc
            z = pts_cam[:, 2]
            fx = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
            fy = height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
            u = fx * (pts_cam[:, 0] / np.maximum(z, 1e-4)) + width * 0.5
            v = fy * (pts_cam[:, 1] / np.maximum(z, 1e-4)) + height * 0.5

            valid = (z > 0) & np.isfinite(z)
            ui = np.round(u[valid]).astype(np.int64)
            vi = np.round(v[valid]).astype(np.int64)
            inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            if not np.any(inside):
                return np.zeros((height, width, 3), dtype=np.float32)
            ui = ui[inside]
            vi = vi[inside]
            idx_valid = np.nonzero(valid)[0][inside]
            depth = z[idx_valid]

            depth_map = np.full((height, width), np.inf, dtype=np.float32)
            index_map = np.full((height, width), -1, dtype=np.int64)
            for px, py, d, idx in zip(ui, vi, depth, idx_valid):
                if d < depth_map[py, px]:
                    depth_map[py, px] = d
                    index_map[py, px] = idx

            chosen = index_map >= 0
            if not np.any(chosen):
                return np.zeros((height, width, 3), dtype=np.float32)
            chosen_idx = index_map[chosen]
            alpha = _sigmoid_np(opacity[chosen_idx])
            colors = (
                colors_override[chosen_idx]
                if colors_override is not None
                else np.clip(0.5 + SH_C0 * sh_dc[chosen_idx], 0.0, 1.0)
            )
            img = np.zeros((height, width, 3), dtype=np.float32)
            img[chosen] = colors * alpha[:, None]
            return img

        means_t, scales_t, rotations_t, opacity_t, colors_full = self._prepare_gsplat_inputs_tensor(
            means, scales, rotations, opacity, sh_dc
        )
        if colors_override is not None:
            colors_full = torch.as_tensor(np.asarray(colors_override, dtype=np.float32), device=self.device)
        viewmats, Ks, near, far = self._build_camera_tensors(rotation_wc, position_w, width, height, fov_deg)
        return self._gsplat_render(
            means_t,
            scales_t,
            rotations_t,
            opacity_t,
            colors_full,
            viewmats,
            Ks,
            width,
            height,
            near,
            far,
            detach=True,
            to_numpy=True,
        )

    def _render_visibility_mask(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        opacity: np.ndarray,
        sh_dc: np.ndarray,
        rotation_wc: np.ndarray,
        position_w: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        selected_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if gsplat_rasterization is not None:
            return self._render_visibility_mask_gsplat(
                means,
                scales,
                rotations,
                opacity,
                sh_dc,
                rotation_wc,
                position_w,
                width,
                height,
                fov_deg,
                selected_mask,
            )
        return self._render_visibility_mask_simple(
            means,
            scales,
            rotations,
            opacity,
            sh_dc,
            rotation_wc,
            position_w,
            width,
            height,
            fov_deg,
            selected_mask,
        )

    def _render_visibility_mask_simple(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        opacity: np.ndarray,
        sh_dc: np.ndarray,
        rotation_wc: np.ndarray,
        position_w: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        selected_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gaussian splat accumulation (no point sampling) for visibility mask."""
        pts_cam = (sanitize_points(means) - sanitize_points(position_w[None, :])) @ rotation_wc
        z = pts_cam[:, 2]
        fx = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        fy = height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        u = fx * (pts_cam[:, 0] / np.maximum(z, 1e-4)) + width * 0.5
        v = fy * (pts_cam[:, 1] / np.maximum(z, 1e-4)) + height * 0.5

        valid = (z > 0) & np.isfinite(z)
        if not np.any(valid):
            zeros = np.zeros((height, width, 3), dtype=np.float32)
            return zeros, zeros, zeros

        # Footprint estimate (isotropic) and alpha.
        sigma = np.exp(scales.mean(axis=1))
        sigma_px = np.zeros_like(z, dtype=np.float32)
        sigma_px[valid] = np.clip(sigma[valid] * fx / np.maximum(z[valid], 1e-4), 1.0, float(max(width, height)))
        alpha_all = _sigmoid_np(opacity)
        colors_full = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)

        color_accum = np.zeros((height, width, 3), dtype=np.float32)
        mask_accum = np.zeros((height, width, 3), dtype=np.float32)
        weight_accum = np.zeros((height, width), dtype=np.float32)

        idxs = np.nonzero(valid)[0]
        for idx in idxs:
            px = int(round(u[idx]))
            py = int(round(v[idx]))
            if px < 0 or px >= width or py < 0 or py >= height:
                continue
            r = int(min(max(1.0, 3.0 * sigma_px[idx]), float(max(width, height))))
            x0, x1 = max(0, px - r), min(width, px + r + 1)
            y0, y1 = max(0, py - r), min(height, py + r + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            dist2 = (gx - u[idx]) ** 2 + (gy - v[idx]) ** 2
            w = np.exp(-0.5 * dist2 / (sigma_px[idx] ** 2 + 1e-6)) * alpha_all[idx]
            w3 = w[..., None]
            color_accum[y0:y1, x0:x1] += w3 * colors_full[idx]
            if selected_mask[idx]:
                mask_accum[y0:y1, x0:x1] += w3
            weight_accum[y0:y1, x0:x1] += w

        weight_safe = np.maximum(weight_accum, 1e-6)
        render_full = color_accum / weight_safe[..., None]
        mask_render = mask_accum / weight_safe[..., None]
        mask_rgb = np.repeat(np.clip(weight_accum / (weight_accum.max() + 1e-6), 0.0, 1.0)[..., None], 3, axis=2)
        return render_full, mask_render, mask_rgb

    def _render_visibility_mask_gsplat(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        opacity: np.ndarray,
        sh_dc: np.ndarray,
        rotation_wc: np.ndarray,
        position_w: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        selected_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use gsplat.rasterization for higher-quality visibility masks."""
        means_t, scales_t, rotations_t, opacity_t, colors_full = self._prepare_gsplat_inputs_tensor(
            means, scales, rotations, opacity, sh_dc
        )
        viewmats, Ks, near, far = self._build_camera_tensors(rotation_wc, position_w, width, height, fov_deg)

        render_full = self._gsplat_render(
            means_t,
            scales_t,
            rotations_t,
            opacity_t,
            colors_full,
            viewmats,
            Ks,
            width,
            height,
            near,
            far,
            detach=True,
            to_numpy=True,
            backgrounds=np.zeros((1, 3), dtype=np.float32),
        )
        color_override = np.full_like(np.asarray(sh_dc, dtype=np.float32), (-0.5 / SH_C0), dtype=np.float32)
        color_override[selected_mask.astype(bool)] = (0.5 / SH_C0)
        mask_render = self._gsplat_render(
            means_t,
            scales_t,
            rotations_t,
            opacity_t,
            torch.as_tensor(color_override, device=self.device, dtype=torch.float32),
            viewmats,
            Ks,
            width,
            height,
            near,
            far,
            detach=True,
            to_numpy=True,
            backgrounds=np.zeros((1, 3), dtype=np.float32),
        )
        mask_gray = mask_render.mean(axis=2, keepdims=True)
        mask_rgb = np.repeat(mask_gray, 3, axis=2)
        return render_full, mask_render, mask_rgb


    def _get_first_frame_gt(
        self,
        frame_token: str | None,
        workspace_root: Path,
        gt_image_path: str | Path | None,
    ) -> np.ndarray | None:
        if frame_token is None:
            return None
        input_img_path = workspace_root / "images" / "input" / f"{frame_token}.png"
        if input_img_path.exists():
            return ensure_rgb(load_image(input_img_path, flip_udlr=False))
        if gt_image_path and Path(gt_image_path).exists():
            return ensure_rgb(load_image(Path(gt_image_path), flip_udlr=False))
        return None

    def _render_interp_video_frames(
        self,
        start_eye: np.ndarray,
        start_wxyz: np.ndarray,
        end_eye: np.ndarray,
        end_wxyz: np.ndarray,
        out_root: Path,
        render_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        video_samples: int,
        frame_token: str | None,
        workspace_root: Path,
        gt_image_path: str | Path | None,
        center_target: np.ndarray,
        up_axis: str,
        width: int,
        height: int,
        fov_deg: float,
    ) -> Dict:
        frames_dir = out_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        Ks_t, near, far = None, None, None
        meta_entries: List[Dict] = []

        first_frame_gt = self._get_first_frame_gt(frame_token, workspace_root, gt_image_path)
        r0, yaw0, pitch0 = self._vec_to_spherical(start_eye)
        r1, yaw1, pitch1 = self._vec_to_spherical(end_eye)
        radius = r0 if r0 > 1e-4 else (r1 if r1 > 1e-4 else 1.5)

        for idx in range(max(1, video_samples)):
            t = 0.0 if video_samples <= 1 else float(idx) / float(video_samples - 1)
            yaw = self._interp_angle(yaw0, yaw1, t)
            pitch = self._interp_angle(pitch0, pitch1, t)
            eye = self._spherical_to_vec(radius, yaw, pitch)
            rotation_wc, wxyz = self._look_at_rotation(eye, center_target, up_axis)

            if idx == 0 and first_frame_gt is not None:
                render_full = first_frame_gt.astype(np.float32) / 255.0
                mask_render = np.zeros_like(render_full, dtype=np.float32)
            else:
                render_full, mask_render = render_fn(rotation_wc, eye)

            rgb_path = frames_dir / f"rgb_{idx:04d}.png"
            mask_path = frames_dir / f"mask_{idx:04d}.png"
            iio.imwrite(rgb_path, (np.clip(render_full, 0, 1) * 255.0).astype(np.uint8))
            iio.imwrite(mask_path, (np.clip(mask_render[..., 0], 0, 1) * 255.0).astype(np.uint8))

            if Ks_t is None:
                viewmats, Ks_t, near, far = self._build_camera_tensors(rotation_wc, eye, width, height, fov_deg)
            cam_entry = {
                "eye": eye.astype(float).tolist(),
                "wxyz": wxyz.astype(float).tolist(),
                "K": Ks_t[0].cpu().numpy().astype(float).tolist(),
                "near": float(near),
                "far": float(far),
                "width": width,
                "height": height,
                "fov_deg": fov_deg,
            }
            cam_path = frames_dir / f"camera_{idx:04d}.json"
            cam_path.write_text(json.dumps(cam_entry, indent=2))
            meta_entries.append({"rgb": str(rgb_path), "mask": str(mask_path), "camera": str(cam_path)})

        meta_path = out_root / "frames_meta.json"
        meta_path.write_text(json.dumps(meta_entries, indent=2))
        return {
            "frames": [m["rgb"] for m in meta_entries],
            "masks": [m["mask"] for m in meta_entries],
            "cameras": [m["camera"] for m in meta_entries],
            "meta": str(meta_path),
        }
    def _render_full_and_mask_numpy(
        self,
        means_np: np.ndarray,
        scales_np: np.ndarray,
        rotations_np: np.ndarray,
        opacity_np: np.ndarray,
        sh_dc_np: np.ndarray,
        rotation_wc: np.ndarray,
        position_w: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        selected_mask_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        render_full = self._render_full_simple(
            means_np,
            scales_np,
            rotations_np,
            opacity_np,
            sh_dc_np,
            rotation_wc,
            position_w,
            width,
            height,
            fov_deg,
        )
        _, mask_render, _ = self._render_visibility_mask(
            means_np,
            scales_np,
            rotations_np,
            opacity_np,
            sh_dc_np,
            rotation_wc,
            position_w,
            width,
            height,
            fov_deg,
            selected_mask_np,
        )
        return render_full, mask_render



    def _render_roi_patch(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        sh_dc: torch.Tensor,
        rotation_wc: np.ndarray,
        position_w: np.ndarray,
        width: int,
        height: int,
        bbox: Dict[str, int],
        fov_deg: float,
        colors_override: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gsplat_rasterization is None:
            device = means.device
            rotation_wc_t = torch.as_tensor(rotation_wc, device=device, dtype=torch.float32)
            position_t = torch.as_tensor(position_w, device=device, dtype=torch.float32)

            pts_cam = (means - position_t[None, :]) @ rotation_wc_t
            z = pts_cam[:, 2]
            valid = torch.isfinite(z) & (z > 1e-4)
            if not torch.any(valid):
                zeros = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
                return zeros, zeros

            pts_cam = pts_cam[valid]
            z = z[valid]
            u = pts_cam[:, 0] / z
            v = pts_cam[:, 1] / z
            fx = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
            fy = height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
            u = fx * u + width * 0.5
            v = fy * v + height * 0.5

            colors = torch.clamp(0.5 + SH_C0 * (sh_dc[valid] if colors_override is None else colors_override[valid]), 0.0, 1.0)
            sigma = torch.exp(scales[valid].mean(dim=1))
            sigma_px = torch.clamp(sigma * fx / torch.clamp(z, min=1e-4), min=1.0, max=float(max(width, height) * 2))
            alpha = torch.sigmoid(opacity[valid])

            x0, x1 = int(bbox["x0"]), int(bbox["x1"])
            y0, y1 = int(bbox["y0"]), int(bbox["y1"])
            x0 = max(0, min(x0, width - 1))
            x1 = max(x0 + 1, min(x1, width))
            y0 = max(0, min(y0, height - 1))
            y1 = max(y0 + 1, min(y1, height))

            margin = 3.0 * sigma_px
            keep = (
                (u + margin >= x0)
                & (u - margin <= x1)
                & (v + margin >= y0)
                & (v - margin <= y1)
            )
            if not torch.any(keep):
                zeros = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
                return zeros, zeros

            u = u[keep]
            v = v[keep]
            colors = colors[keep]
            sigma_px = sigma_px[keep]
            alpha = alpha[keep]

            xs = torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5
            ys = torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            gx = gx.reshape(1, -1)
            gy = gy.reshape(1, -1)

            u = u[:, None]
            v = v[:, None]
            sigma_sq = (sigma_px[:, None] ** 2) + 1e-6
            dist2 = (gx - u) ** 2 + (gy - v) ** 2
            weight = torch.exp(-0.5 * dist2 / sigma_sq)

            color_weights = alpha[:, None] * colors
            color_map = weight.transpose(0, 1) @ color_weights
            alpha_map = weight.transpose(0, 1) @ alpha
            alpha_map = alpha_map.clamp_min(1e-6)
            color_map = color_map / alpha_map[:, None]

            roi_h, roi_w = y1 - y0, x1 - x0
            roi_img = color_map.reshape(roi_h, roi_w, 3)
            roi_alpha = alpha_map.reshape(roi_h, roi_w, 1)

            full = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
            full_alpha = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
            full[y0:y1, x0:x1] = roi_img
            full_alpha[y0:y1, x0:x1] = roi_alpha
            return full, full_alpha

        means_t, scales_t, rotations_t, opacity_t, colors_full = self._prepare_gsplat_inputs_tensor(
            means, scales, rotations, opacity, sh_dc
        )
        if colors_override is not None:
            colors_full = torch.clamp(torch.as_tensor(colors_override, device=self.device, dtype=torch.float32), 0.0, 1.0)

        viewmats, Ks, near, far = self._build_camera_tensors(rotation_wc, position_w, width, height, fov_deg)
        img = self._gsplat_render(
            means_t,
            scales_t,
            rotations_t,
            opacity_t,
            colors_full,
            viewmats,
            Ks,
            width,
            height,
            near,
            far,
            detach=False,
            to_numpy=False,
        )

        x0, x1 = int(bbox["x0"]), int(bbox["x1"])
        y0, y1 = int(bbox["y0"]), int(bbox["y1"])
        x0 = max(0, min(x0, width - 1))
        x1 = max(x0 + 1, min(x1, width))
        y0 = max(0, min(y0, height - 1))
        y1 = max(y0 + 1, min(y1, height))

        mask = torch.zeros_like(img)
        mask[y0:y1, x0:x1] = 1.0
        img_roi = img * mask
        return img_roi, mask

    # --- ROI optimize helpers ---------------------------------------
    def _parse_handle_inputs(self, payload: Dict) -> Dict:
        selection = payload.get("selection", {})
        indices = selection.get("gs_indices") or selection.get("indices") or []
        if not indices:
            raise ValueError("No selection indices provided for fix.")

        paths = payload.get("paths", {})
        if "gs_ply" not in paths:
            raise ValueError("paths.gs_ply is required.")

        gs_path = Path(paths["gs_ply"])
        output_dir = Path(paths.get("output_dir", ROOT / "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        workspace_root = Path(paths.get("workspace_root", output_dir.parent))

        width, height = payload.get("image_size", [512, 512])
        width, height = int(width), int(height)
        cam_info = payload.get("camera", {})
        fov_deg = float(cam_info.get("fov_deg", 49.1))
        up_axis = payload.get("up_axis", "y")
        camera_overrides = payload.get("camera_overrides", {}) or {}
        frame_token = payload.get("frame_token")
        gt_image_path = paths.get("gt_image")
        eval_gt_root = Path(paths.get("eval_gt_dir", ""))
        depth_margin_ratio = float(payload.get("depth_margin_ratio", 0.05))
        min_depth_margin = float(payload.get("min_depth_margin", 0.01))

        opt_params = payload.get("opt_params", {})
        iters = int(opt_params.get("iters", 1500))
        lr_color = float(opt_params.get("lr_color", 0.0025))
        lr_opacity = float(opt_params.get("lr_opacity", 0.05))
        lr_scale = float(opt_params.get("lr_scale", 0.005))
        lr_rotation = float(opt_params.get("lr_rotation", 0.001))
        lr_means = float(opt_params.get("lr_means", 0.00016))
        lr_final_factor = float(opt_params.get("lr_final_factor", 0.02))
        lambda_dssim = float(opt_params.get("lambda_dssim", 0.2))
        full_loss_weight = float(opt_params.get("full_loss_weight", 0.5))
        optimize_means = bool(opt_params.get("optimize_means", True))
        optimize_rotation = bool(opt_params.get("optimize_rotation", True))
        unlock_ratio = float(opt_params.get("unlock_ratio", 0.35))
        video_samples = int(payload.get("video_samples", opt_params.get("video_samples", 16)))
        render_video_only = bool(payload.get("render_video_only", opt_params.get("render_video_only", False)))
        mask_dilate = int(opt_params.get("mask_dilate", payload.get("mask_dilate", 0)))

        split_enable = bool(opt_params.get("split_enable", True))
        split_every = int(opt_params.get("split_every", 20))
        split_start = int(opt_params.get("split_start", int(iters * 0.2)))
        split_max_new = int(opt_params.get("split_max_new", 512))
        split_topk = int(opt_params.get("split_topk", 128))
        split_score = str(opt_params.get("split_score", "scale_opacity_grad"))
        split_thresh = float(opt_params.get("split_thresh", 1e-3))
        split_jitter = float(opt_params.get("split_jitter", 0.3))
        split_scale_shrink = float(opt_params.get("split_scale_shrink", 0.6))
        split_opacity_shrink = float(opt_params.get("split_opacity_shrink", 0.5))
        split_min_scale = float(opt_params.get("split_min_scale", 1e-4))
        split_max_scale = float(opt_params.get("split_max_scale", 0.2))
        split_cooldown = int(opt_params.get("split_cooldown", 3))

        prop_names, vertices, header_lines, tensors = self._load_ply_tensors(gs_path)
        means = tensors["means"]
        scales = tensors["scales"]
        rotations = tensors["rotations"]
        opacity = tensors["opacity"]
        sh_dc = tensors["sh_dc"]

        indices_arr = np.asarray(indices, dtype=np.int64)
        indices_arr = indices_arr[(indices_arr >= 0) & (indices_arr < means.shape[0])]
        if indices_arr.size == 0:
            raise ValueError("Selection indices are out of bounds.")

        sel = torch.as_tensor(indices_arr, device=self.device, dtype=torch.long)
        selected_mask_np = np.zeros((means.shape[0],), dtype=bool)
        selected_mask_np[indices_arr] = True
        center_target = np.zeros(3, dtype=np.float32)

        return {
            "payload": payload,
            "selection": selection,
            "indices_arr": indices_arr,
            "sel": sel,
            "selected_mask_np": selected_mask_np,
            "paths": paths,
            "gs_path": gs_path,
            "output_dir": output_dir,
            "workspace_root": workspace_root,
            "width": width,
            "height": height,
            "fov_deg": fov_deg,
            "up_axis": up_axis,
            "camera_overrides": camera_overrides,
            "frame_token": frame_token,
            "gt_image_path": gt_image_path,
            "eval_gt_root": eval_gt_root,
            "depth_margin_ratio": depth_margin_ratio,
            "min_depth_margin": min_depth_margin,
            "opt_params": opt_params,
            "iters": iters,
            "lr_color": lr_color,
            "lr_opacity": lr_opacity,
            "lr_scale": lr_scale,
            "lr_rotation": lr_rotation,
            "lr_means": lr_means,
            "lr_final_factor": lr_final_factor,
            "lambda_dssim": lambda_dssim,
            "full_loss_weight": full_loss_weight,
            "optimize_means": optimize_means,
            "optimize_rotation": optimize_rotation,
            "unlock_ratio": unlock_ratio,
            "video_samples": video_samples,
            "render_video_only": render_video_only,
            "mask_dilate": mask_dilate,
            "split_enable": split_enable,
            "split_every": split_every,
            "split_start": split_start,
            "split_max_new": split_max_new,
            "split_topk": split_topk,
            "split_score": split_score,
            "split_thresh": split_thresh,
            "split_jitter": split_jitter,
            "split_scale_shrink": split_scale_shrink,
            "split_opacity_shrink": split_opacity_shrink,
            "split_min_scale": split_min_scale,
            "split_max_scale": split_max_scale,
            "split_cooldown": split_cooldown,
            "cam_info": cam_info,
            "prop_names": prop_names,
            "vertices": vertices,
            "header_lines": header_lines,
            "means": means,
            "scales": scales,
            "rotations": rotations,
            "opacity": opacity,
            "sh_dc": sh_dc,
            "center_target": center_target,
        }

    def _collect_visible_views(self, ctx: Dict) -> Tuple[List[Dict], Dict, str, Dict[str, np.ndarray]]:
        eval_gt_root = ctx["eval_gt_root"]
        frame_token = ctx.get("frame_token")
        means = ctx["means"]
        scales = ctx["scales"]
        rotations = ctx["rotations"]
        opacity = ctx["opacity"]
        sh_dc = ctx["sh_dc"]
        selected_mask_np = ctx["selected_mask_np"]
        fov_deg = float(ctx["fov_deg"])
        up_axis = ctx.get("up_axis", "y")
        print("collect visble up axis:", up_axis)
        camera_overrides = ctx.get("camera_overrides", {}) or {}

        centers_np = means.detach().cpu().numpy()
        center_target = ctx.get("center_target", np.zeros(3, dtype=np.float32))
        width = ctx["width"]
        height = ctx["height"]

        candidate_views: List[Tuple[str, Path]] = []
        if eval_gt_root.exists() and frame_token:
            for sub in sorted(eval_gt_root.iterdir()):
                if sub.is_dir():
                    img_path = sub / f"{frame_token}.png"
                    if img_path.exists():
                        candidate_views.append((sub.name, img_path))
                elif sub.is_file() and sub.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    candidate_views.append((sub.stem, sub))

        if not candidate_views:
            raise ValueError("No valid eval_gt view with ROI coverage found for the given frame_token.")

        visible_views: List[Dict] = []
        scales_np = scales.detach().cpu().numpy()
        rotations_np = rotations.detach().cpu().numpy()
        opacity_np = opacity.detach().cpu().numpy()
        sh_dc_np = sh_dc.detach().cpu().numpy()

        for view_name, img_path in candidate_views:
            override_cam = camera_overrides.get(view_name) or {}
            eye = np.asarray(override_cam.get("eye"), dtype=np.float32) if "eye" in override_cam else None
            wxyz = np.asarray(override_cam.get("wxyz"), dtype=np.float32) if "wxyz" in override_cam else None
            print(f"[info] checking view '{view_name}' for visibility...")
            print(f"       eye override: {eye}, wxyz override: {wxyz}")
            if eye is None or wxyz is None:
                eye, wxyz = self._camera_pose_for_view(view_name, center_target, depth=1.5, up_axis=up_axis)

            rotation_wc = self._camera_to_world_rotation(wxyz.tolist())
            gt_img = load_image(img_path, flip_udlr=False)
            h_gt, w_gt = gt_img.shape[:2]
            render_full, mask_render, mask_soft = self._render_visibility_mask(
                centers_np,
                scales_np,
                rotations_np,
                opacity_np,
                sh_dc_np,
                rotation_wc,
                eye,
                w_gt,
                h_gt,
                fov_deg,
                selected_mask_np,
            )



            mask_binary = (mask_render[..., 0] > 0.05).astype(np.uint8)
            coverage = float(mask_binary.sum())
            if coverage <= 0.0:
                continue
            ys, xs = np.nonzero(mask_binary)
            if ys.size > 0 and xs.size > 0:
                bbox_view = {"x0": int(xs.min()), "x1": int(xs.max() + 1), "y0": int(ys.min()), "y1": int(ys.max() + 1)}
            else:
                bbox_view = {"x0": 0, "x1": w_gt, "y0": 0, "y1": h_gt}
            visible_views.append(
                {
                    "name": view_name,
                    "img_path": img_path,
                    "coverage": coverage,
                    "mask": mask_soft,
                    "bbox": bbox_view,
                    "eye": eye.astype(np.float32),
                    "wxyz": wxyz.astype(np.float32),
                    "size": (w_gt, h_gt),
                }
            )

        if not visible_views:
            raise ValueError("Visibility masks are empty across all views; nothing to optimize.")

        visible_views.sort(key=lambda v: v.get("coverage", 0.0), reverse=True)
        best_view = visible_views[0]
        width, height = best_view["size"]
        rotation_wc_best = self._camera_to_world_rotation(best_view["wxyz"].tolist())
        position_w_best = best_view["eye"]
        bbox_best = best_view["bbox"]

        view_tensors: List[Dict] = []
        for view in visible_views:
            w_v, h_v = view["size"]
            if (w_v, h_v) != (width, height):
                raise ValueError(
                    "All GT views must share the same resolution; "
                    f"got {(w_v, h_v)} vs {(width, height)} for view {view['name']}."
                )
            gt_img_np_view = load_image(view["img_path"], flip_udlr=False)
            gt_img_np_view = np.rot90(gt_img_np_view, 2)
            gt_t_view = torch.from_numpy(gt_img_np_view.astype(np.float32) / 255.0).to(self.device).permute(2, 0, 1)
            mask_t_view = torch.from_numpy(view["mask"][..., 0].astype(np.float32)).to(self.device).unsqueeze(0)
            mask_sum_view = float(mask_t_view.sum().item())
            if mask_sum_view <= 0:
                continue
            print(f"[info] using view '{view['name']}' with coverage {view['coverage']:.1f} pixels for optimization.")
            print(f"       bbox: {view['bbox']}, eye: {view['eye']}, wxyz: {view['wxyz']}")
            view_tensors.append(
                {
                    "name": view["name"],
                    "rotation_wc": self._camera_to_world_rotation(view["wxyz"].tolist()),
                    "position_w": view["eye"],
                    "bbox": view["bbox"],
                    "gt": gt_t_view,
                    "mask": mask_t_view,
                    "gt_np": gt_img_np_view,
                }
            )

        if not view_tensors:
            raise ValueError("Visibility masks are empty across all views; nothing to optimize.")

        best_view_data = next(v for v in view_tensors if v["name"] == best_view["name"])
        ctx["width"], ctx["height"] = width, height

        return view_tensors, best_view_data, best_view["name"], {
            "rotation_wc_best": rotation_wc_best,
            "position_w_best": position_w_best,
            "bbox_best": bbox_best,
        }

    def _render_interp_video_only(
        self,
        ctx: Dict,
        best_view: Dict,
        target_view: str,
    ) -> Dict:
        means_np = ctx["means"].detach().cpu().numpy()
        scales_np = ctx["scales"].detach().cpu().numpy()
        rotations_np = ctx["rotations"].detach().cpu().numpy()
        opacity_np = ctx["opacity"].detach().cpu().numpy()
        sh_dc_np = ctx["sh_dc"].detach().cpu().numpy()

        width = ctx["width"]
        height = ctx["height"]
        fov_deg = float(ctx["fov_deg"])
        up_axis = ctx.get("up_axis", "y")
        video_samples = ctx.get("video_samples", 16)
        frame_token = ctx.get("frame_token")
        workspace_root = ctx["workspace_root"]
        cam_info = ctx.get("cam_info", {})
        center_target = ctx.get("center_target", np.zeros(3, dtype=np.float32))

        start_eye, start_wxyz = self._camera_pose_for_view("input", center_target, depth=1.5, up_axis=up_axis)
        cam_end_eye = cam_info.get("position") if isinstance(cam_info, dict) else None
        cam_end_wxyz = cam_info.get("wxyz") if isinstance(cam_info, dict) else None
        end_eye = (
            np.asarray(cam_end_eye, dtype=np.float32)
            if cam_end_eye is not None
            else best_view["eye"]
        )
        end_wxyz = (
            np.asarray(cam_end_wxyz, dtype=np.float32)
            if cam_end_wxyz is not None
            else best_view["wxyz"]
        )

        def render_fn(rotation_wc: np.ndarray, position_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return self._render_full_and_mask_numpy(
                means_np,
                scales_np,
                rotations_np,
                opacity_np,
                sh_dc_np,
                rotation_wc,
                position_w,
                width,
                height,
                fov_deg,
                ctx["selected_mask_np"],
            )

        video_root = workspace_root / f"video_fix_{frame_token or 'default'}"
        traj_info = self._render_interp_video_frames(
            start_eye,
            start_wxyz,
            end_eye,
            end_wxyz,
            video_root,
            render_fn,
            video_samples,
            frame_token,
            workspace_root,
            ctx.get("gt_image_path"),
            center_target,
            up_axis,
            width,
            height,
            fov_deg,
        )
        print("video_root:", video_root)
        return {
            "status": "success",
            "mode": "render_video_only",
            "video_root": str(video_root),
            "target_view": target_view,
            "frames": traj_info["frames"],
            "masks": traj_info["masks"],
            "cameras": traj_info["cameras"],
            "meta": traj_info["meta"],
        }

    def _write_ply_subset(
        self,
        header_lines,
        vertices,
        indices_arr,
        means_sel_cur: torch.Tensor,
        scales_sel_cur: torch.Tensor,
        rotations_sel_cur: torch.Tensor,
        opacity_sel_cur: torch.Tensor,
        sh_dc_sel_cur: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        sh_dc: torch.Tensor,
        out_path: Path,
    ) -> None:
        write_ply_subset_aligned(
            header_lines,
            vertices,
            indices_arr,
            means_sel_cur.detach().cpu().numpy(),
            scales_sel_cur.detach().cpu().numpy(),
            rotations_sel_cur.detach().cpu().numpy(),
            opacity_sel_cur.detach().cpu().numpy(),
            sh_dc_sel_cur.detach().cpu().numpy(),
            means.detach().cpu().numpy(),
            scales.detach().cpu().numpy(),
            rotations.detach().cpu().numpy(),
            opacity.detach().cpu().numpy(),
            sh_dc.detach().cpu().numpy(),
            out_path,
        )

    def _render_full_numpy_view(
        self,
        rotation_wc_best: np.ndarray,
        position_w_best: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        sh_dc: torch.Tensor,
        sel: torch.Tensor,
        means_sel_cur: torch.Tensor,
        scales_sel_cur: torch.Tensor,
        rotations_sel_cur: torch.Tensor,
        opacity_sel_cur: torch.Tensor,
        sh_dc_sel_cur: torch.Tensor,
    ) -> np.ndarray:
        means_all = self._combine_params(means, means_sel_cur, sel).detach().cpu().numpy()
        scales_all = self._combine_params(scales, scales_sel_cur, sel).detach().cpu().numpy()
        rotations_all = self._combine_params(rotations, rotations_sel_cur, sel).detach().cpu().numpy()
        opacity_all = self._combine_params(opacity, opacity_sel_cur, sel).detach().cpu().numpy()
        sh_all = self._combine_params(sh_dc, sh_dc_sel_cur, sel).detach().cpu().numpy()
        return self._render_full_simple(
            means_all,
            scales_all,
            rotations_all,
            opacity_all,
            sh_all,
            rotation_wc_best,
            position_w_best,
            width,
            height,
            fov_deg,
        )



    @torch.no_grad()
    def _split_gaussians_once(
        self,
        means_sel: torch.Tensor,
        scales_sel: torch.Tensor,
        rotations_sel: torch.Tensor,
        opacity_sel: torch.Tensor,
        sh_dc_sel: torch.Tensor,
        grad_means: torch.Tensor | None,
        grad_scales: torch.Tensor | None,
        grad_opacity: torch.Tensor | None,
        split_score: str,
        split_thresh: float,
        split_topk: int,
        split_max_new: int,
        split_jitter: float,
        split_scale_shrink: float,
        split_opacity_shrink: float,
        split_min_scale: float,
        split_max_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Split selected parents into two children while retaining each parent.

        Returns the expanded tensors, the number of children, and parent indices.
        """

        # Fall back to scale and opacity gradients when means are frozen.
        if split_score == "means_grad" and (grad_means is not None):
            score = grad_means.norm(dim=1)  # [N]
        else:
            # Combine scale and opacity gradients into a split score.
            s = 0.0
            if grad_scales is not None:
                s = s + grad_scales.norm(dim=1)  # [N]
            if grad_opacity is not None:
                s = s + grad_opacity.view(-1).abs()  # [N]
            score = s

        if score is None:
            parents = torch.empty((0,), device=means_sel.device, dtype=torch.long)
            return means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel, 0, parents

        # Apply the threshold before selecting the highest-scoring parents.
        cand = torch.where(score > split_thresh)[0]
        if cand.numel() == 0:
            parents = torch.empty((0,), device=means_sel.device, dtype=torch.long)
            return means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel, 0, parents

        # topk
        cand_score = score[cand]
        k = min(int(split_topk), int(cand.numel()), int(split_max_new))
        print("top k:", k, "from candidates:", cand.numel())
        topk_idx = torch.topk(cand_score, k=k, largest=True).indices
        parents = cand[topk_idx]  # [k]

        # Create two children for every selected parent.
        mu = means_sel[parents]                # [k,3]
        sc_log = scales_sel[parents]           # [k,3] (log-space)
        ro = rotations_sel[parents]            # [k,4]
        op_logit = opacity_sel[parents].view(-1)  # [k] (logit-space)
        rgb = sh_dc_sel[parents]               # [k,3]

        # Scale random position jitter by each parent's linear scale.
        sc_lin = torch.exp(sc_log)
        sc_mean = sc_lin.mean(dim=1, keepdim=True).clamp_min(split_min_scale)
        eps = torch.randn_like(mu) * (sc_mean * split_jitter)

        child_mu1 = mu + eps
        child_mu2 = mu - eps

        # scale: operate in linear space, then convert back to log-space
        child_sc_lin = (sc_lin * split_scale_shrink).clamp(min=split_min_scale, max=split_max_scale)
        child_sc = torch.log(child_sc_lin.clamp_min(1e-12))

        # opacity: operate in alpha space, then convert back to logit-space
        alpha = torch.sigmoid(op_logit)
        child_alpha = (alpha * split_opacity_shrink).clamp(1e-6, 1.0 - 1e-6)
        child_op = torch.log(child_alpha / (1.0 - child_alpha))

        # Children inherit rotation and color.
        child_ro1 = ro
        child_ro2 = ro
        child_rgb1 = rgb
        child_rgb2 = rgb

        # Retain parents and append their children.
        means2 = torch.cat([means_sel, child_mu1, child_mu2], dim=0)
        scales2 = torch.cat([scales_sel, child_sc, child_sc], dim=0)
        rots2 = torch.cat([rotations_sel, child_ro1, child_ro2], dim=0)
        op2 = torch.cat([opacity_sel.view(-1), child_op, child_op], dim=0)
        sh2 = torch.cat([sh_dc_sel, child_rgb1, child_rgb2], dim=0)

        num_added = int(2 * parents.numel())
        return means2, scales2, rots2, op2, sh2, num_added, parents

    def _build_optimizer(
        self,
        means_sel: torch.Tensor,
        scales_sel: torch.Tensor,
        rotations_sel: torch.Tensor,
        opacity_sel: torch.Tensor,
        sh_dc_sel: torch.Tensor,
        stage_means_added: bool,
        stage_rot_added: bool,
        lr_color: float,
        lr_opacity: float,
        lr_scale: float,
        lr_means: float,
        lr_rotation: float,
    ) -> torch.optim.Optimizer:
        param_groups = [
            {"params": sh_dc_sel, "lr": lr_color, "base_lr": lr_color},
            {"params": opacity_sel, "lr": lr_opacity, "base_lr": lr_opacity},
            {"params": scales_sel, "lr": lr_scale, "base_lr": lr_scale},
        ]
        if stage_means_added:
            param_groups.append({"params": means_sel, "lr": lr_means, "base_lr": lr_means})
        if stage_rot_added:
            param_groups.append({"params": rotations_sel, "lr": lr_rotation, "base_lr": lr_rotation})

        opt = torch.optim.Adam(param_groups)
        return opt



    def _optimize_roi(
        self,
        ctx: Dict,
        view_tensors: List[Dict],
        best_view: str,
        best_view_data: Dict,
        rotation_wc_best: np.ndarray,
        position_w_best: np.ndarray,
        bbox_best: Dict[str, int],
    ) -> Dict:
        lambda_dssim = ctx["lambda_dssim"]
        full_loss_weight = ctx["full_loss_weight"]
        lr_color = ctx["lr_color"]
        lr_opacity = ctx["lr_opacity"]
        lr_scale = ctx["lr_scale"]
        lr_means = ctx["lr_means"]
        lr_rotation = ctx["lr_rotation"]
        lr_final_factor = ctx["lr_final_factor"]
        optimize_means = ctx["optimize_means"]
        optimize_rotation = ctx["optimize_rotation"]
        iters = ctx["iters"]
        width = ctx["width"]
        height = ctx["height"]
        fov_deg = ctx["fov_deg"]
        up_axis = ctx.get("up_axis", "y")
        video_samples = ctx.get("video_samples", 16)
        frame_token = ctx.get("frame_token")

        # ---- split hyperparams (add near ctx reads) ----
        split_enable = ctx.get("split_enable", True)
        split_every = int(ctx.get("split_every", 20))
        split_start = int(ctx.get("split_start", int(iters * 0.2)))
        split_max_new = int(ctx.get("split_max_new", 512))
        split_topk = int(ctx.get("split_topk", 128))
        split_score = ctx.get("split_score", "scale_opacity_grad")   # or "means_grad"
        split_thresh = float(ctx.get("split_thresh", 1e-3))
        split_jitter = float(ctx.get("split_jitter", 0.3))
        split_scale_shrink = float(ctx.get("split_scale_shrink", 0.6)) # child scale = parent * shrink
        split_opacity_shrink = float(ctx.get("split_opacity_shrink", 0.5)) # child opacity = parent * shrink
        split_min_scale = float(ctx.get("split_min_scale", 1e-4))
        split_max_scale = float(ctx.get("split_max_scale", 0.2))
        mask_dilate = int(ctx.get("mask_dilate", 0))
        # hard clips (linear domain) to prevent exploding/giant gaussians during optimization
        scale_clip_min = float(ctx.get("scale_clip_min", split_min_scale))
        scale_clip_max = float(ctx.get("scale_clip_max", split_max_scale))
        opacity_clip_min = float(ctx.get("opacity_clip_min", 1e-4))  # alpha min
        opacity_clip_max = float(ctx.get("opacity_clip_max", 0.999))  # alpha max
        sh_clip = float(ctx.get("sh_clip", float(0.5 / SH_C0)))  # bounds DC so RGB stays in [0,1] without saturation
        lambda_scale_reg = float(ctx.get("lambda_scale_reg", 0.0))
        scale_reg_max = float(ctx.get("scale_reg_max", scale_clip_max))
        lambda_sh_reg = float(ctx.get("lambda_sh_reg", 0.0))
        lambda_opacity_reg = float(ctx.get("lambda_opacity_reg", 0.0))


        means = ctx["means"]
        scales = ctx["scales"]
        rotations = ctx["rotations"]
        opacity = ctx["opacity"]
        sh_dc = ctx["sh_dc"]
        sel = ctx["sel"]
        indices_arr = ctx["indices_arr"]
        vertices = ctx["vertices"]
        header_lines = ctx["header_lines"]
        output_dir = ctx["output_dir"]
        workspace_root = ctx["workspace_root"]
        center_target = ctx.get("center_target", np.zeros(3, dtype=np.float32))
        cam_info = ctx.get("cam_info", {})
        render_video_only = ctx.get("render_video_only", False)

        best_gt_np = best_view_data["gt_np"]
        best_gt_t = best_view_data["gt"]
        best_mask_t = best_view_data["mask"]
        mask_sum = float(best_mask_t.sum().item())
        if mask_sum <= 0:
            raise ValueError("Visibility mask is empty in the best view; nothing to optimize.")

        means_sel = means[sel].clone().requires_grad_(False)
        scales_sel = scales[sel].clone().requires_grad_(True)
        opacity_sel = opacity[sel].clone().requires_grad_(True)
        sh_dc_sel = sh_dc[sel].clone().requires_grad_(True)
        rotations_sel = rotations[sel].clone().requires_grad_(False)

        optimizer = torch.optim.Adam(
            [
                {"params": sh_dc_sel, "lr": lr_color},
                {"params": opacity_sel, "lr": lr_opacity},
                {"params": scales_sel, "lr": lr_scale},
            ]
        )
        stage_means_added = False
        stage_rot_added = False

        warmup_ratio = 0.05
        lr_final = max(lr_final_factor, 1e-4)
        for g in optimizer.param_groups:
            g["base_lr"] = g["lr"]

        def _apply_lr_schedule(step: int) -> None:
            warmup_steps = max(int(iters * warmup_ratio), 1)
            if step < warmup_steps:
                scale = float(step + 1) / float(warmup_steps)
            else:
                prog = float(step - warmup_steps) / float(max(iters - warmup_steps, 1))
                scale = lr_final + 0.5 * (1.0 - lr_final) * (1.0 + math.cos(math.pi * prog))
            for g in optimizer.param_groups:
                base_lr = g.get("base_lr", g["lr"])
                g["base_lr"] = base_lr
                g["lr"] = base_lr * scale

        bbox_for_draw = bbox_best.copy()
        reference_vis = draw_bbox_outline(best_gt_np, bbox_for_draw, color=(0, 255, 0), thickness=3)
        reference_vis = np.rot90(reference_vis, 2)

        render_before_np = self._render_full_numpy_view(
            rotation_wc_best,
            position_w_best,
            width,
            height,
            fov_deg,
            means,
            scales,
            rotations,
            opacity,
            sh_dc,
            sel,
            means_sel,
            scales_sel,
            rotations_sel,
            opacity_sel,
            sh_dc_sel,
        )
        render_before_vis = draw_bbox_outline(
            (np.clip(render_before_np, 0, 1) * 255.0).astype(np.uint8), bbox_for_draw, color=(0, 255, 0), thickness=3
        )
        render_before_vis = np.rot90(render_before_vis, 2)

        render_before_path = output_dir / "render_before.png"
        iio.imwrite(render_before_path, render_before_vis)

        oracle_path = output_dir / "oracle_patch.png"
        iio.imwrite(oracle_path, reference_vis)

        check_dir = ROOT / "tmp" / "check_loss"
        check_dir.mkdir(parents=True, exist_ok=True)
        preview_path = check_dir / "progress.png"
        loss_render_path = check_dir / "loss_render.png"
        loss_target_path = check_dir / "loss_target.png"
        loss_mask_path = check_dir / "loss_mask.png"
        live_ply_path = output_dir / "gs_live.ply"

        blank = np.zeros((height, width, 3), dtype=np.uint8)
        blank_strip = np.concatenate([blank for _ in range(max(1, len(view_tensors)))], axis=1) if view_tensors else blank
        iio.imwrite(preview_path, blank)
        iio.imwrite(loss_render_path, blank_strip)
        iio.imwrite(loss_target_path, blank_strip)
        iio.imwrite(loss_mask_path, blank_strip)
        self._write_ply_subset(
            header_lines,
            vertices,
            indices_arr,
            means_sel,
            scales_sel,
            rotations_sel,
            opacity_sel,
            sh_dc_sel,
            means,
            scales,
            rotations,
            opacity,
            sh_dc,
            live_ply_path,
        )

        exp_id = ctx["payload"].get("experiment_id", f"exp_{int(time.time())}")
        progress_entry = {
            "status": "running",
            "iter": 0,
            "loss": None,
            "iter_max": iters,
            "render_path": str(render_before_path),
            "oracle_path": str(oracle_path),
            "progress_render": str(preview_path),
            "progress_target": str(loss_target_path),
            "live_ply": str(live_ply_path),
        }
        self.progress[exp_id] = progress_entry

        loss_curve: List[float] = []
        start = time.time()
        last_update = start
        unlock_ratio = float(ctx.get("unlock_ratio", 0.35))
        unlock_ratio = float(min(max(unlock_ratio, 0.0), 1.0))
        unlock_iter = int(iters * unlock_ratio)

        # ---- split runtime state ----
        last_split_it = -10**9
        split_cooldown = int(ctx.get("split_cooldown", 3))
        roi_debug_dir = configured_debug_dir("optimizer/roi", create=True)


        video_only_result = self._maybe_render_video_only(ctx, best_view_data, rotation_wc_best, position_w_best, best_view)
        if video_only_result is not None:
            return video_only_result

        for it in tqdm(range(max(iters, 1))):
            if optimize_means and not stage_means_added and it >= unlock_iter:
                means_sel.requires_grad_(True)
                optimizer.add_param_group({"params": means_sel, "lr": lr_means, "base_lr": lr_means})
                stage_means_added = True
            if optimize_rotation and not stage_rot_added and it >= unlock_iter:
                rotations_sel.requires_grad_(True)
                optimizer.add_param_group({"params": rotations_sel, "lr": lr_rotation, "base_lr": lr_rotation})
                stage_rot_added = True

            self._apply_lr_schedule(optimizer, it, iters, lr_final, warmup_ratio)

            optimizer.zero_grad()
            total_loss = 0.0
            view_count = 0
            best_render_cur: torch.Tensor | None = None
            render_strips: List[np.ndarray] = []
            target_strips: List[np.ndarray] = []
            mask_strips: List[np.ndarray] = []
            for view_data in view_tensors:
                render_cur = self._render_full_for_view(
                    view_data,
                    means,
                    scales,
                    rotations,
                    opacity,
                    sh_dc,
                    sel,
                    means_sel,
                    scales_sel,
                    rotations_sel,
                    opacity_sel,
                    sh_dc_sel,
                    width,
                    height,
                    fov_deg,
                )
                if view_data["name"] == best_view:
                    best_render_cur = render_cur

                render_cur_chw = render_cur.permute(2, 0, 1)
                bbox_v = view_data["bbox"]
                x0, x1 = int(bbox_v["x0"]), int(bbox_v["x1"])
                y0, y1 = int(bbox_v["y0"]), int(bbox_v["y1"])
                x0 = max(0, min(x0, width - 1))
                x1 = max(x0 + 1, min(x1, width))
                y0 = max(0, min(y0, height - 1))
                y1 = max(y0 + 1, min(y1, height))

                render_patch = render_cur_chw[:, y0:y1, x0:x1]
                gt_patch = view_data["gt"][:, y0:y1, x0:x1]
                mask_patch = view_data["mask"][:, y0:y1, x0:x1]
                if mask_dilate > 0:
                    kernel = int(mask_dilate) * 2 + 1
                    mask_patch = F.max_pool2d(
                        mask_patch.unsqueeze(0),
                        kernel_size=kernel,
                        stride=1,
                        padding=kernel // 2,
                    ).squeeze(0)

                mask_sum_patch = mask_patch.sum().clamp_min(1e-6)
                if roi_debug_dir is not None and it % 200 == 0:
                    render_full_vis = (
                        (render_cur_chw.clamp(0, 1) * 255.0)
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    gt_full_vis = (
                        (view_data["gt"].clamp(0, 1) * 255.0)
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    render_patch_vis = (
                        (render_patch.clamp(0, 1) * 255.0)
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    gt_patch_vis = (
                        (gt_patch.clamp(0, 1) * 255.0)
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    mask_patch_vis = (
                        (mask_patch.clamp(0, 1) * 255.0)
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    iio.imwrite(roi_debug_dir / "render_full.png", render_full_vis)
                    iio.imwrite(roi_debug_dir / "gt_full.png", gt_full_vis)
                    iio.imwrite(roi_debug_dir / "render_patch.png", render_patch_vis)
                    iio.imwrite(roi_debug_dir / "gt_patch.png", gt_patch_vis)
                    iio.imwrite(roi_debug_dir / "mask_patch.png", mask_patch_vis)

                l1_term = ((render_patch - gt_patch).abs() * mask_patch).sum() / mask_sum_patch
                ssim_term = 0.0
                if lambda_dssim > 0.0:
                    render_ssim = (render_patch * mask_patch).unsqueeze(0)
                    gt_ssim = (gt_patch * mask_patch).unsqueeze(0)
                    ssim_val = _ssim_torch(render_ssim, gt_ssim)
                    ssim_term = 1.0 - ssim_val
                loss_roi = (1.0 - lambda_dssim) * l1_term + lambda_dssim * ssim_term

                l1_full = (render_cur_chw - view_data["gt"]).abs().mean()
                ssim_full_term = 0.0
                if lambda_dssim > 0.0:
                    ssim_full = _ssim_torch(render_cur_chw.unsqueeze(0), view_data["gt"].unsqueeze(0))
                    ssim_full_term = 1.0 - ssim_full
                loss_full = (1.0 - lambda_dssim) * l1_full + lambda_dssim * ssim_full_term

                loss_view = loss_roi + full_loss_weight * loss_full
                total_loss = total_loss + loss_view
                view_count += 1

                render_cur_np = render_cur.detach().clamp(0, 1).cpu().numpy()
                target_np = (view_data["gt"].permute(1, 2, 0)).detach().clamp(0, 1).cpu().numpy()
                bbox_v = view_data["bbox"]
                target_vis = draw_bbox_outline((target_np * 255.0).astype(np.uint8), bbox_v, color=(0, 255, 0), thickness=3)
                mask_np = view_data["mask"].squeeze(0).detach().clamp(0, 1).cpu().numpy()
                mask_vis = np.repeat((mask_np * 255.0).astype(np.uint8)[..., None], 3, axis=2)
                render_strips.append((render_cur_np * 255.0).astype(np.uint8))
                target_strips.append(target_vis)
                mask_strips.append(mask_vis)

            loss = total_loss / max(view_count, 1)
            if lambda_scale_reg > 0.0:
                # penalize overly large linear scales (log-space params live in `scales_sel`)
                scales_lin = torch.exp(scales_sel)
                excess = torch.relu(scales_lin - scale_reg_max)
                loss = loss + lambda_scale_reg * (excess * excess).mean()
            if lambda_sh_reg > 0.0:
                loss = loss + lambda_sh_reg * (sh_dc_sel * sh_dc_sel).mean()
            if lambda_opacity_reg > 0.0:
                # encourage sparsity / avoid opaque floaters (penalize high alpha)
                alpha = torch.sigmoid(opacity_sel.view(-1))
                loss = loss + lambda_opacity_reg * alpha.mean()
            loss.backward()

            # Clip gradients to stabilize optimization around split operations.
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in (g["params"] if isinstance(g["params"], list) else [g["params"]])], 1.0)

            optimizer.step()
            # hard clamps to avoid "giant GS" degeneracy (operate in log-space / logit-space)
            with torch.no_grad():
                smin = math.log(max(scale_clip_min, 1e-12))
                smax = math.log(max(scale_clip_max, 1e-12))
                scales_sel.clamp_(min=smin, max=smax)
                a_min = min(max(opacity_clip_min, 1e-6), 1.0 - 1e-6)
                a_max = min(max(opacity_clip_max, 1e-6), 1.0 - 1e-6)
                if a_max < a_min:
                    a_min, a_max = a_max, a_min
                omin = math.log(a_min / (1.0 - a_min))
                omax = math.log(a_max / (1.0 - a_max))
                opacity_sel.view(-1).clamp_(min=omin, max=omax)
                if sh_clip > 0:
                    sh_dc_sel.clamp_(min=-sh_clip, max=sh_clip)
            loss_curve.append(float(loss.detach().cpu().item()))

            # GS split: after step
            if split_enable and (it >= split_start) and (split_every > 0) and (it % split_every == 0):
                # Gradients remain available after optimizer.step() for split scoring.
                grad_means = means_sel.grad.detach().clone() if (means_sel.requires_grad and means_sel.grad is not None) else None
                grad_scales = scales_sel.grad.detach().clone() if (scales_sel.grad is not None) else None
                grad_opacity = opacity_sel.grad.detach().clone() if (opacity_sel.grad is not None) else None
                print(f"[SPLIT] it={it} grad_means_norm={(grad_means.norm(dim=1) if grad_means is not None else 'N/A')}")
                # Split returns new parameter tensors.
                means2, scales2, rots2, op2, sh2, num_added, parents = self._split_gaussians_once(
                    means_sel,
                    scales_sel,
                    rotations_sel,
                    opacity_sel,
                    sh_dc_sel,
                    grad_means,
                    grad_scales,
                    grad_opacity,
                    split_score=split_score,
                    split_thresh=split_thresh,
                    split_topk=split_topk,
                    split_max_new=split_max_new,
                    split_jitter=split_jitter,
                    split_scale_shrink=split_scale_shrink,
                    split_opacity_shrink=split_opacity_shrink,
                    split_min_scale=split_min_scale,
                    split_max_scale=split_max_scale,
                )

                if num_added > 0:
                    # Extend all index-aligned state for the new children:
                    # - full tensors: means/scales/rotations/opacity/sh_dc
                    # - selection: sel / indices_arr / selected_mask_np
                    # - PLY vertices, to keep write-back indices aligned.
                    old_total = int(means.shape[0])
                    num_added = int(num_added)
                    if num_added > 0:
                        sel_before = sel
                        means_new = means2[-num_added:].detach()
                        scales_new = scales2[-num_added:].detach()
                        rots_new = rots2[-num_added:].detach()
                        op_new = op2[-num_added:].detach()
                        sh_new = sh2[-num_added:].detach()

                        means = torch.cat([means, means_new], dim=0)
                        scales = torch.cat([scales, scales_new], dim=0)
                        rotations = torch.cat([rotations, rots_new], dim=0)
                        opacity = torch.cat([opacity, op_new], dim=0)
                        sh_dc = torch.cat([sh_dc, sh_new], dim=0)

                        new_indices_arr = np.arange(old_total, old_total + num_added, dtype=np.int64)
                        indices_arr = np.concatenate([indices_arr, new_indices_arr], axis=0)
                        sel = torch.cat(
                            [sel, torch.as_tensor(new_indices_arr, device=self.device, dtype=torch.long)], dim=0
                        )
                        selected_mask_np = ctx.get("selected_mask_np")
                        if selected_mask_np is not None:
                            selected_mask_np = np.concatenate(
                                [selected_mask_np, np.ones((num_added,), dtype=bool)], axis=0
                            )
                            ctx["selected_mask_np"] = selected_mask_np

                        if vertices is not None and parents.numel() > 0:
                            parent_full = sel_before[parents].detach().cpu().numpy().astype(np.int64, copy=False)
                            parent_rows = vertices[parent_full]
                            child_rows = np.concatenate([parent_rows.copy(), parent_rows.copy()], axis=0)
                            vertices = np.concatenate([vertices, child_rows], axis=0)

                        # keep ctx in sync for downstream helpers / final return
                        ctx["means"] = means
                        ctx["scales"] = scales
                        ctx["rotations"] = rotations
                        ctx["opacity"] = opacity
                        ctx["sh_dc"] = sh_dc
                        ctx["indices_arr"] = indices_arr
                        ctx["sel"] = sel
                        ctx["vertices"] = vertices

                    # Splitting creates new tensors, so restore their gradient state.
                    # Keep staged means/rotation unlocking consistent with the optimizer.
                    means_sel = means2.detach().clone()
                    scales_sel = scales2.detach().clone()
                    rotations_sel = rots2.detach().clone()
                    opacity_sel = op2.detach().clone()
                    sh_dc_sel = sh2.detach().clone()

                    means_sel.requires_grad_(bool(stage_means_added))
                    rotations_sel.requires_grad_(bool(stage_rot_added))
                    scales_sel.requires_grad_(True)
                    opacity_sel.requires_grad_(True)
                    sh_dc_sel.requires_grad_(True)

                    if int(means_sel.shape[0]) != int(sel.numel()):
                        raise RuntimeError(
                            f"[SPLIT] bookkeeping mismatch: means_sel N={int(means_sel.shape[0])} "
                            f"but sel N={int(sel.numel())}. This indicates split added gaussians "
                            "but selection indices were not updated consistently."
                        )
                    if int(means.shape[0]) <= int(sel.max().item()):
                        raise RuntimeError(
                            f"[SPLIT] bookkeeping mismatch: full means N={int(means.shape[0])} "
                            f"but sel.max={int(sel.max().item())}."
                        )

                    # Rebuild the optimizer for the expanded tensors.
                    optimizer = self._build_optimizer(
                        means_sel,
                        scales_sel,
                        rotations_sel,
                        opacity_sel,
                        sh_dc_sel,
                        stage_means_added=stage_means_added,
                        stage_rot_added=stage_rot_added,
                        lr_color=lr_color,
                        lr_opacity=lr_opacity,
                        lr_scale=lr_scale,
                        lr_means=lr_means,
                        lr_rotation=lr_rotation,
                    )

                    # Reapply the schedule using base_lr from _build_optimizer.
                    self._apply_lr_schedule(optimizer, it, iters, lr_final, warmup_ratio)

                    print(f"[SPLIT] it={it} added={num_added} new_N={means_sel.shape[0]}")
            now = time.time()
            if now - last_update >= 1.5 and best_render_cur is not None:
                render_full_np = self._render_full_numpy_view(
                    rotation_wc_best,
                    position_w_best,
                    width,
                    height,
                    fov_deg,
                    means,
                    scales,
                    rotations,
                    opacity,
                    sh_dc,
                    sel,
                    means_sel,
                    scales_sel,
                    rotations_sel,
                    opacity_sel,
                    sh_dc_sel,
                )
                render_full_np = draw_bbox_outline(
                    (np.clip(render_full_np, 0, 1) * 255.0).astype(np.uint8), bbox_for_draw, color=(0, 255, 0), thickness=3
                )
                iio.imwrite(preview_path, render_full_np)

                render_strip = self._concat_horizontal(render_strips, width, height)
                target_strip = self._concat_horizontal(target_strips, width, height)
                mask_strip = self._concat_horizontal(mask_strips, width, height)
                iio.imwrite(loss_render_path, render_strip)
                iio.imwrite(loss_target_path, target_strip)
                iio.imwrite(loss_mask_path, mask_strip)
                self._write_ply_subset(
                    header_lines,
                    vertices,
                    indices_arr,
                    means_sel,
                    scales_sel,
                    rotations_sel,
                    opacity_sel,
                    sh_dc_sel,
                    means,
                    scales,
                    rotations,
                    opacity,
                    sh_dc,
                    live_ply_path,
                )

                progress_entry.update(
                    {
                        "status": "running",
                        "iter": it + 1,
                        "loss": loss_curve[-1],
                        "iter_max": iters,
                        "render_path": str(preview_path),
                        "oracle_path": str(oracle_path),
                        "progress_render": str(loss_render_path),
                        "progress_target": str(loss_target_path),
                        "live_ply": str(live_ply_path),
                    }
                )
                last_update = now

        end = time.time()

        render_after_np = self._render_full_numpy_view(
            rotation_wc_best,
            position_w_best,
            width,
            height,
            fov_deg,
            means,
            scales,
            rotations,
            opacity,
            sh_dc,
            sel,
            means_sel,
            scales_sel,
            rotations_sel,
            opacity_sel,
            sh_dc_sel,
        )
        render_after_vis = draw_bbox_outline(
            (np.clip(render_after_np, 0, 1) * 255.0).astype(np.uint8), bbox_for_draw, color=(0, 255, 0), thickness=3
        )
        render_after_vis = np.rot90(render_after_vis, 2)
        render_after_path = output_dir / "render_after.png"
        iio.imwrite(render_after_path, render_after_vis)

        refined_path = output_dir / "gs_refined.ply"
        self._write_ply_subset(
            header_lines,
            vertices,
            indices_arr,
            means_sel,
            scales_sel,
            rotations_sel,
            opacity_sel,
            sh_dc_sel,
            means,
            scales,
            rotations,
            opacity,
            sh_dc,
            refined_path,
        )

        progress_entry.update(
            {
                "status": "done",
                "iter": iters,
                "loss": loss_curve[-1] if loss_curve else None,
                "iter_max": iters,
                "render_path": str(render_after_path),
                "oracle_path": str(oracle_path),
                "progress_render": str(loss_render_path),
                "progress_target": str(loss_target_path),
                "live_ply": str(refined_path),
                "render_before": str(render_before_path),
                "render_after": str(render_after_path),
                "oracle_patch": str(oracle_path),
                "time_sec": round(end - start, 3),
            }
        )

        metrics: Dict[str, float | None] = {}
        try:
            gt_np = best_view_data.get("gt_np")
            if gt_np is not None:
                metrics = self.metric_eval.fix_before_after_metrics(render_before_np, render_after_np, gt_np)
        except Exception as exc:  # noqa: BLE001
            print(f"Metric computation failed: {exc}")
            metrics = {}

        return {
            "status": "success",
            "experiment_id": exp_id,
            "refined_ply": str(refined_path),
            "selection_indices": indices_arr.tolist(),
            "loss_curve": loss_curve,
            "time_sec": round(end - start, 3),
            "message": f"ROI-aware Gaussian optimization finished in {end - start:.2f}s.",
            "outputs": {
                "render_before": str(render_before_path),
                "render_after": str(render_after_path),
                "oracle_patch": str(oracle_path),
                "progress_render": str(loss_render_path),
                "progress_target": str(loss_target_path),
                "live_ply": str(refined_path),
            },
            "target_view": best_view,
            "metrics": metrics,
        }


    @staticmethod
    def _concat_horizontal(strips: List[np.ndarray], width: int, height: int) -> np.ndarray:
        return np.concatenate(strips, axis=1) if strips else np.zeros((height, width, 3), dtype=np.uint8)

    def _render_full_for_view(
        self,
        view_data: Dict,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        sh_dc: torch.Tensor,
        sel: torch.Tensor,
        means_sel_cur: torch.Tensor,
        scales_sel_cur: torch.Tensor,
        rotations_sel_cur: torch.Tensor,
        opacity_sel_cur: torch.Tensor,
        sh_dc_sel_cur: torch.Tensor,
        width: int,
        height: int,
        fov_deg: float,
    ) -> torch.Tensor:
        means_all = self._combine_params(means, means_sel_cur, sel)
        scales_all = self._combine_params(scales, scales_sel_cur, sel)
        rotations_all = self._combine_params(rotations, rotations_sel_cur, sel)
        opacity_all = self._combine_params(opacity, opacity_sel_cur, sel)
        sh_all = self._combine_params(sh_dc, sh_dc_sel_cur, sel)
        render_full, _ = self._render_roi_patch(
            means_all,
            scales_all,
            rotations_all,
            opacity_all,
            sh_all,
            view_data["rotation_wc"],
            view_data["position_w"],
            width,
            height,
            {"x0": 0, "x1": width, "y0": 0, "y1": height},
            fov_deg,
        )
        return render_full

    @staticmethod
    def _apply_lr_schedule(optimizer: torch.optim.Optimizer, step: int, iters: int, lr_final: float, warmup_ratio: float) -> None:
        warmup_steps = max(int(iters * warmup_ratio), 1)
        if step < warmup_steps:
            scale = float(step + 1) / float(warmup_steps)
        else:
            prog = float(step - warmup_steps) / float(max(iters - warmup_steps, 1))
            scale = lr_final + 0.5 * (1.0 - lr_final) * (1.0 + math.cos(math.pi * prog))
        for g in optimizer.param_groups:
            base_lr = g.get("base_lr", g["lr"])
            g["base_lr"] = base_lr
            g["lr"] = base_lr * scale

    def _maybe_render_video_only(
        self,
        ctx: Dict,
        best_view_data: Dict,
        rotation_wc_best: np.ndarray,
        position_w_best: np.ndarray,
        best_view: str,
    ) -> Dict | None:
        render_video_only = ctx.get("render_video_only", False)
        if not render_video_only:
            return None
        up_axis = ctx.get("up_axis", "y")
        center_target = ctx.get("center_target", np.zeros(3, dtype=np.float32))
        cam_info = ctx.get("cam_info", {})
        start_eye, start_wxyz = self._camera_pose_for_view("input", center_target, depth=1.5, up_axis=up_axis)
        cam_end_eye = cam_info.get("position") if isinstance(cam_info, dict) else None
        cam_end_wxyz = cam_info.get("wxyz") if isinstance(cam_info, dict) else None
        end_eye_local = (
            np.asarray(cam_end_eye, dtype=np.float32)
            if cam_end_eye is not None
            else best_view_data.get("position_w", position_w_best)
        )
        end_wxyz_local = (
            np.asarray(cam_end_wxyz, dtype=np.float32)
            if cam_end_wxyz is not None
            else best_view_data.get("rotation_wc", rotation_wc_best)
        )
        return self._render_interp_video_only(ctx, {"eye": end_eye_local, "wxyz": end_wxyz_local}, best_view)

    # --- Public helper for arbitrary multiview optimization ----------
    def optimize_with_views(
        self,
        gs_ply: Path,
        view_specs: List[Dict],
        output_dir: Path,
        image_size: Tuple[int, int] = (512, 512),
        selection_indices: List[int] | None = None,
        frame_token: str = "000000",
        experiment_id: str | None = None,
        opt_params: Dict | None = None,
    ) -> Dict:
        """
        Optimize a GS PLY using arbitrary multi-view GT images.

        view_specs: list of {name, image_path, eye, wxyz}.
        selection_indices: list of splat indices to optimize; defaults to all.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_gt_dir = output_dir / "eval_tmp"
        eval_gt_dir.mkdir(parents=True, exist_ok=True)

        if selection_indices is None:
            _, vertices = read_ply_vertices(gs_ply)
            selection_indices = list(range(len(vertices)))

        camera_overrides: Dict[str, Dict] = {}
        for spec in view_specs:
            name = spec.get("name") or f"view_{len(camera_overrides)}"
            img_path = Path(spec["image_path"])
            view_dir = eval_gt_dir / name
            view_dir.mkdir(parents=True, exist_ok=True)
            target_img = view_dir / f"{frame_token}.png"
            if img_path.resolve() != target_img.resolve():
                try:
                    shutil.copy(img_path, target_img)
                except Exception:
                    pass
            else:
                target_img = img_path
            camera_overrides[name] = {
                "eye": spec["eye"],
                "wxyz": spec["wxyz"],
            }

        payload = {
            "selection": {"gs_indices": selection_indices},
            "paths": {
                "gs_ply": str(gs_ply),
                "eval_gt_dir": str(eval_gt_dir),
                "output_dir": str(output_dir),
            },
            "frame_token": frame_token,
            "camera_overrides": camera_overrides,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "opt_params": opt_params or {},
        }
        if experiment_id:
            payload["experiment_id"] = experiment_id
        ctx = self._parse_handle_inputs(payload)
        view_tensors, best_view_data, best_view_name, best_view_cam = self._collect_visible_views(ctx)
        return self._optimize_roi(
            ctx,
            view_tensors,
            best_view_name,
            best_view_data,
            best_view_cam["rotation_wc_best"],
            best_view_cam["position_w_best"],
            best_view_cam["bbox_best"],
        )

    def archor_view_refine(self, payload: Dict) -> Dict:
        """Anchor-view refinement to stabilize nearby tilted views.

        Expected payload keys:
        - paths.gs_ply: PLY to optimize.
        - anchor_view: one of ["input", "eval_0", "eval_1", "eval_2", "eval_3"].
        - frame_token: used with eval_gt_dir to locate GT when anchor_view is eval_*.
        - paths.anchor_image: optional absolute path; used when anchor_view == "input".
        - paths.eval_gt_dir: root of eval_gt views (eval_0...eval_3/frame_token.png).
        - selection.gs_indices: optional list of indices to optimize; defaults to all.
        - opt_params: overrides for iters / learning rates.
        - tilt_deg / tilt_pitch_deg / tilt_loss_weight: control nearby tilt sampling.
        """

        paths = payload.get("paths", {})
        if "gs_ply" not in paths:
            raise ValueError("paths.gs_ply is required for anchor_view_refine.")
        gs_path = Path(paths["gs_ply"])
        output_dir = Path(paths.get("output_dir", ROOT / "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        anchor_view = payload.get("anchor_view", "input")
        frame_token = payload.get("frame_token")
        up_axis = payload.get("up_axis", "y")
        eval_gt_root = Path(paths.get("eval_gt_dir", ""))
        workspace_root = Path(paths.get("workspace_root", output_dir.parent))

        selection = payload.get("selection", {})
        indices = selection.get("gs_indices") or selection.get("indices")

        opt_params = payload.get("opt_params", {})
        iters = int(opt_params.get("iters", 2000))
        lr_color = float(opt_params.get("lr_color", 0.0015))
        lr_opacity = float(opt_params.get("lr_opacity", 0.02))
        lr_scale = float(opt_params.get("lr_scale", 0.003))
        lr_rotation = float(opt_params.get("lr_rotation", 0.0005))
        lr_means = float(opt_params.get("lr_means", 0.00008))
        lambda_dssim = float(opt_params.get("lambda_dssim", 0.2))
        teacher_consistency_weight = float(opt_params.get("teacher_consistency_weight", 0.15))
        scale_reg_weight = float(opt_params.get("scale_reg_weight", 0.001))
        teacher_extra_views = int(opt_params.get("teacher_extra_views", 8))
        tilt_deg = float(opt_params.get("tilt_deg", payload.get("tilt_deg", 10.0)))
        tilt_pitch_deg = float(opt_params.get("tilt_pitch_deg", payload.get("tilt_pitch_deg", 6.0)))
        optimize_means = bool(opt_params.get("optimize_means", False))
        optimize_rotation = bool(opt_params.get("optimize_rotation", False))

        prop_names, vertices, header_lines, tensors = self._load_ply_tensors(gs_path)
        means = tensors["means"]
        scales = tensors["scales"]
        rotations = tensors["rotations"]
        opacity = tensors["opacity"]
        sh_dc = tensors["sh_dc"]

        # Determine selection (defaults to all points).
        if indices:
            indices_arr = np.asarray(indices, dtype=np.int64)
            indices_arr = indices_arr[(indices_arr >= 0) & (indices_arr < means.shape[0])]
        else:
            indices_arr = np.arange(means.shape[0], dtype=np.int64)
        if indices_arr.size == 0:
            raise ValueError("Selection indices are out of bounds for anchor_view_refine.")
        sel = torch.as_tensor(indices_arr, device=self.device, dtype=torch.long)

        means_sel = means[sel].clone().requires_grad_(False)
        scales_sel = scales[sel].clone().requires_grad_(True)
        opacity_sel = opacity[sel].clone().requires_grad_(True)
        sh_dc_sel = sh_dc[sel].clone().requires_grad_(True)
        rotations_sel = rotations[sel].clone().requires_grad_(False)

        teacher = {
            "means": means[sel].detach().clone(),
            "scales": scales[sel].detach().clone(),
            "rotations": rotations[sel].detach().clone(),
            "opacity": opacity[sel].detach().clone(),
            "sh_dc": sh_dc[sel].detach().clone(),
        }

        center_target = payload.get("center_target", [0.0, 0.0, 0.0])



        # Resolve anchor GT path.
        anchor_image_path: Path | None = None
        if anchor_view.startswith("eval"):
            if not frame_token:
                raise ValueError("frame_token is required when anchor_view is eval_*." )
            anchor_image_path = eval_gt_root / anchor_view / f"{frame_token}.png"
        else:
            if paths.get("anchor_image"):
                anchor_image_path = Path(paths["anchor_image"])
            elif frame_token:
                anchor_image_path = workspace_root / "images" / "input" / f"{frame_token}.png"
        if anchor_image_path is None or not anchor_image_path.exists():
            raise ValueError(f"Anchor image not found for view {anchor_view}.")

        anchor_np = load_image(anchor_image_path, flip_udlr=False)
        anchor_np = anchor_np
        height, width = anchor_np.shape[:2]
        anchor_t = torch.from_numpy(anchor_np.astype(np.float32) / 255.0).to(self.device).permute(2, 0, 1)

        eye_anchor, wxyz_anchor = self._camera_pose_for_view(anchor_view, center_target, depth=1.5, up_axis=up_axis)
        rotation_anchor = self._camera_to_world_rotation(wxyz_anchor.tolist())

        # Prepare tilt poses (yaw +/- tilt_deg, pitch +/- tilt_pitch_deg) for teacher consistency only.
        tilt_offsets = [(-tilt_deg, 0.0), (tilt_deg, 0.0), (0.0, -tilt_pitch_deg), (0.0, tilt_pitch_deg)]
        tilt_views: List[Dict] = []
        teacher_wide_views: List[Dict] = []
        means_np = means.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        rotations_np = rotations.detach().cpu().numpy()
        opacity_np = opacity.detach().cpu().numpy()
        sh_dc_np = sh_dc.detach().cpu().numpy()

        def _render_teacher_np(rot_wc: np.ndarray, pos_w: np.ndarray) -> torch.Tensor:
            render_np = self._render_full_simple(
                means_np,
                scales_np,
                rotations_np,
                opacity_np,
                sh_dc_np,
                rot_wc,
                pos_w,
                width,
                height,
                float(payload.get("fov_deg", 49.1)),
            )
            render_np = np.clip(render_np, 0.0, 1.0)
            render_np = render_np
            return torch.from_numpy(render_np.astype(np.float32)).to(self.device).permute(2, 0, 1)

        for yaw_off, pitch_off in tilt_offsets:
            eye_tilt, wxyz_tilt = self._camera_pose_for_view(
                anchor_view, center_target, depth=1.5, up_axis=up_axis, yaw_off=yaw_off, pitch_off=pitch_off
            )
            rot_tilt = self._camera_to_world_rotation(wxyz_tilt.tolist())
            tilt_views.append(
                {
                    "rotation_wc": rot_tilt,
                    "position_w": eye_tilt,
                    "yaw_off": yaw_off,
                    "pitch_off": pitch_off,
                    "teacher": _render_teacher_np(rot_tilt, eye_tilt),
                }
            )

        # Wider teacher-only viewpoints sampled uniformly around the object to stabilize non-anchor views.
        if teacher_extra_views > 0:
            pitch_mag = max(tilt_pitch_deg * 2.0, 10.0)
            for i in range(max(1, teacher_extra_views)):
                yaw_off = (360.0 / float(max(1, teacher_extra_views))) * i
                pitch_off = pitch_mag if i % 2 == 0 else -pitch_mag
                if abs(yaw_off) <= tilt_deg * 1.1 and abs(pitch_off) <= tilt_pitch_deg * 1.1:
                    continue  # Skip inside the jitter band.
                eye_wide, wxyz_wide = self._camera_pose_for_view(
                    anchor_view, center_target, depth=1.5, up_axis=up_axis, yaw_off=yaw_off, pitch_off=pitch_off
                )
                rot_wide = self._camera_to_world_rotation(wxyz_wide.tolist())
                teacher_wide_views.append(
                    {
                        "rotation_wc": rot_wide,
                        "position_w": eye_wide,
                        "yaw_off": yaw_off,
                        "pitch_off": pitch_off,
                        "teacher": _render_teacher_np(rot_wide, eye_wide),
                    }
                )

        def _combine_all(means_cur: torch.Tensor, scales_cur: torch.Tensor, rotations_cur: torch.Tensor, opacity_cur: torch.Tensor, sh_cur: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            means_all = self._combine_params(means, means_cur, sel)
            scales_all = self._combine_params(scales, scales_cur, sel)
            rotations_all = self._combine_params(rotations, rotations_cur, sel)
            opacity_all = self._combine_params(opacity, opacity_cur, sel)
            sh_all = self._combine_params(sh_dc, sh_cur, sel)
            return means_all, scales_all, rotations_all, opacity_all, sh_all

        def _render_full_torch(rot_wc: np.ndarray, pos_w: np.ndarray, means_cur: torch.Tensor, scales_cur: torch.Tensor, rotations_cur: torch.Tensor, opacity_cur: torch.Tensor, sh_cur: torch.Tensor) -> torch.Tensor:
            means_all, scales_all, rotations_all, opacity_all, sh_all = _combine_all(means_cur, scales_cur, rotations_cur, opacity_cur, sh_cur)
            full_img, _ = self._render_roi_patch(
                means_all,
                scales_all,
                rotations_all,
                opacity_all,
                sh_all,
                rot_wc,
                pos_w,
                width,
                height,
                {"x0": 0, "x1": width, "y0": 0, "y1": height},
                float(payload.get("fov_deg", 49.1)),
            )
            return full_img

        # Reference renders for logging.
        render_before = _render_full_torch(rotation_anchor, eye_anchor, means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel)
        render_before_np = np.clip(render_before.detach().cpu().numpy(), 0.0, 1.0)
        render_before_np = render_before_np
        render_before_path = output_dir / "archor_render_before.png"
        iio.imwrite(render_before_path, (render_before_np * 255.0).astype(np.uint8))

        optimizer = torch.optim.Adam(
            [
                {"params": sh_dc_sel, "lr": lr_color},
                {"params": opacity_sel, "lr": lr_opacity},
                {"params": scales_sel, "lr": lr_scale},
            ]
        )
        if optimize_means:
            optimizer.add_param_group({"params": means_sel, "lr": lr_means})
        if optimize_rotation:
            optimizer.add_param_group({"params": rotations_sel, "lr": lr_rotation})

        warmup_ratio = 0.05
        lr_final_factor = float(opt_params.get("lr_final_factor", 0.05))
        for g in optimizer.param_groups:
            g["base_lr"] = g["lr"]

        def _apply_lr_schedule(step: int) -> None:
            warmup_steps = max(int(iters * warmup_ratio), 1)
            if step < warmup_steps:
                scale = float(step + 1) / float(warmup_steps)
            else:
                prog = float(step - warmup_steps) / float(max(iters - warmup_steps, 1))
                scale = lr_final_factor + 0.5 * (1.0 - lr_final_factor) * (1.0 + math.cos(math.pi * prog))
            for g in optimizer.param_groups:
                base_lr = g.get("base_lr", g["lr"])
                g["lr"] = base_lr * scale

        exp_id = payload.get("experiment_id", f"archor_{int(time.time())}")
        progress_entry = {
            "status": "running",
            "iter": 0,
            "iter_max": iters,
            "render_path": str(render_before_path),
        }
        self.progress[exp_id] = progress_entry

        live_ply_path = output_dir / "archor_gs_live.ply"

        def _write_ply(out_path: Path, means_sel_cur: torch.Tensor, scales_sel_cur: torch.Tensor, rotations_sel_cur: torch.Tensor, opacity_sel_cur: torch.Tensor, sh_dc_sel_cur: torch.Tensor) -> None:
            """Write PLY back to source frame (inverse of COORD_ALIGN_MAT)."""
            means_np = means.detach().cpu().numpy()
            scales_np = scales.detach().cpu().numpy()
            rotations_np = rotations.detach().cpu().numpy()
            opacity_np = opacity.detach().cpu().numpy()
            sh_dc_np = sh_dc.detach().cpu().numpy()

            means_np[indices_arr] = means_sel_cur.detach().cpu().numpy()
            scales_np[indices_arr] = scales_sel_cur.detach().cpu().numpy()
            rotations_np[indices_arr] = rotations_sel_cur.detach().cpu().numpy()
            opacity_np[indices_arr] = opacity_sel_cur.detach().cpu().numpy()
            sh_dc_np[indices_arr] = sh_dc_sel_cur.detach().cpu().numpy()

            # Inverse align centers and rotations back to PLY/world frame.
            centers_src = (means_np @ COORD_ALIGN_MAT).astype(np.float32)
            R_aligned = tf.SO3(rotations_np).as_matrix()
            R_src = (COORD_ALIGN_MAT.T[None, :, :] @ R_aligned @ COORD_ALIGN_MAT).astype(np.float32)
            quat_src = tf.SO3.from_matrix(R_src).wxyz.astype(np.float32)

            vertices_copy = vertices.copy()
            vertices_copy["x"] = centers_src[:, 0]
            vertices_copy["y"] = centers_src[:, 1]
            vertices_copy["z"] = centers_src[:, 2]
            vertices_copy["rot_0"] = quat_src[:, 0]
            vertices_copy["rot_1"] = quat_src[:, 1]
            vertices_copy["rot_2"] = quat_src[:, 2]
            vertices_copy["rot_3"] = quat_src[:, 3]
            vertices_copy["scale_0"] = scales_np[:, 0]
            vertices_copy["scale_1"] = scales_np[:, 1]
            vertices_copy["scale_2"] = scales_np[:, 2]
            vertices_copy["opacity"] = opacity_np
            vertices_copy["f_dc_0"] = sh_dc_np[:, 0]
            vertices_copy["f_dc_1"] = sh_dc_np[:, 1]
            vertices_copy["f_dc_2"] = sh_dc_np[:, 2]
            write_ply_vertices(header_lines, vertices_copy, out_path)



        loss_curve: List[float] = []
        last_update = time.time()
        start_time = last_update
        anchor_debug_dir = configured_debug_dir("optimizer/anchor", create=True)


        for it in tqdm(range(max(iters, 1))):
            _apply_lr_schedule(it)
            optimizer.zero_grad()

            render_anchor = _render_full_torch(rotation_anchor, eye_anchor, means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel)
            render_anchor_chw = render_anchor.permute(2, 0, 1)
            l1_anchor = (render_anchor_chw - anchor_t).abs().mean()
            ssim_term = 0.0
            if lambda_dssim > 0.0:
                ssim_val = _ssim_torch(render_anchor_chw.unsqueeze(0), anchor_t.unsqueeze(0))
                ssim_term = 1.0 - ssim_val
            anchor_loss = (1.0 - lambda_dssim) * l1_anchor + lambda_dssim * ssim_term


            if anchor_debug_dir is not None and it % 200 == 0:
                render_anchor_np = (np.array(render_anchor_chw.detach().cpu()) * 255.0).astype(np.uint8)
                anchor_t_np = (np.array(anchor_t.detach().cpu()) * 255.0).astype(np.uint8)
                iio.imwrite(anchor_debug_dir / f"iter_{it:05d}_anchor_render.png", render_anchor_np)
                iio.imwrite(anchor_debug_dir / f"iter_{it:05d}_anchor_target.png", anchor_t_np)



            teacher_loss = 0.0
            for tv in tilt_views:
                render_tilt = _render_full_torch(tv["rotation_wc"], tv["position_w"], means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel)
                render_tilt_chw = render_tilt.permute(2, 0, 1)
                teacher_loss = teacher_loss + (render_tilt_chw - tv["teacher"]).abs().mean()


            for t_idx, tv in enumerate(teacher_wide_views):
                render_teacher = _render_full_torch(tv["rotation_wc"], tv["position_w"], means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel)
                render_teacher = render_teacher
                render_teacher_chw = render_teacher.permute(2, 0, 1)
                teacher_loss = teacher_loss + (render_teacher_chw - tv["teacher"]).abs().mean()

                if anchor_debug_dir is not None and it % 200 == 0:
                    render_teacher_np = (
                        render_teacher_chw.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255.0
                    ).astype(np.uint8)
                    teacher_np = (
                        tv["teacher"].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255.0
                    ).astype(np.uint8)
                    iio.imwrite(
                        anchor_debug_dir / f"iter_{it:05d}_teacher_wide_{t_idx}_render.png",
                        render_teacher_np,
                    )
                    iio.imwrite(
                        anchor_debug_dir / f"iter_{it:05d}_teacher_wide_{t_idx}_target.png",
                        teacher_np,
                    )


            scale_reg = (scales_sel - teacher["scales"]).pow(2).mean()
            total_loss = (
                anchor_loss
                + teacher_consistency_weight * teacher_loss / max(len(tilt_views) + len(teacher_wide_views), 1)
                + scale_reg_weight * scale_reg
            )
            total_loss.backward()
            optimizer.step()

            loss_curve.append(float(total_loss.detach().cpu().item()))

            now = time.time()
            if now - last_update >= 1.5:
                render_prog_np = np.clip(render_anchor.detach().cpu().numpy(), 0.0, 1.0)
                render_prog_np = render_prog_np
                preview_path = output_dir / "archor_preview.png"
                iio.imwrite(preview_path, (render_prog_np * 255.0).astype(np.uint8))
                try:
                    _write_ply(live_ply_path, means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] archor live_ply write failed: {exc}")
                progress_entry.update(
                    {
                        "status": "running",
                        "iter": it + 1,
                        "iter_max": iters,
                        "loss": loss_curve[-1],
                        "render_path": str(preview_path),
                        "progress_render": str(preview_path),
                        "live_ply": str(live_ply_path),
                    }
                )
                last_update = now

        render_after = _render_full_torch(rotation_anchor, eye_anchor, means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel)
        render_after_np = np.clip(render_after.detach().cpu().numpy(), 0.0, 1.0)
        render_after_np = render_after_np
        render_after_path = output_dir / "archor_render_after.png"
        iio.imwrite(render_after_path, (render_after_np * 255.0).astype(np.uint8))

        refined_path = output_dir / "archor_gs_refined.ply"

        print("refined_path:", refined_path)
        print("render_after_path :", render_after_path )


        _write_ply(refined_path, means_sel, scales_sel, rotations_sel, opacity_sel, sh_dc_sel)

        progress_entry.update(
            {
                "status": "done",
                "iter": iters,
                "iter_max": iters,
                "loss": loss_curve[-1] if loss_curve else None,
                "render_path": str(render_after_path),
                "refined_ply": str(refined_path),
                "live_ply": str(refined_path),
            }
        )

        end_time = time.time()

        return {
            "status": "success",
            "experiment_id": exp_id,
            "refined_ply": str(refined_path),
            "render_before": str(render_before_path),
            "render_after": str(render_after_path),
            "anchor_view": anchor_view,
            "anchor_image": str(anchor_image_path),
            "loss_curve": loss_curve,
            "tilt_views": [{"yaw_off": t["yaw_off"], "pitch_off": t["pitch_off"]} for t in tilt_views],
            "time_sec": round(end_time - start_time, 3),
        }
