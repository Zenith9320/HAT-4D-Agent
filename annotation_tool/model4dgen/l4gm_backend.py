"""
Minimal backend wrapper for L4GM inference to produce a single GS PLY.

This is intentionally small and mirrors the pipeline in model4dgen/L4GM_MoGA/infer_3d.py
for a single image input. It loads the default 'big' config and the pretrained
reconstruction checkpoint `pretrained/recon.safetensors`.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import imageio

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from safetensors.torch import load_file

from kiui.cam import orbit_camera
from utils.debug_utils import configured_debug_dir

ROOT = Path(__file__).resolve().parent
L4GM_ROOT = ROOT / "L4GM_MoGA"
if str(L4GM_ROOT) not in sys.path:
    # Prepend so that L4GM's own `utils` is found before the repo's root `utils` package.
    sys.path.insert(0, str(L4GM_ROOT))

from .L4GM_MoGA.core.options_conf import config_defaults
from .L4GM_MoGA.core_free_fn.models import LGM
from .L4GM_MoGA.mvdream.pipeline_mvdream import MVDreamPipeline
from .L4GM_MoGA.core.options_conf import Options
from .L4GM_MoGA.utils.train_utils import seed_everything
from .L4GM_MoGA.core.gs import GaussianRenderer
from .L4GM_MoGA.utils.gpu_utils import print_gpu_memory

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _rgba_to_rgb_with_white_bg(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGBA")
    white_bg = Image.new("RGB", img.size, (255, 255, 255))
    white_bg.paste(img, mask=img.split()[3])
    return np.array(white_bg)


def _load_single_image(image_path: Path, size: int) -> np.ndarray:
    if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        rgb = _rgba_to_rgb_with_white_bg(image_path)
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    rgb = np.asarray(Image.fromarray(rgb).resize((size, size)))
    return rgb.astype(np.float32) / 255.0


def _ensure_checkpoint(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"L4GM checkpoint not found: {path}. Please place recon.safetensors under pretrained/."
        )
    return path


class L4GMGenerator:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = self._build_default_options()
        self.model, self.rays_embeddings = self._load_model(self.opt)
        self.mv_pipe = self._load_mvdream()
        self.renderer = GaussianRenderer(self.opt)
        self._warmup_renderer()
        # If >0, subsample seed GS for lower VRAM usage. Set to 0 to disable subsampling.
        self.seed_gs_max_points = int(os.environ.get("L4GM_SEED_GS_MAX_POINTS", "0"))

    def _maybe_subsample_gaussians(self, gaussians: torch.Tensor, max_points: int, *, seed: int) -> torch.Tensor:
        """Subsample gaussians [1,N,14] by opacity to reduce GPU memory."""
        if max_points <= 0:
            return gaussians
        if gaussians.dim() == 2:
            gaussians = gaussians.unsqueeze(0)
        if gaussians.dim() != 3:
            return gaussians
        n = int(gaussians.shape[1])
        if n <= max_points:
            return gaussians

        # Opacity is stored at index 3 in this repo's compatible layout.
        op = gaussians[0, :, 3].detach().float().cpu()
        op = torch.nan_to_num(op, nan=0.0, posinf=0.0, neginf=0.0)
        k = int(max_points)
        if float(op.max().item()) <= 0.0:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            idx = torch.randperm(n, generator=g)[:k]
        else:
            idx = torch.topk(op, k=k, largest=True, sorted=False).indices
        return gaussians.index_select(1, idx.to(device=gaussians.device))

    def _build_default_options(self) -> Options:
        base = copy.deepcopy(config_defaults["big"])
        checkpoint = Path(
            os.environ.get("L4GM_CHECKPOINT", self.root / "pretrained" / "recon.safetensors")
        ).expanduser()
        base.resume = str(_ensure_checkpoint(checkpoint))
        base.workspace = str((self.root / "workspace").resolve())
        if base.exp_log_path is None:
            base.exp_log_path = str((self.root / "workspace").resolve())
        return base

    def _warmup_renderer(self):
        """
        Warm up gsplat rasterization by triggering a minimal render.
        This will compile CUDA extensions and initialize kernels.
        """

        print("[L4GMGenerator] Warming up gsplat renderer...")

        device = self.device
        opt = self.opt

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):

            # --------------------------------------------------
            # build minimal gaussian: [1, 1, 14]
            # --------------------------------------------------
            gaussians = torch.zeros((1, 1, 14), device=device, dtype=torch.float32)

            # xyz (in front of camera)
            gaussians[..., 0:3] = torch.tensor([[0.0, 0.0, 2.0]], device=device)

            # opacity
            gaussians[..., 3:4] = 1.0

            # scale
            gaussians[..., 4:7] = 0.05

            # quaternion (identity, wxyz)
            gaussians[..., 7:11] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

            # rgb (red point)
            gaussians[..., 11:14] = torch.tensor([[1.0, 0.0, 0.0]], device=device)

            # --------------------------------------------------
            # build camera
            # --------------------------------------------------
            cam_poses = torch.from_numpy(
                orbit_camera(0, 0, radius=opt.cam_radius, opengl=True)
            ).unsqueeze(0).to(device)

            cam_poses[:, :3, 1:3] *= -1.0

            cam_view = torch.inverse(cam_poses).transpose(1, 2)   # [1,4,4]
            cam_view_proj = cam_view @ self.renderer.proj_matrix.to(device)
            cam_pos = -cam_poses[:, :3, 3]

            bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)

            # --------------------------------------------------
            # trigger rasterization (CUDA compile happens here)
            # --------------------------------------------------
            _ = self.model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),        # [B=1, V=1, 4,4]
                cam_view_proj.unsqueeze(0),   # [B=1, V=1, 4,4]
                cam_pos.unsqueeze(0),         # [B=1, V=1, 3]
                bg_color=bg_color,
            )

        print("[L4GMGenerator] gsplat renderer warmup complete")


    def _load_model(self, opt: Options) -> Tuple[LGM, torch.Tensor]:
        seed_everything(opt.seed)
        model = LGM(opt)
        if opt.resume:
            print("opt.resume:", opt.resume)
            if opt.resume.endswith('safetensors'):
                ckpt = load_file(opt.resume, device='cpu')
            else:
                ckpt = torch.load(opt.resume, map_location='cpu')
            model.load_state_dict(ckpt, strict=False)
        model = model.half().to(self.device).eval()
        rays_embeddings = model.prepare_default_rays(self.device)
        return model, rays_embeddings

    def _load_mvdream(self) -> MVDreamPipeline:
        print("self.cache:", str(self.root / ".cache"))
        pipe = MVDreamPipeline.from_pretrained(
            "ashawkey/imagedream-ipmv-diffusers",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=str(self.root / ".cache"),
            local_files_only=True,
        )
        return pipe.to(self.device)

    def _make_multi_view(self, rgb: np.ndarray, opt: Options) -> np.ndarray:
        # Use mvdream to hallucinate 4 novel views from a single RGB.
        mv_image = self.mv_pipe("", rgb, guidance_scale=5, num_inference_steps=30, elevation=0)
        mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0)

        debug_dir = configured_debug_dir("l4gm/multiview", create=True)
        if debug_dir is not None:
            for view_index in range(mv_image.shape[0]):
                image = (mv_image[view_index] * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(image).save(debug_dir / f"view_{view_index:03d}.png")

        return mv_image.astype(np.float32)


    def _prepare_rays(self, num_frames: int) -> torch.Tensor:
        """Repeat default rays for the requested number of frames."""
        rays = self.model.prepare_default_rays(self.device)
        # rays: [V, C, H, W]; repeat over frames
        rays = torch.cat([rays for _ in range(num_frames)], dim=0)
        return rays

    def _forward_gaussians_sequence(self, mv_video: np.ndarray, opt: Options, num_clip_frames: int) -> torch.Tensor:
        """Forward multiple frames with shared multi-view inputs.

        mv_video: [T, V, H, W, 3] float32 in [0,1]
        """
        T, V, H, W, _ = mv_video.shape
        input_image = (
            torch.from_numpy(mv_video)
            .permute(0, 1, 4, 2, 3)
            .reshape(-1, 3, H, W)
            .float()
            .to(self.device)
        )
        input_image = F.interpolate(
            input_image,
            size=(opt.input_size, opt.input_size),
            mode="bilinear",
            align_corners=False,
        )
        input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        rays_embeddings = self._prepare_rays(num_clip_frames)
        rays_embeddings = F.interpolate(
            rays_embeddings,
            size=(opt.input_size, opt.input_size),
            mode="bilinear",
            align_corners=False,
        )
        rays_embeddings = rays_embeddings.to(dtype=input_image.dtype, device=input_image.device)

        input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16):

            debug_dir = configured_debug_dir("l4gm/tensors", create=True)
            if debug_dir is not None:
                np.savez(debug_dir / "input_image.npz", input_image=input_image.cpu().numpy())
            gaussians_all_frame = self.model.forward_gaussians(
                input_image, num_clip_frames=num_clip_frames
            )
        return gaussians_all_frame

    def _save_ply(self, gaussians: torch.Tensor, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer.save_ply(gaussians, str(out_path), compatible=True)

    def _render_four_views(self, gaussians: torch.Tensor, opt: Options) -> np.ndarray:
        """Render 4 views (azimuth 0,90,180,270; elevation 0) from gaussians."""
        device = gaussians.device
        bg_color = torch.tensor([255, 255, 255], dtype=torch.float32, device=device) / 255.0

        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        proj_matrix[0, 0] = 1.0 / tan_half_fov
        proj_matrix[1, 1] = 1.0 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1.0

        mv_images: List[np.ndarray] = []
        for azi in (0, 90, 180, 270):
            cam_poses = torch.from_numpy(
                orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)
            ).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1.0

            cam_view = torch.inverse(cam_poses).transpose(1, 2)
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = -cam_poses[:, :3, 3]

            result = self.model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
                bg_color=bg_color,
            )
            image = result["image"].squeeze(1)
            image = F.interpolate(image, (opt.input_size, opt.input_size))
            image_np = (
                image.permute(0, 2, 3, 1)
                .contiguous()
                .float()
                .cpu()
                .numpy()
            )
            mv_images.append(image_np[0])

        return np.stack(mv_images, axis=0)

    def _render_four_views_with_offset(self, gaussians: torch.Tensor, opt: Options, azimuth_offset: float) -> np.ndarray:
        """Render 4 views with a global azimuth offset (degrees)."""
        device = gaussians.device
        bg_color = torch.tensor([255, 255, 255], dtype=torch.float32, device=device) / 255.0

        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        proj_matrix[0, 0] = 1.0 / tan_half_fov
        proj_matrix[1, 1] = 1.0 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1.0

        mv_images: List[np.ndarray] = []
        for azi_base in (0, 90, 180, 270):
            azi = azi_base + azimuth_offset
            cam_poses = torch.from_numpy(
                orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)
            ).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1.0

            cam_view = torch.inverse(cam_poses).transpose(1, 2)
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = -cam_poses[:, :3, 3]

            result = self.model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
                bg_color=bg_color,
            )
            image = result["image"].squeeze(1)
            image = F.interpolate(image, (opt.input_size, opt.input_size))
            image_np = (
                image.permute(0, 2, 3, 1)
                .contiguous()
                .float()
                .cpu()
                .numpy()
            )
            mv_images.append(image_np[0])

        return np.stack(mv_images, axis=0)

    def _render_four_views_orthographic_with_offset(
        self,
        gaussians: torch.Tensor,
        opt: Options,
        azimuth_offset: float,
    ) -> np.ndarray:
        """Render 4 orthographic views with a global azimuth offset (degrees)."""
        device = gaussians.device
        bg_color = torch.tensor([255, 255, 255], dtype=torch.float32, device=device) / 255.0

        ortho_scale = float(getattr(opt, "cam_radius", 1.5)) * 0.85
        znear = max(1e-3, float(opt.znear))
        zfar = max(znear + 1e-3, float(opt.zfar))

        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        proj_matrix[0, 0] = 1.0 / max(1e-6, ortho_scale)
        proj_matrix[1, 1] = 1.0 / max(1e-6, ortho_scale)
        proj_matrix[2, 2] = 2.0 / (zfar - znear)
        proj_matrix[3, 2] = -(zfar + znear) / (zfar - znear)
        proj_matrix[3, 3] = 1.0

        mv_images: List[np.ndarray] = []
        for azi_base in (0, 90, 180, 270):
            azi = azi_base + azimuth_offset
            cam_poses = torch.from_numpy(
                orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)
            ).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1.0

            cam_view = torch.inverse(cam_poses).transpose(1, 2)
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = -cam_poses[:, :3, 3]

            result = self.model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
                bg_color=bg_color,
            )
            image = result["image"].squeeze(1)
            image = F.interpolate(image, (opt.input_size, opt.input_size))
            image_np = (
                image.permute(0, 2, 3, 1)
                .contiguous()
                .float()
                .cpu()
                .numpy()
            )
            mv_images.append(image_np[0])

        return np.stack(mv_images, axis=0)

    def _find_best_azimuth(self, gaussians: torch.Tensor, target_rgb: np.ndarray, opt: Options) -> float:
        """Brute-force search for azimuth (degrees) best aligning rendered view to target RGB."""
        device = gaussians.device
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        bg_color = torch.tensor([255, 255, 255], dtype=torch.float32, device=device) / 255.0

        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        proj_matrix[0, 0] = 1.0 / tan_half_fov
        proj_matrix[1, 1] = 1.0 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1.0

        target = torch.from_numpy(target_rgb).to(device=device, dtype=torch.float32)
        best_azi = 0.0
        best_diff = float("inf")
        best_pred_image: torch.Tensor | None = None
        for azi in range(-180, 180):
            cam_poses = torch.from_numpy(
                orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)
            ).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1.0

            cam_view = torch.inverse(cam_poses).transpose(1, 2)
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = -cam_poses[:, :3, 3]

            result = self.model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
                bg_color=bg_color,
            )
            image = result["image"].squeeze(1)
            image = F.interpolate(image, (opt.input_size, opt.input_size))
            image_np = image.permute(0, 2, 3, 1).contiguous().float()
            diff = torch.mean((image_np - target[None, ...]) ** 2).item()
            if diff < best_diff:
                best_diff = diff
                best_azi = float(azi)
                best_pred_image = image_np[0].detach()

        debug_dir = configured_debug_dir("l4gm/alignment", create=True)
        if best_pred_image is not None and debug_dir is not None:
            gt_np = (target.clamp(0, 1).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pred_np = (best_pred_image.clamp(0, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            Image.fromarray(gt_np).save(debug_dir / f"gt_{best_azi:+05.1f}.png")
            Image.fromarray(pred_np).save(debug_dir / f"pred_{best_azi:+05.1f}.png")

        print(f"[generate_segment] best azimuth offset: {best_azi} (mse={best_diff:.6f})")
        return best_azi

    def _rotate_gaussians_y(self, gaussians: torch.Tensor, degrees: float) -> torch.Tensor:
        """Rotate gaussians around Y-axis by degrees (in-place copy)."""
        if abs(degrees) < 1e-3:
            return gaussians

        theta = math.radians(degrees)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        rot_mat = torch.tensor(
            [[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]],
            dtype=gaussians.dtype,
            device=gaussians.device,
        )

        out = gaussians.clone()
        means = gaussians[..., 0:3]
        out[..., 0:3] = means @ rot_mat.t()

        # Quaternion multiplication: q_y * q (wxyz convention)
        q = gaussians[..., 7:11]
        w, x, y, z = q.unbind(-1)
        w_r = torch.full_like(w, math.cos(theta * 0.5))
        x_r = torch.zeros_like(x)
        y_r = torch.full_like(y, math.sin(theta * 0.5))
        z_r = torch.zeros_like(z)

        w_new = w_r * w - x_r * x - y_r * y - z_r * z
        x_new = w_r * x + x_r * w + y_r * z - z_r * y
        y_new = w_r * y - x_r * z + y_r * w + z_r * x
        z_new = w_r * z + x_r * y - y_r * x + z_r * w

        quat = torch.stack((w_new, x_new, y_new, z_new), dim=-1)
        quat = F.normalize(quat, dim=-1, eps=1e-8)
        out[..., 7:11] = quat
        return out

    def _load_gaussians_from_ply(self, ply_path: Path) -> torch.Tensor:
        """Load a single-frame Gaussian from a saved PLY (compatible layout)."""
        if not ply_path.exists():
            raise FileNotFoundError(f"GS PLY not found: {ply_path}")
        gaussians_cpu = self.renderer.load_ply(str(ply_path), compatible=True).to(dtype=torch.float32)
        if gaussians_cpu.dim() == 2:
            gaussians_cpu = gaussians_cpu.unsqueeze(0)

        n_all = int(gaussians_cpu.shape[1]) if gaussians_cpu.dim() == 3 else 0
        max_pts = int(self.seed_gs_max_points)
        if max_pts <= 0:
            try:
                return gaussians_cpu.to(device=self.device, dtype=torch.float32)
            except RuntimeError as exc:
                msg = str(exc)
                if self.device.type == "cuda" and ("CUBLAS_STATUS_ALLOC_FAILED" in msg or "out of memory" in msg):
                    raise RuntimeError(
                        f"Failed to load full seed GS onto GPU ({n_all:,} points). "
                        "Set L4GM_SEED_GS_MAX_POINTS (e.g. 200000) to enable subsampling."
                    ) from exc
                raise
        caps = [max_pts, max(50_000, max_pts // 2), max(20_000, max_pts // 4)]
        last_exc: Exception | None = None
        for cap in caps:
            g_try = self._maybe_subsample_gaussians(gaussians_cpu, cap, seed=n_all + 1337)
            try:
                return g_try.to(device=self.device, dtype=torch.float32)
            except RuntimeError as exc:
                last_exc = exc
                msg = str(exc)
                if self.device.type == "cuda" and ("CUBLAS_STATUS_ALLOC_FAILED" in msg or "out of memory" in msg):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        return gaussians_cpu.to(device=self.device, dtype=torch.float32)

    def _save_mv_images(self, mv_image: np.ndarray, frame_idx: int, preview_dir: Path) -> None:
        """Persist multi-view images to preview/mv for inspection."""
        preview_dir.mkdir(parents=True, exist_ok=True)
        for v in range(mv_image.shape[0]):
            img = (mv_image[v] * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(preview_dir / f"{frame_idx:06d}_{v:03d}.png")

    def _forward_and_save_segment(
        self,
        ref_video: np.ndarray,
        start_frame_idx: int,
        session_root: Path,
        opt: Options,
    ) -> List[Path]:
        num_clip_frames = ref_video.shape[0]
        gaussians_all_frame = self._forward_gaussians_sequence(ref_video, opt, num_clip_frames)
        B = 1
        gaussians_all_frame = gaussians_all_frame.reshape(
            B, gaussians_all_frame.shape[0] // opt.batch_size, *gaussians_all_frame.shape[1:]
        )

        out_paths: List[Path] = []
        for t in range(num_clip_frames):
            out_ply = session_root / "gs_sequence" / f"{start_frame_idx + t:06d}.ply"
            self._save_ply(gaussians_all_frame[:, t], out_ply)
            out_paths.append(out_ply)
        return out_paths

    def generate_from_image(
        self,
        image_path: Path,
        out_ply: Path,
        init_mode: str = "imagedream_init",
        input_frames: Optional[List[str | Path]] = None,
    ) -> Dict:
        assert init_mode == "imagedream_init", "Only 'imagedream_init' is supported in generate_from_image. If you use sv4d, please run add --l4gm-init-sv4d"
        opt = self.opt
        seed_everything(opt.seed)
        rgb = _load_single_image(image_path, opt.input_size)
        mv_image = self._make_multi_view(rgb, opt)

        mv_image_resized = []
        for v in range(mv_image.shape[0]):
            pil_img = Image.fromarray((mv_image[v] * 255.0).clip(0, 255).astype(np.uint8))
            pil_img = pil_img.resize((opt.input_size, opt.input_size))
            mv_image_resized.append(np.asarray(pil_img).astype(np.float32) / 255.0)
        mv_image_resized = np.stack(mv_image_resized, axis=0)

        opt.num_frames = 1
        opt.num_views = 4
        gaussians_all_frame = self._forward_gaussians_sequence(
            mv_image_resized[None, ...], opt, num_clip_frames=1
        )
        B = 1
        gaussians_all_frame = gaussians_all_frame.reshape(
            B, gaussians_all_frame.shape[0] // opt.batch_size, *gaussians_all_frame.shape[1:]
        )
        gaussians = gaussians_all_frame[:, 0]


        best_azimuth = self._find_best_azimuth(gaussians, rgb, opt)
        best_azimuth = - best_azimuth

        gaussians_aligned = self._rotate_gaussians_y(gaussians, best_azimuth)

        # self._save_ply(gaussians, out_ply)
        self._save_ply(gaussians_aligned, out_ply)
        print_gpu_memory()
        return {
            "status": "success",
            "ply": str(out_ply),
            "device": str(self.device),
            "best_azimuth": best_azimuth,
        }

    def generate_segment(
        self,
        frame_paths: List[Path],
        session_root: Path,
        start_frame_idx: int,
        num_clip_frames: int,
    ) -> Dict:
        """Generate a multi-frame GS segment starting at start_frame_idx.

        frame_paths: ordered list of input frames (length == num_clip_frames).
        """
        if num_clip_frames <= 0:
            raise ValueError("num_clip_frames must be positive")

        opt = copy.deepcopy(self.opt)
        opt.num_frames = num_clip_frames
        opt.num_views = 4

        preview_dir = session_root / "preview" / "mv"

        # 1) Multi-view images from the already-generated GS of the current frame
        seed_ply = session_root / "gs_sequence" / f"{start_frame_idx:06d}.ply"
        if not seed_ply.exists():
            # Fallback: use the latest available ply before start_frame_idx.
            for back in range(start_frame_idx - 1, -1, -1):
                candidate = session_root / "gs_sequence" / f"{back:06d}.ply"
                if candidate.exists():
                    seed_ply = candidate
                    break
        gaussians_first = self._load_gaussians_from_ply(seed_ply)
        first_frame_rgb = _load_single_image(frame_paths[0], opt.input_size)
        best_azimuth = self._find_best_azimuth(gaussians_first, first_frame_rgb, opt)
        best_azimuth = - best_azimuth
        mv_image_resized = self._render_four_views_with_offset(gaussians_first, opt, best_azimuth)
        self._save_mv_images(mv_image_resized, start_frame_idx, preview_dir)

        # 2) Build ref_video: view0 from each frame, other views copied from mv_image_resized
        ref_video = []
        for offset, path in enumerate(frame_paths):
            rgb = _load_single_image(path, opt.input_size)
            views = [rgb] + [mv_image_resized[v] for v in range(1, mv_image_resized.shape[0])]
            frame_views = np.stack(views, axis=0)

            frame_idx = start_frame_idx + offset
            for v in range(frame_views.shape[0]):
                img = (frame_views[v] * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(preview_dir / f"{frame_idx:06d}_{v:03d}.png")

            ref_video.append(frame_views)

        ref_video_arr = np.stack(ref_video, axis=0)

        # Optional visualization for debugging: save ref_video as a montage video (T, H, V*W, C).
        viz_ref_video = ref_video_arr.reshape(
            ref_video_arr.shape[0],
            ref_video_arr.shape[2] * ref_video_arr.shape[1],
            ref_video_arr.shape[3],
            ref_video_arr.shape[4],
        )
        viz_dir = session_root / "tmp" / "test_baseline"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_name = f"{start_frame_idx}_{start_frame_idx + opt.num_frames}_ref_video_debug.mp4"
        imageio.mimwrite(viz_dir / viz_name, (viz_ref_video * 255).clip(0, 255).astype(np.uint8), fps=10)

        # 3) Forward & save PLYs
        out_paths = self._forward_and_save_segment(ref_video_arr, start_frame_idx, session_root, opt)

        print_gpu_memory()
        return {
            "status": "success",
            "ply_paths": [str(p) for p in out_paths],
            "start_frame": start_frame_idx,
            "end_frame": start_frame_idx + len(out_paths) - 1,
            "mv_preview_dir": str(preview_dir),
            "device": str(self.device),
        }

    def _load_memory_bank_views(self, memory_bank_dir: Path, input_size: int) -> List[np.ndarray]:
        index_path = memory_bank_dir / "memory_bank.json"
        if not index_path.exists():
            raise FileNotFoundError(f"memory bank metadata not found: {index_path}")
        payload = json.loads(index_path.read_text())
        entries = payload.get("entries") or []
        entries = sorted(entries, key=lambda x: int(x.get("slot", 0)))
        if not entries:
            raise RuntimeError(f"memory bank is empty: {memory_bank_dir}")

        all_views: List[np.ndarray] = []
        for entry in entries:
            view_paths = entry.get("view_paths") or []
            if len(view_paths) != 4:
                raise RuntimeError(f"invalid memory entry (need 4 views): {entry}")
            views: List[np.ndarray] = []
            for p in view_paths:
                path = Path(p)
                if not path.is_absolute():
                    path = memory_bank_dir / path
                if not path.exists():
                    raise FileNotFoundError(f"memory view image not found: {path}")
                img_u8 = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
                if img_u8.shape[0] != input_size or img_u8.shape[1] != input_size:
                    img_u8 = np.asarray(
                        Image.fromarray(img_u8).resize((input_size, input_size))
                    )
                views.append(img_u8.astype(np.float32) / 255.0)
            all_views.append(np.stack(views, axis=0))
        return all_views

    def build_memory_bank(
        self,
        frame_paths: List[Path],
        frame_indices: List[int],
        session_root: Path,
    ) -> Dict:
        if not frame_paths:
            raise ValueError("frame_paths must not be empty")
        if len(frame_paths) != len(frame_indices):
            raise ValueError("frame_paths and frame_indices must have the same length")

        opt = copy.deepcopy(self.opt)
        opt.num_frames = 1
        opt.num_views = 4

        memory_bank_dir = session_root / "memory_bank"
        memory_bank_dir.mkdir(parents=True, exist_ok=True)
        for old in memory_bank_dir.glob("*.png"):
            try:
                old.unlink()
            except Exception:
                pass

        entries: List[Dict] = []
        for slot, (frame_idx, frame_path) in enumerate(zip(frame_indices, frame_paths)):
            ply_path = session_root / "gs_sequence" / f"{int(frame_idx):06d}.ply"
            if not ply_path.exists():
                raise FileNotFoundError(f"GS PLY not found for memory frame {frame_idx}: {ply_path}")
            gaussians = self._load_gaussians_from_ply(ply_path)
            rgb = _load_single_image(Path(frame_path), opt.input_size)
            best_azimuth = self._find_best_azimuth(gaussians, rgb, opt)
            best_azimuth = -best_azimuth
            mv_image = self._render_four_views_orthographic_with_offset(gaussians, opt, best_azimuth)

            view_paths: List[str] = []
            for v in range(mv_image.shape[0]):
                img_u8 = (mv_image[v] * 255.0).clip(0, 255).astype(np.uint8)
                out_path = memory_bank_dir / f"{slot:03d}_frame_{int(frame_idx):06d}_view{v:03d}.png"
                Image.fromarray(img_u8).save(out_path)
                view_paths.append(str(out_path))

            entries.append(
                {
                    "slot": int(slot),
                    "frame_idx": int(frame_idx),
                    "best_azimuth": float(best_azimuth),
                    "view_paths": view_paths,
                }
            )

        meta = {
            "count": len(entries),
            "entries": entries,
        }
        (memory_bank_dir / "memory_bank.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
        return {
            "status": "success",
            "memory_bank_dir": str(memory_bank_dir),
            "num_memory_frames": len(entries),
            "entries": entries,
        }

    def generate_segment_with_memory(
        self,
        frame_paths: List[Path],
        session_root: Path,
        start_frame_idx: int,
        total_clip_frames: int = 16,
        memory_bank_dir: Path | None = None,
        debug_ref_video: bool = False,
    ) -> Dict:
        if total_clip_frames <= 0:
            raise ValueError("total_clip_frames must be positive")
        if not frame_paths:
            raise ValueError("frame_paths must not be empty")

        opt = copy.deepcopy(self.opt)
        opt.num_views = 4
        if memory_bank_dir is None:
            memory_bank_dir = session_root / "memory_bank"

        memory_views = self._load_memory_bank_views(memory_bank_dir, opt.input_size)
        num_memory_frames = len(memory_views)
        if num_memory_frames >= total_clip_frames:
            raise ValueError(
                f"num_memory_frames ({num_memory_frames}) must be < total_clip_frames ({total_clip_frames})"
            )

        num_generate_frames = min(total_clip_frames - num_memory_frames, len(frame_paths))
        if num_generate_frames <= 0:
            raise RuntimeError("No future frames left after memory frames")

        opt.num_frames = num_memory_frames + num_generate_frames
        preview_dir = session_root / "preview" / "mv_memory"
        preview_dir.mkdir(parents=True, exist_ok=True)

        ref_video: List[np.ndarray] = []

        for slot, frame_views in enumerate(memory_views):
            ref_video.append(frame_views.astype(np.float32))
            for v in range(frame_views.shape[0]):
                img_u8 = (frame_views[v] * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_u8).save(preview_dir / f"mem_{slot:03d}_{v:03d}.png")

        template_views = memory_views[-1]
        for offset, path in enumerate(frame_paths[:num_generate_frames]):
            rgb = _load_single_image(path, opt.input_size)
            views = [rgb] + [template_views[v] for v in range(1, template_views.shape[0])]
            frame_views = np.stack(views, axis=0)
            ref_video.append(frame_views)

            frame_idx = start_frame_idx + offset
            for v in range(frame_views.shape[0]):
                img_u8 = (frame_views[v] * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_u8).save(preview_dir / f"{frame_idx:06d}_{v:03d}.png")

        ref_video_arr = np.stack(ref_video, axis=0)
        debug_ref_video_path: str | None = None
        if bool(debug_ref_video):
            viz_dir = session_root / "tmp" / "test_memory_debug"
            viz_dir.mkdir(parents=True, exist_ok=True)
            # [T, V, H, W, C] -> [T, H, V*W, C]
            viz_ref_video = ref_video_arr.transpose(0, 2, 1, 3, 4).reshape(
                ref_video_arr.shape[0],
                ref_video_arr.shape[2],
                ref_video_arr.shape[1] * ref_video_arr.shape[3],
                ref_video_arr.shape[4],
            )
            viz_name = (
                f"{start_frame_idx}_{start_frame_idx + num_generate_frames}_"
                f"memory_ref_video_debug.mp4"
            )
            viz_path = viz_dir / viz_name
            imageio.mimwrite(
                viz_path,
                (viz_ref_video * 255).clip(0, 255).astype(np.uint8),
                fps=10,
            )
            debug_ref_video_path = str(viz_path)

        gaussians_all_frame = self._forward_gaussians_sequence(ref_video_arr, opt, opt.num_frames)
        B = 1
        gaussians_all_frame = gaussians_all_frame.reshape(
            B, gaussians_all_frame.shape[0] // opt.batch_size, *gaussians_all_frame.shape[1:]
        )

        out_paths: List[Path] = []
        for t in range(num_generate_frames):
            clip_idx = num_memory_frames + t
            out_ply = session_root / "gs_sequence" / f"{start_frame_idx + t:06d}.ply"
            self._save_ply(gaussians_all_frame[:, clip_idx], out_ply)
            out_paths.append(out_ply)

        print_gpu_memory()
        return {
            "status": "success",
            "ply_paths": [str(p) for p in out_paths],
            "start_frame": start_frame_idx,
            "end_frame": start_frame_idx + len(out_paths) - 1,
            "mv_preview_dir": str(preview_dir),
            "memory_bank_dir": str(memory_bank_dir),
            "num_memory_frames": int(num_memory_frames),
            "num_generated_frames": int(num_generate_frames),
            "total_clip_frames": int(opt.num_frames),
            "debug_ref_video_path": debug_ref_video_path,
            "device": str(self.device),
        }

    def to_json(self) -> Dict:
        return {
            "device": str(self.device),
            "resume": self.opt.resume,
            "input_size": self.opt.input_size,
        }
