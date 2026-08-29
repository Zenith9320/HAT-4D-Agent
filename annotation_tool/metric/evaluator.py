from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio

from metric.utils import init_lpips, lpips_score, psnr_score


class MetricEvaluator:
	"""Shared image metric helper (PSNR / LPIPS) for numpy or torch inputs."""

	def __init__(self, device: torch.device, lpips_net: str = "vgg") -> None:
		self.device = device
		self.lpips = init_lpips(device, lpips_net)

	def psnr_lpips_from_tensors(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float | None]:
		pred_f = pred.to(self.device, dtype=torch.float32)
		gt_f = gt.to(self.device, dtype=torch.float32)
		pred_max = float(pred_f.max().item()) if pred_f.numel() > 0 else 0.0
		gt_max = float(gt_f.max().item()) if gt_f.numel() > 0 else 0.0
		if pred_max <= 1.5 and gt_max > 1.5:
			pred_f = pred_f * 255.0
		elif pred_max > 1.5 and gt_max <= 1.5:
			gt_f = gt_f * 255.0
		psnr_val = psnr_score(pred_f, gt_f)
		lpips_val = lpips_score(pred_f, gt_f, self.lpips)
		return {"psnr": psnr_val, "lpips": lpips_val}

	def psnr_lpips_from_numpy(self, pred_np: np.ndarray, gt_np: np.ndarray) -> Dict[str, float | None]:
		pred_t = torch.from_numpy(pred_np.astype(np.float32)).permute(2, 0, 1)
		gt_t = torch.from_numpy(gt_np.astype(np.float32)).permute(2, 0, 1)
		return self.psnr_lpips_from_tensors(pred_t, gt_t)

	def fix_before_after_metrics(self, before_np: np.ndarray, after_np: np.ndarray, gt_np: np.ndarray) -> Dict[str, float | None]:
		"""Compute before/after/delta metrics for ROI fix flows."""
		before = self.psnr_lpips_from_numpy(before_np, gt_np)
		after = self.psnr_lpips_from_numpy(after_np, gt_np)
		print({
			"psnr_before": before.get("psnr"),
			"psnr_after": after.get("psnr"),
			"psnr_delta": (after.get("psnr") - before.get("psnr")) if before.get("psnr") is not None else None,
			"lpips_before": before.get("lpips"),
			"lpips_after": after.get("lpips"),
			"lpips_delta": (after.get("lpips") - before.get("lpips")) if None not in (after.get("lpips"), before.get("lpips")) else None,
		})
		return {
			"psnr_before": before.get("psnr"),
			"psnr_after": after.get("psnr"),
			"psnr_delta": (after.get("psnr") - before.get("psnr")) if before.get("psnr") is not None else None,
			"lpips_before": before.get("lpips"),
			"lpips_after": after.get("lpips"),
			"lpips_delta": (after.get("lpips") - before.get("lpips")) if None not in (after.get("lpips"), before.get("lpips")) else None,
		}


class L4GMEvaluator(MetricEvaluator):
	"""Metric runner using L4GM for rendering eval views."""

	def __init__(self, l4gm, eval_azimuths: Sequence[float] | None = None) -> None:  # noqa: ANN001
		super().__init__(device=l4gm.device)
		self.l4gm = l4gm
		self.opt = l4gm.opt
		self.eval_azimuths = list(eval_azimuths) if eval_azimuths is not None else [-75.0, 15.0, 105.0, 195.0]

	def _render_view(self, gaussians: torch.Tensor, azimuth_deg: float) -> np.ndarray:
		device = gaussians.device
		bg_color = torch.tensor([255, 255, 255], dtype=torch.float32, device=device) / 255.0

		tan_half_fov = math.tan(0.5 * math.radians(self.opt.fovy))
		proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
		proj_matrix[0, 0] = 1.0 / tan_half_fov
		proj_matrix[1, 1] = 1.0 / tan_half_fov
		proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
		proj_matrix[3, 2] = -(self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
		proj_matrix[2, 3] = 1.0

		from kiui.cam import orbit_camera  # Lazy import to keep dependencies local.

		cam_poses = torch.from_numpy(
			orbit_camera(0, azimuth_deg, radius=self.opt.cam_radius, opengl=True)
		).unsqueeze(0).to(device)
		cam_poses[:, :3, 1:3] *= -1.0
		cam_view = torch.inverse(cam_poses).transpose(1, 2)
		cam_view_proj = cam_view @ proj_matrix
		cam_pos = -cam_poses[:, :3, 3]

		result = self.l4gm.model.gs.render(
			gaussians,
			cam_view.unsqueeze(0),
			cam_view_proj.unsqueeze(0),
			cam_pos.unsqueeze(0),
			bg_color=bg_color,
		)
		image = result["image"].squeeze(1)
		image = F.interpolate(image, (self.opt.input_size, self.opt.input_size))
		image_np = (
			image.permute(0, 2, 3, 1)
			.contiguous()
			.float()
			.cpu()
			.numpy()[0]
		)
		return image_np

	def _load_gt(self, eval_root: Path, view_idx: int, frame_token: str) -> np.ndarray:
		from PIL import Image  # Lazy import.

		candidates = [
			eval_root / f"eval_{view_idx}" / f"{frame_token}.png",
			eval_root / f"eval_{view_idx}" / f"{frame_token}.jpg",
			eval_root / f"eval_{view_idx}" / f"{frame_token}.jpeg",
			eval_root / f"eval_{view_idx}" / f"{int(frame_token):06d}.png",
		]


		for c in candidates:
			if c.exists():
				img = Image.open(c).convert("RGBA")
				if img.mode == "RGBA":
					bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
					img = Image.alpha_composite(bg, img)
				img = img.convert("RGB").resize((self.opt.input_size, self.opt.input_size))
				return np.asarray(img).astype(np.float32) / 255.0

		raise FileNotFoundError(f"GT image not found for view eval_{view_idx}, token={frame_token}")

	def score_frame(
		self,
		ply_path: Path,
		eval_root: Path,
		frame_token: str,
		preview_dir: Path | None = None,
	) -> Dict:
		if not eval_root or not eval_root.exists():
			raise FileNotFoundError(f"eval_gt_dir not found: {eval_root}")
		gaussians = self.l4gm._load_gaussians_from_ply(ply_path)
		psnrs: List[float] = []
		lpips_scores: List[float] = []

		for idx, azi in enumerate(self.eval_azimuths):
			pred_np = self._render_view(gaussians, azi)
			gt_np = self._load_gt(eval_root, idx, frame_token)

			pred_img = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
			gt_img = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)

			# Match previous behavior: metrics on 0-255 space, PSNR data_range=255, LPIPS input in [-1, 1].
			pred_rgb_255 = pred_img * 255.0
			gt_rgb_255 = gt_img * 255.0
			psnrs.append(float(peak_signal_noise_ratio(pred_rgb_255, gt_rgb_255, data_range=255.0).item()))

			pred_lp = pred_rgb_255 / 127.5 - 1.0
			gt_lp = gt_rgb_255 / 127.5 - 1.0
			lp_val = lpips_score(pred_lp.squeeze(0), gt_lp.squeeze(0), self.lpips)
			lpips_scores.append(lp_val if lp_val is not None else 0.0)

			if preview_dir is not None:
				try:
					preview_dir.mkdir(parents=True, exist_ok=True)
					pred_out = preview_dir / f"{frame_token}_eval{idx}_pred.png"
					gt_out = preview_dir / f"{frame_token}_eval{idx}_gt.png"
					iio.imwrite(pred_out, (pred_np * 255.0).astype(np.uint8))
					iio.imwrite(gt_out, (gt_np * 255.0).astype(np.uint8))
				except Exception as exc:  # noqa: BLE001
					print(f"[metric] failed to dump debug images for frame {frame_token} view {idx}: {exc}")

		psnr_avg = float(np.mean(psnrs)) if psnrs else 0.0
		lpips_avg = float(np.mean(lpips_scores)) if lpips_scores else 0.0
		return {
			"frame": frame_token,
			"psnr": psnr_avg,
			"lpips": lpips_avg,
			"psnr_per_view": psnrs,
			"lpips_per_view": lpips_scores,
		}
