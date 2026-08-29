from __future__ import annotations

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import peak_signal_noise_ratio

__all__ = [
    "init_lpips",
    "psnr_score",
    "lpips_score",
]


def init_lpips(device: torch.device, net_type: str = "vgg") -> LearnedPerceptualImagePatchSimilarity | None:
    """Initialize LPIPS on the given device; return None if it fails."""
    try:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, reduction="mean").to(device)
        lpips.eval()
        return lpips
    except Exception as exc:  # noqa: BLE001
        print(f"LPIPS init failed: {exc}")
        return None


def _detect_data_range(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Infer data range like tmp_metric: 255 for 0-255 inputs, else 1.0."""
    max_val = torch.max(torch.stack([pred.max(), gt.max()])).item()
    return 255.0 if max_val > 1.5 else 1.0


def psnr_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR via torchmetrics (same call style as tmp_metric)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    pred = pred.float()
    gt = gt.float()
    data_range = _detect_data_range(pred, gt)
    pred = pred.clamp(0.0, data_range)
    gt = gt.clamp(0.0, data_range)
    return float(peak_signal_noise_ratio(pred, gt, data_range=data_range).item())


def lpips_score(pred: torch.Tensor, gt: torch.Tensor, lpips: LearnedPerceptualImagePatchSimilarity | None) -> float | None:
    """LPIPS aligned with tmp_metric (inputs scaled to [-1,1])."""
    if lpips is None:
        return None
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    pred = pred.float().to(lpips.device)
    gt = gt.float().to(lpips.device)
    data_range = _detect_data_range(pred, gt)
    if data_range > 1.5:
        pred_n = pred / 127.5 - 1.0
        gt_n = gt / 127.5 - 1.0
    else:
        pred_n = pred.mul(2.0).sub(1.0)
        gt_n = gt.mul(2.0).sub(1.0)
    pred_n = pred_n.clamp(-1.0, 1.0)
    gt_n = gt_n.clamp(-1.0, 1.0)
    with torch.no_grad():
        return float(lpips(pred_n, gt_n).detach().cpu().item())
