from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .rigid_gaussian import (
    apply_rigid_transform_to_gaussians,
    estimate_weighted_rigid_transform,
)


@dataclass(frozen=True)
class ProjectionParameters:
    deformation_gain: float
    spatial_smoothing: float
    temporal_smoothing: float
    scale_preservation: float
    appearance_preservation: float
    max_residual_ratio: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _score(values: dict[str, Any], key: str, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(values.get(key, default))))
    except Exception:
        return default


def derive_projection_parameters(plan: dict[str, Any] | None) -> ProjectionParameters:
    """Convert semantic physics estimates into continuous projection strengths."""
    plan = dict(plan or {})
    properties = dict(plan.get("physical_properties", {}) or {})
    constraints = dict(plan.get("constraints", {}) or {})

    rigidity = _score(properties, "rigidity", 0.25)
    elasticity = _score(properties, "elasticity", 0.35)
    articulatedness = _score(properties, "articulatedness", 0.0)
    sheet_likeness = _score(properties, "sheet_likeness", 0.0)
    topology = _score(properties, "topology_stability", 0.8)
    observed = _score(properties, "deformation_observed", 0.5)
    confidence = _score(plan, "confidence", 0.0)

    # Low-confidence plans stay close to the original model output. Confident plans
    # progressively replace high-frequency deformation with coherent motion.
    physical_deformability = (
        0.50 * (1.0 - rigidity)
        + 0.24 * elasticity
        + 0.16 * articulatedness
        + 0.10 * sheet_likeness
    )
    confident_gain = physical_deformability * (0.45 + 0.55 * observed)
    deformation_gain = (1.0 - confidence) + confidence * confident_gain
    deformation_gain = max(0.0, min(1.0, deformation_gain))

    requested_temporal = _score(constraints, "temporal_smoothness", 0.65)
    spatial_smoothing = confidence * topology * (
        0.35 + 0.35 * elasticity + 0.20 * sheet_likeness + 0.10 * articulatedness
    )
    temporal_smoothing = confidence * requested_temporal * (0.45 + 0.55 * topology)
    scale_preservation = confidence * max(
        rigidity,
        topology,
        _score(constraints, "scale_preservation", 0.0),
    )
    appearance_preservation = confidence * (0.55 + 0.45 * topology)
    max_residual_ratio = 0.02 + 0.28 * deformation_gain + 0.12 * articulatedness

    return ProjectionParameters(
        deformation_gain=max(0.0, min(1.0, deformation_gain)),
        spatial_smoothing=max(0.0, min(1.0, spatial_smoothing)),
        temporal_smoothing=max(0.0, min(0.95, temporal_smoothing)),
        scale_preservation=max(0.0, min(1.0, scale_preservation)),
        appearance_preservation=max(0.0, min(1.0, appearance_preservation)),
        max_residual_ratio=max(0.02, min(0.5, max_residual_ratio)),
    )


def _smooth_residual_voxels(
    source_xyz: torch.Tensor,
    residual: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if strength <= 0.0 or source_xyz.shape[0] < 8:
        return residual
    extent = (source_xyz.amax(dim=0) - source_xyz.amin(dim=0)).clamp_min(1e-5)
    # A bounded grid keeps memory linear in point count and creates a coherent,
    # low-frequency deformation field without assuming an object category.
    bins = max(6, min(32, int(round(8 + 20 * strength))))
    coords = ((source_xyz - source_xyz.amin(dim=0)) / extent * (bins - 1)).long()
    keys = coords[:, 0] + bins * (coords[:, 1] + bins * coords[:, 2])
    unique_keys, inverse = torch.unique(keys, sorted=False, return_inverse=True)
    sums = torch.zeros(
        (unique_keys.shape[0], 3), device=residual.device, dtype=residual.dtype
    )
    counts = torch.zeros(
        (unique_keys.shape[0], 1), device=residual.device, dtype=residual.dtype
    )
    sums.index_add_(0, inverse, residual)
    counts.index_add_(
        0,
        inverse,
        torch.ones((residual.shape[0], 1), device=residual.device, dtype=residual.dtype),
    )
    averaged = sums[inverse] / counts[inverse].clamp_min(1.0)
    return torch.lerp(residual, averaged, strength)


def _limit_residual(
    source_xyz: torch.Tensor,
    residual: torch.Tensor,
    max_residual_ratio: float,
) -> torch.Tensor:
    center = source_xyz.mean(dim=0)
    radius = (
        torch.linalg.vector_norm(source_xyz.float() - center.float(), dim=-1)
        .quantile(0.9)
        .clamp_min(1e-4)
        .to(dtype=residual.dtype)
    )
    limit = radius * max_residual_ratio
    norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True).clamp_min(1e-8)
    return residual * torch.clamp(limit / norm, max=1.0)


def project_gaussian_sequence(
    gaussians: torch.Tensor,
    plan: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply one category-agnostic physical projection to a [T,N,14] sequence."""
    if gaussians.ndim != 3 or gaussians.shape[-1] < 14:
        raise ValueError("gaussians must have shape [T, N, >=14]")
    if gaussians.shape[0] == 0:
        raise ValueError("gaussian sequence is empty")

    params = derive_projection_parameters(plan)
    reference = gaussians[0]
    source_xyz = reference[:, 0:3]
    projected_frames: list[torch.Tensor] = []
    transforms: list[dict[str, Any]] = []
    previous_residual: torch.Tensor | None = None

    for frame_index, predicted in enumerate(gaussians):
        if frame_index == 0:
            rotation = torch.eye(3, device=gaussians.device, dtype=gaussians.dtype)
            translation = torch.zeros(3, device=gaussians.device, dtype=gaussians.dtype)
        else:
            weights = (reference[:, 3].clamp_min(0) * predicted[:, 3].clamp_min(0)).sqrt()
            rotation, translation = estimate_weighted_rigid_transform(
                source_xyz, predicted[:, 0:3], weights
            )

        rigid_frame = apply_rigid_transform_to_gaussians(
            reference.unsqueeze(0), rotation, translation
        )[0]
        residual = predicted[:, 0:3] - rigid_frame[:, 0:3]
        residual = _smooth_residual_voxels(
            source_xyz, residual, params.spatial_smoothing
        )
        residual = _limit_residual(source_xyz, residual, params.max_residual_ratio)
        residual = residual * params.deformation_gain
        if previous_residual is not None:
            residual = torch.lerp(
                residual, previous_residual, params.temporal_smoothing
            )
        previous_residual = residual

        output = predicted.clone()
        output[:, 0:3] = rigid_frame[:, 0:3] + residual
        output[:, 3:7] = torch.lerp(
            predicted[:, 3:7], reference[:, 3:7], params.scale_preservation
        )
        output[:, 7:11] = F.normalize(
            torch.lerp(
                predicted[:, 7:11],
                rigid_frame[:, 7:11],
                max(params.scale_preservation, 1.0 - params.deformation_gain),
            ),
            dim=-1,
            eps=1e-8,
        )
        output[:, 11:14] = torch.lerp(
            predicted[:, 11:14],
            reference[:, 11:14],
            params.appearance_preservation,
        )
        projected_frames.append(output)
        transforms.append(
            {
                "frame_offset": frame_index,
                "rotation": rotation.detach().float().cpu().tolist(),
                "translation": translation.detach().float().cpu().tolist(),
                "mean_residual": float(
                    torch.linalg.vector_norm(residual, dim=-1).mean().detach().cpu()
                ),
            }
        )

    metadata = {
        "version": 1,
        "representation": str((plan or {}).get("representation", "free_deform")),
        "parameters": params.to_dict(),
        "transforms": transforms,
    }
    return torch.stack(projected_frames, dim=0), metadata
