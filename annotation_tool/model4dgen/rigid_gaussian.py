from __future__ import annotations

import torch
import torch.nn.functional as F


def estimate_weighted_rigid_transform(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate target ~= source @ R.T + t without scale."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[-1] != 3:
        raise ValueError("source and target must have matching [N, 3] shapes")

    dtype = source.dtype
    source_f = source.float()
    target_f = target.float()
    valid = torch.isfinite(source_f).all(dim=-1) & torch.isfinite(target_f).all(dim=-1)
    if weights is None:
        weights_f = torch.ones(source.shape[0], device=source.device, dtype=torch.float32)
    else:
        weights_f = torch.nan_to_num(weights.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid &= weights_f > 0
    if int(valid.sum()) < 3:
        raise RuntimeError("not enough valid Gaussian correspondences for rigid fitting")

    source_f = source_f[valid]
    target_f = target_f[valid]
    weights_f = weights_f[valid]
    weights_f = weights_f / weights_f.sum().clamp_min(1e-8)

    source_center = (source_f * weights_f[:, None]).sum(dim=0)
    target_center = (target_f * weights_f[:, None]).sum(dim=0)
    source_zero = source_f - source_center
    target_zero = target_f - target_center
    covariance = source_zero.T @ (weights_f[:, None] * target_zero)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if torch.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation.to(dtype=dtype), translation.to(dtype=dtype)


def rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")
    matrix = rotation.float()
    candidates = torch.stack(
        (
            1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2],
            1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
            1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2],
            1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2],
        )
    ).clamp_min(0)
    index = int(torch.argmax(candidates))
    root = torch.sqrt(candidates[index]).clamp_min(1e-8)
    if index == 0:
        quat = torch.stack((root, matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]))
    elif index == 1:
        quat = torch.stack((matrix[2, 1] - matrix[1, 2], root, matrix[0, 1] + matrix[1, 0], matrix[0, 2] + matrix[2, 0]))
    elif index == 2:
        quat = torch.stack((matrix[0, 2] - matrix[2, 0], matrix[0, 1] + matrix[1, 0], root, matrix[1, 2] + matrix[2, 1]))
    else:
        quat = torch.stack((matrix[1, 0] - matrix[0, 1], matrix[0, 2] + matrix[2, 0], matrix[1, 2] + matrix[2, 1], root))
    quat[index] *= 2
    other = torch.arange(4, device=quat.device) != index
    quat[other] /= 2 * root
    return F.normalize(quat.to(dtype=rotation.dtype), dim=0, eps=1e-8)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply wxyz quaternions with broadcasting."""
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def apply_rigid_transform_to_gaussians(
    canonical: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply one rigid transform while preserving Gaussian shape and appearance."""
    if canonical.ndim != 3 or canonical.shape[0] != 1 or canonical.shape[-1] < 14:
        raise ValueError("canonical must have shape [1, N, >=14]")
    output = canonical.clone()
    output[..., 0:3] = canonical[..., 0:3] @ rotation.T + translation
    global_quat = rotation_matrix_to_quaternion(rotation).view(1, 1, 4)
    output[..., 7:11] = F.normalize(
        quaternion_multiply(global_quat, canonical[..., 7:11]), dim=-1, eps=1e-8
    )
    return output
