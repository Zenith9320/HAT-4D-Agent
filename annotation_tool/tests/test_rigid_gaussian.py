import math

import torch

from model4dgen.rigid_gaussian import (
    apply_rigid_transform_to_gaussians,
    estimate_weighted_rigid_transform,
)


def _rotation_y(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_weighted_rigid_transform_recovers_motion():
    torch.manual_seed(7)
    source = torch.randn(200, 3)
    rotation = _rotation_y(0.4)
    translation = torch.tensor([0.3, -0.2, 0.5])
    target = source @ rotation.T + translation

    estimated_rotation, estimated_translation = estimate_weighted_rigid_transform(
        source, target, torch.ones(source.shape[0])
    )

    assert torch.allclose(estimated_rotation, rotation, atol=1e-5)
    assert torch.allclose(estimated_translation, translation, atol=1e-5)


def test_apply_rigid_transform_preserves_non_pose_attributes():
    canonical = torch.zeros(1, 4, 14)
    canonical[..., 0:3] = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    canonical[..., 3:7] = torch.rand(1, 4, 4)
    canonical[..., 7] = 1.0
    canonical[..., 11:14] = torch.rand(1, 4, 3)
    rotation = _rotation_y(0.25)
    translation = torch.tensor([1.0, 2.0, 3.0])

    transformed = apply_rigid_transform_to_gaussians(canonical, rotation, translation)

    assert torch.allclose(transformed[..., 0:3], canonical[..., 0:3] @ rotation.T + translation)
    assert torch.equal(transformed[..., 3:7], canonical[..., 3:7])
    assert torch.equal(transformed[..., 11:14], canonical[..., 11:14])
    assert torch.allclose(torch.linalg.vector_norm(transformed[..., 7:11], dim=-1), torch.ones(1, 4))
