import math

import torch

from model4dgen.gaussian_physics import (
    derive_projection_parameters,
    project_gaussian_sequence,
)


def _sequence() -> torch.Tensor:
    torch.manual_seed(13)
    base = torch.zeros(64, 14)
    base[:, 0:3] = torch.randn(64, 3)
    base[:, 3] = 0.9
    base[:, 4:7] = 0.05
    base[:, 7] = 1.0
    base[:, 11:14] = torch.rand(64, 3)
    frames = []
    for index in range(3):
        angle = 0.08 * index
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        frame = base.clone()
        frame[:, 0:3] = base[:, 0:3] @ rotation.T + torch.tensor([0.1 * index, 0.0, 0.0])
        frame[:, 0] += torch.linspace(-0.2, 0.2, 64) * index
        frames.append(frame)
    return torch.stack(frames)


def _plan(**properties):
    return {
        "representation": "test",
        "confidence": 1.0,
        "constraints": {"temporal_smoothness": 0.8},
        "physical_properties": {
            "topology_stability": 1.0,
            "deformation_observed": 0.8,
            **properties,
        },
    }


def test_rigidity_continuously_reduces_deformation():
    sequence = _sequence()
    rigid, _ = project_gaussian_sequence(
        sequence, _plan(rigidity=1.0, elasticity=0.0)
    )
    elastic, _ = project_gaussian_sequence(
        sequence, _plan(rigidity=0.1, elasticity=0.9)
    )
    rigid_change = torch.linalg.vector_norm(rigid[2, :, 0:3] - rigid[0, :, 0:3], dim=-1).std()
    elastic_change = torch.linalg.vector_norm(elastic[2, :, 0:3] - elastic[0, :, 0:3], dim=-1).std()
    assert rigid_change < elastic_change


def test_all_representations_use_same_projector_contract():
    sequence = _sequence()
    property_sets = {
        "rigid": {"rigidity": 0.95},
        "elastic": {"rigidity": 0.2, "elasticity": 0.9},
        "cloth": {"rigidity": 0.1, "sheet_likeness": 0.95},
        "articulated": {"rigidity": 0.5, "articulatedness": 0.95},
        "free_deform": {"rigidity": 0.2, "elasticity": 0.2},
    }
    for representation, properties in property_sets.items():
        plan = _plan(**properties)
        plan["representation"] = representation
        output, metadata = project_gaussian_sequence(sequence, plan)
        assert output.shape == sequence.shape
        assert torch.isfinite(output).all()
        assert metadata["representation"] == representation
        assert len(metadata["transforms"]) == sequence.shape[0]


def test_low_confidence_keeps_projection_conservative():
    uncertain = derive_projection_parameters(
        {"confidence": 0.0, "physical_properties": {"rigidity": 1.0}}
    )
    confident = derive_projection_parameters(
        {"confidence": 1.0, "physical_properties": {"rigidity": 1.0}}
    )
    assert uncertain.deformation_gain > confident.deformation_gain
    assert uncertain.deformation_gain == 1.0
