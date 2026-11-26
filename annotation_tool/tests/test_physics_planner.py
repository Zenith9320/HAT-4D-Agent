from types import SimpleNamespace

from viewer_plugins.vlm_assistor.agents.core import PhysicsPlanningAgent
from viewer_plugins.vlm_assistor.state import VLMAssistorState


class _FakeVLM:
    def __init__(self, properties):
        self.properties = properties

    def infer_physical_properties(self, **_kwargs):
        return {"ok": True, "physical_properties": dict(self.properties)}


def _plan(properties: dict) -> dict:
    state = VLMAssistorState()
    state.object_description = "unseen object"
    agent = PhysicsPlanningAgent(SimpleNamespace(), state, None, _FakeVLM(properties))
    return agent.run(start_frame=0, num_frames=8)["plan"]


def test_rigid_object_plan():
    plan = _plan({"rigidity": 0.95, "elasticity": 0.05, "deformation_observed": 0.05, "confidence": 0.9})
    assert plan["representation"] == "rigid"
    assert plan["tool"] == "physics.gaussian_constraint_project"


def test_elastic_object_plan():
    plan = _plan({"rigidity": 0.3, "elasticity": 0.9, "deformation_observed": 0.8, "confidence": 0.85})
    assert plan["representation"] == "elastic"
    assert "topology_change" in plan["forbidden_motion"]


def test_articulated_object_plan():
    plan = _plan({"rigidity": 0.5, "articulatedness": 0.9, "confidence": 0.8})
    assert plan["representation"] == "articulated"
    assert plan["tool"] == "physics.gaussian_constraint_project"


def test_cloth_object_plan():
    plan = _plan({"rigidity": 0.1, "sheet_likeness": 0.9, "confidence": 0.8})
    assert plan["representation"] == "cloth"
    assert plan["tool"] == "physics.gaussian_constraint_project"


def test_unknown_object_uses_conservative_fallback():
    plan = _plan({"rigidity": 0.45, "elasticity": 0.4, "deformation_observed": 0.5, "confidence": 0.4})
    assert plan["representation"] == "free_deform"
    assert plan["confidence"] < 0.5
