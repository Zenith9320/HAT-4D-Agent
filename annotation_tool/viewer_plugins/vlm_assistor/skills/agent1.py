from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

SKILL_VLM_POINT_PROMPT_PLANNER = "vlm.point_prompt_planner"
SKILL_VLM_MASK_EVALUATOR = "vlm.mask_evaluator"
SKILL_SAM_SEGMENT_WITH_POINTS = "sam.segment_with_points"
SKILL_SAM_EVALUATE_MASK = "sam.evaluate_mask"
SKILL_SAM3D_ENSURE_SESSION = "sam3d.session.ensure"
SKILL_SAM3D_GENERATE_AND_WAIT = "sam3d.generate_and_wait"
SKILL_SAM3D_NEW_OBJECT = "sam3d.object.new"


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    fn: Callable[..., dict[str, Any]]


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillEntry] = {}

    def register(self, *, name: str, description: str, fn: Callable[..., dict[str, Any]]) -> None:
        self._skills[str(name)] = SkillEntry(name=str(name), description=str(description), fn=fn)

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        entry = self._skills[str(name)]
        return entry.fn(**kwargs)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": x.name, "description": x.description}
            for x in self._skills.values()
        ]


class Agent1SkillBundle:
    """Skill bundle for Agent1 (point planning + mask eval + SAM/SAM3D ops)."""

    def __init__(self, *, bridge, vlm_client) -> None:
        self.bridge = bridge
        self.vlm_client = vlm_client
        self.registry = SkillRegistry()
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.registry.register(
            name=SKILL_VLM_POINT_PROMPT_PLANNER,
            description="Plan positive/negative click points for SAM from image + object context.",
            fn=self._skill_vlm_point_prompt_planner,
        )
        self.registry.register(
            name=SKILL_VLM_MASK_EVALUATOR,
            description="Evaluate segmentation mask quality with VLM.",
            fn=self._skill_vlm_mask_evaluator,
        )
        self.registry.register(
            name=SKILL_SAM_SEGMENT_WITH_POINTS,
            description="Run SAM segmentation with positive/negative points.",
            fn=self._skill_sam_segment_with_points,
        )
        self.registry.register(
            name=SKILL_SAM_EVALUATE_MASK,
            description="Run SAM-side mask heuristic evaluation.",
            fn=self._skill_sam_evaluate_mask,
        )
        self.registry.register(
            name=SKILL_SAM3D_ENSURE_SESSION,
            description="Ensure SAM3D session is active.",
            fn=self._skill_sam3d_ensure_session,
        )
        self.registry.register(
            name=SKILL_SAM3D_GENERATE_AND_WAIT,
            description="Generate current object in SAM3D and wait until done.",
            fn=self._skill_sam3d_generate_and_wait,
        )
        self.registry.register(
            name=SKILL_SAM3D_NEW_OBJECT,
            description="Create/select next object slot in SAM3D.",
            fn=self._skill_sam3d_new_object,
        )

    def _skill_vlm_point_prompt_planner(
        self,
        *,
        object_name: str,
        image_rgb_u8,
        non_target_objects: list[str] | None = None,
        points_per_object: int = 3,
        negative_points: int = 1,
        prior_fail_reason: str | None = None,
        prior_fail_analysis: dict[str, Any] | None = None,
        backend_mode: str | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        return self.vlm_client.suggest_click_points_for_object(
            object_name=object_name,
            image_rgb_u8=image_rgb_u8,
            non_target_objects=non_target_objects,
            points_per_object=points_per_object,
            negative_points=negative_points,
            prior_fail_reason=prior_fail_reason,
            prior_fail_analysis=prior_fail_analysis,
            backend_mode=backend_mode,
            timeout=timeout,
        )

    def _skill_vlm_mask_evaluator(
        self,
        *,
        object_name: str,
        image_rgb_u8,
        mask_bool,
        positive_points: list[Any] | None = None,
        negative_points: list[Any] | None = None,
        non_target_objects: list[str] | None = None,
        backend_mode: str | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        return self.vlm_client.evaluate_mask_with_vlm(
            object_name=object_name,
            image_rgb_u8=image_rgb_u8,
            mask_bool=mask_bool,
            positive_points=positive_points,
            negative_points=negative_points,
            non_target_objects=non_target_objects,
            backend_mode=backend_mode,
            timeout=timeout,
        )

    def _skill_sam_segment_with_points(
        self,
        *,
        client,
        points: list[Any] | None = None,
        negative_points: list[Any] | None = None,
        clear_existing: bool = True,
    ) -> dict[str, Any]:
        return self.bridge.registry.call(
            "sam.segment_with_points",
            client=client,
            points=points or [],
            negative_points=negative_points or [],
            clear_existing=bool(clear_existing),
        )

    def _skill_sam_evaluate_mask(
        self,
        *,
        positive_points: list[Any] | None = None,
        negative_points: list[Any] | None = None,
        min_area_ratio: float = 0.0,
        max_area_ratio: float = 1.0,
        min_positive_coverage: float = 0.0,
        max_negative_coverage: float = 1.0,
    ) -> dict[str, Any]:
        return self.bridge.registry.call(
            "sam.evaluate_mask",
            client=None,
            positive_points=positive_points or [],
            negative_points=negative_points or [],
            min_area_ratio=float(min_area_ratio),
            max_area_ratio=float(max_area_ratio),
            min_positive_coverage=float(min_positive_coverage),
            max_negative_coverage=float(max_negative_coverage),
        )

    def _skill_sam3d_ensure_session(self, *, client) -> dict[str, Any]:
        return self.bridge.registry.call("session.ensure_sam3d", client=client)

    def _skill_sam3d_generate_and_wait(
        self,
        *,
        client,
        timeout: float = 2400.0,
        poll_interval: float = 0.5,
    ) -> dict[str, Any]:
        return self.bridge.registry.call(
            "sam3d.generate_and_wait",
            client=client,
            timeout=float(timeout),
            poll_interval=float(poll_interval),
        )

    def _skill_sam3d_new_object(self, *, client) -> dict[str, Any]:
        return self.bridge.registry.call("object.new", client=client)
