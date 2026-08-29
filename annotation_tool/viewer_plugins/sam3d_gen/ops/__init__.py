"""SAM3D generation operations module.

This package provides the Sam3DGenOps class composed from multiple mixins
for handling SAM segmentation, SAM3D 3D generation, transforms, pose optimization,
and backend communication.
"""
from __future__ import annotations

from .base import Sam3DGenOpsBase
from .sam import SamMixin
from .sam3d import Sam3DMixin
from .transform import TransformMixin
from .pose_opt import PoseOptMixin
from .backend_client import BackendClientMixin


class Sam3DGenOps(
    SamMixin,
    Sam3DMixin,
    TransformMixin,
    PoseOptMixin,
    BackendClientMixin,
    Sam3DGenOpsBase,
):
    """Complete SAM3D generation operations class composed from mixins.

    Inherits functionality from:
    - Sam3DGenOpsBase: Core initialization, camera, status, UI setters, object management
    - SamMixin: SAM segmentation, frustum rendering, pixel picking
    - Sam3DMixin: SAM3D 3D generation, splat loading, session PLY management
    - TransformMixin: BBox gizmo transforms (translate, rotate, scale)
    - PoseOptMixin: Pose optimization to align generated objects with masks
    - BackendClientMixin: HTTP communication with SAM3D backend server
    """

    pass
