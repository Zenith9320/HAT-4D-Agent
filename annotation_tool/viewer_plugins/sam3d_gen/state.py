from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.ply_utils import SplatData


@dataclass
class Sam3DObjectState:
    points_xy: List[Tuple[int, int]] = field(default_factory=list)  # (x,y) in original image pixels
    point_labels: List[int] = field(default_factory=list)  # 1=positive, 0=negative
    mask: Optional[np.ndarray] = None  # bool[H,W] in original resolution
    logits: Optional[np.ndarray] = None  # float[256,256] SAM logits
    logits_key: Optional[str] = None  # backend handle for prev_logits (to refine)
    mask_path: Optional[str] = None  # path to mask png written by backend

    # Generation outputs
    sam3d_output: Optional[dict] = None
    splat: Optional[SplatData] = None  # in camera-local coords (viser convention)
    frame_handle: Any = None  # viser.FrameHandle (camera frame parent)
    gs_handle: Any = None  # viser.GaussianSplatHandle

    # Manual global scaling (used for interactive size adjustment during pose opt workflow).
    manual_scale: float = 1.0
    manual_scale_base_splat: Optional[SplatData] = None

    # Pose-application bookkeeping (to avoid double-application and enable pose refinement).
    pose_applied: bool = False
    gs_base_xyz: Any = None
    gs_base_rot: Any = None
    gs_base_scale: Any = None
    gs_base_min_kernel: Optional[float] = None

    # BBox transform gizmo (similar to manual_edit)
    bbox_frame_handle: Any = None
    bbox_box_handle: Any = None
    bbox_face_handles: Dict[str, Any] = field(default_factory=dict)
    bbox_face_defs: Dict[str, Tuple[np.ndarray, int]] = field(default_factory=dict)
    bbox_dimensions: Optional[np.ndarray] = None
    bbox_base_splat: Optional[SplatData] = None
    bbox_base_center: Optional[np.ndarray] = None
    bbox_base_wxyz: Optional[np.ndarray] = None
    bbox_base_dims: Optional[np.ndarray] = None
    bbox_updating: bool = False
    bbox_visible: bool = False
    transform_mode: str = "Translate"  # Translate | Rotate | Scale
    axis_mode: str = "World"
    axis_constraint: Tuple[bool, bool, bool] = (False, False, False)


class Sam3DGenState:
    def __init__(self) -> None:
        self.lock = threading.Lock()

        # Session state (frontend-local)
        self.gen_active: bool = False
        # If enabled, starting SAM3D Gen deletes the current frame's session PLY so subsequent
        # "Add Object" creates a fresh file (overwrite workflow).
        self.total_cover: bool = False
        # If enabled, automatically run pose optimization right after backend/local generation.
        self.pose_opt_after_gen: bool = True
        # If enabled, pad input image to square for frustum display + SAM/SAM3D.
        self.pad_square: bool = True
        self.pad_value: int = 255
        self.pad_info: tuple[int, int, int, int] | None = None  # (top, bottom, left, right)
        self.padded_image_path: str | None = None
        self.image_alpha_u8: Optional[np.ndarray] = None
        self.image_alpha_base_u8: Optional[np.ndarray] = None

        # Pose-opt settings (used for "pose_opt_after_gen" autop-run).
        self.pose_opt_steps: int = 160
        self.pose_opt_lr: float = 1e-3
        self.pose_opt_min_side: int = 512
        self.pose_opt_rgb_weight: float = 0.4
        self.pose_opt_center_reg_weight: float = 1e-2
        self.pose_opt_pose_reg_weight: float = 1e-3
        self.pose_opt_scale_reg_weight: float = 1e-5
        self.pose_opt_box_reg_weight: float = 1e-1
        self.pose_opt_debug_mode: bool = False
        # If enabled, pose optimization is allowed to refine rotation.
        self.pose_opt_optimize_rotation: bool = False
        # Optional per-axis hard lock for object center in world coordinates during pose optimization.
        self.pose_opt_lock_tx: bool = False
        self.pose_opt_lock_ty: bool = False
        self.pose_opt_lock_tz: bool = False
        # Pose opt running flag - used to disable transform controls
        self.pose_opt_running: bool = False

        # Image + frustum state
        self.frame_token: str | None = None
        self.image_flip_udlr: bool = False

        # If True, roll the frustum camera 180° around its forward axis instead of
        # rotating the image pixels. This keeps SAM3D inputs unmodified while fixing
        # the visual orientation in the frustum.
        self.frustum_roll_180: bool = False
        self.image_rgb_u8: Optional[np.ndarray] = None  # original (after flip_udlr for display consistency)
        self.image_disp_rgb_u8: Optional[np.ndarray] = None  # downsampled for frustum display
        self.image_downsample: int = 1

        self.frustum_handle: Any = None
        self.frustum_wxyz: Optional[np.ndarray] = None  # world_from_camera (wxyz)
        self.frustum_position: Optional[np.ndarray] = None  # world position
        self.frustum_aspect: float = 1.0
        self.frustum_scale: float = 0.35
        self.frustum_opacity: float = 1.0

        # Interaction flags
        self.interaction_mode: str = "segment"  # segment | transform
        # Pixel-pick mode is per-client (multiple browser tabs can connect).
        # Use string keys to avoid client_id type mismatches across viser versions.
        self.pixel_pick_mode_by_client: Dict[str, bool] = {}
        self.negative_point: bool = False
        self.eraser_mode: bool = False
        self.eraser_radius_px: int = 18

        # Display/perf controls
        # Large GS updates can stall the viewer; cap display points for interactive rendering.
        self.display_max_points: int = 150_000
        self.display_sample: str = "opacity"

        self.locked_pose: Dict[int, Dict[str, np.ndarray]] = {}

        # SAM / SAM3D config (defaults are repo-local where possible)
        self.sam_checkpoint: str | None = None
        self.sam_model_type: str = "vit_h"
        self.sam_device: str = "cuda"
        self.align_pytorch3d_camera_to_viser: bool = True

        self.sam3d_pipeline_config: str | None = None
        self.sam3d_depth_model_path: str | None = None
        self.sam3d_compile: bool = False
        self.sam3d_seed: int = 42

        # Lazy-loaded heavy objects
        self._sam_predictor: Any = None
        self._sam_model: Any = None
        self._sam3d_pipeline: Any = None
        self.sam_status: str = "uninitialized"
        self.sam3d_status: str = "uninitialized"
        self._sam_init_notified: bool = False
        self._sam3d_init_notified: bool = False
        self._sam_ready_notified: bool = False
        self._sam3d_ready_notified: bool = False
        self._sam_failed_notified: bool = False
        self._sam3d_failed_notified: bool = False

        # Backend polling (per client_id)
        self.backend_poll_events: Dict[int, threading.Event] = {}
        self.backend_poll_threads: Dict[int, threading.Thread] = {}

        # Objects
        self.objects: List[Sam3DObjectState] = [Sam3DObjectState()]
        self.current_object_idx: int = 0

        # UI handles (optional)
        self.status_handle: Any = None
        self.sam_status_handle: Any = None
        self.sam3d_status_handle: Any = None
        self.object_status_handle: Any = None
        self.mask_status_handle: Any = None
        self.gen_status_handle: Any = None
        # Per-client UI handles
        self.pose_opt_overlay_handles: Dict[Any, Any] = {}
        self.pose_opt_debug_handles: Dict[Any, Any] = {}
