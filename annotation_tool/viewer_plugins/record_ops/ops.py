"""Recording operations for the record_ops plugin.

Handles event callbacks from SAM3D plugin and saves recording data.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .state import RecordOpsState, RecordedClick, RecordedObject
from .renderer import SixViewRenderer


class RecordOps:
    """Operations class for recording SAM3D events."""

    def __init__(self, ctx, state: RecordOpsState):
        self.ctx = ctx
        self.state = state
        self.renderer: Optional[SixViewRenderer] = None
        self.workspace = None

    def _app(self):
        return self.ctx.app

    def set_recording_enabled(self, enabled: bool) -> None:
        print(f"[RecordOps] set_recording_enabled({enabled})")
        with self.state.lock:
            self.state.recording_enabled = bool(enabled)

    def is_recording(self) -> bool:
        with self.state.lock:
            return self.state.recording_enabled and self.state.session_active

    # ----------------------------
    # Event handlers (called from plugin hooks)
    # ----------------------------

    def on_sam3d_begin(self, client, frame_token: str, frame_idx: int) -> None:
        """Called when SAM3D Gen begins. Saves pre-edit 6-view images."""
        print(f"[RecordOps] on_sam3d_begin called: frame_token={frame_token}, frame_idx={frame_idx}")
        print(f"[RecordOps] recording_enabled={self.state.recording_enabled}")

        with self.state.lock:
            if not self.state.recording_enabled:
                print("[RecordOps] Recording is disabled, skipping...")
                return
            # Start new session
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.state.session_id = session_id
            self.state.frame_token = frame_token
            self.state.started_at = datetime.now().isoformat()
            self.state.session_active = True
            self.state.objects = [RecordedObject()]
            self.state.current_object_idx = 0

            # Create session directory
            # workspace is a WorkspaceManager, use .root to get the Path
            workspace_root = self.workspace.root if hasattr(self.workspace, 'root') else Path(self.workspace)
            session_dir = workspace_root / "record_ops" / f"session_{session_id}"
            session_dir.mkdir(parents=True, exist_ok=True)
            self.state.session_dir = session_dir
            print(f"[RecordOps] Session started: {session_dir}")

        # Render and save pre-edit 6-view images
        self._save_pre_edit_views(frame_idx)

        # Save initial meta.json
        self._save_meta()

    def on_pixel_pick(
        self,
        client,
        x: int,
        y: int,
        label: int,
    ) -> None:
        """Called when user clicks a pixel during SAM3D segmentation."""
        with self.state.lock:
            if not self.state.recording_enabled or not self.state.session_active:
                return
            # Record click
            click = RecordedClick(
                x=int(x),
                y=int(y),
                label=int(label),
                timestamp=time.time(),
            )
            obj_idx = self.state.current_object_idx
            if obj_idx < len(self.state.objects):
                self.state.objects[obj_idx].clicks.append(click)

    def on_object_complete(
        self,
        client,
        obj_idx: int,
        mask_path: Optional[str],
        rgb_image: Optional[np.ndarray],
        bbox_center: Optional[np.ndarray],
        bbox_dimensions: Optional[np.ndarray],
        bbox_wxyz: Optional[np.ndarray],
    ) -> None:
        """Called when an object is completed (before New Obj or Add All)."""
        with self.state.lock:
            if not self.state.recording_enabled or not self.state.session_active:
                return
            session_dir = self.state.session_dir
            frame_token = self.state.frame_token

        if session_dir is None or frame_token is None:
            return

        # Save object data
        obj_dir = session_dir / f"frame_{frame_token}" / "objects" / f"obj_{obj_idx}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        with self.state.lock:
            if obj_idx >= len(self.state.objects):
                return
            obj = self.state.objects[obj_idx]

            # Save clicks
            clicks_data = [
                {"x": c.x, "y": c.y, "label": c.label, "timestamp": c.timestamp}
                for c in obj.clicks
            ]
            clicks_path = obj_dir / "clicks.json"
            try:
                with open(clicks_path, "w") as f:
                    json.dump(clicks_data, f, indent=2)
            except Exception:
                pass

            # Save mask
            if mask_path is not None:
                mask_dst = obj_dir / "mask.png"
                try:
                    shutil.copy2(mask_path, mask_dst)
                    obj.mask_path = str(mask_dst)
                except Exception:
                    pass

            # Save RGB image
            if rgb_image is not None:
                rgb_dst = obj_dir / "rgb.png"
                try:
                    from PIL import Image
                    Image.fromarray(rgb_image).save(rgb_dst)
                    obj.rgb_path = str(rgb_dst)
                except Exception:
                    pass

            # Save bbox
            bbox_data = {}
            if bbox_center is not None:
                bbox_data["center"] = bbox_center.tolist() if hasattr(bbox_center, "tolist") else list(bbox_center)
                obj.bbox_center = bbox_data["center"]
            if bbox_dimensions is not None:
                bbox_data["dimensions"] = bbox_dimensions.tolist() if hasattr(bbox_dimensions, "tolist") else list(bbox_dimensions)
                obj.bbox_dimensions = bbox_data["dimensions"]
            if bbox_wxyz is not None:
                bbox_data["wxyz"] = bbox_wxyz.tolist() if hasattr(bbox_wxyz, "tolist") else list(bbox_wxyz)
                obj.bbox_wxyz = bbox_data["wxyz"]

            if bbox_data:
                bbox_path = obj_dir / "bbox.json"
                try:
                    with open(bbox_path, "w") as f:
                        json.dump(bbox_data, f, indent=2)
                except Exception:
                    pass

    def on_new_object(self, client) -> None:
        """Called when user clicks New Object button."""
        with self.state.lock:
            if not self.state.recording_enabled or not self.state.session_active:
                return
            # Add new object to track
            self.state.objects.append(RecordedObject())
            self.state.current_object_idx = len(self.state.objects) - 1

    def on_session_complete(self, client, frame_idx: int) -> None:
        """Called when SAM3D session ends (Add All Objects & Exit)."""
        with self.state.lock:
            if not self.state.recording_enabled or not self.state.session_active:
                return
            session_dir = self.state.session_dir
            frame_token = self.state.frame_token

        if session_dir is None or frame_token is None:
            return

        # Render and save post-edit 6-view images
        self._save_post_edit_views(frame_idx)

        # Save all_bboxes.json
        self._save_all_bboxes()

        # Update meta.json with final object count
        self._save_meta()

        # End session
        with self.state.lock:
            self.state.session_active = False

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _save_pre_edit_views(self, frame_idx: int) -> None:
        """Save 6-view images before any edits."""
        print(f"[RecordOps] _save_pre_edit_views called, frame_idx={frame_idx}")
        with self.state.lock:
            session_dir = self.state.session_dir
            frame_token = self.state.frame_token

        if session_dir is None or frame_token is None:
            print(f"[RecordOps] _save_pre_edit_views: session_dir or frame_token is None")
            return

        if self.renderer is None:
            self.renderer = SixViewRenderer(self._app())

        views_dir = session_dir / f"frame_{frame_token}" / "pre_edit_views"
        print(f"[RecordOps] Rendering pre-edit views to: {views_dir}")
        try:
            saved_paths = self.renderer.render_six_views(
                output_dir=views_dir,
                frame_idx=frame_idx,
                distance=1.5,
                image_size=512,
            )
            print(f"[RecordOps] Pre-edit views saved: {list(saved_paths.keys())}")
        except Exception as e:
            print(f"[RecordOps] Failed to render pre-edit views: {e}")
            import traceback
            traceback.print_exc()

    def _save_post_edit_views(self, frame_idx: int) -> None:
        """Save 6-view images after all edits."""
        print(f"[RecordOps] _save_post_edit_views called, frame_idx={frame_idx}")
        with self.state.lock:
            session_dir = self.state.session_dir
            frame_token = self.state.frame_token

        if session_dir is None or frame_token is None:
            print(f"[RecordOps] _save_post_edit_views: session_dir or frame_token is None")
            return

        if self.renderer is None:
            self.renderer = SixViewRenderer(self._app())

        views_dir = session_dir / f"frame_{frame_token}" / "post_edit_views"
        print(f"[RecordOps] Rendering post-edit views to: {views_dir}")
        try:
            saved_paths = self.renderer.render_six_views(
                output_dir=views_dir,
                frame_idx=frame_idx,
                distance=1.5,
                image_size=512,
            )
            print(f"[RecordOps] Post-edit views saved: {list(saved_paths.keys())}")
        except Exception as e:
            print(f"[RecordOps] Failed to render post-edit views: {e}")
            import traceback
            traceback.print_exc()

    def _save_all_bboxes(self) -> None:
        """Save all object bboxes to all_bboxes.json."""
        with self.state.lock:
            session_dir = self.state.session_dir
            frame_token = self.state.frame_token
            objects = list(self.state.objects)

        if session_dir is None or frame_token is None:
            return

        all_bboxes = []
        for idx, obj in enumerate(objects):
            bbox = {
                "object_idx": idx,
                "center": obj.bbox_center,
                "dimensions": obj.bbox_dimensions,
                "wxyz": obj.bbox_wxyz,
            }
            all_bboxes.append(bbox)

        bboxes_path = session_dir / f"frame_{frame_token}" / "all_bboxes.json"
        try:
            bboxes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bboxes_path, "w") as f:
                json.dump(all_bboxes, f, indent=2)
        except Exception as e:
            print(f"[RecordOps] Failed to save all_bboxes.json: {e}")

    def _save_meta(self) -> None:
        """Save session metadata to meta.json."""
        with self.state.lock:
            session_dir = self.state.session_dir
            meta = {
                "session_id": self.state.session_id,
                "frame_token": self.state.frame_token,
                "started_at": self.state.started_at,
                "objects_count": len(self.state.objects),
            }

        if session_dir is None:
            return

        meta_path = session_dir / "meta.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[RecordOps] Failed to save meta.json: {e}")
