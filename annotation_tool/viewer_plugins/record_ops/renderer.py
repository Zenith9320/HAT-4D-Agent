"""6-view renderer for record_ops plugin.

Renders 6 canonical views (front, back, left, right, up, down) of the current
Gaussian splat scene using gsplat rasterization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from utils.camera_utils import eye_wxyz_from_angles


# 6 canonical views: (azimuth_deg, elevation_deg)
SIX_VIEW_ANGLES: Dict[str, Tuple[float, float]] = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (90.0, 0.0),
    "right": (270.0, 0.0),
    "up": (0.0, 90.0),
    "down": (0.0, -90.0),
}


class SixViewRenderer:
    """Renders 6 canonical views of the current GS scene."""

    def __init__(self, app):
        self.app = app

    def render_six_views(
        self,
        output_dir: Path,
        frame_idx: int,
        distance: float = 1.5,
        image_size: int = 512,
    ) -> Dict[str, Path]:
        """Render 6 canonical views and save to output_dir.

        Args:
            output_dir: Directory to save the 6 view images.
            frame_idx: Frame index to render.
            distance: Camera distance from center.
            image_size: Output image size (square).

        Returns:
            Dict mapping view name to saved image path.
        """
        print(f"[SixViewRenderer] render_six_views: output_dir={output_dir}, frame_idx={frame_idx}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for view_name, (az_deg, el_deg) in SIX_VIEW_ANGLES.items():
            print(f"[SixViewRenderer] Rendering view: {view_name} (az={az_deg}, el={el_deg})")
            img = self._render_view(
                frame_idx=frame_idx,
                az_deg=az_deg,
                el_deg=el_deg,
                distance=distance,
                image_size=image_size,
            )
            if img is not None:
                out_path = output_dir / f"{view_name}.png"
                self._save_image(img, out_path)
                saved_paths[view_name] = out_path
                print(f"[SixViewRenderer] Saved: {out_path}")
            else:
                print(f"[SixViewRenderer] Failed to render view: {view_name}")

        return saved_paths

    def _render_view(
        self,
        frame_idx: int,
        az_deg: float,
        el_deg: float,
        distance: float,
        image_size: int,
    ) -> Optional[np.ndarray]:
        """Render a single view of the GS scene.

        Returns HxWx3 uint8 RGB image or None on failure.
        """
        # Use sequence.load() to ensure data is loaded (not just cache.get())
        try:
            data = self.app.sequence.load(int(frame_idx))
        except Exception as e:
            print(f"[SixViewRenderer] Failed to load sequence data: {e}")
            data = None
        if data is None:
            print(f"[SixViewRenderer] No data for frame_idx={frame_idx}")
            return None

        try:
            import torch
            from gsplat.rendering import rasterization
            import viser.transforms as tf
        except Exception as e:
            print(f"[SixViewRenderer] Failed to import dependencies: {e}")
            return None

        # Camera pose from angles
        center = np.asarray(self.app.camera_target, dtype=np.float32)
        up_axis = getattr(self.app, "up_axis", "y")
        eye, wxyz = eye_wxyz_from_angles(az_deg, el_deg, distance, center, up_axis)

        # Build camera matrices
        R_cw = tf.SO3(wxyz).as_matrix().astype(np.float32)
        t_cw = eye.astype(np.float32)

        T_cw = np.eye(4, dtype=np.float32)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw

        T_wc = np.linalg.inv(T_cw).astype(np.float32)
        viewmat = torch.from_numpy(T_wc[None]).float()

        # Intrinsics
        fov_deg = getattr(self.app, "fov_deg", 49.1)
        tan_half_fov = np.tan(0.5 * np.deg2rad(float(fov_deg)))
        f = image_size / (2.0 * tan_half_fov)
        K = torch.tensor(
            [[f, 0.0, image_size / 2.0],
             [0.0, f, image_size / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        # GS attributes
        centers = data.centers.astype(np.float32, copy=False)
        rotations = data.rotations.astype(np.float32, copy=False)
        scales = data.scales.astype(np.float32, copy=False)
        opacities = data.opacities.reshape(-1).astype(np.float32, copy=False)
        rgbs = data.rgbs.astype(np.float32, copy=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        means = torch.from_numpy(centers).to(device)
        quats = torch.from_numpy(rotations).to(device)
        scales_t = torch.from_numpy(scales).to(device)
        opacities_t = torch.from_numpy(opacities).to(device)
        colors_t = torch.from_numpy(rgbs).to(device)

        viewmats = viewmat.to(device)
        Ks = torch.stack([K]).to(device)
        bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

        try:
            rendered_image_all, rendered_alpha_all, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales_t,
                opacities=opacities_t,
                colors=colors_t,
                viewmats=viewmats,
                Ks=Ks,
                width=image_size,
                height=image_size,
                near_plane=0.01,
                far_plane=100.0,
                packed=False,
                backgrounds=torch.stack([bg]),
                render_mode="RGB",
            )

            if rendered_image_all is None or len(rendered_image_all) == 0:
                print(f"[SixViewRenderer] Rasterization returned empty result")
                return None

            img = rendered_image_all[0]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()

            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
            return img
        except Exception as e:
            print(f"[SixViewRenderer] Rasterization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_image(self, img: np.ndarray, path: Path) -> None:
        """Save image to file."""
        try:
            from PIL import Image
            Image.fromarray(img).save(path)
        except Exception:
            pass
