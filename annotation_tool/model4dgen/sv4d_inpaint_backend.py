from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as TT
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import InterpolationMode

# Ensure generative-models is importable when this backend is used as a module.
ROOT = Path(__file__).resolve().parent
GM_ROOT = ROOT / "generative-models"
if str(GM_ROOT) not in sys.path:
    sys.path.insert(0, str(GM_ROOT))

from scripts.demo.sv4d_helpers import (  # noqa: E402
    load_model,
    run_img2vid,
    run_img2vid_masked_denoising,
    save_video,
)
from sgm.modules.encoders.modules import (  # noqa: E402
    VideoPredictionEmbedderWithEncoder,
)


SV4D2_CONFIGS = {
    "sv4d2_with_encoder": {
        "T": 12,
        "V": 4,
        "model_config": "scripts/sampling/configs/sv4d2_with_encoder.yaml",
        "version_dict": {
            "T": 12 * 4,
            "options": {
                "discretization": 1,
                "cfg": 2.0,
                "min_cfg": 2.0,
                "num_views": 4,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 2,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    }
}


@dataclass
class SV4DInpaintResult:
    videos: List[str]
    frames: int
    views: int
    output_dir: str
    device: str
    png_root: str
    view_azimuths: List[float]


class SV4DInpaintingBackend:
    """Minimal backend wrapper to run SV4D2 masked denoising on multi-view clips."""

    def __init__(
        self,
        model_path: str | Path = "./checkpoints/sv4d2.0/sv4d2_with_first_stage_encoder_v2.safetensors",
        config_key: str = "sv4d2_with_encoder",
        device: Optional[str] = None,
    ) -> None:
        if config_key not in SV4D2_CONFIGS:
            raise ValueError(f"Unknown SV4D2 config key: {config_key}")

        self.config = copy.deepcopy(SV4D2_CONFIGS[config_key])
        self.model_config_path = GM_ROOT / self.config["model_config"]

        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = GM_ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"SV4D2 checkpoint not found: {model_path}. Place the weights under checkpoints/."
            )
        self.model_path = model_path

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model = None
        self._last_total_frames: Optional[int] = None
        self._last_num_steps: Optional[int] = None

        # Eagerly load the model once at startup with default settings to avoid per-request init.
        self._default_total_frames = self.config["version_dict"].get("T", self.config["T"] * self.config["V"])
        self._default_num_steps = 50
        # try:
        self._get_or_load_model(
            total_frames=self._default_total_frames,
            num_steps=self._default_num_steps,
            decoding_t=4,
            encoding_t=8,
            force=False,
            )
        # except Exception as exc:  # noqa: BLE001
        #     # Allow backend construction even if weights are missing; will raise on first use.

    # -------------------------- public API --------------------------
    def gen3d(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        azimuths_deg: Optional[List[float]] = None,
        img_size: int = 576,
        num_steps: int = 50,
        seed: int = 23,
        encoding_t: int = 8,
        decoding_t: int = 4,
        elevations_deg: float | List[float] = 0.0,
        frame_token: str = "000000",
        cond_mv: bool = False,
        return_img_rgb: bool = False
    ) -> Dict:
        """Generate multi-view images from a single input image.

        Args:
            image_path: Input image path (RGB/RGBA).
            output_dir: Folder to save output PNGs.
            azimuths_deg: Novel-view azimuths in degrees. Default [0,90,180,270].
            img_size: Square output resolution.
            num_steps: Diffusion steps.
            seed: Random seed.
            encoding_t: Frames per chunk for encoder embedder.
            decoding_t: Frames per chunk for decoder.
            elevations_deg: Single value or list of per-view elevations (degrees).
            frame_token: Filename stem for output PNGs.
            cond_mv: Whether to condition on multi-view inputs (usually False for single image).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        png_root = output_dir / "png"
        png_root.mkdir(parents=True, exist_ok=True)

        V_cfg = int(self.config["V"])

        # Interpret azimuths: if length == V, treat as novel views; if length == V+1, first is anchor.
        if azimuths_deg is None:
            novel_azimuths = [0.0, 90.0, 180.0, 270.0]
            anchor_az = 0.0
            requested_azimuths = list(novel_azimuths)
        else:
            azimuths = [float(a) for a in azimuths_deg]
            if len(azimuths) == V_cfg + 1:
                anchor_az = azimuths[0]
                novel_azimuths = azimuths[1:]
                requested_azimuths = list(azimuths)
            elif len(azimuths) == V_cfg:
                anchor_az = 0.0
                novel_azimuths = azimuths
                requested_azimuths = list(azimuths)
            else:
                anchor_az = azimuths[0] if azimuths else 0.0
                novel_azimuths = azimuths[1:] if len(azimuths) > 1 else []
                if not novel_azimuths:
                    step = 360.0 / max(V_cfg, 1)
                    novel_azimuths = [float((anchor_az + step * (i + 1)) % 360.0) for i in range(V_cfg)]
                while len(novel_azimuths) < V_cfg:
                    novel_azimuths.append(novel_azimuths[-1])
                novel_azimuths = novel_azimuths[:V_cfg]
                requested_azimuths = list(novel_azimuths)

        # Load input image (RGBA -> RGB on white background)
        rgba = Image.open(image_path).convert("RGBA")
        white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
        white_bg.paste(rgba, mask=rgba.split()[3])
        rgb_img = white_bg.resize((img_size, img_size), Image.BILINEAR)

        to_tensor = TT.ToTensor()
        image_t = to_tensor(rgb_img).unsqueeze(0).to(self.device)
        image_t = image_t * 2.0 - 1.0

        # Build conditioning for a single-image sequence
        total_frames = self.config["T"] * V_cfg
        T_time = max(1, int(total_frames // V_cfg))
        cond_motion = image_t.repeat(T_time, 1, 1, 1)
        cond_view = torch.zeros((V_cfg, 3, img_size, img_size), device=self.device)

        version_dict = self._build_version_dict(img_size, num_steps, total_frames)
        model = self._prepare_model(decoding_t, encoding_t, total_frames)

        batch_azimuths = [anchor_az] + novel_azimuths
        polars_rad, azimuths_rad = self._build_view_angles(
            [elevations_deg] * (V_cfg + 1) if isinstance(elevations_deg, (int, float)) else elevations_deg,
            batch_azimuths,
            V_cfg + 1,
        )
        polars = polars_rad[1:][None].repeat(T_time, 0).flatten()
        azims = azimuths_rad[1:][None].repeat(T_time, 0).flatten()
        polars = (polars - polars_rad[0] + torch.pi / 2) % (torch.pi * 2)
        azims = (azims - azimuths_rad[0]) % (torch.pi * 2)

        torch.manual_seed(seed)
        samples = run_img2vid(
            version_dict,
            model,
            image_t,
            seed=seed,
            polar_rad=polars,
            azim_rad=azims,
            cond_motion=cond_motion,
            cond_view=cond_view,
            decoding_t=decoding_t,
            cond_mv=cond_mv,
        )

        # samples: [T*V, 3, H, W] -> take first frame views
        samples = samples.view(T_time, V_cfg, 3, img_size, img_size)
        first_frame = samples[0]

        output_paths: Dict[str, str] = {}

        # Save novel views
        for v, az in enumerate(novel_azimuths):
            out_path = png_root / f"{frame_token}_az{int(round(az)) % 360:03d}.png"
            vutils.save_image(first_frame[v].clamp(0, 1), out_path)
            output_paths[f"az{int(round(az)) % 360:03d}"] = str(out_path)

        # Optionally save anchor if explicitly requested (len == V+1 case)
        if azimuths_deg is not None and len(azimuths_deg) == V_cfg + 1:
            anchor_path = png_root / f"{frame_token}_az{int(round(anchor_az)) % 360:03d}.png"
            rgb_img.save(anchor_path)
            output_paths[f"az{int(round(anchor_az)) % 360:03d}"] = str(anchor_path)

        return {
            "status": "success",
            "png_root": str(png_root),
            "output_paths": output_paths,
            "view_azimuths": requested_azimuths,
            "device": str(self.device)
        }

    def gen3d_input_video(
        self,
        input_frames: List[str | Path] | str | Path,
        output_dir: str | Path,
        azimuths_deg: Optional[List[float]] = None,
        img_size: int = 576,
        num_steps: int = 50,
        seed: int = 23,
        encoding_t: int = 8,
        decoding_t: int = 4,
        elevations_deg: float | List[float] = 0.0,
        frame_token: str = "000000",
        cond_mv: bool = False,
    ) -> Dict:
        """Generate multi-view images from a short input video clip.

        Args:
            input_frames: Either a list of frame paths (frame0..frameT) or a directory containing frames.
            output_dir: Folder to save output PNGs.
            azimuths_deg: Novel-view azimuths in degrees. Default [0,90,180,270].
            img_size: Square output resolution.
            num_steps: Diffusion steps.
            seed: Random seed.
            encoding_t: Frames per chunk for encoder embedder.
            decoding_t: Frames per chunk for decoder.
            elevations_deg: Single value or list of per-view elevations (degrees).
            frame_token: Filename stem for output PNGs.
            cond_mv: Whether to condition on multi-view inputs (usually False for input video).
        """
        if isinstance(input_frames, (str, Path)):
            input_path = Path(input_frames)
            if input_path.is_dir():
                frame_list = sorted(
                    [p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
                )
            else:
                frame_list = [input_path]
        else:
            frame_list = [Path(p) for p in input_frames]

        if not frame_list:
            raise ValueError("No input frames provided for gen3d_input_video")
        for p in frame_list:
            if not p.exists():
                raise FileNotFoundError(f"Input frame not found: {p}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        png_root = output_dir / "png"
        png_root.mkdir(parents=True, exist_ok=True)

        V_cfg = int(self.config["V"])

        # Interpret azimuths: if length == V, treat as novel views; if length == V+1, first is anchor.
        if azimuths_deg is None:
            novel_azimuths = [0.0, 90.0, 180.0, 270.0]
            anchor_az = 0.0
            requested_azimuths = list(novel_azimuths)
        else:
            azimuths = [float(a) for a in azimuths_deg]
            if len(azimuths) == V_cfg + 1:
                anchor_az = azimuths[0]
                novel_azimuths = azimuths[1:]
                requested_azimuths = list(azimuths)
            elif len(azimuths) == V_cfg:
                anchor_az = 0.0
                novel_azimuths = azimuths
                requested_azimuths = list(azimuths)
            else:
                anchor_az = azimuths[0] if azimuths else 0.0
                novel_azimuths = azimuths[1:] if len(azimuths) > 1 else []
                if not novel_azimuths:
                    step = 360.0 / max(V_cfg, 1)
                    novel_azimuths = [float((anchor_az + step * (i + 1)) % 360.0) for i in range(V_cfg)]
                while len(novel_azimuths) < V_cfg:
                    novel_azimuths.append(novel_azimuths[-1])
                novel_azimuths = novel_azimuths[:V_cfg]
                requested_azimuths = list(novel_azimuths)

        # Load frames (RGBA -> RGB on white background)
        def _load_frame(path: Path) -> torch.Tensor:
            rgba = Image.open(path).convert("RGBA")
            white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
            white_bg.paste(rgba, mask=rgba.split()[3])
            rgb_img = white_bg.resize((img_size, img_size), Image.BILINEAR)
            image_t = TT.ToTensor()(rgb_img).unsqueeze(0)
            return image_t

        T_cfg = int(self.config["T"])
        frames_t = [_load_frame(p) for p in frame_list]
        if len(frames_t) < T_cfg:
            pad = frames_t[-1]
            frames_t = frames_t + [pad] * (T_cfg - len(frames_t))
        elif len(frames_t) > T_cfg:
            frames_t = frames_t[:T_cfg]
        T_time = T_cfg
        cond_motion = torch.cat(frames_t, dim=0).to(self.device)
        cond_motion = cond_motion * 2.0 - 1.0
        image_t = frames_t[0].to(self.device) * 2.0 - 1.0

        cond_view = torch.zeros((V_cfg, 3, img_size, img_size), device=self.device)
        total_frames = T_cfg * V_cfg
        version_dict = self._build_version_dict(img_size, num_steps, total_frames)
        model = self._prepare_model(decoding_t, encoding_t, total_frames)

        batch_azimuths = [anchor_az] + novel_azimuths
        polars_rad, azimuths_rad = self._build_view_angles(
            [elevations_deg] * (V_cfg + 1) if isinstance(elevations_deg, (int, float)) else elevations_deg,
            batch_azimuths,
            V_cfg + 1,
        )
        polars = polars_rad[1:][None].repeat(T_time, 0).flatten()
        azims = azimuths_rad[1:][None].repeat(T_time, 0).flatten()
        polars = (polars - polars_rad[0] + torch.pi / 2) % (torch.pi * 2)
        azims = (azims - azimuths_rad[0]) % (torch.pi * 2)

        torch.manual_seed(seed)
        samples = run_img2vid(
            version_dict,
            model,
            image_t,
            seed=seed,
            polar_rad=polars,
            azim_rad=azims,
            cond_motion=cond_motion,
            cond_view=cond_view,
            decoding_t=decoding_t,
            cond_mv=cond_mv,
        )

        # samples: [T*V, 3, H, W] -> take first frame views
        samples = samples.view(T_time, V_cfg, 3, img_size, img_size)
        first_frame = samples[0]

        output_paths: Dict[str, str] = {}
        for v, az in enumerate(novel_azimuths):
            out_path = png_root / f"{frame_token}_az{int(round(az)) % 360:03d}.png"
            vutils.save_image(first_frame[v].clamp(0, 1), out_path)
            output_paths[f"az{int(round(az)) % 360:03d}"] = str(out_path)

        if azimuths_deg is not None and len(azimuths_deg) == V_cfg + 1:
            anchor_path = png_root / f"{frame_token}_az{int(round(anchor_az)) % 360:03d}.png"
            Image.open(frame_list[0]).convert("RGB").resize((img_size, img_size), Image.BILINEAR).save(anchor_path)
            output_paths[f"az{int(round(anchor_az)) % 360:03d}"] = str(anchor_path)

        return {
            "status": "success",
            "png_root": str(png_root),
            "output_paths": output_paths,
            "view_azimuths": requested_azimuths,
            "device": str(self.device)
        }


    def inpaint(
        self,
        render_root: str | Path,
        mask_root: str | Path,
        input_root: str | Path,
        output_dir: str | Path,
        num_steps: int = 50,
        img_size: int = 576,
        seed: int = 23,
        encoding_t: int = 8,
        decoding_t: int = 4,
        elevations_deg: float | List[float] = 0.0,
        azimuths_deg: Optional[List[float]] = None,
        cond_mv: bool = True,
        frame_token: str = "000000",
        sigma_start: float = 250.0,
        num_refine_steps: int = 25,
        batch_runs: int = 7,
        batch_views: int = 4,
    ) -> SV4DInpaintResult:
        """
        Args:
            render_root: Folder containing multi-view RGB frames (PNG) named as `{frame_id:06d}_azXXX.png`.
            mask_root: Folder containing corresponding masks with the same naming pattern.
            input_root: Folder of RGBA frames for the source view (view0) used as cond_motion.
            output_dir: Target directory where per-view mp4 videos will be written.
            img_size: Square resolution for conditioning and sampling.
            seed: Random seed.
            encoding_t: Frames per chunk for the encoder embedder.
            decoding_t: Frames per chunk for first-stage decoding.
            elevations_deg: Single value or list of per-view elevations (degrees).
            azimuths_deg: List of per-view azimuths (degrees); defaults to canonical spacing.
            cond_mv: Whether to condition on multi-view inputs (cond_view/cond_motion).
            batch_runs: Number of SV4D runs (each run generates `batch_views` novel views).
            batch_views: Novel views per run (anchor view0 is always included as input).
        """
        render_root = Path(render_root)
        mask_root = Path(mask_root)
        input_root = Path(input_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        V_cfg = self.config["V"]
        n_views_per_batch = V_cfg + 1
        if batch_views <= 0:
            batch_views = V_cfg
        batch_runs = max(1, int(batch_runs))

        if azimuths_deg is None:
            azimuths_deg = [0.0, 60.0, 120.0, 180.0, 240.0]

        anchor_az = float(azimuths_deg[0]) if azimuths_deg else 0.0
        total_novel = batch_runs * batch_views
        if len(azimuths_deg) >= 1 + total_novel:
            novel_azimuths = [float(a) for a in azimuths_deg[1:1 + total_novel]]
        else:
            step = 360.0 / max(total_novel, 1)
            novel_azimuths = [float((anchor_az + step * (i + 1)) % 360.0) for i in range(total_novel)]

        if isinstance(elevations_deg, (list, tuple)):
            elev_value = float(elevations_deg[0]) if elevations_deg else 0.0
        else:
            elev_value = float(elevations_deg)

        png_root = output_dir / "png"
        png_root.mkdir(parents=True, exist_ok=True)
        videos: List[str] = []
        saved_view_azimuths: List[float] = []

        torch.manual_seed(seed)

        global_view_idx = 0
        for run_idx in range(batch_runs):
            batch_start = run_idx * batch_views
            batch_end = batch_start + batch_views
            batch_novel = novel_azimuths[batch_start:batch_end]
            if len(batch_novel) < batch_views:
                batch_novel = batch_novel + [batch_novel[-1]] * (batch_views - len(batch_novel))

            batch_azimuths = [anchor_az] + batch_novel

            (
                render_clip,
                mask_clip,
                cond_motion,
                cond_view,
                T_time,
            ) = self._load_inputs(render_root, mask_root, input_root, img_size, batch_azimuths)

            total_frames = T_time * V_cfg
            version_dict = self._build_version_dict(img_size, num_steps, total_frames)
            model = self._prepare_model(decoding_t, encoding_t, total_frames)

            polars_rad, azimuths_rad = self._build_view_angles(
                [elev_value] * n_views_per_batch,
                batch_azimuths,
                n_views_per_batch,
            )
            polars = polars_rad[1:][None].repeat(T_time, 0).flatten()
            azims = azimuths_rad[1:][None].repeat(T_time, 0).flatten()
            polars = (polars - polars_rad[0] + torch.pi / 2) % (torch.pi * 2)
            azims = (azims - azimuths_rad[0]) % (torch.pi * 2)

            samples = run_img2vid_masked_denoising(
                version_dict,
                model,
                render_clip[:, 1:],
                mask_clip[:, 1:],
                seed=seed,
                polar_rad=polars,
                azim_rad=azims,
                cond_motion=cond_motion,
                cond_view=cond_view,
                decoding_t=decoding_t,
                cond_mv=cond_mv,
                sigma_start=sigma_start,
                num_refine_steps=num_refine_steps,
            )

            for v in range(samples.shape[1]):
                vid_file = output_dir / f"sample_v{global_view_idx + 1:03d}.mp4"
                save_video(
                    str(vid_file),
                    [samples[t, v].unsqueeze(0).cpu() * 2.0 - 1.0 for t in range(samples.shape[0])],
                )
                videos.append(str(vid_file))

                view_dir = png_root / f"eval_{global_view_idx}"
                view_dir.mkdir(parents=True, exist_ok=True)
                first_frame = samples[-1, v].unsqueeze(0).cpu()  # The final frame is the target.
                vutils.save_image(first_frame, view_dir / f"{frame_token}.png")

                saved_view_azimuths.append(batch_novel[v])
                global_view_idx += 1

        return SV4DInpaintResult(
            videos=videos,
            frames=T_time,
            views=global_view_idx,
            output_dir=str(output_dir),
            device=str(self.device),
            png_root=str(png_root),
            view_azimuths=saved_view_azimuths,
        )

    # -------------------------- helpers --------------------------
    def _build_version_dict(
        self, img_size: int, num_steps: int, total_frames: int
    ) -> Dict:
        version = copy.deepcopy(self.config["version_dict"])
        version["H"] = img_size
        version["W"] = img_size
        version["C"] = 4
        version["f"] = 8
        version["T"] = total_frames
        version["options"] = copy.deepcopy(version["options"])
        version["options"]["num_steps"] = num_steps
        return version

    def _prepare_model(self, decoding_t: int, encoding_t: int, total_frames: int):
        # Reuse the eagerly loaded model; warn if total_frames differs from the loaded config.
        if self._model is None:
            self._get_or_load_model(
                total_frames=self._default_total_frames,
                num_steps=self._default_num_steps,
                decoding_t=decoding_t,
                encoding_t=encoding_t,
                force=True,
            )
        elif total_frames != self._last_total_frames:
            print(
                f"[SV4DInpaintingBackend] Using preloaded model with T={self._last_total_frames}"
                f" while request needs T={total_frames} (no reload)."
            )

        self._model.en_and_decode_n_samples_a_time = decoding_t
        for emb in self._model.conditioner.embedders:
            if isinstance(emb, VideoPredictionEmbedderWithEncoder):
                emb.en_and_decode_n_samples_a_time = encoding_t
        return self._model

    def _get_or_load_model(
        self,
        total_frames: int,
        num_steps: int,
        decoding_t: int,
        encoding_t: int,
        force: bool = False,
    ):
        need_reload = force or self._model is None
        if need_reload:
            self._model, _ = load_model(
                str(self.model_config_path),
                str(self.device),
                total_frames,
                num_steps,
                False,
                str(self.model_path),
            )
            self._last_total_frames = total_frames
            self._last_num_steps = num_steps

        self._model.en_and_decode_n_samples_a_time = decoding_t
        for emb in self._model.conditioner.embedders:
            if isinstance(emb, VideoPredictionEmbedderWithEncoder):
                emb.en_and_decode_n_samples_a_time = encoding_t
        return self._model

    def _load_inputs(
        self,
        render_root: Path,
        mask_root: Path,
        input_root: Path,
        img_size: int,
        azimuths_deg: List[float],
    ):
        n_views = len(azimuths_deg)
        angle_to_view: Dict[str, List[int]] = {}
        for idx, az in enumerate(azimuths_deg):
            token = f"az{int(round(float(az))) % 360:03d}"
            angle_to_view.setdefault(token, []).append(idx)

        def _load_flat_frames(root: Path) -> List[Path]:
            files = sorted(root.glob("*.png"))
            if not files:
                raise ValueError(f"No frames found in {root}")
            return files

        img_files_all = _load_flat_frames(render_root)
        mask_files_all = _load_flat_frames(mask_root)

        def _organize(files: List[Path]) -> Dict[int, Dict[int, Path]]:
            buckets: Dict[int, Dict[int, Path]] = {}
            for f in files:
                name = f.stem
                parts = name.split("_")
                if len(parts) < 2:
                    continue
                frame_id, az = parts[0], parts[1]
                if az not in angle_to_view:
                    continue
                fidx = int(frame_id)
                for vidx in angle_to_view[az]:
                    buckets.setdefault(fidx, {})[vidx] = f
            return buckets

        img_buckets = _organize(img_files_all)
        mask_buckets = _organize(mask_files_all)
        frame_ids = sorted(img_buckets.keys())
        if not frame_ids:
            raise ValueError("Could not parse frames from file names")

        T_cfg = self.config["T"]
        T_time = min(T_cfg, len(frame_ids))
        frame_ids = frame_ids[:T_time]
        if len(frame_ids) < T_cfg:
            print(
                f"Warning: only {len(frame_ids)} frames found, expected {T_cfg} (trimming to available)"
            )

        to_tensor = TT.ToTensor()
        resize = TT.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)
        resize_mask = TT.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)

        all_views = []
        all_masks = []
        for v in range(n_views):
            imgs_v = []
            masks_v = []
            for fidx in frame_ids:
                if v not in img_buckets.get(fidx, {}):
                    raise ValueError(f"Missing view {v} frame {fidx:06d} in {render_root}")
                if v not in mask_buckets.get(fidx, {}):
                    raise ValueError(f"Missing view {v} frame {fidx:06d} in {mask_root}")
                img = Image.open(img_buckets[fidx][v]).convert("RGB")
                msk = Image.open(mask_buckets[fidx][v]).convert("L")
                img_t = to_tensor(resize(img)).unsqueeze(0).to(self.device)
                img_t = img_t * 2.0 - 1.0
                msk_t = to_tensor(resize_mask(msk)).unsqueeze(0).to(self.device)
                imgs_v.append(img_t)
                masks_v.append(msk_t)
            all_views.append(torch.cat(imgs_v, dim=0))
            all_masks.append(torch.cat(masks_v, dim=0))

        render_clip = torch.stack(all_views, dim=1)
        mask_clip = torch.stack(all_masks, dim=1)

        input_files = sorted(input_root.glob("*.png"))
        if len(input_files) < T_time:
            raise ValueError(f"Need at least {T_time} input frames in {input_root}, found {len(input_files)}")

        cond_list = []
        for f in input_files[:T_time]:
            rgba = Image.open(f).convert("RGBA").resize((img_size, img_size), Image.BILINEAR)
            arr = np.array(rgba).astype(np.float32) / 255.0
            rgb = arr[..., :3]
            alpha = arr[..., 3:4]
            comp = rgb * alpha + (1.0 - alpha)
            comp_img = torch.from_numpy(comp).permute(2, 0, 1)
            cond_list.append(comp_img.unsqueeze(0))
        cond_motion = torch.cat(cond_list, dim=0).to(self.device) * 2.0 - 1.0

        cond_view = render_clip[0, 1:]
        return render_clip, mask_clip, cond_motion, cond_view, T_time

    def _build_view_angles(
        self,
        elevations_deg: float | List[float],
        azimuths_deg: Optional[List[float]],
        n_views: int,
    ):
        if isinstance(elevations_deg, (float, int)):
            elevations_deg = [float(elevations_deg)] * n_views
        if len(elevations_deg) != n_views:
            raise ValueError(
                f"Provide 1 value or {n_views} values for elevations_deg; got {len(elevations_deg)}"
            )

        if azimuths_deg is None:
            if n_views == 5:
                azimuths_deg = [0, 60, 120, 180, 240]
            else:
                azimuths_deg = list(np.linspace(0, 360, n_views + 1)[1:] % 360)
        if len(azimuths_deg) != n_views:
            raise ValueError(
                f"Provide {n_views} values for azimuths_deg; got {len(azimuths_deg)}"
            )

        polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg], dtype=np.float32)
        azimuths_rad = np.array(
            [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg], dtype=np.float32
        )
        return polars_rad, azimuths_rad


__all__ = ["SV4DInpaintingBackend", "SV4DInpaintResult"]
