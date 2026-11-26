#!/usr/bin/env python3
"""Render a Gaussian-splat PLY sequence to PNG frames and an MP4 video."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gs-dir", type=Path, required=True, help="Directory containing frame PLY files.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--frames-dir", type=Path, help="PNG output directory (default: <output stem>_frames).")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--fov", type=float, default=49.1, help="Vertical field of view in degrees.")
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=10.0)
    parser.add_argument("--distance", type=float, help="Camera distance; automatically estimated when omitted.")
    parser.add_argument("--up-axis", choices=("y", "-y", "z"), default="z")
    parser.add_argument("--orbit-degrees", type=float, default=0.0, help="Camera orbit over the full sequence.")
    parser.add_argument("--start", type=int, default=0, help="First sorted PLY index to render.")
    parser.add_argument("--end", type=int, help="Exclusive sorted PLY index.")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_runtime(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output.parent / ".render_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(temp_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMP", str(temp_dir))

    cuda_home = Path(os.environ.get("CUDA_HOME", "/root/miniforge/envs/hat4d-roi"))
    if (cuda_home / "bin" / "nvcc").exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"
        cc = cuda_home / "bin" / "x86_64-conda-linux-gnu-gcc"
        cxx = cuda_home / "bin" / "x86_64-conda-linux-gnu-g++"
        if cc.exists() and cxx.exists():
            os.environ.setdefault("CC", str(cc))
            os.environ.setdefault("CXX", str(cxx))
        target_include = cuda_home / "targets" / "x86_64-linux" / "include"
        target_lib = cuda_home / "targets" / "x86_64-linux" / "lib"
        os.environ["CPLUS_INCLUDE_PATH"] = f"{target_include}:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
        os.environ["LIBRARY_PATH"] = f"{target_lib}:{os.environ.get('LIBRARY_PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{target_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    extensions_dir = Path(
        os.environ.get("TORCH_EXTENSIONS_DIR", "/data/hat4d/cache/torch-extensions/hat4d-frontend")
    )
    extensions_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extensions_dir))


def robust_bounds(centers: np.ndarray) -> tuple[np.ndarray, float]:
    low, high = np.percentile(centers, [1.0, 99.0], axis=0)
    center = ((low + high) * 0.5).astype(np.float32)
    radius = float(np.linalg.norm(high - low) * 0.5)
    return center, max(radius, 1e-3)


def camera_pose(azimuth: float, elevation: float, distance: float, center: np.ndarray, up_axis: str):
    from utils.camera_utils import eye_wxyz_from_angles
    import viser.transforms as tf

    eye, wxyz = eye_wxyz_from_angles(azimuth, elevation, distance, center, up_axis)
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = tf.SO3(wxyz).as_matrix().astype(np.float32)
    camera_to_world[:3, 3] = eye
    return np.linalg.inv(camera_to_world).astype(np.float32)


def render_frame(path: Path, viewmat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import torch
    from gsplat.rendering import rasterization
    from utils.ply_utils import load_splat_from_ply

    data = load_splat_from_ply(path)
    device = torch.device(args.device)
    fy = args.height / (2.0 * np.tan(0.5 * np.deg2rad(args.fov)))
    fx = fy
    K = torch.tensor(
        [[fx, 0.0, args.width / 2.0], [0.0, fy, args.height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    bg_value = 1.0 if args.background == "white" else 0.0
    background = torch.full((1, 3), bg_value, dtype=torch.float32, device=device)

    with torch.inference_mode():
        images, _, _ = rasterization(
            means=torch.from_numpy(data.centers).to(device),
            quats=torch.from_numpy(data.rotations).to(device),
            scales=torch.from_numpy(data.scales).to(device),
            opacities=torch.from_numpy(data.opacities.reshape(-1)).to(device),
            colors=torch.from_numpy(data.rgbs).to(device),
            viewmats=torch.from_numpy(viewmat[None]).to(device),
            Ks=K[None],
            width=args.width,
            height=args.height,
            near_plane=0.01,
            far_plane=150.0,
            packed=False,
            backgrounds=background,
            render_mode="RGB",
        )
    image = images[0].clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    return image


def write_video(frame_paths: list[Path], output: Path, fps: float) -> None:
    import imageio.v2 as imageio

    try:
        with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        return
    except Exception as imageio_exc:
        output.unlink(missing_ok=True)

    try:
        import cv2

        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Could not read rendered frame: {frame_paths[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter could not open the MP4 output")
        try:
            writer.write(first)
            for frame_path in frame_paths[1:]:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Could not read rendered frame: {frame_path}")
                writer.write(frame)
        finally:
            writer.release()
    except Exception as cv2_exc:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "PNG frames were rendered, but MP4 encoding failed with both imageio "
            f"({imageio_exc}) and OpenCV ({cv2_exc})"
        ) from cv2_exc


def main() -> int:
    args = parse_args()
    prepare_runtime(args.output)

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")
    if args.width <= 0 or args.height <= 0 or args.step <= 0 or args.fps <= 0:
        raise ValueError("width, height, step, and fps must be positive")

    all_paths = sorted(args.gs_dir.glob("*.ply"))
    selected = all_paths[args.start : args.end : args.step]
    if not selected:
        raise FileNotFoundError(f"No PLY frames selected from {args.gs_dir}")

    from PIL import Image
    from utils.ply_utils import load_splat_from_ply

    first = load_splat_from_ply(selected[0])
    center, radius = robust_bounds(first.centers)
    distance = args.distance if args.distance is not None else radius / np.sin(np.deg2rad(args.fov * 0.45))
    frames_dir = args.frames_dir or args.output.with_name(f"{args.output.stem}_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    total = len(selected)
    print(f"Rendering {total} frames; center={center.tolist()}, radius={radius:.4f}, distance={distance:.4f}")
    for i, ply_path in enumerate(selected):
        phase = 0.0 if total <= 1 else i / (total - 1)
        azimuth = args.azimuth + args.orbit_degrees * phase
        viewmat = camera_pose(azimuth, args.elevation, distance, center, args.up_axis)
        image = render_frame(ply_path, viewmat, args)
        frame_path = frames_dir / f"{i:06d}.png"
        Image.fromarray(image, mode="RGB").save(frame_path)
        frame_paths.append(frame_path)
        print(f"[{i + 1:04d}/{total:04d}] {ply_path.name} -> {frame_path.name}", flush=True)

    write_video(frame_paths, args.output, args.fps)
    print(f"Frames: {frames_dir}")
    print(f"Video:  {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
