from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from .constants import COORD_ALIGN_MAT, SH_C0
from .geometry_utils import sanitize_points, transform_gaussians

# Mapping from PLY property types to numpy dtype strings.
PLY_DTYPE = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "uchar": "u1",
    "uint8": "u1",
}


def read_ply_vertices(
    path: Path, return_header: bool = False
) -> Union[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray, List[bytes]]]:
    """Read vertex properties and optionally return raw header lines."""
    header_lines: List[bytes] = []
    with path.open("rb") as f:
        for line in f:
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header_len = sum(len(line) for line in header_lines)
    header = [line.decode("ascii").strip() for line in header_lines]

    vertex_props: List[Tuple[str, str]] = []
    vertex_count = None
    ply_format = None
    in_vertex = False
    for line in header:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "format" and len(tokens) >= 2:
            ply_format = tokens[1]
        elif tokens[0] == "element":
            in_vertex = tokens[1] == "vertex"
            if in_vertex:
                vertex_count = int(tokens[2])
        elif tokens[0] == "property" and in_vertex:
            dtype = PLY_DTYPE.get(tokens[1])
            if dtype is None:
                raise ValueError(f"Unsupported PLY property type: {tokens[1]}")
            vertex_props.append((tokens[-1], dtype))
        elif tokens[0] == "end_header":
            break

    if vertex_count is None or not vertex_props:
        raise ValueError(f"Invalid or unsupported PLY header in {path}")

    dtype = np.dtype(vertex_props)
    with path.open("rb") as f:
        f.seek(header_len)
        if ply_format == "ascii":
            rows = []
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                values = line.split()
                if values:
                    rows.append(values)
            if len(rows) != vertex_count:
                raise ValueError(
                    f"Expected {vertex_count} ASCII PLY vertices in {path}, found {len(rows)}"
                )
            data = np.empty(vertex_count, dtype=dtype)
            for index, (name, prop_dtype) in enumerate(vertex_props):
                try:
                    data[name] = np.asarray(
                        [row[index] for row in rows],
                        dtype=np.dtype(prop_dtype),
                    )
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid ASCII PLY vertex data in {path}") from exc
        elif ply_format == "binary_little_endian":
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
        else:
            raise ValueError(f"Unsupported PLY format {ply_format!r} in {path}")

    if return_header:
        return [p[0] for p in vertex_props], data, header_lines
    return [p[0] for p in vertex_props], data


def write_ply_vertices(header_lines: List[bytes], vertices: np.ndarray, path: Path) -> None:
    """Atomically write PLY vertices using a preserved header."""
    header_out = list(header_lines)
    vertex_count = int(len(vertices))
    ply_format = None
    for idx, line in enumerate(header_out):
        stripped = line.strip()
        if stripped.startswith(b"format "):
            parts = stripped.split()
            if len(parts) >= 2:
                ply_format = parts[1].decode("ascii")
        if stripped.startswith(b"element") and b"vertex" in stripped:
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == b"element" and parts[1] == b"vertex":
                newline = b"\r\n" if line.endswith(b"\r\n") else b"\n"
                header_out[idx] = b"element vertex " + str(vertex_count).encode("ascii") + newline
            break
    if ply_format not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format {ply_format!r} for {path}")
    if vertices.dtype.names is None:
        raise ValueError("PLY vertices must use a structured NumPy dtype")

    def _ascii_value(value: np.generic) -> str:
        if value.dtype.kind == "f":
            precision = 9 if value.dtype.itemsize <= 4 else 17
            return format(float(value), f".{precision}g")
        return str(int(value))

    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            for line in header_out:
                f.write(line)
            if ply_format == "ascii":
                for vertex in vertices:
                    values = [_ascii_value(vertex[name]) for name in vertices.dtype.names]
                    f.write((" ".join(values) + "\n").encode("ascii"))
            else:
                vertices.tofile(f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@dataclass
class SplatData:
    centers: np.ndarray
    rgbs: np.ndarray
    opacities: np.ndarray
    covariances: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    sh: np.ndarray
    opacity_raw: np.ndarray
    centers_raw: np.ndarray


def load_splat_from_ply(path: Path) -> SplatData:
    """Load Gaussian splat data from PLY and apply front-end transforms.

    This mirrors the logic used by the viewer: compute RGB from SH,
    apply `exp` to scales and `sigmoid` to opacity, then run
    `transform_gaussians` with `COORD_ALIGN_MAT` to get transformed
    centers, rotations, scales and covariance.
    """
    _, vertices = read_ply_vertices(path)

    centers_raw = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)

    # SH DC -> RGB (front-end uses SH_C0)
    sh = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
    rgbs = np.clip(0.5 + SH_C0 * sh, 0.0, 1.0).astype(np.float32)

    # opacity: apply sigmoid and make as (N,1)
    opacity_raw = vertices["opacity"].astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacity_raw)))[:, None].astype(np.float32)

    # scales: exp of stored log-scales
    scales_raw = np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1).astype(np.float32)
    scales = np.exp(scales_raw).astype(np.float32)

    # rotations quaternion (wxyz)
    rotations_raw = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1).astype(np.float32)

    # Transform gaussians (centers, rotations, scales) via covariance method
    centers, rotations, scales, Sigma = transform_gaussians(
        centers=centers_raw, rotations=rotations_raw, scales=scales, A=COORD_ALIGN_MAT, return_cov=True
    )


    return SplatData(
        centers=centers,
        rgbs=rgbs,
        opacities=opacities,
        covariances=Sigma,
        scales=scales,
        rotations=rotations,
        sh=sh,
        opacity_raw=opacity_raw,
        centers_raw=centers_raw,
    )


def splatdata_to_backend_dict(splat: SplatData) -> dict:
    """Convert SplatData to backend dict expected by roi_backend.

    Returns keys: means, scales, rotations, opacity, sh_dc
    - `means`: (N,3) transformed centers
    - `scales`: (N,3) transformed scales (NOT log)
    - `rotations`: (N,4) quaternion wxyz
    - `opacity`: (N,) sigmoided values flattened
    - `sh_dc`: (N,3) raw SH coefficients
    """
    return {
        "means": splat.centers.astype(np.float32),
        "scales": splat.scales.astype(np.float32),
        "rotations": splat.rotations.astype(np.float32),
        "opacity": splat.opacities.reshape(-1).astype(np.float32),
        "sh_dc": splat.sh.astype(np.float32),
    }


def load_gaussian_tensors_raw(
    path: Path, return_header: bool = False
) -> Union[Dict[str, np.ndarray], Tuple[List[str], np.ndarray, List[bytes], Dict[str, np.ndarray]]]:
    """Load raw Gaussian tensors aligned to viewer coordinates.

    - centers are aligned by COORD_ALIGN_MAT
    - rotations are aligned by COORD_ALIGN_MAT
    - scales remain raw (log-space)
    - opacity remains raw (logit)
    """
    if return_header:
        prop_names, vertices, header_lines = read_ply_vertices(path, return_header=True)
    else:
        prop_names, vertices = read_ply_vertices(path)
        header_lines = []

    centers = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    centers = sanitize_points(centers) @ COORD_ALIGN_MAT.T

    scales = np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1).astype(np.float32)
    rotations_raw = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1).astype(np.float32)

    import viser.transforms as tf  # local import to avoid hard dependency at import time

    Rs = tf.SO3(rotations_raw).as_matrix()
    Rs = (COORD_ALIGN_MAT[None, :, :] @ Rs).astype(np.float32) @ COORD_ALIGN_MAT.T
    rotations = tf.SO3.from_matrix(Rs).wxyz.astype(np.float32)

    opacity = vertices["opacity"].astype(np.float32)
    sh_dc = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)

    data = {
        "means": centers,
        "scales": scales,
        "rotations": rotations,
        "opacity": opacity,
        "sh_dc": sh_dc,
    }

    if return_header:
        return prop_names, vertices, header_lines, data
    return data


# Backward-compat alias (typoed name).
load_gaussian_tensors_rawl = load_gaussian_tensors_raw


def inverse_transform_gaussians(
    centers: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    A: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of `transform_gaussians` using A^{-1}."""
    A_inv = np.linalg.inv(A.astype(np.float32))
    out = transform_gaussians(centers=centers, rotations=rotations, scales=scales, A=A_inv, eps=eps, return_cov=False)
    return out[0], out[1], out[2]


def load_gaussian_tensors_transformed(
    path: Path, return_header: bool = False, eps: float = 1e-8
) -> Union[Dict[str, np.ndarray], Tuple[List[str], np.ndarray, List[bytes], Dict[str, np.ndarray]]]:
    """Load Gaussian tensors in aligned space using `transform_gaussians`.

    - centers/rotations/scales are transformed by COORD_ALIGN_MAT
    - scales are returned in log-space (so later exp() is correct)
    - opacity remains raw (logit)
    """
    if return_header:
        prop_names, vertices, header_lines = read_ply_vertices(path, return_header=True)
    else:
        prop_names, vertices = read_ply_vertices(path)
        header_lines = []

    centers_raw = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    scales_raw = np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1).astype(np.float32)
    rotations_raw = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1).astype(np.float32)
    opacity = vertices["opacity"].astype(np.float32)
    sh_dc = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)

    scales = np.exp(scales_raw).astype(np.float32)
    centers, rotations, scales_t = transform_gaussians(
        centers=centers_raw,
        rotations=rotations_raw,
        scales=scales,
        A=COORD_ALIGN_MAT,
        eps=eps,
        return_cov=False,
    )
    scales_log = np.log(np.clip(scales_t, eps, None)).astype(np.float32)

    data = {
        "means": centers.astype(np.float32),
        "scales": scales_log,
        "rotations": rotations.astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "sh_dc": sh_dc.astype(np.float32),
    }

    if return_header:
        return prop_names, vertices, header_lines, data
    return data


def write_ply_subset_aligned(
    header_lines: List[bytes],
    vertices: np.ndarray,
    indices_arr: np.ndarray,
    means_sel_cur: np.ndarray,
    scales_sel_cur: np.ndarray,
    rotations_sel_cur: np.ndarray,
    opacity_sel_cur: np.ndarray,
    sh_dc_sel_cur: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacity: np.ndarray,
    sh_dc: np.ndarray,
    out_path: Path,
    eps: float = 1e-8,
) -> None:
    """Write a PLY subset from aligned tensors, applying inverse transform.

    Inputs are expected in aligned space where scales are log-space.
    """
    means_np = np.asarray(means, dtype=np.float32).copy()
    scales_np = np.asarray(scales, dtype=np.float32).copy()
    rotations_np = np.asarray(rotations, dtype=np.float32).copy()
    opacity_np = np.asarray(opacity, dtype=np.float32).copy()
    sh_dc_np = np.asarray(sh_dc, dtype=np.float32).copy()

    means_np[indices_arr] = np.asarray(means_sel_cur, dtype=np.float32)
    scales_np[indices_arr] = np.asarray(scales_sel_cur, dtype=np.float32)
    rotations_np[indices_arr] = np.asarray(rotations_sel_cur, dtype=np.float32)
    opacity_np[indices_arr] = np.asarray(opacity_sel_cur, dtype=np.float32)
    sh_dc_np[indices_arr] = np.asarray(sh_dc_sel_cur, dtype=np.float32)

    scales_lin = np.exp(scales_np).astype(np.float32)
    centers_src, rotations_src, scales_src = inverse_transform_gaussians(
        centers=means_np,
        rotations=rotations_np,
        scales=scales_lin,
        A=COORD_ALIGN_MAT,
        eps=eps,
    )
    scales_src_log = np.log(np.clip(scales_src, eps, None)).astype(np.float32)

    vertices_copy = vertices.copy()
    vertices_copy["x"] = centers_src[:, 0]
    vertices_copy["y"] = centers_src[:, 1]
    vertices_copy["z"] = centers_src[:, 2]
    vertices_copy["rot_0"] = rotations_src[:, 0]
    vertices_copy["rot_1"] = rotations_src[:, 1]
    vertices_copy["rot_2"] = rotations_src[:, 2]
    vertices_copy["rot_3"] = rotations_src[:, 3]
    vertices_copy["scale_0"] = scales_src_log[:, 0]
    vertices_copy["scale_1"] = scales_src_log[:, 1]
    vertices_copy["scale_2"] = scales_src_log[:, 2]
    vertices_copy["opacity"] = opacity_np
    vertices_copy["f_dc_0"] = sh_dc_np[:, 0]
    vertices_copy["f_dc_1"] = sh_dc_np[:, 1]
    vertices_copy["f_dc_2"] = sh_dc_np[:, 2]
    write_ply_vertices(header_lines, vertices_copy, out_path)
