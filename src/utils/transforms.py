"""Coordinate transforms and BEV projection geometry (numpy only).

These functions implement the *geometric* part of the perception layer:
LiDAR point-cloud voxelization / pillarization and world↔BEV mapping. They are
deliberately numpy-only so they can be unit-tested without any ML framework,
and later re-used / mirrored by the C++ runtime.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "rotation_matrix_2d",
    "world_to_bev_index",
    "bev_index_to_world",
    "point_cloud_to_bev_tensor",
]


def rotation_matrix_2d(yaw: float) -> np.ndarray:
    """2×2 rotation matrix for a heading ``yaw`` (rad), float32."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _grid_dims(x_range, y_range, resolution) -> Tuple[float, float, int, int]:
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    W = max(1, int(round((x1 - x0) / resolution)))
    H = max(1, int(round((y1 - y0) / resolution)))
    return x0, y0, H, W


def world_to_bev_index(points, x_range, y_range, resolution):
    """Map (N,2) world points → integer BEV ``(row, col)`` + validity mask.

    Returns ``(row, col, valid, (H, W))``.
    """
    pts = np.asarray(points, dtype=np.float32)[..., :2]
    x0, y0, H, W = _grid_dims(x_range, y_range, resolution)
    col = ((pts[..., 0] - x0) / resolution).astype(np.int64)
    row = ((pts[..., 1] - y0) / resolution).astype(np.int64)
    valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    return row, col, valid, (H, W)


def bev_index_to_world(row, col, x_range, y_range, resolution):
    """Inverse of :func:`world_to_bev_index` for a single cell (returns x,y)."""
    x0, y0, _, _ = _grid_dims(x_range, y_range, resolution)
    return np.array([col * resolution + x0, row * resolution + y0], dtype=np.float32)


def point_cloud_to_bev_tensor(
    points,
    x_range=(-50.0, 50.0),
    y_range=(-50.0, 50.0),
    z_range=(-3.0, 3.0),
    resolution=0.5,
    intensity_range=(0.0, 1.0),
):
    """Voxelized BEV feature map from a raw point cloud.

    Returns a ``(4, H, W)`` float32, C-contiguous tensor with channels::

        [0] normalized max height
        [1] normalized mean height
        [2] point density (clipped to 1.0)
        [3] mean intensity

    This is the geometric primitive verified by the perception unit test
    (dimension + memory-contiguity checks).
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points must be (N, >=3), got {pts.shape}")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    intensity = pts[:, 3] if pts.shape[1] >= 4 else np.zeros_like(z)
    i0, i1 = intensity_range
    intensity = np.clip((intensity - i0) / max(i1 - i0, 1e-6), 0.0, 1.0)

    x0, y0, H, W = _grid_dims(x_range, y_range, resolution)
    col = np.floor((x - x0) / resolution).astype(np.int64)
    row = np.floor((y - y0) / resolution).astype(np.int64)
    valid = (
        (col >= 0) & (col < W) & (row >= 0) & (row < H)
        & (z >= z_range[0]) & (z <= z_range[1])
    )
    col, row = col[valid], row[valid]
    z, intensity = z[valid], intensity[valid]

    flat = row * W + col
    n_cells = H * W

    max_h = np.full(n_cells, z_range[0], dtype=np.float32)
    np.maximum.at(max_h, flat, z)

    sum_h = np.zeros(n_cells, dtype=np.float32)
    np.add.at(sum_h, flat, z)
    cnt = np.zeros(n_cells, dtype=np.float32)
    np.add.at(cnt, flat, 1.0)
    mean_h = np.where(cnt > 0, sum_h / np.maximum(cnt, 1.0), z_range[0])

    density = np.clip(cnt / 64.0, 0.0, 1.0)

    sum_i = np.zeros(n_cells, dtype=np.float32)
    np.add.at(sum_i, flat, intensity)
    mean_i = np.where(cnt > 0, sum_i / np.maximum(cnt, 1.0), 0.0)

    z_span = max(z_range[1] - z_range[0], 1e-6)
    bev = np.stack(
        [
            (max_h - z_range[0]) / z_span,
            (mean_h - z_range[0]) / z_span,
            density,
            mean_i,
        ],
        axis=0,
    ).reshape(4, H, W).astype(np.float32, order="C")
    return np.ascontiguousarray(bev)
