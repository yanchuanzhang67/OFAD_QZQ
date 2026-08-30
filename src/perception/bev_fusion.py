"""Multi-modal BEV fusion (RGB + LiDAR + IMU) -> fused BEV feature volume.

Architecture (Phase 2):
  * Camera branch : compact Lift-Splat-Shoot (LSS). A small CNN produces a
    per-pixel feature map and a categorical depth distribution; the outer
    product is scattered into a BEV grid via a *precomputed, static* frustum
    scatter pattern (one batched ``scatter_add`` -> ONNX ``ScatterElements``).
  * LiDAR branch  : PointPillars-lite. Each point is encoded by an MLP and
    scattered into the same BEV grid with a batched ``scatter_add``. The point
    cloud is canonicalised to a FIXED ``num_points`` (zero-pad + valid mask)
    so every shape in the graph is static.
  * IMU branch     : GRU over the IMU history -> broadcast bias on the BEV.
  * Attitude      : an optional body->world rotation matrix ``imu_attitude``
    rotates the LiDAR points into a gravity-aligned frame before projection
    (off-road pitch/roll compensation).

TensorRT / deployment notes
--------------------------
The whole ``forward`` is traceable and uses only **static-shape** ops:

* no Python loops over the batch dimension,
* no ``nonzero`` / data-dependent reshape / dynamic-output ops,
* all scatter indices are precomputed (camera) or data-valued but fixed-size
  (LiDAR) with a *fixed* output tensor.

Dynamic batch is the only dynamic axis allowed in ONNX export (supported by
TensorRT optimization profiles); every other dim is static.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.types import BEVFeature

__all__ = ["BEVFusionConfig", "BEVFusion"]


def _grid_n(rng: Tuple[float, float], res: float) -> int:
    return max(1, int(round((float(rng[1]) - float(rng[0])) / res)))


@dataclass
class BEVFusionConfig:
    """Configuration for :class:`BEVFusion` (all dims static for TensorRT)."""

    num_cameras: int = 2
    image_channels: int = 3
    image_size: Tuple[int, int] = (64, 64)       # (H, W)
    num_points: int = 256                        # FIXED LiDAR point count
    # camera / LSS
    cam_feat_channels: int = 16
    depth_bins: int = 4
    depth_min: float = 1.0
    depth_max: float = 20.0
    # BEV grid (static)
    bev_x_range: Tuple[float, float] = (-12.5, 12.5)
    bev_y_range: Tuple[float, float] = (-12.5, 12.5)
    bev_resolution: float = 0.5
    bev_channels: int = 32                       # fused output channels
    # LiDAR pillar encoder
    lidar_in_channels: int = 4                   # x, y, z, intensity
    pillar_feat_channels: int = 16
    # IMU history encoder
    imu_in_channels: int = 6                     # ax, ay, az, gx, gy, gz
    imu_hidden: int = 32
    imu_steps: int = 10
    # cameras (optional; auto-generated pinhole if None)
    camera_intrinsics: Optional[List[np.ndarray]] = None  # list of (3, 3)
    camera_extrinsics: Optional[List[np.ndarray]] = None  # list of (4, 4) cam->ego


class _ImageEncoder(nn.Module):
    """Tiny CNN -> (feat, depth-distribution) per camera."""

    def __init__(self, in_ch: int, feat_ch: int, depth_bins: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1), nn.ReLU(),
        )
        self.depth_head = nn.Conv2d(feat_ch, depth_bins, 1)

    def forward(self, x):
        feat = self.backbone(x)                         # (B*N, Cf, Hf, Wf)
        depth = F.softmax(self.depth_head(feat), dim=1)  # (B*N, D, Hf, Wf)
        return feat, depth


class _ImuEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_ch: int):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_ch)

    def forward(self, imu):                            # (B, T, in_ch)
        _, h = self.gru(imu)                            # (1, B, hidden)
        return self.proj(h.squeeze(0))                  # (B, out_ch)


def _default_cameras(num_cameras: int, image_size: Tuple[int, int]):
    """Plausible pinhole intrinsics + spread forward extrinsics."""
    H, W = image_size
    fx = fy = float(W)
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsics, extrinsics = [], []
    for i in range(num_cameras):
        yaw = (i - (num_cameras - 1) / 2.0) * 0.6
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = R
        E[:3, 3] = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        intrinsics.append(K.copy())
        extrinsics.append(E)
    return intrinsics, extrinsics


class BEVFusion(nn.Module):
    """Modular E2E perception head producing a fused BEV feature volume.

    All public methods are batched (no per-sample Python loops) and operate on
    static shapes, making the module directly exportable to ONNX / TensorRT.
    """

    def __init__(self, config: Optional[BEVFusionConfig] = None):
        super().__init__()
        self.config = config or BEVFusionConfig()
        c = self.config
        # BEV grid follows image convention (C, H, W): H <- y_range (rows),
        # W <- x_range (cols). This aligns with utils.transforms._grid_dims
        # and OccupancyGrid (rows=y, cols=x). [fix for BUG-1]
        self.bev_h = _grid_n(c.bev_y_range, c.bev_resolution)
        self.bev_w = _grid_n(c.bev_x_range, c.bev_resolution)
        self.x0 = float(c.bev_x_range[0])
        self.y0 = float(c.bev_y_range[0])

        self.image_encoder = _ImageEncoder(c.image_channels, c.cam_feat_channels,
                                           c.depth_bins)
        self.pillar_mlp = nn.Sequential(
            nn.Linear(c.lidar_in_channels, c.pillar_feat_channels), nn.ReLU(),
            nn.Linear(c.pillar_feat_channels, c.pillar_feat_channels), nn.ReLU(),
        )
        self.imu_encoder = _ImuEncoder(c.imu_in_channels, c.imu_hidden, c.bev_channels)

        in_ch = c.cam_feat_channels + c.pillar_feat_channels
        self.fuser = nn.Sequential(
            nn.Conv2d(in_ch, c.bev_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c.bev_channels, c.bev_channels, 3, padding=1), nn.ReLU(),
        )
        self.occupancy_head = nn.Conv2d(c.bev_channels, 1, kernel_size=1)

        # Discover the actual feature-map size via a single dry forward.
        with torch.no_grad():
            dummy = torch.zeros(1, c.image_channels, *c.image_size)
            feat, _ = self.image_encoder(dummy)
        self._feat_h, self._feat_w = int(feat.shape[-2]), int(feat.shape[-1])

        Ks, Es = (c.camera_intrinsics, c.camera_extrinsics)
        if Ks is None or Es is None:
            Ks, Es = _default_cameras(c.num_cameras, c.image_size)
        cam_index, cam_valid = self._build_frustum(Ks, Es)  # (P_total,) long/bool
        self.register_buffer("cam_index", cam_index)
        self.register_buffer("cam_valid", cam_valid)

    # -- geometry (init-time, static) -----------------------------------
    def _build_frustum(self, Ks, Es):
        """Precompute the static per-camera frustum -> BEV scatter pattern.

        Returns flattened indices ``(P_total,)`` and a validity mask, ordered
        as ``(n, d, hf, wf)`` to match :meth:`encode_images`' reshape.
        """
        c = self.config
        D, Hf, Wf = c.depth_bins, self._feat_h, self._feat_w
        depths = c.depth_min + (c.depth_max - c.depth_min) * (np.arange(D) + 0.5) / D
        stride_h = c.image_size[0] / Hf
        stride_w = c.image_size[1] / Wf
        v, u = np.meshgrid(np.arange(Hf), np.arange(Wf), indexing="ij")
        idx_list, valid_list = [], []
        for K, E in zip(Ks, Es):
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            u_img = u * stride_w + stride_w / 2.0
            v_img = v * stride_h + stride_h / 2.0
            xc = (u_img - cx) / fx
            yc = (v_img - cy) / fy
            Xc = depths[:, None, None] * xc[None]
            Yc = depths[:, None, None] * yc[None]
            Zc = np.broadcast_to(depths[:, None, None], Xc.shape)
            pts = np.stack([Xc, Yc, Zc], axis=-1).reshape(-1, 3).astype(np.float32)
            homo = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=1)
            ego = (np.asarray(E, np.float32) @ homo.T).T[:, :3]
            col = np.floor((ego[:, 0] - self.x0) / c.bev_resolution).astype(np.int64)
            row = np.floor((ego[:, 1] - self.y0) / c.bev_resolution).astype(np.int64)
            valid = (col >= 0) & (col < self.bev_w) & (row >= 0) & (row < self.bev_h)
            idx = row * self.bev_w + col
            idx[~valid] = 0
            idx_list.append(idx)
            valid_list.append(valid)
        cam_index = torch.from_numpy(np.concatenate(idx_list).astype(np.int64))
        cam_valid = torch.from_numpy(np.concatenate(valid_list))
        return cam_index, cam_valid

    # -- IMU attitude coordinate transform ------------------------------
    def transform_points_to_world(self, points: torch.Tensor,
                                  attitude: torch.Tensor) -> torch.Tensor:
        """Rotate point-cloud xyz by a body->world attitude matrix.

        Args:
            points:  (B, P, >=3) [..., x, y, z, (intensity)]
            attitude: (B, 3, 3) rotation matrix (body -> world)
        Returns:
            (B, P, >=3) with xyz rotated; extra channels (e.g. intensity)
            preserved. Uses a single batched ``einsum`` (MatMul) -> static shape.
        """
        xyz = points[..., :3]
        world = torch.einsum("bij,bpj->bpi", attitude, xyz)
        out = points.clone()
        out[..., :3] = world
        return out

    # -- point cloud canonicalisation (static shape) --------------------
    def _canonicalize_points(self, points: torch.Tensor):
        """Zero-pad / truncate to ``num_points`` and return a valid mask."""
        B, n = points.shape[0], points.shape[1]
        P = self.config.num_points
        device, dtype = points.device, points.dtype
        if n == P:
            return points, torch.ones(B, P, dtype=torch.bool, device=device)
        if n > P:
            return points[:, :P], torch.ones(B, P, dtype=torch.bool, device=device)
        pad = torch.zeros(B, P - n, points.shape[-1], device=device, dtype=dtype)
        out = torch.cat([points, pad], dim=1)
        valid = torch.zeros(B, P, dtype=torch.bool, device=device)
        valid[:, :n] = True
        return out, valid

    # -- LiDAR branch (vectorised, static-shape scatter) ----------------
    def encode_lidar(self, points: torch.Tensor,
                     attitude: Optional[torch.Tensor]) -> torch.Tensor:
        pts, valid = self._canonicalize_points(points)        # (B, P, in_ch)
        if attitude is not None:
            pts = self.transform_points_to_world(pts, attitude)
        feats = self.pillar_mlp(pts)                          # (B, P, Cl)
        feats = feats * valid.unsqueeze(-1).to(feats.dtype)
        col = ((pts[..., 0] - self.x0) / self.config.bev_resolution).floor().long()
        row = ((pts[..., 1] - self.y0) / self.config.bev_resolution).floor().long()
        inb = valid & (col >= 0) & (col < self.bev_w) & (row >= 0) & (row < self.bev_h)
        col = col.clamp(0, self.bev_w - 1)
        row = row.clamp(0, self.bev_h - 1)
        feats = feats * inb.unsqueeze(-1).to(feats.dtype)
        Cl = self.config.pillar_feat_channels
        idx = (row * self.bev_w + col)                        # (B, P) in [0, H*W)
        bev_flat = torch.zeros(pts.shape[0], Cl, self.bev_h * self.bev_w,
                               device=pts.device, dtype=feats.dtype)
        index = idx.unsqueeze(1).expand(-1, Cl, -1)            # (B, Cl, P)
        src = feats.permute(0, 2, 1).contiguous()             # (B, Cl, P)
        bev_flat.scatter_add_(2, index, src)                  # (B, Cl, H*W)
        return bev_flat.view(pts.shape[0], Cl, self.bev_h, self.bev_w)

    # -- camera branch (vectorised LSS splat) ---------------------------
    def encode_images(self, images: torch.Tensor):
        B, N = images.shape[0], images.shape[1]
        feat, depth = self.image_encoder(images.view(B * N, *images.shape[2:]))
        Cf, Hf, Wf = feat.shape[-3], feat.shape[-2], feat.shape[-1]
        feat = feat.view(B, N, Cf, Hf, Wf)
        depth = depth.view(B, N, -1, Hf, Wf)
        lifted = feat.unsqueeze(3) * depth.unsqueeze(2)        # (B, N, Cf, D, Hf, Wf)
        lifted = lifted.permute(0, 2, 1, 3, 4, 5)              # (B, Cf, N, D, Hf, Wf)
        P_total = N * self.config.depth_bins * Hf * Wf
        lifted = lifted.reshape(B, Cf, P_total)               # (B, Cf, P_total)
        return lifted, Cf, P_total

    def _splat_cameras(self, lifted: torch.Tensor, Cf: int) -> torch.Tensor:
        B = lifted.shape[0]
        lifted = lifted * self.cam_valid.to(lifted.dtype).view(1, 1, -1)
        bev_flat = torch.zeros(B, Cf, self.bev_h * self.bev_w,
                               device=lifted.device, dtype=lifted.dtype)
        index = self.cam_index.view(1, 1, -1).expand(B, Cf, -1)  # (B, Cf, P_total)
        bev_flat.scatter_add_(2, index, lifted)
        return bev_flat.view(B, Cf, self.bev_h, self.bev_w)

    # -- fusion ---------------------------------------------------------
    def fuse(self, bev_cam: torch.Tensor, bev_lidar: torch.Tensor,
             imu: torch.Tensor) -> torch.Tensor:
        fused = self.fuser(torch.cat([bev_cam, bev_lidar], dim=1))   # (B, C, H, W)
        imu_feat = self.imu_encoder(imu).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return (fused + imu_feat).contiguous().float()

    # -- main forward ---------------------------------------------------
    def forward(self, images: torch.Tensor, points: torch.Tensor,
                imu: torch.Tensor,
                imu_attitude: Optional[torch.Tensor] = None) -> BEVFeature:
        """Run full multi-modal fusion.

        Args:
            images:       (B, N, C, H, W) multi-view RGB
            points:       (B, P', 4) LiDAR (x, y, z, intensity); canonicalised to num_points
            imu:          (B, T, 6) IMU history [ax, ay, az, gx, gy, gz]
            imu_attitude: (B, 3, 3) optional body->world rotation for gravity-aligned BEV
        Returns:
            :class:`BEVFeature` with ``bev`` (B, C_out, H_bev, W_bev) float32.
        """
        bev_lidar = self.encode_lidar(points, imu_attitude)
        lifted, Cf, _ = self.encode_images(images)
        bev_cam = self._splat_cameras(lifted, Cf)
        bev = self.fuse(bev_cam, bev_lidar, imu)
        occupancy = torch.sigmoid(self.occupancy_head(bev))
        return BEVFeature(bev=bev, occupancy=occupancy, imu_state=imu,
                          metadata={"bev_hw": (self.bev_h, self.bev_w),
                                    "num_points": self.config.num_points})

    # -- deployment (static-shape export) --------------------------------
    def export_onnx(self, path: str, sample_images: torch.Tensor,
                    sample_points: torch.Tensor, sample_imu: torch.Tensor,
                    sample_attitude: Optional[torch.Tensor] = None) -> None:
        """Export the tensor path to ONNX. Only the batch axis is dynamic."""
        class _Wrapper(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def forward(self, images, points, imu, attitude):
                return self.parent(images, points, imu, imu_attitude=attitude).bev
        wrapper = _Wrapper(self).eval()
        inputs = (sample_images, sample_points, sample_imu,
                  sample_attitude if sample_attitude is not None
                  else torch.eye(3).repeat(sample_images.shape[0], 1, 1))
        with torch.no_grad():
            torch.onnx.export(
                wrapper, inputs, path,
                input_names=["images", "points", "imu", "attitude"],
                output_names=["bev"],
                dynamic_axes={"images": {0: "batch"}, "points": {0: "batch"},
                              "imu": {0: "batch"}, "attitude": {0: "batch"},
                              "bev": {0: "batch"}},
                opset_version=17,
            )
