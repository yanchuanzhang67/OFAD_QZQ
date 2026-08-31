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

from perception.encoders import ImageEncoder, ImuEncoder
from perception.fusion import ModalityAwareBEVFuser
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
    Camera and LiDAR form one strict safety contract: either branch being
    absent, malformed or non-finite invalidates the complete fusion request.
    """

    def __init__(self, config: Optional[BEVFusionConfig] = None):
        super().__init__()
        self.config = config or BEVFusionConfig()
        c = self.config
        # 关键坐标约定：BEV 使用图像 (C,H,W) 布局，H 对应 y/row，W 对应
        # x/col。后续相机、点云、占据栅格均必须遵守该约定，避免非对称
        # BEV 范围下出现 x/y 互换。
        self.bev_h = _grid_n(c.bev_y_range, c.bev_resolution)
        self.bev_w = _grid_n(c.bev_x_range, c.bev_resolution)
        self.x0 = float(c.bev_x_range[0])
        self.y0 = float(c.bev_y_range[0])

        # 三个编码支路保持独立；Camera/LiDAR 只在公共 BEV 坐标系融合，
        # IMU 则作为时序姿态特征在融合后注入。
        self.image_encoder = ImageEncoder(
            c.image_channels, c.cam_feat_channels, c.depth_bins)
        self.pillar_mlp = nn.Sequential(
            nn.Linear(c.lidar_in_channels, c.pillar_feat_channels), nn.ReLU(),
            nn.Linear(c.pillar_feat_channels, c.pillar_feat_channels), nn.ReLU(),
        )
        self.imu_encoder = ImuEncoder(
            c.imu_in_channels, c.imu_hidden, c.bev_channels)

        self.fuser = ModalityAwareBEVFuser(
            c.cam_feat_channels, c.pillar_feat_channels, c.bev_channels)
        self.occupancy_head = nn.Conv2d(c.bev_channels, 1, kernel_size=1)

        # 初始化时探测特征图尺寸，随后将 frustum->BEV 映射固化为 buffer；
        # 运行时不创建依赖输入内容的动态索引，便于 ONNX/TensorRT 导出。
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
        """Pad/truncate a *valid, non-empty* cloud to the deployment shape.

        输入有效性在 :meth:`_validate_input_shapes` 中先行检查。这里的零填充
        仅用于静态图对齐，不代表“空 LiDAR 数据仍然有效”。
        """
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
        # valid 区分真实点与静态 shape 的补零点；越界点保留在输入中，但不
        # 参与 scatter，防止 clamp 后错误堆积到 BEV 边缘栅格。
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
        # Compact LSS：每个像素特征与离散深度概率做外积，将 2D 特征
        # lift 成 (feature, depth) 体，再交给静态 frustum 索引 splat。
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
             imu: torch.Tensor, modality_mask=None) -> torch.Tensor:
        # modality_mask 的列顺序固定为 [camera_valid, lidar_valid]。
        # 显式 mask 下两列必须全部为 True；任一 False 都由 fuser 直接拒绝。
        fused = self.fuser(bev_cam, bev_lidar, modality_mask)
        # IMU 不替代视觉/点云。它只提供广播到整个 BEV 的时序运动偏置。
        imu_feat = self.imu_encoder(imu).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return (fused + imu_feat).contiguous().float()

    @staticmethod
    def _validate_finite(name: str, value: torch.Tensor) -> None:
        """Reject poisoned sensor tensors on eager paths.

        Finite-value decisions are data-dependent and therefore deliberately
        omitted while tracing/exporting. Online Python/ROS2/CARLA entry points
        execute this eager guard before any encoder consumes sensor data.
        """
        if (torch.jit.is_tracing()
                or torch.onnx.is_in_onnx_export()):
            return
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")

    def _validate_input_shapes(self, images: torch.Tensor,
                               points: torch.Tensor,
                               imu: torch.Tensor,
                               attitude: Optional[torch.Tensor]) -> None:
        c = self.config
        expected_image = (
            c.num_cameras, c.image_channels, *c.image_size)
        if images.ndim != 5 or tuple(images.shape[1:]) != expected_image:
            raise ValueError(
                f"images shape must be (B,{','.join(map(str, expected_image))})")
        if points.ndim != 3 or points.shape[-1] != c.lidar_in_channels:
            raise ValueError(
                f"points shape must be (B,P,{c.lidar_in_channels})")
        if points.shape[1] == 0:
            raise ValueError("points must contain at least one LiDAR return")
        expected_imu = (c.imu_steps, c.imu_in_channels)
        if imu.ndim != 3 or tuple(imu.shape[1:]) != expected_imu:
            raise ValueError(
                f"imu shape must be (B,{c.imu_steps},{c.imu_in_channels})")
        if not (images.shape[0] == points.shape[0] == imu.shape[0]):
            raise ValueError("images, points and imu batch sizes must match")
        if attitude is not None and tuple(attitude.shape) != (
                images.shape[0], 3, 3):
            raise ValueError("imu_attitude shape must be (B,3,3)")
        # 形状合法并不代表数据可用：NaN/Inf 会污染卷积、scatter 和最终
        # occupancy，因此必须在编码前 fail-fast。Camera 或 LiDAR 任一失败
        # 都会使本次融合整体无效。
        self._validate_finite("images", images)
        self._validate_finite("points", points)

    # -- main forward ---------------------------------------------------
    def forward(self, images: torch.Tensor, points: torch.Tensor,
                imu: torch.Tensor,
                imu_attitude: Optional[torch.Tensor] = None,
                modality_mask=None) -> BEVFeature:
        """Run full multi-modal fusion.

        Args:
            images:       (B, N, C, H, W) multi-view RGB
            points:       (B, P', 4) LiDAR (x, y, z, intensity); canonicalised to num_points
            imu:          (B, T, 6) IMU history [ax, ay, az, gx, gy, gz]
            imu_attitude: (B, 3, 3) optional body->world rotation for gravity-aligned BEV
            modality_mask: optional ``(B, 2)`` boolean mask ordered as
                ``[camera_valid, lidar_valid]``. If supplied, both values must
                be true for every sample. ``None`` means both modalities have
                already passed the validation above.
        Returns:
            :class:`BEVFeature` with ``bev`` (B, C_out, H_bev, W_bev) float32.
        """
        # 严格校验先于任何神经网络运算，确保坏帧不会被编码成貌似正常的
        # BEV；上层闭环捕获 ValueError 后应进入紧急制动/故障安全路径。
        self._validate_input_shapes(images, points, imu, imu_attitude)
        if modality_mask is not None:
            # mask 必须在 encoder 前校验；已声明某传感器无效时不应继续消耗
            # 计算资源，也不应产生可被误用的中间 BEV feature。
            self.fuser.validate_availability(
                modality_mask, images.shape[0], images.device)
        bev_lidar = self.encode_lidar(points, imu_attitude)
        lifted, Cf, _ = self.encode_images(images)
        bev_cam = self._splat_cameras(lifted, Cf)
        bev = self.fuse(bev_cam, bev_lidar, imu, modality_mask)
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
