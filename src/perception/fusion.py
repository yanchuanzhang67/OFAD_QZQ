"""BEV-space feature fusion with a strict dual-sensor health contract."""

import torch
import torch.nn as nn

__all__ = ["ModalityAwareBEVFuser"]


class ModalityAwareBEVFuser(nn.Sequential):
    """Fuse independent camera/LiDAR BEV streams using a compact Conv fuser.

    ``availability`` is an optional ``(B, 2)`` camera/LiDAR mask, ordered as
    ``[camera_valid, lidar_valid]``. ORAD's online fusion contract is strict:
    every sample must have both entries set. A missing/invalid branch is
    rejected instead of being silently replaced by a zero feature map. With no
    mask, callers assert that both already passed upstream validation.

    Subclassing ``nn.Sequential`` intentionally preserves historical state
    dict keys such as ``fuser.0.weight``.
    """

    def __init__(self, camera_channels: int, lidar_channels: int,
                 out_channels: int):
        self.camera_channels = int(camera_channels)
        self.lidar_channels = int(lidar_channels)
        self.out_channels = int(out_channels)
        super().__init__(
            nn.Conv2d(
                self.camera_channels + self.lidar_channels,
                self.out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def _validate_features(self, camera: torch.Tensor,
                           lidar: torch.Tensor) -> None:
        if camera.ndim != 4 or lidar.ndim != 4:
            raise ValueError("camera and lidar BEV features must be 4-D")
        if camera.shape[0] != lidar.shape[0] or (
                camera.shape[-2:] != lidar.shape[-2:]):
            raise ValueError("camera and lidar BEV batch/spatial shapes differ")
        if camera.shape[1] != self.camera_channels:
            raise ValueError("camera BEV channel count does not match fuser")
        if lidar.shape[1] != self.lidar_channels:
            raise ValueError("lidar BEV channel count does not match fuser")

    def validate_availability(self, availability, batch_size: int,
                              device) -> torch.Tensor:
        """Validate and return the canonical ``[camera, lidar]`` mask."""
        mask = torch.as_tensor(availability, device=device)
        expected = (batch_size, 2)
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"modality availability shape must be {expected}")
        if not torch.isfinite(mask.to(torch.float32)).all():
            raise ValueError("modality availability must be finite")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("modality availability values must be boolean")
        if not torch.all(mask == 1):
            raise ValueError(
                "both camera and lidar modalities must be available")
        return mask

    def _apply_availability(self, camera: torch.Tensor, lidar: torch.Tensor,
                            availability) -> tuple:
        mask = self.validate_availability(
            availability, camera.shape[0], camera.device)
        mask = mask.to(dtype=camera.dtype)
        camera_mask = mask[:, 0].view(-1, 1, 1, 1)
        lidar_mask = mask[:, 1].view(-1, 1, 1, 1)
        return camera * camera_mask, lidar * lidar_mask

    def forward(self, camera: torch.Tensor, lidar: torch.Tensor,
                availability=None) -> torch.Tensor:
        self._validate_features(camera, lidar)
        if availability is not None:
            camera, lidar = self._apply_availability(
                camera, lidar, availability)
        return super().forward(torch.cat([camera, lidar], dim=1))
