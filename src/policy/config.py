"""Configuration contract for the deterministic Pure BC policy."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BCPolicyConfig"]


@dataclass(frozen=True)
class BCPolicyConfig:
    """Static dimensions and loss weights for ``pure_bc_v1``."""

    family: str = "pure_bc_v1"
    bev_channels: int = 32
    bev_h: int = 50
    bev_w: int = 50
    imu_in_channels: int = 6
    imu_steps: int = 10
    ego_dim: int = 8
    encoder_hidden: int = 256
    ego_hidden: int = 64
    latent_dim: int = 256
    horizon: int = 20
    traj_dim: int = 4
    waypoint_dt: float = 0.1
    bc_xy_weight: float = 1.0
    bc_heading_weight: float = 0.2
    bc_speed_weight: float = 0.2
    bc_smooth_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.family != "pure_bc_v1":
            raise ValueError("family must be pure_bc_v1")
        dimensions = {
            "bev_channels": self.bev_channels,
            "bev_h": self.bev_h,
            "bev_w": self.bev_w,
            "imu_in_channels": self.imu_in_channels,
            "imu_steps": self.imu_steps,
            "ego_dim": self.ego_dim,
            "encoder_hidden": self.encoder_hidden,
            "ego_hidden": self.ego_hidden,
            "latent_dim": self.latent_dim,
            "horizon": self.horizon,
            "traj_dim": self.traj_dim,
        }
        invalid = [name for name, value in dimensions.items()
                   if not isinstance(value, int) or value <= 0]
        if invalid:
            raise ValueError(
                f"Pure BC dimensions must be positive integers: {invalid}")
        if self.ego_dim != 8:
            raise ValueError("ego_dim must be 8 for ego-dynamics-v1")
        if self.traj_dim != 4:
            raise ValueError("traj_dim must be 4 for [x,y,heading,velocity]")
        if self.waypoint_dt <= 0:
            raise ValueError("waypoint_dt must be positive")
        weights = {
            "bc_xy_weight": self.bc_xy_weight,
            "bc_heading_weight": self.bc_heading_weight,
            "bc_speed_weight": self.bc_speed_weight,
            "bc_smooth_weight": self.bc_smooth_weight,
        }
        invalid_weights = [name for name, value in weights.items()
                           if value < 0]
        if invalid_weights:
            raise ValueError(
                f"Pure BC loss weights must be non-negative: {invalid_weights}")
