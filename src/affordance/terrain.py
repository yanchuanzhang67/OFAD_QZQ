"""Minimal terrain-affordance v1: traversability and roughness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.schema import AffordanceOutput


@dataclass(frozen=True)
class TerrainAffordanceConfig:
    in_channels: int = 32
    hidden_channels: int = 32
    traversability_weight: float = 1.0
    roughness_weight: float = 1.0
    speed_epsilon: float = 0.5

    def __post_init__(self):
        if self.in_channels <= 0 or self.hidden_channels <= 0:
            raise ValueError("affordance channels must be > 0")
        if self.traversability_weight < 0 or self.roughness_weight < 0:
            raise ValueError("affordance loss weights must be >= 0")
        if self.speed_epsilon <= 0:
            raise ValueError("speed_epsilon must be > 0")


class TerrainAffordance(nn.Module):
    """A deliberately small multi-task head; BEV remains the stable baseline."""

    def __init__(self, config: Optional[TerrainAffordanceConfig] = None):
        super().__init__()
        self.config = config or TerrainAffordanceConfig()
        c = self.config
        self.encoder = nn.Sequential(
            nn.Conv2d(c.in_channels, c.hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c.hidden_channels, c.hidden_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.traversability_head = nn.Conv2d(c.hidden_channels, 1, 1)
        self.roughness_head = nn.Conv2d(c.hidden_channels, 1, 1)

    def forward(self, bev: torch.Tensor) -> AffordanceOutput:
        if bev.ndim != 4 or bev.shape[1] != self.config.in_channels:
            raise ValueError(
                f"BEV channels must be {self.config.in_channels}, "
                f"got shape {tuple(bev.shape)}")
        feature = self.encoder(bev)
        return AffordanceOutput(
            feature=feature,
            traversability=torch.sigmoid(self.traversability_head(feature)),
            roughness=F.softplus(self.roughness_head(feature)),
        )

    def loss(self, output: AffordanceOutput, traversability: torch.Tensor,
             roughness: torch.Tensor,
             roughness_mask: Optional[torch.Tensor] = None
             ) -> Dict[str, torch.Tensor]:
        if traversability.shape != output.traversability.shape:
            raise ValueError("traversability target shape mismatch")
        if roughness.shape != output.roughness.shape:
            raise ValueError("roughness target shape mismatch")
        trav_loss = F.binary_cross_entropy(
            output.traversability, traversability.to(output.traversability.dtype))
        error = (output.roughness - roughness).abs()
        if roughness_mask is None:
            rough_loss = error.mean()
        else:
            if roughness_mask.shape != error.shape:
                raise ValueError("roughness mask shape mismatch")
            mask = roughness_mask.to(error.dtype)
            rough_loss = (error * mask).sum() / mask.sum().clamp(min=1.0)
        total = (self.config.traversability_weight * trav_loss
                 + self.config.roughness_weight * rough_loss)
        return {"traversability": trav_loss, "roughness": rough_loss,
                "total": total}


def roughness_target_from_imu(imu: torch.Tensor, speed: torch.Tensor,
                              epsilon: float = 0.5) -> torch.Tensor:
    """Speed-conditioned weak roughness proxy RMS(a_z-mean)/(abs(v)+eps)."""
    if imu.ndim != 3 or imu.shape[-1] != 6:
        raise ValueError(f"imu must be (B,T,6), got {tuple(imu.shape)}")
    if speed.ndim != 1 or speed.shape[0] != imu.shape[0]:
        raise ValueError("speed must be (B,) and match IMU batch")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    az = imu[..., 2]
    centred = az - az.mean(dim=1, keepdim=True)
    rms = torch.sqrt((centred.square()).mean(dim=1) + 1e-8)
    return rms / (speed.abs() + epsilon)
