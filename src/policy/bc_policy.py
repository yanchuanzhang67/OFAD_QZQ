"""Deterministic B0 Behavior-Cloning trajectory policy."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .config import BCPolicyConfig

__all__ = [
    "BCPolicy",
    "TrajectoryDecoder",
    "masked_bc_loss_components",
]


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, out_dim),
        nn.ELU(),
    )


class TrajectoryDecoder(nn.Module):
    """Decode a deterministic latent into fixed-horizon ego waypoints."""

    def __init__(self, config: BCPolicyConfig):
        super().__init__()
        self.horizon = config.horizon
        self.projection = _mlp(
            config.latent_dim, config.latent_dim, config.latent_dim)
        self.gru = nn.GRU(
            config.latent_dim, config.latent_dim, batch_first=True)
        self.head = nn.Linear(config.latent_dim, config.traj_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        repeated = self.projection(latent).unsqueeze(1).expand(
            -1, self.horizon, -1)
        decoded, _ = self.gru(repeated)
        return self.head(decoded)


def _validate_loss_inputs(
        prediction: torch.Tensor, expert: torch.Tensor,
        valid_mask: torch.Tensor, config: BCPolicyConfig) -> None:
    expected_tail = (config.horizon, config.traj_dim)
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != expected_tail:
        raise ValueError(
            f"prediction shape must be (B,{config.horizon},{config.traj_dim})")
    if expert.shape != prediction.shape:
        raise ValueError(
            f"expert shape {tuple(expert.shape)} must match prediction shape "
            f"{tuple(prediction.shape)}")
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} must match "
            f"{tuple(prediction.shape[:2])}")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must have bool dtype")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction must contain only finite values")
    if not torch.isfinite(expert).all():
        raise ValueError("expert must contain only finite values")
    if not valid_mask.any(dim=1).all():
        raise ValueError("every sample must contain at least one valid waypoint")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    return values.masked_select(expanded).mean()


def masked_bc_loss_components(
        prediction: torch.Tensor, expert: torch.Tensor,
        valid_mask: torch.Tensor,
        config: BCPolicyConfig) -> Dict[str, torch.Tensor | int]:
    """Compute the mask-aware, angle-wrapped B0 supervised objective."""
    _validate_loss_inputs(prediction, expert, valid_mask, config)
    xy = _masked_mean(
        (prediction[..., :2] - expert[..., :2]).square(), valid_mask)
    heading_delta = prediction[..., 2] - expert[..., 2]
    wrapped_heading = torch.atan2(
        torch.sin(heading_delta), torch.cos(heading_delta))
    heading = _masked_mean(wrapped_heading.square(), valid_mask)
    speed = _masked_mean(
        (prediction[..., 3] - expert[..., 3]).square(), valid_mask)

    smooth = prediction.new_zeros(())
    if config.horizon >= 3:
        triplet_mask = (
            valid_mask[:, :-2]
            & valid_mask[:, 1:-1]
            & valid_mask[:, 2:])
        if triplet_mask.any():
            second_difference = (
                prediction[:, 2:, :2]
                - 2.0 * prediction[:, 1:-1, :2]
                + prediction[:, :-2, :2])
            smooth = _masked_mean(second_difference.square(), triplet_mask)

    total = (
        config.bc_xy_weight * xy
        + config.bc_heading_weight * heading
        + config.bc_speed_weight * speed
        + config.bc_smooth_weight * smooth)
    return {
        "xy": xy,
        "heading": heading,
        "speed": speed,
        "smooth": smooth,
        "total": total,
        "valid_waypoints": int(valid_mask.sum().item()),
    }


class BCPolicy(nn.Module):
    """Pure BC policy with BEV, IMU and ego-dynamics encoders."""

    def __init__(self, config: BCPolicyConfig | None = None):
        super().__init__()
        self.config = config or BCPolicyConfig()
        c = self.config
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(c.bev_channels, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, c.encoder_hidden),
            nn.ELU(),
        )
        self.imu_encoder = nn.GRU(
            c.imu_in_channels, c.encoder_hidden, batch_first=True)
        self.ego_encoder = _mlp(c.ego_dim, c.ego_hidden, c.ego_hidden)
        self.fusion = _mlp(
            2 * c.encoder_hidden + c.ego_hidden,
            c.latent_dim,
            c.latent_dim)
        self.decoder = TrajectoryDecoder(c)
        self.register_buffer("ego_mean", torch.zeros(c.ego_dim))
        self.register_buffer("ego_std", torch.ones(c.ego_dim))

    def set_ego_normalization(
            self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected = (self.config.ego_dim,)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                f"ego normalization shape must be {expected}")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("ego normalization values must be finite")
        if not (std > 1e-6).all():
            raise ValueError("ego normalization std must be positive above 1e-6")
        self.ego_mean.copy_(mean.to(self.ego_mean))
        self.ego_std.copy_(std.to(self.ego_std))

    def _validate_inputs(
            self, bev: torch.Tensor, imu: torch.Tensor,
            ego: torch.Tensor) -> None:
        c = self.config
        expected = {
            "bev": (c.bev_channels, c.bev_h, c.bev_w),
            "imu": (c.imu_steps, c.imu_in_channels),
            "ego": (c.ego_dim,),
        }
        values = {"bev": bev, "imu": imu, "ego": ego}
        batch_size = None
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != len(expected[name]) + 1:
                raise ValueError(
                    f"{name} shape must be (B,{','.join(map(str, expected[name]))})")
            if tuple(value.shape[1:]) != expected[name]:
                raise ValueError(
                    f"{name} shape must be (B,{','.join(map(str, expected[name]))})")
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("BEV, IMU and ego batch sizes must match")
            if not value.is_floating_point():
                raise TypeError(f"{name} must use a floating dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")

    def forward(
            self, bev: torch.Tensor, imu: torch.Tensor,
            ego: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(bev, imu, ego)
        bev_feature = self.bev_encoder(bev)
        _, imu_hidden = self.imu_encoder(imu)
        normalized_ego = (ego - self.ego_mean) / self.ego_std
        ego_feature = self.ego_encoder(normalized_ego)
        latent = self.fusion(torch.cat([
            bev_feature, imu_hidden.squeeze(0), ego_feature], dim=-1))
        trajectory = self.decoder(latent)
        if not torch.isfinite(trajectory).all():
            raise ValueError("BC trajectory must contain only finite values")
        return trajectory

    def bc_loss_components(
            self, bev: torch.Tensor, imu: torch.Tensor, ego: torch.Tensor,
            expert: torch.Tensor,
            valid_mask: torch.Tensor) -> Dict[str, torch.Tensor | int]:
        prediction = self.forward(bev, imu, ego)
        return masked_bc_loss_components(
            prediction, expert, valid_mask, self.config)

    def bc_loss(
            self, bev: torch.Tensor, imu: torch.Tensor, ego: torch.Tensor,
            expert: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        result = self.bc_loss_components(
            bev, imu, ego, expert, valid_mask)["total"]
        if not isinstance(result, torch.Tensor):  # pragma: no cover - invariant
            raise TypeError("BC total loss must be a tensor")
        return result
