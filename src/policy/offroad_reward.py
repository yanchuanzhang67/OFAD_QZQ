"""Off-road reward function for RL fine-tuning (Phase 3).

A single *differentiable* (PyTorch) reward aggregating four off-road-specific
components. It is a pure function of a candidate trajectory (and optional
context) so it serves two roles without duplication:

1. Real trajectory evaluation  -- after an env rollout, yields the scalar
   reward used to supervise the world-model reward predictor.
2. Latent imagination rollouts -- inside the Dreamer-style policy gradient,
   backpropagates through the actor (trajectory is actor-output).

Components (weighted SUM -> per-batch scalar r in R^B)::

    R1 progress   : forward displacement of consecutive waypoints projected
                    onto the heading direction (>0 when advancing).
    R2 collision  : occupancy-grid sampling at every waypoint; penalty
                    proportional to occupancy probability (soft, bilinear
                    ``grid_sample``).
    R3 attitude   : |d pitch/dt| + |d roll/dt| (angular-rate, comfort) PLUS a
                    hard limit penalty when |pitch|/|roll| exceed the rollover
                    threshold (anti-rollover -- off-road specific term).
    R4 jerk       : second time-derivative of speed (jerk = d accel/dt);
                    squared sum (longitudinal smoothness / comfort).

BEV grid convention follows perception (rows <-> y, cols <-> x); the collision
sampler maps world (x, y) -> (col <- x, row <- y) via ``F.grid_sample``, keeping
the whole reward graph TensorRT/trace-friendly (no nonzero / dynamic shapes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OffRoadRewardConfig", "OffRoadReward"]


def _as_tensor(x, dtype=torch.float32, device=None) -> torch.Tensor:
    """Accept torch or numpy; convert to a float32 torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


@dataclass
class OffRoadRewardConfig:
    """Weights and physical thresholds for the off-road reward (SI units)."""

    dt: float = 0.1                              # trajectory sampling period [s]
    # component weights (the reward is a weighted SUM of the four terms)
    w_progress: float = 1.0
    w_collision: float = 5.0
    w_attitude_jitter: float = 0.3               # angular-rate (comfort)
    w_attitude_limit: float = 2.0                # rollover hard-limit
    w_jerk: float = 0.02                         # longitudinal jerk
    # thresholds
    collision_threshold: float = 0.5             # occupancy prob above this = "hit"
    pitch_limit: float = 0.40                    # rad (~23 deg) static rollover
    roll_limit: float = 0.40                     # rad (~23 deg)
    # BEV geometry (MUST match perception BEVFusionConfig for collision sampling)
    bev_x_range: Tuple[float, float] = (-12.5, 12.5)
    bev_y_range: Tuple[float, float] = (-12.5, 12.5)
    bev_resolution: float = 0.5


class OffRoadReward(nn.Module):
    """Differentiable multi-component off-road reward.

    Args to :meth:`forward`:
        trajectory : (B, N, 4)  [x, y, heading, velocity] (world frame).
        occupancy  : (B, H, W) or (B, 1, H, W) optional occupancy grid in BEV
                     convention (rows=y, cols=x); values in [0,1].
        attitude   : (B, N, 2) optional [pitch, roll] per waypoint (rad).
                     If None, the attitude component is skipped.
        dt         : override ``config.dt``.

    Returns:
        reward     : (B,) total reward (weighted sum of components).
        components : dict of (B,) per-component contributions for logging.
    """

    def __init__(self, config: Optional[OffRoadRewardConfig] = None):
        super().__init__()
        self.config = config or OffRoadRewardConfig()
        c = self.config
        x0, x1 = c.bev_x_range
        y0, y1 = c.bev_y_range
        # Precompute BEV grid dims (static) for the collision sampler.
        self.bev_w = max(1, int(round((x1 - x0) / c.bev_resolution)))  # cols <- x
        self.bev_h = max(1, int(round((y1 - y0) / c.bev_resolution)))  # rows <- y
        self.x0, self.y0 = float(x0), float(y0)

    def _sample_occupancy(
        self, xy: torch.Tensor, occupancy: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear occupancy lookup at world points (differentiable).

        Args:
            xy: (B, N, 2) world (x, y).
            occupancy: (B, H, W) or (B, 1, H, W), rows=y cols=x, vals [0,1].
        Returns:
            occ: (B, N) occupancy probability at each point (0=free).
        """
        B, N, _ = xy.shape
        if occupancy.dim() == 3:
            occupancy = occupancy.unsqueeze(1)            # (B,1,H,W)
        x = xy[..., 0]
        y = xy[..., 1]
        # world (x,y) -> BEV cell (col<-x, row<-y) -> normalized [-1, 1]
        col = (x - self.x0) / (self.config.bev_resolution * max(self.bev_w - 1, 1)) * 2.0 - 1.0
        row = (y - self.y0) / (self.config.bev_resolution * max(self.bev_h - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([col, row], dim=-1)             # (B, N, 2): (x=col, y=row)
        grid = grid[:, None, :, :]                         # (B, 1, N, 2)
        occ = F.grid_sample(occupancy, grid, mode="bilinear",
                            padding_mode="zeros",         # overridden to occupied below
                            align_corners=True)           # (B, 1, 1, N)
        occ = occ.reshape(B, N)
        outside = (col.abs() > 1.0) | (row.abs() > 1.0)
        return torch.where(outside, torch.ones_like(occ), occ)

    # ------------------------------------------------------------------
    def per_step(self, trajectory: torch.Tensor,
                 occupancy: Optional[torch.Tensor] = None,
                 attitude: Optional[torch.Tensor] = None,
                 dt: Optional[float] = None) -> torch.Tensor:
        """Per-waypoint reward (B, N); ``sum(-1)`` equals the episode reward.

        Components are credited to the waypoint they *end* at (displacement
        ending at step t is credited to t).
        """
        traj = _as_tensor(trajectory)
        if traj.dim() != 3 or traj.shape[-1] != 4:
            raise ValueError(f"trajectory must be (B,N,4), got {tuple(traj.shape)}")
        B, N, _ = traj.shape
        c = self.config
        dt = c.dt if dt is None else float(dt)
        r = torch.zeros(B, N, dtype=traj.dtype, device=traj.device)

        xy = traj[..., :2]
        heading = traj[..., 2]
        speed = traj[..., 3]

        # --- R1: progress (forward displacement along heading) -----------
        if N >= 2:
            dxy = xy[:, 1:] - xy[:, :-1]                 # (B, N-1, 2)
            fwd = torch.stack([torch.cos(heading[:, :-1]),
                               torch.sin(heading[:, :-1])], dim=-1)  # (B,N-1,2)
            progress = (dxy * fwd).sum(-1)              # (B, N-1)
            r[:, 1:] = r[:, 1:] + c.w_progress * progress

        # --- R2: collision (soft occupancy sampling) ----------------------
        if occupancy is not None:
            occ = self._sample_occupancy(xy, _as_tensor(occupancy))   # (B, N)
            r = r - c.w_collision * occ

        # --- R3: attitude jitter + rollover limit -------------------------
        if attitude is not None:
            att = _as_tensor(attitude)                   # (B, N, 2) [pitch, roll]
            limits = torch.tensor([c.pitch_limit, c.roll_limit],
                                  dtype=att.dtype, device=att.device)
            if N >= 2:
                rate = (att[:, 1:] - att[:, :-1]) / dt  # (B, N-1, 2)
                r[:, 1:] = r[:, 1:] - c.w_attitude_jitter * rate.abs().sum(-1)
            over = (att.abs() - limits).clamp(min=0.0).sum(-1)        # (B, N)
            r = r - c.w_attitude_limit * over

        # --- R4: jerk (second derivative of speed) ------------------------
        if N >= 3:
            accel = (speed[:, 1:] - speed[:, :-1]) / dt            # (B, N-1)
            jerk = (accel[:, 1:] - accel[:, :-1]) / dt             # (B, N-2)
            r[:, 2:] = r[:, 2:] - c.w_jerk * (jerk ** 2)

        return r

    # ------------------------------------------------------------------
    def forward(self, trajectory: torch.Tensor,
                occupancy: Optional[torch.Tensor] = None,
                attitude: Optional[torch.Tensor] = None,
                dt: Optional[float] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Total reward (B,) and per-component breakdown (each (B,))."""
        traj = _as_tensor(trajectory)
        B, N, _ = traj.shape
        c = self.config
        dt = c.dt if dt is None else float(dt)
        components: Dict[str, torch.Tensor] = {}

        # R1 progress
        prog = torch.zeros(B, dtype=traj.dtype, device=traj.device)
        if N >= 2:
            dxy = traj[:, 1:, :2] - traj[:, :-1, :2]
            fwd = torch.stack([torch.cos(traj[:, :-1, 2]),
                               torch.sin(traj[:, :-1, 2])], dim=-1)
            prog = c.w_progress * (dxy * fwd).sum(-1).sum(-1)
        components["progress"] = prog

        # R2 collision (total occupancy exposure over the trajectory; uses SUM
        # to match per_step so the reward_head learns the SAME quantity that is
        # bootstrapped in RL imagination rollouts -- max would be inconsistent)
        coll = torch.zeros(B, dtype=traj.dtype, device=traj.device)
        if occupancy is not None:
            occ = self._sample_occupancy(traj[..., :2], _as_tensor(occupancy))
            coll = -c.w_collision * occ.sum(-1)
        components["collision"] = coll

        # R3 attitude jitter + rollover limit
        att_total = torch.zeros(B, dtype=traj.dtype, device=traj.device)
        if attitude is not None:
            att = _as_tensor(attitude)
            limits = torch.tensor([c.pitch_limit, c.roll_limit],
                                  dtype=att.dtype, device=att.device)
            if N >= 2:
                rate = (att[:, 1:] - att[:, :-1]) / dt
                att_total = att_total - c.w_attitude_jitter * rate.abs().sum(-1).sum(-1)
            over = (att.abs() - limits).clamp(min=0.0).sum(-1).sum(-1)
            att_total = att_total - c.w_attitude_limit * over
        components["attitude"] = att_total

        # R4 jerk (second derivative of speed)
        jerk_t = torch.zeros(B, dtype=traj.dtype, device=traj.device)
        if N >= 3:
            v = traj[..., 3]
            accel = (v[:, 1:] - v[:, :-1]) / dt
            jerk = (accel[:, 1:] - accel[:, :-1]) / dt
            jerk_t = -c.w_jerk * (jerk ** 2).sum(-1)
        components["jerk"] = jerk_t

        reward = prog + coll + att_total + jerk_t
        components["total"] = reward
        return reward, components
