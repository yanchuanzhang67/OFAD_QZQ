"""Deterministic, testable domain-randomization sampling and LiDAR augmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DomainRandomizationConfig:
    tire_friction: Tuple[float, float] = (0.4, 1.4)
    vehicle_mass: Tuple[float, float] = (1500.0, 1900.0)
    lidar_sigma: Tuple[float, float] = (0.0, 0.1)
    lidar_dropout: Tuple[float, float] = (0.0, 0.05)
    suspension_damping: Tuple[float, float] = (400.0, 1200.0)

    def __post_init__(self):
        for name, bounds in self.__dict__.items():
            if (len(bounds) != 2 or not np.isfinite(bounds).all()
                    or bounds[0] > bounds[1]):
                raise ValueError(f"invalid {name} range: {bounds}")


def sample_domain(config: DomainRandomizationConfig, rng=None) -> dict:
    """Sample one reproducible episode manifest from configured ranges."""
    rng = np.random.default_rng() if rng is None else rng
    return {name: float(rng.uniform(*bounds))
            for name, bounds in config.__dict__.items()}


def augment_lidar(points, sigma: float, dropout: float, rng=None):
    """Add xyz Gaussian noise and point dropout without mutating the input."""
    if sigma < 0 or not 0.0 <= dropout <= 1.0:
        raise ValueError("sigma must be >= 0 and dropout must be in [0,1]")
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"points must be (N,3) or (N,4), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite values")
    rng = np.random.default_rng() if rng is None else rng
    out = pts.copy()
    out[:, :3] += rng.normal(0.0, sigma, out[:, :3].shape).astype(np.float32)
    keep = rng.random(out.shape[0]) >= dropout
    return np.ascontiguousarray(out[keep])
