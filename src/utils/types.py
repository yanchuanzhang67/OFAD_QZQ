"""Shared, framework-agnostic data structures for the ORAD system.

These dataclasses are intentionally numpy-backed so that the *same* structures
can be (de)serialized between the Python perception/policy stack and the C++
safety/control runtime. Torch tensors are accepted via the ``Any``-typed
fields but are never *required* here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = [
    "VehicleState",
    "Waypoint",
    "Trajectory",
    "OccupancyGrid",
    "BEVFeature",
    "SensorPacket",
]


@dataclass
class VehicleState:
    """Ego-vehicle kinematic + attitude state.

    ``x`` / ``y`` / ``yaw`` describe the planar pose in the *world* frame.
    Body rates (``vx``, ``vy``, ``yaw_rate``) and attitude (``pitch``,
    ``roll``) are in the vehicle body frame. ``accel_z`` is the vertical
    acceleration consumed by the off-road stability reward term.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    steering: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    accel_z: float = 0.0

    _FIELDS = (
        "x", "y", "yaw", "speed", "steering",
        "pitch", "roll", "vx", "vy", "yaw_rate", "accel_z",
    )

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in self._FIELDS], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VehicleState":
        arr = np.asarray(arr, dtype=np.float32).ravel()
        return cls(**{k: float(arr[i]) for i, k in enumerate(cls._FIELDS[: len(arr)])})


@dataclass
class Waypoint:
    """A single point on a candidate trajectory."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    steering: float = 0.0
    t: float = 0.0


@dataclass
class Trajectory:
    """Ordered sequence of :class:`Waypoint` (candidate path from policy)."""

    waypoints: list[Waypoint] = field(default_factory=list)
    frame: str = "world"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.frame not in ("ego", "world"):
            raise ValueError("trajectory frame must be 'ego' or 'world'")
        if not np.isfinite(self.timestamp):
            raise ValueError("trajectory timestamp must be finite")

    @property
    def length(self) -> int:
        return len(self.waypoints)

    @property
    def duration(self) -> float:
        return (self.waypoints[-1].t - self.waypoints[0].t) if self.waypoints else 0.0

    def to_array(self) -> np.ndarray:
        if not self.waypoints:
            return np.zeros((0, 6), dtype=np.float32)
        return np.array(
            [[w.x, w.y, w.yaw, w.speed, w.steering, w.t] for w in self.waypoints],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray, frame: str = "world",
                   timestamp: float = 0.0) -> "Trajectory":
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 6)
        return cls(waypoints=[Waypoint(*map(float, row)) for row in arr],
                   frame=frame, timestamp=timestamp)


@dataclass
class OccupancyGrid:
    """2D probabilistic occupancy grid in the BEV plane (1.0 = occupied)."""

    data: np.ndarray
    resolution: float = 0.5
    origin: tuple = (0.0, 0.0, 0.0)  # (x, y, yaw) world pose of cell (row=0, col=0)
    free_threshold: float = 0.3
    occupied_threshold: float = 0.7

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError(f"occupancy data must be 2D, got {self.data.shape}")
        if not np.isfinite(self.data).all():
            raise ValueError("occupancy data must contain only finite values")
        if not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("resolution must be finite and > 0")
        if len(self.origin) != 3 or not np.isfinite(self.origin).all():
            raise ValueError("origin must be finite (x, y, yaw)")

    @property
    def shape(self):
        return self.data.shape

    def world_to_index(self, x: float, y: float) -> tuple[int, int]:
        dx, dy = float(x) - self.origin[0], float(y) - self.origin[1]
        yaw = float(self.origin[2])
        ca, sa = np.cos(yaw), np.sin(yaw)
        local_x = ca * dx + sa * dy
        local_y = -sa * dx + ca * dy
        c = int(np.floor(local_x / self.resolution))
        r = int(np.floor(local_y / self.resolution))
        return r, c

    def is_occupied(self, x: float, y: float, inflation: float = 0.0) -> bool:
        """Conservative occupancy test: outside-grid ⇒ occupied."""
        r, c = self.world_to_index(x, y)
        H, W = self.data.shape
        rr = int(round(inflation / self.resolution))
        r0, r1 = max(0, r - rr), min(H, r + rr + 1)
        c0, c1 = max(0, c - rr), min(W, c + rr + 1)
        if r0 >= r1 or c0 >= c1:
            return True
        return bool(self.data[r0:r1, c0:c1].max() >= self.occupied_threshold)


@dataclass
class BEVFeature:
    """Output of the perception / BEV-fusion stage."""

    bev: Any                       # fused BEV tensor (B, C, H, W) (torch.Tensor)
    occupancy: Optional[Any] = None       # batched tensor in perception; grid in safety
    imu_state: Optional[Any] = None        # (B, T, F) IMU history features
    metadata: dict = field(default_factory=dict)


@dataclass
class SensorPacket:
    """A time-synchronized multi-modal sensor sample."""

    images: list                  # list of (H, W, 3) uint8 arrays / tensors
    point_cloud: Any              # (N, 3) or (N, 4) float32
    imu: Any                      # (T, 6) [ax, ay, az, gx, gy, gz]
    timestamps: dict = field(default_factory=dict)
