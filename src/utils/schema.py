"""Validated cross-module schemas for synchronized ORAD observations."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Tuple

import numpy as np

from utils.frames import CoordinateFrame
from utils.types import VehicleState


@dataclass
class AffordanceOutput:
    """Explicit interface between generic BEV, world model and policy."""

    feature: Any
    traversability: Any
    roughness: Any


@dataclass
class Observation:
    """One synchronized model input at a simulator/control timestamp."""

    timestamp: float
    simulator_frame: int
    images: list
    point_cloud: Any
    imu_history: Any
    ego_state: VehicleState
    camera_frames: Tuple[int, ...]
    lidar_frame: int
    imu_frames: Tuple[int, ...] = ()
    frame: str = CoordinateFrame.EGO.value
    sensor_timestamps: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame != CoordinateFrame.EGO.value:
            raise ValueError("Observation frame must be 'ego'")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not self.images or len(self.images) != len(self.camera_frames):
            raise ValueError("images and camera_frames must be non-empty and aligned")
        try:
            valid_images = all(
                np.asarray(image).size > 0 and np.isfinite(image).all()
                for image in self.images
            )
        except (TypeError, ValueError):
            valid_images = False
        if not valid_images:
            raise ValueError("images must be non-empty finite arrays")
        points = np.asarray(self.point_cloud, dtype=np.float32)
        imu = np.asarray(self.imu_history, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError(f"point_cloud must be (N,3/4), got {points.shape}")
        if imu.ndim != 2 or imu.shape[1] != 6:
            raise ValueError(f"imu_history must be (T,6), got {imu.shape}")
        if not np.isfinite(points).all() or not np.isfinite(imu).all():
            raise ValueError("observation arrays must contain only finite values")
        sensor_frames = tuple(self.camera_frames) + (self.lidar_frame,)
        if any(frame != self.simulator_frame for frame in sensor_frames):
            raise ValueError("camera and lidar frames must be synchronized")
        if self.imu_frames and len(self.imu_frames) > imu.shape[0]:
            raise ValueError("imu_frames cannot exceed imu history length")
        if (self.sensor_timestamps
                and not np.isfinite(list(self.sensor_timestamps.values())).all()):
            raise ValueError("sensor timestamps must be finite")
        self.point_cloud = points
        self.imu_history = imu

    @property
    def max_sensor_skew_frames(self) -> int:
        frames = tuple(self.camera_frames) + (self.lidar_frame,)
        return max(frames) - min(frames)

    @property
    def max_sensor_skew_seconds(self) -> float:
        if not self.sensor_timestamps:
            return 0.0
        values = tuple(float(v) for v in self.sensor_timestamps.values())
        return max(values) - min(values)
