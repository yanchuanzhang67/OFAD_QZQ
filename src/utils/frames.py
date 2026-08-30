"""Canonical ORAD coordinate frames and explicit ego/world transforms.

Ego axes are fixed system-wide: x forward, y left, z up. Angles are radians,
distance is metres, speed is metres/second.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from utils.types import Trajectory, VehicleState, Waypoint


class CoordinateFrame(str, Enum):
    EGO = "ego"
    WORLD = "world"


def _xy(points) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"points must be (N,2), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("points must contain only finite values")
    return value


def ego_to_world_points(points, state: VehicleState) -> np.ndarray:
    points = _xy(points)
    c, s = np.cos(state.yaw), np.sin(state.yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    translation = np.array([state.x, state.y], dtype=np.float32)
    return np.ascontiguousarray(points @ rotation.T + translation)


def world_to_ego_points(points, state: VehicleState) -> np.ndarray:
    points = _xy(points)
    c, s = np.cos(state.yaw), np.sin(state.yaw)
    inverse = np.array([[c, s], [-s, c]], dtype=np.float32)
    translation = np.array([state.x, state.y], dtype=np.float32)
    return np.ascontiguousarray((points - translation) @ inverse.T)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def transform_trajectory(trajectory: Trajectory, state: VehicleState,
                         target: CoordinateFrame) -> Trajectory:
    """Transform a trajectory between canonical ego and world frames."""
    source = CoordinateFrame(trajectory.frame)
    target = CoordinateFrame(target)
    if source is target:
        return Trajectory(list(trajectory.waypoints), frame=target.value,
                          timestamp=trajectory.timestamp)
    array = trajectory.to_array()
    if array.shape[0] == 0:
        return Trajectory([], frame=target.value, timestamp=trajectory.timestamp)
    xy = (ego_to_world_points(array[:, :2], state)
          if source is CoordinateFrame.EGO
          else world_to_ego_points(array[:, :2], state))
    yaw_sign = 1.0 if target is CoordinateFrame.WORLD else -1.0
    waypoints = [
        Waypoint(x=float(xy[i, 0]), y=float(xy[i, 1]),
                 yaw=_wrap(float(row[2]) + yaw_sign * state.yaw),
                 speed=float(row[3]), steering=float(row[4]), t=float(row[5]))
        for i, row in enumerate(array)
    ]
    return Trajectory(waypoints, frame=target.value,
                      timestamp=trajectory.timestamp)
