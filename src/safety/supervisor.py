"""Fail-safe policy-trajectory supervisor and degradation state machine."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from utils.frames import CoordinateFrame
from utils.types import Trajectory


class SafetyMode(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class SafetySupervisorConfig:
    expected_frame: str = CoordinateFrame.EGO.value
    min_waypoints: int = 2
    max_speed: float = 30.0
    max_sensor_age: float = 0.2
    max_sensor_skew: float = 0.05
    max_model_latency: float = 0.1


@dataclass(frozen=True)
class SafetyDecision:
    mode: SafetyMode
    reason: str


class SafetySupervisor:
    """Reject invalid policy output before it reaches kinematics/control."""

    def __init__(self, config: Optional[SafetySupervisorConfig] = None):
        self.config = config or SafetySupervisorConfig()

    def evaluate(self, trajectory: Optional[Trajectory], sensor_age: float = 0.0,
                 sensor_skew: float = 0.0,
                 model_latency: float = 0.0,
                 safety_intervened: bool = False) -> SafetyDecision:
        c = self.config
        timing = np.asarray(
            [sensor_age, sensor_skew, model_latency], dtype=np.float64)
        if not np.isfinite(timing).all() or np.any(timing < 0.0):
            return SafetyDecision(SafetyMode.EMERGENCY_STOP, "invalid_timing")
        if sensor_age > c.max_sensor_age:
            return SafetyDecision(SafetyMode.EMERGENCY_STOP, "sensor_timeout")
        if sensor_skew > c.max_sensor_skew:
            return SafetyDecision(SafetyMode.EMERGENCY_STOP, "sensor_skew")
        if model_latency > c.max_model_latency:
            return SafetyDecision(SafetyMode.EMERGENCY_STOP, "model_timeout")
        if trajectory is None:
            return SafetyDecision(SafetyMode.EMERGENCY_STOP, "missing_trajectory")
        if trajectory.length < c.min_waypoints:
            return SafetyDecision(SafetyMode.EMERGENCY_STOP, "too_few_waypoints")
        if trajectory.frame != c.expected_frame:
            return SafetyDecision(SafetyMode.EMERGENCY_STOP, "frame_mismatch")
        values = trajectory.to_array()
        if not np.isfinite(values).all():
            return SafetyDecision(SafetyMode.EMERGENCY_STOP,
                                  "non_finite_trajectory")
        speeds = values[:, 3]
        if np.any(speeds < 0.0) or np.any(speeds > c.max_speed):
            return SafetyDecision(SafetyMode.EMERGENCY_STOP, "invalid_speed")
        if safety_intervened:
            return SafetyDecision(SafetyMode.DEGRADED, "safety_intervention")
        return SafetyDecision(SafetyMode.NORMAL, "ok")

    def emergency_trajectory(self, timestamp: float = 0.0) -> Trajectory:
        return Trajectory([], frame=self.config.expected_frame,
                          timestamp=timestamp)
