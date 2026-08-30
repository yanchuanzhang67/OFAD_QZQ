"""Kinematic bicycle model (rear-axle reference) for the safety layer.

Pure-numpy forward simulation so that feasibility checks and trajectory
re-projection run without any ML dependency. The same equations are mirrored
in the C++ ``safety_filter_node``.

State equations (rear-axle bicycle model)::

    x'   = x   + v * cos(yaw) * dt
    y'   = y   + v * sin(yaw) * dt
    yaw' = yaw + (v / L) * tan(steer) * dt
    v'   = clip(v + accel * dt, 0, max_speed)
"""
from __future__ import annotations

import numpy as np

from utils.types import Trajectory, VehicleState, Waypoint

__all__ = ["KinematicBicycleModel"]


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class KinematicBicycleModel:
    """Differentiable-free, numpy-only kinematic bicycle model."""

    def __init__(
        self,
        wheelbase: float = 2.5,
        max_steer: float = 0.5,
        max_steer_rate: float = 0.5,
        max_accel: float = 3.0,
        max_decel: float = -5.0,
        max_speed: float = 20.0,
    ):
        self.wheelbase = float(wheelbase)
        self.max_steer = float(max_steer)
        self.max_steer_rate = float(max_steer_rate)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.max_speed = float(max_speed)

    def clip_controls(self, steering, accel, prev_steering=None, dt=None):
        """Clip steering/acceleration to physical limits (rate-limited if
        ``prev_steering`` and ``dt`` are given)."""
        steer = float(np.clip(steering, -self.max_steer, self.max_steer))
        accel = float(np.clip(accel, self.max_decel, self.max_accel))
        if prev_steering is not None and dt is not None:
            dmax = self.max_steer_rate * dt
            steer = float(np.clip(steer, prev_steering - dmax, prev_steering + dmax))
        return steer, accel

    def step(self, state: VehicleState, steering: float, accel: float,
             dt: float = 0.1) -> VehicleState:
        """Single Euler step of the bicycle model."""
        steer = float(np.clip(steering, -self.max_steer, self.max_steer))
        accel = float(np.clip(accel, self.max_decel, self.max_accel))
        v = float(np.clip(state.speed + accel * dt, 0.0, self.max_speed))
        x = state.x + v * np.cos(state.yaw) * dt
        y = state.y + v * np.sin(state.yaw) * dt
        yaw = state.yaw + (v / self.wheelbase) * np.tan(steer) * dt
        yaw_rate = (v / self.wheelbase) * np.tan(steer)
        return VehicleState(
            x=x, y=y, yaw=yaw, speed=v, steering=steer,
            pitch=state.pitch, roll=state.roll,
            vx=v * np.cos(yaw), vy=v * np.sin(yaw),
            yaw_rate=yaw_rate, accel_z=state.accel_z,
        )

    def rollout(self, state: VehicleState, controls, dt: float = 0.1) -> Trajectory:
        """Roll out ``controls`` ``(N, 2)`` = [steer, accel] from ``state``."""
        controls = np.asarray(controls, dtype=np.float32).reshape(-1, 2)
        s = state
        wps: list[Waypoint] = []
        t = dt
        for i in range(controls.shape[0]):
            steer, accel = float(controls[i, 0]), float(controls[i, 1])
            s = self.step(s, steer, accel, dt=dt)
            wps.append(Waypoint(x=s.x, y=s.y, yaw=s.yaw, speed=s.speed,
                                steering=s.steering, t=t))
            t += dt
        return Trajectory(wps)

    def is_feasible(self, trajectory: Trajectory, state: VehicleState,
                    dt: float = 0.1, tol: float = 1e-2) -> bool:
        """True iff ``trajectory`` is reproducible by this model from
        ``state`` within ``tol`` (controls within limits, dynamics consistent).
        """
        wp = trajectory.waypoints
        if not wp:
            return True
        steers = np.array([w.steering for w in wp], dtype=np.float32)
        if np.any(np.abs(steers) > self.max_steer + 1e-6):
            return False
        speeds = np.concatenate([[state.speed], [w.speed for w in wp]]).astype(np.float32)
        accels = np.diff(speeds) / dt
        if np.any(accels > self.max_accel + 1e-3) or np.any(accels < self.max_decel - 1e-3):
            return False
        controls = np.stack([steers, accels], axis=1).astype(np.float32)
        recon = self.rollout(state, controls, dt=dt)
        if recon.length != len(wp):
            return False
        for a, b in zip(recon.waypoints, wp):
            if (abs(a.x - b.x) > tol or abs(a.y - b.y) > tol
                    or abs(_wrap_angle(a.yaw - b.yaw)) > tol):
                return False
        return True
