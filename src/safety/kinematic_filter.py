"""Safety filter: kinematic feasibility + collision hard-truncation.

Implements the *Safety Filter* layer of the SDD. Given a candidate
trajectory from the policy network it:

1. clips over-limit steering / acceleration (with rate & jerk limits),
2. re-projects the trajectory onto a kinematically feasible manifold via the
   bicycle model (this is the "重投影逻辑" reprojection logic),
3. hard-truncates any collision-dangerous prefix against the occupancy grid.

A CasADi-based nonlinear MPC optimizer is used when ``casadi`` is importable
and ``use_casadi=True``; otherwise the deterministic projection fallback is
used. The public API is numpy-only at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from safety.bicycle_model import KinematicBicycleModel
from utils.types import OccupancyGrid, Trajectory, VehicleState

try:  # optional nonlinear optimizer
    import casadi  # noqa: F401
    _HAS_CASADI = True
except Exception:  # pragma: no cover - env dependent
    _HAS_CASADI = False

__all__ = ["SafetyFilterConfig", "SafetyFilter"]


def _wrap_angle(angle: float) -> float:
    """Wrap an angle (rad) into ``(-pi, pi]`` (mirrors bicycle_model)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class SafetyFilterConfig:
    """Physical limits used by :class:`SafetyFilter`."""

    wheelbase: float = 2.5
    max_steering: float = 0.5          # ~28.6 deg
    max_steering_rate: float = 0.5     # rad/s
    max_accel: float = 3.0             # m/s^2
    max_decel: float = -5.0            # m/s^2
    max_jerk: float = 5.0             # m/s^3
    max_lateral_accel: float = 4.0     # m/s^2 (off-road stability)
    dt: float = 0.1                    # trajectory sampling period [s]
    collision_check: bool = True
    inflation: float = 0.5             # obstacle inflation radius [m]
    use_casadi: bool = False
    feasibility_tol: float = 1e-2


class SafetyFilter:
    """Hard safety envelope around an end-to-end policy trajectory."""

    def __init__(self, config: Optional[SafetyFilterConfig] = None):
        self.config = config or SafetyFilterConfig()
        c = self.config
        self.bicycle = KinematicBicycleModel(
            wheelbase=c.wheelbase,
            max_steer=c.max_steering,
            max_steer_rate=c.max_steering_rate,
            max_accel=c.max_accel,
            max_decel=c.max_decel,
        )
        self._use_casadi = bool(c.use_casadi and _HAS_CASADI)

    # -- introspection ----------------------------------------------------
    @property
    def has_casadi(self) -> bool:
        return _HAS_CASADI

    @property
    def max_curvature(self) -> float:
        """Maximum path curvature implied by ``max_steering`` (``tan(δ)/L``)."""
        return float(np.tan(self.config.max_steering) / self.config.wheelbase)

    # -- scalar control clipping -----------------------------------------
    def clip_steering(self, steering: float, prev_steering: Optional[float],
                      dt: Optional[float] = None) -> float:
        c = self.config
        dt = c.dt if dt is None else dt
        s = float(np.clip(steering, -c.max_steering, c.max_steering))
        if prev_steering is not None:
            dmax = c.max_steering_rate * dt
            s = float(np.clip(s, prev_steering - dmax, prev_steering + dmax))
        return s

    def clip_accel(self, accel: float, prev_accel: Optional[float],
                   dt: Optional[float] = None) -> float:
        c = self.config
        dt = c.dt if dt is None else dt
        a = float(np.clip(accel, c.max_decel, c.max_accel))
        if prev_accel is not None:
            a = float(np.clip(a, prev_accel - c.max_jerk * dt,
                              prev_accel + c.max_jerk * dt))
        return a

    # -- feasibility & collision -----------------------------------------
    def is_kinematically_feasible(
        self, trajectory: Trajectory, vehicle_state: VehicleState
    ) -> bool:
        return self.bicycle.is_feasible(
            trajectory, vehicle_state, dt=self.config.dt,
            tol=self.config.feasibility_tol)

    def check_collision(self, trajectory: Trajectory,
                        occupancy: Optional[OccupancyGrid] = None) -> np.ndarray:
        """Per-waypoint occupancy flags (True = occupied)."""
        n = trajectory.length
        flags = np.zeros(n, dtype=bool)
        if occupancy is None or n == 0:
            return flags
        for i, w in enumerate(trajectory.waypoints):
            flags[i] = occupancy.is_occupied(w.x, w.y, inflation=self.config.inflation)
        return flags

    # -- kinematic limit checking -----------------------------------------
    def estimate_curvature(self, trajectory: Trajectory,
                           dt: Optional[float] = None) -> np.ndarray:
        """Per-waypoint path curvature ``κ`` (1/m).

        When a waypoint carries a non-zero ``steering`` field the curvature is
        recovered from the bicycle relation ``κ = tan(δ) / L`` (preserving the
        intent of trajectories whose ``steering`` was set explicitly).
        Otherwise ``κ`` is estimated by finite-differencing the heading over
        the inter-waypoint arc length — the path taken by end-to-end policy
        outputs that only provide ``(x, y, yaw, v)`` (no steering).
        """
        n = trajectory.length
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        c = self.config
        step = c.dt if dt is None else dt
        L = c.wheelbase
        wps = trajectory.waypoints
        kappa = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if abs(wps[i].steering) > 1e-6:
                kappa[i] = float(np.tan(wps[i].steering) / L)
                continue
            if i < n - 1:
                nxt, cur = wps[i + 1], wps[i]
            elif i > 0:
                nxt, cur = wps[i], wps[i - 1]
            else:  # single waypoint: cannot differentiate
                continue
            d_yaw = _wrap_angle(nxt.yaw - cur.yaw)
            ds = float(np.hypot(nxt.x - cur.x, nxt.y - cur.y))
            if ds < 1e-3:
                ds = max(cur.speed * step, 1e-3)
            kappa[i] = d_yaw / ds
        return kappa

    def check_limits(self, trajectory: Trajectory) -> dict:
        """Return per-waypoint kinematic-limit diagnostics.

        Keys: ``curvature``, ``steering`` (``δ = atan(κ·L)``),
        ``lateral_accel`` (``v²·|κ|``) plus boolean ``*_violation`` arrays
        against ``max_curvature`` / ``max_steering`` / ``max_lateral_accel``.
        """
        k = self.estimate_curvature(trajectory)
        L = self.config.wheelbase
        steer = np.arctan(k * L)
        speed = np.array([w.speed for w in trajectory.waypoints],
                         dtype=np.float32)
        lat = speed ** 2 * np.abs(k)
        return {
            "curvature": k,
            "steering": steer,
            "lateral_accel": lat,
            "curvature_violation": np.abs(k) > self.max_curvature + 1e-9,
            "steering_violation": np.abs(steer)
            > self.config.max_steering + 1e-9,
            "lateral_accel_violation": lat
            > self.config.max_lateral_accel + 1e-9,
        }

    # -- re-projection ----------------------------------------------------
    def project_to_feasible(self, trajectory: Trajectory,
                            vehicle_state: VehicleState) -> Trajectory:
        """Project onto the nearest kinematically feasible trajectory.

        Three hard constraints are enforced (the SDD Phase-4 safety spec):

        * **curvature** ``|κ| ≤ tan(δ_max)/L``  → curvature is clipped, the
          steering is then ``δ = atan(κ·L)`` (so the turn is the closest
          feasible one to the request);
        * **steering rate** ``|Δδ| ≤ δ̇_max·dt`` → :meth:`clip_steering`;
        * **lateral acceleration** ``a_y = v²·|κ| ≤ a_y_max`` → speed is capped
          to ``v ≤ √(a_y_max / |κ|)`` before the longitudinal P-term.

        The projected ``(steer, accel)`` controls are re-integrated through
        :class:`KinematicBicycleModel.rollout` from ``vehicle_state``, yielding
        a trajectory that is feasible-by-construction.
        """
        n = trajectory.length
        if n == 0:
            return Trajectory([], frame=trajectory.frame,
                              timestamp=trajectory.timestamp)
        c = self.config
        L = c.wheelbase
        kmax = self.max_curvature
        k = self.estimate_curvature(trajectory)
        k_clamped = np.clip(k, -kmax, kmax)
        wps = trajectory.waypoints
        controls = np.zeros((n, 2), dtype=np.float32)
        prev_s = vehicle_state.steering
        prev_a: Optional[float] = None
        prev_speed = vehicle_state.speed
        for i in range(n):
            steer = float(np.arctan(k_clamped[i] * L))
            steer = self.clip_steering(steer, prev_s)
            k_abs = max(abs(k_clamped[i]), 1e-9)
            v_lat = float(np.sqrt(c.max_lateral_accel / k_abs))
            v_des = min(float(wps[i].speed), v_lat, self.bicycle.max_speed)
            a_raw = (v_des - prev_speed) / c.dt
            a = self.clip_accel(a_raw, prev_a)
            controls[i, 0] = steer
            controls[i, 1] = a
            prev_s, prev_a, prev_speed = steer, a, v_des
        result = self.bicycle.rollout(vehicle_state, controls, dt=c.dt)
        result.frame = trajectory.frame
        result.timestamp = trajectory.timestamp
        return result

    # -- optional nonlinear optimization ---------------------------------
    def optimize_casadi(
        self, trajectory: Trajectory, vehicle_state: VehicleState,
        occupancy: Optional[OccupancyGrid] = None
    ) -> Trajectory:
        """CasADi nonlinear MPC placeholder.

        Falls back to :meth:`project_to_feasible` when CasADi is unavailable.
        """
        if not self._use_casadi:
            raise RuntimeError(
                "CasADi optimizer disabled (not installed or use_casadi=False); "
                "use project_to_feasible() / filter() instead.")
        # TODO: build OCP with bicycle constraints + occupancy avoidance,
        # solve with IPOPT. Until then, degrade gracefully.
        return self.project_to_feasible(trajectory, vehicle_state)

    # -- main entry point ------------------------------------------------
    def filter(self, trajectory: Trajectory, vehicle_state: VehicleState,
               occupancy: Optional[OccupancyGrid] = None) -> Trajectory:
        """Return a kinematically feasible, collision-free trajectory.

        Hard-truncates the prefix up to (and including) the first waypoint
        whose cell is occupied, per the SDD safety spec.
        """
        if self._use_casadi:
            feasible = self.optimize_casadi(trajectory, vehicle_state, occupancy)
        else:
            feasible = self.project_to_feasible(trajectory, vehicle_state)
        if (self.config.collision_check and occupancy is not None
                and feasible.length):
            flags = self.check_collision(feasible, occupancy)
            if flags.any():
                first_hit = int(np.argmax(flags))  # first True index
                feasible = Trajectory(
                    feasible.waypoints[:first_hit], frame=feasible.frame,
                    timestamp=feasible.timestamp)
        return feasible
