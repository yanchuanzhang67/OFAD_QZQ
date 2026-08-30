"""Unit tests for SafetyFilter (SDD: over-limit steering truncation & re-projection)."""
import numpy as np
import pytest

from safety.kinematic_filter import SafetyFilter, SafetyFilterConfig
from utils.types import OccupancyGrid, Trajectory, VehicleState, Waypoint

pytestmark = pytest.mark.unit


def _straight_traj(n=10, speed=2.0, steering=0.0, dt=0.1):
    return Trajectory([
        Waypoint(x=(i + 1) * speed * dt, y=0.0, yaw=0.0, speed=speed,
                 steering=steering, t=(i + 1) * dt)
        for i in range(n)
    ])


def test_overlimit_steering_truncated():
    sf = SafetyFilter(SafetyFilterConfig(max_steering=0.5))
    s0 = VehicleState(speed=2.0)
    traj = _straight_traj(steering=1.0)  # over limit
    out = sf.filter(traj, s0)
    assert out.length > 0
    assert all(abs(w.steering) <= 0.5 + 1e-6 for w in out.waypoints)


def test_filter_output_kinematically_feasible():
    sf = SafetyFilter()
    s0 = VehicleState(speed=2.0)
    traj = _straight_traj(steering=0.2)
    out = sf.filter(traj, s0)
    assert out.length > 0
    assert sf.is_kinematically_feasible(out, s0)


def test_collision_hard_truncation():
    sf = SafetyFilter(SafetyFilterConfig(max_steering=0.5, inflation=0.0))
    s0 = VehicleState(speed=2.0)
    traj = _straight_traj(n=15)
    grid = np.zeros((200, 200), dtype=np.float32)
    occ = OccupancyGrid(data=grid, resolution=0.5, origin=(-50.0, -50.0, 0.0))
    r, c = occ.world_to_index(1.0, 0.0)
    occ.data[r, c] = 1.0
    out = sf.filter(traj, s0, occupancy=occ)
    assert out.length > 0
    assert all(w.x < 1.0 - 1e-3 for w in out.waypoints)


def test_safe_trajectory_passes_through():
    sf = SafetyFilter()
    s0 = VehicleState(speed=2.0)
    traj = _straight_traj(steering=0.1)
    out = sf.filter(traj, s0)
    assert out.length == traj.length
