"""Phase 4 tests: curvature / lateral-acceleration hard limits.

These complement the Phase-1 suite (steer clip + collision). The filter now also
enforces (i) path curvature ``|κ| ≤ tan(δ_max)/L``, (ii) the steering-angle
limit and (iii) lateral acceleration ``a_y = v²·|κ| ≤ a_y_max`` by projecting
onto the nearest feasible trajectory through bicycle-model re-integration.
"""
import numpy as np
import pytest

from safety.bicycle_model import KinematicBicycleModel
from safety.kinematic_filter import SafetyFilter, SafetyFilterConfig
from utils.types import OccupancyGrid, Trajectory, VehicleState, Waypoint

pytestmark = pytest.mark.unit


def _straight_traj(n=10, speed=2.0, steering=0.0, dt=0.1):
    return Trajectory([
        Waypoint(x=(i + 1) * speed * dt, y=0.0, yaw=0.0, speed=speed,
                 steering=steering, t=(i + 1) * dt)
        for i in range(n)
    ])


def _arc_via_model(n=8, steer=0.0, speed=2.0, dt=0.1):
    m = KinematicBicycleModel()
    s0 = VehicleState(speed=speed)
    controls = np.tile(np.array([steer, 0.0], dtype=np.float32), (n, 1))
    return m.rollout(s0, controls, dt=dt), s0


def _heading_arc(n=8, d_yaw=0.02, speed=2.0, dt=0.1):
    """Arc with steering=0 but increasing yaw (end-to-end policy style)."""
    wps = []
    x = y = yaw = 0.0
    ds = speed * dt
    for i in range(n):
        wps.append(Waypoint(x=x, y=y, yaw=yaw, speed=speed, t=(i + 1) * dt))
        yaw += d_yaw
        x += ds * np.cos(yaw)
        y += ds * np.sin(yaw)
    return Trajectory(wps)


def _fast_arc(n=12, steer_equiv=0.4, speed=8.0, dt=0.1):
    """High-speed arc, steering=0 (curvature inferred from heading)."""
    L = 2.5
    k = float(np.tan(steer_equiv) / L)
    ds = speed * dt
    wps = []
    x = y = yaw = 0.0
    for i in range(n):
        wps.append(Waypoint(x=x, y=y, yaw=yaw, speed=speed, t=(i + 1) * dt))
        yaw += k * ds
        x += ds * np.cos(yaw)
        y += ds * np.sin(yaw)
    return Trajectory(wps)


def test_max_curvature_derived():
    sf = SafetyFilter(SafetyFilterConfig(max_steering=0.5, wheelbase=2.5))
    assert sf.max_curvature == pytest.approx(np.tan(0.5) / 2.5)


def test_estimate_curvature_from_steering():
    sf = SafetyFilter()
    traj, _ = _arc_via_model(n=8, steer=0.2, speed=2.0)
    k = sf.estimate_curvature(traj)
    expected = float(np.tan(0.2) / 2.5)
    assert k.shape == (8,)
    assert np.allclose(k, expected, atol=1e-3)


def test_estimate_curvature_from_heading():
    sf = SafetyFilter()
    traj = _heading_arc(n=8, d_yaw=0.02, speed=2.0)  # kappa ~= 0.02/0.2 = 0.1
    k = sf.estimate_curvature(traj)
    assert k.shape == (8,)
    assert np.allclose(k[1:-1], 0.1, atol=2e-2)
    assert traj.waypoints[0].steering == 0.0  # indeed policy-style (no steer)


def test_check_limits_flags_violations():
    sf = SafetyFilter(SafetyFilterConfig(max_steering=0.3, max_lateral_accel=2.0))
    hard, _ = _arc_via_model(n=8, steer=0.5, speed=8.0)  # aggressive
    d = sf.check_limits(hard)
    assert d["curvature_violation"].any()
    assert d["lateral_accel_violation"].any()
    mild, _ = _arc_via_model(n=8, steer=0.1, speed=1.0)
    dm = sf.check_limits(mild)
    assert not dm["curvature_violation"].any()
    assert not dm["lateral_accel_violation"].any()


def test_project_clamps_curvature_and_steer():
    sf = SafetyFilter(SafetyFilterConfig(max_steering=0.3, max_lateral_accel=4.0))
    traj = _straight_traj(n=8, speed=2.0, steering=1.0)  # over-limit steer
    out = sf.project_to_feasible(traj, VehicleState(speed=2.0))
    assert out.length > 0
    assert all(abs(w.steering) <= 0.3 + 1e-6 for w in out.waypoints)
    k_out = sf.estimate_curvature(out)
    assert np.all(np.abs(k_out) <= sf.max_curvature + 1e-6)


def test_project_caps_lateral_acceleration():
    sf = SafetyFilter(SafetyFilterConfig(max_lateral_accel=2.0))
    traj = _fast_arc(n=12, steer_equiv=0.4, speed=8.0)  # aggressive
    out = sf.project_to_feasible(traj, VehicleState(speed=2.0))
    assert out.length > 0
    k_out = sf.estimate_curvature(out)
    v_out = np.array([w.speed for w in out.waypoints], dtype=np.float32)
    lat = v_out ** 2 * np.abs(k_out)
    assert np.all(lat <= sf.config.max_lateral_accel + 2e-2)


def test_project_preserves_safe_trajectory_length():
    sf = SafetyFilter()
    traj = _straight_traj(n=10, speed=2.0, steering=0.1)
    out = sf.filter(traj, VehicleState(speed=2.0))
    assert out.length == traj.length


def test_filter_empty_trajectory():
    sf = SafetyFilter()
    out = sf.filter(Trajectory([]), VehicleState())
    assert out.length == 0


def test_policy_style_arc_filtered_feasible():
    sf = SafetyFilter()
    traj = _heading_arc(n=10, d_yaw=0.02, speed=2.0)  # steering=0, mild arc
    out = sf.filter(traj, VehicleState(speed=2.0))
    assert out.length > 0
    assert sf.is_kinematically_feasible(out, VehicleState(speed=2.0))


def test_collision_truncation_preserved():
    sf = SafetyFilter(SafetyFilterConfig(inflation=0.0))
    s0 = VehicleState(speed=2.0)
    traj = _straight_traj(n=15)
    grid = np.zeros((200, 200), dtype=np.float32)
    occ = OccupancyGrid(data=grid, resolution=0.5, origin=(-50.0, -50.0, 0.0))
    r, c = occ.world_to_index(1.0, 0.0)
    occ.data[r, c] = 1.0
    out = sf.filter(traj, s0, occupancy=occ)
    assert out.length > 0
    assert all(w.x < 1.0 - 1e-3 for w in out.waypoints)
