"""Unit tests for the numpy-only CARLA closed-loop helpers (no CARLA/torch)."""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# `src/` is on sys.path via conftest.py; this guard keeps the file importable
# when run standalone.
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sim.carla_closed_loop import (  # noqa: E402
    ClosedLoopConfig,
    EpisodeMetrics,
    aggregate_episodes,
    canonicalize_points,
    carla_control_from_command,
    ego_trajectory_to_world,
    imu_attitude_from_accel,
    imu_samples_to_array,
    occupancy_from_points,
    policy_trajectory_to_waypoints,
)
from utils.types import VehicleState  # noqa: E402


class _Cmd:
    def __init__(self, steering, speed, accel):
        self.steering = steering
        self.speed = speed
        self.accel = accel


def test_canonicalize_points_shape_and_pad():
    pts = np.arange(10 * 3).reshape(10, 3).astype(np.float32)
    out = canonicalize_points(pts, 256)
    assert out.shape == (256, 4)
    assert out.dtype == np.float32
    assert np.all(out[10:] == 0.0)          # padded region zero
    assert np.all(out[:10, :3] == pts)      # xyz preserved
    assert np.all(out[:10, 3] == 0.0)       # intensity filled with 0


def test_canonicalize_points_truncates():
    pts = np.ones((1000, 4), np.float32)
    out = canonicalize_points(pts, 256)
    assert out.shape == (256, 4)


def test_canonicalize_empty():
    out = canonicalize_points(np.zeros((0, 4), np.float32), 64)
    assert out.shape == (64, 4)


def test_imu_samples_preserve_accel_and_gyro():
    samples = [
        (np.array([1, 2, 3], np.float32), np.array([4, 5, 6], np.float32)),
        (np.array([7, 8, 9], np.float32), np.array([10, 11, 12], np.float32)),
    ]
    imu = imu_samples_to_array(samples, steps=4)
    assert imu.shape == (4, 6)
    assert imu.dtype == np.float32
    assert np.all(imu[:2] == 0.0)
    assert np.allclose(imu[-2], [1, 2, 3, 4, 5, 6])
    assert np.allclose(imu[-1], [7, 8, 9, 10, 11, 12])


def test_occupancy_from_points_marks_cells():
    rng, res = (-10.0, 10.0), 1.0
    pts = np.array([[5.0, 5.0, 0.5], [-5.0, -5.0, 0.5],
                    [5.0, 5.0, 3.0]], np.float32)  # last out of band
    grid = occupancy_from_points(pts, rng, rng, res)
    assert grid.data.shape == (20, 20)
    assert grid.is_occupied(5.0, 5.0)
    assert grid.is_occupied(-5.0, -5.0)
    assert not grid.is_occupied(0.0, 0.0)
    assert grid.origin[0] == rng[0] and grid.origin[1] == rng[0]


def test_occupancy_empty_grid():
    grid = occupancy_from_points(np.zeros((0, 4)), (-5, 5), (-5, 5), 0.5)
    assert grid.data.shape == (20, 20)
    assert not grid.data.any()


def test_occupancy_grid_world_pose_rotates_queries():
    points = np.array([[2.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    state = VehicleState(x=10.0, y=5.0, yaw=np.pi / 2)
    grid = occupancy_from_points(points, (-5, 5), (-5, 5), 0.5,
                                 vehicle_state=state)
    assert grid.is_occupied(10.0, 7.0)
    assert not grid.is_occupied(12.0, 5.0)


def test_imu_attitude_level_is_identity():
    a = np.tile([0.0, 0.0, 9.81], (10, 1)).astype(np.float32)
    R = imu_attitude_from_accel(a)
    assert np.allclose(R, np.eye(3, dtype=np.float32), atol=1e-5)


def test_imu_attitude_aligns_up():
    # tilt forward: body up = (sin p, 0, cos p)
    p = 0.2
    a = np.tile([np.sin(p), 0.0, np.cos(p)], (5, 1)).astype(np.float32) * 9.81
    R = imu_attitude_from_accel(a)
    up_world = R @ np.array([np.sin(p), 0.0, np.cos(p)], np.float32)
    assert np.allclose(up_world, [0.0, 0.0, 1.0], atol=1e-4)


def test_ego_trajectory_to_world_identity_yaw():
    state = VehicleState(x=10.0, y=5.0, yaw=0.0)
    traj = np.array([[2.0, 0.0, 0.0, 1.0]], np.float32)
    out = ego_trajectory_to_world(traj, state)
    wp = out.waypoints[0]
    assert wp.x == pytest.approx(12.0)
    assert wp.y == pytest.approx(5.0)
    assert wp.yaw == pytest.approx(0.0)
    assert wp.speed == pytest.approx(1.0)


def test_ego_trajectory_to_world_yaw_rotation():
    state = VehicleState(x=10.0, y=5.0, yaw=np.pi / 2)
    traj = np.array([[2.0, 0.0, 0.5, 1.0]], np.float32)
    out = ego_trajectory_to_world(traj, state)
    wp = out.waypoints[0]
    assert wp.x == pytest.approx(10.0, abs=1e-4)
    assert wp.y == pytest.approx(7.0, abs=1e-4)
    assert wp.yaw == pytest.approx(np.pi / 2 + 0.5)


def test_policy_trajectory_world_frame_passthrough():
    state = VehicleState(x=1.0, y=2.0, yaw=0.5)
    traj = np.array([[[3.0, 4.0, 0.1, 2.0]]], np.float32)  # (1,1,4)
    out = policy_trajectory_to_waypoints(traj, state, frame="world")
    wp = out.waypoints[0]
    assert wp.x == 3.0 and wp.y == 4.0 and wp.speed == 2.0


def test_carla_control_zero_and_limits():
    cfg = ClosedLoopConfig()
    z = carla_control_from_command(_Cmd(0.0, 0.0, 0.0), cfg)
    assert z["throttle"] == 0.0 and z["steer"] == 0.0 and z["brake"] == 0.0
    full = carla_control_from_command(
        _Cmd(cfg.max_steering, cfg.max_speed, cfg.max_accel), cfg)
    assert full["steer"] == pytest.approx(1.0)
    assert full["throttle"] == pytest.approx(1.0)
    assert full["brake"] == 0.0
    brake = carla_control_from_command(
        _Cmd(-cfg.max_steering, 0.0, cfg.max_decel), cfg)
    assert brake["steer"] == pytest.approx(-1.0)
    assert brake["throttle"] == 0.0
    assert brake["brake"] == pytest.approx(1.0)


def test_episode_metrics_progress_and_alarms():
    cfg = ClosedLoopConfig()
    m = EpisodeMetrics()
    s0 = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=2.0, yaw_rate=0.0)
    s1 = VehicleState(x=1.0, y=0.0, pitch=cfg.pitch_alarm + 0.1,
                      speed=5.0, yaw_rate=1.0)  # lat accel 5 > 4
    m.update(s0, None, cfg)
    m.update(s1, s0, cfg)
    assert m.progress == pytest.approx(1.0)
    assert m.pitch_alarms == 1
    assert m.lat_accel_max == pytest.approx(5.0)
    assert m.lat_accel_alarms == 1
    summ = m.to_summary()
    assert summ["attitude_alarms"] == 2
    assert summ["success"] is False   # not reached_goal + collisions


def test_aggregate_episodes_rates():
    a = EpisodeMetrics()
    a.reached_goal = True              # success, no collision
    b = EpisodeMetrics()
    b.collisions = 1                   # crash
    out = aggregate_episodes([a, b])
    assert out["n_episodes"] == 2
    assert out["pass_rate"] == 0.5
    assert out["collision_rate"] == 0.5
    assert aggregate_episodes([])["pass_rate"] == 0.0
