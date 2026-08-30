"""Unit tests for the kinematic bicycle model (pure-numpy forward dynamics)."""
import numpy as np
import pytest

from safety.bicycle_model import KinematicBicycleModel
from utils.types import VehicleState

pytestmark = pytest.mark.unit


def test_step_straight_line():
    m = KinematicBicycleModel(wheelbase=2.5)
    s0 = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=2.0)
    s1 = m.step(s0, steering=0.0, accel=0.0, dt=0.1)
    assert s1.x == pytest.approx(0.2)
    assert s1.y == pytest.approx(0.0)
    assert s1.yaw == pytest.approx(0.0)
    assert s1.speed == pytest.approx(2.0)


def test_step_turning_changes_yaw():
    m = KinematicBicycleModel()
    s0 = VehicleState(speed=2.0)
    s1 = m.step(s0, steering=0.3, accel=0.0, dt=0.1)
    assert s1.yaw > 0.0


def test_steer_clipping():
    m = KinematicBicycleModel(max_steer=0.5)
    s0 = VehicleState(speed=1.0)
    s1 = m.step(s0, steering=1.0, accel=0.0, dt=0.1)
    assert abs(s1.steering) <= 0.5 + 1e-6


def test_rollout_length_and_time():
    m = KinematicBicycleModel()
    s0 = VehicleState(speed=2.0)
    controls = np.zeros((10, 2), dtype=np.float32)
    traj = m.rollout(s0, controls, dt=0.1)
    assert traj.length == 10
    assert traj.waypoints[-1].t == pytest.approx(1.0)


def test_rollout_is_feasible():
    m = KinematicBicycleModel()
    s0 = VehicleState(speed=2.0)
    controls = np.full((10, 2), 0.1, dtype=np.float32)
    traj = m.rollout(s0, controls, dt=0.1)
    assert m.is_feasible(traj, s0, dt=0.1)
