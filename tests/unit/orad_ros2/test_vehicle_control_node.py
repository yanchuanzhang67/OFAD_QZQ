"""Unit tests for the Pure-Pursuit controller (numpy-only) and rclpy node.

The controller is exercised directly with numpy data (no ROS required). The
:class:`VehicleControlNode` is only constructed when rclpy is importable.
"""
import numpy as np
import pytest

from orad_ros2.vehicle_control_node import (
    ControlCommand,
    PurePursuitConfig,
    PurePursuitController,
)
from utils.types import Trajectory, VehicleState, Waypoint

pytestmark = pytest.mark.unit


def _line_traj(n=10, speed=2.0, dt=0.1):
    return Trajectory([
        Waypoint(x=(i + 1) * speed * dt, y=0.0, yaw=0.0, speed=speed,
                 t=(i + 1) * dt)
        for i in range(n)
    ])


def test_controller_straight_line_zero_steer():
    ctrl = PurePursuitController()
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=2.0)
    cmd = ctrl.compute(state, _line_traj(n=12, speed=2.0))
    assert isinstance(cmd, ControlCommand)
    assert abs(cmd.steering) < 1e-2
    assert cmd.speed > 0.0


def test_controller_turns_toward_target():
    ctrl = PurePursuitController()
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0)
    traj = Trajectory([Waypoint(x=2.0, y=2.0, yaw=0.785, speed=1.0, t=0.1)])
    cmd = ctrl.compute(state, traj)
    assert cmd.steering > 0.01  # target to the left -> steer left


def test_controller_steer_clipped():
    cfg = PurePursuitConfig(max_steering=0.3)
    ctrl = PurePursuitController(cfg)
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0)
    traj = Trajectory([Waypoint(x=-2.0, y=2.0, yaw=0.0, speed=1.0, t=0.1)])
    cmd = ctrl.compute(state, traj)
    assert abs(cmd.steering) <= 0.3 + 1e-6


def test_controller_speed_capped_by_lateral_accel():
    cfg = PurePursuitConfig(
        max_lateral_accel=2.0, wheelbase=2.5, max_steering=0.5)
    ctrl = PurePursuitController(cfg)
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0)
    traj = Trajectory([Waypoint(x=2.0, y=2.0, yaw=0.785, speed=8.0, t=0.1)])
    cmd = ctrl.compute(state, traj)
    k = float(np.tan(abs(cmd.steering)) / cfg.wheelbase)
    assert k * cmd.speed ** 2 <= cfg.max_lateral_accel + 1e-2


def test_controller_empty_trajectory_stops():
    ctrl = PurePursuitController()
    cmd = ctrl.compute(VehicleState(), Trajectory([]))
    assert cmd.steering == 0.0
    assert cmd.speed == 0.0
    assert cmd.accel == ctrl.config.max_decel


def test_controller_short_trajectory_uses_last_point():
    ctrl = PurePursuitController(PurePursuitConfig(lookahead=10.0))
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0)
    traj = _line_traj(n=3, speed=2.0)  # all within lookahead
    target = ctrl.select_target(state, traj)
    assert target is not None
    assert target[0] == 2  # last index


def test_controller_lookahead_grows_with_speed():
    ctrl = PurePursuitController(
        PurePursuitConfig(lookahead=3.0, lookahead_gain=0.6))
    assert ctrl.lookahead_distance(0.0) == pytest.approx(3.0)
    assert ctrl.lookahead_distance(10.0) > 3.0


def test_node_skip_without_full_ros_stack():
    """The node needs rclpy + ackermann_msgs + nav_msgs + std_msgs.

    ``_HAS_RCLPY`` is set at import time by the lazy rclpy block; when any
    required message package is missing the node cannot run, so we skip
    rather than fail (CI without the full ROS stack).
    """
    from orad_ros2.vehicle_control_node import _HAS_RCLPY, VehicleControlNode
    if not _HAS_RCLPY:
        pytest.skip("full ROS 2 stack (rclpy + ackermann/nav/std msgs) "
                    "required to instantiate VehicleControlNode")
    import rclpy
    rclpy.init()
    try:
        node = VehicleControlNode()
        assert node.get_name() == "vehicle_control"
        node.destroy_node()
    finally:
        rclpy.shutdown()
