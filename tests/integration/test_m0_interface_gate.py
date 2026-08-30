"""M0 Gate: 1000 synthetic samples cannot leak invalid trajectories to control."""
import numpy as np
import pytest

from orad_ros2.vehicle_control_node import PurePursuitController
from safety.supervisor import SafetyMode, SafetySupervisor
from utils.types import Trajectory, VehicleState, Waypoint

pytestmark = pytest.mark.integration


def _sample(i):
    trajectory = Trajectory([
        Waypoint(x=float(k + 1), y=0.01 * i, yaw=0.0, speed=2.0, t=0.1 * (k + 1))
        for k in range(5)
    ], frame="ego", timestamp=i * 0.1)
    if i % 10 == 0:
        trajectory.waypoints[2].x = np.nan
    return trajectory


def test_1000_sample_interface_and_failsafe_gate():
    supervisor = SafetySupervisor()
    controller = PurePursuitController()
    state = VehicleState()
    emergency_count = 0
    for i in range(1000):
        raw = _sample(i)
        decision = supervisor.evaluate(raw)
        if decision.mode is SafetyMode.EMERGENCY_STOP:
            emergency_count += 1
            command = controller.compute(
                state, supervisor.emergency_trajectory(raw.timestamp))
            assert command.speed == 0.0
            assert command.accel == controller.config.max_decel
        else:
            assert np.isfinite(raw.to_array()).all()
    assert emergency_count == 100
