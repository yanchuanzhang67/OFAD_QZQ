import numpy as np
import pytest

from safety.supervisor import SafetyMode, SafetySupervisor, SafetySupervisorConfig
from utils.types import Trajectory, Waypoint

pytestmark = pytest.mark.unit


def _trajectory(frame="ego", n=4, speed=2.0):
    return Trajectory([
        Waypoint(x=float(i + 1), y=0.0, yaw=0.0, speed=speed, t=0.1 * (i + 1))
        for i in range(n)
    ], frame=frame)


@pytest.mark.parametrize("trajectory,reason", [
    (None, "missing_trajectory"),
    (Trajectory([], frame="ego"), "too_few_waypoints"),
    (_trajectory(n=1), "too_few_waypoints"),
    (_trajectory(frame="world"), "frame_mismatch"),
    (_trajectory(speed=-1.0), "invalid_speed"),
    (_trajectory(speed=1000.0), "invalid_speed"),
])
def test_supervisor_rejects_invalid_trajectory(trajectory, reason):
    decision = SafetySupervisor().evaluate(trajectory)
    assert decision.mode is SafetyMode.EMERGENCY_STOP
    assert decision.reason == reason


def test_supervisor_rejects_nan_inf_and_timeouts():
    for value in (np.nan, np.inf):
        trajectory = _trajectory()
        trajectory.waypoints[1].x = value
        assert SafetySupervisor().evaluate(trajectory).mode is SafetyMode.EMERGENCY_STOP
    supervisor = SafetySupervisor(SafetySupervisorConfig(
        max_sensor_age=0.2, max_model_latency=0.1))
    assert supervisor.evaluate(_trajectory(), sensor_age=0.21).reason == "sensor_timeout"
    assert supervisor.evaluate(_trajectory(), model_latency=0.11).reason == "model_timeout"
    assert supervisor.evaluate(_trajectory(), sensor_skew=0.06).reason == "sensor_skew"


@pytest.mark.parametrize("field,value", [
    ("sensor_age", -0.1),
    ("sensor_age", np.nan),
    ("sensor_skew", -0.1),
    ("sensor_skew", np.inf),
    ("model_latency", -0.1),
    ("model_latency", np.nan),
])
def test_supervisor_rejects_invalid_timing_metrics(field, value):
    decision = SafetySupervisor().evaluate(_trajectory(), **{field: value})
    assert decision.mode is SafetyMode.EMERGENCY_STOP
    assert decision.reason == "invalid_timing"


def test_supervisor_normal_and_degraded_modes():
    supervisor = SafetySupervisor()
    normal = supervisor.evaluate(_trajectory())
    degraded = supervisor.evaluate(_trajectory(), safety_intervened=True)
    assert normal.mode is SafetyMode.NORMAL
    assert degraded.mode is SafetyMode.DEGRADED
    assert degraded.reason == "safety_intervention"


def test_supervisor_emergency_trajectory_keeps_contract():
    stop = SafetySupervisor().emergency_trajectory(timestamp=2.5)
    assert stop.length == 0
    assert stop.frame == "ego"
    assert stop.timestamp == pytest.approx(2.5)
