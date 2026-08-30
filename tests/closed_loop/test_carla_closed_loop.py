"""Closed-loop CARLA / Gazebo off-road driving tests (SDD Closed-Loop tier).

Pass criteria: success rate > 90%, collision rate 0, attitude-overload alarms 0.
"""
import pytest

pytestmark = pytest.mark.closed_loop


def test_steep_slope_passability():
    pytest.skip("requires CARLA + 20deg off-road map")


def test_attitude_stability():
    pytest.skip("requires Gazebo suspension dynamics")
