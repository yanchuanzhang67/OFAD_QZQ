"""Closed-loop CARLA / Gazebo off-road driving tests (SDD Closed-Loop tier).

Pass criteria: success rate > 90%, collision rate 0, attitude-overload alarms 0.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.closed_loop


def test_steep_slope_passability():
    pytest.skip("requires CARLA + 20deg off-road map")


def test_attitude_stability():
    pytest.skip("requires Gazebo suspension dynamics")


def test_fixed_carla_baseline_is_reproducible_across_three_runs():
    run_id = os.environ.get("ORAD_CARLA_BASELINE_RUN_ID")
    if not run_id:
        pytest.skip(
            "requires running CARLA 0.9.16 Town10HD_Opt and "
            "ORAD_CARLA_BASELINE_RUN_ID")
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable, "scripts/verify_carla_baseline.py",
            "--run-id", run_id,
        ], cwd=root, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary_path = root / "runs" / "carla_baseline" / run_id / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["repetitions"] == 3
    assert summary["config_consistent"] is True
    assert summary["sensor_timing_consistent"] is True
    assert summary["reproducible"] is True


def test_normal_carla_sensor_stream_has_zero_false_rejections():
    run_id = os.environ.get("ORAD_CARLA_HEALTH_RUN_ID")
    if not run_id:
        pytest.skip(
            "requires running CARLA 0.9.16 Town10HD_Opt and "
            "ORAD_CARLA_HEALTH_RUN_ID")
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable, "scripts/evaluate_carla_sensor_health.py",
            "--run-id", run_id, "--ticks", "1000",
        ], cwd=root, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary_path = (
        root / "runs" / "carla_sensor_health" / run_id / "summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["evaluated_ticks"] >= 1000
    assert summary["false_rejections"] == 0
    assert summary["frame_continuity_violations"] == 0
    assert summary["imu_continuity_violations"] == 0
    assert summary["sensor_skew_seconds"]["max"] <= 0.05
    assert summary["sensor_age_seconds"]["max"] <= 0.20
    assert summary["health_latency_ms"]["p95"] < 100.0
    assert summary["exit_gate_passed"] is True
