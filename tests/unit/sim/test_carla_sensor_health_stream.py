import json
from types import SimpleNamespace

import pytest

from sim.carla_baseline import SensorTimingSample
from sim.carla_sensor_health import (
    CarlaNormalHealthMetrics,
    reference_timestamp_from_snapshot,
    write_jsonl_exclusive,
)
from utils.sensor_health import SensorHealthReason, SensorHealthReport

pytestmark = pytest.mark.unit


def _timing(frame: int, timestamp: float) -> SensorTimingSample:
    return SensorTimingSample(
        frame=frame,
        camera_frames=(frame, frame, frame),
        lidar_frame=frame,
        imu_frames=tuple(range(frame - 9, frame + 1)),
        camera_timestamps=(timestamp, timestamp, timestamp),
        lidar_timestamp=timestamp,
        imu_timestamps=tuple(timestamp - 0.1 * index
                             for index in range(9, -1, -1)),
    )


def _report(frame: int, timestamp: float, *, valid=True,
            age=0.0, skew=0.0) -> SensorHealthReport:
    return SensorHealthReport(
        camera_valid=valid,
        lidar_valid=valid,
        imu_valid=valid,
        frame_id=frame,
        reference_timestamp=timestamp,
        max_sensor_age_seconds=age,
        max_sensor_skew_seconds=skew,
        reasons=(() if valid else (SensorHealthReason.CAMERA_BLACK,)),
    )


def test_reference_timestamp_uses_matching_carla_snapshot_elapsed_time():
    snapshot = SimpleNamespace(
        frame=42, timestamp=SimpleNamespace(elapsed_seconds=12.3))

    assert reference_timestamp_from_snapshot(snapshot, expected_frame=42) == 12.3

    with pytest.raises(ValueError, match="snapshot frame"):
        reference_timestamp_from_snapshot(snapshot, expected_frame=43)


def test_normal_metrics_report_false_rejections_and_distributions():
    metrics = CarlaNormalHealthMetrics(
        expected_ticks=3, max_sensor_age_seconds=0.2,
        max_sensor_skew_seconds=0.05, control_period_seconds=0.1)
    metrics.record(_report(10, 1.0, age=0.01, skew=0.01), 0.001,
                   _timing(10, 1.0))
    metrics.record(_report(11, 1.1, age=0.02, skew=0.02), 0.002,
                   _timing(11, 1.1))
    metrics.record(_report(12, 1.2, valid=False, age=0.03, skew=0.03),
                   0.003, _timing(12, 1.2))

    summary = metrics.to_summary()

    assert summary["evaluated_ticks"] == 3
    assert summary["accepted_ticks"] == 2
    assert summary["false_rejections"] == 1
    assert summary["false_rejection_rate"] == pytest.approx(1 / 3)
    assert summary["failure_reasons"] == {"camera_black": 1}
    assert summary["frame_continuity_violations"] == 0
    assert summary["imu_continuity_violations"] == 0
    assert summary["sensor_age_seconds"] == {
        "p50": 0.02, "p95": 0.029, "max": 0.03}
    assert summary["sensor_skew_seconds"] == {
        "p50": 0.02, "p95": 0.029, "max": 0.03}
    assert summary["health_latency_ms"] == {
        "p50": 2.0, "p95": 2.9, "max": 3.0}
    assert summary["exit_gate_passed"] is False


def test_metrics_detect_sensor_frame_and_imu_continuity_gaps():
    metrics = CarlaNormalHealthMetrics(
        expected_ticks=2, max_sensor_age_seconds=0.2,
        max_sensor_skew_seconds=0.05, control_period_seconds=0.1)
    metrics.record(_report(20, 2.0), 0.001, _timing(20, 2.0))
    broken = SensorTimingSample(
        frame=22,
        camera_frames=(22, 21, 22),
        lidar_frame=21,
        imu_frames=(13, 14, 15, 16, 17, 18, 19, 20, 20, 22),
        camera_timestamps=(2.2, 2.2, 2.2),
        lidar_timestamp=2.2,
        imu_timestamps=(1.3, 1.4, 1.5, 1.6, 1.7,
                        1.8, 1.9, 2.0, 2.1, 2.2),
    )
    metrics.record(_report(22, 2.2), 0.001, broken)

    summary = metrics.to_summary()

    assert summary["frame_continuity_violations"] == 1
    assert summary["imu_continuity_violations"] == 1
    assert summary["exit_gate_passed"] is False


def test_1000_normal_ticks_close_the_cpu_metrics_contract():
    metrics = CarlaNormalHealthMetrics(
        expected_ticks=1000, max_sensor_age_seconds=0.2,
        max_sensor_skew_seconds=0.05, control_period_seconds=0.1)
    for index in range(1000):
        frame = 100 + index
        timestamp = 10.0 + index * 0.1
        metrics.record(
            _report(frame, timestamp, age=0.01, skew=0.01),
            0.0005, _timing(frame, timestamp))

    summary = metrics.to_summary()

    assert summary["evaluated_ticks"] == 1000
    assert summary["false_rejections"] == 0
    assert summary["frame_continuity_violations"] == 0
    assert summary["imu_continuity_violations"] == 0
    assert summary["exit_gate_passed"] is True


def test_jsonl_evidence_writer_is_exclusive(tmp_path):
    path = tmp_path / "ticks.jsonl"
    write_jsonl_exclusive(path, [{"frame": 1}, {"frame": 2}])
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"frame": 1}, {"frame": 2}]

    with pytest.raises(FileExistsError):
        write_jsonl_exclusive(path, [{"frame": 3}])


def test_acquisition_error_keeps_a_failed_audit_summary():
    metrics = CarlaNormalHealthMetrics(
        expected_ticks=1000, max_sensor_age_seconds=0.2,
        max_sensor_skew_seconds=0.05, control_period_seconds=0.1)
    metrics.record_acquisition_error("sensor timeout at tick 7")

    summary = metrics.to_summary()

    assert summary["acquisition_errors"] == ["sensor timeout at tick 7"]
    assert summary["exit_gate_passed"] is False
