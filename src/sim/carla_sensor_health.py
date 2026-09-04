"""Metrics for normal CARLA sensor streams evaluated before model inference."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from sim.carla_baseline import SensorTimingSample
from utils.sensor_health import SensorHealthReport


def reference_timestamp_from_snapshot(snapshot, *, expected_frame: int) -> float:
    """Return CARLA simulation time and reject a snapshot from another tick."""
    if int(snapshot.frame) != int(expected_frame):
        raise ValueError(
            f"snapshot frame {snapshot.frame} != sensor frame {expected_frame}")
    timestamp = float(snapshot.timestamp.elapsed_seconds)
    if not np.isfinite(timestamp):
        raise ValueError("snapshot timestamp must be finite")
    return timestamp


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": round(float(np.percentile(values, 50)), 12),
        "p95": round(float(np.percentile(values, 95)), 12),
        "max": round(float(np.max(values)), 12),
    }


class CarlaNormalHealthMetrics:
    """Aggregate false rejection, timing, and continuity for normal ticks."""

    def __init__(self, *, expected_ticks: int, max_sensor_age_seconds: float,
                 max_sensor_skew_seconds: float,
                 control_period_seconds: float):
        if expected_ticks <= 0:
            raise ValueError("expected_ticks must be positive")
        self.expected_ticks = int(expected_ticks)
        self.max_sensor_age_seconds = float(max_sensor_age_seconds)
        self.max_sensor_skew_seconds = float(max_sensor_skew_seconds)
        self.control_period_seconds = float(control_period_seconds)
        self.reports: list[SensorHealthReport] = []
        self.latencies_ms: list[float] = []
        self.sensor_ages: list[float] = []
        self.sensor_skews: list[float] = []
        self.failure_reasons: Counter = Counter()
        self.frame_continuity_violations = 0
        self.imu_continuity_violations = 0
        self.acquisition_errors: list[str] = []
        self._last_frame = None

    def record_acquisition_error(self, message: str) -> None:
        self.acquisition_errors.append(str(message))

    def record(self, report: SensorHealthReport, latency_seconds: float,
               timing: SensorTimingSample) -> dict:
        frame_bad = (
            (self._last_frame is not None
             and timing.frame != self._last_frame + 1)
            or any(frame != timing.frame for frame in timing.camera_frames)
            or timing.lidar_frame != timing.frame)
        if frame_bad:
            self.frame_continuity_violations += 1
        imu_bad = (
            not timing.imu_frames
            or timing.imu_frames[-1] != timing.frame
            or any(right != left + 1 for left, right in zip(
                timing.imu_frames, timing.imu_frames[1:])))
        if imu_bad:
            self.imu_continuity_violations += 1
        self._last_frame = timing.frame
        self.reports.append(report)
        self.latencies_ms.append(max(float(latency_seconds), 0.0) * 1000.0)
        self.sensor_ages.append(float(report.max_sensor_age_seconds))
        self.sensor_skews.append(float(report.max_sensor_skew_seconds))
        self.failure_reasons.update(reason.value for reason in report.reasons)
        return {
            **timing.to_dict(),
            "reference_timestamp": report.reference_timestamp,
            "valid": report.valid,
            "reasons": [reason.value for reason in report.reasons],
            "max_sensor_age_seconds": report.max_sensor_age_seconds,
            "max_sensor_skew_seconds": report.max_sensor_skew_seconds,
            "health_latency_ms": self.latencies_ms[-1],
        }

    def to_summary(self) -> dict:
        evaluated = len(self.reports)
        rejected = sum(not report.valid for report in self.reports)
        age = _distribution(self.sensor_ages)
        skew = _distribution(self.sensor_skews)
        latency = _distribution(self.latencies_ms)
        gate_passed = (
            self.expected_ticks >= 1000
            and evaluated >= self.expected_ticks
            and rejected == 0
            and self.frame_continuity_violations == 0
            and self.imu_continuity_violations == 0
            and not self.acquisition_errors
            and age["max"] <= self.max_sensor_age_seconds
            and skew["max"] <= self.max_sensor_skew_seconds
            and latency["p95"] < self.control_period_seconds * 1000.0)
        return {
            "evaluated_ticks": evaluated,
            "accepted_ticks": evaluated - rejected,
            "false_rejections": rejected,
            "false_rejection_rate": rejected / max(evaluated, 1),
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
            "frame_continuity_violations": (
                self.frame_continuity_violations),
            "imu_continuity_violations": self.imu_continuity_violations,
            "acquisition_errors": list(self.acquisition_errors),
            "sensor_age_seconds": age,
            "sensor_skew_seconds": skew,
            "health_latency_ms": latency,
            "thresholds": {
                "max_sensor_age_seconds": self.max_sensor_age_seconds,
                "max_sensor_skew_seconds": self.max_sensor_skew_seconds,
                "control_period_seconds": self.control_period_seconds,
            },
            "exit_gate_passed": gate_passed,
        }


def write_jsonl_exclusive(path: Path, records: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True))
            stream.write("\n")
