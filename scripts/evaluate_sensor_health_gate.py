#!/usr/bin/env python3
"""Run the deterministic Stage 1A CPU sensor-health fault matrix."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

from configuration.system import load_system_stack  # noqa: E402
from orad_ros2.vehicle_control_node import PurePursuitController  # noqa: E402
from safety.supervisor import (  # noqa: E402
    SafetySupervisor,
    SafetySupervisorConfig,
)
from sim.carla_closed_loop import carla_control_from_command  # noqa: E402
from utils.sensor_health import SensorHealthGate  # noqa: E402
from utils.types import VehicleState  # noqa: E402


DEFAULT_CONFIG = Path(_HERE).parents[0] / "configs" / "system.yaml"


def _valid_packet(stack, frame: int = 100) -> dict:
    cfg = stack.sensor_health
    height, width = cfg.image_size
    yy, xx = np.indices((height, width))
    image = ((xx + yy) % 180 + 30).astype(np.uint8)
    image = np.repeat(image[:, :, None], 3, axis=2)
    images = [np.roll(image, shift=index, axis=1).copy()
              for index in range(cfg.num_cameras)]

    point_count = max(stack.bev.num_points, cfg.min_lidar_points, 128)
    points = np.zeros((point_count, 4), dtype=np.float32)
    points[:, 0] = np.linspace(1.0, 20.0, point_count, dtype=np.float32)
    points[:, 1] = np.linspace(-10.0, 10.0, point_count, dtype=np.float32)
    points[:, 2] = np.linspace(-1.0, 1.0, point_count, dtype=np.float32)
    points[:, 3] = np.linspace(0.0, 1.0, point_count, dtype=np.float32)

    imu = np.zeros((cfg.imu_steps, 6), dtype=np.float32)
    imu[:, 2] = 9.8
    imu[:, 0] = np.linspace(0.0, 0.2, cfg.imu_steps, dtype=np.float32)
    reference = frame * stack.closed_loop.dt
    latest_imu = reference - 0.01
    imu_gap = min(stack.closed_loop.dt,
                  cfg.max_imu_sample_gap_seconds * 0.9)
    imu_timestamps = tuple(
        latest_imu - (cfg.imu_steps - index - 1) * imu_gap
        for index in range(cfg.imu_steps))
    return {
        "images": images,
        "point_cloud": points,
        "imu_history": imu,
        "frame_id": frame,
        "reference_timestamp": reference,
        "camera_frames": (frame,) * cfg.num_cameras,
        "lidar_frame": frame,
        "imu_frames": tuple(range(frame - cfg.imu_steps + 1, frame + 1)),
        "camera_timestamps": (reference - 0.01,) * cfg.num_cameras,
        "lidar_timestamp": reference - 0.015,
        "imu_timestamps": imu_timestamps,
        "calibration_version": cfg.expected_calibration_version,
    }


def _inject_fault(packet: dict, fault: str, stack) -> None:
    cfg = stack.sensor_health
    if fault == "camera_black":
        packet["images"] = [np.zeros_like(value)
                            for value in packet["images"]]
    elif fault == "camera_saturated":
        packet["images"] = [np.full_like(value, 255)
                            for value in packet["images"]]
    elif fault == "camera_dtype":
        packet["images"][0] = packet["images"][0].astype(np.float32)
    elif fault == "camera_frame_mismatch":
        frames = list(packet["camera_frames"])
        frames[0] -= 1
        packet["camera_frames"] = tuple(frames)
    elif fault == "lidar_empty":
        packet["point_cloud"] = np.zeros((0, 4), dtype=np.float32)
    elif fault == "lidar_sparse":
        packet["point_cloud"] = packet["point_cloud"][
            :cfg.min_lidar_points - 1]
    elif fault == "lidar_out_of_range":
        packet["point_cloud"][:, :3] = (
            cfg.max_lidar_abs_coordinate + 1.0)
    elif fault == "lidar_repeated_points":
        packet["point_cloud"][:] = packet["point_cloud"][0]
    elif fault == "lidar_nonfinite":
        packet["point_cloud"][0, 0] = np.nan
    elif fault == "lidar_stale":
        packet["lidar_timestamp"] = (
            packet["reference_timestamp"]
            - cfg.max_sensor_age_seconds - 0.01)
    elif fault == "imu_incomplete":
        packet["imu_frames"] = packet["imu_frames"][:-1]
    elif fault == "imu_frequency":
        latest = packet["imu_timestamps"][-1]
        gap = cfg.max_imu_sample_gap_seconds + 0.01
        packet["imu_timestamps"] = tuple(
            latest - (cfg.imu_steps - index - 1) * gap
            for index in range(cfg.imu_steps))
    elif fault == "imu_accel_range":
        packet["imu_history"][0, 0] = cfg.max_accel_abs + 1.0
    elif fault == "calibration_mismatch":
        packet["calibration_version"] = "unexpected-calibration"
    else:  # pragma: no cover - internal matrix definition
        raise ValueError(f"unsupported fault: {fault}")


_FAULTS = (
    "camera_black",
    "camera_saturated",
    "camera_dtype",
    "camera_frame_mismatch",
    "lidar_empty",
    "lidar_sparse",
    "lidar_out_of_range",
    "lidar_repeated_points",
    "lidar_nonfinite",
    "lidar_stale",
    "imu_incomplete",
    "imu_frequency",
    "imu_accel_range",
    "calibration_mismatch",
)


def run_fault_matrix(stack, samples: int = 1000) -> dict:
    """Evaluate deterministic faults and their real CPU emergency command."""
    if samples <= 0:
        raise ValueError("samples must be > 0")
    gate = SensorHealthGate(stack.sensor_health)
    supervisor = SafetySupervisor(
        SafetySupervisorConfig(expected_frame="world"))
    controller = PurePursuitController(stack.controller)
    state = VehicleState()
    injected = Counter()
    detected_reasons = Counter()
    rejected = 0
    false_accepts = 0
    maximum_brake = 0
    latencies_ms = []

    for index in range(samples):
        gate.reset()
        fault = _FAULTS[index % len(_FAULTS)]
        packet = deepcopy(_valid_packet(stack, frame=100 + index))
        _inject_fault(packet, fault, stack)
        injected[fault] += 1
        started = time.perf_counter()
        report = gate.evaluate(**packet)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if report.valid:
            false_accepts += 1
            continue
        rejected += 1
        detected_reasons.update(reason.value for reason in report.reasons)
        trajectory = supervisor.emergency_trajectory(
            packet["reference_timestamp"])
        command = controller.compute(
            state, trajectory, stack.closed_loop.dt)
        control = carla_control_from_command(command, stack.closed_loop)
        if control["throttle"] == 0.0 and control["brake"] == 1.0:
            maximum_brake += 1

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    budget_ms = float(stack.closed_loop.dt * 1000.0)
    return {
        "total_faults": samples,
        "rejected_faults": rejected,
        "false_accepts": false_accepts,
        "detection_rate": rejected / samples,
        "maximum_brake_count": maximum_brake,
        "fail_safe_rate": maximum_brake / samples,
        "injected_fault_counts": dict(sorted(injected.items())),
        "detected_reason_counts": dict(sorted(detected_reasons.items())),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "control_budget_ms": budget_ms,
        "control_budget_pass": p95 < budget_ms,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1A CPU sensor-health fault matrix")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    stack = load_system_stack(args.config)
    metrics = run_fault_matrix(stack, samples=args.samples)
    metrics["run_id"] = args.run_id
    metrics["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_dir = Path("runs") / "sensor_health" / args.run_id
    output_path = output_dir / "metrics.json"
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if (
        metrics["false_accepts"] == 0
        and metrics["fail_safe_rate"] == 1.0
        and metrics["control_budget_pass"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
