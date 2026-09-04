#!/usr/bin/env python3
"""Evaluate SensorHealthGate on a normal CARLA stream without ML models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from configuration.system import load_system_stack  # noqa: E402
from sim.carla_baseline import (  # noqa: E402
    build_baseline_manifest,
    capture_sensor_timing,
    carla_sensor_attributes,
    collect_git_metadata,
    resolved_baseline_values,
    validate_carla_environment,
    write_json_exclusive,
)
from sim.carla_closed_loop import CarlaSensorStack, _HAS_CARLA  # noqa: E402
from sim.carla_sensor_health import (  # noqa: E402
    CarlaNormalHealthMetrics, write_jsonl_exclusive)
from utils.sensor_health import SensorHealthGate  # noqa: E402

DEFAULT_CONFIG = _ROOT / "configs" / "system.yaml"


def connect_carla(host: str, port: int):
    import carla
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    return client


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run normal CARLA sensors through SensorHealthGate")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument(
        "--runs-root", type=Path,
        default=_ROOT / "runs" / "carla_sensor_health")
    args = parser.parse_args(argv)
    if not _HAS_CARLA:
        raise SystemExit("carla PythonAPI not installed")
    if args.ticks < 1000:
        raise SystemExit("--ticks must be at least 1000")

    stack = load_system_stack(args.config)
    if args.warmup_ticks < stack.bev.imu_steps:
        raise SystemExit(
            f"--warmup-ticks must be at least {stack.bev.imu_steps}")
    baseline = stack.carla_baseline
    output_dir = args.runs_root / args.run_id
    if output_dir.exists():
        raise SystemExit(f"sensor-health run already exists: {output_dir}")

    client = connect_carla(args.host, args.port)
    world = client.get_world()
    original_settings = world.get_settings()
    sensors = None
    vehicle = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = stack.closed_loop.dt
        world.apply_settings(settings)
        blueprints = world.get_blueprint_library()
        validate_carla_environment(
            baseline,
            client_version=client.get_client_version(),
            server_version=client.get_server_version(),
            map_name=world.get_map().name,
            available_blueprints=tuple(bp.id for bp in blueprints),
            fixed_delta_seconds=world.get_settings().fixed_delta_seconds,
            expected_control_period=stack.closed_loop.dt,
        )
        random.seed(baseline.random_seed)
        np.random.seed(baseline.random_seed)
        if hasattr(world, "set_pedestrians_seed"):
            world.set_pedestrians_seed(baseline.random_seed)

        spawn = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(
            blueprints.find(baseline.vehicle_blueprint), spawn)
        sensors = CarlaSensorStack(
            world, vehicle, stack.closed_loop,
            num_cameras=stack.bev.num_cameras,
            image_size=stack.bev.image_size,
            calibration_version=(
                stack.sensor_health.expected_calibration_version),
            sensor_attributes=carla_sensor_attributes(
                baseline, stack.bev.image_size),
            baseline=baseline,
        )

        import carla
        normal_control = carla.VehicleControl(
            throttle=0.15, steer=0.05, brake=0.0)
        vehicle.apply_control(normal_control)
        for warmup_index in range(args.warmup_ticks):
            if sensors.tick() is None:
                raise RuntimeError(
                    f"sensor timeout during warm-up tick {warmup_index + 1}")

        gate = SensorHealthGate(stack.sensor_health)
        metrics = CarlaNormalHealthMetrics(
            expected_ticks=args.ticks,
            max_sensor_age_seconds=(
                stack.sensor_health.max_sensor_age_seconds),
            max_sensor_skew_seconds=(
                stack.sensor_health.max_sensor_skew_seconds),
            control_period_seconds=stack.closed_loop.dt,
        )
        tick_records = []
        for tick_index in range(args.ticks):
            packet = sensors.tick()
            if packet is None:
                metrics.record_acquisition_error(
                    f"sensor timeout at evaluated tick {tick_index + 1}")
                break
            images, points, imu = packet
            started = time.perf_counter()
            report = gate.evaluate(
                images=images,
                point_cloud=points,
                imu_history=imu,
                frame_id=sensors.last_frame,
                reference_timestamp=sensors.last_reference_timestamp,
                camera_frames=sensors.last_camera_frames,
                lidar_frame=sensors.last_lidar_frame,
                imu_frames=sensors.last_imu_frames,
                camera_timestamps=sensors.last_camera_timestamps,
                lidar_timestamp=sensors.last_lidar_timestamp,
                imu_timestamps=sensors.last_imu_timestamps,
                calibration_version=sensors.calibration_version,
            )
            latency = time.perf_counter() - started
            tick_records.append(metrics.record(
                report, latency, capture_sensor_timing(sensors)))

        manifest = build_baseline_manifest(
            baseline=baseline,
            config_path=args.config,
            calibration_version=(
                stack.sensor_health.expected_calibration_version),
            control_period_seconds=stack.closed_loop.dt,
            git_metadata=collect_git_metadata(_ROOT),
            client_version=client.get_client_version(),
            server_version=client.get_server_version(),
            observed_map_name=world.get_map().name,
        )
        manifest.update({
            "evaluation": "normal_carla_sensor_health",
            "warmup_ticks": args.warmup_ticks,
            "requested_ticks": args.ticks,
            "normal_control": {
                "throttle": 0.15, "steer": 0.05, "brake": 0.0},
            "resolved_baseline": resolved_baseline_values(
                baseline,
                calibration_version=(
                    stack.sensor_health.expected_calibration_version),
                control_period_seconds=stack.closed_loop.dt,
                num_cameras=stack.bev.num_cameras,
                image_size=stack.bev.image_size),
            "health_thresholds": {
                "max_sensor_age_seconds": (
                    stack.sensor_health.max_sensor_age_seconds),
                "max_sensor_skew_seconds": (
                    stack.sensor_health.max_sensor_skew_seconds),
                "max_imu_sample_gap_seconds": (
                    stack.sensor_health.max_imu_sample_gap_seconds),
            },
        })
        summary = metrics.to_summary()
        write_json_exclusive(output_dir / "manifest.json", manifest)
        write_jsonl_exclusive(output_dir / "ticks.jsonl", tick_records)
        write_json_exclusive(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["exit_gate_passed"] else 1
    finally:
        if vehicle is not None:
            try:
                import carla
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
            except Exception:
                pass
        if sensors is not None:
            sensors.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    raise SystemExit(main())
