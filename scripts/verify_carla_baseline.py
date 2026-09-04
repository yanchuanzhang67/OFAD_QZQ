#!/usr/bin/env python3
"""Run the fixed CARLA sensor baseline three times and save audit evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

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
    compare_repetition_records,
    resolved_baseline_values,
    validate_carla_environment,
    write_json_exclusive,
)
from sim.carla_closed_loop import CarlaSensorStack, _HAS_CARLA  # noqa: E402

DEFAULT_CONFIG = _ROOT / "configs" / "system.yaml"


def connect_carla(host: str, port: int):
    import carla
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    return client


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the fixed three-run CARLA sensor baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--runs-root", type=Path, default=_ROOT / "runs" / "carla_baseline")
    args = parser.parse_args(argv)
    if not _HAS_CARLA:
        raise SystemExit("carla PythonAPI not installed")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    stack = load_system_stack(args.config)
    baseline = stack.carla_baseline
    output_dir = args.runs_root / args.run_id
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"baseline run already exists: {output_dir}")

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
        blueprint_ids = tuple(bp.id for bp in world.get_blueprint_library())
        validate_carla_environment(
            baseline,
            client_version=client.get_client_version(),
            server_version=client.get_server_version(),
            map_name=world.get_map().name,
            available_blueprints=blueprint_ids,
            fixed_delta_seconds=world.get_settings().fixed_delta_seconds,
            expected_control_period=stack.closed_loop.dt,
        )

        random.seed(baseline.random_seed)
        np.random.seed(baseline.random_seed)
        if hasattr(world, "set_pedestrians_seed"):
            world.set_pedestrians_seed(baseline.random_seed)
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(baseline.random_seed)

        blueprint = world.get_blueprint_library().find(
            baseline.vehicle_blueprint)
        spawn = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(blueprint, spawn)
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
        manifest["steps_per_repetition"] = args.steps
        manifest["resolved_baseline"] = resolved_baseline_values(
            baseline,
            calibration_version=(
                stack.sensor_health.expected_calibration_version),
            control_period_seconds=stack.closed_loop.dt,
            num_cameras=stack.bev.num_cameras,
            image_size=stack.bev.image_size,
        )
        write_json_exclusive(manifest_path, manifest)

        records = []
        for repetition in range(1, baseline.repetitions + 1):
            vehicle.set_transform(spawn)
            trace = []
            for _ in range(args.steps):
                if sensors.tick() is None:
                    raise RuntimeError(
                        f"sensor timeout in repetition {repetition}")
                trace.append(capture_sensor_timing(sensors))
            record = {
                "repetition": repetition,
                "resolved_baseline": manifest["resolved_baseline"],
                "sensor_timing": trace,
            }
            records.append(record)
            write_json_exclusive(
                output_dir / f"repetition-{repetition:02d}.json",
                {
                    "repetition": repetition,
                    "resolved_baseline": record["resolved_baseline"],
                    "sensor_timing": [sample.to_dict() for sample in trace],
                },
            )

        summary = compare_repetition_records(
            records, expected_repetitions=baseline.repetitions)
        summary["manifest"] = os.path.relpath(manifest_path, _ROOT)
        write_json_exclusive(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["reproducible"] else 1
    finally:
        if sensors is not None:
            sensors.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    raise SystemExit(main())
