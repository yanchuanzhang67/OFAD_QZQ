#!/usr/bin/env python3
"""Collect one synchronized CARLA episode for New_ORAD pipeline smoke tests.

CARLA is imported lazily so the data helpers remain testable without a running
simulator.  This initial collector uses Traffic Manager to prove the sensor and
storage path; its controls are not expert labels for off-road behaviour cloning.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from queue import Empty, Queue
import shutil
import sys
import time
from typing import Iterable
import uuid

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from configuration.system import load_system_stack  # noqa: E402
from sim.carla_baseline import (  # noqa: E402
    canonical_yaml_sha256,
    carla_sensor_attributes,
    sha256_file,
    validate_carla_environment,
)


def canonicalize_lidar(
    points: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Convert CARLA XYZI into deterministic New_ORAD ego-frame points.

    CARLA sensor coordinates are x-forward/y-right/z-up. New_ORAD canonical
    ego coordinates are x-forward/y-left/z-up, hence the explicit Y flip.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    raw = np.asarray(points, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(f"points must have shape (N,4), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("points must contain only finite values")

    canonical_source = raw.copy()
    canonical_source[:, 1] *= -1.0
    raw_count = int(canonical_source.shape[0])
    valid_count = min(raw_count, count)

    if raw_count > count:
        indices = np.random.default_rng(seed).choice(
            raw_count, size=count, replace=False
        )
        canonical = canonical_source[indices]
    else:
        padding = np.zeros((count - raw_count, 4), dtype=np.float32)
        canonical = np.concatenate([canonical_source, padding], axis=0)

    metadata = {
        "raw_point_count": raw_count,
        "valid_point_count": valid_count,
        "padding_count": count - valid_count,
        "seed": int(seed),
    }
    return np.ascontiguousarray(canonical, dtype=np.float32), metadata


def build_imu_history(
    samples: Iterable[tuple[int, float, np.ndarray]],
    steps: int,
) -> tuple[np.ndarray, list[int], list[float | None], int]:
    """Return a front-padded old-to-new IMU window and its timing metadata."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    selected = list(samples)[-steps:]
    valid_count = len(selected)
    padding_count = steps - valid_count

    rows = [np.zeros(6, dtype=np.float32) for _ in range(padding_count)]
    frames = [-1] * padding_count
    timestamps: list[float | None] = [None] * padding_count
    for frame, timestamp, values in selected:
        row = np.asarray(values, dtype=np.float32).reshape(6)
        if not np.isfinite(row).all():
            raise ValueError("IMU values must contain only finite values")
        rows.append(row)
        frames.append(int(frame))
        timestamps.append(float(timestamp))

    return np.asarray(rows, dtype=np.float32), frames, timestamps, valid_count


def get_exact_frame(
    data_queue,
    expected_frame: int,
    timeout: float,
    sensor_name: str = "sensor",
):
    """Discard stale sensor packets and return exactly ``expected_frame``."""
    while True:
        try:
            data = data_queue.get(timeout=timeout)
        except Empty as error:
            raise TimeoutError(
                f"{sensor_name} timed out waiting for sensor frame "
                f"{expected_frame}"
            ) from error
        frame = int(data.frame)
        if frame < expected_frame:
            continue
        if frame > expected_frame:
            raise RuntimeError(
                f"{sensor_name} expected frame {expected_frame}, "
                f"received future frame {frame}"
            )
        return data


def camera_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Build a pinhole intrinsic matrix from horizontal field of view."""
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError("fov_deg must be in (0, 180)")
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a synchronized CARLA smoke-test episode."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("datasets/carla_initial"))
    parser.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    return parser


def recording_plan_from_stack(stack) -> dict:
    """Resolve every physical collector value from the canonical stack."""
    baseline = stack.carla_baseline
    attributes = carla_sensor_attributes(baseline, stack.bev.image_size)
    cameras = {
        camera.name: {
            "blueprint": camera.blueprint,
            "location_m": camera.transform.location_m,
            "rotation_degrees": camera.transform.rotation_degrees,
            "fov": camera.fov_degrees,
            "attributes": attributes["cameras"][camera.name],
        }
        for camera in baseline.cameras
    }
    return {
        "carla_version": baseline.version,
        "map_name": baseline.map_name,
        "vehicle_blueprint": baseline.vehicle_blueprint,
        "random_seed": baseline.random_seed,
        "fixed_delta_seconds": stack.closed_loop.dt,
        "calibration_version": (
            stack.sensor_health.expected_calibration_version),
        "image_size": stack.bev.image_size,
        "camera": cameras,
        "lidar": {
            "blueprint": baseline.lidar.blueprint,
            "location_m": baseline.lidar.transform.location_m,
            "rotation_degrees": baseline.lidar.transform.rotation_degrees,
            "attributes": attributes["lidar"],
            "points_per_second": baseline.lidar.points_per_second,
            "canonical_points": baseline.lidar.canonical_points,
        },
        "imu": {
            "blueprint": baseline.imu.blueprint,
            "location_m": baseline.imu.transform.location_m,
            "rotation_degrees": baseline.imu.transform.rotation_degrees,
            "attributes": attributes["imu"],
            "history_steps": baseline.imu.history_steps,
            "channels": baseline.imu.channels,
        },
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _vector(vector) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def _rotation_rad(rotation) -> list[float]:
    return [
        math.radians(float(rotation.roll)),
        math.radians(float(rotation.pitch)),
        math.radians(float(rotation.yaw)),
    ]


def _transform_dict(transform) -> dict:
    return {
        "location_m": _vector(transform.location),
        "rotation_rad": _rotation_rad(transform.rotation),
        "matrix": np.asarray(transform.get_matrix(), dtype=np.float64).tolist(),
    }


def _weather_dict(weather) -> dict:
    names = (
        "cloudiness", "precipitation", "precipitation_deposits",
        "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
        "fog_density", "fog_distance", "fog_falloff", "wetness",
        "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm",
    )
    return {
        name: float(getattr(weather, name))
        for name in names
        if hasattr(weather, name)
    }


def _control_dict(control) -> dict:
    return {
        "steer": float(control.steer),
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "hand_brake": bool(control.hand_brake),
        "reverse": bool(control.reverse),
        "gear": int(control.gear),
    }


def _spawn_vehicle(world, rng: np.random.Generator, blueprint_id: str):
    library = world.get_blueprint_library()
    try:
        blueprint = library.find(blueprint_id)
    except RuntimeError as error:
        raise RuntimeError(
            f"canonical vehicle blueprint unavailable: {blueprint_id}") from error
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("the current map has no vehicle spawn points")
    for index in rng.permutation(len(spawn_points)):
        vehicle = world.try_spawn_actor(blueprint, spawn_points[int(index)])
        if vehicle is not None:
            goal_index = (int(index) + max(1, len(spawn_points) // 2)) % len(
                spawn_points
            )
            return vehicle, spawn_points[goal_index]
    raise RuntimeError("all vehicle spawn points are occupied")


def _spawn_sensor(world, vehicle, blueprint_name, transform, attributes):
    import carla

    blueprint = world.get_blueprint_library().find(blueprint_name)
    for name, value in attributes.items():
        blueprint.set_attribute(name, str(value))
    return world.spawn_actor(
        blueprint,
        transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )


def _drain_collision_events(collision_queue: Queue, frame: int) -> list[dict]:
    events: list[dict] = []
    deferred = []
    while True:
        try:
            event = collision_queue.get_nowait()
        except Empty:
            break
        if int(event.frame) > frame:
            deferred.append(event)
            continue
        impulse = _vector(event.normal_impulse)
        events.append({
            "frame_id": int(event.frame),
            "other_actor_id": int(event.other_actor.id),
            "other_actor_type": str(event.other_actor.type_id),
            "impulse": impulse,
            "impulse_norm": float(np.linalg.norm(impulse)),
        })
    for event in deferred:
        collision_queue.put(event)
    return events


def _update_spectator(world, vehicle) -> None:
    import carla

    vehicle_transform = vehicle.get_transform()
    forward = vehicle_transform.get_forward_vector()
    location = vehicle_transform.location - forward * 7.0
    location.z += 3.0
    rotation = carla.Rotation(
        pitch=-15.0,
        yaw=vehicle_transform.rotation.yaw,
        roll=0.0,
    )
    world.get_spectator().set_transform(carla.Transform(location, rotation))


def _wheel_angles(vehicle) -> tuple[float | None, float | None, bool]:
    try:
        import carla

        left_deg = vehicle.get_wheel_steer_angle(
            carla.VehicleWheelLocation.FL_Wheel
        )
        right_deg = vehicle.get_wheel_steer_angle(
            carla.VehicleWheelLocation.FR_Wheel
        )
        return math.radians(left_deg), math.radians(right_deg), True
    except (AttributeError, RuntimeError):
        return None, None, False


def collect(args: argparse.Namespace) -> Path:
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    stack = load_system_stack(args.config)
    plan = recording_plan_from_stack(stack)

    try:
        import carla
    except ImportError as error:
        raise RuntimeError(
            "CARLA Python API is missing; activate the environment containing "
            "carla==0.9.16"
        ) from error

    output_root = args.output.resolve()
    incomplete_root = output_root / ".incomplete"
    episodes_root = output_root / "episodes"
    incomplete_root.mkdir(parents=True, exist_ok=True)
    episodes_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < 1_000_000_000:
        raise RuntimeError("less than 1 GB free at the output location")

    episode_name = datetime.now(timezone.utc).strftime("episode_%Y%m%dT%H%M%SZ")
    working_dir = incomplete_root / f"{episode_name}_{uuid.uuid4().hex[:8]}"
    final_dir = episodes_root / episode_name
    working_dir.mkdir()
    for relative in (
        "camera_front", "camera_rear", "camera_top",
        "lidar_raw", "lidar_256",
    ):
        (working_dir / relative).mkdir()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    actors = []
    sensors = []
    frames_file = None
    committed = False
    rng = np.random.default_rng(plan["random_seed"])

    def carla_transform(spec):
        location = spec["location_m"]
        rotation = spec["rotation_degrees"]
        return carla.Transform(
            carla.Location(x=location[0], y=location[1], z=location[2]),
            carla.Rotation(
                roll=rotation[0], pitch=rotation[1], yaw=rotation[2]),
        )

    camera_specs = {
        name: {**spec, "transform": carla_transform(spec)}
        for name, spec in plan["camera"].items()
    }
    lidar_transform = carla_transform(plan["lidar"])
    imu_transform = carla_transform(plan["imu"])

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = plan["fixed_delta_seconds"]
        world.apply_settings(settings)
        library = world.get_blueprint_library()
        validate_carla_environment(
            stack.carla_baseline,
            client_version=client.get_client_version(),
            server_version=client.get_server_version(),
            map_name=world.get_map().name,
            available_blueprints=tuple(blueprint.id for blueprint in library),
            fixed_delta_seconds=world.get_settings().fixed_delta_seconds,
            expected_control_period=stack.closed_loop.dt,
        )
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(plan["random_seed"])

        vehicle, goal_transform = _spawn_vehicle(
            world, rng, plan["vehicle_blueprint"])
        actors.append(vehicle)

        camera_queues: dict[str, Queue] = {}
        camera_actors = {}
        for name, spec in camera_specs.items():
            sensor = _spawn_sensor(
                world,
                vehicle,
                spec["blueprint"],
                spec["transform"],
                spec["attributes"],
            )
            sensor_queue: Queue = Queue()
            sensor.listen(sensor_queue.put)
            camera_queues[name] = sensor_queue
            camera_actors[name] = sensor
            sensors.append(sensor)
            actors.append(sensor)

        lidar = _spawn_sensor(
            world,
            vehicle,
            plan["lidar"]["blueprint"],
            lidar_transform,
            plan["lidar"]["attributes"],
        )
        lidar_queue: Queue = Queue()
        lidar.listen(lidar_queue.put)
        sensors.append(lidar)
        actors.append(lidar)

        imu = _spawn_sensor(
            world,
            vehicle,
            plan["imu"]["blueprint"],
            imu_transform,
            plan["imu"]["attributes"],
        )
        imu_queue: Queue = Queue()
        imu.listen(imu_queue.put)
        sensors.append(imu)
        actors.append(imu)

        collision = _spawn_sensor(
            world,
            vehicle,
            "sensor.other.collision",
            carla.Transform(),
            {},
        )
        collision_queue: Queue = Queue()
        collision.listen(collision_queue.put)
        sensors.append(collision)
        actors.append(collision)

        calibration = {
            "schema_version": "new-orad-carla-v1",
            "calibration_version": plan["calibration_version"],
            "carla_sensor_axes": "x-forward,y-right,z-up",
            "new_orad_ego_axes": "x-forward,y-left,z-up",
            "camera": {
                name: {
                    "intrinsic": camera_intrinsic(
                        plan["image_size"][1], plan["image_size"][0],
                        spec["fov"]
                    ).tolist(),
                    "sensor_to_ego_carla": _transform_dict(spec["transform"]),
                    "image_size": list(plan["image_size"]),
                    "fov_deg": spec["fov"],
                }
                for name, spec in camera_specs.items()
            },
            "lidar": {
                "sensor_to_ego_carla": _transform_dict(lidar_transform),
                "canonical_y_flip": True,
                "canonical_points": plan["lidar"]["canonical_points"],
            },
            "imu": {
                "sensor_to_ego_carla": _transform_dict(imu_transform),
                "history_shape": [
                    plan["imu"]["history_steps"], plan["imu"]["channels"]],
            },
        }
        calibration_path = working_dir / "calibration.json"
        _write_json(calibration_path, calibration)

        vehicle.set_autopilot(True, traffic_manager.get_port())
        imu_samples: deque = deque(maxlen=plan["imu"]["history_steps"])
        for _ in range(args.warmup_frames):
            frame = world.tick()
            for name, sensor_queue in camera_queues.items():
                get_exact_frame(
                    sensor_queue, frame, args.timeout, f"camera_{name}"
                )
            get_exact_frame(lidar_queue, frame, args.timeout, "lidar")
            imu_data = get_exact_frame(
                imu_queue, frame, args.timeout, "imu"
            )
            imu_samples.append((
                int(imu_data.frame),
                float(imu_data.timestamp),
                np.array(
                    [imu_data.accelerometer.x, imu_data.accelerometer.y,
                     imu_data.accelerometer.z, imu_data.gyroscope.x,
                     imu_data.gyroscope.y, imu_data.gyroscope.z],
                    dtype=np.float32,
                ),
            ))

        frames_file = (working_dir / "frames.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        start_time = time.monotonic()
        start_frame = None
        end_frame = None
        collision_count = 0

        for sample_index in range(args.frames):
            frame = world.tick()
            start_frame = frame if start_frame is None else start_frame
            end_frame = frame

            camera_data = {
                name: get_exact_frame(
                    sensor_queue, frame, args.timeout, f"camera_{name}"
                )
                for name, sensor_queue in camera_queues.items()
            }
            lidar_data = get_exact_frame(
                lidar_queue, frame, args.timeout, "lidar"
            )
            imu_data = get_exact_frame(
                imu_queue, frame, args.timeout, "imu"
            )
            imu_row = np.array(
                [imu_data.accelerometer.x, imu_data.accelerometer.y,
                 imu_data.accelerometer.z, imu_data.gyroscope.x,
                 imu_data.gyroscope.y, imu_data.gyroscope.z],
                dtype=np.float32,
            )
            imu_samples.append(
                (int(imu_data.frame), float(imu_data.timestamp), imu_row)
            )
            imu_history, imu_frames, imu_timestamps, imu_valid = build_imu_history(
                imu_samples, steps=plan["imu"]["history_steps"]
            )

            file_stem = f"{frame:08d}"
            camera_paths = {}
            for name, image in camera_data.items():
                relative = Path(f"camera_{name}") / f"{file_stem}.png"
                image.save_to_disk(str(working_dir / relative))
                camera_paths[name] = relative.as_posix()

            raw_lidar = np.frombuffer(
                lidar_data.raw_data, dtype=np.float32
            ).reshape((-1, 4)).copy()
            canonical_seed = (int(plan["random_seed"]) << 32) ^ int(frame)
            canonical_lidar, lidar_meta = canonicalize_lidar(
                raw_lidar, count=plan["lidar"]["canonical_points"],
                seed=canonical_seed
            )
            raw_relative = Path("lidar_raw") / f"{file_stem}.npy"
            canonical_relative = Path("lidar_256") / f"{file_stem}.npy"
            np.save(working_dir / raw_relative, raw_lidar)
            np.save(working_dir / canonical_relative, canonical_lidar)

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            acceleration = vehicle.get_acceleration()
            angular_velocity = vehicle.get_angular_velocity()
            speed = math.sqrt(
                velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
            )
            wheel_left, wheel_right, wheel_available = _wheel_angles(vehicle)
            control = vehicle.get_control()
            action = _control_dict(control)
            collision_events = _drain_collision_events(collision_queue, frame)
            collision_count += len(collision_events)
            distance_to_goal = float(
                transform.location.distance(goal_transform.location)
            )

            sensor_timestamps = [
                *(float(value.timestamp) for value in camera_data.values()),
                float(lidar_data.timestamp),
                float(imu_data.timestamp),
            ]
            sensor_skew = max(sensor_timestamps) - min(sensor_timestamps)
            record = {
                "sample_index": sample_index,
                "frame_id": int(frame),
                "timestamp": float(lidar_data.timestamp),
                "sensors": {
                    "camera": {
                        name: {
                            "frame_id": int(value.frame),
                            "timestamp": float(value.timestamp),
                            "path": camera_paths[name],
                        }
                        for name, value in camera_data.items()
                    },
                    "lidar": {
                        "frame_id": int(lidar_data.frame),
                        "timestamp": float(lidar_data.timestamp),
                        "raw_path": raw_relative.as_posix(),
                        "canonical_path": canonical_relative.as_posix(),
                        **lidar_meta,
                    },
                    "imu": {
                        "frame_id": int(imu_data.frame),
                        "timestamp": float(imu_data.timestamp),
                        "history": imu_history.tolist(),
                        "history_frame_ids": imu_frames,
                        "history_timestamps": imu_timestamps,
                        "valid_count": imu_valid,
                    },
                    "timestamp_skew_seconds": float(sensor_skew),
                },
                "ego_state": {
                    "position_world_m": _vector(transform.location),
                    "rotation_world_rad": _rotation_rad(transform.rotation),
                    "velocity_world_mps": _vector(velocity),
                    "speed_mps": float(speed),
                    "angular_velocity_world_radps": [
                        math.radians(value) for value in _vector(angular_velocity)
                    ],
                    "acceleration_world_mps2": _vector(acceleration),
                    "steer_normalized": float(control.steer),
                    "front_left_wheel_steer_rad": wheel_left,
                    "front_right_wheel_steer_rad": wheel_right,
                    "wheel_steer_available": wheel_available,
                },
                "action_raw": action,
                "action_applied": action,
                "action_source": "traffic_manager_smoke",
                "expert_label": False,
                "event": {
                    "collision": bool(collision_events),
                    "collision_events": collision_events,
                    "reached_goal": distance_to_goal <= 2.0,
                    "distance_to_goal_m": distance_to_goal,
                    "emergency_stop": False,
                    "safety_intervention": False,
                    "safety_reason": None,
                },
            }
            frames_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (sample_index + 1) % 10 == 0:
                frames_file.flush()
            _update_spectator(world, vehicle)
            print(
                f"\r采集 {sample_index + 1:4d}/{args.frames} | "
                f"frame={frame} | speed={speed * 3.6:5.1f} km/h | "
                f"lidar={raw_lidar.shape[0]:5d}",
                end="",
                flush=True,
            )

        frames_file.flush()
        frames_file.close()
        frames_file = None
        manifest = {
            "schema_version": "new-orad-carla-v1",
            "status": "complete",
            "action_source": "traffic_manager_smoke",
            "expert_labels": False,
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": world.get_map().name,
            "weather": _weather_dict(world.get_weather()),
            "fixed_delta_seconds": float(plan["fixed_delta_seconds"]),
            "random_seed": int(plan["random_seed"]),
            "vehicle_blueprint": str(vehicle.type_id),
            "calibration_version": plan["calibration_version"],
            "calibration_sha256": sha256_file(calibration_path),
            "config_sha256": sha256_file(args.config),
            "config_canonical_sha256": canonical_yaml_sha256(args.config),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "sample_count": int(args.frames),
            "collision_count": int(collision_count),
            "goal_world_m": _vector(goal_transform.location),
            "wall_time_seconds": float(time.monotonic() - start_time),
        }
        _write_json(working_dir / "episode.json", manifest)
        (working_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        if final_dir.exists():
            final_dir = episodes_root / f"{episode_name}_{uuid.uuid4().hex[:8]}"
        working_dir.replace(final_dir)
        committed = True
        print(f"\n完成：{final_dir}")
        return final_dir
    finally:
        if frames_file is not None:
            frames_file.close()
        if actors:
            vehicle = actors[0]
            try:
                vehicle.set_autopilot(False, traffic_manager.get_port())
                vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                )
            except RuntimeError:
                pass
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        for actor in reversed(actors):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        try:
            traffic_manager.set_synchronous_mode(False)
        finally:
            world.apply_settings(original_settings)
        if not committed:
            print(f"\n采集未提交，诊断文件保留在：{working_dir}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        collect(args)
    except KeyboardInterrupt:
        print("\n用户中止采集。")
        return 130
    except Exception as error:
        print(f"\n采集失败：{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
