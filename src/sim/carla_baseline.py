"""Reproducible CARLA environment baseline contracts and evidence helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


@dataclass(frozen=True)
class CarlaTransformConfig:
    """CARLA sensor pose relative to the ego actor."""

    location_m: tuple[float, float, float]
    rotation_degrees: tuple[float, float, float]

    def __post_init__(self) -> None:
        values = (*self.location_m, *self.rotation_degrees)
        if len(self.location_m) != 3 or len(self.rotation_degrees) != 3:
            raise ValueError("CARLA transform vectors must contain three values")
        if not np.isfinite(values).all():
            raise ValueError("CARLA transform values must be finite")


@dataclass(frozen=True)
class CarlaCameraConfig:
    name: str
    blueprint: str
    transform: CarlaTransformConfig
    fov_degrees: float
    sensor_tick_seconds: float

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.blueprint.strip():
            raise ValueError("camera name and blueprint must be non-empty")
        if not 0.0 < self.fov_degrees < 180.0:
            raise ValueError("camera fov_degrees must be in (0, 180)")
        if self.sensor_tick_seconds <= 0.0:
            raise ValueError("camera sensor_tick_seconds must be positive")


@dataclass(frozen=True)
class CarlaLidarConfig:
    blueprint: str
    transform: CarlaTransformConfig
    channels: int
    range_meters: float
    points_per_second: int
    rotation_frequency_hz: float
    upper_fov_degrees: float
    lower_fov_degrees: float
    sensor_tick_seconds: float
    canonical_points: int
    canonical_y_flip: bool

    def __post_init__(self) -> None:
        if not self.blueprint.strip():
            raise ValueError("lidar blueprint must be non-empty")
        for name in (
                "channels", "range_meters", "points_per_second",
                "rotation_frequency_hz", "sensor_tick_seconds",
                "canonical_points"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"lidar {name} must be positive")
        if self.lower_fov_degrees >= self.upper_fov_degrees:
            raise ValueError(
                "lidar lower_fov_degrees must be below upper_fov_degrees")
        if not self.canonical_y_flip:
            raise ValueError("lidar canonical_y_flip must be true")


@dataclass(frozen=True)
class CarlaImuConfig:
    blueprint: str
    transform: CarlaTransformConfig
    sensor_tick_seconds: float
    history_steps: int
    channels: int

    def __post_init__(self) -> None:
        if not self.blueprint.strip():
            raise ValueError("imu blueprint must be non-empty")
        if self.sensor_tick_seconds <= 0.0:
            raise ValueError("imu sensor_tick_seconds must be positive")
        if self.history_steps <= 0 or self.channels != 6:
            raise ValueError("imu history_steps must be positive and channels must be 6")


@dataclass(frozen=True)
class CarlaBaselineConfig:
    """Immutable CARLA environment choices owned by ``system.yaml``."""

    version: str
    map_name: str
    vehicle_blueprint: str
    random_seed: int
    cameras: tuple[CarlaCameraConfig, ...]
    lidar: CarlaLidarConfig
    imu: CarlaImuConfig
    repetitions: int = 3

    def __post_init__(self) -> None:
        for field_name in ("version", "map_name", "vehicle_blueprint"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.repetitions != 3:
            raise ValueError("repetitions must be exactly 3")
        camera_names = tuple(camera.name for camera in self.cameras)
        if camera_names != ("front", "rear", "top"):
            raise ValueError(
                "camera order must be exactly [front, rear, top]")


def _carla_transform_matrix(transform: CarlaTransformConfig) -> np.ndarray:
    roll, pitch, yaw = np.radians(transform.rotation_degrees)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrix = np.array([
        [cy * cp, cy * sp * sr - sy * cr,
         -cy * sp * cr - sy * sr, transform.location_m[0]],
        [sy * cp, sy * sp * sr + cy * cr,
         -sy * sp * cr + cy * sr, transform.location_m[1]],
        [sp, -cp * sr, cp * cr, transform.location_m[2]],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return matrix


def camera_intrinsic_matrix(
        camera: CarlaCameraConfig,
        image_size: tuple[int, int]) -> np.ndarray:
    height, width = (int(value) for value in image_size)
    focal = width / (2.0 * np.tan(np.radians(camera.fov_degrees) / 2.0))
    return np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def camera_extrinsic_new_orad(camera: CarlaCameraConfig) -> np.ndarray:
    """Return optical-camera to New_ORAD ego axes as a homogeneous matrix."""
    optical_to_carla = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    carla_ego_to_new_orad = np.diag([1.0, -1.0, 1.0, 1.0]).astype(
        np.float32)
    return (carla_ego_to_new_orad
            @ _carla_transform_matrix(camera.transform)
            @ optical_to_carla)


def canonical_yaml_sha256(path: Path) -> str:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BaselineMismatchError(ValueError):
    """Raised when the running simulator differs from the fixed baseline."""


@dataclass(frozen=True)
class SensorTimingSample:
    frame: int
    camera_frames: tuple[int, ...]
    lidar_frame: int
    imu_frames: tuple[int, ...]
    camera_timestamps: tuple[float, ...]
    lidar_timestamp: float
    imu_timestamps: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "camera_frames": list(self.camera_frames),
            "lidar_frame": self.lidar_frame,
            "imu_frames": list(self.imu_frames),
            "camera_timestamps": list(self.camera_timestamps),
            "lidar_timestamp": self.lidar_timestamp,
            "imu_timestamps": list(self.imu_timestamps),
        }


def capture_sensor_timing(sensors: Any) -> SensorTimingSample:
    return SensorTimingSample(
        frame=int(sensors.last_frame),
        camera_frames=tuple(int(value)
                            for value in sensors.last_camera_frames),
        lidar_frame=int(sensors.last_lidar_frame),
        imu_frames=tuple(int(value) for value in sensors.last_imu_frames),
        camera_timestamps=tuple(
            float(value) for value in sensors.last_camera_timestamps),
        lidar_timestamp=float(sensors.last_lidar_timestamp),
        imu_timestamps=tuple(
            float(value) for value in sensors.last_imu_timestamps),
    )


def carla_sensor_attributes(
        baseline: CarlaBaselineConfig,
        image_size: tuple[int, int]) -> dict[str, Any]:
    height, width = (int(value) for value in image_size)
    return {
        "cameras": {
            camera.name: {
                "image_size_x": str(width),
                "image_size_y": str(height),
                "fov": str(float(camera.fov_degrees)),
                "sensor_tick": str(float(camera.sensor_tick_seconds)),
            }
            for camera in baseline.cameras
        },
        "lidar": {
            "channels": str(int(baseline.lidar.channels)),
            "range": str(float(baseline.lidar.range_meters)),
            "points_per_second": str(int(baseline.lidar.points_per_second)),
            "rotation_frequency": str(
                float(baseline.lidar.rotation_frequency_hz)),
            "upper_fov": str(float(baseline.lidar.upper_fov_degrees)),
            "lower_fov": str(float(baseline.lidar.lower_fov_degrees)),
            "sensor_tick": str(float(baseline.lidar.sensor_tick_seconds)),
        },
        "imu": {"sensor_tick": str(float(baseline.imu.sensor_tick_seconds))},
    }


def resolved_baseline_values(
        baseline: CarlaBaselineConfig, *, calibration_version: str,
        control_period_seconds: float, num_cameras: int,
        image_size: tuple[int, int]) -> dict[str, Any]:
    return {
        "version": baseline.version,
        "map_name": baseline.map_name,
        "vehicle_blueprint": baseline.vehicle_blueprint,
        "random_seed": baseline.random_seed,
        "control_period_seconds": float(control_period_seconds),
        "calibration_version": str(calibration_version),
        "num_cameras": int(num_cameras),
        "image_size": [int(value) for value in image_size],
        "sensor_attributes": carla_sensor_attributes(baseline, image_size),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_git_metadata(repository: Path) -> dict[str, Any]:
    root = Path(repository).resolve()

    def run(*args: str) -> str:
        result = subprocess.run(
            ("git", "-C", str(root), *args), check=True,
            capture_output=True, text=True)
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=all")
    return {"commit": commit, "dirty": bool(status)}


def validate_carla_environment(
        baseline: CarlaBaselineConfig, *, client_version: str,
        server_version: str, map_name: str,
        available_blueprints: Sequence[str], fixed_delta_seconds: float,
        expected_control_period: float) -> None:
    actual_map = str(map_name).rstrip("/").rsplit("/", 1)[-1]
    mismatches = []
    if str(client_version) != baseline.version:
        mismatches.append(
            f"client version {client_version!r} != {baseline.version!r}")
    if str(server_version) != baseline.version:
        mismatches.append(
            f"server version {server_version!r} != {baseline.version!r}")
    if actual_map != baseline.map_name:
        mismatches.append(f"map {actual_map!r} != {baseline.map_name!r}")
    if baseline.vehicle_blueprint not in set(available_blueprints):
        mismatches.append(
            f"vehicle blueprint {baseline.vehicle_blueprint!r} unavailable")
    if abs(float(fixed_delta_seconds) - float(expected_control_period)) > 1e-9:
        mismatches.append(
            f"fixed delta {fixed_delta_seconds!r} != control period "
            f"{expected_control_period!r}")
    if mismatches:
        raise BaselineMismatchError("; ".join(mismatches))


def _timing_signature(samples: Sequence[SensorTimingSample]) -> tuple:
    if not samples:
        return ()
    first_frame = samples[0].frame
    first_timestamp = samples[0].lidar_timestamp

    def relative_timestamp(value: float) -> float:
        return round(float(value) - first_timestamp, 9)

    return tuple((
        sample.frame - first_frame,
        tuple(frame - first_frame for frame in sample.camera_frames),
        sample.lidar_frame - first_frame,
        tuple(frame - first_frame for frame in sample.imu_frames),
        tuple(relative_timestamp(value)
              for value in sample.camera_timestamps),
        relative_timestamp(sample.lidar_timestamp),
        tuple(relative_timestamp(value) for value in sample.imu_timestamps),
    ) for sample in samples)


def compare_repetition_records(records: Sequence[Mapping[str, Any]], *,
                               expected_repetitions: int) -> dict[str, Any]:
    mismatches = []
    if len(records) != expected_repetitions:
        mismatches.append(
            f"expected {expected_repetitions} repetitions, got {len(records)}")
    if not records:
        return {
            "reproducible": False,
            "repetitions": 0,
            "config_consistent": False,
            "sensor_timing_consistent": False,
            "mismatches": mismatches,
        }

    reference_config = records[0]["resolved_baseline"]
    reference_timing = _timing_signature(records[0]["sensor_timing"])
    config_consistent = True
    timing_consistent = True
    for index, record in enumerate(records[1:], start=2):
        if record["resolved_baseline"] != reference_config:
            config_consistent = False
            mismatches.append(
                f"repetition {index} resolved baseline differs from repetition 1")
        if _timing_signature(record["sensor_timing"]) != reference_timing:
            timing_consistent = False
            mismatches.append(
                f"repetition {index} sensor timing differs from repetition 1")
    count_matches = len(records) == expected_repetitions
    return {
        "reproducible": (
            count_matches and config_consistent and timing_consistent),
        "repetitions": len(records),
        "config_consistent": config_consistent and count_matches,
        "sensor_timing_consistent": timing_consistent and count_matches,
        "mismatches": mismatches,
    }


def build_baseline_manifest(
        *, baseline: CarlaBaselineConfig, config_path: Path,
        calibration_version: str, control_period_seconds: float,
        git_metadata: Mapping[str, Any], client_version: str,
        server_version: str, observed_map_name: str) -> dict[str, Any]:
    map_name = str(observed_map_name).rstrip("/").rsplit("/", 1)[-1]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": dict(git_metadata),
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": sha256_file(config_path),
            "canonical_sha256": canonical_yaml_sha256(config_path),
        },
        "carla": {
            "expected_version": baseline.version,
            "client_version": str(client_version),
            "server_version": str(server_version),
            "map_name": map_name,
            "vehicle_blueprint": baseline.vehicle_blueprint,
        },
        "sensors": {"calibration_version": str(calibration_version)},
        "control_period_seconds": float(control_period_seconds),
        "random_seed": baseline.random_seed,
        "repetitions": baseline.repetitions,
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
