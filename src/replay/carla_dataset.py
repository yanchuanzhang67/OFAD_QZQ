"""Strict read-only loader for synchronized CARLA episode directories."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

from sim.carla_baseline import CarlaTransformConfig, sha256_file
from utils.types import VehicleState

if TYPE_CHECKING:
    from configuration.system import SystemStackConfig


_SCHEMA_VERSION = "new-orad-carla-v1"
_CAMERA_NAMES = ("front", "rear", "top")
_PROVENANCE_FIELDS = (
    "vehicle_blueprint", "config_sha256", "calibration_sha256")


class DatasetContractError(RuntimeError):
    """A stable code plus evidence describing a recorded-data violation."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class RecordedCarlaFrame:
    sample_index: int
    frame_id: int
    timestamp: float
    images: tuple[np.ndarray, ...]
    raw_point_cloud: np.ndarray
    canonical_point_cloud: np.ndarray
    imu_history: np.ndarray
    ego_state: VehicleState
    camera_frames: tuple[int, ...]
    lidar_frame: int
    imu_frames: tuple[int, ...]
    camera_timestamps: tuple[float, ...]
    lidar_timestamp: float
    imu_timestamps: tuple[float, ...]
    sensor_skew_seconds: float
    action_source: str
    expert_label: bool


def _error(code: str, message: str):
    raise DatasetContractError(code, message)


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _error(code, f"cannot read {path.name}: {error}")
    if not isinstance(value, dict):
        _error(code, f"{path.name} must contain a JSON object")
    return value


def _map_basename(value: Any) -> str:
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def _require_finite(value: Any, *, code: str, message: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        _error(code, message)
    if not np.isfinite(result):
        _error(code, message)
    return result


class CarlaRecordedEpisode:
    """Validated metadata index with lazy, copy-on-read sensor decoding."""

    def __init__(
            self, path: Path, stack: "SystemStackConfig",
            manifest: Mapping[str, Any], calibration: Mapping[str, Any],
            records: tuple[Mapping[str, Any], ...],
            provenance_gaps: tuple[str, ...], calibration_sha256: str,
            episode_sha256: str):
        self.path = path
        self.stack = stack
        self.manifest = dict(manifest)
        self.calibration = dict(calibration)
        self._records = records
        self.provenance_gaps = provenance_gaps
        self.calibration_sha256 = calibration_sha256
        self.episode_sha256 = episode_sha256

    @classmethod
    def open(
            cls, path: Path, stack: "SystemStackConfig", *,
            allow_legacy_provenance: bool = False) -> "CarlaRecordedEpisode":
        root = Path(path).resolve()
        if not (root / "_SUCCESS").is_file():
            _error("episode_incomplete", f"{root} has no _SUCCESS marker")
        for name in ("episode.json", "calibration.json", "frames.jsonl"):
            if not (root / name).is_file():
                _error("metadata_missing", f"{root} is missing {name}")

        manifest = _read_json(
            root / "episode.json", code="manifest_invalid")
        calibration = _read_json(
            root / "calibration.json", code="calibration_invalid")
        cls._validate_manifest(manifest, stack)
        calibration_hash = sha256_file(root / "calibration.json")
        cls._validate_calibration(calibration, stack)

        recorded_calibration_hash = manifest.get("calibration_sha256")
        if (recorded_calibration_hash is not None
                and str(recorded_calibration_hash) != calibration_hash):
            _error(
                "calibration_hash_mismatch",
                "recorded calibration_sha256 does not match calibration.json")
        recorded_canonical_hash = manifest.get("config_canonical_sha256")
        recorded_config_hash = manifest.get("config_sha256")
        if recorded_canonical_hash is not None:
            if str(recorded_canonical_hash) != stack.config_canonical_sha256:
                _error(
                    "config_hash_mismatch",
                    "recorded canonical config differs from current system.yaml")
        elif (recorded_config_hash is not None
              and str(recorded_config_hash) != stack.config_sha256):
            _error(
                "config_hash_mismatch",
                "recorded config_sha256 differs from current system.yaml")

        gaps = tuple(
            field for field in _PROVENANCE_FIELDS
            if manifest.get(field) in (None, ""))
        if gaps and not allow_legacy_provenance:
            _error(
                "legacy_provenance_requires_opt_in",
                f"manifest is missing provenance fields: {list(gaps)}")

        records = cls._read_records(root / "frames.jsonl")
        if int(manifest.get("sample_count", -1)) != len(records):
            _error(
                "sample_count_mismatch",
                "episode sample_count does not match frames.jsonl")
        cls._validate_record_index(root, records)
        episode_hash = cls._episode_digest(root)
        return cls(
            root, stack, manifest, calibration, records, gaps,
            calibration_hash, episode_hash)

    @staticmethod
    def _validate_manifest(
            manifest: Mapping[str, Any], stack: "SystemStackConfig") -> None:
        baseline = stack.carla_baseline
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            _error("schema_mismatch", "unsupported episode schema_version")
        if manifest.get("status") != "complete":
            _error("episode_incomplete", "manifest status is not complete")
        if (str(manifest.get("carla_client_version")) != baseline.version
                or str(manifest.get("carla_server_version")) != baseline.version):
            _error(
                "carla_version_mismatch",
                "recorded client/server version differs from Canonical")
        if _map_basename(manifest.get("map")) != baseline.map_name:
            _error("map_mismatch", "recorded map differs from Canonical")
        if int(manifest.get("random_seed", -1)) != baseline.random_seed:
            _error("random_seed_mismatch", "recorded seed differs from Canonical")
        delta = _require_finite(
            manifest.get("fixed_delta_seconds"), code="control_period_mismatch",
            message="fixed_delta_seconds must be finite")
        if abs(delta - stack.closed_loop.dt) > 1e-9:
            _error(
                "control_period_mismatch",
                "recorded fixed delta differs from Canonical")
        blueprint = manifest.get("vehicle_blueprint")
        if blueprint is not None and str(blueprint) != baseline.vehicle_blueprint:
            _error(
                "vehicle_blueprint_mismatch",
                "recorded vehicle differs from Canonical")

    @staticmethod
    def _validate_calibration(
            calibration: Mapping[str, Any],
            stack: "SystemStackConfig") -> None:
        if calibration.get("schema_version") != _SCHEMA_VERSION:
            _error("calibration_schema_mismatch", "calibration schema differs")
        if (calibration.get("calibration_version")
                != stack.sensor_health.expected_calibration_version):
            _error(
                "calibration_version_mismatch",
                "calibration version differs from Canonical")
        if (calibration.get("carla_sensor_axes")
                != "x-forward,y-right,z-up"
                or calibration.get("new_orad_ego_axes")
                != "x-forward,y-left,z-up"):
            _error("calibration_axes_mismatch", "coordinate axes are not canonical")
        cameras = calibration.get("camera")
        if not isinstance(cameras, dict) or tuple(cameras) != _CAMERA_NAMES:
            _error(
                "camera_order_mismatch",
                "calibration cameras must be ordered front/rear/top")
        for camera in stack.carla_baseline.cameras:
            recorded = cameras[camera.name]
            if tuple(recorded.get("image_size", ())) != stack.bev.image_size:
                _error(
                    "camera_shape_mismatch",
                    f"camera {camera.name} image_size differs")
            if abs(float(recorded.get("fov_deg", -1)) - camera.fov_degrees) > 1e-6:
                _error(
                    "camera_fov_mismatch",
                    f"camera {camera.name} FOV differs")
            CarlaRecordedEpisode._validate_transform(
                recorded.get("sensor_to_ego_carla"), camera.transform,
                sensor_name=f"camera {camera.name}",
                code="camera_transform_mismatch")
        lidar = calibration.get("lidar", {})
        if (int(lidar.get("canonical_points", -1)) != stack.bev.num_points
                or lidar.get("canonical_y_flip") is not True):
            _error("lidar_calibration_mismatch", "LiDAR canonicalization differs")
        imu_shape = tuple(calibration.get("imu", {}).get("history_shape", ()))
        if imu_shape != (stack.bev.imu_steps, stack.bev.imu_in_channels):
            _error("imu_calibration_mismatch", "IMU history shape differs")

    @staticmethod
    def _validate_transform(
            recorded: Any, expected: CarlaTransformConfig, *,
            sensor_name: str, code: str) -> None:
        if not isinstance(recorded, Mapping):
            _error(code, f"{sensor_name} transform is missing")
        try:
            location = np.asarray(recorded.get("location_m"), dtype=np.float64)
            rotation = np.asarray(recorded.get("rotation_rad"), dtype=np.float64)
        except (TypeError, ValueError):
            _error(code, f"{sensor_name} transform is malformed")
        expected_rotation = np.radians(expected.rotation_degrees)
        if (location.shape != (3,) or rotation.shape != (3,)
                or not np.isfinite(location).all()
                or not np.isfinite(rotation).all()
                or not np.allclose(location, expected.location_m, atol=1e-6)
                or not np.allclose(rotation, expected_rotation, atol=1e-6)):
            _error(code, f"{sensor_name} transform differs from Canonical")

    @staticmethod
    def _read_records(path: Path) -> tuple[Mapping[str, Any], ...]:
        records = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if line.strip():
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            _error(
                                "frame_record_invalid",
                                f"line {line_number} is not a JSON object")
                        records.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            _error("frames_jsonl_invalid", str(error))
        if not records:
            _error("frames_jsonl_empty", "frames.jsonl contains no records")
        return tuple(records)

    @classmethod
    def _validate_record_index(
            cls, root: Path,
            records: tuple[Mapping[str, Any], ...]) -> None:
        frames = [int(record.get("frame_id", -1)) for record in records]
        timestamps = [
            _require_finite(
                record.get("timestamp"), code="timestamp_order_invalid",
                message="record timestamp must be finite")
            for record in records
        ]
        if (frames != sorted(set(frames))
                or any(right <= left for left, right in zip(frames, frames[1:]))):
            _error("frame_order_invalid", "frame ids must strictly increase")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            _error("timestamp_order_invalid", "timestamps must strictly increase")
        for record in records:
            frame = int(record["frame_id"])
            sensors = record.get("sensors", {})
            cameras = sensors.get("camera", {})
            if tuple(cameras) != _CAMERA_NAMES:
                _error("camera_order_mismatch", f"frame {frame} camera order differs")
            for name in _CAMERA_NAMES:
                camera = cameras[name]
                if int(camera.get("frame_id", -1)) != frame:
                    _error("sensor_frame_mismatch", f"frame {frame} camera {name}")
                cls._resolve_sensor_path(root, camera.get("path"), frame)
            lidar = sensors.get("lidar", {})
            imu = sensors.get("imu", {})
            if (int(lidar.get("frame_id", -1)) != frame
                    or int(imu.get("frame_id", -1)) != frame):
                _error("sensor_frame_mismatch", f"frame {frame} LiDAR/IMU")
            cls._resolve_sensor_path(root, lidar.get("raw_path"), frame)
            cls._resolve_sensor_path(root, lidar.get("canonical_path"), frame)

    @staticmethod
    def _resolve_sensor_path(root: Path, relative: Any, frame: int) -> Path:
        value = Path(str(relative))
        if value.is_absolute():
            _error("unsafe_sensor_path", f"frame {frame} uses an absolute path")
        resolved = (root / value).resolve()
        if resolved != root and root not in resolved.parents:
            _error("unsafe_sensor_path", f"frame {frame} escapes episode root")
        if not resolved.is_file():
            _error("sensor_file_missing", f"frame {frame} is missing {value}")
        return resolved

    @staticmethod
    def _episode_digest(root: Path) -> str:
        digest = hashlib.sha256()
        for name in ("episode.json", "calibration.json", "frames.jsonl"):
            digest.update(name.encode("utf-8"))
            digest.update((root / name).read_bytes())
        return digest.hexdigest()

    def iter_frames(
            self, max_frames: Optional[int] = None) -> Iterator[RecordedCarlaFrame]:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        records = self._records if max_frames is None else self._records[:max_frames]
        for record in records:
            yield self._decode_frame(record)

    def _decode_frame(self, record: Mapping[str, Any]) -> RecordedCarlaFrame:
        frame = int(record["frame_id"])
        sensors = record["sensors"]
        cameras = sensors["camera"]
        images = tuple(
            self._decode_image(
                self._resolve_sensor_path(self.path, cameras[name]["path"], frame),
                frame, name)
            for name in _CAMERA_NAMES
        )
        lidar = sensors["lidar"]
        raw = self._decode_lidar(lidar["raw_path"], frame, canonical=False)
        canonical = self._decode_lidar(
            lidar["canonical_path"], frame, canonical=True)
        imu = sensors["imu"]
        history = np.asarray(imu.get("history"), dtype=np.float32)
        if history.shape != (self.stack.bev.imu_steps, self.stack.bev.imu_in_channels):
            _error("imu_shape_mismatch", f"frame {frame} IMU history shape differs")
        if not np.isfinite(history).all():
            _error("imu_nonfinite", f"frame {frame} IMU contains NaN/Inf")
        imu_frames = tuple(int(value) for value in imu.get("history_frame_ids", ()))
        imu_timestamps = tuple(
            _require_finite(
                value, code="imu_timestamp_invalid",
                message=f"frame {frame} IMU timestamp is invalid")
            for value in imu.get("history_timestamps", ())
        )
        if (len(imu_frames) != self.stack.bev.imu_steps
                or len(imu_timestamps) != self.stack.bev.imu_steps):
            _error("imu_timing_shape_mismatch", f"frame {frame} IMU timing differs")
        state = self._vehicle_state(record.get("ego_state", {}), frame)
        return RecordedCarlaFrame(
            sample_index=int(record.get("sample_index", -1)),
            frame_id=frame,
            timestamp=float(record["timestamp"]),
            images=images,
            raw_point_cloud=raw,
            canonical_point_cloud=canonical,
            imu_history=np.ascontiguousarray(history),
            ego_state=state,
            camera_frames=tuple(
                int(cameras[name]["frame_id"]) for name in _CAMERA_NAMES),
            lidar_frame=int(lidar["frame_id"]),
            imu_frames=imu_frames,
            camera_timestamps=tuple(
                float(cameras[name]["timestamp"]) for name in _CAMERA_NAMES),
            lidar_timestamp=float(lidar["timestamp"]),
            imu_timestamps=imu_timestamps,
            sensor_skew_seconds=float(sensors["timestamp_skew_seconds"]),
            action_source=str(record.get("action_source", "unknown")),
            expert_label=bool(record.get("expert_label", False)),
        )

    def _decode_image(self, path: Path, frame: int, name: str) -> np.ndarray:
        try:
            with Image.open(path) as source:
                image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        except (OSError, ValueError) as error:
            _error("camera_decode_failed", f"frame {frame} camera {name}: {error}")
        expected = (*self.stack.bev.image_size, 3)
        if image.shape != expected:
            _error("camera_shape_mismatch", f"frame {frame} camera {name}")
        return np.ascontiguousarray(image)

    def _decode_lidar(
            self, relative: Any, frame: int, *, canonical: bool) -> np.ndarray:
        path = self._resolve_sensor_path(self.path, relative, frame)
        try:
            points = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            _error("lidar_decode_failed", f"frame {frame}: {error}")
        if points.dtype != np.float32:
            _error("lidar_dtype_mismatch", f"frame {frame} LiDAR is not float32")
        if points.ndim != 2 or points.shape[1] != 4:
            _error("lidar_shape_mismatch", f"frame {frame} LiDAR must be (N,4)")
        if canonical and points.shape != (self.stack.bev.num_points, 4):
            _error("lidar_shape_mismatch", f"frame {frame} canonical LiDAR")
        if not np.isfinite(points).all():
            _error("lidar_nonfinite", f"frame {frame} LiDAR contains NaN/Inf")
        return np.ascontiguousarray(points.copy())

    def _vehicle_state(
            self, raw: Mapping[str, Any], frame: int) -> VehicleState:
        try:
            position = np.asarray(raw["position_world_m"], dtype=np.float64)
            rotation = np.asarray(raw["rotation_world_rad"], dtype=np.float64)
            velocity = np.asarray(raw["velocity_world_mps"], dtype=np.float64)
            angular = np.asarray(
                raw["angular_velocity_world_radps"], dtype=np.float64)
            acceleration = np.asarray(
                raw["acceleration_world_mps2"], dtype=np.float64)
            values = np.concatenate(
                [position, rotation, velocity, angular, acceleration])
        except (KeyError, TypeError, ValueError) as error:
            _error("ego_state_invalid", f"frame {frame}: {error}")
        if (position.shape != (3,) or rotation.shape != (3,)
                or velocity.shape != (3,) or angular.shape != (3,)
                or acceleration.shape != (3,) or not np.isfinite(values).all()):
            _error("ego_state_invalid", f"frame {frame} ego vectors are invalid")
        yaw = float(rotation[2])
        cy, sy = np.cos(yaw), np.sin(yaw)
        vx_body = cy * velocity[0] + sy * velocity[1]
        vy_body = -sy * velocity[0] + cy * velocity[1]
        speed = _require_finite(
            raw.get("speed_mps"), code="ego_state_invalid",
            message=f"frame {frame} speed is invalid")
        steer = _require_finite(
            raw.get("steer_normalized"), code="ego_state_invalid",
            message=f"frame {frame} steering is invalid")
        return VehicleState(
            x=float(position[0]), y=float(position[1]), yaw=yaw,
            speed=speed, steering=steer * self.stack.safety.max_steering,
            pitch=float(rotation[1]), roll=float(rotation[0]),
            vx=float(vx_body), vy=float(vy_body),
            yaw_rate=float(angular[2]), accel_z=float(acceleration[2]),
        )
