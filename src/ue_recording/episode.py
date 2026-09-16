"""Strict read-only consumer for frozen ofad-ue-episode-v1.

Format constants describe the versioned on-disk protocol, not runtime model
configuration. No CARLA, Torch, BC weights, padding, or steering conversions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image

from utils.types import RecordedUEFrame, ScoutDynamicsV1


CAMERAS = ("front", "rear", "top")
OPTICAL_TO_SENSOR = np.array([
    [0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


class UERecordingError(ValueError):
    """An incomplete, inconsistent, or malformed recorded episode."""


def require(condition, message):
    if not condition:
        raise UERecordingError(message)


def numbers(value, shape, label):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in "iuf", f"invalid {label} shape/type")
    require(np.isfinite(result).all(), f"non-finite {label}")
    require(np.abs(result).max(initial=0) <= np.finfo(np.float32).max, f"overflow {label}")
    return result.astype(np.float64)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {label}")
    return value


def close(a, b):
    return np.allclose(a, b, rtol=0, atol=1e-6)


def rigid(value, label):
    transform = numbers(value, (16,), label).reshape(4, 4)
    rotation = transform[:3, :3]
    require(close(transform[3], [0, 0, 0, 1])
            and close(rotation.T @ rotation, np.eye(3))
            and close(np.linalg.det(rotation), 1), f"non-rigid {label}")
    return transform


def safe_file(root, relative):
    require(isinstance(relative, str) and bool(relative), "invalid file path")
    path = PurePosixPath(relative)
    require("\\" not in relative and ":" not in relative
            and not path.is_absolute() and not PureWindowsPath(relative).drive
            and ".." not in path.parts, f"unsafe file path: {relative}")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root) and resolved.is_file(),
            f"missing/outside episode file: {relative}")
    return resolved


def json_object(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=pairs)
    require(isinstance(value, dict), "JSON object required")
    return value


class UERecordedEpisode:
    """Open validates every record and binary payload; iteration decodes on demand."""

    @classmethod
    def open(cls, path):
        try:
            episode = cls()
            episode.path = Path(path).resolve()
            require(not episode.path.name.endswith(".incomplete"), "incomplete recording")
            safe_file(episode.path, "_SUCCESS")
            episode.metadata = json_object(safe_file(episode.path, "episode.json")
                                           .read_text(encoding="utf-8"))
            calibration_bytes = safe_file(episode.path, "calibration.json").read_bytes()
            episode.calibration = json_object(calibration_bytes.decode("utf-8"))
            episode._validate_metadata(calibration_bytes)
            episode._validate_calibration()
            frame_text = safe_file(episode.path, "frames.jsonl").read_text(encoding="utf-8")
            lines = frame_text.splitlines()
            require(bool(lines) and all(line.strip() for line in lines), "empty frame records")
            episode._records = [json_object(line) for line in lines]
            require(len(episode) == episode.metadata["sample_count"], "sample_count mismatch")
            previous = None
            used_paths = set()
            for index, record in enumerate(episode._records):
                episode._validate_record(record, index, previous, used_paths)
                episode._decode(record)
                previous = record
            return episode
        except UERecordingError:
            raise
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            raise UERecordingError(f"invalid UE episode: {error}") from error

    def __len__(self):
        return len(self._records)

    def _validate_metadata(self, calibration_bytes):
        meta = self.metadata
        for key, expected in {
            "schema_version": "ofad-ue-episode-v1", "status": "complete",
            "vehicle_type": "scout_mini_skid_steer",
        }.items():
            require(meta[key] == expected, f"unsupported {key}")
        require(meta["expert_labels"] is False, "M1 is not expert data")
        require(meta["action_source"] in ("manual_keyboard", "scripted_smoke"), "action_source")
        require(meta["termination_reason"] in ("capture_limit", "manual_stop"),
                "termination_reason")
        integer(meta["sample_count"], "sample_count", 1)
        for key in ("engine_version", "map", "episode_id"):
            require(isinstance(meta[key], str) and bool(meta[key].strip()), f"invalid {key}")
        for key, expected in (("fixed_delta_seconds", 0.1), ("physics_delta_seconds", 1 / 60)):
            require(close(numbers(meta[key], (), key), expected), f"invalid {key}")
        require(meta["calibration_sha1"] == hashlib.sha1(calibration_bytes).hexdigest(),
                "calibration SHA1 mismatch")

    def _validate_calibration(self):
        calibration = self.calibration
        for key, expected in {
            "schema_version": "ofad-ue-calibration-v1", "calibration_version": "scout-m1-v1",
            "axes": "x-forward,y-left,z-up", "lidar_frame": "sensor",
            "imu_semantics": "specific_force_body_mps2,angular_velocity_body_radps",
            "camera_order": list(CAMERAS),
        }.items():
            require(calibration[key] == expected, f"unsupported calibration {key}")
        require(close(rigid(calibration["optical_to_sensor"], "optical_to_sensor"),
                      OPTICAL_TO_SENSOR), "invalid optical axis mapping")
        require(set(calibration["camera"]) == set(CAMERAS), "camera calibration set")
        for name in CAMERAS:
            camera = calibration["camera"][name]
            require(camera["image_size"] == [192, 192], "camera image_size")
            intrinsic = numbers(camera["intrinsic"], (9,), "intrinsic").reshape(3, 3)
            require(intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0
                    and 0 <= intrinsic[0, 2] < 192 and 0 <= intrinsic[1, 2] < 192
                    and close(intrinsic[[0, 1], [1, 0]], [0, 0])
                    and close(intrinsic[2], [0, 0, 1]), "invalid camera intrinsic")
            rigid(camera["sensor_to_ego"], f"camera {name}")
        self._lidar_to_ego = rigid(calibration["lidar"]["sensor_to_ego"], "lidar")
        require(close(rigid(calibration["imu"]["sensor_to_ego"], "imu"), np.eye(4)),
                "M1 IMU must be at root body reference")

    def _validate_record(self, record, index, previous, used_paths):
        require(integer(record["sample_index"], "sample_index") == index, "sample index gap")
        frame_id = integer(record["frame_id"], "frame_id", 54)
        stamp = numbers(record["timestamp"], (), "timestamp")
        require(frame_id % 6 == 0 and close(stamp, frame_id / 60), "physics timestamp mismatch")
        if previous is not None:
            require(frame_id - previous["frame_id"] == 6
                    and close(stamp - previous["timestamp"], 0.1), "sample cadence mismatch")
        require(record["expert_label"] is False, "M1 expert_label must be false")
        require(record["action_source"] == self.metadata["action_source"],
                "action_source mismatch")
        sensors = record["sensors"]
        require(set(sensors["camera"]) == set(CAMERAS), "camera set mismatch")
        packets = list(sensors["camera"].values()) + [sensors["lidar"], sensors["imu"]]
        for packet in packets:
            require(integer(packet["frame_id"], "sensor frame") == frame_id
                    and close(numbers(packet["timestamp"], (), "sensor timestamp"), stamp),
                    "sensor frame/timestamp mismatch")
        for packet in packets[:-1]:
            artifact = safe_file(self.path, packet["path"])
            require(artifact not in used_paths, "reused sensor file across samples")
            used_paths.add(artifact)
        integer(sensors["lidar"]["point_count"], "point_count", 256)
        imu = sensors["imu"]
        history = numbers(imu["history"], (10, 6), "imu history")
        ids = imu["history_frame_ids"]
        require(isinstance(ids, list) and len(ids) == 10, "IMU history frame shape")
        for value in ids:
            integer(value, "IMU history frame")
        times = numbers(imu["history_timestamps"], (10,), "IMU history timestamps")
        require(ids == list(range(frame_id - 54, frame_id + 1, 6))
                and close(times, np.asarray(ids) / 60), "IMU history continuity mismatch")
        if previous is not None:
            require(close(history[:-1], previous["sensors"]["imu"]["history"][1:]),
                    "IMU overlapping history mismatch")
        ego = record["ego_state"]
        for field in ("position_world_m", "velocity_world_mps", "velocity_body_mps",
                      "angular_velocity_body_radps", "acceleration_body_mps2"):
            numbers(ego[field], (3,), field)
        quaternion = numbers(ego["orientation_world_xyzw"], (4,), "orientation")
        require(close(np.linalg.norm(quaternion), 1), "non-unit quaternion")
        for field in ("linear_velocity_mps", "angular_velocity_radps"):
            numbers(record["action_applied"][field], (), field)

    def _decode(self, record):
        sensors = record["sensors"]
        images = []
        for name in CAMERAS:
            with Image.open(safe_file(self.path, sensors["camera"][name]["path"])) as image:
                require(image.format == "PNG" and image.mode in ("RGB", "RGBA")
                        and image.size == (192, 192), "invalid camera PNG format/size")
                images.append(np.array(image.convert("RGB"), dtype=np.uint8))
        raw_bytes = safe_file(self.path, sensors["lidar"]["path"]).read_bytes()
        count = sensors["lidar"]["point_count"]
        require(len(raw_bytes) == count * 16, "LiDAR byte length mismatch")
        raw = np.frombuffer(raw_bytes, dtype="<f4").reshape(count, 4).copy()
        require(np.isfinite(raw).all(), "non-finite raw LiDAR points")
        selected = np.random.default_rng(record["sample_index"]).choice(count, 256, replace=False)
        canonical = raw[selected].copy()
        transformed = (canonical[:, :3].astype(float) @ self._lidar_to_ego[:3, :3].T
                       + self._lidar_to_ego[:3, 3])
        numbers(transformed, (256, 3), "transformed LiDAR")
        canonical[:, :3] = transformed
        ego = record["ego_state"]
        x, y, z, w = ego["orientation_world_xyzw"]
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        vx, vy, _ = ego["velocity_body_mps"]
        dynamics = ScoutDynamicsV1((vx, ego["angular_velocity_body_radps"][2], pitch, roll,
                                   vx, vy, ego["acceleration_body_mps2"][2],
                                   record["action_applied"]["angular_velocity_radps"]))
        return RecordedUEFrame(
            record["sample_index"], record["frame_id"], record["timestamp"], tuple(images), raw,
            canonical, np.asarray(sensors["imu"]["history"], dtype=np.float32), dynamics,
            ego.copy(), record["action_applied"].copy(), record["action_source"])

    def iter_frames(self):
        """Decode validated files; surface later filesystem corruption as a contract error."""
        for record in self._records:
            try:
                safe_file(self.path, "_SUCCESS")
                yield self._decode(record)
            except UERecordingError:
                raise
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise UERecordingError(f"cannot decode UE frame: {error}") from error
