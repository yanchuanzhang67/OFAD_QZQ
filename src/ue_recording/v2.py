"""Strict read-only consumer for the frozen Forest ``ofad-ue-episode-v2``.

V2 is a separate protocol from M1/V1: one 320x180 front RGB camera, one
versioned LiDAR profile, derived IMU history, Scout state and applied command.
Integration-smoke recordings remain non-expert data.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image

from utils.types import RecordedUEFrame, ScoutDynamicsV1
from .episode import (
    OPTICAL_TO_SENSOR,
    UERecordingError,
    close,
    integer,
    json_object,
    numbers,
    require,
    rigid,
    safe_file,
)


PROFILE_CONTRACTS = {
    "solidstate-placeholder-v1": {
        "scan_type": "solidstate_uniform_grid",
        "nominal_rays_per_scan": 27648,
        "nominal_rays_per_second": 276480,
        "horizontal_samples": 192,
        "vertical_samples": 144,
        "horizontal_fov_degrees": 120,
        "vertical_fov_degrees": 90,
    },
    "mid360-approx-v1": {
        "scan_type": "mechanical_uniform",
        "nominal_rays_per_scan": 20000,
        "nominal_rays_per_second": 200000,
        "channels": 40,
        "points_per_channel": 500,
        "horizontal_fov_degrees": 360,
        "vertical_min_degrees": -7,
        "vertical_max_degrees": 52,
    },
}


class UERecordedEpisodeV2:
    """Validate all V2 metadata and payloads, then decode frames on demand."""

    @classmethod
    def open(cls, path):
        try:
            episode = cls()
            episode.path = Path(path).resolve()
            require(not episode.path.name.endswith(".incomplete"),
                    "incomplete recording")
            safe_file(episode.path, "_SUCCESS")
            episode.metadata = json_object(
                safe_file(episode.path, "episode.json").read_text(
                    encoding="utf-8"))
            calibration_bytes = safe_file(
                episode.path, "calibration.json").read_bytes()
            episode.calibration = json_object(
                calibration_bytes.decode("utf-8"))
            episode._validate_metadata(calibration_bytes)
            episode._validate_calibration()
            lines = safe_file(episode.path, "frames.jsonl").read_text(
                encoding="utf-8").splitlines()
            require(bool(lines) and all(line.strip() for line in lines),
                    "empty frame records")
            episode._records = [json_object(line) for line in lines]
            require(len(episode) == episode.metadata["sample_count"],
                    "sample_count mismatch")
            previous = None
            used_paths = set()
            for index, record in enumerate(episode._records):
                episode._validate_record(record, index, previous, used_paths)
                episode._decode(record)
                previous = record
            return episode
        except UERecordingError:
            raise
        except (
                OSError, ValueError, KeyError, TypeError,
                OverflowError) as error:
            raise UERecordingError(
                f"invalid UE Forest V2 episode: {error}") from error

    def __len__(self):
        return len(self._records)

    def _validate_metadata(self, calibration_bytes):
        metadata = self.metadata
        expected = {
            "schema_version": "ofad-ue-episode-v2",
            "status": "complete",
            "map": "forest111",
            "scene_version": "forest-smoke-v1",
            "rig_version": "scout-forest-rig-v1",
            "vehicle_type": "scout_mini_skid_steer",
            "imu_source": "derived_from_root_motion",
            "task": "integration_smoke",
        }
        for key, value in expected.items():
            require(metadata[key] == value, f"unsupported {key}")
        profile = metadata["sensor_profile"]
        require(profile in PROFILE_CONTRACTS, "unsupported sensor_profile")
        require(metadata["expert_labels"] is False,
                "integration smoke is not expert data")
        require(metadata["goal_valid"] is False,
                "integration_smoke cannot carry a valid goal")
        require(metadata["action_source"] in ("manual", "scripted"),
                "action_source")
        require(metadata["termination_reason"] in (
            "capture_limit", "manual_stop"),
                "termination_reason")
        require(type(metadata["source_worktree_dirty"]) is bool,
                "source_worktree_dirty")
        require(
            isinstance(metadata["source_commit"], str) and
            re.fullmatch(
                r"[0-9a-fA-F]{7,40}", metadata["source_commit"]),
            "source_commit")
        count = integer(metadata["sample_count"], "sample_count", 1)
        require(
            metadata["episode_id"] == self.path.name,
            "episode_id mismatch")
        require(
            isinstance(metadata["engine_version"], str) and
            metadata["engine_version"].strip(), "engine_version")
        require(
            metadata["calibration_sha1"] ==
            hashlib.sha1(calibration_bytes).hexdigest(),
            "calibration SHA1 mismatch")
        timing = metadata["timing"]
        require(set(timing) == {"physics_hz", "sample_hz", "frame_count"},
                "timing fields")
        require(integer(timing["physics_hz"], "physics_hz", 1) == 60,
                "physics_hz")
        require(integer(timing["sample_hz"], "sample_hz", 1) == 10,
                "sample_hz")
        require(integer(timing["frame_count"], "frame_count", 1) == count,
                "frame_count mismatch")
        inventory = metadata["actor_inventory"]
        require(set(inventory) == {"scout", "camera", "lidar"},
                "actor_inventory fields")
        for kind, actor in inventory.items():
            require(isinstance(actor["name"], str) and actor["name"].strip(),
                    f"{kind} name")
            require(isinstance(actor["class"], str) and actor["class"].strip(),
                    f"{kind} class")
            transform = rigid(actor["relative_pose"], f"{kind} relative_pose")
            if kind == "scout":
                require(close(transform, np.eye(4)), "scout relative pose")

    def _validate_calibration(self):
        calibration = self.calibration
        expected = {
            "schema_version": "ofad-ue-calibration-v2",
            "calibration_version": "scout-forest-rig-v1",
            "axes": "x-forward,y-left,z-up",
            "lidar_frame": "sensor",
            "imu_semantics":
                "specific_force_body_mps2,angular_velocity_body_radps",
        }
        for key, value in expected.items():
            require(
                calibration[key] == value,
                f"unsupported calibration {key}")
        optical_to_sensor = rigid(
            calibration["optical_to_sensor"], "optical_to_sensor")
        require(
            close(optical_to_sensor, OPTICAL_TO_SENSOR),
            "invalid optical axis mapping")
        require(calibration["camera_order"] == ["camera_0"], "camera_order")
        require(set(calibration["camera"]) == {"camera_0"}, "camera set")
        camera = calibration["camera"]["camera_0"]
        require(camera["image_size"] == [180, 320], "camera image_size")
        native_hz = numbers(
            camera["native_capture_hz"], (), "native_capture_hz")
        require(close(native_hz, 30), "native_capture_hz")
        intrinsic = numbers(
            camera["intrinsic"], (9,), "intrinsic").reshape(3, 3)
        require(
            intrinsic[0, 0] > 0 and
            intrinsic[1, 1] > 0 and
            0 <= intrinsic[0, 2] < 320 and
            0 <= intrinsic[1, 2] < 180 and
            close(intrinsic[[0, 1], [1, 0]], [0, 0]) and
            close(intrinsic[2], [0, 0, 1]),
            "invalid camera intrinsic")
        camera_to_ego = rigid(camera["sensor_to_ego"], "camera sensor_to_ego")
        self._lidar_to_ego = rigid(
            calibration["lidar"]["sensor_to_ego"], "lidar sensor_to_ego")
        inventory = self.metadata["actor_inventory"]
        camera_inventory = rigid(
            inventory["camera"]["relative_pose"], "camera inventory")
        lidar_inventory = rigid(
            inventory["lidar"]["relative_pose"], "lidar inventory")
        require(close(camera_to_ego, camera_inventory),
                "camera pose/inventory mismatch")
        require(close(self._lidar_to_ego, lidar_inventory),
                "lidar pose/inventory mismatch")
        imu = calibration["imu"]
        require(
            imu["source"] == self.metadata["imu_source"],
            "IMU source mismatch")
        imu_to_ego = rigid(
            imu["sensor_to_ego"], "imu sensor_to_ego")
        require(close(imu_to_ego, np.eye(4)),
                "derived IMU must use the body reference")
        self._validate_lidar_profile()

    def _validate_lidar_profile(self):
        lidar = self.calibration["lidar"]
        profile = self.metadata["sensor_profile"]
        require(lidar["sensor_profile"] == profile, "sensor profile mismatch")
        require(lidar["scan_model"] == "instantaneous_raycast", "scan_model")
        require(
            lidar["intensity_model"] == "incidence_cosine",
            "intensity_model")
        for key, expected in {
            "scan_hz": 10, "minimum_range_m": 0.1, "maximum_range_m": 50,
        }.items():
            require(
                close(numbers(lidar[key], (), key), expected),
                f"invalid {key}")
        for key, expected in PROFILE_CONTRACTS[profile].items():
            if isinstance(expected, str):
                require(lidar[key] == expected, f"invalid profile {key}")
            else:
                require(close(numbers(lidar[key], (), key), expected),
                        f"invalid profile {key}")
        self.nominal_rays_per_scan = PROFILE_CONTRACTS[profile][
            "nominal_rays_per_scan"]

    def _validate_record(self, record, index, previous, used_paths):
        require(integer(record["sample_index"], "sample_index") == index,
                "sample index gap")
        frame_id = integer(record["frame_id"], "frame_id", 54)
        timestamp = numbers(record["timestamp"], (), "timestamp")
        require(frame_id % 6 == 0 and close(timestamp, frame_id / 60),
                "physics timestamp mismatch")
        if previous is not None:
            require(
                frame_id - previous["frame_id"] == 6 and
                close(timestamp - previous["timestamp"], 0.1),
                "sample cadence mismatch")
        require(record["expert_label"] is False, "expert_label must be false")
        require(record["action_source"] == self.metadata["action_source"],
                "action_source mismatch")
        require(
            record["task"] == "integration_smoke" and
            record["goal_valid"] is False,
            "frame task mismatch")
        sensors = record["sensors"]
        require(set(sensors) == {"camera_0", "lidar_0", "imu"},
                "sensor set mismatch")
        for packet in sensors.values():
            packet_frame = integer(packet["frame_id"], "sensor frame")
            packet_time = numbers(
                packet["timestamp"], (), "sensor timestamp")
            require(
                packet_frame == frame_id and
                close(packet_time, timestamp),
                "sensor frame/timestamp mismatch")
        for packet in (sensors["camera_0"], sensors["lidar_0"]):
            artifact = safe_file(self.path, packet["path"])
            require(artifact not in used_paths, "reused sensor file")
            used_paths.add(artifact)
        count = integer(
            sensors["lidar_0"]["point_count"], "point_count", 256)
        require(
            count <= self.nominal_rays_per_scan,
            "point_count exceeds ray budget")
        imu = sensors["imu"]
        history = numbers(imu["history"], (10, 6), "imu history")
        ids = imu["history_frame_ids"]
        require(
            isinstance(ids, list) and len(ids) == 10,
            "IMU history frame shape")
        for value in ids:
            integer(value, "IMU history frame")
        times = numbers(imu["history_timestamps"], (10,), "IMU timestamps")
        expected_ids = list(range(frame_id - 54, frame_id + 1, 6))
        require(ids == expected_ids and close(times, np.asarray(ids) / 60),
                "IMU history continuity mismatch")
        if previous is not None:
            previous_history = previous["sensors"]["imu"]["history"]
            require(close(history[:-1], previous_history[1:]),
                    "IMU overlapping history mismatch")
        ego = record["ego_state"]
        for field in (
                "position_world_m", "velocity_world_mps",
                "velocity_body_mps", "angular_velocity_body_radps",
                "acceleration_body_mps2"):
            numbers(ego[field], (3,), field)
        quaternion = numbers(
            ego["orientation_world_xyzw"], (4,), "orientation")
        require(close(np.linalg.norm(quaternion), 1), "non-unit quaternion")
        action = record["action_applied"]
        for field in ("linear_velocity_mps", "angular_velocity_radps"):
            numbers(action[field], (), field)
        require(
            close(
                action["linear_velocity_mps"],
                ego["command_linear_velocity_mps"]) and
            close(
                action["angular_velocity_radps"],
                ego["command_angular_velocity_radps"]),
            "ego/action command mismatch")

    def _decode(self, record):
        sensors = record["sensors"]
        image_file = safe_file(
            self.path, sensors["camera_0"]["path"])
        with Image.open(image_file) as image:
            require(
                image.format == "PNG" and
                image.mode in ("RGB", "RGBA") and
                image.size == (320, 180),
                "invalid camera PNG")
            camera = np.array(image.convert("RGB"), dtype=np.uint8)
        lidar_file = safe_file(
            self.path, sensors["lidar_0"]["path"])
        raw_bytes = lidar_file.read_bytes()
        count = sensors["lidar_0"]["point_count"]
        require(len(raw_bytes) == count * 16, "LiDAR byte length mismatch")
        raw = np.frombuffer(raw_bytes, dtype="<f4").reshape(count, 4).copy()
        require(np.isfinite(raw).all(), "non-finite raw LiDAR points")
        selected = np.random.default_rng(record["sample_index"]).choice(
            count, 256, replace=False)
        canonical = raw[selected].copy()
        transformed = (
            canonical[:, :3].astype(float) @
            self._lidar_to_ego[:3, :3].T +
            self._lidar_to_ego[:3, 3])
        numbers(transformed, (256, 3), "transformed LiDAR")
        canonical[:, :3] = transformed
        ego = record["ego_state"]
        x, y, z, w = ego["orientation_world_xyzw"]
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        vx, vy, _ = ego["velocity_body_mps"]
        dynamics = ScoutDynamicsV1((
            vx, ego["angular_velocity_body_radps"][2], pitch, roll, vx, vy,
            ego["acceleration_body_mps2"][2],
            record["action_applied"]["angular_velocity_radps"]))
        return RecordedUEFrame(
            record["sample_index"], record["frame_id"], record["timestamp"],
            (camera,), raw, canonical,
            np.asarray(sensors["imu"]["history"], dtype=np.float32), dynamics,
            ego.copy(), record["action_applied"].copy(),
            record["action_source"])

    def iter_frames(self):
        """Decode frames and reject later payload removal or corruption."""
        for record in self._records:
            try:
                safe_file(self.path, "_SUCCESS")
                yield self._decode(record)
            except UERecordingError:
                raise
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise UERecordingError(
                    f"cannot decode UE Forest V2 frame: {error}") from error
