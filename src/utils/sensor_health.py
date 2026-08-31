"""Deterministic pre-model health gate for raw ORAD sensor packets.

The gate is intentionally NumPy-only so CARLA, replay and future ROS 2
adapters can share one reason vocabulary without importing a simulator or a
model runtime.  It validates raw LiDAR before any fixed-shape padding.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Optional, Tuple

import numpy as np


class SensorHealthReason(str, Enum):
    """Stable machine-readable reasons emitted by :class:`SensorHealthGate`."""

    PACKET_TIMEOUT = "packet_timeout"
    PACKET_MALFORMED = "packet_malformed"
    FRAME_ID_INVALID = "frame_id_invalid"
    REFERENCE_TIMESTAMP_INVALID = "reference_timestamp_invalid"
    SENSOR_SKEW = "sensor_skew"
    CALIBRATION_MISSING = "calibration_missing"
    CALIBRATION_MISMATCH = "calibration_mismatch"

    CAMERA_COUNT = "camera_count"
    CAMERA_SHAPE = "camera_shape"
    CAMERA_DTYPE = "camera_dtype"
    CAMERA_NONFINITE = "camera_nonfinite"
    CAMERA_BLACK = "camera_black"
    CAMERA_SATURATED = "camera_saturated"
    CAMERA_FROZEN = "camera_frozen"
    CAMERA_FRAME_MISMATCH = "camera_frame_mismatch"
    CAMERA_TIMESTAMP_MISSING = "camera_timestamp_missing"
    CAMERA_TIMESTAMP_NONFINITE = "camera_timestamp_nonfinite"
    CAMERA_TIMESTAMP_FUTURE = "camera_timestamp_future"
    CAMERA_STALE = "camera_stale"

    LIDAR_SHAPE = "lidar_shape"
    LIDAR_NONFINITE = "lidar_nonfinite"
    LIDAR_EMPTY = "lidar_empty"
    LIDAR_SPARSE = "lidar_sparse"
    LIDAR_OUT_OF_RANGE = "lidar_out_of_range"
    LIDAR_REPEATED_POINTS = "lidar_repeated_points"
    LIDAR_FROZEN = "lidar_frozen"
    LIDAR_FRAME_MISMATCH = "lidar_frame_mismatch"
    LIDAR_TIMESTAMP_NONFINITE = "lidar_timestamp_nonfinite"
    LIDAR_TIMESTAMP_FUTURE = "lidar_timestamp_future"
    LIDAR_STALE = "lidar_stale"

    IMU_SHAPE = "imu_shape"
    IMU_NONFINITE = "imu_nonfinite"
    IMU_INCOMPLETE = "imu_incomplete"
    IMU_FRAME_INVALID = "imu_frame_invalid"
    IMU_FREQUENCY = "imu_frequency"
    IMU_ACCEL_RANGE = "imu_accel_range"
    IMU_GYRO_RANGE = "imu_gyro_range"
    IMU_JUMP = "imu_jump"
    IMU_TIMESTAMP_NONFINITE = "imu_timestamp_nonfinite"
    IMU_TIMESTAMP_FUTURE = "imu_timestamp_future"
    IMU_STALE = "imu_stale"


@dataclass(frozen=True)
class SensorHealthConfig:
    """Raw sensor dimensions and health thresholds derived from system YAML."""

    num_cameras: int
    image_size: Tuple[int, int]
    imu_steps: int
    expected_calibration_version: str
    max_sensor_age_seconds: float
    max_sensor_skew_seconds: float
    black_pixel_threshold: int
    saturation_pixel_threshold: int
    max_black_ratio: float
    max_saturation_ratio: float
    max_consecutive_identical_frames: int
    min_lidar_points: int
    min_lidar_valid_ratio: float
    max_lidar_abs_coordinate: float
    max_lidar_duplicate_ratio: float
    max_accel_abs: float
    max_gyro_abs: float
    max_accel_step_delta: float
    max_gyro_step_delta: float
    max_imu_sample_gap_seconds: float

    def __post_init__(self) -> None:
        if self.num_cameras <= 0:
            raise ValueError("num_cameras must be > 0")
        if (len(self.image_size) != 2
                or any(int(value) <= 0 for value in self.image_size)):
            raise ValueError("image_size must contain two positive values")
        if self.imu_steps <= 0:
            raise ValueError("imu_steps must be > 0")
        if not str(self.expected_calibration_version).strip():
            raise ValueError("expected_calibration_version must be non-empty")
        for name in (
                "max_sensor_age_seconds", "max_sensor_skew_seconds"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not 0 <= int(self.black_pixel_threshold) <= 255:
            raise ValueError("black_pixel_threshold must be in [0, 255]")
        if not 0 <= int(self.saturation_pixel_threshold) <= 255:
            raise ValueError("saturation_pixel_threshold must be in [0, 255]")
        if self.black_pixel_threshold >= self.saturation_pixel_threshold:
            raise ValueError(
                "black_pixel_threshold must be below saturation_pixel_threshold")
        for name in (
                "max_black_ratio", "max_saturation_ratio",
                "min_lidar_valid_ratio", "max_lidar_duplicate_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
                "max_consecutive_identical_frames", "min_lidar_points"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in (
                "max_lidar_abs_coordinate", "max_accel_abs", "max_gyro_abs",
                "max_accel_step_delta", "max_gyro_step_delta",
                "max_imu_sample_gap_seconds"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True)
class SensorHealthReport:
    """Health decision for one synchronized raw sensor packet."""

    camera_valid: bool
    lidar_valid: bool
    imu_valid: bool
    frame_id: int
    reference_timestamp: float
    max_sensor_age_seconds: float
    max_sensor_skew_seconds: float
    reasons: Tuple[SensorHealthReason, ...]

    @property
    def valid(self) -> bool:
        return self.camera_valid and self.lidar_valid and self.imu_valid

    @property
    def modality_mask(self) -> np.ndarray:
        return np.asarray(
            [self.camera_valid, self.lidar_valid], dtype=np.bool_)


class SensorHealthGate:
    """Stateful Camera/LiDAR/IMU validator executed before model inference."""

    def __init__(self, config: SensorHealthConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Clear cross-frame freeze state at an episode boundary."""
        self._camera_state = None
        self._camera_repeats = 0
        self._lidar_state = None
        self._lidar_repeats = 0

    @staticmethod
    def _append(reasons, reason: SensorHealthReason) -> None:
        if reason not in reasons:
            reasons.append(reason)

    @staticmethod
    def _digest(arrays) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        for value in arrays:
            array = np.ascontiguousarray(value)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(array.tobytes())
        return digest.digest()

    def _advance_state(self, modality: str, digest: bytes, frame_id: int,
                       timestamp: float) -> bool:
        state_name = f"_{modality}_state"
        repeats_name = f"_{modality}_repeats"
        state = getattr(self, state_name)
        repeats = getattr(self, repeats_name)
        if (state is not None and digest == state[0]
                and frame_id > state[1] and timestamp > state[2]):
            repeats += 1
        else:
            repeats = 0
        setattr(self, state_name, (digest, frame_id, timestamp))
        setattr(self, repeats_name, repeats)
        return repeats >= self.config.max_consecutive_identical_frames

    def _reset_modality_state(self, modality: str) -> None:
        setattr(self, f"_{modality}_state", None)
        setattr(self, f"_{modality}_repeats", 0)

    def evaluate(
        self,
        *,
        images,
        point_cloud,
        imu_history,
        frame_id: int,
        reference_timestamp: float,
        camera_frames,
        lidar_frame: int,
        imu_frames,
        camera_timestamps,
        lidar_timestamp: float,
        imu_timestamps,
        calibration_version: Optional[str],
    ) -> SensorHealthReport:
        """Return a report for one raw packet; malformed data never escapes."""
        try:
            return self._evaluate(
                images=images, point_cloud=point_cloud,
                imu_history=imu_history, frame_id=frame_id,
                reference_timestamp=reference_timestamp,
                camera_frames=camera_frames, lidar_frame=lidar_frame,
                imu_frames=imu_frames, camera_timestamps=camera_timestamps,
                lidar_timestamp=lidar_timestamp,
                imu_timestamps=imu_timestamps,
                calibration_version=calibration_version)
        except Exception:
            self.reset()
            safe_frame = frame_id if isinstance(frame_id, int) else -1
            try:
                safe_timestamp = float(reference_timestamp)
            except (TypeError, ValueError):
                safe_timestamp = float("nan")
            return SensorHealthReport(
                False, False, False, safe_frame, safe_timestamp,
                float("inf"), float("inf"),
                (SensorHealthReason.PACKET_MALFORMED,))

    def _evaluate(
        self, *, images, point_cloud, imu_history, frame_id,
        reference_timestamp, camera_frames, lidar_frame, imu_frames,
        camera_timestamps, lidar_timestamp, imu_timestamps,
        calibration_version,
    ) -> SensorHealthReport:
        cfg = self.config
        reasons = []
        camera_valid = True
        lidar_valid = True
        imu_valid = True

        frame_ok = isinstance(frame_id, (int, np.integer)) and frame_id >= 0
        if not frame_ok:
            self._append(reasons, SensorHealthReason.FRAME_ID_INVALID)
            camera_valid = lidar_valid = imu_valid = False
        reference_ok = bool(np.isfinite(reference_timestamp))
        if not reference_ok:
            self._append(reasons,
                         SensorHealthReason.REFERENCE_TIMESTAMP_INVALID)
            camera_valid = lidar_valid = imu_valid = False
        reference = float(reference_timestamp) if reference_ok else 0.0

        image_values = list(images) if isinstance(images, (list, tuple)) else []
        camera_structural = len(image_values) == cfg.num_cameras
        if not camera_structural:
            self._append(reasons, SensorHealthReason.CAMERA_COUNT)
            camera_valid = False
        converted_images = []
        expected_shape = tuple(int(value) for value in cfg.image_size) + (3,)
        for image in image_values:
            array = np.asarray(image)
            converted_images.append(array)
            if array.shape != expected_shape:
                self._append(reasons, SensorHealthReason.CAMERA_SHAPE)
                camera_valid = camera_structural = False
            if array.dtype != np.uint8:
                self._append(reasons, SensorHealthReason.CAMERA_DTYPE)
                camera_valid = camera_structural = False
            try:
                finite = bool(np.isfinite(array).all())
            except TypeError:
                finite = False
            if not finite:
                self._append(reasons, SensorHealthReason.CAMERA_NONFINITE)
                camera_valid = camera_structural = False
        if converted_images and all(array.size for array in converted_images):
            pixels = np.concatenate([array.reshape(-1)
                                     for array in converted_images])
            try:
                black_ratio = float(np.mean(
                    pixels <= cfg.black_pixel_threshold))
                saturation_ratio = float(np.mean(
                    pixels >= cfg.saturation_pixel_threshold))
            except TypeError:
                black_ratio = saturation_ratio = 1.0
            if black_ratio > cfg.max_black_ratio:
                self._append(reasons, SensorHealthReason.CAMERA_BLACK)
                camera_valid = False
            if saturation_ratio > cfg.max_saturation_ratio:
                self._append(reasons, SensorHealthReason.CAMERA_SATURATED)
                camera_valid = False

        camera_frames_value = tuple(camera_frames)
        if (len(camera_frames_value) != cfg.num_cameras
                or not frame_ok
                or any(value != frame_id for value in camera_frames_value)):
            self._append(reasons, SensorHealthReason.CAMERA_FRAME_MISMATCH)
            camera_valid = False

        camera_times = tuple(camera_timestamps)
        latest_times = []
        ages = []
        camera_time_ok = len(camera_times) == cfg.num_cameras
        if not camera_time_ok:
            self._append(reasons,
                         SensorHealthReason.CAMERA_TIMESTAMP_MISSING)
            camera_valid = False
        elif not np.isfinite(camera_times).all():
            self._append(reasons,
                         SensorHealthReason.CAMERA_TIMESTAMP_NONFINITE)
            camera_valid = camera_time_ok = False
        if camera_time_ok and reference_ok:
            camera_ages = [reference - float(value) for value in camera_times]
            if any(age < 0.0 for age in camera_ages):
                self._append(reasons,
                             SensorHealthReason.CAMERA_TIMESTAMP_FUTURE)
                camera_valid = False
            if any(age > cfg.max_sensor_age_seconds for age in camera_ages):
                self._append(reasons, SensorHealthReason.CAMERA_STALE)
                camera_valid = False
            latest_times.extend(float(value) for value in camera_times)
            ages.extend(camera_ages)

        points = np.asarray(point_cloud)
        lidar_structural = points.ndim == 2 and points.shape[1] in (3, 4)
        if not lidar_structural:
            self._append(reasons, SensorHealthReason.LIDAR_SHAPE)
            lidar_valid = False
        elif points.shape[0] == 0:
            self._append(reasons, SensorHealthReason.LIDAR_EMPTY)
            lidar_valid = lidar_structural = False
        else:
            try:
                points_finite = bool(np.isfinite(points).all())
            except TypeError:
                points_finite = False
            if not points_finite:
                self._append(reasons, SensorHealthReason.LIDAR_NONFINITE)
                lidar_valid = lidar_structural = False
            if points.shape[0] < cfg.min_lidar_points:
                self._append(reasons, SensorHealthReason.LIDAR_SPARSE)
                lidar_valid = False
            if points_finite:
                xyz = np.asarray(points[:, :3], dtype=np.float64)
                in_range = np.all(
                    np.abs(xyz) <= cfg.max_lidar_abs_coordinate, axis=1)
                if float(np.mean(in_range)) < cfg.min_lidar_valid_ratio:
                    self._append(reasons,
                                 SensorHealthReason.LIDAR_OUT_OF_RANGE)
                    lidar_valid = False
                unique_ratio = np.unique(xyz, axis=0).shape[0] / xyz.shape[0]
                if 1.0 - unique_ratio > cfg.max_lidar_duplicate_ratio:
                    self._append(reasons,
                                 SensorHealthReason.LIDAR_REPEATED_POINTS)
                    lidar_valid = False

        if not frame_ok or lidar_frame != frame_id:
            self._append(reasons, SensorHealthReason.LIDAR_FRAME_MISMATCH)
            lidar_valid = False
        lidar_time_ok = bool(np.isfinite(lidar_timestamp))
        if not lidar_time_ok:
            self._append(reasons,
                         SensorHealthReason.LIDAR_TIMESTAMP_NONFINITE)
            lidar_valid = False
        elif reference_ok:
            lidar_age = reference - float(lidar_timestamp)
            if lidar_age < 0.0:
                self._append(reasons,
                             SensorHealthReason.LIDAR_TIMESTAMP_FUTURE)
                lidar_valid = False
            if lidar_age > cfg.max_sensor_age_seconds:
                self._append(reasons, SensorHealthReason.LIDAR_STALE)
                lidar_valid = False
            latest_times.append(float(lidar_timestamp))
            ages.append(lidar_age)

        imu = np.asarray(imu_history)
        imu_structural = imu.shape == (cfg.imu_steps, 6)
        if not imu_structural:
            self._append(reasons, SensorHealthReason.IMU_SHAPE)
            imu_valid = False
        else:
            try:
                imu_finite = bool(np.isfinite(imu).all())
            except TypeError:
                imu_finite = False
            if not imu_finite:
                self._append(reasons, SensorHealthReason.IMU_NONFINITE)
                imu_valid = False
            else:
                accel = np.asarray(imu[:, :3], dtype=np.float64)
                gyro = np.asarray(imu[:, 3:], dtype=np.float64)
                if np.max(np.abs(accel)) > cfg.max_accel_abs:
                    self._append(reasons, SensorHealthReason.IMU_ACCEL_RANGE)
                    imu_valid = False
                if np.max(np.abs(gyro)) > cfg.max_gyro_abs:
                    self._append(reasons, SensorHealthReason.IMU_GYRO_RANGE)
                    imu_valid = False
                if cfg.imu_steps > 1 and (
                        np.max(np.abs(np.diff(accel, axis=0)))
                        > cfg.max_accel_step_delta
                        or np.max(np.abs(np.diff(gyro, axis=0)))
                        > cfg.max_gyro_step_delta):
                    self._append(reasons, SensorHealthReason.IMU_JUMP)
                    imu_valid = False

        imu_frames_value = tuple(imu_frames)
        if len(imu_frames_value) != cfg.imu_steps:
            self._append(reasons, SensorHealthReason.IMU_INCOMPLETE)
            imu_valid = False
        elif (not frame_ok
              or any(b <= a for a, b in zip(
                  imu_frames_value, imu_frames_value[1:]))
              or imu_frames_value[-1] > frame_id):
            self._append(reasons, SensorHealthReason.IMU_FRAME_INVALID)
            imu_valid = False

        imu_times = tuple(imu_timestamps)
        imu_time_ok = len(imu_times) == cfg.imu_steps
        if not imu_time_ok:
            self._append(reasons, SensorHealthReason.IMU_INCOMPLETE)
            imu_valid = False
        elif not np.isfinite(imu_times).all():
            self._append(reasons,
                         SensorHealthReason.IMU_TIMESTAMP_NONFINITE)
            imu_valid = imu_time_ok = False
        if imu_time_ok and reference_ok:
            gaps = np.diff(np.asarray(imu_times, dtype=np.float64))
            if (np.any(gaps <= 0.0)
                    or np.any(gaps > cfg.max_imu_sample_gap_seconds)):
                self._append(reasons, SensorHealthReason.IMU_FREQUENCY)
                imu_valid = False
            imu_age = reference - float(imu_times[-1])
            if imu_age < 0.0:
                self._append(reasons, SensorHealthReason.IMU_TIMESTAMP_FUTURE)
                imu_valid = False
            if imu_age > cfg.max_sensor_age_seconds:
                self._append(reasons, SensorHealthReason.IMU_STALE)
                imu_valid = False
            latest_times.append(float(imu_times[-1]))
            ages.append(imu_age)

        if calibration_version is None or not str(calibration_version).strip():
            self._append(reasons, SensorHealthReason.CALIBRATION_MISSING)
            camera_valid = lidar_valid = False
        elif calibration_version != cfg.expected_calibration_version:
            self._append(reasons, SensorHealthReason.CALIBRATION_MISMATCH)
            camera_valid = lidar_valid = False

        max_skew = (max(latest_times) - min(latest_times)
                    if latest_times else float("inf"))
        if (len(latest_times) == cfg.num_cameras + 2
                and max_skew > cfg.max_sensor_skew_seconds):
            self._append(reasons, SensorHealthReason.SENSOR_SKEW)
            camera_valid = lidar_valid = imu_valid = False
        max_age = max(ages) if ages else float("inf")

        if camera_structural and camera_time_ok and frame_ok:
            if self._advance_state(
                    "camera", self._digest(converted_images), frame_id,
                    max(camera_times)):
                self._append(reasons, SensorHealthReason.CAMERA_FROZEN)
                camera_valid = False
        else:
            self._reset_modality_state("camera")
        if lidar_structural and lidar_time_ok and frame_ok:
            if self._advance_state(
                    "lidar", self._digest((points,)), frame_id,
                    float(lidar_timestamp)):
                self._append(reasons, SensorHealthReason.LIDAR_FROZEN)
                lidar_valid = False
        else:
            self._reset_modality_state("lidar")

        return SensorHealthReport(
            camera_valid=bool(camera_valid), lidar_valid=bool(lidar_valid),
            imu_valid=bool(imu_valid), frame_id=int(frame_id),
            reference_timestamp=float(reference_timestamp),
            max_sensor_age_seconds=float(max_age),
            max_sensor_skew_seconds=float(max_skew),
            reasons=tuple(reasons))


__all__ = [
    "SensorHealthReason",
    "SensorHealthConfig",
    "SensorHealthReport",
    "SensorHealthGate",
]
