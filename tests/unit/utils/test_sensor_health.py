from copy import deepcopy

import numpy as np
import pytest

from utils.sensor_health import (
    SensorHealthConfig,
    SensorHealthGate,
    SensorHealthReason,
)

pytestmark = pytest.mark.unit


def _config_values():
    return {
        "num_cameras": 2,
        "image_size": (4, 6),
        "imu_steps": 3,
        "expected_calibration_version": "cal-v1",
        "max_sensor_age_seconds": 0.2,
        "max_sensor_skew_seconds": 0.05,
        "black_pixel_threshold": 2,
        "saturation_pixel_threshold": 253,
        "max_black_ratio": 0.98,
        "max_saturation_ratio": 0.98,
        "max_consecutive_identical_frames": 2,
        "min_lidar_points": 4,
        "min_lidar_valid_ratio": 0.8,
        "max_lidar_abs_coordinate": 100.0,
        "max_lidar_duplicate_ratio": 0.6,
        "max_accel_abs": 50.0,
        "max_gyro_abs": 5.0,
        "max_accel_step_delta": 5.0,
        "max_gyro_step_delta": 1.0,
        "max_imu_sample_gap_seconds": 0.11,
    }


def _config():
    return SensorHealthConfig(**_config_values())


def _packet(frame=10):
    timestamp = frame * 0.1
    image = (np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
             + np.uint8(20))
    points = np.array([
        [1.0, 0.0, 0.0, 0.1],
        [2.0, 0.5, 0.1, 0.2],
        [3.0, -0.5, 0.2, 0.3],
        [4.0, 1.0, -0.1, 0.4],
        [5.0, -1.0, 0.3, 0.5],
    ], dtype=np.float32)
    imu = np.array([
        [0.0, 0.0, 9.7, 0.00, 0.00, 0.00],
        [0.1, 0.0, 9.8, 0.01, 0.00, 0.00],
        [0.1, 0.1, 9.8, 0.01, 0.01, 0.00],
    ], dtype=np.float32)
    return {
        "images": [image.copy(), np.flip(image, axis=1).copy()],
        "point_cloud": points,
        "imu_history": imu,
        "frame_id": frame,
        "reference_timestamp": timestamp,
        "camera_frames": (frame, frame),
        "lidar_frame": frame,
        "imu_frames": (frame - 2, frame - 1, frame),
        "camera_timestamps": (timestamp - 0.01, timestamp - 0.01),
        "lidar_timestamp": timestamp - 0.02,
        "imu_timestamps": (timestamp - 0.20, timestamp - 0.10,
                           timestamp - 0.03),
        "calibration_version": "cal-v1",
    }


def test_valid_packet_returns_strict_online_mask_and_timing():
    report = SensorHealthGate(_config()).evaluate(**_packet())

    assert report.valid
    assert report.reasons == ()
    np.testing.assert_array_equal(report.modality_mask, [True, True])
    assert report.modality_mask.dtype == np.bool_
    assert report.max_sensor_age_seconds == pytest.approx(0.03)
    assert report.max_sensor_skew_seconds == pytest.approx(0.02)


@pytest.mark.parametrize("field,value", [
    ("num_cameras", 0),
    ("image_size", (0, 6)),
    ("imu_steps", 0),
    ("expected_calibration_version", ""),
    ("max_sensor_age_seconds", -0.1),
    ("max_sensor_skew_seconds", -0.1),
    ("black_pixel_threshold", -1),
    ("saturation_pixel_threshold", 256),
    ("max_black_ratio", 1.1),
    ("max_saturation_ratio", -0.1),
    ("max_consecutive_identical_frames", 0),
    ("min_lidar_points", 0),
    ("min_lidar_valid_ratio", 1.1),
    ("max_lidar_abs_coordinate", 0.0),
    ("max_lidar_duplicate_ratio", -0.1),
    ("max_accel_abs", 0.0),
    ("max_gyro_abs", 0.0),
    ("max_accel_step_delta", 0.0),
    ("max_gyro_step_delta", 0.0),
    ("max_imu_sample_gap_seconds", 0.0),
])
def test_health_config_rejects_invalid_thresholds(field, value):
    values = _config_values()
    values[field] = value

    with pytest.raises(ValueError):
        SensorHealthConfig(**values)


def test_health_config_rejects_overlapping_pixel_thresholds():
    values = _config_values()
    values["black_pixel_threshold"] = 254

    with pytest.raises(ValueError, match="black_pixel_threshold"):
        SensorHealthConfig(**values)


def _mutate(packet, mutation):
    if mutation == "camera_count":
        packet["images"].pop()
        packet["camera_frames"] = (packet["frame_id"],)
        packet["camera_timestamps"] = packet["camera_timestamps"][:1]
    elif mutation == "camera_shape":
        packet["images"][0] = packet["images"][0][:, :, :2]
    elif mutation == "camera_dtype":
        packet["images"][0] = packet["images"][0].astype(np.float32)
    elif mutation == "camera_nonfinite":
        packet["images"][0] = packet["images"][0].astype(np.float32)
        packet["images"][0][0, 0, 0] = np.nan
    elif mutation == "camera_black":
        packet["images"] = [np.zeros_like(image) for image in packet["images"]]
    elif mutation == "camera_saturated":
        packet["images"] = [np.full_like(image, 255)
                            for image in packet["images"]]
    elif mutation == "lidar_shape":
        packet["point_cloud"] = np.zeros((5, 2), dtype=np.float32)
    elif mutation == "lidar_empty":
        packet["point_cloud"] = np.zeros((0, 4), dtype=np.float32)
    elif mutation == "lidar_sparse":
        packet["point_cloud"] = packet["point_cloud"][:2]
    elif mutation == "lidar_nonfinite":
        packet["point_cloud"][0, 0] = np.nan
    elif mutation == "lidar_out_of_range":
        packet["point_cloud"][:4, 0] = 101.0
    elif mutation == "lidar_duplicates":
        packet["point_cloud"][:] = packet["point_cloud"][0]
    elif mutation == "imu_shape":
        packet["imu_history"] = packet["imu_history"][:, :5]
    elif mutation == "imu_nonfinite":
        packet["imu_history"][0, 0] = np.inf
    elif mutation == "imu_incomplete":
        packet["imu_frames"] = packet["imu_frames"][:-1]
    elif mutation == "imu_gap":
        timestamp = packet["reference_timestamp"]
        packet["imu_timestamps"] = (timestamp - 0.40,
                                    timestamp - 0.20,
                                    timestamp - 0.03)
    elif mutation == "imu_accel_range":
        packet["imu_history"][1, 0] = 51.0
    elif mutation == "imu_gyro_range":
        packet["imu_history"][1, 3] = 5.1
    elif mutation == "imu_jump":
        packet["imu_history"][1, 0] = 6.0
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(mutation)
    return packet


@pytest.mark.parametrize("mutation,reason,attribute", [
    ("camera_count", SensorHealthReason.CAMERA_COUNT, "camera_valid"),
    ("camera_shape", SensorHealthReason.CAMERA_SHAPE, "camera_valid"),
    ("camera_dtype", SensorHealthReason.CAMERA_DTYPE, "camera_valid"),
    ("camera_nonfinite", SensorHealthReason.CAMERA_NONFINITE, "camera_valid"),
    ("camera_black", SensorHealthReason.CAMERA_BLACK, "camera_valid"),
    ("camera_saturated", SensorHealthReason.CAMERA_SATURATED, "camera_valid"),
    ("lidar_shape", SensorHealthReason.LIDAR_SHAPE, "lidar_valid"),
    ("lidar_empty", SensorHealthReason.LIDAR_EMPTY, "lidar_valid"),
    ("lidar_sparse", SensorHealthReason.LIDAR_SPARSE, "lidar_valid"),
    ("lidar_nonfinite", SensorHealthReason.LIDAR_NONFINITE, "lidar_valid"),
    ("lidar_out_of_range", SensorHealthReason.LIDAR_OUT_OF_RANGE,
     "lidar_valid"),
    ("lidar_duplicates", SensorHealthReason.LIDAR_REPEATED_POINTS,
     "lidar_valid"),
    ("imu_shape", SensorHealthReason.IMU_SHAPE, "imu_valid"),
    ("imu_nonfinite", SensorHealthReason.IMU_NONFINITE, "imu_valid"),
    ("imu_incomplete", SensorHealthReason.IMU_INCOMPLETE, "imu_valid"),
    ("imu_gap", SensorHealthReason.IMU_FREQUENCY, "imu_valid"),
    ("imu_accel_range", SensorHealthReason.IMU_ACCEL_RANGE, "imu_valid"),
    ("imu_gyro_range", SensorHealthReason.IMU_GYRO_RANGE, "imu_valid"),
    ("imu_jump", SensorHealthReason.IMU_JUMP, "imu_valid"),
])
def test_deterministic_sensor_faults_are_rejected(
        mutation, reason, attribute):
    packet = _mutate(deepcopy(_packet()), mutation)

    report = SensorHealthGate(_config()).evaluate(**packet)

    assert reason in report.reasons
    assert getattr(report, attribute) is False
    assert report.valid is False


@pytest.mark.parametrize("mutation,reason,invalid_modalities", [
    ("frame_invalid", SensorHealthReason.FRAME_ID_INVALID,
     ("camera", "lidar", "imu")),
    ("reference_nonfinite", SensorHealthReason.REFERENCE_TIMESTAMP_INVALID,
     ("camera", "lidar", "imu")),
    ("camera_frame", SensorHealthReason.CAMERA_FRAME_MISMATCH, ("camera",)),
    ("lidar_frame", SensorHealthReason.LIDAR_FRAME_MISMATCH, ("lidar",)),
    ("imu_frame", SensorHealthReason.IMU_FRAME_INVALID, ("imu",)),
    ("camera_timestamp_missing", SensorHealthReason.CAMERA_TIMESTAMP_MISSING,
     ("camera",)),
    ("camera_timestamp_nonfinite",
     SensorHealthReason.CAMERA_TIMESTAMP_NONFINITE, ("camera",)),
    ("camera_timestamp_future", SensorHealthReason.CAMERA_TIMESTAMP_FUTURE,
     ("camera",)),
    ("camera_stale", SensorHealthReason.CAMERA_STALE, ("camera",)),
    ("lidar_timestamp_nonfinite", SensorHealthReason.LIDAR_TIMESTAMP_NONFINITE,
     ("lidar",)),
    ("lidar_timestamp_future", SensorHealthReason.LIDAR_TIMESTAMP_FUTURE,
     ("lidar",)),
    ("imu_timestamp_nonfinite", SensorHealthReason.IMU_TIMESTAMP_NONFINITE,
     ("imu",)),
    ("imu_timestamp_future", SensorHealthReason.IMU_TIMESTAMP_FUTURE, ("imu",)),
    ("imu_stale", SensorHealthReason.IMU_STALE, ("imu",)),
    ("lidar_stale", SensorHealthReason.LIDAR_STALE, ("lidar",)),
    ("skew", SensorHealthReason.SENSOR_SKEW, ("camera", "lidar", "imu")),
    ("calibration_missing", SensorHealthReason.CALIBRATION_MISSING,
     ("camera", "lidar")),
    ("calibration_mismatch", SensorHealthReason.CALIBRATION_MISMATCH,
     ("camera", "lidar")),
])
def test_metadata_faults_are_rejected(mutation, reason, invalid_modalities):
    packet = _packet()
    if mutation == "frame_invalid":
        packet["frame_id"] = -1
    elif mutation == "reference_nonfinite":
        packet["reference_timestamp"] = np.nan
    elif mutation == "camera_frame":
        packet["camera_frames"] = (9, 10)
    elif mutation == "lidar_frame":
        packet["lidar_frame"] = 9
    elif mutation == "imu_frame":
        packet["imu_frames"] = (8, 10, 9)
    elif mutation == "camera_timestamp_missing":
        packet["camera_timestamps"] = (0.99,)
    elif mutation == "camera_timestamp_nonfinite":
        packet["camera_timestamps"] = (np.inf, 0.99)
    elif mutation == "camera_timestamp_future":
        packet["camera_timestamps"] = (1.01, 0.99)
    elif mutation == "camera_stale":
        packet["camera_timestamps"] = (0.79, 0.99)
    elif mutation == "lidar_timestamp_nonfinite":
        packet["lidar_timestamp"] = np.nan
    elif mutation == "lidar_timestamp_future":
        packet["lidar_timestamp"] = 1.01
    elif mutation == "imu_timestamp_nonfinite":
        packet["imu_timestamps"] = (0.8, np.nan, 0.97)
    elif mutation == "imu_timestamp_future":
        packet["imu_timestamps"] = (0.8, 0.9, 1.01)
    elif mutation == "imu_stale":
        packet["imu_timestamps"] = (0.59, 0.69, 0.79)
    elif mutation == "lidar_stale":
        packet["lidar_timestamp"] = 0.79
    elif mutation == "skew":
        packet["lidar_timestamp"] = 0.91
    elif mutation == "calibration_missing":
        packet["calibration_version"] = None
    elif mutation == "calibration_mismatch":
        packet["calibration_version"] = "cal-v2"

    report = SensorHealthGate(_config()).evaluate(**packet)

    assert reason in report.reasons
    for modality in invalid_modalities:
        assert getattr(report, f"{modality}_valid") is False


def test_advancing_identical_camera_content_eventually_freezes():
    gate = SensorHealthGate(_config())
    report = None
    for frame in (10, 11, 12):
        packet = _packet(frame)
        packet["point_cloud"][0, 0] += frame * 0.001
        report = gate.evaluate(**packet)

    assert SensorHealthReason.CAMERA_FROZEN in report.reasons
    assert report.camera_valid is False


def test_advancing_identical_lidar_content_eventually_freezes():
    gate = SensorHealthGate(_config())
    report = None
    for frame in (10, 11, 12):
        packet = _packet(frame)
        packet["images"][0][0, 0, 0] = 20 + frame
        report = gate.evaluate(**packet)

    assert SensorHealthReason.LIDAR_FROZEN in report.reasons
    assert report.lidar_valid is False


def test_reset_prevents_cross_episode_freeze_state():
    gate = SensorHealthGate(_config())
    for frame in (10, 11):
        gate.evaluate(**_packet(frame))

    gate.reset()
    report = gate.evaluate(**_packet(1))

    assert SensorHealthReason.CAMERA_FROZEN not in report.reasons
    assert SensorHealthReason.LIDAR_FROZEN not in report.reasons


def test_malformed_objects_return_a_report_instead_of_escaping():
    packet = _packet()
    packet["images"] = [object(), object()]

    report = SensorHealthGate(_config()).evaluate(**packet)

    assert report.valid is False
    assert SensorHealthReason.CAMERA_SHAPE in report.reasons


def test_uniterable_metadata_is_contained_as_malformed_packet():
    packet = _packet()
    packet["camera_frames"] = None

    report = SensorHealthGate(_config()).evaluate(**packet)

    assert report.valid is False
    assert report.reasons == (SensorHealthReason.PACKET_MALFORMED,)
    assert np.isinf(report.max_sensor_age_seconds)
