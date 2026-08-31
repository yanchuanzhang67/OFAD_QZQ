from types import SimpleNamespace

import numpy as np
import pytest

import sim.carla_closed_loop as closed_loop
from orad_ros2.vehicle_control_node import PurePursuitController
from safety.kinematic_filter import SafetyFilter, SafetyFilterConfig
from sim.carla_closed_loop import CarlaClosedLoopRunner, ClosedLoopConfig
from utils.sensor_health import (
    SensorHealthConfig,
    SensorHealthGate,
    SensorHealthReason,
)
from utils.types import BEVFeature

pytestmark = pytest.mark.unit


class _Vehicle:
    def __init__(self):
        self.applied = []

    def get_transform(self):
        return SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0),
            rotation=SimpleNamespace(pitch=0.0, roll=0.0),
            get_forward_vector=lambda: SimpleNamespace(x=1.0, y=0.0),
        )

    def get_velocity(self):
        return SimpleNamespace(x=0.0, y=0.0)

    def get_control(self):
        return SimpleNamespace(steer=0.0)

    def apply_control(self, control):
        self.applied.append(control)


class _Sensors:
    last_frame = 7
    last_camera_frames = (7,)
    last_lidar_frame = 7
    last_imu_frames = (6, 7)
    last_camera_timestamps = (0.7,)
    last_lidar_timestamp = 0.7
    last_imu_timestamps = (0.6, 0.7)
    calibration_version = "test-cal-v1"
    last_sensor_timestamps = {
        "camera_0": 0.7, "lidar": 0.7, "imu_latest": 0.7,
    }

    def reset_episode(self):
        pass

    def tick(self):
        return (
            [np.arange(48, dtype=np.uint8).reshape(4, 4, 3) + 20],
            np.array([[10.0, 10.0, 0.0, 0.5]], dtype=np.float32),
            np.array([
                [0.0, 0.0, 9.7, 0.0, 0.0, 0.0],
                [0.0, 0.0, 9.8, 0.0, 0.0, 0.0],
            ], dtype=np.float32),
        )

    def collision_stats(self):
        return 0, 0.0


class _TimeoutSensors(_Sensors):
    def tick(self):
        return None


class _EmptyLidarSensors(_Sensors):
    def tick(self):
        images, _, imu = super().tick()
        return images, np.zeros((0, 4), dtype=np.float32), imu


class _BlackCameraSensors(_Sensors):
    def tick(self):
        images, points, imu = super().tick()
        return [np.zeros_like(images[0])], points, imu


class _StaleLidarSensors(_Sensors):
    last_lidar_timestamp = 0.4


class _CalibrationMismatchSensors(_Sensors):
    calibration_version = "test-cal-v2"


def _health_config(**overrides):
    values = {
        "num_cameras": 1,
        "image_size": (4, 4),
        "imu_steps": 2,
        "expected_calibration_version": "test-cal-v1",
        "max_sensor_age_seconds": 0.2,
        "max_sensor_skew_seconds": 0.05,
        "black_pixel_threshold": 2,
        "saturation_pixel_threshold": 253,
        "max_black_ratio": 0.98,
        "max_saturation_ratio": 0.98,
        "max_consecutive_identical_frames": 2,
        "min_lidar_points": 1,
        "min_lidar_valid_ratio": 1.0,
        "max_lidar_abs_coordinate": 100.0,
        "max_lidar_duplicate_ratio": 1.0,
        "max_accel_abs": 50.0,
        "max_gyro_abs": 5.0,
        "max_accel_step_delta": 20.0,
        "max_gyro_step_delta": 2.0,
        "max_imu_sample_gap_seconds": 0.11,
    }
    values.update(overrides)
    return SensorHealthConfig(**values)


def _health_gate(**overrides):
    return SensorHealthGate(_health_config(**overrides))


def _install_fake_carla(monkeypatch):
    monkeypatch.setattr(
        closed_loop, "carla",
        SimpleNamespace(VehicleControl=lambda **kwargs: kwargs), raising=False)


def test_runner_routes_learned_occupancy_to_safety_without_payload_swap(
        monkeypatch):
    _install_fake_carla(monkeypatch)
    prediction = np.zeros((1, 1, 4, 4), dtype=np.float32)
    prediction[0, 0, 2, 2] = 0.95

    def perceive(images, points, imu, attitude):
        return BEVFeature(
            bev=np.zeros((1, 8, 4, 4), dtype=np.float32),
            occupancy=prediction)

    def policy(bev, imu):
        return np.array([[
            [0.5, 0.5, 0.0, 1.0],
            [1.0, 0.5, 0.0, 1.0],
            [1.5, 0.5, 0.0, 1.0],
        ]], dtype=np.float32)

    vehicle = _Vehicle()
    config = ClosedLoopConfig(
        bev_x_range=(-2.0, 2.0), bev_y_range=(-2.0, 2.0),
        bev_resolution=1.0, num_points=4, imu_steps=2,
        occupancy_source="learned")
    runner = CarlaClosedLoopRunner(
        world=object(), vehicle=vehicle, sensors=_Sensors(),
        perceive=perceive, policy=policy,
        safety_filter=SafetyFilter(SafetyFilterConfig(dt=config.dt)),
        controller=PurePursuitController(), config=config,
        health_gate=_health_gate(), max_steps=1)

    metrics = runner.run()

    assert metrics.raw_policy_risk_steps == 1
    assert metrics.post_safety_risk_steps == 0
    assert metrics.emergency_stops == 1
    assert vehicle.applied[-1]["throttle"] == 0.0
    assert vehicle.applied[-1]["brake"] == 1.0


def test_runner_sensor_timeout_applies_emergency_brake(monkeypatch):
    _install_fake_carla(monkeypatch)
    vehicle = _Vehicle()
    runner = CarlaClosedLoopRunner(
        world=object(), vehicle=vehicle, sensors=_TimeoutSensors(),
        perceive=lambda *args: None, policy=lambda *args: None,
        safety_filter=SafetyFilter(), controller=PurePursuitController(),
        config=ClosedLoopConfig(), health_gate=_health_gate(), max_steps=1)

    metrics = runner.run()

    assert metrics.emergency_stops == 1
    assert vehicle.applied[-1]["brake"] == 1.0


def test_runner_empty_lidar_is_invalid_and_applies_emergency_brake(
        monkeypatch):
    _install_fake_carla(monkeypatch)
    vehicle = _Vehicle()
    perceive_calls = []
    runner = CarlaClosedLoopRunner(
        world=object(), vehicle=vehicle, sensors=_EmptyLidarSensors(),
        perceive=lambda *args: perceive_calls.append(args),
        policy=lambda *args: None,
        safety_filter=SafetyFilter(), controller=PurePursuitController(),
        config=ClosedLoopConfig(), health_gate=_health_gate(), max_steps=1)

    metrics = runner.run()

    assert perceive_calls == []
    assert metrics.emergency_stops == 1
    assert metrics.sensor_health_failures == 1
    assert metrics.sensor_health_failure_reasons == {"lidar_empty": 1}
    assert metrics.to_summary()["sensor_health_latency_p95_ms"] >= 0.0
    assert vehicle.applied[-1]["throttle"] == 0.0
    assert vehicle.applied[-1]["brake"] == 1.0


@pytest.mark.parametrize("sensors,reason", [
    (_BlackCameraSensors(), SensorHealthReason.CAMERA_BLACK),
    (_StaleLidarSensors(), SensorHealthReason.LIDAR_STALE),
    (_CalibrationMismatchSensors(), SensorHealthReason.CALIBRATION_MISMATCH),
])
def test_runner_health_fault_never_calls_models_and_applies_max_brake(
        monkeypatch, sensors, reason):
    _install_fake_carla(monkeypatch)
    vehicle = _Vehicle()
    perceive_calls = []
    policy_calls = []
    runner = CarlaClosedLoopRunner(
        world=object(), vehicle=vehicle, sensors=sensors,
        perceive=lambda *args: perceive_calls.append(args),
        policy=lambda *args: policy_calls.append(args),
        safety_filter=SafetyFilter(), controller=PurePursuitController(),
        config=ClosedLoopConfig(), health_gate=_health_gate(), max_steps=1)

    summary = runner.run().to_summary()

    assert perceive_calls == []
    assert policy_calls == []
    assert summary["sensor_health_failures"] == 1
    assert summary["sensor_health_failure_reasons"][reason.value] == 1
    assert vehicle.applied[-1]["throttle"] == 0.0
    assert vehicle.applied[-1]["brake"] == 1.0


@pytest.mark.parametrize("perceive", [
    lambda *args: (_ for _ in ()).throw(RuntimeError("model failed")),
    lambda *args: BEVFeature(
        bev=np.zeros((1, 8, 4, 4), dtype=np.float32), occupancy=None),
])
def test_runner_model_or_required_occupancy_failure_is_fail_safe(
        monkeypatch, perceive):
    _install_fake_carla(monkeypatch)
    vehicle = _Vehicle()
    config = ClosedLoopConfig(
        bev_x_range=(-2.0, 2.0), bev_y_range=(-2.0, 2.0),
        bev_resolution=1.0, num_points=4, imu_steps=2,
        occupancy_source="learned")
    runner = CarlaClosedLoopRunner(
        world=object(), vehicle=vehicle, sensors=_Sensors(),
        perceive=perceive,
        policy=lambda *args: np.zeros((1, 3, 4), dtype=np.float32),
        safety_filter=SafetyFilter(), controller=PurePursuitController(),
        config=config, health_gate=_health_gate(), max_steps=1)

    metrics = runner.run()

    assert metrics.emergency_stops == 1
    assert vehicle.applied[-1]["brake"] == 1.0
