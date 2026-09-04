from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configuration.system import load_system_stack
from replay.carla_dataset import RecordedCarlaFrame
from utils.sensor_health import SensorHealthGate, SensorHealthReason
from utils.types import VehicleState

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs" / "system.yaml"


def make_frame(*, invalid_imu: bool = False) -> RecordedCarlaFrame:
    images = tuple(
        np.full((192, 192, 3), 30 + index, dtype=np.uint8)
        for index in range(3))
    raw = np.array(
        [[float(index + 1), 0.2, 0.0, 0.5] for index in range(32)],
        dtype=np.float32)
    canonical = np.zeros((256, 4), dtype=np.float32)
    canonical[:32] = raw
    imu = np.tile(
        np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        (10, 1))
    if invalid_imu:
        imu[0, :3] = [1000.0, -800.0, 200.0]
    return RecordedCarlaFrame(
        sample_index=0 if invalid_imu else 1,
        frame_id=100,
        timestamp=10.0,
        images=images,
        raw_point_cloud=raw,
        canonical_point_cloud=canonical,
        imu_history=imu,
        ego_state=VehicleState(speed=2.0),
        camera_frames=(100, 100, 100),
        lidar_frame=100,
        imu_frames=tuple(range(91, 101)),
        camera_timestamps=(10.0, 10.0, 10.0),
        lidar_timestamp=10.0,
        imu_timestamps=tuple(value / 10.0 for value in range(91, 101)),
        sensor_skew_seconds=0.0,
        action_source="traffic_manager_smoke",
        expert_label=False,
    )


def _pipeline_module():
    return importlib.import_module("replay.carla_pipeline")


def test_invalid_startup_imu_is_rejected_before_observation() -> None:
    module = _pipeline_module()
    stack = load_system_stack(_CONFIG)

    prepared = module.prepare_recorded_frame(
        make_frame(invalid_imu=True), SensorHealthGate(stack.sensor_health),
        calibration_version=stack.sensor_health.expected_calibration_version)

    assert prepared.observation is None
    assert prepared.startup_context == "startup"
    assert SensorHealthReason.IMU_ACCEL_RANGE in prepared.health.reasons
    assert SensorHealthReason.IMU_JUMP in prepared.health.reasons
    assert prepared.health_latency_ms >= 0.0
    assert module.maximum_brake_record(
        prepared, stack.controller.max_decel) == {
            "steering": 0.0,
            "speed": 0.0,
            "accel": -5.0,
            "safety_mode": "emergency_stop",
            "model_invoked": False,
        }


def test_valid_recorded_frame_constructs_canonical_observation() -> None:
    module = _pipeline_module()
    stack = load_system_stack(_CONFIG)

    prepared = module.prepare_recorded_frame(
        make_frame(), SensorHealthGate(stack.sensor_health),
        calibration_version=stack.sensor_health.expected_calibration_version)

    observation = prepared.observation
    assert prepared.health.valid is True
    assert prepared.startup_context == "startup"
    assert observation is not None
    assert observation.frame == "ego"
    assert observation.simulator_frame == 100
    assert observation.point_cloud.shape == (256, 4)
    assert observation.imu_history.shape == (10, 6)
    assert observation.max_sensor_skew_seconds == 0.0


def test_pipeline_never_calls_models_for_health_rejection() -> None:
    module = _pipeline_module()
    stack = load_system_stack(_CONFIG)

    class ForbiddenModel:
        calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("invalid frame reached a model")

    perception = ForbiddenModel()
    policy = ForbiddenModel()
    models = module.ReplayModelBundle(
        perception=perception,
        policy=policy,
        model_mode="test-models",
        checkpoint_hashes={},
        model_performance_valid=False,
        closed_loop_acceptance_valid=False,
    )
    pipeline = module.CarlaReplayPipeline(
        stack, SensorHealthGate(stack.sensor_health), models)

    record = pipeline.process(make_frame(invalid_imu=True))

    assert perception.calls == policy.calls == 0
    assert record["model_invoked"] is False
    assert record["safety_mode"] == "emergency_stop"
    assert record["command"]["accel"] == stack.controller.max_decel
    assert record["maximum_brake"] is True


def test_pipeline_passes_canonical_tensors_through_fake_models() -> None:
    module = _pipeline_module()
    stack = load_system_stack(_CONFIG)

    class FakePerception:
        def __init__(self):
            self.inputs = None

        def __call__(self, images, points, imu, *, modality_mask):
            self.inputs = (images, points, imu, modality_mask)
            return SimpleNamespace(
                bev=torch.zeros(1, 32, 50, 50),
                occupancy=torch.zeros(1, 1, 50, 50),
            )

    class FakePolicy:
        def __init__(self):
            self.calls = 0

        def __call__(self, bev, imu):
            self.calls += 1
            trajectory = torch.zeros(1, 20, 4)
            trajectory[0, :, 0] = torch.arange(1.0, 21.0)
            trajectory[0, :, 3] = 2.0
            return trajectory

    perception = FakePerception()
    policy = FakePolicy()
    models = module.ReplayModelBundle(
        perception=perception,
        policy=policy,
        model_mode="test-models",
        checkpoint_hashes={},
        model_performance_valid=False,
        closed_loop_acceptance_valid=False,
    )
    record = module.CarlaReplayPipeline(
        stack, SensorHealthGate(stack.sensor_health), models).process(make_frame())

    images, points, imu, mask = perception.inputs
    assert images.shape == (1, 3, 3, 192, 192)
    assert points.shape == (1, 256, 4)
    assert imu.shape == (1, 10, 6)
    assert mask.tolist() == [[True, True]]
    assert images.dtype == points.dtype == imu.dtype == torch.float32
    assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0
    assert policy.calls == 1
    assert record["shapes"]["bev"] == [1, 32, 50, 50]
    assert record["shapes"]["policy"] == [1, 20, 4]
    assert record["finite"]["bev"] is True
    assert record["finite"]["policy"] is True
    assert np.isfinite(list(record["command"].values())).all()
    assert abs(record["command"]["steering"]) <= stack.controller.max_steering
    assert stack.controller.max_decel <= record["command"]["accel"] <= (
        stack.controller.max_accel)
    assert record["maximum_brake"] == (
        record["command"]["accel"] == stack.controller.max_decel)
