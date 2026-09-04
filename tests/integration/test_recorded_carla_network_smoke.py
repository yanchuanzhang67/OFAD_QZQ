from pathlib import Path

import numpy as np
import pytest
import torch

from configuration.system import load_system_stack
from perception.bev_fusion import BEVFusion
from policy.hybrid_policy import HybridPolicy
from replay.carla_dataset import RecordedCarlaFrame
import replay.carla_pipeline as pipeline_module
from utils.sensor_health import SensorHealthGate
from utils.types import VehicleState

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).parents[2]
_CONFIG = _ROOT / "configs" / "system.yaml"


def _valid_frame() -> RecordedCarlaFrame:
    rng = np.random.default_rng(42)
    images = tuple(
        rng.integers(10, 240, size=(192, 192, 3), dtype=np.uint8)
        for _ in range(3))
    raw = rng.uniform(
        [-10.0, -10.0, -1.0, 0.0],
        [10.0, 10.0, 1.0, 1.0],
        size=(512, 4)).astype(np.float32)
    canonical = raw[:256].copy()
    imu = np.tile(
        np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], np.float32),
        (10, 1))
    return RecordedCarlaFrame(
        sample_index=10, frame_id=200, timestamp=20.0, images=images,
        raw_point_cloud=raw, canonical_point_cloud=canonical,
        imu_history=imu, ego_state=VehicleState(speed=1.0),
        camera_frames=(200, 200, 200), lidar_frame=200,
        imu_frames=tuple(range(191, 201)),
        camera_timestamps=(20.0, 20.0, 20.0), lidar_timestamp=20.0,
        imu_timestamps=tuple(value / 10.0 for value in range(191, 201)),
        sensor_skew_seconds=0.0, action_source="test", expert_label=False)


def test_real_network_modules_accept_one_recorded_carla_frame_on_cpu():
    stack = load_system_stack(_CONFIG)
    torch.manual_seed(42)
    np.random.seed(42)
    models = pipeline_module.ReplayModelBundle(
        perception=BEVFusion(stack.bev).eval(),
        policy=HybridPolicy(stack.policy).eval(),
        model_mode="random-model-network-smoke",
        checkpoint_hashes={},
        model_performance_valid=False,
        closed_loop_acceptance_valid=False,
    )
    pipeline = pipeline_module.CarlaReplayPipeline(
        stack, SensorHealthGate(stack.sensor_health), models)

    record = pipeline.process(_valid_frame())

    assert record["model_invoked"] is True
    assert record["shapes"]["bev"] == [1, 32, 50, 50]
    assert record["shapes"]["policy"] == [1, 20, 4]
    assert record["finite"] == {"bev": True, "policy": True, "command": True}
    assert record["safety_mode"] in {"normal", "degraded", "emergency_stop"}
    assert record["model_performance_valid"] is False
    assert record["closed_loop_acceptance_valid"] is False
