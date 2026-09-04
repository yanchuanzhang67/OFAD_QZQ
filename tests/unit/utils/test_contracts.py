import numpy as np
import pytest

from orad_ros2.vehicle_control_node import PurePursuitConfig
from perception.bev_fusion import BEVFusionConfig
from policy.config import BCPolicyConfig
from policy.hybrid_policy import HybridPolicyConfig
from safety.kinematic_filter import SafetyFilterConfig
from sim.carla_closed_loop import ClosedLoopConfig
from utils.contracts import validate_stack_configs
from utils.types import (
    EGO_DYNAMICS_V1_FIELDS,
    EgoDynamicsV1,
    VehicleState,
)

pytestmark = pytest.mark.unit


def test_default_stack_configs_are_compatible():
    validate_stack_configs(
        BEVFusionConfig(), HybridPolicyConfig(), SafetyFilterConfig(),
        PurePursuitConfig(), ClosedLoopConfig())


def test_stack_config_drift_fails_fast():
    with pytest.raises(ValueError, match="policy.bev_w"):
        validate_stack_configs(
            BEVFusionConfig(), HybridPolicyConfig(bev_w=49),
            SafetyFilterConfig(), PurePursuitConfig(), ClosedLoopConfig())


def test_pure_bc_stack_config_drift_fails_fast():
    with pytest.raises(ValueError, match="bc_policy.bev_w"):
        validate_stack_configs(
            BEVFusionConfig(), HybridPolicyConfig(), SafetyFilterConfig(),
            PurePursuitConfig(), ClosedLoopConfig(),
            bc_policy=BCPolicyConfig(bev_w=49))


def test_ego_dynamics_v1_excludes_world_pose_and_preserves_order():
    state = VehicleState(
        x=100.0, y=-20.0, yaw=1.5, speed=3.0, steering=0.1,
        pitch=0.2, roll=-0.3, vx=2.5, vy=-0.4,
        yaw_rate=0.05, accel_z=9.7)

    dynamics = EgoDynamicsV1.from_vehicle_state(state)

    assert EGO_DYNAMICS_V1_FIELDS == (
        "speed", "steering", "pitch", "roll",
        "vx", "vy", "yaw_rate", "accel_z")
    np.testing.assert_array_equal(
        dynamics.to_array(),
        np.array(
            [3.0, 0.1, 0.2, -0.3, 2.5, -0.4, 0.05, 9.7],
            dtype=np.float32))


def test_ego_dynamics_v1_rejects_nonfinite_vehicle_state():
    with pytest.raises(ValueError, match="ego-dynamics-v1.*finite"):
        EgoDynamicsV1.from_vehicle_state(VehicleState(speed=float("nan")))


def test_ego_dynamics_v1_rejects_wrong_direct_length():
    with pytest.raises(ValueError, match="length 8"):
        EgoDynamicsV1((1.0, 2.0))
