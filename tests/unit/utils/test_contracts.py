import pytest

from orad_ros2.vehicle_control_node import PurePursuitConfig
from perception.bev_fusion import BEVFusionConfig
from policy.hybrid_policy import HybridPolicyConfig
from safety.kinematic_filter import SafetyFilterConfig
from sim.carla_closed_loop import ClosedLoopConfig
from utils.contracts import validate_stack_configs

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
