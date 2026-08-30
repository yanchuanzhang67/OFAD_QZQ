"""CPU contract: sensor tensors -> BEV -> policy -> safety -> control."""
import numpy as np
import pytest
import torch

from orad_ros2.vehicle_control_node import PurePursuitController
from perception.bev_fusion import BEVFusion, BEVFusionConfig
from policy.hybrid_policy import HybridPolicy, HybridPolicyConfig
from safety.kinematic_filter import SafetyFilter, SafetyFilterConfig
from sim.carla_closed_loop import ego_trajectory_to_world, occupancy_from_points
from utils.types import VehicleState

pytestmark = pytest.mark.integration


def test_cpu_pipeline_shape_frame_safety_contract():
    bev_cfg = BEVFusionConfig(
        num_cameras=2, image_size=(16, 16), num_points=16,
        cam_feat_channels=4, pillar_feat_channels=4, bev_channels=8,
        imu_hidden=8, imu_steps=4, bev_x_range=(-5, 5),
        bev_y_range=(-5, 5), bev_resolution=1.0)
    policy_cfg = HybridPolicyConfig(
        bev_channels=8, bev_h=10, bev_w=10, imu_steps=4,
        enc_hidden=16, hidden_dim=16, deter_dim=16, stoch_dim=4,
        action_dim=4, horizon=5, latent_dim=16)
    perception = BEVFusion(bev_cfg).eval()
    policy = HybridPolicy(policy_cfg).eval()
    images = torch.zeros(1, 2, 3, 16, 16)
    points = torch.zeros(1, 16, 4)
    imu = torch.zeros(1, 4, 6)
    with torch.no_grad():
        feature = perception(images, points, imu)
        raw = policy(feature.bev, imu)
    assert feature.bev.shape == (1, 8, 10, 10)
    assert feature.occupancy.shape == (1, 1, 10, 10)
    assert raw.shape == (1, 5, 4)
    assert torch.isfinite(raw).all()

    # Use a known physically meaningful policy trajectory for downstream frame
    # and safety contracts; random untrained network values are not acceptance data.
    ego = np.array([[i + 1.0, 0.0, 0.0, 2.0] for i in range(5)], np.float32)
    state = VehicleState(x=10.0, y=5.0, yaw=np.pi / 2, speed=1.0)
    trajectory = ego_trajectory_to_world(ego, state)
    occupancy = occupancy_from_points(
        np.zeros((0, 4), np.float32), (-5, 5), (-5, 5), 1.0,
        vehicle_state=state)
    safe = SafetyFilter(SafetyFilterConfig(collision_check=False)).filter(
        trajectory, state, occupancy)
    command = PurePursuitController().compute(state, safe)
    assert safe.length == 5
    assert np.isfinite([command.steering, command.speed, command.accel]).all()
    assert abs(command.steering) <= 0.5
    assert -5.0 <= command.accel <= 3.0
