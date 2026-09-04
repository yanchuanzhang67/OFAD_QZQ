"""Finite and deterministic training-engine contracts for Pure BC."""
from __future__ import annotations

from types import MethodType

import pytest

torch = pytest.importorskip("torch")

from perception.bev_fusion import BEVFusion, BEVFusionConfig  # noqa: E402
from policy.bc_policy import BCPolicy  # noqa: E402
from policy.config import BCPolicyConfig  # noqa: E402
from training.bc_dataset import BCBatch  # noqa: E402
from training.bc_engine import (  # noqa: E402
    fit_ego_normalization,
    set_deterministic_seed,
    train_one_epoch,
    validate_one_epoch,
)

pytestmark = pytest.mark.unit


def _perception() -> BEVFusion:
    return BEVFusion(BEVFusionConfig(
        num_cameras=2,
        image_size=(16, 16),
        num_points=16,
        bev_channels=8,
        bev_x_range=(-2.5, 2.5),
        bev_y_range=(-2.5, 2.5),
        bev_resolution=0.5,
        imu_steps=4,
    ))


def _policy() -> BCPolicy:
    return BCPolicy(BCPolicyConfig(
        bev_channels=8,
        bev_h=10,
        bev_w=10,
        imu_steps=4,
        encoder_hidden=16,
        ego_hidden=8,
        latent_dim=16,
        horizon=5,
    ))


def _batch() -> BCBatch:
    return BCBatch(
        images=torch.rand(2, 2, 3, 16, 16),
        lidar=torch.rand(2, 16, 4),
        imu=torch.rand(2, 4, 6),
        ego=torch.tensor([
            [1.0, 0.1, 0.01, 0.02, 0.8, 0.1, 0.03, 9.8],
            [2.0, 0.2, 0.02, 0.03, 1.8, 0.2, 0.04, 9.9],
        ]),
        expert=torch.rand(2, 5, 4),
        mask=torch.tensor([
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]),
        sample_ids=("ep-1:1", "ep-2:1"),
        episode_ids=("ep-1", "ep-2"),
        tags=({"terrain": "dirt"}, {"terrain": "rock"}),
    )


def _clone_state(module):
    return {name: value.detach().clone()
            for name, value in module.state_dict().items()}


def _state_equal(module, before) -> bool:
    return all(torch.equal(value, before[name])
               for name, value in module.state_dict().items())


def test_train_step_freezes_perception_and_updates_bc_policy():
    set_deterministic_seed(41)
    perception = _perception()
    policy = _policy()
    before_perception = _clone_state(perception)
    before_policy = _clone_state(policy)

    result = train_one_epoch(
        perception,
        policy,
        [_batch()],
        torch.optim.Adam(policy.parameters(), lr=1e-3),
        torch.device("cpu"),
    )

    assert result["steps"] == 1
    assert result["sample_count"] == 2
    assert result["valid_waypoint_count"] == 8
    assert _state_equal(perception, before_perception)
    assert not _state_equal(policy, before_policy)
    assert all(parameter.grad is None for parameter in perception.parameters())


def test_train_epoch_fails_on_nonfinite_gradient():
    perception = _perception()
    policy = _policy()

    def _nan_loss(self, bev, imu, ego, expert, valid_mask):
        total = next(self.parameters()).sum() * float("nan")
        zero = total.detach().new_zeros(())
        return {
            "xy": zero,
            "heading": zero,
            "speed": zero,
            "smooth": zero,
            "total": total,
            "valid_waypoints": int(valid_mask.sum().item()),
        }

    policy.bc_loss_components = MethodType(_nan_loss, policy)
    with pytest.raises(FloatingPointError, match="loss|gradient"):
        train_one_epoch(
            perception,
            policy,
            [_batch()],
            torch.optim.Adam(policy.parameters()),
            torch.device("cpu"),
        )


def test_normalization_is_fit_only_from_train_dataset():
    class _EgoDataset:
        def __init__(self):
            self.values = [
                torch.tensor([float(index) for index in range(8)]),
                torch.tensor([float(index + 2) for index in range(8)]),
            ]

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return type("Sample", (), {"ego": self.values[index]})()

    mean, std = fit_ego_normalization(_EgoDataset())

    assert torch.equal(mean, torch.arange(8, dtype=torch.float32) + 1.0)
    assert torch.equal(std, torch.ones(8))


def test_normalization_rejects_degenerate_or_nonfinite_training_field():
    class _Dataset:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return type("Sample", (), {"ego": self.values[index]})()

    constant = _Dataset([torch.ones(8), torch.ones(8)])
    with pytest.raises(ValueError, match="standard deviation"):
        fit_ego_normalization(constant)

    nonfinite = _Dataset([torch.ones(8), torch.full((8,), float("nan"))])
    with pytest.raises(ValueError, match="finite"):
        fit_ego_normalization(nonfinite)


def test_validation_never_updates_either_model_and_reports_ade():
    perception = _perception()
    policy = _policy()
    before_perception = _clone_state(perception)
    before_policy = _clone_state(policy)

    result = validate_one_epoch(
        perception, policy, [_batch()], torch.device("cpu"), max_speed=12.0)

    assert result["steps"] == 1
    assert result["sample_count"] == 2
    assert result["ade_m"] >= 0.0
    assert _state_equal(perception, before_perception)
    assert _state_equal(policy, before_policy)
    assert all(parameter.grad is None for parameter in perception.parameters())
    assert all(parameter.grad is None for parameter in policy.parameters())
