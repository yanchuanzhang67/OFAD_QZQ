import pytest
import torch

from affordance.terrain import (
    TerrainAffordance, TerrainAffordanceConfig,
    roughness_target_from_imu)

pytestmark = pytest.mark.unit


def test_affordance_output_contract_and_ranges():
    cfg = TerrainAffordanceConfig(in_channels=8, hidden_channels=12)
    model = TerrainAffordance(cfg)
    output = model(torch.randn(2, 8, 10, 14))
    assert output.feature.shape == (2, 12, 10, 14)
    assert output.traversability.shape == (2, 1, 10, 14)
    assert output.roughness.shape == (2, 1, 10, 14)
    assert torch.all((output.traversability >= 0)
                     & (output.traversability <= 1))
    assert torch.all(output.roughness >= 0)


def test_affordance_masked_multitask_loss_is_finite_and_backward():
    model = TerrainAffordance(TerrainAffordanceConfig(
        in_channels=4, hidden_channels=8))
    bev = torch.randn(2, 4, 6, 6, requires_grad=True)
    output = model(bev)
    traversability = torch.ones_like(output.traversability)
    roughness = torch.full_like(output.roughness, 0.3)
    mask = torch.zeros_like(output.roughness)
    mask[:, :, 2:4, 2:4] = 1.0
    losses = model.loss(output, traversability, roughness, mask)
    assert set(losses) == {"traversability", "roughness", "total"}
    assert all(value.ndim == 0 and torch.isfinite(value)
               for value in losses.values())
    losses["total"].backward()
    assert bev.grad is not None and torch.isfinite(bev.grad).all()


def test_roughness_weak_target_is_speed_conditioned_and_stable():
    smooth = torch.zeros(2, 10, 6)
    bumpy = smooth.clone()
    bumpy[:, :, 2] = torch.tensor([[-2.0, 2.0] * 5] * 2)
    speed = torch.tensor([2.0, 0.0])
    smooth_target = roughness_target_from_imu(smooth, speed)
    bumpy_target = roughness_target_from_imu(bumpy, speed)
    assert smooth_target.shape == (2,)
    assert torch.isfinite(bumpy_target).all()
    assert bumpy_target[0] > smooth_target[0]
    assert bumpy_target[1] >= 0.0


def test_affordance_rejects_wrong_bev_channels():
    model = TerrainAffordance(TerrainAffordanceConfig(in_channels=8))
    with pytest.raises(ValueError, match="channels"):
        model(torch.zeros(1, 7, 5, 5))


@pytest.mark.parametrize("kwargs", [
    {"in_channels": 0}, {"hidden_channels": 0},
    {"traversability_weight": -1.0}, {"roughness_weight": -1.0},
    {"speed_epsilon": 0.0},
])
def test_affordance_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TerrainAffordanceConfig(**kwargs)


def test_affordance_loss_rejects_shape_mismatches_and_supports_full_map():
    model = TerrainAffordance(TerrainAffordanceConfig(in_channels=4))
    output = model(torch.zeros(1, 4, 3, 3))
    target = torch.zeros_like(output.traversability)
    losses = model.loss(output, target, torch.zeros_like(output.roughness))
    assert torch.isfinite(losses["total"])
    with pytest.raises(ValueError, match="traversability"):
        model.loss(output, target[:, :, :2], torch.zeros_like(output.roughness))
    with pytest.raises(ValueError, match="roughness target"):
        model.loss(output, target, target[:, :, :2])
    with pytest.raises(ValueError, match="mask"):
        model.loss(output, target, torch.zeros_like(output.roughness),
                   target[:, :, :2])


@pytest.mark.parametrize("imu,speed,match", [
    (torch.zeros(1, 2, 3), torch.ones(1), "imu"),
    (torch.zeros(1, 2, 6), torch.ones(2), "speed"),
])
def test_roughness_target_rejects_bad_shapes(imu, speed, match):
    with pytest.raises(ValueError, match=match):
        roughness_target_from_imu(imu, speed)
    if match == "speed":
        with pytest.raises(ValueError, match="epsilon"):
            roughness_target_from_imu(imu, torch.ones(1), epsilon=0.0)
