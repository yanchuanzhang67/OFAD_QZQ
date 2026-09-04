"""Unit contracts for the deterministic B0 Pure BC policy."""
import pytest

torch = pytest.importorskip("torch")

from policy.bc_policy import BCPolicy, masked_bc_loss_components  # noqa: E402
from policy.config import BCPolicyConfig  # noqa: E402

pytestmark = pytest.mark.unit


def _cfg(**overrides):
    values = {
        "bev_channels": 8,
        "bev_h": 10,
        "bev_w": 10,
        "imu_in_channels": 6,
        "imu_steps": 4,
        "ego_dim": 8,
        "encoder_hidden": 32,
        "ego_hidden": 16,
        "latent_dim": 32,
        "horizon": 5,
        "traj_dim": 4,
    }
    values.update(overrides)
    return BCPolicyConfig(**values)


def _inputs(config, batch=2):
    return (
        torch.randn(
            batch, config.bev_channels, config.bev_h, config.bev_w),
        torch.randn(batch, config.imu_steps, config.imu_in_channels),
        torch.randn(batch, config.ego_dim),
    )


def test_pure_bc_forward_is_deterministic_and_has_no_world_model():
    config = _cfg()
    policy = BCPolicy(config).eval()
    bev, imu, ego = _inputs(config)

    with torch.no_grad():
        first = policy(bev, imu, ego)
        second = policy(bev, imu, ego)

    assert first.shape == (2, config.horizon, config.traj_dim)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    forbidden = ("rssm", "prior", "posterior", "critic", "reward_head")
    assert not any(
        any(token in name for token in forbidden)
        for name, _ in policy.named_parameters())


@pytest.mark.parametrize("input_name", ["bev", "imu", "ego"])
def test_pure_bc_rejects_nonfinite_inputs(input_name):
    config = _cfg()
    policy = BCPolicy(config)
    bev, imu, ego = _inputs(config, batch=1)
    values = {"bev": bev, "imu": imu, "ego": ego}
    values[input_name].flatten()[0] = float("nan")

    with pytest.raises(ValueError, match=f"{input_name}.*finite"):
        policy(values["bev"], values["imu"], values["ego"])


@pytest.mark.parametrize(
    ("input_name", "replacement"),
    [
        ("bev", torch.zeros(1, 8, 9, 10)),
        ("imu", torch.zeros(1, 3, 6)),
        ("ego", torch.zeros(1, 7)),
    ],
)
def test_pure_bc_rejects_wrong_input_shapes(input_name, replacement):
    config = _cfg()
    policy = BCPolicy(config)
    bev, imu, ego = _inputs(config, batch=1)
    values = {"bev": bev, "imu": imu, "ego": ego}
    values[input_name] = replacement

    with pytest.raises(ValueError, match=f"{input_name} shape"):
        policy(values["bev"], values["imu"], values["ego"])


def test_pure_bc_rejects_nonfloating_input():
    config = _cfg()
    policy = BCPolicy(config)
    bev, imu, ego = _inputs(config, batch=1)

    with pytest.raises(TypeError, match="ego.*floating"):
        policy(bev, imu, ego.to(torch.int64))


def test_pure_bc_rejects_non_tensor_rank_and_batch_mismatch():
    config = _cfg()
    policy = BCPolicy(config)
    bev, imu, ego = _inputs(config, batch=2)
    with pytest.raises(TypeError, match="bev.*torch.Tensor"):
        policy(bev.numpy(), imu, ego)
    with pytest.raises(ValueError, match="bev shape"):
        policy(bev[0], imu, ego)
    with pytest.raises(ValueError, match="batch sizes"):
        policy(bev, imu[:1], ego)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"family": "hybrid"}, "family"),
        ({"bev_channels": 0}, "dimensions"),
        ({"ego_dim": 7}, "ego_dim"),
        ({"traj_dim": 3}, "traj_dim"),
        ({"waypoint_dt": 0.0}, "waypoint_dt"),
        ({"waypoint_dt": float("nan")}, "waypoint_dt"),
        ({"bc_xy_weight": -1.0}, "weights"),
        ({"bc_xy_weight": float("nan")}, "weights"),
    ],
)
def test_pure_bc_config_rejects_invalid_dimensions_and_weights(
        overrides, message):
    with pytest.raises(ValueError, match=message):
        _cfg(**overrides)


def test_ego_normalization_is_a_strict_checkpoint_buffer():
    policy = BCPolicy(_cfg())
    mean = torch.arange(8.0)
    std = torch.ones(8)

    policy.set_ego_normalization(mean, std)

    state = policy.state_dict()
    assert torch.equal(state["ego_mean"], mean)
    assert torch.equal(state["ego_std"], std)
    with pytest.raises(ValueError, match="std.*positive"):
        policy.set_ego_normalization(torch.zeros(8), torch.zeros(8))
    with pytest.raises(ValueError, match="normalization shape"):
        policy.set_ego_normalization(torch.zeros(7), torch.ones(7))
    with pytest.raises(ValueError, match="normalization.*finite"):
        policy.set_ego_normalization(
            torch.full((8,), float("nan")), torch.ones(8))


def test_masked_loss_wraps_heading_and_ignores_padding():
    config = _cfg(horizon=3)
    prediction = torch.tensor([[[0.0, 0.0, 3.13, 1.0],
                                [1.0, 0.0, 0.0, 2.0],
                                [99.0, 99.0, 0.0, 99.0]]])
    expert = torch.tensor([[[0.0, 0.0, -3.13, 1.0],
                            [1.0, 0.0, 0.0, 2.0],
                            [0.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])

    losses = masked_bc_loss_components(prediction, expert, mask, config)

    assert losses["heading"].item() < 0.001
    assert losses["xy"].item() == 0.0
    assert losses["speed"].item() == 0.0
    assert losses["smooth"].item() == 0.0
    assert losses["valid_waypoints"] == 2


def test_masked_loss_applies_only_complete_smoothness_triplets():
    config = _cfg(horizon=4)
    prediction = torch.tensor([[[0.0, 0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 1.0],
                                [4.0, 0.0, 0.0, 1.0],
                                [9.0, 0.0, 0.0, 1.0]]])
    expert = prediction.clone()

    full = masked_bc_loss_components(
        prediction, expert, torch.ones(1, 4, dtype=torch.bool), config)
    interrupted = masked_bc_loss_components(
        prediction, expert,
        torch.tensor([[True, True, False, True]]), config)

    assert full["smooth"].item() == pytest.approx(2.0)
    assert interrupted["smooth"].item() == 0.0


def test_masked_loss_rejects_empty_mask_and_nonfinite_target():
    config = _cfg()
    prediction = torch.zeros(1, config.horizon, config.traj_dim)
    expert = torch.zeros_like(prediction)
    with pytest.raises(ValueError, match="valid waypoint"):
        masked_bc_loss_components(
            prediction, expert,
            torch.zeros(1, config.horizon, dtype=torch.bool), config)
    expert[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="expert.*finite"):
        masked_bc_loss_components(
            prediction, expert,
            torch.ones(1, config.horizon, dtype=torch.bool), config)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("prediction_shape", "prediction shape"),
        ("expert_shape", "expert shape"),
        ("mask_shape", "valid_mask shape"),
        ("mask_dtype", "bool dtype"),
        ("prediction_nonfinite", "prediction.*finite"),
    ],
)
def test_masked_loss_rejects_malformed_prediction_contract(kind, message):
    config = _cfg()
    prediction = torch.zeros(1, config.horizon, config.traj_dim)
    expert = torch.zeros_like(prediction)
    mask = torch.ones(1, config.horizon, dtype=torch.bool)
    if kind == "prediction_shape":
        prediction = prediction[:, :-1]
        expert = expert[:, :-1]
        mask = mask[:, :-1]
    elif kind == "expert_shape":
        expert = expert[:, :-1]
    elif kind == "mask_shape":
        mask = mask[:, :-1]
    elif kind == "mask_dtype":
        mask = mask.float()
    else:
        prediction[0, 0, 0] = float("inf")

    with pytest.raises((TypeError, ValueError), match=message):
        masked_bc_loss_components(prediction, expert, mask, config)


def test_pure_bc_rejects_nonfinite_decoder_output_and_bc_loss_is_scalar():
    config = _cfg()
    policy = BCPolicy(config)
    bev, imu, ego = _inputs(config, batch=1)
    expert = torch.zeros(1, config.horizon, config.traj_dim)
    mask = torch.ones(1, config.horizon, dtype=torch.bool)
    assert policy.bc_loss(bev, imu, ego, expert, mask).ndim == 0

    class _NonfiniteDecoder(torch.nn.Module):
        def forward(self, latent):
            return torch.full(
                (latent.shape[0], config.horizon, config.traj_dim),
                float("nan"), device=latent.device)

    policy.decoder = _NonfiniteDecoder()
    with pytest.raises(ValueError, match="trajectory.*finite"):
        policy(bev, imu, ego)


def test_bc_loss_backpropagates_to_all_three_encoders_and_decoder():
    config = _cfg()
    policy = BCPolicy(config)
    bev, imu, ego = _inputs(config)
    expert = torch.randn(2, config.horizon, config.traj_dim)
    mask = torch.ones(2, config.horizon, dtype=torch.bool)

    losses = policy.bc_loss_components(bev, imu, ego, expert, mask)
    losses["total"].backward()

    for module in (
            policy.bev_encoder, policy.imu_encoder, policy.ego_encoder,
            policy.fusion, policy.decoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert any(
            gradient is not None and gradient.abs().sum().item() > 0
            for gradient in gradients)
