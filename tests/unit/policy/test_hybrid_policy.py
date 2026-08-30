"""Unit tests for the hybrid policy network (TDD, Phase 3).

Covers: BC forward shape, bc_loss backprop to actor, imagination rollout
shapes, world_model_loss backprop, actor_critic_loss backprop to actor & critic,
and RSSM imagine_seq shapes. Uses a small config so CPU tests are fast.
"""
import pytest

torch = pytest.importorskip("torch")  # skip the whole module if torch missing

from policy.hybrid_policy import (  # noqa: E402
    HybridPolicy,
    HybridPolicyConfig,
)

pytestmark = pytest.mark.unit


def _cfg(**kw):
    base = dict(
        bev_channels=8, bev_h=10, bev_w=10,
        imu_in_channels=6, imu_steps=4,
        enc_hidden=32, hidden_dim=32,
        deter_dim=32, stoch_dim=8, action_dim=4,
        horizon=5, traj_dim=4, latent_dim=32,
        imagine_horizon=4,
    )
    base.update(kw)
    return HybridPolicyConfig(**base)


def _obs(cfg, B=2, device="cpu"):
    bev = torch.randn(B, cfg.bev_channels, cfg.bev_h, cfg.bev_w, device=device)
    imu = torch.randn(B, cfg.imu_steps, cfg.imu_in_channels, device=device)
    return bev, imu


def _wm_batch(cfg, B=2, S=3, device="cpu"):
    bev_seq = torch.randn(B, S, cfg.bev_channels, cfg.bev_h, cfg.bev_w, device=device)
    imu_seq = torch.randn(B, S, cfg.imu_steps, cfg.imu_in_channels, device=device)
    act_seq = torch.randn(B, S, cfg.action_dim, device=device)
    rew_seq = torch.randn(B, S, device=device)
    cont_seq = torch.ones(B, S, device=device)
    return bev_seq, imu_seq, act_seq, rew_seq, cont_seq


def test_forward_output_shape():
    cfg = _cfg()
    pol = HybridPolicy(cfg)
    bev, imu = _obs(cfg, B=2)
    traj = pol(bev, imu)
    assert traj.shape == (2, cfg.horizon, cfg.traj_dim)


def test_eval_forward_is_deterministic():
    cfg = _cfg()
    policy = HybridPolicy(cfg).eval()
    bev, imu = _obs(cfg, B=1)
    with torch.no_grad():
        first = policy(bev, imu)
        second = policy(bev, imu)
    assert torch.equal(first, second)


def test_bc_loss_backward_to_actor():
    cfg = _cfg()
    pol = HybridPolicy(cfg)
    bev, imu = _obs(cfg, B=2)
    expert = torch.randn(2, cfg.horizon, cfg.traj_dim)
    loss = pol.bc_loss(bev, imu, expert)
    assert loss.dim() == 0
    loss.backward()
    grads = [p.grad for p in pol.actor.parameters()]
    assert all(g is not None for g in grads)
    assert any((g.abs().sum() > 0).item() for g in grads)


def test_bc_loss_reports_weighted_physical_components():
    cfg = _cfg(bc_xy_weight=1.0, bc_heading_weight=0.2,
               bc_speed_weight=0.2, bc_smooth_weight=0.05)
    policy = HybridPolicy(cfg)
    bev, imu = _obs(cfg, B=2)
    expert = torch.zeros(2, cfg.horizon, cfg.traj_dim)
    losses = policy.bc_loss_components(bev, imu, expert)
    assert set(losses) == {"xy", "heading", "speed", "smooth", "total"}
    expected = (losses["xy"] + 0.2 * losses["heading"]
                + 0.2 * losses["speed"] + 0.05 * losses["smooth"])
    assert torch.allclose(losses["total"], expected)
    assert all(value.ndim == 0 and torch.isfinite(value)
               for value in losses.values())


def test_imagine_rollout_shapes():
    cfg = _cfg()
    pol = HybridPolicy(cfg)
    bev, imu = _obs(cfg, B=2)
    out = pol.imagine(bev, imu)
    H = cfg.imagine_horizon
    assert out["latents"].shape == (2, H, cfg.latent_dim)
    assert out["trajectories"].shape == (2, H, cfg.horizon, cfg.traj_dim)
    assert out["rewards"].shape == (2, H)
    assert out["values"].shape == (2, H)
    assert out["continue"].shape == (2, H)


def test_world_model_loss_backward():
    cfg = _cfg()
    pol = HybridPolicy(cfg)
    bev_seq, imu_seq, act_seq, rew_seq, cont_seq = _wm_batch(cfg, S=3)
    losses = pol.world_model.world_model_loss(
        bev_seq, imu_seq, act_seq, rew_seq, cont_seq)
    for k in ("recon", "reward", "kl", "cont", "total"):
        assert k in losses and losses[k].dim() == 0
    assert losses["prior_std"].item() > 0
    assert losses["posterior_std"].item() > 0
    losses["total"].backward()
    g = [p.grad for p in pol.world_model.parameters()]
    assert any((x is not None and (x.abs().sum() > 0).item()) for x in g)


def test_actor_critic_loss_backward():
    cfg = _cfg()
    pol = HybridPolicy(cfg)
    bev, imu = _obs(cfg, B=2)
    out = pol.imagine(bev, imu)
    losses = pol.actor_critic_loss(out)
    losses["actor_loss"].backward()
    assert any((p.grad is not None and (p.grad.abs().sum() > 0).item())
               for p in pol.actor.parameters())
    pol.zero_grad(set_to_none=True)
    out2 = pol.imagine(bev, imu)
    losses2 = pol.actor_critic_loss(out2)
    losses2["critic_loss"].backward()
    assert any((p.grad is not None and (p.grad.abs().sum() > 0).item())
               for p in pol.critic.parameters())


def test_rssm_imagine_seq_shapes():
    cfg = _cfg()
    pol = HybridPolicy(cfg)
    bev, imu = _obs(cfg, B=2)
    h0, z0 = pol.world_model.initial_state(bev, imu)
    acts = torch.randn(2, 3, cfg.action_dim)
    hs, zs = pol.world_model.imagine_seq(h0, z0, acts)
    assert hs.shape == (2, 3, cfg.deter_dim)
    assert zs.shape == (2, 3, cfg.stoch_dim)
