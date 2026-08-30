"""Unit tests for the off-road reward function (TDD, Phase 3).

Covers: shape/keys, sign of each component, collision grid-sampling correctness,
attitude jitter + rollover limit, jerk, per_step==forward-sum consistency,
differentiability (through occupancy grid_sample), and numpy input support.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip the whole module if torch missing

from policy.offroad_reward import OffRoadReward, OffRoadRewardConfig  # noqa: E402

pytestmark = pytest.mark.unit


def _reward(**kw):
    return OffRoadReward(OffRoadRewardConfig(**kw))


def test_forward_shape_and_keys():
    rew = _reward()
    traj = torch.zeros(2, 6, 4)
    traj[..., 2] = 0.0                      # heading 0 (forward = +x)
    traj[..., 0] = torch.arange(6).float()  # move +x
    traj[..., 3] = 1.0
    r, comps = rew(traj)
    assert r.shape == (2,)
    for k in ("progress", "collision", "attitude", "jerk", "total"):
        assert k in comps and comps[k].shape == (2,)
    assert torch.allclose(comps["total"], r)


def test_progress_positive_when_advancing():
    rew = _reward(w_collision=0.0, w_attitude_jitter=0.0,
                  w_attitude_limit=0.0, w_jerk=0.0)
    traj = torch.zeros(1, 5, 4)
    traj[..., 2] = 0.0
    traj[..., 0] = torch.arange(5).float()
    _, comps = rew(traj)
    assert comps["progress"].item() > 0


def test_collision_penalty_at_obstacle():
    cfg = dict(w_progress=0.0, w_attitude_jitter=0.0,
               w_attitude_limit=0.0, w_jerk=0.0,
               bev_x_range=(-10, 10), bev_y_range=(-10, 10),
               bev_resolution=1.0)
    rew = _reward(**cfg)                      # BEV 20x20, cols<-x, rows<-y
    H = W = 20
    occ = torch.zeros(1, H, W)
    occ[0, 10, 15] = 1.0                      # obstacle at (x=5, y=0)
    traj = torch.zeros(1, 1, 4)
    traj[0, 0, 0] = 5.0                       # hits obstacle
    _, c1 = rew(traj, occupancy=occ)
    assert c1["collision"].item() < 0
    traj[0, 0, 0] = 0.0                       # free point
    _, c2 = rew(traj, occupancy=occ)
    assert abs(c2["collision"].item()) < 1e-6


def test_attitude_jitter_and_rollover_limit():
    rew = _reward(w_progress=0.0, w_collision=0.0, w_jerk=0.0)
    traj = torch.zeros(1, 4, 4)
    # jitter with sub-limit amplitudes -> negative (jitter term only)
    att = torch.zeros(1, 4, 2)
    att[0, :, 0] = torch.tensor([0., 0.1, -0.1, 0.1])
    _, c_jit = rew(traj, attitude=att)
    assert c_jit["attitude"].item() < 0
    # single waypoint under the rollover limit -> ~0 (no jitter, no breach)
    traj1 = torch.zeros(1, 1, 4)
    att_small = torch.zeros(1, 1, 2)
    att_small[0, 0, 0] = 0.1                       # |pitch|=0.1 < 0.4 limit
    _, c_small = rew(traj1, attitude=att_small)
    assert abs(c_small["attitude"].item()) < 1e-6
    # single waypoint beyond the rollover limit -> negative (limit term only)
    att_big = torch.zeros(1, 1, 2)
    att_big[0, 0, 0] = 1.0                         # |pitch|=1.0 > 0.4 limit
    _, c_big = rew(traj1, attitude=att_big)
    assert c_big["attitude"].item() < 0
    assert c_big["attitude"].item() < c_small["attitude"].item()


def test_jerk_penalty_zero_for_constant_speed():
    rew = _reward(w_progress=0.0, w_collision=0.0,
                  w_attitude_jitter=0.0, w_attitude_limit=0.0)
    traj = torch.zeros(1, 6, 4)
    traj[..., 3] = 2.0                       # constant speed -> no jerk
    _, c1 = rew(traj)
    assert abs(c1["jerk"].item()) < 1e-6
    traj[..., 3] = torch.tensor([0., 1., 2., 2., 1., 0.])  # accel/decel
    _, c2 = rew(traj)
    assert c2["jerk"].item() < 0


def test_per_step_equals_forward_sum():
    rew = _reward()
    torch.manual_seed(0)
    traj = torch.randn(3, 7, 4) * 0.5
    att = torch.randn(3, 7, 2) * 0.05           # below limit; small jitter
    occ = torch.rand(3, 20, 20) * 0.3
    r_total, _ = rew(traj, occupancy=occ, attitude=att)
    r_step = rew.per_step(traj, occupancy=occ, attitude=att)
    assert torch.allclose(r_step.sum(-1), r_total, rtol=1e-4, atol=1e-4)


def test_differentiable_through_occupancy():
    rew = _reward(w_progress=0.0, w_attitude_jitter=0.0,
                  w_attitude_limit=0.0, w_jerk=0.0,
                  bev_x_range=(-10, 10), bev_y_range=(-10, 10),
                  bev_resolution=1.0)
    traj = torch.zeros(1, 3, 4, requires_grad=True)
    traj.data[..., 0] = torch.tensor([3., 4., 5.])
    occ = torch.zeros(1, 20, 20, requires_grad=True)
    r, _ = rew(traj, occupancy=occ)
    r.backward()
    assert traj.grad is not None
    assert torch.isfinite(traj.grad).all()


def test_numpy_input_supported():
    rew = _reward()
    traj = np.zeros((1, 5, 4), dtype=np.float32)
    traj[0, :, 0] = [0, 1, 2, 3, 4]
    traj[0, :, 2] = 0.0
    r, _ = rew(traj)
    assert r.shape[0] == 1


def test_outside_occupancy_grid_is_conservatively_penalized():
    reward = OffRoadReward()
    traj = torch.tensor([[[1000.0, 1000.0, 0.0, 1.0]]])
    occupancy = torch.zeros(1, reward.bev_h, reward.bev_w)
    _, components = reward(traj, occupancy=occupancy)
    assert components["collision"].item() < 0.0
