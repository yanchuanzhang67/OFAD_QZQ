"""Hybrid policy network: Behavior Cloning pretrain + Dreamer-style RL finetune.

Architecture (Phase 3):

  * Shared ``PolicyEncoder``  -- BEV (B,C,H,W) conv stack + IMU history GRU
    fused into an observation feature ``obs_feat`` (B, hidden). Handles both
    single-step (deploy/BC) and sequence (world-model training) inputs.
  * ``RSSM``                  -- Gaussian Recurrent State-Space Model
    (deterministic ``h`` + diagonal-Gaussian stochastic ``z``), providing
    posterior states from real obs and prior imagination rollouts.
  * ``WorldModel``            -- RSSM + observation/BEV/reward/continue heads;
    ``world_model_loss`` = recon + reward MSE + balanced KL (Dreamer training).
  * ``Actor``                 -- latent -> trajectory (B, N, 4) via a GRU
    decoder (the IL/RL policy OUTPUT, also fed to the safety filter).
  * ``ActionProj``            -- trajectory -> compact action for the RSSM.
  * ``Critic``                -- latent -> state value (for the RL return).
  * ``HybridPolicy``          -- ties it together:
      - ``forward(bev, imu)``      BC / deployment (encode -> actor -> traj).
      - ``imagine(bev, imu, H)``   Dreamer imagination rollout in latent space.
      - ``world_model_loss(batch)`` dynamics learning (recon+reward+KL).
      - ``actor_critic_loss(rollout)`` policy-gradient + value losses.

All ops are static-shape (H/N/S fixed); the only Python loop is over fixed
lengths, so the deploy path remains traceable / TensorRT-friendly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .offroad_reward import OffRoadReward

__all__ = ["HybridPolicyConfig", "HybridPolicy", "WorldModel", "RSSM"]


def _mlp(in_dim, hid, out, layers=2, act=nn.ELU) -> nn.Module:
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hid), act()]
        d = hid
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


def _diag_gaussian(params: torch.Tensor):
    """Split (B, 2D) -> (mean, std); std = softplus + 0.1 floor."""
    mean, std = params.chunk(2, dim=-1)
    std = F.softplus(std) + 0.1
    return mean, std


@dataclass
class HybridPolicyConfig:
    """Static dims for the hybrid policy (match perception defaults)."""

    # perception input (must align with BEVFusionConfig outputs)
    bev_channels: int = 32
    bev_h: int = 50                       # rows <- y
    bev_w: int = 50                       # cols <- x
    imu_in_channels: int = 6
    imu_steps: int = 10
    # shared encoder
    enc_hidden: int = 256
    hidden_dim: int = 256
    # RSSM
    deter_dim: int = 256                 # deterministic h
    stoch_dim: int = 32                   # stochastic z (Gaussian)
    action_dim: int = 8                   # compact action for the world model
    # policy output
    horizon: int = 20                     # N trajectory waypoints
    traj_dim: int = 4                     # [x, y, heading, velocity]
    latent_dim: int = 256
    # RL (Dreamer)
    imagine_horizon: int = 5
    discount: float = 0.99
    lambda_: float = 0.95                  # GAE lambda for lambda-return
    ent_coef: float = 1e-4
    cont_coef: float = 1.0
    reward_coef: float = 1.0
    kl_coef: float = 1.0
    balanced_kl: float = 0.8             # Dreamer balanced KL factor
    # BC physical-component weights (relatetalk Stage-2 baseline)
    bc_xy_weight: float = 1.0
    bc_heading_weight: float = 0.2
    bc_speed_weight: float = 0.2
    bc_smooth_weight: float = 0.05


class PolicyEncoder(nn.Module):
    """BEV conv + IMU GRU -> observation feature ``obs_feat``.

    Accepts single-step ``(B,C,H,W)`` + ``(B,T,6)`` OR sequences
    ``(B,S,C,H,W)`` + ``(B,S,T,6)``; returns ``(B, hidden)`` or ``(B,S,hidden)``.
    """

    def __init__(self, cfg: HybridPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.bev_conv = nn.Sequential(
            nn.Conv2d(cfg.bev_channels, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, cfg.enc_hidden), nn.ELU(),
        )
        self.imu_gru = nn.GRU(cfg.imu_in_channels, cfg.enc_hidden,
                              batch_first=True)
        self.imu_proj = nn.Linear(cfg.enc_hidden, cfg.enc_hidden)
        self.fuse = _mlp(2 * cfg.enc_hidden, cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, bev: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        seq = bev.dim() == 5
        if not seq:
            bev = bev.unsqueeze(1)             # (B,1,C,H,W)
            imu = imu.unsqueeze(1)             # (B,1,T,6)
        B, S, C, H, W = bev.shape
        Ti = imu.shape[2]
        bev = bev.reshape(B * S, C, H, W)
        imu = imu.reshape(B * S, Ti, self.cfg.imu_in_channels)
        bf = self.bev_conv(bev)                       # (B*S, enc)
        _, h = self.imu_gru(imu)                      # (1, B*S, enc)
        imf = self.imu_proj(h.squeeze(0))            # (B*S, enc)
        feat = self.fuse(torch.cat([bf, imf], dim=-1))   # (B*S, hidden)
        feat = feat.reshape(B, S, -1)
        return feat if seq else feat.squeeze(1)


class RSSM(nn.Module):
    """Gaussian Recurrent State-Space Model (Dreamer-lite, continuous z).

    State = deterministic ``h`` (GRU) + diagonal-Gaussian stochastic ``z``.

    * ``observe_step`` uses the **posterior** (q(z|h, obs)) -- for world-model
      training on real transitions.
    * ``imagine_step`` uses the **prior** (p(z|h)) -- for RL imagination
      rollouts (no observation available).
    """

    def __init__(self, cfg: HybridPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.cell = nn.GRUCell(cfg.stoch_dim + cfg.action_dim, cfg.deter_dim)
        # prior p(z_t | h_t) -- imagination
        self.prior = _mlp(cfg.deter_dim, cfg.hidden_dim, 2 * cfg.stoch_dim)
        # posterior q(z_t | h_t, obs_t) -- observation
        self.post = _mlp(cfg.deter_dim + cfg.hidden_dim,
                         cfg.hidden_dim, 2 * cfg.stoch_dim)

    def init_state(self, B, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(B, self.cfg.deter_dim, device=device, dtype=dtype)
        z = torch.zeros(B, self.cfg.stoch_dim, device=device, dtype=dtype)
        return h, z

    def observe_step(self, h, z, a, obs_feat):
        """One real step: h = GRU(h, [z,a]); z ~ q(z|h, obs)."""
        h = self.cell(torch.cat([z, a], dim=-1), h)           # next deterministic
        pm, ps = _diag_gaussian(self.prior(h))               # prior params
        qm, qs = _diag_gaussian(self.post(torch.cat([h, obs_feat], dim=-1)))
        z_post = (qm + qs * torch.randn_like(qs)
                  if self.training else qm)                  # deterministic deploy
        return h, z_post, (pm, ps), (qm, qs)

    def imagine_step(self, h, z, a):
        """One imagined step: h = GRU(h, [z,a]); z = p(z|h) mean."""
        h = self.cell(torch.cat([z, a], dim=-1), h)
        pm, ps = _diag_gaussian(self.prior(h))
        z_prior = pm                                          # deterministic rollout
        return h, z_prior, (pm, ps)


class BevDecoder(nn.Module):
    """latent -> (B, C, H, W) BEV reconstruction (matches perception output)."""

    def __init__(self, cfg: HybridPolicyConfig):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.ELU(),
            nn.Linear(cfg.hidden_dim, 128 * 4 * 4), nn.ELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, cfg.bev_channels, 3, padding=1),
            nn.AdaptiveAvgPool2d((cfg.bev_h, cfg.bev_w)),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent).reshape(-1, 128, 4, 4)
        return self.deconv(x)


def _kl_diag(qm, qs, pm, ps) -> torch.Tensor:
    """KL( N(qm,qs) || N(pm,ps) ) per row, summed over the latent dims."""
    kl = torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5
    return kl.sum(-1)


class WorldModel(nn.Module):
    """RSSM + observation/BEV/reward/continue heads."""

    def __init__(self, cfg: HybridPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PolicyEncoder(cfg)
        self.rssm = RSSM(cfg)
        self.latent_proj = _mlp(cfg.deter_dim + cfg.stoch_dim,
                                cfg.hidden_dim, cfg.latent_dim)
        self.obs_decoder = _mlp(cfg.latent_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.bev_decoder = BevDecoder(cfg)
        self.reward_head = _mlp(cfg.latent_dim, cfg.hidden_dim, 1)
        self.cont_head = _mlp(cfg.latent_dim, cfg.hidden_dim, 1)

    def _latent(self, h, z) -> torch.Tensor:
        return self.latent_proj(torch.cat([h, z], dim=-1))

    def initial_state(self, bev, imu):
        """Posterior state estimate from a single (bev, imu) observation."""
        obs = self.encoder(bev, imu)                  # (B, hidden)
        B = obs.shape[0]
        h, z = self.rssm.init_state(B, obs.device, obs.dtype)
        a0 = torch.zeros(B, self.cfg.action_dim, device=obs.device, dtype=obs.dtype)
        h, z_post, _, _ = self.rssm.observe_step(h, z, a0, obs)
        return h, z_post

    def observe_seq(self, bev_seq, imu_seq, action_seq):
        """Roll the posterior over S real steps; return states + obs + dists."""
        obs = self.encoder(bev_seq, imu_seq)          # (B, S, hidden)
        B, S, _ = obs.shape
        h, z = self.rssm.init_state(B, obs.device, obs.dtype)
        hs, zs, priors, posts = [], [], [], []
        for t in range(S):
            h, z, prior, post = self.rssm.observe_step(
                h, z, action_seq[:, t], obs[:, t])
            hs.append(h)
            zs.append(z)
            priors.append(prior)
            posts.append(post)
        return (torch.stack(hs, dim=1), torch.stack(zs, dim=1),
                priors, posts, obs)

    def imagine_seq(self, h0, z0, action_seq):
        """Prior (no-observation) rollout over S imagined steps."""
        h, z = h0, z0
        hs, zs = [], []
        for t in range(action_seq.shape[1]):
            h, z, _ = self.rssm.imagine_step(h, z, action_seq[:, t])
            hs.append(h)
            zs.append(z)
        return torch.stack(hs, dim=1), torch.stack(zs, dim=1)

    def world_model_loss(self, bev_seq, imu_seq, action_seq,
                         reward_seq, cont_seq=None) -> Dict[str, torch.Tensor]:
        """Dynamics objective = recon(obs+BEV) + reward + balanced KL (+ cont).

        Args (sequences; S = static rollout length):
            bev_seq    : (B, S, C, H, W)
            imu_seq    : (B, S, T, 6)
            action_seq : (B, S, action)  -- PREV action aligned to step t.
            reward_seq : (B, S)
            cont_seq   : (B, S) 1=continue / 0=terminal (optional).
        """
        cfg = self.cfg
        B, S = reward_seq.shape
        hs, zs, priors, posts, obs = self.observe_seq(bev_seq, imu_seq, action_seq)
        latents = self._latent(hs, zs)                 # (B, S, latent)

        # -- reconstruction (obs feature + BEV) --
        obs_pred = self.obs_decoder(latents)
        recon = F.mse_loss(obs_pred, obs)
        flat = latents.reshape(B * S, -1)
        bev_pred = self.bev_decoder(flat)             # (B*S, C, H, W)
        bev_tgt = bev_seq.reshape(B * S, *bev_seq.shape[2:])
        recon = recon + F.mse_loss(bev_pred, bev_tgt)

        # -- reward prediction --
        r_pred = self.reward_head(latents).squeeze(-1)   # (B, S)
        reward_loss = F.mse_loss(r_pred, reward_seq)

        # -- continue (terminal) prediction --
        cont_loss = latents.new_zeros(())
        if cont_seq is not None:
            c_pred = self.cont_head(latents).squeeze(-1)   # logits (B, S)
            cont_loss = F.binary_cross_entropy_with_logits(c_pred, cont_seq)

        # -- balanced KL between posterior and prior (Dreamer balancing) --
        bal = cfg.balanced_kl
        prior_kl = latents.new_zeros(())
        post_kl = latents.new_zeros(())
        for t in range(S):
            pm, ps = priors[t]
            qm, qs = posts[t]
            prior_kl = prior_kl + _kl_diag(qm.detach(), qs.detach(), pm, ps)
            post_kl = post_kl + _kl_diag(qm, qs, pm.detach(), ps.detach())
        kl = (bal * prior_kl + (1 - bal) * post_kl).mean()
        prior_std = torch.stack([params[1].mean() for params in priors]).mean()
        posterior_std = torch.stack([params[1].mean() for params in posts]).mean()

        total = (cfg.reward_coef * reward_loss + recon
                 + cfg.cont_coef * cont_loss + cfg.kl_coef * kl)
        return {
            "recon": recon, "reward": reward_loss, "kl": kl,
            "cont": cont_loss, "total": total,
            "prior_std": prior_std.detach(),
            "posterior_std": posterior_std.detach(),
        }


class Actor(nn.Module):
    """latent -> trajectory (B, N, traj_dim) via a GRU decoder.

    The IL/RL policy OUTPUT (also consumed by the safety filter). The same
    module serves BC (supervised on expert trajectories) and RL (differentiable
    through imagination rollouts)."""

    def __init__(self, cfg: HybridPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = _mlp(cfg.latent_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Linear(cfg.hidden_dim, cfg.traj_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:   # (B, latent_dim)
        N = self.cfg.horizon
        x = self.proj(latent).unsqueeze(1).expand(-1, N, -1)   # (B, N, hidden)
        out, _ = self.gru(x)
        return self.head(out)                                  # (B, N, traj_dim)


class ActionProj(nn.Module):
    """Project a full trajectory into a compact action for the RSSM."""

    def __init__(self, cfg: HybridPolicyConfig):
        super().__init__()
        self.mlp = _mlp(cfg.horizon * cfg.traj_dim,
                        cfg.hidden_dim, cfg.action_dim)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:     # (B, N, traj_dim)
        return self.mlp(traj.flatten(1))                       # (B, action_dim)


class Critic(nn.Module):
    """latent -> state value V(s)."""

    def __init__(self, cfg: HybridPolicyConfig):
        super().__init__()
        self.mlp = _mlp(cfg.latent_dim, cfg.hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:    # (B, latent_dim)
        return self.mlp(latent).squeeze(-1)                     # (B,)


class HybridPolicy(nn.Module):
    """IL (BC) pretrain + Dreamer-style RL finetune policy."""

    def __init__(self, config: Optional[HybridPolicyConfig] = None):
        super().__init__()
        self.config = config or HybridPolicyConfig()
        self.world_model = WorldModel(self.config)
        self.actor = Actor(self.config)
        self.action_proj = ActionProj(self.config)
        self.critic = Critic(self.config)
        # The off-road reward function (used to label real rollouts so the
        # reward_head learns it; also usable directly on imagined trajectories).
        self.reward = OffRoadReward()

    # -- BC / deployment ---------------------------------------------------
    def encode(self, bev, imu):
        """Single-step posterior state -> latent (B, latent_dim)."""
        h, z = self.world_model.initial_state(bev, imu)
        return self.world_model._latent(h, z), h, z

    def forward(self, bev, imu) -> torch.Tensor:
        """BC / deployment: (bev, imu) -> trajectory (B, N, 4)."""
        latent, _, _ = self.encode(bev, imu)
        return self.actor(latent)

    def bc_loss(self, bev, imu, expert_traj) -> torch.Tensor:
        """Weighted physical-component BC objective."""
        return self.bc_loss_components(bev, imu, expert_traj)["total"]

    def bc_loss_components(self, bev, imu, expert_traj
                           ) -> Dict[str, torch.Tensor]:
        prediction = self.forward(bev, imu)
        if prediction.shape != expert_traj.shape:
            raise ValueError(
                f"expert trajectory shape {tuple(expert_traj.shape)} does not "
                f"match prediction {tuple(prediction.shape)}")
        xy = F.mse_loss(prediction[..., :2], expert_traj[..., :2])
        heading = F.mse_loss(prediction[..., 2], expert_traj[..., 2])
        speed = F.mse_loss(prediction[..., 3], expert_traj[..., 3])
        if prediction.shape[1] >= 3:
            second_difference = (prediction[:, 2:, :2]
                                 - 2.0 * prediction[:, 1:-1, :2]
                                 + prediction[:, :-2, :2])
            smooth = second_difference.square().mean()
        else:
            smooth = prediction.new_zeros(())
        c = self.config
        total = (c.bc_xy_weight * xy + c.bc_heading_weight * heading
                 + c.bc_speed_weight * speed + c.bc_smooth_weight * smooth)
        return {"xy": xy, "heading": heading, "speed": speed,
                "smooth": smooth, "total": total}

    # -- RL: Dreamer imagination ------------------------------------------
    def imagine(self, bev, imu, horizon: Optional[int] = None
                ) -> Dict[str, torch.Tensor]:
        """Latent-space rollout; returns trajectories/rewards/values.

        The initial (posterior) state is detached so that actor gradients flow
        only through the imagined actions (the world model is treated as a
        fixed dynamics function during the RL step; train it separately with
        :meth:`world_model_loss`)."""
        cfg = self.config
        H = cfg.imagine_horizon if horizon is None else int(horizon)
        h, z = self.world_model.initial_state(bev, imu)
        h, z = h.detach(), z.detach()
        lats, trajs, rews, vals, conts = [], [], [], [], []
        for _ in range(H):
            latent = self.world_model._latent(h, z)
            traj = self.actor(latent)                           # (B, N, 4)
            a = self.action_proj(traj)                          # (B, action)
            r = self.world_model.reward_head(latent).squeeze(-1)   # (B,)
            v = self.critic(latent)                             # (B,)
            c = self.world_model.cont_head(latent).squeeze(-1)      # logits (B,)
            lats.append(latent)
            trajs.append(traj)
            rews.append(r)
            vals.append(v)
            conts.append(c)
            h, z, _ = self.world_model.rssm.imagine_step(h, z, a)
        return {
            "latents": torch.stack(lats, dim=1),            # (B, H, latent)
            "trajectories": torch.stack(trajs, dim=1),      # (B, H, N, 4)
            "rewards": torch.stack(rews, dim=1),             # (B, H)
            "values": torch.stack(vals, dim=1),             # (B, H)
            "continue": torch.stack(conts, dim=1),          # (B, H) logits
        }

    def actor_critic_loss(self, rollout: Dict[str, torch.Tensor]
                          ) -> Dict[str, torch.Tensor]:
        """Dreamer actor-critic: lambda-return over the imagined rollout.

        * Actor maximizes the imagined discounted lambda-return (gradients flow
          through the imagined rewards -> states -> actions -> actor; the world
          model is frozen / treated as fixed dynamics here).
        * Critic regresses V toward the (detached) lambda-return.
        """
        cfg = self.config
        r = rollout["rewards"]                                  # (B, H)
        v = rollout["values"]                                  # (B, H)
        cont = torch.sigmoid(rollout["continue"])             # (B, H)
        disc, lam = cfg.discount, cfg.lambda_
        H = r.shape[1]
        vd = v.detach()                                        # value as baseline
        returns = torch.empty_like(r)
        g = vd[:, -1]
        for t in reversed(range(H)):
            next_v = vd[:, t + 1] if t + 1 < H else vd[:, -1]
            delta = r[:, t] + disc * cont[:, t] * next_v - vd[:, t]
            g = delta + disc * cont[:, t] * lam * g
            returns[:, t] = g
        # entropy proxy: temporal variance of the trajectory (exploration)
        ent = rollout["trajectories"].var(dim=2).mean()
        actor_loss = -returns.mean() - cfg.ent_coef * ent
        critic_loss = F.mse_loss(v, returns.detach())
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "return": returns.mean().detach(),
            "entropy": ent.detach(),
        }
