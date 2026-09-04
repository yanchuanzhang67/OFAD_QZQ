"""Layer 3: hybrid policy network (IL + RL).

Implemented modules (Phase 3):

* ``hybrid_policy.py``  - ``HybridPolicy``: Behavior-Cloning pretrain head +
                          Dreamer-style world-model (RSSM) RL finetune.
  Components: ``PolicyEncoder``, ``RSSM``, ``WorldModel``, ``Actor``,
  ``ActionProj``, ``Critic``.
* ``offroad_reward.py`` - ``OffRoadReward``: differentiable multi-component
                          reward (progress / collision / attitude / jerk).

Public re-exports for convenience.
"""
from .offroad_reward import OffRoadReward, OffRoadRewardConfig  # noqa: E402,F401
from .config import BCPolicyConfig  # noqa: E402,F401
from .hybrid_policy import (  # noqa: E402,F401
    HybridPolicy,
    HybridPolicyConfig,
    WorldModel,
    RSSM,
)

__all__ = [
    "HybridPolicy",
    "HybridPolicyConfig",
    "WorldModel",
    "RSSM",
    "OffRoadReward",
    "OffRoadRewardConfig",
    "BCPolicyConfig",
]
