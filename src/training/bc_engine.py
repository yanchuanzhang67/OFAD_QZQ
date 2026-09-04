"""Deterministic frozen-perception training engine for the Pure BC baseline."""
from __future__ import annotations

import random
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn

from policy.bc_policy import BCPolicy, masked_bc_loss_components

from .bc_dataset import BCBatch
from .bc_metrics import BCMetricAccumulator

__all__ = [
    "fit_ego_normalization",
    "set_deterministic_seed",
    "train_one_epoch",
    "validate_one_epoch",
]


_LOSS_NAMES = ("xy", "heading", "speed", "smooth", "total")


def set_deterministic_seed(seed: int) -> dict[str, object]:
    """Seed supported RNGs and report the requested deterministic backend."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic_algorithms = True
    limitation = None
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError) as error:  # pragma: no cover - old/backend
        deterministic_algorithms = False
        limitation = f"{type(error).__name__}: {error}"
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "torch_deterministic_algorithms": deterministic_algorithms,
        "bitwise_determinism_claimed": deterministic_algorithms,
        "limitation": limitation,
    }


def fit_ego_normalization(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute float64 population statistics from the training split only."""
    if len(dataset) == 0:
        raise ValueError("training dataset must not be empty")
    values = []
    for index in range(len(dataset)):
        sample = dataset[index]
        ego = getattr(sample, "ego", None)
        if not isinstance(ego, torch.Tensor):
            raise TypeError("training sample ego must be a torch.Tensor")
        if tuple(ego.shape) != (8,):
            raise ValueError("training sample ego shape must be (8,)")
        values.append(ego.detach().to(device="cpu", dtype=torch.float64))
    stacked = torch.stack(values)
    if not torch.isfinite(stacked).all():
        raise ValueError("training ego normalization values must be finite")
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("training ego normalization statistics must be finite")
    if not (std > 1e-6).all():
        indices = torch.nonzero(std <= 1e-6, as_tuple=False).flatten().tolist()
        raise ValueError(
            "training ego standard deviation must exceed 1e-6 for every "
            f"field; degenerate indices: {indices}")
    return mean.to(torch.float32), std.to(torch.float32)


def _move_batch(batch: BCBatch, device: torch.device) -> BCBatch:
    if not isinstance(batch, BCBatch):
        raise TypeError("BC loader must yield BCBatch values")
    return BCBatch(
        images=batch.images.to(device),
        lidar=batch.lidar.to(device),
        imu=batch.imu.to(device),
        ego=batch.ego.to(device),
        expert=batch.expert.to(device),
        mask=batch.mask.to(device),
        sample_ids=batch.sample_ids,
        episode_ids=batch.episode_ids,
        tags=batch.tags,
    )


def _freeze_perception(perception: nn.Module, device: torch.device) -> None:
    perception.to(device)
    perception.eval()
    perception.requires_grad_(False)
    for parameter in perception.parameters():
        parameter.grad = None


def _extract_losses(
        losses: Mapping[str, torch.Tensor | int]) -> tuple[dict[str, float], int]:
    missing = [name for name in _LOSS_NAMES if name not in losses]
    if missing or "valid_waypoints" not in losses:
        raise ValueError(f"BC loss components are missing keys: {missing}")
    values = {}
    for name in _LOSS_NAMES:
        value = losses[name]
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TypeError(f"BC loss component {name} must be a scalar tensor")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"BC loss component {name} must be finite")
        values[name] = float(value.detach().item())
    valid_waypoints = losses["valid_waypoints"]
    if not isinstance(valid_waypoints, int) or valid_waypoints <= 0:
        raise ValueError("BC valid_waypoints must be a positive integer")
    return values, valid_waypoints


class _EpochTotals:
    def __init__(self) -> None:
        self.loss_sums = {name: 0.0 for name in _LOSS_NAMES}
        self.valid_waypoints = 0
        self.steps = 0
        self.samples = 0

    def update(
            self, losses: Mapping[str, torch.Tensor | int],
            batch_size: int) -> None:
        values, valid_waypoints = _extract_losses(losses)
        for name, value in values.items():
            self.loss_sums[name] += value * valid_waypoints
        self.valid_waypoints += valid_waypoints
        self.steps += 1
        self.samples += batch_size

    def result(self) -> dict[str, float | int]:
        if self.steps == 0 or self.valid_waypoints == 0:
            raise ValueError("BC loader must yield at least one valid batch")
        result: dict[str, float | int] = {
            f"loss_{name}": total / self.valid_waypoints
            for name, total in self.loss_sums.items()
        }
        result.update({
            "steps": self.steps,
            "sample_count": self.samples,
            "valid_waypoint_count": self.valid_waypoints,
        })
        return result


def _bev_feature(
        perception: nn.Module, batch: BCBatch,
        device: torch.device) -> torch.Tensor:
    modality_mask = torch.ones(
        batch.size, 2, dtype=torch.bool, device=device)
    feature = perception(
        batch.images,
        batch.lidar,
        batch.imu,
        modality_mask=modality_mask,
    )
    bev = getattr(feature, "bev", None)
    if not isinstance(bev, torch.Tensor):
        raise TypeError("perception must return a BEVFeature with tensor bev")
    if not torch.isfinite(bev).all():
        raise FloatingPointError("frozen perception BEV must be finite")
    return bev


def train_one_epoch(
        perception: nn.Module, policy: BCPolicy,
        loader: Iterable[BCBatch], optimizer: torch.optim.Optimizer,
        device: torch.device) -> dict[str, float | int]:
    """Train only ``BCPolicy`` while perception remains frozen and grad-free."""
    _freeze_perception(perception, device)
    policy.to(device)
    policy.train()
    totals = _EpochTotals()
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.no_grad():
            bev = _bev_feature(perception, batch, device)
        losses = policy.bc_loss_components(
            bev, batch.imu, batch.ego, batch.expert, batch.mask)
        _extract_losses(losses)
        optimizer.zero_grad(set_to_none=True)
        total_loss = losses["total"]
        if not isinstance(total_loss, torch.Tensor):  # pragma: no cover - checked
            raise TypeError("BC total loss must be a tensor")
        total_loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in policy.parameters())
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("BC gradient must be finite")
        optimizer.step()
        if not all(torch.isfinite(parameter).all()
                   for parameter in policy.parameters()):
            raise FloatingPointError("BC optimizer update produced non-finite weights")
        totals.update(losses, batch.size)
    return totals.result()


def validate_one_epoch(
        perception: nn.Module, policy: BCPolicy,
        loader: Iterable[BCBatch], device: torch.device,
        *, max_speed: float) -> dict[str, object]:
    """Evaluate validation loss and metrics without updating either network."""
    _freeze_perception(perception, device)
    policy.to(device)
    policy.eval()
    totals = _EpochTotals()
    metrics = BCMetricAccumulator(max_speed=max_speed)
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            bev = _bev_feature(perception, batch, device)
            prediction = policy(bev, batch.imu, batch.ego)
            losses = masked_bc_loss_components(
                prediction, batch.expert, batch.mask, policy.config)
            totals.update(losses, batch.size)
            metrics.update(
                prediction, batch.expert, batch.mask,
                episode_ids=batch.episode_ids, tags=batch.tags)
    result: dict[str, object] = totals.result()
    result.update(metrics.compute())
    return result
