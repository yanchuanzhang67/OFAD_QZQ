"""Mask-aware open-loop metrics for B0 trajectory policies."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence

import torch

__all__ = ["BCMetricAccumulator", "wrapped_angle_error"]


def wrapped_angle_error(
        prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return absolute angular error wrapped to ``[-pi, pi]``."""
    delta = prediction - target
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


@dataclass
class _MetricTotals:
    distance_sum: float = 0.0
    waypoint_count: int = 0
    fde_sum: float = 0.0
    sample_count: int = 0
    heading_sum: float = 0.0
    velocity_sum: float = 0.0
    smoothness_sum: float = 0.0
    smoothness_count: int = 0
    negative_speed_count: int = 0
    overspeed_count: int = 0
    reverse_count: int = 0
    transition_count: int = 0

    @staticmethod
    def _ratio(numerator: float, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    def result(self, *, nonfinite_count: int = 0) -> dict[str, float | int]:
        return {
            "ade_m": self._ratio(self.distance_sum, self.waypoint_count),
            "fde_m": self._ratio(self.fde_sum, self.sample_count),
            "heading_mae_rad": self._ratio(
                self.heading_sum, self.waypoint_count),
            "velocity_mae_mps": self._ratio(
                self.velocity_sum, self.waypoint_count),
            "smoothness_m": self._ratio(
                self.smoothness_sum, self.smoothness_count),
            "negative_speed_rate": self._ratio(
                self.negative_speed_count, self.waypoint_count),
            "overspeed_rate": self._ratio(
                self.overspeed_count, self.waypoint_count),
            "reverse_waypoint_rate": self._ratio(
                self.reverse_count, self.transition_count),
            "nonfinite_trajectory_count": int(nonfinite_count),
            "valid_waypoint_count": self.waypoint_count,
            "sample_count": self.sample_count,
        }


class BCMetricAccumulator:
    """Accumulate metrics by counts so batching cannot change the result."""

    def __init__(self, max_speed: float):
        if not math.isfinite(max_speed) or max_speed <= 0:
            raise ValueError("max_speed must be finite and positive")
        self.max_speed = float(max_speed)
        self._total = _MetricTotals()
        self._by_episode: dict[str, _MetricTotals] = {}
        self._by_tag: dict[str, dict[str, _MetricTotals]] = {}
        self._nonfinite_count = 0

    def update(
            self, prediction: torch.Tensor, target: torch.Tensor,
            valid_mask: torch.Tensor, *, episode_ids: Sequence[str],
            tags: Optional[Sequence[Mapping[str, str]]] = None) -> None:
        self._validate_batch(
            prediction, target, valid_mask, episode_ids, tags)
        if not torch.isfinite(prediction).all():
            invalid = ~torch.isfinite(prediction).flatten(1).all(dim=1)
            self._nonfinite_count += int(invalid.sum().item())
            raise ValueError("prediction must contain only finite values")
        if not torch.isfinite(target).all():
            raise ValueError("target must contain only finite values")
        if not valid_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid waypoint")

        tag_values = tags or tuple({} for _ in episode_ids)
        for index, episode_id in enumerate(episode_ids):
            sample_prediction = prediction[index]
            sample_target = target[index]
            sample_mask = valid_mask[index]
            states = [self._total]
            states.append(self._by_episode.setdefault(
                str(episode_id), _MetricTotals()))
            for name, value in tag_values[index].items():
                value_states = self._by_tag.setdefault(
                    str(name), {})
                states.append(value_states.setdefault(
                    str(value), _MetricTotals()))
            for state in states:
                self._update_state(
                    state, sample_prediction, sample_target, sample_mask)

    @staticmethod
    def _validate_batch(
            prediction: torch.Tensor, target: torch.Tensor,
            valid_mask: torch.Tensor, episode_ids: Sequence[str],
            tags: Optional[Sequence[Mapping[str, str]]]) -> None:
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("prediction must be a torch.Tensor")
        if prediction.ndim != 3 or prediction.shape[-1] != 4:
            raise ValueError("prediction shape must be (B,N,4)")
        if target.shape != prediction.shape:
            raise ValueError("target shape must match prediction shape")
        if valid_mask.shape != prediction.shape[:2]:
            raise ValueError("valid_mask shape must be (B,N)")
        if valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must have bool dtype")
        if len(episode_ids) != prediction.shape[0]:
            raise ValueError("episode_ids length must match batch size")
        if tags is not None and len(tags) != prediction.shape[0]:
            raise ValueError("tags length must match batch size")

    def _update_state(
            self, state: _MetricTotals, prediction: torch.Tensor,
            target: torch.Tensor, mask: torch.Tensor) -> None:
        distances = torch.linalg.vector_norm(
            prediction[..., :2] - target[..., :2], dim=-1)
        heading = wrapped_angle_error(prediction[..., 2], target[..., 2])
        velocity = (prediction[..., 3] - target[..., 3]).abs()
        valid_count = int(mask.sum().item())
        last_index = int(torch.nonzero(mask, as_tuple=False)[-1].item())

        state.distance_sum += float(distances[mask].sum().item())
        state.heading_sum += float(heading[mask].sum().item())
        state.velocity_sum += float(velocity[mask].sum().item())
        state.waypoint_count += valid_count
        state.fde_sum += float(distances[last_index].item())
        state.sample_count += 1
        valid_speed = prediction[..., 3][mask]
        state.negative_speed_count += int((valid_speed < 0).sum().item())
        state.overspeed_count += int(
            (valid_speed > self.max_speed).sum().item())

        if prediction.shape[0] >= 2:
            pair_mask = mask[:-1] & mask[1:]
            delta_x = prediction[1:, 0] - prediction[:-1, 0]
            state.reverse_count += int((delta_x[pair_mask] < 0).sum().item())
            state.transition_count += int(pair_mask.sum().item())
        if prediction.shape[0] >= 3:
            triplet_mask = mask[:-2] & mask[1:-1] & mask[2:]
            second_difference = (
                prediction[2:, :2]
                - 2.0 * prediction[1:-1, :2]
                + prediction[:-2, :2])
            norms = torch.linalg.vector_norm(second_difference, dim=-1)
            state.smoothness_sum += float(norms[triplet_mask].sum().item())
            state.smoothness_count += int(triplet_mask.sum().item())

    def compute(self) -> dict[str, object]:
        """Return global and grouped metrics as JSON-compatible values."""
        result: dict[str, object] = self._total.result(
            nonfinite_count=self._nonfinite_count)
        result["by_episode"] = {
            name: state.result()
            for name, state in sorted(self._by_episode.items())
        }
        result["by_tag"] = {
            tag_name: {
                tag_value: state.result()
                for tag_value, state in sorted(values.items())
            }
            for tag_name, values in sorted(self._by_tag.items())
        }
        return result
