"""Exact-value tests for mask-aware B0 open-loop metrics."""
import pytest

torch = pytest.importorskip("torch")

from training.bc_metrics import (  # noqa: E402
    BCMetricAccumulator,
    wrapped_angle_error,
)

pytestmark = pytest.mark.unit


def test_metrics_use_last_valid_waypoint_for_fde_and_ignore_padding():
    prediction = torch.tensor([[[0.0, 0.0, 0.0, 1.0],
                                [2.0, 0.0, 0.0, 2.0],
                                [99.0, 99.0, 0.0, 99.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    metrics = BCMetricAccumulator(max_speed=12.0)

    metrics.update(
        prediction, target, mask, episode_ids=["episode-a"])
    result = metrics.compute()

    assert result["ade_m"] == pytest.approx(0.5)
    assert result["fde_m"] == pytest.approx(1.0)
    assert result["velocity_mae_mps"] == pytest.approx(0.5)
    assert result["valid_waypoint_count"] == 2
    assert result["sample_count"] == 1


def test_heading_metric_wraps_at_pi():
    error = wrapped_angle_error(
        torch.tensor([3.13]), torch.tensor([-3.13]))

    assert error.item() == pytest.approx(0.023185, abs=1e-5)


def test_metrics_reject_empty_masks_and_count_nonfinite_trajectory():
    prediction = torch.zeros(1, 3, 4)
    target = torch.zeros_like(prediction)
    accumulator = BCMetricAccumulator(max_speed=12.0)

    with pytest.raises(ValueError, match="valid waypoint"):
        accumulator.update(
            prediction, target, torch.zeros(1, 3, dtype=torch.bool),
            episode_ids=["episode-a"])

    prediction[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="prediction.*finite"):
        accumulator.update(
            prediction, target, torch.ones(1, 3, dtype=torch.bool),
            episode_ids=["episode-a"])

    assert accumulator.compute()["nonfinite_trajectory_count"] == 1


def test_metrics_report_episode_and_tag_groups():
    prediction = torch.tensor([
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
    ])
    target = torch.zeros_like(prediction)
    mask = torch.ones(2, 2, dtype=torch.bool)
    accumulator = BCMetricAccumulator(max_speed=12.0)

    accumulator.update(
        prediction, target, mask,
        episode_ids=["episode-a", "episode-b"],
        tags=[{"terrain": "dirt"}, {"terrain": "rock"}])
    result = accumulator.compute()

    assert set(result["by_episode"]) == {"episode-a", "episode-b"}
    assert result["by_episode"]["episode-a"]["ade_m"] == pytest.approx(1.5)
    assert result["by_episode"]["episode-b"]["ade_m"] == pytest.approx(1.5)
    assert result["by_tag"]["terrain"]["dirt"]["sample_count"] == 1
    assert result["by_tag"]["terrain"]["rock"]["sample_count"] == 1


def test_metrics_report_physical_diagnostic_rates():
    prediction = torch.tensor([[[0.0, 0.0, 0.0, 1.0],
                                [-1.0, 0.0, 0.0, -0.5],
                                [2.0, 0.0, 0.0, 13.0]]])
    target = prediction.clone()
    accumulator = BCMetricAccumulator(max_speed=12.0)

    accumulator.update(
        prediction, target, torch.ones(1, 3, dtype=torch.bool),
        episode_ids=["episode-a"])
    result = accumulator.compute()

    assert result["negative_speed_rate"] == pytest.approx(1.0 / 3.0)
    assert result["overspeed_rate"] == pytest.approx(1.0 / 3.0)
    assert result["reverse_waypoint_rate"] == pytest.approx(0.5)
    assert result["smoothness_m"] == pytest.approx(4.0)


def test_metric_accumulation_is_invariant_to_batch_partitioning():
    prediction = torch.tensor([
        [[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 1.0]],
    ])
    target = torch.zeros_like(prediction)
    mask = torch.ones(2, 2, dtype=torch.bool)
    together = BCMetricAccumulator(max_speed=12.0)
    separate = BCMetricAccumulator(max_speed=12.0)

    together.update(
        prediction, target, mask,
        episode_ids=["episode-a", "episode-b"])
    for index, episode in enumerate(("episode-a", "episode-b")):
        separate.update(
            prediction[index:index + 1], target[index:index + 1],
            mask[index:index + 1], episode_ids=[episode])

    together_result = together.compute()
    separate_result = separate.compute()
    for name in ("ade_m", "fde_m", "heading_mae_rad", "velocity_mae_mps"):
        assert together_result[name] == pytest.approx(separate_result[name])
