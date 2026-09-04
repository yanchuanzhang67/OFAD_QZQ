from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from configuration.system import load_system_stack
import replay.carla_pipeline as pipeline_module

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs" / "system.yaml"


def _artifacts_module():
    return importlib.import_module("replay.artifacts")


def test_model_loader_requires_complete_checkpoints_or_explicit_random_mode(
        tmp_path):
    stack = load_system_stack(_CONFIG)
    load = pipeline_module.load_model_bundle

    with pytest.raises(ValueError, match="both checkpoints.*allow_random_models"):
        load(
            stack, perception_checkpoint=None, policy_checkpoint=None,
            allow_random_models=False, seed=42)

    checkpoint = tmp_path / "perception.pt"
    torch.save({}, checkpoint)
    with pytest.raises(ValueError, match="both perception and policy"):
        load(
            stack, perception_checkpoint=checkpoint, policy_checkpoint=None,
            allow_random_models=False, seed=42)


def test_random_model_loader_is_deterministic_and_explicitly_non_acceptance():
    stack = load_system_stack(_CONFIG)
    load = pipeline_module.load_model_bundle

    first = load(
        stack, perception_checkpoint=None, policy_checkpoint=None,
        allow_random_models=True, seed=42)
    second = load(
        stack, perception_checkpoint=None, policy_checkpoint=None,
        allow_random_models=True, seed=42)

    first_parameter = next(first.perception.parameters()).detach()
    second_parameter = next(second.perception.parameters()).detach()
    assert torch.equal(first_parameter, second_parameter)
    assert first.model_mode == "random-model-network-smoke"
    assert first.checkpoint_hashes == {}
    assert first.model_performance_valid is False
    assert first.closed_loop_acceptance_valid is False


def test_checkpoint_loader_is_strict_and_never_falls_back_to_random(tmp_path):
    stack = load_system_stack(_CONFIG)
    perception = tmp_path / "perception.pt"
    policy = tmp_path / "policy.pt"
    torch.save({}, perception)
    torch.save({}, policy)

    with pytest.raises(RuntimeError, match="Missing key"):
        pipeline_module.load_model_bundle(
            stack, perception_checkpoint=perception,
            policy_checkpoint=policy, allow_random_models=False, seed=42)


def test_run_writer_is_transactional_exclusive_and_preserves_gaps(tmp_path):
    module = _artifacts_module()
    metadata = {
        "recorded_vehicle_blueprint": None,
        "recorded_config_sha256": None,
        "provenance_gaps": [
            "vehicle_blueprint", "config_sha256", "calibration_sha256"],
    }
    writer = module.ReplayRunWriter.create(tmp_path, "run-001", metadata)

    assert writer.working_path.parent.name == ".incomplete"
    assert not (writer.working_path / "_SUCCESS").exists()
    writer.write_frame({
        "frame_id": 1, "health_valid": False, "health_reasons": ["imu_jump"],
        "model_invoked": False, "safety_mode": "emergency_stop",
        "maximum_brake": True,
        "health_latency_ms": 1.0, "model_latency_ms": 0.0,
        "end_to_end_latency_ms": 1.5,
        "finite": {"command": True},
    })
    summary = module.summarize_replay_records(writer.records)
    final_path = writer.complete(summary)

    assert final_path == tmp_path / "run-001"
    assert (final_path / "_SUCCESS").read_text(encoding="utf-8") == "complete\n"
    assert json.loads((final_path / "run.json").read_text()) == metadata
    saved_summary = json.loads((final_path / "summary.json").read_text())
    assert saved_summary["total_frames"] == 1
    assert saved_summary["rejected_frames"] == 1
    assert saved_summary["fail_safe_frames"] == 1
    assert saved_summary["maximum_brake_frames"] == 1
    assert saved_summary["safety_mode_counts"] == {"emergency_stop": 1}

    with pytest.raises(FileExistsError):
        module.ReplayRunWriter.create(tmp_path, "run-001", metadata)


def test_summary_reports_distributions_and_observed_rejection_rate():
    module = _artifacts_module()
    records = [
        {
            "health_valid": False, "health_reasons": ["imu_jump"],
            "model_invoked": False, "safety_mode": "emergency_stop",
            "maximum_brake": True,
            "health_latency_ms": 1.0, "model_latency_ms": 0.0,
            "end_to_end_latency_ms": 2.0,
            "finite": {"command": True},
        },
        {
            "health_valid": True, "health_reasons": [],
            "model_invoked": True, "safety_mode": "normal",
            "maximum_brake": False,
            "health_latency_ms": 3.0, "model_latency_ms": 4.0,
            "end_to_end_latency_ms": 8.0,
            "finite": {"bev": True, "policy": True, "command": True},
        },
    ]

    summary = module.summarize_replay_records(records)

    assert summary["total_frames"] == 2
    assert summary["valid_frames"] == 1
    assert summary["observed_rejection_rate"] == 0.5
    assert summary["rejection_reasons"] == {"imu_jump": 1}
    assert summary["maximum_brake_frames"] == 1
    assert summary["safety_mode_counts"] == {"emergency_stop": 1, "normal": 1}
    assert summary["health_latency_ms"] == {"p50": 2.0, "p95": 2.9, "max": 3.0}
    assert summary["network_boundary_finite_rate"] == 1.0


def test_replay_cli_help_exposes_only_explicit_model_and_provenance_modes():
    result = subprocess.run(
        [sys.executable, "scripts/replay_carla_pipeline.py", "--help"],
        cwd=_ROOT, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    for option in (
            "--config", "--episode", "--output-root", "--max-frames",
            "--allow-legacy-provenance", "--allow-random-models",
            "--perception-checkpoint", "--policy-checkpoint", "--seed"):
        assert option in result.stdout
