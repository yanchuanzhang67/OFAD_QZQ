"""CPU integration contract for Pure BC fixture training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from configuration.system import load_system_stack
from perception.bev_fusion import BEVFusion, BEVFusionConfig
from policy.bc_policy import BCPolicy
from policy.config import BCPolicyConfig
from replay.carla_dataset import CarlaRecordedEpisode
from tests.unit.replay.test_carla_dataset import make_episode
from training.bc_artifacts import resolve_checkpoint_index
from training.bc_dataset import BCBatch
from training.bc_engine import evaluate_loader

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).parents[2]
_CONFIG = _ROOT / "configs" / "system.yaml"


def _vary_ego_dynamics(episode: Path) -> None:
    path = episode / "frames.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["ego_state"].update({
        "rotation_world_rad": [0.12, 0.22, 0.31],
        "velocity_world_mps": [2.5, 1.1, 0.0],
        "speed_mps": 3.0,
        "angular_velocity_world_radps": [0.0, 0.0, 0.08],
        "acceleration_world_mps2": [0.0, 0.0, 0.2],
        "steer_normalized": 0.2,
    })
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8")


def _build_bc_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    stack = load_system_stack(_CONFIG)
    entries = {}
    for split in ("train", "validation", "test"):
        created = make_episode(
            data_root / split, complete_provenance=True, expert=True)
        episode = data_root / f"episode_{split}"
        created.rename(episode)
        _vary_ego_dynamics(episode)
        opened = CarlaRecordedEpisode.open(episode, stack)
        entries[split] = [{
            "episode": episode.name,
            "sha256": opened.episode_sha256,
            "tags": {"terrain": "dirt", "weather": "clear"},
        }]
    manifest = data_root / "bc_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "new-orad-bc-manifest-v1",
        "dataset_version": "expert-offroad-fixture-v1",
        "calibration_version": "carla-default-v1",
        "fixture": True,
        "splits": entries,
    }, indent=2), encoding="utf-8")
    perception_checkpoint = tmp_path / "perception.pt"
    torch.save(BEVFusion(stack.bev).state_dict(), perception_checkpoint)
    return manifest, perception_checkpoint


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _train(
        manifest: Path, perception: Path,
        output: Path) -> subprocess.CompletedProcess:
    return _run([
        sys.executable,
        "scripts/train_bc.py",
        "--config", str(_CONFIG),
        "--data-manifest", str(manifest),
        "--perception-checkpoint", str(perception),
        "--run-dir", str(output),
        "--seed", "41",
        "--epochs", "1",
        "--batch-size", "2",
        "--num-workers", "0",
        "--device", "cpu",
        "--smoke-test",
    ])


def _evaluate(
        config: Path, manifest: Path, perception: Path,
        policy: Path, output: Path) -> subprocess.CompletedProcess:
    return _run([
        sys.executable,
        "scripts/evaluate_bc.py",
        "--config", str(config),
        "--data-manifest", str(manifest),
        "--perception-checkpoint", str(perception),
        "--policy-checkpoint", str(policy),
        "--run-dir", str(output),
        "--batch-size", "2",
        "--num-workers", "0",
        "--device", "cpu",
        "--smoke-test",
    ])


def test_fixture_train_then_evaluate_is_explicitly_non_performance(tmp_path):
    manifest, perception = _build_bc_fixture(tmp_path)
    train = _train(manifest, perception, tmp_path / "train")
    assert train.returncode == 0, train.stderr
    last_index = next(
        (tmp_path / "train").glob("*/checkpoints/last.json"))
    checkpoint = resolve_checkpoint_index(last_index)

    evaluate = _evaluate(
        _CONFIG, manifest, perception, checkpoint, tmp_path / "evaluate")

    assert evaluate.returncode == 0, evaluate.stderr
    result_path = next((tmp_path / "evaluate").glob("*/metrics.json"))
    metrics = json.loads(result_path.read_text(encoding="utf-8"))
    assert metrics["model_family"] == "pure_bc_v1"
    assert metrics["model_performance_valid"] is False
    assert metrics["sample_count"] == 2
    assert metrics["nonfinite_trajectory_count"] == 0
    assert (result_path.parent / "_SUCCESS").is_file()
    assert (result_path.parent / "failure_samples.jsonl").is_file()

    modified_config = tmp_path / "modified-system.yaml"
    config_values = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    config_values["vehicle"]["max_speed"] = 19.0
    modified_config.write_text(
        yaml.safe_dump(config_values, sort_keys=False), encoding="utf-8")
    mismatch = _evaluate(
        modified_config, manifest, perception, checkpoint,
        tmp_path / "mismatch")
    assert mismatch.returncode != 0
    assert "config" in mismatch.stderr.lower()
    assert "mismatch" in mismatch.stderr.lower()


def test_evaluate_never_updates_policy_or_normalization():
    perception = BEVFusion(BEVFusionConfig(
        num_cameras=2,
        image_size=(16, 16),
        num_points=16,
        bev_channels=8,
        bev_x_range=(-2.5, 2.5),
        bev_y_range=(-2.5, 2.5),
        bev_resolution=0.5,
        imu_steps=4,
    ))
    policy = BCPolicy(BCPolicyConfig(
        bev_channels=8,
        bev_h=10,
        bev_w=10,
        imu_steps=4,
        encoder_hidden=16,
        ego_hidden=8,
        latent_dim=16,
        horizon=5,
    ))
    batch = BCBatch(
        images=torch.rand(1, 2, 3, 16, 16),
        lidar=torch.rand(1, 16, 4),
        imu=torch.rand(1, 4, 6),
        ego=torch.rand(1, 8),
        expert=torch.rand(1, 5, 4),
        mask=torch.ones(1, 5, dtype=torch.bool),
        sample_ids=("episode:1",),
        episode_ids=("episode",),
        tags=({"terrain": "dirt"},),
    )
    perception_before = {
        name: value.clone() for name, value in perception.state_dict().items()}
    policy_before = {
        name: value.clone() for name, value in policy.state_dict().items()}

    result = evaluate_loader(
        perception, policy, [batch], torch.device("cpu"), max_speed=12.0)

    assert result["sample_count"] == 1
    assert all(torch.equal(value, perception_before[name])
               for name, value in perception.state_dict().items())
    assert all(torch.equal(value, policy_before[name])
               for name, value in policy.state_dict().items())
