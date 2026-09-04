"""Unit contracts for expert BC dataset manifests and batching."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from configuration.system import load_system_stack
from replay.carla_dataset import CarlaRecordedEpisode, DatasetContractError
from tests.unit.replay.test_carla_dataset import make_episode
from training.bc_dataset import ExpertBCDataset, collate_bc_samples

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs" / "system.yaml"


def _write_manifest(
        path: Path, *, train=(), validation=(), test=(),
        calibration_version="carla-default-v1", fixture=True) -> Path:
    payload = {
        "schema_version": "new-orad-bc-manifest-v1",
        "dataset_version": "expert-offroad-fixture-v1",
        "calibration_version": calibration_version,
        "fixture": fixture,
        "splits": {
            "train": list(train),
            "validation": list(validation),
            "test": list(test),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _entry(manifest_dir: Path, episode: Path, *, tags=None) -> dict:
    stack = load_system_stack(_CONFIG)
    opened = CarlaRecordedEpisode.open(episode, stack)
    return {
        "episode": str(episode.relative_to(manifest_dir)),
        "sha256": opened.episode_sha256,
        "tags": tags or {"terrain": "dirt", "weather": "clear"},
    }


def _mutate_first_record(path: Path, mutate) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(records[0])
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8")


def test_expert_dataset_collates_canonical_b0_batch(tmp_path):
    stack = load_system_stack(_CONFIG)
    episode = make_episode(
        tmp_path, complete_provenance=True, expert=True)
    manifest = _write_manifest(
        tmp_path / "bc_manifest.json",
        train=[_entry(tmp_path, episode)])

    dataset = ExpertBCDataset.open(manifest, "train", stack)
    batch = collate_bc_samples([dataset[0], dataset[1]])

    assert len(dataset) == 2
    assert dataset.fixture is True
    assert dataset.rejections == ()
    assert batch.images.shape == (2, 3, 3, 192, 192)
    assert batch.lidar.shape == (2, 256, 4)
    assert batch.imu.shape == (2, 10, 6)
    assert batch.ego.shape == (2, 8)
    assert batch.expert.shape == (2, 20, 4)
    assert batch.mask.shape == (2, 20)
    assert batch.images.dtype == batch.lidar.dtype == torch.float32
    assert batch.mask.dtype == torch.bool
    assert batch.size == 2
    assert batch.episode_ids == ("episode_test", "episode_test")
    assert batch.tags[0]["terrain"] == "dirt"
    assert 0.0 <= batch.images.min().item() <= batch.images.max().item() <= 1.0
    assert len(dataset.manifest_sha256) == len(dataset.split_sha256) == 64
    assert dataset.calibration_sha256 == CarlaRecordedEpisode.open(
        episode, stack).calibration_sha256


def test_expert_dataset_rejects_non_expert_episode(tmp_path):
    stack = load_system_stack(_CONFIG)
    episode = make_episode(tmp_path, complete_provenance=True)
    manifest = _write_manifest(
        tmp_path / "bc_manifest.json",
        train=[_entry(tmp_path, episode)])

    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(manifest, "train", stack)

    assert exc_info.value.code == "expert_label_required"


def test_manifest_rejects_episode_leakage_across_splits(tmp_path):
    stack = load_system_stack(_CONFIG)
    episode = make_episode(
        tmp_path, complete_provenance=True, expert=True)
    entry = _entry(tmp_path, episode)
    manifest = _write_manifest(
        tmp_path / "bc_manifest.json",
        train=[entry], validation=[entry])

    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(manifest, "train", stack)

    assert exc_info.value.code == "split_episode_overlap"


def test_manifest_rejects_unsafe_path_hash_and_calibration_drift(tmp_path):
    stack = load_system_stack(_CONFIG)
    episode = make_episode(
        tmp_path, complete_provenance=True, expert=True)
    valid = _entry(tmp_path, episode)
    cases = (
        ({**valid, "episode": "../outside"}, "unsafe_episode_path"),
        ({**valid, "sha256": "0" * 64}, "episode_hash_mismatch"),
    )
    for index, (entry, code) in enumerate(cases):
        manifest = _write_manifest(
            tmp_path / f"bc_manifest_{index}.json", train=[entry])
        with pytest.raises(DatasetContractError) as exc_info:
            ExpertBCDataset.open(manifest, "train", stack)
        assert exc_info.value.code == code

    manifest = _write_manifest(
        tmp_path / "bad_calibration.json", train=[valid],
        calibration_version="carla-default-v2")
    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(manifest, "train", stack)
    assert exc_info.value.code == "calibration_version_mismatch"


def test_expert_dataset_records_sensor_health_rejections(tmp_path):
    stack = load_system_stack(_CONFIG)
    episode = make_episode(
        tmp_path, complete_provenance=True, expert=True)
    _mutate_first_record(
        episode / "frames.jsonl",
        lambda record: record["sensors"]["imu"].update({
            "history": [[1000.0, 0.0, 9.81, 0.0, 0.0, 0.0]] * 10,
        }))
    entry = _entry(tmp_path, episode)
    manifest = _write_manifest(
        tmp_path / "bc_manifest.json", train=[entry])

    dataset = ExpertBCDataset.open(manifest, "train", stack)

    assert len(dataset) == 1
    assert dataset.rejections[0]["frame_id"] == 100
    assert "imu_accel_range" in dataset.rejections[0]["reasons"]


def test_manifest_file_hash_matches_file_bytes(tmp_path):
    stack = load_system_stack(_CONFIG)
    episode = make_episode(
        tmp_path, complete_provenance=True, expert=True)
    manifest = _write_manifest(
        tmp_path / "bc_manifest.json",
        train=[_entry(tmp_path, episode)])

    dataset = ExpertBCDataset.open(manifest, "train", stack)

    assert dataset.manifest_sha256 == hashlib.sha256(
        manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update({"schema_version": "old"}),
         "bc_schema_mismatch"),
        (lambda value: value.update({"dataset_version": ""}),
         "dataset_version_missing"),
        (lambda value: value.update({"fixture": "yes"}),
         "fixture_flag_invalid"),
        (lambda value: value.update({"splits": {"train": []}}),
         "split_schema_invalid"),
        (lambda value: value["splits"].update({"train": {}}),
         "split_schema_invalid"),
    ],
)
def test_manifest_header_rejects_invalid_schema_fields(tmp_path, mutation, code):
    stack = load_system_stack(_CONFIG)
    payload = {
        "schema_version": "new-orad-bc-manifest-v1",
        "dataset_version": "expert-fixture-v1",
        "calibration_version": "carla-default-v1",
        "fixture": True,
        "splits": {"train": [], "validation": [], "test": []},
    }
    mutation(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(path, "train", stack)

    assert exc_info.value.code == code


def test_manifest_rejects_invalid_json_root_and_split_name(tmp_path):
    stack = load_system_stack(_CONFIG)
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(invalid_json, "train", stack)
    assert exc_info.value.code == "bc_manifest_invalid"

    list_root = tmp_path / "list.json"
    list_root.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(list_root, "train", stack)
    assert exc_info.value.code == "bc_manifest_invalid"

    valid = _write_manifest(tmp_path / "valid.json")
    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(valid, "development", stack)
    assert exc_info.value.code == "split_name_invalid"


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        ("episode", "split_entry_invalid"),
        ({"episode": "/absolute", "sha256": "1" * 64},
         "unsafe_episode_path"),
        ({"episode": "episode", "sha256": "short"},
         "episode_hash_invalid"),
        ({"episode": "episode", "sha256": "z" * 64},
         "episode_hash_invalid"),
        ({"episode": "episode", "sha256": "1" * 64, "tags": []},
         "episode_tags_invalid"),
    ],
)
def test_manifest_rejects_malformed_split_entries(tmp_path, entry, code):
    stack = load_system_stack(_CONFIG)
    manifest = _write_manifest(tmp_path / "manifest.json", train=[entry])

    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(manifest, "train", stack)

    assert exc_info.value.code == code


def test_dataset_rejects_multiple_calibration_hashes_in_one_split(tmp_path):
    stack = load_system_stack(_CONFIG)
    first = make_episode(
        tmp_path / "first", complete_provenance=True, expert=True)
    second = make_episode(
        tmp_path / "second", complete_provenance=True, expert=True)
    calibration_path = second / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["fixture_variant"] = "second"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    episode_manifest_path = second / "episode.json"
    episode_manifest = json.loads(
        episode_manifest_path.read_text(encoding="utf-8"))
    episode_manifest["calibration_sha256"] = hashlib.sha256(
        calibration_path.read_bytes()).hexdigest()
    episode_manifest_path.write_text(
        json.dumps(episode_manifest), encoding="utf-8")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        train=[_entry(tmp_path, first), _entry(tmp_path, second)])

    with pytest.raises(DatasetContractError) as exc_info:
        ExpertBCDataset.open(manifest, "train", stack)

    assert exc_info.value.code == "calibration_hash_mismatch"


def test_collate_rejects_empty_sample_list():
    with pytest.raises(ValueError, match="empty"):
        collate_bc_samples([])
