from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from configuration.system import load_system_stack
from sim.carla_baseline import canonical_yaml_sha256

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs" / "system.yaml"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _calibration() -> dict:
    cameras = {}
    for name, fov, location, rotation in (
        ("front", 90.0, [1.5, 0.0, 1.6], [0.0, 0.0, 0.0]),
        ("rear", 90.0, [-1.5, 0.0, 1.6], [0.0, 0.0, np.pi]),
        ("top", 100.0, [0.0, 0.0, 1.9], [0.0, -np.pi / 12.0, 0.0]),
    ):
        cameras[name] = {
            "intrinsic": np.eye(3).tolist(),
            "sensor_to_ego_carla": {
                "location_m": location,
                "rotation_rad": rotation,
                "matrix": np.eye(4).tolist(),
            },
            "image_size": [192, 192],
            "fov_deg": fov,
        }
    return {
        "schema_version": "new-orad-carla-v1",
        "calibration_version": "carla-default-v1",
        "carla_sensor_axes": "x-forward,y-right,z-up",
        "new_orad_ego_axes": "x-forward,y-left,z-up",
        "camera": cameras,
        "lidar": {
            "sensor_to_ego_carla": {"matrix": np.eye(4).tolist()},
            "canonical_y_flip": True,
            "canonical_points": 256,
        },
        "imu": {
            "sensor_to_ego_carla": {"matrix": np.eye(4).tolist()},
            "history_shape": [10, 6],
        },
    }


def make_episode(
        tmp_path: Path, *, complete_provenance: bool = False,
        expert: bool = False) -> Path:
    episode = tmp_path / "episode_test"
    episode.mkdir(parents=True)
    for directory in (
            "camera_front", "camera_rear", "camera_top",
            "lidar_raw", "lidar_256"):
        (episode / directory).mkdir()

    calibration_path = episode / "calibration.json"
    _write_json(calibration_path, _calibration())
    calibration_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    records = []
    for sample_index, frame in enumerate((100, 101)):
        stem = f"{frame:08d}"
        camera = {}
        for camera_index, name in enumerate(("front", "rear", "top")):
            relative = f"camera_{name}/{stem}.png"
            pixels = np.full(
                (192, 192, 3), 20 + sample_index + camera_index,
                dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(episode / relative)
            camera[name] = {
                "frame_id": frame, "timestamp": frame / 10.0,
                "path": relative,
            }
        raw_path = f"lidar_raw/{stem}.npy"
        canonical_path = f"lidar_256/{stem}.npy"
        raw = np.array(
            [[1.0 + index, 0.2, 0.0, 0.5] for index in range(32)],
            dtype=np.float32)
        canonical = np.zeros((256, 4), dtype=np.float32)
        canonical[:32] = raw
        np.save(episode / raw_path, raw)
        np.save(episode / canonical_path, canonical)
        record = {
            "sample_index": sample_index,
            "frame_id": frame,
            "timestamp": frame / 10.0,
            "sensors": {
                "camera": camera,
                "lidar": {
                    "frame_id": frame, "timestamp": frame / 10.0,
                    "raw_path": raw_path, "canonical_path": canonical_path,
                    "raw_point_count": 32, "valid_point_count": 32,
                    "padding_count": 224, "seed": frame,
                },
                "imu": {
                    "frame_id": frame, "timestamp": frame / 10.0,
                    "history": np.tile(
                        [0.0, 0.0, 9.81, 0.0, 0.0, 0.0], (10, 1)).tolist(),
                    "history_frame_ids": list(range(frame - 9, frame + 1)),
                    "history_timestamps": [
                        value / 10.0 for value in range(frame - 9, frame + 1)],
                    "valid_count": 10,
                },
                "timestamp_skew_seconds": 0.0,
            },
            "ego_state": {
                "position_world_m": [1.0, 2.0, 0.0],
                "rotation_world_rad": [0.1, 0.2, 0.3],
                "velocity_world_mps": [2.0, 1.0, 0.0],
                "speed_mps": np.sqrt(5.0),
                "angular_velocity_world_radps": [0.0, 0.0, 0.05],
                "acceleration_world_mps2": [0.0, 0.0, 0.1],
                "steer_normalized": 0.1,
            },
            "action_source": (
                "human-teleop-v1" if expert else "traffic_manager_smoke"),
            "expert_label": expert,
            "event": {"collision": False},
        }
        if expert:
            record.update({
                "expert_source": "human-teleop-v1",
                "expert_trajectory": [
                    [0.2 * (index + 1), 0.0, 0.0, 2.0]
                    for index in range(20)
                ],
                "trajectory_mask": [True] * 20,
            })
        records.append(record)
    with (episode / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    manifest = {
        "schema_version": "new-orad-carla-v1",
        "status": "complete",
        "action_source": (
            "human-teleop-v1" if expert else "traffic_manager_smoke"),
        "expert_labels": expert,
        "carla_client_version": "0.9.16",
        "carla_server_version": "0.9.16",
        "map": "Carla/Maps/Town10HD_Opt",
        "fixed_delta_seconds": 0.1,
        "random_seed": 42,
        "start_frame": 100,
        "end_frame": 101,
        "sample_count": 2,
    }
    if complete_provenance:
        manifest.update({
            "vehicle_blueprint": "vehicle.lincoln.mkz_2020",
            "config_sha256": hashlib.sha256(_CONFIG.read_bytes()).hexdigest(),
            "config_canonical_sha256": canonical_yaml_sha256(_CONFIG),
            "calibration_sha256": calibration_hash,
            "calibration_version": "carla-default-v1",
        })
    _write_json(episode / "episode.json", manifest)
    (episode / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return episode


def _dataset_module():
    return importlib.import_module("replay.carla_dataset")


def test_legacy_episode_requires_opt_in_and_decodes_read_only_frames(tmp_path):
    module = _dataset_module()
    stack = load_system_stack(_CONFIG)
    episode_path = make_episode(tmp_path)

    with pytest.raises(module.DatasetContractError) as exc_info:
        module.CarlaRecordedEpisode.open(episode_path, stack)
    assert exc_info.value.code == "legacy_provenance_requires_opt_in"

    episode = module.CarlaRecordedEpisode.open(
        episode_path, stack, allow_legacy_provenance=True)
    frame = next(episode.iter_frames(max_frames=1))

    assert episode.provenance_gaps == (
        "vehicle_blueprint", "config_sha256", "calibration_sha256")
    assert len(episode.calibration_sha256) == 64
    assert len(episode.episode_sha256) == 64
    assert frame.images[0].shape == (192, 192, 3)
    assert frame.images[0].dtype == np.uint8
    assert frame.raw_point_cloud.shape == (32, 4)
    assert frame.canonical_point_cloud.shape == (256, 4)
    assert frame.imu_history.shape == (10, 6)
    assert frame.ego_state.yaw == pytest.approx(0.3)
    assert frame.ego_state.vx == pytest.approx(2.206193, rel=1e-5)
    assert frame.ego_state.vy == pytest.approx(0.364296, rel=1e-5)
    assert frame.expert_source == "unknown"
    assert frame.expert_trajectory is None
    assert frame.trajectory_mask is None


def test_recorded_episode_decodes_versioned_expert_trajectory(tmp_path):
    module = _dataset_module()
    path = make_episode(
        tmp_path, complete_provenance=True, expert=True)

    frame = next(module.CarlaRecordedEpisode.open(
        path, load_system_stack(_CONFIG)).iter_frames())

    assert frame.expert_label is True
    assert frame.expert_source == "human-teleop-v1"
    assert frame.expert_trajectory.shape == (20, 4)
    assert frame.expert_trajectory.dtype == np.float32
    assert frame.trajectory_mask.shape == (20,)
    assert frame.trajectory_mask.dtype == np.bool_


def test_recorded_episode_rejects_malformed_expert_trajectory(tmp_path):
    module = _dataset_module()
    path = make_episode(
        tmp_path, complete_provenance=True, expert=True)
    _mutate_first_record(
        path / "frames.jsonl",
        lambda record: record.update({"expert_trajectory": [[0.0] * 4]}))
    episode = module.CarlaRecordedEpisode.open(path, load_system_stack(_CONFIG))

    with pytest.raises(module.DatasetContractError) as exc_info:
        next(episode.iter_frames())

    assert exc_info.value.code == "expert_trajectory_shape_mismatch"


@pytest.mark.parametrize(("mutation", "code"), [
    (lambda path: (path / "_SUCCESS").unlink(), "episode_incomplete"),
    (lambda path: _mutate_json(
        path / "episode.json", "schema_version", "unknown"),
     "schema_mismatch"),
    (lambda path: _mutate_json(
        path / "episode.json", "sample_count", 3),
     "sample_count_mismatch"),
    (lambda path: _mutate_json(
        path / "episode.json", "carla_server_version", "0.9.15"),
     "carla_version_mismatch"),
])
def test_loader_rejects_episode_contract_drift(tmp_path, mutation, code):
    module = _dataset_module()
    path = make_episode(tmp_path, complete_provenance=True)
    mutation(path)

    with pytest.raises(module.DatasetContractError) as exc_info:
        module.CarlaRecordedEpisode.open(path, load_system_stack(_CONFIG))
    assert exc_info.value.code == code


def _mutate_json(path: Path, key: str, value) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    _write_json(path, payload)


def _mutate_first_record(path: Path, mutate) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(records[0])
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8")


def test_loader_rejects_unsafe_path_and_missing_sensor_file(tmp_path):
    module = _dataset_module()
    stack = load_system_stack(_CONFIG)
    path = make_episode(tmp_path, complete_provenance=True)
    _mutate_first_record(
        path / "frames.jsonl",
        lambda record: record["sensors"]["camera"]["front"].update(
            {"path": "../outside.png"}))
    with pytest.raises(module.DatasetContractError) as exc_info:
        module.CarlaRecordedEpisode.open(path, stack)
    assert exc_info.value.code == "unsafe_sensor_path"

    path = make_episode(tmp_path / "missing", complete_provenance=True)
    (path / "camera_front" / "00000100.png").unlink()
    with pytest.raises(module.DatasetContractError) as exc_info:
        module.CarlaRecordedEpisode.open(path, stack)
    assert exc_info.value.code == "sensor_file_missing"


def test_loader_rejects_frame_order_calibration_and_nonfinite_arrays(tmp_path):
    module = _dataset_module()
    stack = load_system_stack(_CONFIG)
    path = make_episode(tmp_path / "order", complete_provenance=True)
    _mutate_first_record(
        path / "frames.jsonl", lambda record: record.update({"frame_id": 101}))
    with pytest.raises(module.DatasetContractError) as exc_info:
        module.CarlaRecordedEpisode.open(path, stack)
    assert exc_info.value.code == "frame_order_invalid"

    path = make_episode(tmp_path / "calibration", complete_provenance=True)
    calibration = json.loads(
        (path / "calibration.json").read_text(encoding="utf-8"))
    calibration["new_orad_ego_axes"] = "x-forward,y-right,z-up"
    _write_json(path / "calibration.json", calibration)
    with pytest.raises(module.DatasetContractError) as exc_info:
        module.CarlaRecordedEpisode.open(path, stack)
    assert exc_info.value.code in {
        "calibration_hash_mismatch", "calibration_axes_mismatch"}

    path = make_episode(tmp_path / "finite", complete_provenance=True)
    raw_path = path / "lidar_raw" / "00000100.npy"
    raw = np.load(raw_path, allow_pickle=False)
    raw[0, 0] = np.nan
    np.save(raw_path, raw)
    episode = module.CarlaRecordedEpisode.open(path, stack)
    with pytest.raises(module.DatasetContractError) as exc_info:
        next(episode.iter_frames(max_frames=1))
    assert exc_info.value.code == "lidar_nonfinite"


def test_loader_rejects_camera_transform_drift_even_with_updated_file_hash(
        tmp_path):
    module = _dataset_module()
    stack = load_system_stack(_CONFIG)
    path = make_episode(tmp_path, complete_provenance=True)
    calibration_path = path / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["camera"]["front"]["sensor_to_ego_carla"]["location_m"][0] = 9.0
    _write_json(calibration_path, calibration)
    _mutate_json(
        path / "episode.json", "calibration_sha256",
        hashlib.sha256(calibration_path.read_bytes()).hexdigest())

    with pytest.raises(module.DatasetContractError) as exc_info:
        module.CarlaRecordedEpisode.open(path, stack)
    assert exc_info.value.code == "camera_transform_mismatch"


def test_loader_uses_canonical_config_hash_for_semantic_identity(tmp_path):
    module = _dataset_module()
    stack = load_system_stack(_CONFIG)
    path = make_episode(tmp_path, complete_provenance=True)
    _mutate_json(path / "episode.json", "config_sha256", "0" * 64)

    episode = module.CarlaRecordedEpisode.open(path, stack)

    assert episode.manifest["config_canonical_sha256"] == (
        stack.config_canonical_sha256)
