"""Synthetic format contracts; these fixtures do not establish UE physics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def episode_path(tmp_path):
    root = tmp_path / "episode_fixture"
    root.mkdir()
    identity = np.eye(4).ravel().tolist()
    lidar = np.eye(4)
    lidar[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    lidar[:3, 3] = [1, 2, 3]
    calibration = {
        "schema_version": "ofad-ue-calibration-v1",
        "calibration_version": "scout-m1-v1", "axes": "x-forward,y-left,z-up",
        "imu_semantics": "specific_force_body_mps2,angular_velocity_body_radps",
        "lidar_frame": "sensor", "camera_order": ["front", "rear", "top"],
        "optical_to_sensor": [0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
        "camera": {name: {"image_size": [192, 192],
                          "intrinsic": [96, 0, 96, 0, 96, 96, 0, 0, 1],
                          "sensor_to_ego": identity}
                   for name in ("front", "rear", "top")},
        "lidar": {"sensor_to_ego": lidar.ravel().tolist()},
        "imu": {"sensor_to_ego": identity},
    }
    write_json(root / "calibration.json", calibration)
    metadata = {
        "schema_version": "ofad-ue-episode-v1", "status": "complete", "sample_count": 2,
        "fixed_delta_seconds": 0.1, "physics_delta_seconds": 1 / 60,
        "vehicle_type": "scout_mini_skid_steer", "expert_labels": False,
        "action_source": "scripted_smoke", "engine_version": "5.0", "map": "Entry",
        "episode_id": root.name, "termination_reason": "capture_limit",
        "calibration_sha1": hashlib.sha1((root / "calibration.json").read_bytes()).hexdigest(),
    }
    write_json(root / "episode.json", metadata)
    (root / "_SUCCESS").touch()
    frames = []
    for index in range(2):
        frame_id = 66 + 6 * index
        stamp = frame_id / 60
        sensor = {"frame_id": frame_id, "timestamp": stamp}
        cameras = {}
        for camera, color in zip(("front", "rear", "top"), (12, 45, 78)):
            relative = f"camera_{camera}/{index:08d}.png"
            (root / relative).parent.mkdir(exist_ok=True)
            Image.new("RGB", (192, 192), (color, 2, 3)).save(root / relative)
            cameras[camera] = dict(sensor, path=relative)
        relative = f"lidar/{index:08d}.bin"
        (root / relative).parent.mkdir(exist_ok=True)
        points = np.zeros((300, 4), dtype="<f4")
        points[:, 0] = np.arange(300)
        points[:, 1:] = [4, 5, 0.75]
        points.tofile(root / relative)
        imu = dict(sensor, history=[[0, 0, 9.8, 0, 0, 0]] * 10,
                   history_frame_ids=list(range(frame_id - 54, frame_id + 1, 6)),
                   history_timestamps=[n / 60 for n in range(frame_id - 54, frame_id + 1, 6)])
        frames.append({
            "sample_index": index, **sensor,
            "sensors": {"camera": cameras, "lidar": dict(sensor, path=relative, point_count=300),
                        "imu": imu},
            "ego_state": {"position_world_m": [0, 0, 0],
                          "orientation_world_xyzw": [0, 0, 0, 1],
                          "velocity_world_mps": [-2, 3, 0], "velocity_body_mps": [-2, 3, 0],
                          "angular_velocity_body_radps": [0, 0, 0.4],
                          "acceleration_body_mps2": [0, 0, 0.2]},
            "action_applied": {"linear_velocity_mps": 1, "angular_velocity_radps": -0.7},
            "action_source": "scripted_smoke", "expert_label": False,
        })
    (root / "frames.jsonl").write_text("\n".join(map(json.dumps, frames)), encoding="utf-8")
    return root


def loader():
    assert importlib.util.find_spec("ue_recording") is not None, "UE consumer is missing"
    from ue_recording import UERecordedEpisode
    return UERecordedEpisode


def edit_frame(root, callback, index=0):
    path = root / "frames.jsonl"
    frames = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    callback(frames[index])
    path.write_text("\n".join(map(json.dumps, frames)), encoding="utf-8")


def test_decodes_real_pixels_raw_points_imu_and_separate_scout_state(episode_path):
    episode = loader().open(episode_path)
    frame = next(episode.iter_frames())
    assert len(episode) == 2
    assert [image.shape for image in frame.images] == [(192, 192, 3)] * 3
    assert all(image.dtype == np.uint8 for image in frame.images)
    assert [int(image[0, 0, 0]) for image in frame.images] == [12, 45, 78]
    assert frame.raw_point_cloud.shape == (300, 4)
    assert frame.canonical_point_cloud.shape == (256, 4)
    assert len(np.unique(frame.canonical_point_cloud[:, 1])) == 256
    np.testing.assert_allclose(frame.canonical_point_cloud[:, [0, 2, 3]],
                               np.tile([-3, 8, 0.75], (256, 1)))
    np.testing.assert_array_equal(frame.canonical_point_cloud,
                                  next(episode.iter_frames()).canonical_point_cloud)
    assert frame.imu.shape == (10, 6)
    np.testing.assert_allclose(frame.imu[:, 2], 9.8)
    assert frame.scout_dynamics.schema_version == "scout-dynamics-v1"
    np.testing.assert_allclose(frame.scout_dynamics.to_array(),
                               [-2, 0.4, 0, 0, -2, 3, 0.2, -0.7])
    assert frame.action_applied["linear_velocity_mps"] == 1


@pytest.mark.parametrize("name", ["_SUCCESS", "episode.json", "calibration.json", "frames.jsonl",
                                  "camera_front/00000000.png", "lidar/00000000.bin"])
def test_rejects_missing_artifact(episode_path, name):
    (episode_path / name).unlink()
    with pytest.raises(ValueError):
        loader().open(episode_path)


@pytest.mark.parametrize("field,value", [("schema_version", "unknown"), ("status", "partial"),
                                         ("vehicle_type", "car"), ("expert_labels", True),
                                         ("action_source", "expert"), ("sample_count", 3),
                                         ("fixed_delta_seconds", 0.2),
                                         ("physics_delta_seconds", 0.1),
                                         ("calibration_sha1", "bad")])
def test_rejects_invalid_metadata(episode_path, field, value):
    path = episode_path / "episode.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    write_json(path, metadata)
    with pytest.raises(ValueError):
        loader().open(episode_path)


@pytest.mark.parametrize("mutation", [
    lambda f: f.update(timestamp=1.2),
    lambda f: f.update(sample_index=4),
    lambda f: f.update(expert_label=True),
    lambda f: f.update(action_source="manual_keyboard"),
    lambda f: f["sensors"]["camera"]["front"].update(timestamp=1.2),
    lambda f: f["sensors"]["lidar"].update(point_count=299),
    lambda f: f["sensors"]["imu"].update(history=[]),
    lambda f: f["sensors"]["imu"]["history"][0].__setitem__(0, float("nan")),
    lambda f: f["sensors"]["imu"]["history_frame_ids"].__setitem__(0, 1),
    lambda f: f["sensors"]["imu"]["history_timestamps"].__setitem__(0, 0.4),
    lambda f: f["ego_state"].update(orientation_world_xyzw=[0, 0, 0, 2]),
    lambda f: f["action_applied"].update(angular_velocity_radps=float("inf")),
])
def test_rejects_invalid_frame(episode_path, mutation):
    edit_frame(episode_path, mutation)
    with pytest.raises(ValueError):
        loader().open(episode_path)


@pytest.mark.parametrize("path", ["../escape.png", "/absolute.png", "C:/escape.png",
                                  "camera_front/../../escape.png", "camera_front\\..\\x.png"])
def test_rejects_path_escape(episode_path, path):
    edit_frame(episode_path, lambda f: f["sensors"]["camera"]["front"].update(path=path))
    with pytest.raises(ValueError):
        loader().open(episode_path)


@pytest.mark.parametrize("corruption", ["short", "nan", "few"])
def test_rejects_bad_raw_points(episode_path, corruption):
    path = episode_path / "lidar/00000000.bin"
    if corruption == "short":
        path.write_bytes(b"abc")
    elif corruption == "nan":
        points = np.fromfile(path, dtype="<f4")
        points[0] = np.nan
        points.tofile(path)
    else:
        np.ones((255, 4), dtype="<f4").tofile(path)
        edit_frame(episode_path, lambda f: f["sensors"]["lidar"].update(point_count=255))
    with pytest.raises(ValueError):
        loader().open(episode_path)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(axes="left-handed"),
    lambda c: c["lidar"]["sensor_to_ego"].__setitem__(0, 2),
    lambda c: c["camera"]["front"]["intrinsic"].__setitem__(0, -1),
    lambda c: c["imu"]["sensor_to_ego"].__setitem__(3, 1),
    lambda c: c["optical_to_sensor"].__setitem__(2, -1),
])
def test_rejects_invalid_calibration_even_with_matching_hash(episode_path, mutation):
    path = episode_path / "calibration.json"
    calibration = json.loads(path.read_text())
    mutation(calibration)
    write_json(path, calibration)
    meta = json.loads((episode_path / "episode.json").read_text())
    meta["calibration_sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
    write_json(episode_path / "episode.json", meta)
    with pytest.raises(ValueError):
        loader().open(episode_path)


def test_cli_outputs_shapes_and_rejects_incomplete(episode_path):
    command = [sys.executable, str(Path(__file__).parents[3] / "scripts/verify_ue_episode.py"),
               str(episode_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["sample_count"] == 2
    assert summary["canonical_shape"] == [256, 4]
    assert summary["state_schema"] == "scout-dynamics-v1"
    (episode_path / "_SUCCESS").unlink()
    assert subprocess.run(command, capture_output=True).returncode != 0


@pytest.mark.parametrize("corruption", [
    "json", "duplicate", "missing_field", "image_size", "image_greyscale", "image_bytes",
    "incomplete_name", "empty_frames", "history_overlap", "cadence",
])
def test_rejects_other_corrupted_artifacts(episode_path, corruption):
    if corruption == "json":
        (episode_path / "episode.json").write_text("{")
    elif corruption == "duplicate":
        (episode_path / "episode.json").write_text('{"status": "complete", "status": "bad"}')
    elif corruption == "missing_field":
        edit_frame(episode_path, lambda f: f.pop("ego_state"))
    elif corruption.startswith("image"):
        path = episode_path / "camera_front/00000000.png"
        if corruption == "image_size":
            Image.new("RGB", (10, 10)).save(path)
        elif corruption == "image_greyscale":
            Image.new("L", (192, 192)).save(path)
        else:
            path.write_bytes(b"invalid PNG")
    elif corruption == "incomplete_name":
        new_path = episode_path.with_suffix(".incomplete")
        episode_path.rename(new_path)
        episode_path = new_path
    elif corruption == "empty_frames":
        (episode_path / "frames.jsonl").write_text("")
    elif corruption == "history_overlap":
        edit_frame(episode_path, lambda f: f["sensors"]["imu"]["history"][0].__setitem__(0, 1), 1)
    elif corruption == "cadence":
        edit_frame(episode_path, lambda f: f.update(frame_id=84, timestamp=1.4), 1)
    with pytest.raises(ValueError):
        loader().open(episode_path)


@pytest.mark.parametrize("artifact", ["_SUCCESS", "camera_front/00000000.png"])
def test_iteration_rechecks_removed_or_corrupted_artifacts(episode_path, artifact):
    episode = loader().open(episode_path)
    if artifact == "_SUCCESS":
        (episode_path / artifact).unlink()
    else:
        (episode_path / artifact).write_bytes(b"broken")
    with pytest.raises(ValueError):
        next(episode.iter_frames())


def test_iteration_all_frames_and_rgba_conversion(episode_path):
    Image.new("RGBA", (192, 192), (10, 20, 30, 40)).save(
        episode_path / "camera_front/00000000.png")
    frames = list(loader().open(episode_path).iter_frames())
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0].images[0][0, 0], [10, 20, 30])
    assert not np.array_equal(frames[0].canonical_point_cloud, frames[1].canonical_point_cloud)


def test_import_does_not_require_torch_or_carla():
    result = subprocess.run([sys.executable, "-c",
                             "import ue_recording, sys; "
                             "assert 'torch' not in sys.modules; "
                             "assert 'carla' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
