"""Strict Forest V2 fixtures; real UE evidence is verified separately."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(params=("solidstate-placeholder-v1", "mid360-approx-v1"))
def v2_episode(tmp_path, request):
    profile = request.param
    root = tmp_path / f"episode_{profile}"
    (root / "camera_front").mkdir(parents=True)
    (root / "lidar").mkdir()
    identity = np.eye(4).ravel().tolist()
    lidar_profile = {
        "sensor_to_ego": identity, "sensor_profile": profile,
        "scan_model": "instantaneous_raycast", "scan_hz": 10,
        "minimum_range_m": 0.1, "maximum_range_m": 50,
        "intensity_model": "incidence_cosine",
    }
    if profile == "solidstate-placeholder-v1":
        lidar_profile.update(
            nominal_rays_per_scan=27648, nominal_rays_per_second=276480,
            scan_type="solidstate_uniform_grid", horizontal_samples=192,
            vertical_samples=144, horizontal_fov_degrees=120,
            vertical_fov_degrees=90)
    else:
        lidar_profile.update(
            nominal_rays_per_scan=20000, nominal_rays_per_second=200000,
            scan_type="mechanical_uniform", channels=40,
            points_per_channel=500, horizontal_fov_degrees=360,
            vertical_min_degrees=-7, vertical_max_degrees=52)
    calibration = {
        "schema_version": "ofad-ue-calibration-v2",
        "calibration_version": "scout-forest-rig-v1",
        "axes": "x-forward,y-left,z-up",
        "imu_semantics": (
            "specific_force_body_mps2,angular_velocity_body_radps"),
        "lidar_frame": "sensor",
        "optical_to_sensor": [
            0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
        "camera_order": ["camera_0"],
        "camera": {"camera_0": {
            "image_size": [180, 320],
            "intrinsic": [160, 0, 159.5, 0, 160, 89.5, 0, 0, 1],
            "sensor_to_ego": identity, "native_capture_hz": 30}},
        "lidar": lidar_profile,
        "imu": {"sensor_to_ego": identity,
                "source": "derived_from_root_motion",
                "measurement_model":
                    "root_point_velocity_difference_0.1s_no_noise"},
    }
    _write_json(root / "calibration.json", calibration)
    actor = {"name": "actor", "class": "/Script/Test.Actor",
             "relative_pose": identity}
    metadata = {
        "schema_version": "ofad-ue-episode-v2", "status": "complete",
        "episode_id": root.name, "map": "forest111",
        "scene_version": "forest-smoke-v1",
        "rig_version": "scout-forest-rig-v1", "sensor_profile": profile,
        "vehicle_type": "scout_mini_skid_steer", "expert_labels": False,
        "source_commit": "84344f7", "source_worktree_dirty": True,
        "imu_source": "derived_from_root_motion", "action_source": "scripted",
        "task": "integration_smoke", "goal_valid": False,
        "termination_reason": "capture_limit", "engine_version": "5.0.3",
        "sample_count": 2,
        "calibration_sha1": hashlib.sha1(
            (root / "calibration.json").read_bytes()).hexdigest(),
        "timing": {"physics_hz": 60, "sample_hz": 10, "frame_count": 2},
        "actor_inventory": {"scout": actor, "camera": actor, "lidar": actor},
    }
    _write_json(root / "episode.json", metadata)
    (root / "_SUCCESS").touch()
    frames = []
    for index in range(2):
        frame_id = 126 + index * 6
        stamp = frame_id / 60
        packet = {"frame_id": frame_id, "timestamp": stamp}
        image_path = f"camera_front/{index:08d}.png"
        Image.new(
            "RGB", (320, 180), (20 + index, 30, 40)
        ).save(root / image_path)
        lidar_path = f"lidar/{index:08d}.bin"
        points = np.zeros((300, 4), dtype="<f4")
        points[:, 0] = np.arange(300)
        points[:, 3] = 0.5
        points.tofile(root / lidar_path)
        history_ids = list(range(frame_id - 54, frame_id + 1, 6))
        frames.append({
            "sample_index": index, **packet,
            "sensors": {
                "camera_0": {**packet, "path": image_path},
                "lidar_0": {**packet, "path": lidar_path, "point_count": 300},
                "imu": {**packet, "history": [[0, 0, 9.8, 0, 0, 0]] * 10,
                        "history_frame_ids": history_ids,
                        "history_timestamps": [
                            value / 60 for value in history_ids]}},
            "ego_state": {
                "position_world_m": [0, 0, 0],
                "orientation_world_xyzw": [0, 0, 0, 1],
                "velocity_world_mps": [1, 0, 0],
                "velocity_body_mps": [1, 0, 0],
                "angular_velocity_body_radps": [0, 0, 0.2],
                "acceleration_body_mps2": [0, 0, 0.1],
                "command_linear_velocity_mps": 0.4,
                "command_angular_velocity_radps": 0.0},
            "action_applied": {"linear_velocity_mps": 0.4,
                               "angular_velocity_radps": 0.0},
            "action_source": "scripted", "expert_label": False,
            "task": "integration_smoke", "goal_valid": False,
        })
    (root / "frames.jsonl").write_text(
        "\n".join(json.dumps(frame) for frame in frames), encoding="utf-8")
    return root


def _loader():
    from ue_recording import UERecordedEpisodeV2
    return UERecordedEpisodeV2


def _edit_json(path, callback):
    value = json.loads(path.read_text(encoding="utf-8"))
    callback(value)
    _write_json(path, value)


def _edit_frame(root, callback, index=0):
    path = root / "frames.jsonl"
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    callback(frames[index])
    path.write_text(
        "\n".join(json.dumps(frame) for frame in frames),
        encoding="utf-8")


def test_v2_decodes_single_camera_lidar_imu_and_profile(v2_episode):
    episode = _loader().open(v2_episode)
    frame = next(episode.iter_frames())
    assert len(episode) == 2
    assert frame.images[0].shape == (180, 320, 3)
    assert len(frame.images) == 1
    assert frame.raw_point_cloud.shape == (300, 4)
    assert frame.canonical_point_cloud.shape == (256, 4)
    assert frame.imu.shape == (10, 6)
    assert episode.metadata["expert_labels"] is False
    assert episode.metadata["sensor_profile"] in {
        "solidstate-placeholder-v1", "mid360-approx-v1"}


@pytest.mark.parametrize("field,value", [
    ("schema_version", "ofad-ue-episode-v1"), ("map", "Entry"),
    ("scene_version", "other"), ("rig_version", "other"),
    ("expert_labels", True), ("action_source", "expert"),
    ("task", "goal_navigation"), ("goal_valid", True),
    ("imu_source", "unknown"), ("source_commit", "dirty"),
])
def test_v2_rejects_invalid_manifest(v2_episode, field, value):
    _edit_json(
        v2_episode / "episode.json",
        lambda data: data.update({field: value}))
    with pytest.raises(ValueError):
        _loader().open(v2_episode)


@pytest.mark.parametrize("mutation", [
    lambda frame: frame["sensors"]["camera_0"].update(timestamp=0),
    lambda frame: frame["sensors"]["lidar_0"].update(point_count=299),
    lambda frame: frame["sensors"]["imu"].update(history=[]),
    lambda frame: frame.update(expert_label=True),
    lambda frame: frame.update(goal_valid=True),
    lambda frame: frame["ego_state"].update(
        orientation_world_xyzw=[0, 0, 0, 2]),
])
def test_v2_rejects_unsynchronised_or_invalid_frame(v2_episode, mutation):
    _edit_frame(v2_episode, mutation)
    with pytest.raises(ValueError):
        _loader().open(v2_episode)


def test_v2_rejects_profile_geometry_mismatch(v2_episode):
    path = v2_episode / "calibration.json"
    _edit_json(
        path,
        lambda data: data["lidar"].update(nominal_rays_per_second=1))
    _edit_json(v2_episode / "episode.json", lambda data: data.update(
        calibration_sha1=hashlib.sha1(path.read_bytes()).hexdigest()))
    with pytest.raises(ValueError):
        _loader().open(v2_episode)


def test_v2_cli_outputs_profile_and_shapes(v2_episode):
    script = Path(__file__).parents[3] / "scripts" / "verify_ue_episode_v2.py"
    result = subprocess.run(
        [sys.executable, str(script), str(v2_episode)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["sample_count"] == 2
    assert summary["image_shape"] == [180, 320, 3]
    assert summary["sensor_profile"] == v2_episode.name.removeprefix(
        "episode_")
