from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.verify_carla_initial_dataset import VerificationError, verify_episode


def test_verify_episode_rejects_directory_without_success_marker(tmp_path) -> None:
    episode = tmp_path / "episode_000001"
    episode.mkdir()

    with pytest.raises(VerificationError, match="_SUCCESS"):
        verify_episode(episode)


def test_verify_episode_accepts_consistent_single_frame_episode(tmp_path) -> None:
    episode = tmp_path / "episode_000001"
    episode.mkdir()
    (episode / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    (episode / "calibration.json").write_text("{}", encoding="utf-8")
    (episode / "episode.json").write_text(
        json.dumps({"sample_count": 1}), encoding="utf-8"
    )
    camera_paths = {}
    for name in ("front", "rear", "top"):
        directory = episode / f"camera_{name}"
        directory.mkdir()
        image = directory / "00000010.png"
        image.write_bytes(b"png-smoke-placeholder")
        camera_paths[name] = f"camera_{name}/00000010.png"
    for directory_name in ("lidar_raw", "lidar_256"):
        (episode / directory_name).mkdir()
    np.save(episode / "lidar_raw/00000010.npy", np.ones((20, 4), np.float32))
    np.save(episode / "lidar_256/00000010.npy", np.ones((256, 4), np.float32))

    frame_record = {
        "frame_id": 10,
        "sensors": {
            "camera": {
                name: {"frame_id": 10, "timestamp": 1.0, "path": path}
                for name, path in camera_paths.items()
            },
            "lidar": {
                "frame_id": 10,
                "timestamp": 1.0,
                "raw_path": "lidar_raw/00000010.npy",
                "canonical_path": "lidar_256/00000010.npy",
            },
            "imu": {
                "frame_id": 10,
                "timestamp": 1.0,
                "history": np.zeros((10, 6), np.float32).tolist(),
            },
            "timestamp_skew_seconds": 0.0,
        },
    }
    (episode / "frames.jsonl").write_text(
        json.dumps(frame_record) + "\n", encoding="utf-8"
    )

    summary = verify_episode(episode)

    assert summary == {
        "episode": str(episode),
        "frames": 1,
        "first_frame": 10,
        "last_frame": 10,
        "max_timestamp_skew_seconds": 0.0,
    }
