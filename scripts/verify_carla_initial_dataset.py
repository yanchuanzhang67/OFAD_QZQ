#!/usr/bin/env python3
"""Verify CARLA smoke episodes before New_ORAD consumes them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class VerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify_episode(episode: Path) -> dict:
    episode = Path(episode)
    _require((episode / "_SUCCESS").is_file(), "episode has no _SUCCESS marker")
    for name in ("episode.json", "calibration.json", "frames.jsonl"):
        _require((episode / name).is_file(), f"episode is missing {name}")

    manifest = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (episode / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(bool(records), "frames.jsonl is empty")
    _require(
        int(manifest["sample_count"]) == len(records),
        "episode sample_count does not match frames.jsonl",
    )

    frame_ids = []
    max_skew = 0.0
    for record in records:
        frame = int(record["frame_id"])
        frame_ids.append(frame)
        sensors = record["sensors"]
        cameras = sensors["camera"]
        _require(
            set(cameras) == {"front", "rear", "top"},
            f"frame {frame} does not contain exactly front/rear/top cameras",
        )
        for name, camera in cameras.items():
            _require(
                int(camera["frame_id"]) == frame,
                f"camera {name} frame mismatch at {frame}",
            )
            _require(
                (episode / camera["path"]).is_file(),
                f"camera {name} file is missing at frame {frame}",
            )

        lidar = sensors["lidar"]
        imu = sensors["imu"]
        _require(int(lidar["frame_id"]) == frame, f"LiDAR frame mismatch at {frame}")
        _require(int(imu["frame_id"]) == frame, f"IMU frame mismatch at {frame}")
        raw_path = episode / lidar["raw_path"]
        canonical_path = episode / lidar["canonical_path"]
        _require(raw_path.is_file(), f"raw LiDAR file is missing at frame {frame}")
        _require(
            canonical_path.is_file(),
            f"canonical LiDAR file is missing at frame {frame}",
        )
        raw = np.load(raw_path, mmap_mode="r")
        canonical = np.load(canonical_path, mmap_mode="r")
        _require(
            raw.ndim == 2 and raw.shape[1] == 4,
            f"raw LiDAR must be (N,4) at frame {frame}",
        )
        _require(
            canonical.shape == (256, 4),
            f"canonical LiDAR must be (256,4) at frame {frame}",
        )
        imu_history = np.asarray(imu["history"], dtype=np.float32)
        _require(
            imu_history.shape == (10, 6),
            f"IMU history must be (10,6) at frame {frame}",
        )
        skew = float(sensors["timestamp_skew_seconds"])
        _require(skew <= 0.05 + 1e-9, f"timestamp skew exceeds 0.05 s at {frame}")
        max_skew = max(max_skew, skew)

    _require(
        frame_ids == sorted(set(frame_ids)),
        "frame IDs must be unique and strictly increasing",
    )
    return {
        "episode": str(episode),
        "frames": len(records),
        "first_frame": frame_ids[0],
        "last_frame": frame_ids[-1],
        "max_timestamp_skew_seconds": max_skew,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify CARLA initial episodes.")
    parser.add_argument("root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    episodes = sorted((root / "episodes").glob("episode_*"))
    if not episodes:
        print(f"验证失败：{root / 'episodes'} 中没有 episode")
        return 1
    try:
        for episode in episodes:
            summary = verify_episode(episode)
            print(
                f"通过：{episode.name} | frames={summary['frames']} | "
                f"range={summary['first_frame']}..{summary['last_frame']} | "
                f"max_skew={summary['max_timestamp_skew_seconds']:.6f}s"
            )
    except (VerificationError, KeyError, ValueError, OSError) as error:
        print(f"验证失败：{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
