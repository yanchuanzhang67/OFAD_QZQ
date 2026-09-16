"""Validate a committed UE Forest V2 episode and print data statistics."""
import argparse
import json
import sys

import numpy as np

from ue_recording import UERecordedEpisodeV2, UERecordingError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        help="committed V2 episode directory with _SUCCESS")
    args = parser.parse_args(argv)
    try:
        episode = UERecordedEpisodeV2.open(args.episode)
        counts = []
        imu_latest = []
        frame_ids = []
        image_shape = None
        for frame in episode.iter_frames():
            counts.append(len(frame.raw_point_cloud))
            imu_latest.append(frame.imu[-1])
            frame_ids.append(frame.frame_id)
            image_shape = list(frame.images[0].shape)
        imu = np.asarray(imu_latest)
        print(json.dumps({
            "status": "offline_verified",
            "schema_version": episode.metadata["schema_version"],
            "sample_count": len(episode),
            "sensor_profile": episode.metadata["sensor_profile"],
            "image_shape": image_shape,
            "canonical_shape": [256, 4],
            "imu_shape": [10, 6],
            "frame_id_first": min(frame_ids),
            "frame_id_last": max(frame_ids),
            "point_count_min": min(counts),
            "point_count_max": max(counts),
            "point_count_mean": float(np.mean(counts)),
            "imu_latest_mean": imu.mean(axis=0).tolist(),
            "action_source": episode.metadata["action_source"],
            "task": episode.metadata["task"],
            "goal_valid": episode.metadata["goal_valid"],
            "expert_labels": episode.metadata["expert_labels"],
            "source_commit": episode.metadata["source_commit"],
            "source_worktree_dirty": episode.metadata["source_worktree_dirty"],
        }, indent=2, allow_nan=False))
        return 0
    except UERecordingError as error:
        print(f"Invalid UE Forest V2 episode: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
