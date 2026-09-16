"""Validate genuine UE M1 files and print data statistics, not BC performance."""
import argparse
import json
import sys

import numpy as np

from ue_recording import UERecordedEpisode, UERecordingError
from utils.types import SCOUT_DYNAMICS_V1_FIELDS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="committed episode directory with _SUCCESS")
    args = parser.parse_args(argv)
    try:
        episode = UERecordedEpisode.open(args.episode)
        counts, dynamics, imu_last = [], [], []
        image_shapes = None
        for frame in episode.iter_frames():
            counts.append(len(frame.raw_point_cloud))
            dynamics.append(frame.scout_dynamics.to_array())
            imu_last.append(frame.imu[-1])
            image_shapes = [list(image.shape) for image in frame.images]
        states, imu = np.asarray(dynamics), np.asarray(imu_last)
        print(json.dumps({
            "status": "offline_verified", "sample_count": len(episode),
            "image_shapes": image_shapes, "canonical_shape": [256, 4], "imu_shape": [10, 6],
            "point_count_min": min(counts), "point_count_max": max(counts),
            "imu_latest_mean": imu.mean(axis=0).tolist(),
            "imu_latest_min": imu.min(axis=0).tolist(), "imu_latest_max": imu.max(axis=0).tolist(),
            "state_schema": "scout-dynamics-v1", "state_fields": SCOUT_DYNAMICS_V1_FIELDS,
            "state_min": states.min(axis=0).tolist(), "state_max": states.max(axis=0).tolist(),
            "action_source": episode.metadata["action_source"], "expert_labels": False,
        }, indent=2, allow_nan=False))
        return 0
    except UERecordingError as error:
        print(f"Invalid UE episode: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
