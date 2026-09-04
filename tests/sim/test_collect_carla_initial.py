from __future__ import annotations

from queue import Queue
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from configuration.system import load_system_stack
import scripts.collect_carla_initial as collector
from scripts.collect_carla_initial import (
    build_parser,
    build_imu_history,
    camera_intrinsic,
    canonicalize_lidar,
    get_exact_frame,
)


_ROOT = Path(__file__).parents[2]
_CONFIG = _ROOT / "configs" / "system.yaml"


class FrameData:
    def __init__(self, frame: int) -> None:
        self.frame = frame


def test_canonicalize_lidar_is_deterministic_and_flips_y_axis() -> None:
    points = np.array(
        [[float(i), float(i + 1), float(i + 2), 0.5] for i in range(20)],
        dtype=np.float32,
    )

    first, first_meta = canonicalize_lidar(points, count=8, seed=123)
    second, second_meta = canonicalize_lidar(points, count=8, seed=123)

    np.testing.assert_array_equal(first, second)
    assert first_meta == second_meta
    assert first.shape == (8, 4)
    assert first_meta == {
        "raw_point_count": 20,
        "valid_point_count": 8,
        "padding_count": 0,
        "seed": 123,
    }
    assert np.all(first[:, 1] <= 0.0)


def test_canonicalize_lidar_pads_short_cloud() -> None:
    points = np.array([[1.0, 2.0, 3.0, 0.4]], dtype=np.float32)

    canonical, meta = canonicalize_lidar(points, count=4, seed=9)

    np.testing.assert_allclose(canonical[0], [1.0, -2.0, 3.0, 0.4])
    np.testing.assert_array_equal(canonical[1:], np.zeros((3, 4), np.float32))
    assert meta["valid_point_count"] == 1
    assert meta["padding_count"] == 3


def test_build_imu_history_front_pads_and_preserves_old_to_new_order() -> None:
    samples = [
        (10, 1.0, np.arange(6, dtype=np.float32)),
        (11, 1.1, np.arange(6, dtype=np.float32) + 10),
    ]

    history, frames, timestamps, valid_count = build_imu_history(samples, steps=4)

    assert history.shape == (4, 6)
    np.testing.assert_array_equal(history[:2], np.zeros((2, 6), np.float32))
    np.testing.assert_array_equal(history[2], np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(history[3], np.arange(6, dtype=np.float32) + 10)
    assert frames == [-1, -1, 10, 11]
    assert timestamps == [None, None, 1.0, 1.1]
    assert valid_count == 2


def test_get_exact_frame_discards_old_frame() -> None:
    data_queue: Queue = Queue()
    data_queue.put(FrameData(4))
    expected = FrameData(5)
    data_queue.put(expected)

    assert get_exact_frame(data_queue, expected_frame=5, timeout=0.1) is expected


def test_get_exact_frame_rejects_future_frame() -> None:
    data_queue: Queue = Queue()
    data_queue.put(FrameData(6))

    with pytest.raises(RuntimeError, match="expected frame 5, received future frame 6"):
        get_exact_frame(data_queue, expected_frame=5, timeout=0.1)


def test_get_exact_frame_timeout_identifies_sensor() -> None:
    data_queue: Queue = Queue()

    with pytest.raises(
        TimeoutError,
        match=r"camera_front timed out waiting for sensor frame 5",
    ):
        get_exact_frame(
            data_queue,
            expected_frame=5,
            timeout=0.01,
            sensor_name="camera_front",
        )


def test_parser_defaults_to_one_200_frame_episode() -> None:
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 2000
    assert args.frames == 200
    assert args.config == Path("configs/system.yaml")
    assert not hasattr(args, "seed")
    assert not hasattr(args, "fixed_delta")
    assert str(args.output) == "datasets/carla_initial"


def test_camera_intrinsic_uses_horizontal_fov() -> None:
    intrinsic = camera_intrinsic(width=192, height=192, fov_deg=90.0)

    np.testing.assert_allclose(
        intrinsic,
        [[96.0, 0.0, 96.0], [0.0, 96.0, 96.0], [0.0, 0.0, 1.0]],
        rtol=1e-6,
    )


def test_recording_plan_comes_from_canonical_system_stack() -> None:
    plan = collector.recording_plan_from_stack(load_system_stack(_CONFIG))

    assert plan["vehicle_blueprint"] == "vehicle.lincoln.mkz_2020"
    assert plan["fixed_delta_seconds"] == 0.1
    assert list(plan["camera"]) == ["front", "rear", "top"]
    assert plan["camera"]["top"]["fov"] == 100.0
    assert plan["lidar"]["points_per_second"] == 320000
    assert plan["imu"]["history_steps"] == 10


def test_spawn_vehicle_rejects_missing_canonical_blueprint() -> None:
    class MissingLibrary:
        def find(self, blueprint_id):
            raise RuntimeError(f"unknown blueprint {blueprint_id}")

    class FakeWorld:
        def get_blueprint_library(self):
            return MissingLibrary()

    with pytest.raises(RuntimeError, match="vehicle.lincoln.mkz_2020"):
        collector._spawn_vehicle(
            FakeWorld(), np.random.default_rng(42),
            "vehicle.lincoln.mkz_2020")


def test_collector_help_runs_as_a_direct_script() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/collect_carla_initial.py", "--help"],
        cwd=_ROOT, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
