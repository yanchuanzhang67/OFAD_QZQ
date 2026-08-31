import numpy as np
import pytest

from utils.frames import (
    CoordinateFrame, ego_to_world_points, transform_trajectory,
    world_to_ego_points)
from utils.schema import Observation
from utils.types import Trajectory, VehicleState, Waypoint

pytestmark = pytest.mark.unit


def test_ego_world_ego_points_round_trip():
    state = VehicleState(x=12.0, y=-3.0, yaw=0.73)
    ego = np.array([[0.0, 0.0], [3.0, -1.0], [-2.0, 4.0]], np.float32)
    world = ego_to_world_points(ego, state)
    restored = world_to_ego_points(world, state)
    assert world.dtype == np.float32
    assert np.allclose(restored, ego, atol=1e-6)


def test_trajectory_frame_round_trip_preserves_physical_fields():
    state = VehicleState(x=10.0, y=5.0, yaw=np.pi / 2)
    ego = Trajectory([
        Waypoint(x=2.0, y=0.5, yaw=0.2, speed=3.0, steering=0.1, t=0.2)
    ], frame=CoordinateFrame.EGO.value, timestamp=4.5)
    world = transform_trajectory(ego, state, CoordinateFrame.WORLD)
    restored = transform_trajectory(world, state, CoordinateFrame.EGO)
    assert world.frame == "world"
    assert restored.frame == "ego"
    assert restored.timestamp == pytest.approx(4.5)
    assert np.allclose(restored.to_array(), ego.to_array(), atol=1e-6)


def test_observation_enforces_imu6_and_sensor_frames():
    with pytest.raises(ValueError, match="imu_history"):
        Observation(
            timestamp=1.0, simulator_frame=10,
            images=[np.zeros((4, 4, 3), np.uint8)],
            point_cloud=np.zeros((3, 4), np.float32),
            imu_history=np.zeros((5, 3), np.float32),
            ego_state=VehicleState(), camera_frames=(10,), lidar_frame=10)


def test_observation_reports_zero_camera_lidar_skew():
    obs = Observation(
        timestamp=1.0, simulator_frame=10,
        images=[np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8)],
        point_cloud=np.zeros((3, 4), np.float32),
        imu_history=np.zeros((5, 6), np.float32),
        ego_state=VehicleState(), camera_frames=(10, 10), lidar_frame=10,
        imu_frames=(6, 7, 8, 9, 10),
        sensor_timestamps={"camera_0": 1.000, "camera_1": 1.002,
                           "lidar": 1.001})
    assert obs.frame == CoordinateFrame.EGO.value
    assert obs.max_sensor_skew_frames == 0
    assert obs.max_sensor_skew_seconds == pytest.approx(0.002)


def test_observation_rejects_mixed_camera_lidar_frames():
    with pytest.raises(ValueError, match="synchronized"):
        Observation(
            timestamp=1.0, simulator_frame=10,
            images=[np.zeros((4, 4, 3), np.uint8)],
            point_cloud=np.zeros((3, 4), np.float32),
            imu_history=np.zeros((5, 6), np.float32),
            ego_state=VehicleState(), camera_frames=(9,), lidar_frame=10)


def test_frame_helpers_reject_bad_points_and_support_noop_empty():
    state = VehicleState()
    with pytest.raises(ValueError, match="N,2"):
        ego_to_world_points(np.zeros((2, 3), np.float32), state)
    with pytest.raises(ValueError, match="finite"):
        world_to_ego_points(np.array([[np.nan, 0.0]], np.float32), state)
    empty = Trajectory([], frame="ego", timestamp=1.0)
    same = transform_trajectory(empty, state, CoordinateFrame.EGO)
    world = transform_trajectory(empty, state, CoordinateFrame.WORLD)
    assert same.length == world.length == 0
    assert same.frame == "ego" and world.frame == "world"


def test_observation_rejects_bad_arrays_timestamps_and_frame_metadata():
    base = dict(
        timestamp=1.0, simulator_frame=10,
        images=[np.zeros((4, 4, 3), np.uint8)],
        point_cloud=np.zeros((2, 4), np.float32),
        imu_history=np.zeros((2, 6), np.float32), ego_state=VehicleState(),
        camera_frames=(10,), lidar_frame=10)
    with pytest.raises(ValueError, match="frame must"):
        Observation(**base, frame="world")
    with pytest.raises(ValueError, match="timestamp"):
        Observation(**{**base, "timestamp": np.nan})
    with pytest.raises(ValueError, match="point_cloud"):
        Observation(**{**base, "point_cloud": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="at least one"):
        Observation(**{**base, "point_cloud": np.zeros((0, 4))})
    with pytest.raises(ValueError, match="aligned"):
        Observation(**{**base, "images": []})
    with pytest.raises(ValueError, match="imu_frames"):
        Observation(**base, imu_frames=(1, 2, 3))
    with pytest.raises(ValueError, match="sensor timestamps"):
        Observation(**base, sensor_timestamps={"lidar": np.inf})
    with pytest.raises(ValueError, match="images"):
        Observation(**{
            **base,
            "images": [np.full((4, 4, 3), np.nan, dtype=np.float32)],
        })


def test_observation_without_sensor_timestamps_has_zero_time_skew():
    obs = Observation(
        timestamp=1.0, simulator_frame=10,
        images=[np.zeros((4, 4, 3), np.uint8)],
        point_cloud=np.zeros((2, 4), np.float32),
        imu_history=np.zeros((2, 6), np.float32),
        ego_state=VehicleState(), camera_frames=(10,), lidar_frame=10)
    assert obs.max_sensor_skew_seconds == 0.0
