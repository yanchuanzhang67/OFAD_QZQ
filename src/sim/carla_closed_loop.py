"""CARLA closed-loop evaluation: sensor -> BEV fusion -> policy -> safety -> control.

Two layers:

* **numpy-only helpers** (unit-testable without CARLA / torch): trajectory
  ego->world frame conversion, Ackermann->CARLA ``VehicleControl`` mapping,
  LiDAR canonicalisation, occupancy-grid derivation, IMU attitude estimation,
  and :class:`EpisodeMetrics` aggregation (pass / collision / attitude
  stability).
* **CARLA glue** (:class:`CarlaSensorStack`, :class:`CarlaClosedLoopRunner`)
  whose ``import carla`` is *lazy* -- the module imports cleanly when CARLA is
  absent so the helpers stay exercisable in CI. Run the loop through
  ``scripts/evaluate_carla_closed_loop.py``.

Convention: the policy outputs ``(B, N, 4)`` ego-relative waypoints
``[x_forward, y_left, heading_rel, velocity]``; :func:`ego_trajectory_to_world`
rotates them into the world frame expected by the safety filter and the
pure-pursuit controller (override with ``policy_frame='world'``).
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import time
from typing import Callable, Optional, Tuple

import numpy as np

from safety.supervisor import SafetyMode, SafetySupervisor, SafetySupervisorConfig
from sim.carla_baseline import CarlaBaselineConfig, carla_sensor_attributes
from utils.schema import Observation
from utils.sensor_health import (
    SensorHealthGate,
    SensorHealthReason,
    SensorHealthReport,
)
from utils.types import OccupancyGrid, Trajectory, VehicleState, Waypoint

try:  # lazy: CARLA PythonAPI is optional
    import carla  # noqa: F401
    _HAS_CARLA = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_CARLA = False

__all__ = [
    "ClosedLoopConfig",
    "EpisodeMetrics",
    "CarlaSensorStack",
    "CarlaClosedLoopRunner",
    "canonicalize_points",
    "imu_samples_to_array",
    "occupancy_from_prediction",
    "occupancy_from_points",
    "select_safety_occupancy",
    "imu_attitude_from_accel",
    "ego_trajectory_to_world",
    "policy_trajectory_to_waypoints",
    "carla_control_from_command",
    "aggregate_episodes",
]


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _grid_n(rng: Tuple[float, float], res: float) -> int:
    return max(1, int(round((float(rng[1]) - float(rng[0])) / res)))


@dataclass
class ClosedLoopConfig:
    """Geometry, control-mapping and metric thresholds for the closed loop."""

    # BEV geometry (must align with BEVFusionConfig / HybridPolicyConfig)
    bev_x_range: Tuple[float, float] = (-12.5, 12.5)
    bev_y_range: Tuple[float, float] = (-12.5, 12.5)
    bev_resolution: float = 0.5
    num_points: int = 256           # LiDAR canonicalisation length
    imu_steps: int = 10
    dt: float = 0.1
    # policy output frame
    policy_frame: str = "ego"       # "ego" | "world"
    occupancy_source: str = "lidar"  # "lidar" | "learned" | "fused"
    # Ackermann -> CARLA VehicleControl mapping
    max_steering: float = 0.5
    max_accel: float = 3.0
    max_decel: float = -5.0
    max_speed: float = 20.0
    # metric thresholds
    goal_tolerance: float = 2.0
    pitch_alarm: float = 0.35       # rad (~20 deg)
    roll_alarm: float = 0.35
    lat_accel_alarm: float = 4.0    # m/s^2

    def __post_init__(self) -> None:
        if self.policy_frame not in ("ego", "world"):
            raise ValueError("policy_frame must be 'ego' or 'world'")
        if self.occupancy_source not in ("lidar", "learned", "fused"):
            raise ValueError(
                "occupancy_source must be 'lidar', 'learned' or 'fused'")


# --------------------------------------------------------------------------
# numpy-only helpers
# --------------------------------------------------------------------------

def canonicalize_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """Pad/truncate raw LiDAR to a FIXED ``(num_points, 4)`` float32 array.

    Accepts ``(N', 3)`` or ``(N', 4)``; missing intensity is filled with 0.
    Zero-pads when the cloud is shorter; truncates when longer. Static shape is
    required by the ONNX/TensorRT graph.
    """
    if num_points <= 0:
        raise ValueError("num_points must be > 0")
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((num_points, 4), dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"points must be (N,3) or (N,4), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite values")
    if pts.shape[1] == 3:
        pts = np.concatenate(
            [pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)
    pts = pts[:, :4]
    if pts.shape[0] >= num_points:
        return pts[:num_points].copy()
    pad = np.zeros((num_points - pts.shape[0], 4), dtype=np.float32)
    return np.concatenate([pts, pad], axis=0)


def imu_samples_to_array(samples, steps: int) -> np.ndarray:
    """Pack ``(accel, gyro)`` samples into a padded ``(steps, 6)`` history."""
    if steps <= 0:
        raise ValueError("steps must be > 0")
    rows = []
    for accel, gyro in samples:
        a = np.asarray(accel, dtype=np.float32).reshape(3)
        g = np.asarray(gyro, dtype=np.float32).reshape(3)
        rows.append(np.concatenate([a, g]))
    if len(rows) < steps:
        rows = [np.zeros(6, dtype=np.float32)] * (steps - len(rows)) + rows
    return np.asarray(rows[-steps:], dtype=np.float32)


def _occupancy_origin(bev_x_range, bev_y_range,
                      vehicle_state: Optional[VehicleState]) -> tuple:
    x0, y0 = float(bev_x_range[0]), float(bev_y_range[0])
    if vehicle_state is None:
        return x0, y0, 0.0
    ca, sa = np.cos(vehicle_state.yaw), np.sin(vehicle_state.yaw)
    return (vehicle_state.x + ca * x0 - sa * y0,
            vehicle_state.y + sa * x0 + ca * y0,
            vehicle_state.yaw)


def occupancy_from_points(
    points: np.ndarray,
    bev_x_range: Tuple[float, float],
    bev_y_range: Tuple[float, float],
    resolution: float,
    height_range: Tuple[float, float] = (-1.5, 2.0),
    vehicle_state: Optional[VehicleState] = None,
) -> OccupancyGrid:
    """Coarse occupancy grid from LiDAR points (cell occupied if >=1 point in
    the ``height_range`` band). Used as the safety-filter collision map when no
    dedicated occupancy head is available.
    """
    h = _grid_n(bev_y_range, resolution)
    w = _grid_n(bev_x_range, resolution)
    grid = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    origin = _occupancy_origin(bev_x_range, bev_y_range, vehicle_state)
    if pts.size == 0 or pts.ndim != 2 or pts.shape[1] < 3:
        return OccupancyGrid(grid, resolution=resolution,
                             origin=origin)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    band = (z >= height_range[0]) & (z <= height_range[1])
    x, y = x[band], y[band]
    cx = ((x - bev_x_range[0]) / resolution).astype(np.int64)
    cy = ((y - bev_y_range[0]) / resolution).astype(np.int64)
    ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    grid[cy[ok], cx[ok]] = 1.0
    return OccupancyGrid(grid, resolution=resolution,
                         origin=origin)


def occupancy_from_prediction(
    prediction,
    bev_x_range: Tuple[float, float],
    bev_y_range: Tuple[float, float],
    resolution: float,
    vehicle_state: Optional[VehicleState] = None,
) -> OccupancyGrid:
    """Convert a planning-oriented occupancy head into the safety grid.

    Only batch-1, channel-1 probabilities are accepted in the online runner;
    ambiguous shapes and uncalibrated values are rejected fail-safe.
    """
    if prediction is None:
        raise ValueError("learned occupancy is required")
    value = prediction.detach().cpu().numpy() if hasattr(
        prediction, "detach") else np.asarray(prediction)
    h = _grid_n(bev_y_range, resolution)
    w = _grid_n(bev_x_range, resolution)
    if value.shape != (1, 1, h, w):
        raise ValueError(
            f"learned occupancy shape must be (1,1,{h},{w}), "
            f"got {value.shape}")
    grid = np.asarray(value[0, 0], dtype=np.float32)
    if not np.isfinite(grid).all():
        raise ValueError("learned occupancy must contain finite values")
    if np.any(grid < 0.0) or np.any(grid > 1.0):
        raise ValueError("learned occupancy values must be in [0, 1]")
    return OccupancyGrid(
        grid, resolution=resolution,
        origin=_occupancy_origin(bev_x_range, bev_y_range, vehicle_state))


def select_safety_occupancy(
    points, prediction,
    bev_x_range: Tuple[float, float],
    bev_y_range: Tuple[float, float],
    resolution: float,
    vehicle_state: VehicleState,
    source: str,
) -> OccupancyGrid:
    """Select an explicit occupancy dependency for planning and safety."""
    source = str(source).lower()
    if source not in ("lidar", "learned", "fused"):
        raise ValueError(f"unsupported occupancy source: {source}")
    lidar = occupancy_from_points(
        points, bev_x_range, bev_y_range, resolution,
        vehicle_state=vehicle_state)
    if source == "lidar":
        return lidar
    learned = occupancy_from_prediction(
        prediction, bev_x_range, bev_y_range, resolution,
        vehicle_state=vehicle_state)
    if source == "learned":
        return learned
    return OccupancyGrid(
        np.maximum(lidar.data, learned.data), resolution=resolution,
        origin=lidar.origin)


def _rotation_align(u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit vector ``u`` onto unit vector ``t``
    (Rodrigues). Identity when ``u == t``; 180 deg about any perpendicular
    axis when ``u == -t``."""
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    t = t / max(float(np.linalg.norm(t)), 1e-9)
    d = float(np.dot(u, t))
    if d > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float32)
    if d < -1.0 + 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(u, axis))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = axis - u * np.dot(u, axis)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float32)
        return (np.eye(3, dtype=np.float32)
                + 2.0 * (K @ K)).astype(np.float32)
    c = np.cross(u, t)
    s = float(np.linalg.norm(c))
    K = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]],
                 dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + K + K @ K * (1.0 - d) / (s * s)
            ).astype(np.float32)


def imu_attitude_from_accel(accel_history: np.ndarray) -> np.ndarray:
    """Estimate a body->gravity-aligned rotation ``(3, 3)`` from IMU accel.

    ``accel_history`` is ``(T, 3)`` body-frame specific force ``[ax, ay, az]``.
    The low-passed body "up" direction (``a / |a|``) is aligned onto the world
    up ``[0, 0, 1]`` via Rodrigues, compensating off-road pitch/roll before
    the LiDAR points are splatted into the BEV grid. Returns the identity when
    the platform is level.
    """
    a = np.asarray(accel_history, dtype=np.float32).reshape(-1, 3)
    if a.shape[0] == 0:
        return np.eye(3, dtype=np.float32)
    ab = a.mean(axis=0)
    norm = float(np.linalg.norm(ab))
    if norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    up = ab / norm
    return _rotation_align(up, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def ego_trajectory_to_world(traj: np.ndarray, state: VehicleState,
                            dt: float = 0.1) -> Trajectory:
    """Rotate ego-relative ``(N, 4)`` waypoints ``[x, y, heading, v]`` into the
    world frame (``x`` forward / ``y`` left convention)."""
    arr = np.asarray(traj, dtype=np.float32).reshape(-1, 4)
    if arr.shape[0] == 0:
        return Trajectory([])
    ca, sa = np.cos(state.yaw), np.sin(state.yaw)
    wps: list[Waypoint] = []
    for i, row in enumerate(arr):
        xe, ye, he, v = (float(row[0]), float(row[1]),
                         float(row[2]), float(row[3]))
        wps.append(Waypoint(
            x=state.x + xe * ca - ye * sa,
            y=state.y + xe * sa + ye * ca,
            yaw=_wrap(state.yaw + he), speed=v, steering=0.0,
            t=(i + 1) * dt))
    return Trajectory(wps, frame="world")


def policy_trajectory_to_waypoints(
        traj: np.ndarray, state: VehicleState,
        frame: str = "ego", dt: float = 0.1) -> Trajectory:
    """Wrap a policy ``(B, N, 4)`` or ``(N, 4)`` output into a world-frame
    :class:`Trajectory`. ``frame='ego'`` rotates by the current pose;
    ``'world'`` assumes the policy already emitted world coordinates."""
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.reshape(-1, 4)
    frame = str(frame).lower()
    if frame not in ("ego", "world"):
        raise ValueError(f"unsupported policy trajectory frame: {frame}")
    if frame == "world":
        return Trajectory([
            Waypoint(x=float(r[0]), y=float(r[1]), yaw=_wrap(float(r[2])),
                     speed=float(r[3]), steering=0.0, t=(i + 1) * dt)
            for i, r in enumerate(arr)], frame="world")
    return ego_trajectory_to_world(arr, state, dt=dt)


def carla_control_from_command(cmd, config: ClosedLoopConfig) -> dict:
    """Map an Ackermann control (duck-typed ``.steering / .speed / .accel``) to
    ``carla.VehicleControl`` kwargs: ``throttle``, ``steer`` and ``brake`` each
    in ``[0, 1]`` (CARLA steer is normalised, not an angle)."""
    steer = float(np.clip(cmd.steering / max(config.max_steering, 1e-6),
                          -1.0, 1.0))
    if cmd.accel >= 0.0:
        throttle = float(np.clip(
            cmd.accel / max(config.max_accel, 1e-6), 0.0, 1.0))
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(np.clip(
            -cmd.accel / max(abs(config.max_decel), 1e-6), 0.0, 1.0))
    return {"throttle": throttle, "steer": steer, "brake": brake}


@dataclass
class EpisodeMetrics:
    """Per-episode pass / collision / attitude-stability counters."""
    steps: int = 0
    collisions: int = 0
    max_impulse: float = 0.0
    progress: float = 0.0
    pitch_max: float = 0.0
    roll_max: float = 0.0
    lat_accel_max: float = 0.0
    pitch_alarms: int = 0
    roll_alarms: int = 0
    lat_accel_alarms: int = 0
    reached_goal: bool = False
    goal_dist: float = float("inf")
    safety_interventions: int = 0
    emergency_stops: int = 0
    raw_policy_risk_steps: int = 0
    post_safety_risk_steps: int = 0
    sensor_health_failures: int = 0
    sensor_health_failure_reasons: dict = field(default_factory=dict)
    sensor_health_latencies_ms: list = field(default_factory=list)

    def record_sensor_health(self, report: SensorHealthReport,
                             latency_seconds: float) -> None:
        """Record one gate decision without changing control counters."""
        self.sensor_health_latencies_ms.append(
            max(float(latency_seconds), 0.0) * 1000.0)
        if report.valid:
            return
        self.sensor_health_failures += 1
        for reason in report.reasons:
            key = reason.value
            self.sensor_health_failure_reasons[key] = (
                self.sensor_health_failure_reasons.get(key, 0) + 1)

    def record_sensor_timeout(self) -> None:
        """Record a missing packet as a health failure before emergency stop."""
        self.sensor_health_failures += 1
        key = SensorHealthReason.PACKET_TIMEOUT.value
        self.sensor_health_failure_reasons[key] = (
            self.sensor_health_failure_reasons.get(key, 0) + 1)

    def update(self, state: VehicleState, prev: Optional[VehicleState],
               config: ClosedLoopConfig) -> None:
        self.steps += 1
        if prev is not None:
            self.progress += float(np.hypot(state.x - prev.x,
                                            state.y - prev.y))
        p, r = abs(state.pitch), abs(state.roll)
        self.pitch_max = max(self.pitch_max, p)
        self.roll_max = max(self.roll_max, r)
        if p > config.pitch_alarm:
            self.pitch_alarms += 1
        if r > config.roll_alarm:
            self.roll_alarms += 1
        a_lat = float(abs(state.speed) * abs(state.yaw_rate))  # v*|yaw_rate|
        self.lat_accel_max = max(self.lat_accel_max, a_lat)
        if a_lat > config.lat_accel_alarm:
            self.lat_accel_alarms += 1

    def to_summary(self) -> dict:
        success = self.reached_goal and self.collisions == 0
        alarms = (self.pitch_alarms + self.roll_alarms
                  + self.lat_accel_alarms)
        latency_p50 = (float(np.percentile(
            self.sensor_health_latencies_ms, 50))
            if self.sensor_health_latencies_ms else 0.0)
        latency_p95 = (float(np.percentile(
            self.sensor_health_latencies_ms, 95))
            if self.sensor_health_latencies_ms else 0.0)
        return {
            "success": success, "collisions": self.collisions,
            "max_impulse": self.max_impulse, "progress": self.progress,
            "pitch_max": self.pitch_max, "roll_max": self.roll_max,
            "lat_accel_max": self.lat_accel_max,
            "attitude_alarms": alarms, "steps": self.steps,
            "goal_dist": self.goal_dist,
            "safety_interventions": self.safety_interventions,
            "emergency_stops": self.emergency_stops,
            "safety_intervention_rate": (
                self.safety_interventions / max(self.steps, 1)),
            "raw_policy_risk_rate": (
                self.raw_policy_risk_steps / max(self.steps, 1)),
            "post_safety_risk_rate": (
                self.post_safety_risk_steps / max(self.steps, 1)),
            "sensor_health_failures": self.sensor_health_failures,
            "sensor_health_failure_reasons": dict(
                self.sensor_health_failure_reasons),
            "sensor_health_latency_p50_ms": latency_p50,
            "sensor_health_latency_p95_ms": latency_p95,
        }


def aggregate_episodes(metrics: list) -> dict:
    """Pool per-episode summaries into rates (pass / collision / stability)."""
    n = len(metrics)
    if n == 0:
        return {"n_episodes": 0, "pass_rate": 0.0, "collision_rate": 0.0}
    sums = [m.to_summary() for m in metrics]
    passes = sum(1 for s in sums if s["success"])
    crashes = sum(1 for s in sums if s["collisions"] > 0)
    health_reasons = Counter()
    for summary in sums:
        health_reasons.update(summary["sensor_health_failure_reasons"])
    return {
        "n_episodes": n,
        "pass_rate": passes / n,
        "collision_rate": crashes / n,
        "mean_progress": float(np.mean([s["progress"] for s in sums])),
        "max_impulse": float(np.max([s["max_impulse"] for s in sums])),
        "pitch_max": float(np.max([s["pitch_max"] for s in sums])),
        "roll_max": float(np.max([s["roll_max"] for s in sums])),
        "lat_accel_max": float(np.max([s["lat_accel_max"] for s in sums])),
        "mean_attitude_alarms":
            float(np.mean([s["attitude_alarms"] for s in sums])),
        "mean_safety_intervention_rate": float(np.mean(
            [s["safety_intervention_rate"] for s in sums])),
        "total_emergency_stops": int(sum(
            s["emergency_stops"] for s in sums)),
        "mean_raw_policy_risk_rate": float(np.mean(
            [s["raw_policy_risk_rate"] for s in sums])),
        "mean_post_safety_risk_rate": float(np.mean(
            [s["post_safety_risk_rate"] for s in sums])),
        "total_sensor_health_failures": int(sum(
            s["sensor_health_failures"] for s in sums)),
        "sensor_health_failure_reasons": dict(sorted(health_reasons.items())),
        "max_sensor_health_latency_p95_ms": float(max(
            s["sensor_health_latency_p95_ms"] for s in sums)),
    }


# --------------------------------------------------------------------------
# CARLA glue (`import carla` is lazy; helpers above stay CI-friendly)
# --------------------------------------------------------------------------

class CarlaSensorStack:
    """Spawn & synchronously poll RGB cameras, LiDAR, IMU and collision sensors
    on a CARLA ego vehicle. Requires synchronous world mode."""

    def __init__(self, world, vehicle, config: ClosedLoopConfig,
                 num_cameras: int = 3, image_size: Tuple[int, int] = (192, 192),
                 lidar_channels: int = 32, lidar_range: float = 50.0,
                 calibration_version: str = "carla-default-v1",
                 sensor_attributes: Optional[dict] = None,
                 baseline: Optional[CarlaBaselineConfig] = None):
        if not _HAS_CARLA:
            raise RuntimeError("carla PythonAPI not installed; run inside a "
                               "CARLA environment (`pip install carla`).")
        self._world = world
        self._vehicle = vehicle
        self._config = config
        self.calibration_version = str(calibration_version)
        self._cam_q = [deque(maxlen=1) for _ in range(num_cameras)]
        self._lidar_q = deque(maxlen=1)
        self._imu_q = deque(maxlen=config.imu_steps)
        self._collision_count = 0
        self._max_impulse = 0.0
        bp = world.get_blueprint_library()
        sensor_attributes = sensor_attributes or (
            carla_sensor_attributes(baseline, image_size)
            if baseline is not None else {
                "camera": {
                    "image_size_x": str(image_size[0]),
                    "image_size_y": str(image_size[1]),
                    "fov": "90",
                },
                "lidar": {
                    "channels": str(lidar_channels),
                    "range": str(lidar_range),
                    "points_per_second": str(lidar_channels * 1000),
                    "rotation_frequency": "20",
                    "upper_fov": "15",
                    "lower_fov": "-25",
                },
            })

        def transform_from_config(sensor_transform):
            location = sensor_transform.location_m
            rotation = sensor_transform.rotation_degrees
            return carla.Transform(
                carla.Location(x=location[0], y=location[1], z=location[2]),
                carla.Rotation(
                    roll=rotation[0], pitch=rotation[1], yaw=rotation[2]))

        if baseline is not None:
            if len(baseline.cameras) != num_cameras:
                raise ValueError("baseline camera count differs from num_cameras")
            camera_specs = [
                (camera.name, camera.blueprint,
                 transform_from_config(camera.transform))
                for camera in baseline.cameras
            ]
            lidar_blueprint = baseline.lidar.blueprint
            lidar_transform = transform_from_config(baseline.lidar.transform)
            imu_blueprint = baseline.imu.blueprint
            imu_transform = transform_from_config(baseline.imu.transform)
        else:
            cam_tfs = [
                carla.Transform(carla.Location(x=1.5, z=1.6)),
                carla.Transform(carla.Location(x=-1.5, z=1.6),
                                carla.Rotation(yaw=180)),
                carla.Transform(carla.Location(x=0.0, z=1.9)),
            ][:num_cameras]
            camera_specs = [
                (f"camera_{index}", "sensor.camera.rgb", transform)
                for index, transform in enumerate(cam_tfs)
            ]
            lidar_blueprint = "sensor.lidar.ray_cast"
            lidar_transform = carla.Transform(carla.Location(x=0.0, z=1.9))
            imu_blueprint = "sensor.other.imu"
            imu_transform = carla.Transform(carla.Location(x=0.0, z=0.5))
        self._cams = []
        for i, (camera_name, blueprint_name, tf) in enumerate(camera_specs):
            bpc = bp.find(blueprint_name)
            camera_attributes = (
                sensor_attributes["cameras"][camera_name]
                if "cameras" in sensor_attributes
                else sensor_attributes["camera"])
            for name, value in camera_attributes.items():
                bpc.set(name, value)
            cam = world.spawn_actor(bpc, tf, attach_to=vehicle)
            cam.listen(self._cam_q[i].append)
            self._cams.append(cam)
        bpl = bp.find(lidar_blueprint)
        for name, value in sensor_attributes["lidar"].items():
            bpl.set(name, value)
        self._lidar = world.spawn_actor(
            bpl, lidar_transform,
            attach_to=vehicle)
        self._lidar.listen(self._on_lidar)
        bpi = bp.find(imu_blueprint)
        for name, value in sensor_attributes.get("imu", {}).items():
            bpi.set(name, value)
        self._imu = world.spawn_actor(
            bpi, imu_transform,
            attach_to=vehicle)
        self._imu.listen(self._on_imu)
        bpc = bp.find("sensor.other.collision")
        self._col = world.spawn_actor(
            bpc, carla.Transform(), attach_to=vehicle)
        self._col.listen(self._on_collision)
        self._actors = self._cams + [self._lidar, self._imu, self._col]
        self.last_frame = -1
        self.last_camera_frames: Tuple[int, ...] = ()
        self.last_lidar_frame = -1
        self.last_imu_frames: Tuple[int, ...] = ()
        self.last_camera_timestamps: Tuple[float, ...] = ()
        self.last_lidar_timestamp = float("nan")
        self.last_imu_timestamps: Tuple[float, ...] = ()
        self.last_sensor_timestamps: dict = {}
        self.last_reference_timestamp = float("nan")

    def _on_lidar(self, event):
        data = np.frombuffer(event.raw_data,
                             dtype=np.float32).reshape(-1, 4)
        self._lidar_q.append((event.frame, event.timestamp,
                              np.ascontiguousarray(data)))

    def _on_imu(self, event):
        self._imu_q.append((event.frame, event.timestamp,
                            np.array([
                                event.accelerometer.x, event.accelerometer.y,
                                event.accelerometer.z], dtype=np.float32),
                            np.array([
                                event.gyroscope.x, event.gyroscope.y,
                                event.gyroscope.z], dtype=np.float32)))

    def _on_collision(self, event):
        self._collision_count += 1
        ni = event.normal_impulse
        self._max_impulse = max(
            self._max_impulse,
            float(np.sqrt(ni.x ** 2 + ni.y ** 2 + ni.z ** 2)))

    def tick(self, timeout: float = 2.0):
        """Advance the synchronous world one step and return the latest
        ``(images, points, imu)`` packet, or ``None`` on timeout."""
        frame = self._world.tick()
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while (any(not q or q[-1].frame != frame for q in self._cam_q)
               or not self._lidar_q or self._lidar_q[-1][0] != frame):
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.001)
        cams = [q[-1] for q in self._cam_q]
        images = [
            np.frombuffer(c.raw_data, dtype=np.uint8).reshape(
                c.height, c.width, 4)[:, :, :3].copy() for c in cams]
        points = (self._lidar_q[-1][2] if self._lidar_q
                  else np.zeros((0, 4), np.float32))
        samples = [(a, g) for f, _, a, g in self._imu_q if f <= frame]
        imu = imu_samples_to_array(samples, self._config.imu_steps)
        self.last_frame = int(frame)
        self.last_camera_frames = tuple(int(c.frame) for c in cams)
        self.last_lidar_frame = int(self._lidar_q[-1][0])
        self.last_imu_frames = tuple(
            int(f) for f, _, _, _ in self._imu_q if f <= frame
        )[-self._config.imu_steps:]
        imu_timestamps = tuple(
            float(ts) for f, ts, _, _ in self._imu_q if f <= frame
        )[-self._config.imu_steps:]
        self.last_camera_timestamps = tuple(
            float(c.timestamp) for c in cams)
        self.last_lidar_timestamp = float(self._lidar_q[-1][1])
        self.last_imu_timestamps = imu_timestamps
        self.last_sensor_timestamps = {
            **{f"camera_{i}": float(c.timestamp)
               for i, c in enumerate(cams)},
            "lidar": float(self._lidar_q[-1][1]),
        }
        if imu_timestamps:
            self.last_sensor_timestamps["imu_latest"] = imu_timestamps[-1]
        snapshot = self._world.get_snapshot()
        if int(snapshot.frame) != int(frame):
            raise RuntimeError(
                f"CARLA snapshot frame {snapshot.frame} != tick frame {frame}")
        self.last_reference_timestamp = float(
            snapshot.timestamp.elapsed_seconds)
        return images, points, imu

    def collision_stats(self) -> Tuple[int, float]:
        return self._collision_count, self._max_impulse

    def reset_episode(self) -> None:
        self._collision_count = 0
        self._max_impulse = 0.0

    def destroy(self) -> None:
        for a in self._actors:
            try:
                a.destroy()
            except Exception:
                pass


class CarlaClosedLoopRunner:
    """One CARLA episode: tick -> perceive -> policy -> safety -> control.

    ``perceive``/``policy``/``safety_filter``/``controller`` are duck-typed so
    this module stays importable without torch / rclpy (wired in the script).
    """

    def __init__(self, world, vehicle, sensors: CarlaSensorStack,
                 perceive: Callable, policy: Callable, safety_filter,
                 controller, config: ClosedLoopConfig,
                 health_gate: SensorHealthGate,
                 goal: Optional[Tuple[float, float]] = None,
                 max_steps: int = 1000,
                 supervisor: Optional[SafetySupervisor] = None):
        self._world = world
        self._vehicle = vehicle
        self._sensors = sensors
        self._perceive = perceive
        self._policy = policy
        self._safety = safety_filter
        self._controller = controller
        self._config = config
        self._health_gate = health_gate
        self._goal = goal
        self._max_steps = max_steps
        self._supervisor = supervisor or SafetySupervisor(
            SafetySupervisorConfig(expected_frame="world"))

    def _apply_emergency(self, state: VehicleState, metrics: EpisodeMetrics,
                         timestamp: Optional[float] = None) -> None:
        emergency_timestamp = 0.0 if timestamp is None else float(timestamp)
        cmd = self._controller.compute(
            state, self._supervisor.emergency_trajectory(emergency_timestamp),
            self._config.dt)
        self._vehicle.apply_control(
            carla.VehicleControl(**carla_control_from_command(
                cmd, self._config)))
        metrics.emergency_stops += 1

    def _state_from_vehicle(self, prev: Optional[VehicleState]) -> VehicleState:
        tr = self._vehicle.get_transform()
        v = self._vehicle.get_velocity()
        ctl = self._vehicle.get_control()
        fwd = tr.get_forward_vector()
        yaw = float(np.arctan2(fwd.y, fwd.x))
        pitch = float(np.radians(tr.rotation.pitch))
        roll = float(np.radians(tr.rotation.roll))
        speed = float(np.hypot(v.x, v.y))
        yaw_rate = 0.0
        if prev is not None:
            yaw_rate = _wrap(yaw - prev.yaw) / max(self._config.dt, 1e-6)
        return VehicleState(
            x=float(tr.location.x), y=float(tr.location.y), yaw=yaw,
            speed=speed, steering=float(ctl.steer) * self._config.max_steering,
            pitch=pitch, roll=roll, vx=float(v.x), vy=float(v.y),
            yaw_rate=yaw_rate, accel_z=0.0)

    def run(self) -> EpisodeMetrics:
        metrics = EpisodeMetrics()
        self._sensors.reset_episode()
        self._health_gate.reset()
        prev: Optional[VehicleState] = None
        for _ in range(self._max_steps):
            packet = self._sensors.tick()
            if packet is None:
                state = prev or VehicleState()
                metrics.record_sensor_timeout()
                self._apply_emergency(state, metrics)
                break
            images, points, imu = packet
            state = self._state_from_vehicle(prev)
            reference_timestamp = float(getattr(
                self._sensors, "last_reference_timestamp",
                self._sensors.last_frame * self._config.dt))
            health_started = time.monotonic()
            health_report = self._health_gate.evaluate(
                images=images, point_cloud=points, imu_history=imu,
                frame_id=self._sensors.last_frame,
                reference_timestamp=reference_timestamp,
                camera_frames=self._sensors.last_camera_frames,
                lidar_frame=self._sensors.last_lidar_frame,
                imu_frames=self._sensors.last_imu_frames,
                camera_timestamps=getattr(
                    self._sensors, "last_camera_timestamps", ()),
                lidar_timestamp=getattr(
                    self._sensors, "last_lidar_timestamp", float("nan")),
                imu_timestamps=getattr(
                    self._sensors, "last_imu_timestamps", ()),
                calibration_version=getattr(
                    self._sensors, "calibration_version", None))
            metrics.record_sensor_health(
                health_report, time.monotonic() - health_started)
            if not health_report.valid:
                self._apply_emergency(
                    state, metrics, reference_timestamp)
                break
            try:
                observation = Observation(
                    timestamp=reference_timestamp,
                    simulator_frame=self._sensors.last_frame,
                    images=images, point_cloud=points, imu_history=imu,
                    ego_state=state,
                    camera_frames=self._sensors.last_camera_frames,
                    lidar_frame=self._sensors.last_lidar_frame,
                    imu_frames=self._sensors.last_imu_frames,
                    sensor_timestamps=self._sensors.last_sensor_timestamps)
                attitude = imu_attitude_from_accel(imu)
                model_started = time.monotonic()
                perceived = self._perceive(
                    observation.images, observation.point_cloud,
                    observation.imu_history, attitude)
                bev = getattr(perceived, "bev", perceived)
                learned_occupancy = getattr(perceived, "occupancy", None)
                traj_raw = self._policy(bev, observation.imu_history)
                model_latency = time.monotonic() - model_started
                traj_np = (traj_raw.detach().cpu().numpy()
                           if hasattr(traj_raw, "detach")
                           else np.asarray(traj_raw))
                traj = policy_trajectory_to_waypoints(
                    traj_np, state, frame=self._config.policy_frame,
                    dt=self._config.dt)
                traj.timestamp = observation.timestamp
                occ = select_safety_occupancy(
                    points, learned_occupancy,
                    self._config.bev_x_range, self._config.bev_y_range,
                    self._config.bev_resolution, state,
                    self._config.occupancy_source)
            except Exception:
                self._apply_emergency(
                    state, metrics, reference_timestamp)
                break
            decision = self._supervisor.evaluate(
                traj, sensor_skew=observation.max_sensor_skew_seconds,
                model_latency=model_latency)
            if (decision.mode is not SafetyMode.EMERGENCY_STOP and traj.length
                    and self._safety.check_collision(traj, occ).any()):
                metrics.raw_policy_risk_steps += 1
            if decision.mode is SafetyMode.EMERGENCY_STOP:
                safe_traj = self._supervisor.emergency_trajectory(
                    observation.timestamp)
                metrics.emergency_stops += 1
            else:
                safe_traj = self._safety.filter(traj, state, occ)
                intervened = (safe_traj.length != traj.length
                              or not np.allclose(safe_traj.to_array(),
                                                 traj.to_array(), atol=1e-4))
                post = self._supervisor.evaluate(
                    safe_traj, safety_intervened=intervened)
                if post.mode is SafetyMode.DEGRADED:
                    metrics.safety_interventions += 1
                elif post.mode is SafetyMode.EMERGENCY_STOP:
                    safe_traj = self._supervisor.emergency_trajectory(
                        observation.timestamp)
                    metrics.emergency_stops += 1
            if (safe_traj.length
                    and self._safety.check_collision(safe_traj, occ).any()):
                metrics.post_safety_risk_steps += 1
            cmd = self._controller.compute(state, safe_traj, self._config.dt)
            self._vehicle.apply_control(
                carla.VehicleControl(**carla_control_from_command(
                    cmd, self._config)))
            metrics.update(state, prev, self._config)
            cols, imp = self._sensors.collision_stats()
            metrics.collisions = cols
            metrics.max_impulse = max(metrics.max_impulse, imp)
            if self._goal is not None:
                d = float(np.hypot(state.x - self._goal[0],
                                   state.y - self._goal[1]))
                metrics.goal_dist = d
                if d <= self._config.goal_tolerance:
                    metrics.reached_goal = True
                    break
            prev = state
        return metrics
