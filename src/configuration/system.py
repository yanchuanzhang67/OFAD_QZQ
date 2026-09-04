"""Build one validated ORAD runtime stack from ``configs/system.yaml``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from affordance.terrain import TerrainAffordanceConfig
from orad_ros2.vehicle_control_node import PurePursuitConfig
from perception.bev_fusion import BEVFusionConfig
from policy.hybrid_policy import HybridPolicyConfig
from safety.kinematic_filter import SafetyFilterConfig
from sim.carla_closed_loop import ClosedLoopConfig
from sim.carla_baseline import CarlaBaselineConfig
from sim.carla_baseline import CarlaCameraConfig
from sim.carla_baseline import CarlaImuConfig
from sim.carla_baseline import CarlaLidarConfig
from sim.carla_baseline import CarlaTransformConfig
from sim.carla_baseline import camera_extrinsic_new_orad
from sim.carla_baseline import camera_intrinsic_matrix
from sim.carla_baseline import canonical_yaml_sha256
from sim.carla_baseline import sha256_file
from sim.domain_randomization import DomainRandomizationConfig
from utils.contracts import validate_stack_configs
from utils.sensor_health import SensorHealthConfig


_SECTION_KEYS = {
    "geometry": {
        "bev_x_range", "bev_y_range", "bev_resolution", "bev_channels",
    },
    "sensors": {
        "num_cameras", "image_size", "num_points", "imu_steps",
        "imu_channels", "calibration_version",
    },
    "sensor_health": {
        "max_sensor_age_seconds", "max_sensor_skew_seconds",
        "black_pixel_threshold", "saturation_pixel_threshold",
        "max_black_ratio", "max_saturation_ratio",
        "max_consecutive_identical_frames", "min_lidar_points",
        "min_lidar_valid_ratio", "max_lidar_abs_coordinate",
        "max_lidar_duplicate_ratio", "max_accel_abs", "max_gyro_abs",
        "max_accel_step_delta", "max_gyro_step_delta",
        "max_imu_sample_gap_seconds",
    },
    "vehicle": {
        "wheelbase", "max_steering", "max_steering_rate", "max_speed",
        "max_accel", "max_decel", "max_lateral_accel",
    },
    "control": {"dt"},
    "runtime": {"policy_frame", "occupancy_source"},
    "carla_baseline": {
        "version", "map_name", "vehicle_blueprint", "random_seed",
        "repetitions", "cameras", "lidar", "imu",
    },
    "domain_randomization": {
        "tire_friction", "vehicle_mass", "lidar_sigma", "lidar_dropout",
        "suspension_damping",
    },
    "affordance": {
        "enabled", "hidden_channels", "traversability_weight",
        "roughness_weight", "speed_epsilon",
    },
    "policy_training": {
        "imagination_horizon", "bc_xy_weight", "bc_heading_weight",
        "bc_speed_weight", "bc_smooth_weight",
    },
}

_CARLA_CAMERA_KEYS = {
    "name", "blueprint", "location_m", "rotation_degrees",
    "fov_degrees", "sensor_tick_seconds",
}
_CARLA_LIDAR_KEYS = {
    "blueprint", "location_m", "rotation_degrees", "channels",
    "range_meters", "points_per_second", "rotation_frequency_hz",
    "upper_fov_degrees", "lower_fov_degrees", "sensor_tick_seconds",
    "canonical_points", "canonical_y_flip",
}
_CARLA_IMU_KEYS = {
    "blueprint", "location_m", "rotation_degrees", "sensor_tick_seconds",
    "history_steps", "channels",
}


@dataclass(frozen=True)
class SystemStackConfig:
    config_path: Path
    config_sha256: str
    config_canonical_sha256: str
    bev: BEVFusionConfig
    policy: HybridPolicyConfig
    safety: SafetyFilterConfig
    controller: PurePursuitConfig
    closed_loop: ClosedLoopConfig
    affordance: TerrainAffordanceConfig
    affordance_enabled: bool
    domain_randomization: DomainRandomizationConfig
    sensor_health: SensorHealthConfig
    carla_baseline: CarlaBaselineConfig


def _tuple2(value, name: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return tuple(value)


def _tuple3(value, name: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(float(item) for item in value)


def _validate_keys(values, allowed, name: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = sorted(set(values) - allowed)
    missing = sorted(allowed - set(values))
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    if missing:
        raise ValueError(f"missing {name} keys: {missing}")


def _validate_schema(raw: dict) -> None:
    if not isinstance(raw, dict):
        raise ValueError("system config root must be a mapping")
    actual = set(raw)
    expected = set(_SECTION_KEYS)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"unknown system config sections: {unknown}")
    if missing:
        raise ValueError(f"missing system config sections: {missing}")
    baseline = raw.get("carla_baseline")
    if isinstance(baseline, dict) and "camera_fov_degrees" in baseline:
        raise ValueError(
            "legacy carla_baseline.camera_fov_degrees is unsupported; "
            "migrate to carla_baseline.cameras")
    for section, allowed in _SECTION_KEYS.items():
        values = raw[section]
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        unknown_keys = sorted(set(values) - allowed)
        missing_keys = sorted(allowed - set(values))
        if unknown_keys:
            raise ValueError(f"unknown {section} keys: {unknown_keys}")
        if missing_keys:
            raise ValueError(f"missing {section} keys: {missing_keys}")
    cameras = raw["carla_baseline"]["cameras"]
    if not isinstance(cameras, list) or len(cameras) != 3:
        raise ValueError("carla_baseline.cameras must contain exactly three items")
    for index, camera in enumerate(cameras):
        _validate_keys(
            camera, _CARLA_CAMERA_KEYS,
            f"carla_baseline.cameras[{index}]")
    _validate_keys(
        raw["carla_baseline"]["lidar"], _CARLA_LIDAR_KEYS,
        "carla_baseline.lidar")
    _validate_keys(
        raw["carla_baseline"]["imu"], _CARLA_IMU_KEYS,
        "carla_baseline.imu")


def _transform_config(raw: dict, name: str) -> CarlaTransformConfig:
    return CarlaTransformConfig(
        location_m=_tuple3(raw["location_m"], f"{name}.location_m"),
        rotation_degrees=_tuple3(
            raw["rotation_degrees"], f"{name}.rotation_degrees"),
    )


def _load_carla_baseline(raw: dict) -> CarlaBaselineConfig:
    cameras = tuple(
        CarlaCameraConfig(
            name=str(camera["name"]),
            blueprint=str(camera["blueprint"]),
            transform=_transform_config(
                camera, f"carla_baseline.cameras[{index}]"),
            fov_degrees=float(camera["fov_degrees"]),
            sensor_tick_seconds=float(camera["sensor_tick_seconds"]),
        )
        for index, camera in enumerate(raw["cameras"])
    )
    lidar_raw = raw["lidar"]
    lidar = CarlaLidarConfig(
        blueprint=str(lidar_raw["blueprint"]),
        transform=_transform_config(lidar_raw, "carla_baseline.lidar"),
        channels=int(lidar_raw["channels"]),
        range_meters=float(lidar_raw["range_meters"]),
        points_per_second=int(lidar_raw["points_per_second"]),
        rotation_frequency_hz=float(lidar_raw["rotation_frequency_hz"]),
        upper_fov_degrees=float(lidar_raw["upper_fov_degrees"]),
        lower_fov_degrees=float(lidar_raw["lower_fov_degrees"]),
        sensor_tick_seconds=float(lidar_raw["sensor_tick_seconds"]),
        canonical_points=int(lidar_raw["canonical_points"]),
        canonical_y_flip=bool(lidar_raw["canonical_y_flip"]),
    )
    imu_raw = raw["imu"]
    imu = CarlaImuConfig(
        blueprint=str(imu_raw["blueprint"]),
        transform=_transform_config(imu_raw, "carla_baseline.imu"),
        sensor_tick_seconds=float(imu_raw["sensor_tick_seconds"]),
        history_steps=int(imu_raw["history_steps"]),
        channels=int(imu_raw["channels"]),
    )
    return CarlaBaselineConfig(
        version=str(raw["version"]),
        map_name=str(raw["map_name"]),
        vehicle_blueprint=str(raw["vehicle_blueprint"]),
        random_seed=int(raw["random_seed"]),
        repetitions=int(raw["repetitions"]),
        cameras=cameras,
        lidar=lidar,
        imu=imu,
    )


def _validate_carla_consistency(
        baseline: CarlaBaselineConfig, *, control_dt: float,
        num_points: int, imu_steps: int, imu_channels: int) -> None:
    for camera in baseline.cameras:
        if abs(camera.sensor_tick_seconds - control_dt) > 1e-9:
            raise ValueError(
                f"camera {camera.name} sensor_tick_seconds must equal control.dt")
    if abs(baseline.lidar.sensor_tick_seconds - control_dt) > 1e-9:
        raise ValueError("lidar sensor_tick_seconds must equal control.dt")
    if abs(baseline.imu.sensor_tick_seconds - control_dt) > 1e-9:
        raise ValueError("imu sensor_tick_seconds must equal control.dt")
    if baseline.lidar.canonical_points != num_points:
        raise ValueError(
            "lidar canonical_points must equal sensors.num_points")
    if baseline.imu.history_steps != imu_steps:
        raise ValueError("imu history_steps must equal sensors.imu_steps")
    if baseline.imu.channels != imu_channels:
        raise ValueError("imu channels must equal sensors.imu_channels")


def load_system_stack(path, *, policy_frame: Optional[str] = None,
                      occupancy_source: Optional[str] = None
                      ) -> SystemStackConfig:
    """Load strict YAML and derive every duplicated runtime dataclass.

    Command-line overrides are deliberately limited to run-mode choices;
    geometry, tensor shapes and physical limits always come from the YAML.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_schema(raw)
    geometry = raw["geometry"]
    sensors = raw["sensors"]
    health_raw = raw["sensor_health"]
    vehicle = raw["vehicle"]
    control = raw["control"]
    runtime = raw["runtime"]
    carla_baseline_raw = raw["carla_baseline"]
    training = raw["policy_training"]
    affordance_raw = raw["affordance"]
    dr = raw["domain_randomization"]

    x_range = _tuple2(geometry["bev_x_range"], "geometry.bev_x_range")
    y_range = _tuple2(geometry["bev_y_range"], "geometry.bev_y_range")
    image_size = _tuple2(sensors["image_size"], "sensors.image_size")
    resolution = float(geometry["bev_resolution"])
    bev_h = max(1, int(round((y_range[1] - y_range[0]) / resolution)))
    bev_w = max(1, int(round((x_range[1] - x_range[0]) / resolution)))

    carla_baseline = _load_carla_baseline(carla_baseline_raw)
    _validate_carla_consistency(
        carla_baseline, control_dt=float(control["dt"]),
        num_points=int(sensors["num_points"]),
        imu_steps=int(sensors["imu_steps"]),
        imu_channels=int(sensors["imu_channels"]))
    bev = BEVFusionConfig(
        num_cameras=int(sensors["num_cameras"]),
        image_size=tuple(int(v) for v in image_size),
        num_points=int(sensors["num_points"]),
        bev_x_range=x_range, bev_y_range=y_range,
        bev_resolution=resolution,
        bev_channels=int(geometry["bev_channels"]),
        imu_in_channels=int(sensors["imu_channels"]),
        imu_steps=int(sensors["imu_steps"]),
        camera_intrinsics=[
            camera_intrinsic_matrix(camera, tuple(int(v) for v in image_size))
            for camera in carla_baseline.cameras
        ],
        camera_extrinsics=[
            camera_extrinsic_new_orad(camera)
            for camera in carla_baseline.cameras
        ],
    )
    policy = HybridPolicyConfig(
        bev_channels=bev.bev_channels, bev_h=bev_h, bev_w=bev_w,
        imu_in_channels=bev.imu_in_channels, imu_steps=bev.imu_steps,
        imagine_horizon=int(training["imagination_horizon"]),
        bc_xy_weight=float(training["bc_xy_weight"]),
        bc_heading_weight=float(training["bc_heading_weight"]),
        bc_speed_weight=float(training["bc_speed_weight"]),
        bc_smooth_weight=float(training["bc_smooth_weight"]),
    )
    safety = SafetyFilterConfig(
        wheelbase=float(vehicle["wheelbase"]),
        max_steering=float(vehicle["max_steering"]),
        max_steering_rate=float(vehicle["max_steering_rate"]),
        max_accel=float(vehicle["max_accel"]),
        max_decel=float(vehicle["max_decel"]),
        max_lateral_accel=float(vehicle["max_lateral_accel"]),
        dt=float(control["dt"]),
    )
    controller = PurePursuitConfig(
        wheelbase=safety.wheelbase, max_steering=safety.max_steering,
        max_speed=float(vehicle["max_speed"]),
        max_lateral_accel=safety.max_lateral_accel,
        max_accel=safety.max_accel, max_decel=safety.max_decel,
    )
    closed_loop = ClosedLoopConfig(
        bev_x_range=x_range, bev_y_range=y_range,
        bev_resolution=resolution, num_points=bev.num_points,
        imu_steps=bev.imu_steps, dt=safety.dt,
        policy_frame=(policy_frame or runtime["policy_frame"]),
        occupancy_source=(occupancy_source or runtime["occupancy_source"]),
        max_steering=safety.max_steering,
        max_accel=safety.max_accel, max_decel=safety.max_decel,
        max_speed=controller.max_speed,
        lat_accel_alarm=safety.max_lateral_accel,
    )
    affordance = TerrainAffordanceConfig(
        in_channels=bev.bev_channels,
        hidden_channels=int(affordance_raw["hidden_channels"]),
        traversability_weight=float(
            affordance_raw["traversability_weight"]),
        roughness_weight=float(affordance_raw["roughness_weight"]),
        speed_epsilon=float(affordance_raw["speed_epsilon"]),
    )
    domain_randomization = DomainRandomizationConfig(**{
        name: _tuple2(value, f"domain_randomization.{name}")
        for name, value in dr.items()
    })
    sensor_health = SensorHealthConfig(
        num_cameras=bev.num_cameras,
        image_size=bev.image_size,
        imu_steps=bev.imu_steps,
        expected_calibration_version=str(sensors["calibration_version"]),
        max_sensor_age_seconds=float(
            health_raw["max_sensor_age_seconds"]),
        max_sensor_skew_seconds=float(
            health_raw["max_sensor_skew_seconds"]),
        black_pixel_threshold=int(health_raw["black_pixel_threshold"]),
        saturation_pixel_threshold=int(
            health_raw["saturation_pixel_threshold"]),
        max_black_ratio=float(health_raw["max_black_ratio"]),
        max_saturation_ratio=float(health_raw["max_saturation_ratio"]),
        max_consecutive_identical_frames=int(
            health_raw["max_consecutive_identical_frames"]),
        min_lidar_points=int(health_raw["min_lidar_points"]),
        min_lidar_valid_ratio=float(health_raw["min_lidar_valid_ratio"]),
        max_lidar_abs_coordinate=float(
            health_raw["max_lidar_abs_coordinate"]),
        max_lidar_duplicate_ratio=float(
            health_raw["max_lidar_duplicate_ratio"]),
        max_accel_abs=float(health_raw["max_accel_abs"]),
        max_gyro_abs=float(health_raw["max_gyro_abs"]),
        max_accel_step_delta=float(health_raw["max_accel_step_delta"]),
        max_gyro_step_delta=float(health_raw["max_gyro_step_delta"]),
        max_imu_sample_gap_seconds=float(
            health_raw["max_imu_sample_gap_seconds"]),
    )
    stack = SystemStackConfig(
        config_path=config_path.resolve(),
        config_sha256=sha256_file(config_path),
        config_canonical_sha256=canonical_yaml_sha256(config_path),
        bev=bev, policy=policy, safety=safety, controller=controller,
        closed_loop=closed_loop, affordance=affordance,
        affordance_enabled=bool(affordance_raw["enabled"]),
        domain_randomization=domain_randomization,
        sensor_health=sensor_health,
        carla_baseline=carla_baseline,
    )
    validate_stack_configs(
        stack.bev, stack.policy, stack.safety, stack.controller,
        stack.closed_loop)
    return stack
