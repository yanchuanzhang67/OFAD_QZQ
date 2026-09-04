from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from configuration.system import load_system_stack
from utils.contracts import validate_stack_configs

pytestmark = pytest.mark.unit


_ROOT_CONFIG = Path(__file__).parents[3] / "configs" / "system.yaml"


def test_system_yaml_builds_one_compatible_runtime_stack():
    stack = load_system_stack(_ROOT_CONFIG)

    assert stack.config_path == _ROOT_CONFIG.resolve()
    assert len(stack.config_sha256) == 64
    assert len(stack.config_canonical_sha256) == 64
    assert stack.bev.num_cameras == 3
    assert stack.bev.image_size == (192, 192)
    assert stack.bev.bev_channels == stack.policy.bev_channels == 32
    assert stack.bc_policy.family == "pure_bc_v1"
    assert stack.bc_policy.bev_channels == stack.bev.bev_channels == 32
    assert stack.bc_policy.bev_h == stack.bc_policy.bev_w == 50
    assert stack.bc_policy.imu_steps == 10
    assert stack.bc_policy.ego_dim == 8
    assert stack.bc_policy.horizon == 20
    assert stack.bc_policy.traj_dim == 4
    assert stack.bc_policy.waypoint_dt == stack.closed_loop.dt == 0.1
    assert stack.policy.imagine_horizon == 5
    assert stack.closed_loop.occupancy_source == "lidar"
    assert stack.sensor_health.num_cameras == stack.bev.num_cameras
    assert stack.sensor_health.image_size == stack.bev.image_size
    assert stack.sensor_health.imu_steps == stack.bev.imu_steps
    assert stack.sensor_health.expected_calibration_version == "carla-default-v1"
    assert stack.sensor_health.min_lidar_points == 16
    assert stack.sensor_health.max_sensor_age_seconds == 0.2
    assert stack.carla_baseline.version == "0.9.16"
    assert stack.carla_baseline.map_name == "Town10HD_Opt"
    assert stack.carla_baseline.vehicle_blueprint == "vehicle.lincoln.mkz_2020"
    assert stack.carla_baseline.random_seed == 42
    assert stack.carla_baseline.repetitions == 3
    assert [camera.name for camera in stack.carla_baseline.cameras] == [
        "front", "rear", "top"]
    assert [camera.fov_degrees for camera in stack.carla_baseline.cameras] == [
        90.0, 90.0, 100.0]
    assert stack.carla_baseline.cameras[2].transform.rotation_degrees == (
        0.0, -15.0, 0.0)
    assert stack.carla_baseline.lidar.channels == 32
    assert stack.carla_baseline.lidar.range_meters == 50.0
    assert stack.carla_baseline.lidar.points_per_second == 320000
    assert stack.carla_baseline.lidar.rotation_frequency_hz == 10.0
    assert stack.carla_baseline.lidar.upper_fov_degrees == 15.0
    assert stack.carla_baseline.lidar.lower_fov_degrees == -25.0
    assert stack.carla_baseline.lidar.canonical_points == stack.bev.num_points
    assert stack.carla_baseline.imu.history_steps == stack.bev.imu_steps
    assert stack.carla_baseline.imu.channels == stack.bev.imu_in_channels
    assert stack.bev.camera_intrinsics is not None
    assert stack.bev.camera_extrinsics is not None
    assert stack.affordance_enabled is False
    assert stack.affordance.in_channels == stack.bev.bev_channels
    assert stack.domain_randomization.tire_friction == (0.4, 1.4)
    validate_stack_configs(
        stack.bev, stack.policy, stack.safety, stack.controller,
        stack.closed_loop, bc_policy=stack.bc_policy)


def test_system_config_applies_runtime_overrides_without_hidden_drift():
    stack = load_system_stack(
        _ROOT_CONFIG, policy_frame="world", occupancy_source="fused")
    assert stack.closed_loop.policy_frame == "world"
    assert stack.closed_loop.occupancy_source == "fused"
    validate_stack_configs(
        stack.bev, stack.policy, stack.safety, stack.controller,
        stack.closed_loop)


def test_system_config_rejects_unknown_and_missing_sections(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    unknown = deepcopy(raw)
    unknown["typo_section"] = {}
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text(yaml.safe_dump(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown system config sections"):
        load_system_stack(unknown_path)

    missing = deepcopy(raw)
    missing.pop("vehicle")
    missing_path = tmp_path / "missing.yaml"
    missing_path.write_text(yaml.safe_dump(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="missing system config sections"):
        load_system_stack(missing_path)


def test_system_config_rejects_unknown_nested_key(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["geometry"]["bev_chanels"] = 64
    path = tmp_path / "bad-key.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry.*bev_chanels"):
        load_system_stack(path)


def test_system_config_rejects_unknown_policy_model_key(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["policy_model"]["hidden_typo"] = 128
    path = tmp_path / "bad-policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown policy_model keys.*hidden_typo"):
        load_system_stack(path)


def test_system_config_rejects_wrong_policy_family_and_ego_schema(tmp_path):
    for field, value in (
            ("family", "hybrid_legacy"),
            ("ego_dynamics_schema", "ego-dynamics-v2")):
        raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
        raw["policy_model"][field] = value
        path = tmp_path / f"bad-policy-{field}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        with pytest.raises(ValueError, match=field):
            load_system_stack(path)


def test_system_config_rejects_missing_file_and_non_mapping_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_system_stack(tmp_path / "absent.yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_system_stack(path)


def test_system_config_rejects_non_mapping_and_missing_nested_key(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["geometry"] = []
    path = tmp_path / "bad-section.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry must be a mapping"):
        load_system_stack(path)

    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["sensors"].pop("image_size")
    path = tmp_path / "missing-key.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing sensors keys.*image_size"):
        load_system_stack(path)


def test_system_config_rejects_non_pair_geometry(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["geometry"]["bev_x_range"] = [-1.0, 0.0, 1.0]
    path = tmp_path / "bad-range.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        load_system_stack(path)


def test_system_config_rejects_unknown_and_missing_sensor_health_keys(
        tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["sensor_health"]["max_sensor_ag_seconds"] = 0.3
    unknown_path = tmp_path / "unknown-health-key.yaml"
    unknown_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(
            ValueError, match="unknown sensor_health keys.*max_sensor_ag_seconds"):
        load_system_stack(unknown_path)

    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["sensor_health"].pop("max_sensor_age_seconds")
    missing_path = tmp_path / "missing-health-key.yaml"
    missing_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(
            ValueError, match="missing sensor_health keys.*max_sensor_age_seconds"):
        load_system_stack(missing_path)


def test_system_config_rejects_invalid_sensor_health_threshold(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["sensor_health"]["max_black_ratio"] = 1.1
    path = tmp_path / "invalid-health-threshold.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="max_black_ratio"):
        load_system_stack(path)


def test_system_config_rejects_unknown_and_missing_carla_baseline_keys(
        tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["carla_baseline"]["vehicle_blueprint_typo"] = "vehicle.test"
    unknown_path = tmp_path / "unknown-carla-key.yaml"
    unknown_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(
            ValueError,
            match="unknown carla_baseline keys.*vehicle_blueprint_typo"):
        load_system_stack(unknown_path)

    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["carla_baseline"].pop("map_name")
    missing_path = tmp_path / "missing-carla-key.yaml"
    missing_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError,
                       match="missing carla_baseline keys.*map_name"):
        load_system_stack(missing_path)


@pytest.mark.parametrize(("field", "value"), [
    ("version", ""),
    ("map_name", ""),
    ("vehicle_blueprint", ""),
    ("random_seed", -1),
    ("repetitions", 2),
])
def test_system_config_rejects_invalid_carla_baseline(field, value, tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["carla_baseline"][field] = value
    path = tmp_path / f"invalid-carla-{field}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_system_stack(path)


def test_system_config_rejects_sensor_tick_and_cross_module_drift(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["carla_baseline"]["cameras"][0]["sensor_tick_seconds"] = 0.05
    path = tmp_path / "camera-tick-drift.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="camera.*sensor_tick_seconds.*control.dt"):
        load_system_stack(path)

    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["carla_baseline"]["lidar"]["canonical_points"] = 128
    path = tmp_path / "lidar-points-drift.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical_points.*sensors.num_points"):
        load_system_stack(path)

    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["carla_baseline"]["imu"]["history_steps"] = 9
    path = tmp_path / "imu-history-drift.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="history_steps.*sensors.imu_steps"):
        load_system_stack(path)


def test_system_config_rejects_duplicate_or_reordered_cameras(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["carla_baseline"]["cameras"][1]["name"] = "front"
    path = tmp_path / "duplicate-camera.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"camera order.*front.*rear.*top"):
        load_system_stack(path)


def test_system_config_rejects_legacy_scalar_sensor_schema(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["carla_baseline"].pop("cameras")
    raw["carla_baseline"]["camera_fov_degrees"] = 90.0
    path = tmp_path / "legacy-carla-schema.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy.*camera_fov_degrees.*cameras"):
        load_system_stack(path)
