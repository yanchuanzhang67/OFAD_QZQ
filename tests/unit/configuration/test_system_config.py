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

    assert stack.bev.num_cameras == 3
    assert stack.bev.image_size == (192, 192)
    assert stack.bev.bev_channels == stack.policy.bev_channels == 32
    assert stack.policy.imagine_horizon == 5
    assert stack.closed_loop.occupancy_source == "lidar"
    assert stack.sensor_health.num_cameras == stack.bev.num_cameras
    assert stack.sensor_health.image_size == stack.bev.image_size
    assert stack.sensor_health.imu_steps == stack.bev.imu_steps
    assert stack.sensor_health.expected_calibration_version == "carla-default-v1"
    assert stack.sensor_health.min_lidar_points == 16
    assert stack.sensor_health.max_sensor_age_seconds == 0.2
    assert stack.affordance_enabled is False
    assert stack.affordance.in_channels == stack.bev.bev_channels
    assert stack.domain_randomization.tire_friction == (0.4, 1.4)
    validate_stack_configs(
        stack.bev, stack.policy, stack.safety, stack.controller,
        stack.closed_loop)


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
