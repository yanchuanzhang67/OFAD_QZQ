import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from configuration.system import load_system_stack
import sim.carla_baseline as baseline_module
from sim.carla_baseline import (
    BaselineMismatchError,
    SensorTimingSample,
    build_baseline_manifest,
    capture_sensor_timing,
    collect_git_metadata,
    compare_repetition_records,
    carla_sensor_attributes,
    sha256_file,
    validate_carla_environment,
    write_json_exclusive,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs" / "system.yaml"


def _trace(frame_offset=0, timestamp_offset=0.0):
    return [
        SensorTimingSample(
            frame=frame_offset + index,
            camera_frames=(frame_offset + index,) * 3,
            lidar_frame=frame_offset + index,
            imu_frames=(frame_offset + index - 1, frame_offset + index),
            camera_timestamps=(timestamp_offset + index * 0.1,) * 3,
            lidar_timestamp=timestamp_offset + index * 0.1,
            imu_timestamps=(timestamp_offset + (index - 1) * 0.1,
                            timestamp_offset + index * 0.1),
        )
        for index in (1, 2, 3)
    ]


def test_environment_validation_rejects_every_baseline_drift():
    baseline = load_system_stack(_CONFIG).carla_baseline

    with pytest.raises(BaselineMismatchError) as exc_info:
        validate_carla_environment(
            baseline,
            client_version="0.9.14",
            server_version="0.9.13",
            map_name="Town01",
            available_blueprints=("vehicle.audi.tt",),
            fixed_delta_seconds=0.05,
            expected_control_period=0.1,
        )

    message = str(exc_info.value)
    assert "client version" in message
    assert "server version" in message
    assert "map" in message
    assert "vehicle blueprint" in message
    assert "fixed delta" in message


def test_environment_validation_accepts_full_carla_map_path():
    baseline = load_system_stack(_CONFIG).carla_baseline

    validate_carla_environment(
        baseline,
        client_version="0.9.16",
        server_version="0.9.16",
        map_name="/Game/Carla/Maps/Town10HD_Opt",
        available_blueprints=("vehicle.lincoln.mkz_2020",),
        fixed_delta_seconds=0.1,
        expected_control_period=0.1,
    )


def test_three_repetitions_compare_relative_sensor_timing_and_config():
    resolved = {
        "version": "0.9.16",
        "map_name": "Town10HD_Opt",
        "vehicle_blueprint": "vehicle.lincoln.mkz_2020",
        "random_seed": 42,
        "control_period_seconds": 0.1,
        "calibration_version": "carla-default-v1",
    }
    records = [
        {"resolved_baseline": resolved, "sensor_timing": _trace(100, 10.0)},
        {"resolved_baseline": resolved, "sensor_timing": _trace(200, 20.0)},
        {"resolved_baseline": resolved, "sensor_timing": _trace(300, 30.0)},
    ]

    result = compare_repetition_records(records, expected_repetitions=3)

    assert result == {
        "reproducible": True,
        "repetitions": 3,
        "config_consistent": True,
        "sensor_timing_consistent": True,
        "mismatches": [],
    }


def test_repetition_comparison_reports_timing_and_config_drift():
    records = [
        {"resolved_baseline": {"random_seed": 7},
         "sensor_timing": _trace(100, 10.0)},
        {"resolved_baseline": {"random_seed": 8},
         "sensor_timing": _trace(200, 20.0)},
        {"resolved_baseline": {"random_seed": 7},
         "sensor_timing": _trace(300, 30.0)[:-1]},
    ]

    result = compare_repetition_records(records, expected_repetitions=3)

    assert result["reproducible"] is False
    assert result["config_consistent"] is False
    assert result["sensor_timing_consistent"] is False
    assert result["mismatches"] == [
        "repetition 2 resolved baseline differs from repetition 1",
        "repetition 3 sensor timing differs from repetition 1",
    ]


def test_manifest_contains_traceable_code_config_and_environment():
    stack = load_system_stack(_CONFIG)
    git = collect_git_metadata(_ROOT)

    manifest = build_baseline_manifest(
        baseline=stack.carla_baseline,
        config_path=_CONFIG,
        calibration_version=(
            stack.sensor_health.expected_calibration_version),
        control_period_seconds=stack.closed_loop.dt,
        git_metadata=git,
        client_version="0.9.16",
        server_version="0.9.16",
        observed_map_name="/Game/Carla/Maps/Town10HD_Opt",
    )

    assert manifest["schema_version"] == 1
    assert manifest["git"]["commit"] == git["commit"]
    assert manifest["config"]["sha256"] == sha256_file(_CONFIG)
    assert manifest["carla"]["client_version"] == "0.9.16"
    assert manifest["carla"]["server_version"] == "0.9.16"
    assert manifest["carla"]["map_name"] == "Town10HD_Opt"
    assert manifest["sensors"]["calibration_version"] == "carla-default-v1"
    assert manifest["control_period_seconds"] == 0.1
    assert manifest["random_seed"] == 42


def test_evidence_writer_refuses_to_overwrite_existing_log(tmp_path):
    path = tmp_path / "manifest.json"
    write_json_exclusive(path, {"run": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"run": 1}

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"run": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"run": 1}


def test_sensor_timing_capture_preserves_each_sensor_frame_and_timestamp():
    sensors = SimpleNamespace(
        last_frame=17,
        last_camera_frames=(17, 17, 17),
        last_lidar_frame=17,
        last_imu_frames=(16, 17),
        last_camera_timestamps=(1.7, 1.7, 1.7),
        last_lidar_timestamp=1.7,
        last_imu_timestamps=(1.6, 1.7),
    )

    sample = capture_sensor_timing(sensors)

    assert sample.to_dict() == {
        "frame": 17,
        "camera_frames": [17, 17, 17],
        "lidar_frame": 17,
        "imu_frames": [16, 17],
        "camera_timestamps": [1.7, 1.7, 1.7],
        "lidar_timestamp": 1.7,
        "imu_timestamps": [1.6, 1.7],
    }


def test_carla_sensor_attributes_are_derived_from_canonical_baseline():
    stack = load_system_stack(_CONFIG)

    attributes = carla_sensor_attributes(
        stack.carla_baseline, stack.bev.image_size)

    assert attributes == {
        "cameras": {
            "front": {
                "image_size_x": "192", "image_size_y": "192",
                "fov": "90.0", "sensor_tick": "0.1",
            },
            "rear": {
                "image_size_x": "192", "image_size_y": "192",
                "fov": "90.0", "sensor_tick": "0.1",
            },
            "top": {
                "image_size_x": "192", "image_size_y": "192",
                "fov": "100.0", "sensor_tick": "0.1",
            },
        },
        "lidar": {
            "channels": "32",
            "range": "50.0",
            "points_per_second": "320000",
            "rotation_frequency": "10.0",
            "upper_fov": "15.0",
            "lower_fov": "-25.0",
            "sensor_tick": "0.1",
        },
        "imu": {"sensor_tick": "0.1"},
    }


def test_camera_calibration_maps_principal_rays_into_new_orad_ego_axes():
    stack = load_system_stack(_CONFIG)
    extrinsic = baseline_module.camera_extrinsic_new_orad

    front = extrinsic(stack.carla_baseline.cameras[0])
    rear = extrinsic(stack.carla_baseline.cameras[1])
    principal_ray = [0.0, 0.0, 1.0, 1.0]

    assert front.shape == rear.shape == (4, 4)
    assert (front @ principal_ray)[0] > 0.0
    assert (rear @ principal_ray)[0] < 0.0


def test_canonical_yaml_hash_ignores_mapping_order_and_whitespace(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("a: 1\nb: [2, 3]\n", encoding="utf-8")
    second.write_text("b:\n  - 2\n  - 3\na: 1\n", encoding="utf-8")

    digest = baseline_module.canonical_yaml_sha256
    assert digest(first) == digest(second)
