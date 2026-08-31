# Stage 1 Sensor Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Stage 1 CPU sensor-health gate by rejecting deterministic Camera/LiDAR/IMU faults before model execution, applying maximum braking, recording reasons and latency, and publishing reproducible evidence.

**Architecture:** A NumPy-only stateful `SensorHealthGate` in `utils` owns raw-packet validation and stable reason enums. The strict system loader derives its configuration from the canonical YAML, while `CarlaClosedLoopRunner` consumes reports before `Observation` and routes invalid packets directly to the existing emergency-control path.

**Tech Stack:** Python 3, dataclasses, enum, NumPy, PyYAML, pytest, pytest-cov, flake8.

**Spec:** `docs/superpowers/specs/2026-08-31-sensor-health-stage1-design.md`

## Global Constraints

- `configs/system.yaml` remains the only runtime configuration source.
- Online Camera/LiDAR mask order is exactly `[camera_valid, lidar_valid]`, and online inference requires `[True, True]`.
- Any invalid Camera, LiDAR, or IMU packet selects `EMERGENCY_STOP`; no normal perception or policy call is allowed.
- Raw LiDAR is checked before any padding/canonicalization.
- CPU/Fake evidence may close only the software Gate; real CARLA, ROS 2/C++, actuator, and target-hardware acceptance remain Stage 1B.
- Preserve all pre-existing uncommitted work. The user requested execution in the current `main` workspace; do not auto-commit or stage the dirty tree.

---

### Task 1: Public sensor-health contract and deterministic checks

**Files:**
- Create: `src/utils/sensor_health.py`
- Create: `tests/unit/utils/test_sensor_health.py`

**Interfaces:**
- Consumes: NumPy raw images/points/IMU plus frame, timestamp, and calibration metadata.
- Produces: `SensorHealthReason`, `SensorHealthConfig`, `SensorHealthReport`, and `SensorHealthGate.evaluate(...)` / `reset()`.

- [x] **Step 1: Add RED tests for config and a valid packet**

```python
def test_valid_packet_returns_strict_online_mask(health_gate, valid_packet):
    report = health_gate.evaluate(**valid_packet)
    assert report.valid
    assert report.reasons == ()
    np.testing.assert_array_equal(report.modality_mask, [True, True])
    assert report.modality_mask.dtype == np.bool_

@pytest.mark.parametrize("field,value", [
    ("max_sensor_age_seconds", -0.1),
    ("max_black_ratio", 1.1),
    ("min_lidar_points", 0),
])
def test_health_config_rejects_invalid_thresholds(field, value):
    values = valid_config_values()
    values[field] = value
    with pytest.raises(ValueError):
        SensorHealthConfig(**values)
```

- [x] **Step 2: Run Task 1 RED test**

Run: `python -m pytest -q tests/unit/utils/test_sensor_health.py`

Expected: collection fails because `utils.sensor_health` does not exist.

- [x] **Step 3: Implement the minimal public dataclasses and structural checks**

Implement the exact types from the spec. `SensorHealthConfig.__post_init__` validates positive dimensions/ranges, ratios in `[0,1]`, `black_pixel_threshold < saturation_pixel_threshold`, and a non-empty calibration version. `SensorHealthReport.valid` requires all three validity flags; `modality_mask` returns a new `(2,)` bool NumPy array.

- [x] **Step 4: Add RED table tests for deterministic fault reasons**

Tests mutate one valid literal packet at a time and assert the corresponding enum and per-modality boolean:

```python
@pytest.mark.parametrize("mutation,reason,modality", [
    ("camera_count", SensorHealthReason.CAMERA_COUNT, "camera_valid"),
    ("camera_shape", SensorHealthReason.CAMERA_SHAPE, "camera_valid"),
    ("camera_dtype", SensorHealthReason.CAMERA_DTYPE, "camera_valid"),
    ("camera_nonfinite", SensorHealthReason.CAMERA_NONFINITE, "camera_valid"),
    ("camera_black", SensorHealthReason.CAMERA_BLACK, "camera_valid"),
    ("camera_saturated", SensorHealthReason.CAMERA_SATURATED, "camera_valid"),
    ("lidar_empty", SensorHealthReason.LIDAR_EMPTY, "lidar_valid"),
    ("lidar_sparse", SensorHealthReason.LIDAR_SPARSE, "lidar_valid"),
    ("lidar_nonfinite", SensorHealthReason.LIDAR_NONFINITE, "lidar_valid"),
    ("lidar_out_of_range", SensorHealthReason.LIDAR_OUT_OF_RANGE, "lidar_valid"),
    ("lidar_duplicates", SensorHealthReason.LIDAR_REPEATED_POINTS, "lidar_valid"),
    ("imu_shape", SensorHealthReason.IMU_SHAPE, "imu_valid"),
    ("imu_nonfinite", SensorHealthReason.IMU_NONFINITE, "imu_valid"),
    ("imu_incomplete", SensorHealthReason.IMU_INCOMPLETE, "imu_valid"),
    ("imu_gap", SensorHealthReason.IMU_FREQUENCY, "imu_valid"),
    ("imu_accel_range", SensorHealthReason.IMU_ACCEL_RANGE, "imu_valid"),
    ("imu_gyro_range", SensorHealthReason.IMU_GYRO_RANGE, "imu_valid"),
    ("imu_jump", SensorHealthReason.IMU_JUMP, "imu_valid"),
]):
    packet = mutate_valid_packet(valid_packet, mutation)
    report = health_gate.evaluate(**packet)
    assert reason in report.reasons
    assert getattr(report, modality) is False
    assert report.valid is False
```

Add independent tests for camera/lidar frame mismatch, missing/non-finite/future/stale timestamps, skew overflow, missing/mismatched calibration, and exception containment.

- [x] **Step 5: Verify fault tests fail because checks are absent**

Run: `python -m pytest -q tests/unit/utils/test_sensor_health.py`

Expected: the initial valid/config tests pass while newly added fault cases fail on missing reasons or wrong validity flags.

- [x] **Step 6: Implement minimal stateless Camera/LiDAR/IMU/time checks**

Use small private functions returning reason lists. Preserve deterministic check order, deduplicate before report construction, and catch malformed objects at the public boundary so `evaluate()` always returns a report.

- [x] **Step 7: Add and verify RED stateful freeze/repeat tests**

```python
def test_advancing_frames_with_identical_camera_content_eventually_freeze(...):
    for frame in (10, 11, 12):
        report = gate.evaluate(**packet_for_frame(frame, same_content=True))
    assert SensorHealthReason.CAMERA_FROZEN in report.reasons

def test_reset_prevents_cross_episode_freeze_state(...):
    evaluate_identical_sequence(gate, start=10, count=2)
    gate.reset()
    report = gate.evaluate(**packet_for_frame(1, same_content=True))
    assert SensorHealthReason.CAMERA_FROZEN not in report.reasons
```

Run the two tests and observe failure before implementing content digests and counters.

- [x] **Step 8: Implement freeze/repeated-frame state and turn Task 1 Green**

Use deterministic array byte digests, increment counters only when frame/timestamp advances, reset invalid modality state, and reject only after the configured number of allowed identical frames.

Run: `python -m pytest -q tests/unit/utils/test_sensor_health.py`

Expected: all Task 1 tests pass.

---

### Task 2: Canonical YAML and strict loader

**Files:**
- Modify: `configs/system.yaml`
- Modify: `src/configuration/system.py`
- Modify: `tests/unit/configuration/test_system_config.py`
- Modify: `configs/README.md`

**Interfaces:**
- Consumes: existing `sensors`, `control`, and new `sensor_health` YAML values.
- Produces: `SystemStackConfig.sensor_health: SensorHealthConfig`.

- [x] **Step 1: Add RED loader tests**

Extend the successful stack test to assert the derived camera count/image size/IMU steps/calibration version and threshold values. Add cases proving missing/unknown `sensor_health` keys and invalid ratios fail fast.

- [x] **Step 2: Run loader tests and observe RED**

Run: `python -m pytest -q tests/unit/configuration/test_system_config.py`

Expected: failure because the stack has no `sensor_health` field and the strict schema does not recognize the section.

- [x] **Step 3: Implement YAML section and loader derivation**

Add `sensors.calibration_version: carla-default-v1` and the exact health thresholds from the spec. Extend `_SECTION_KEYS`, build `SensorHealthConfig`, expose it from `SystemStackConfig`, and rely on its validation for invalid threshold errors.

- [x] **Step 4: Verify configuration Green**

Run:

```bash
python -m pytest -q tests/unit/configuration
python -c 'from configuration.system import load_system_stack; load_system_stack("configs/system.yaml")'
```

Expected: exit code 0 for both commands.

---

### Task 3: CARLA pre-model gate, fail-safe metrics, and raw-data ordering

**Files:**
- Modify: `src/sim/carla_closed_loop.py`
- Modify: `scripts/evaluate_carla_closed_loop.py`
- Modify: `tests/unit/sim/test_carla_runner.py`
- Modify: `tests/unit/sim/test_carla_helpers.py`

**Interfaces:**
- Consumes: `SensorHealthGate`, raw sensor packet and `CarlaSensorStack` metadata.
- Produces: mandatory pre-model health evaluation, emergency braking, reason counts, and health p50/p95 summary.

- [x] **Step 1: Add RED metric and Runner fault-injection tests**

Use real `SensorHealthGate` with Fake sensors. Assert an invalid packet produces exactly one health failure, includes the literal reason string in summary, never calls perception/policy, and applies `throttle == 0.0`, `brake == 1.0`. Parameterize P0 cases for black/saturated/frozen camera, empty/sparse LiDAR, stale/skewed timestamps, calibration mismatch, and incomplete IMU.

- [x] **Step 2: Run RED Runner tests**

Run: `python -m pytest -q tests/unit/sim/test_carla_runner.py tests/unit/sim/test_carla_helpers.py`

Expected: failure because Runner does not accept/use the gate and metrics have no health fields.

- [x] **Step 3: Implement metrics and Runner wiring**

Add `EpisodeMetrics.record_sensor_health(report, latency_seconds)`, reason counts, latency samples, and p50/p95 summary. Require a `health_gate` constructor argument, reset it per episode, evaluate immediately after `tick()`, and centralize emergency-control application in a private Runner method so timeout, invalid health, and model exceptions remain maximum-brake paths.

- [x] **Step 4: Expose calibration metadata and wire the executable**

`CarlaSensorStack` stores the configured calibration version. The evaluation script creates `SensorHealthGate(stack.sensor_health)`, passes it to Runner, and passes calibration version into the sensor stack. Do not canonicalize or pad raw LiDAR before `evaluate()`.

- [x] **Step 5: Verify Runner Green and safety tests**

Run:

```bash
python -m pytest -q tests/unit/sim
python -m pytest -q tests/unit/safety tests/unit/orad_ros2
```

Expected: all tests pass; environment tests are not invoked by this task.

---

### Task 4: Deterministic 1000-case CPU fault matrix and audit artifact

**Files:**
- Create: `scripts/evaluate_sensor_health_gate.py`
- Create: `tests/integration/test_sensor_health_gate.py`
- Runtime output: `runs/sensor_health/<run_id>/metrics.json` (Git ignored)

**Interfaces:**
- Consumes: canonical system config and real `SensorHealthGate`.
- Produces: JSON metrics with total cases, accepted/rejected counts, false accepts, per-reason detections, fail-safe count/rate, and p50/p95 latency.

- [x] **Step 1: Add RED integration test for the evaluation function**

```python
def test_1000_fault_cases_have_zero_false_accepts_and_fit_control_budget():
    metrics = run_fault_matrix(load_system_stack(CONFIG), samples=1000)
    assert metrics["total_faults"] == 1000
    assert metrics["false_accepts"] == 0
    assert metrics["fail_safe_rate"] == 1.0
    assert metrics["latency_p95_ms"] < 100.0
```

- [x] **Step 2: Run integration test and observe RED**

Run: `python -m pytest -q tests/integration/test_sensor_health_gate.py`

Expected: collection fails because the evaluation module/function does not exist.

- [x] **Step 3: Implement deterministic matrix and CLI artifact writer**

Cycle through literal mutations with a fresh gate or reset between cases. Measure only `evaluate()` with `time.perf_counter()`. The CLI accepts `--config`, `--samples`, and required/new `--run-id`, creates only `runs/sensor_health/<run-id>/`, refuses to overwrite an existing metrics file, and returns nonzero when any Gate assertion fails.

- [x] **Step 4: Verify integration Green and generate evidence**

Run:

```bash
python -m pytest -q tests/integration/test_sensor_health_gate.py tests/integration/test_m0_interface_gate.py
python scripts/evaluate_sensor_health_gate.py --samples 1000 --run-id stage1a-20260831
```

Expected: tests pass; JSON reports 1000 faults, zero false accepts, fail-safe rate 1.0, and p95 below 100 ms.

---

### Task 5: Full verification and synchronized documentation

**Files:**
- Modify: `docs/ENGINEERING_EXECUTION_ROADMAP.md`
- Create: `docs/SENSOR_HEALTH_STAGE1A_2026-08-31.md`
- Modify: `docs/SDD_AUDIT_2026-08-28.md`
- Modify: `docs/TDD_AUDIT_2026-08-28.md`
- Modify: `docs/SDD_REMEDIATION_2026-08-28.md`
- Modify: `docs/技术原理与代码架构.md`
- Modify: `docs/README.md`
- Modify: `README.md`
- Modify: `System_overview.md`

**Interfaces:**
- Consumes: fresh command outputs and generated CPU metrics JSON.
- Produces: synchronized Stage 1A status, commands, counts, limitations, reason/action matrix, and Stage 1B backlog.

- [x] **Step 1: Run fresh verification before writing result numbers**

```bash
python -m pytest -ra
python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-branch
python -m flake8 src tests scripts --max-line-length=99
git diff --check
```

Capture passed/failed/skipped counts, branch coverage, lint outcome, and the CPU metrics values. Do not reuse older numbers as the new baseline.

- [x] **Step 2: Update current and historical documents without rewriting history**

Mark Stage 1A as `CPU offline verified` only if its Gate evidence passes. Keep Stage 1B visibly pending. Append dated sections to SDD/TDD/remediation logs, document the health data flow and public contracts, synchronize README/System overview, and add the new Stage 1A report/spec/plan to `docs/README.md`.

- [x] **Step 3: Validate documentation and final diff**

Check every changed/new relative Markdown link exists, then rerun:

```bash
git diff --check
python -m pytest -ra
```

Expected: no whitespace errors and the same full-suite Green result. Report every environment-dependent skip separately; do not call Stage 1B complete.
