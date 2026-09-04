# CARLA 0.9.16 Data-First Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the repository on the recorded-data CARLA 0.9.16 baseline and prove the read-only Camera/LiDAR/IMU→HealthGate→Observation→BEV→Policy→Safety→Control software path with traceable artifacts.

**Architecture:** `configs/system.yaml` owns structured CARLA vehicle and sensor definitions, and `load_system_stack()` derives both simulator attributes and calibration-aware BEV parameters. A new read-only `replay` package validates an episode, gates raw sensors before model execution, and runs either strict checkpoints or an explicitly labelled deterministic random-model smoke; a thin script only composes dependencies and writes exclusive artifacts.

**Tech Stack:** Python 3.9+, dataclasses, PyYAML, NumPy, Pillow, PyTorch, pytest, pytest-cov, flake8.

**Spec:** `docs/superpowers/specs/2026-09-03-carla-0916-data-first-replay-design.md`

## Global Constraints

- Canonical CARLA client/server is exactly `0.9.16`; map is `Town10HD_Opt`; vehicle is `vehicle.lincoln.mkz_2020`; seed is `42`; synchronous fixed delta/control period is `0.1 s`.
- Camera order is exactly `[front, rear, top]`, image shape is `192×192`, FOV values are `[90°, 90°, 100°]`, and top pitch is `-15°`.
- LiDAR is 32 channels, 50 m, 320000 points/s, 10 Hz, vertical FOV `-25°..15°`; canonical points are `(256,4)` with explicit CARLA y-right→New_ORAD y-left conversion.
- IMU history is `(10,6)` ordered `[ax,ay,az,gx,gy,gz]`; existing HealthGate thresholds remain unchanged.
- `configs/system.yaml` is the only runtime configuration source; physical/sensor values are not silently overridden by CLI arguments.
- Existing datasets and experiment artifacts are read-only. Every new report uses a new explicit directory and exclusive file creation.
- A rejected sensor frame never invokes perception or policy and must produce an offline maximum-brake fail-safe record.
- Random models require explicit opt-in and always report `model_performance_valid=false` and `closed_loop_acceptance_valid=false`.
- Historical missing provenance remains missing: do not infer recorded vehicle/config hashes from current source code or current config.
- Preserve `CarlaBaselineConfig`, `BEVFusion`, `HybridPolicy`, `Observation`, serialized dataset schema, and checkpoint key names.
- The shared worktree is already dirty. Do not stage or commit mixed user changes; mark task checkpoints in this plan and defer atomic commits until the user establishes/approves a clean boundary.

## File Structure

| Path | Responsibility |
|---|---|
| `src/sim/carla_baseline.py` | Frozen structured CARLA configs, calibration math, canonical hash and baseline validation |
| `src/configuration/system.py` | Strict YAML schema parsing and full-stack assembly |
| `configs/system.yaml` | Sole Canonical physical/runtime values |
| `scripts/collect_carla_initial.py` | CARLA composition and immutable episode collection using the loaded stack |
| `src/replay/__init__.py` | Stable replay public imports |
| `src/replay/carla_dataset.py` | Episode contract, safe paths, PNG/NPY/JSONL loading, provenance gaps |
| `src/replay/carla_pipeline.py` | Health gating, Observation construction, network/safety/control orchestration and metrics |
| `src/replay/artifacts.py` | Transactional run writer, provenance manifest and summary aggregation |
| `scripts/replay_carla_pipeline.py` | CLI, model loading, deterministic seeds and exclusive artifact directory |
| `tests/unit/configuration/test_system_config.py` | Strict structured-config and cross-module tests |
| `tests/unit/sim/test_carla_baseline.py` | Sensor attributes, calibration axes and baseline provenance tests |
| `tests/sim/test_collect_carla_initial.py` | Collector CLI/config ownership tests |
| `tests/unit/replay/test_carla_dataset.py` | Loader, path, schema, dtype/shape and provenance-gap tests |
| `tests/unit/replay/test_carla_pipeline.py` | Health-first and fail-safe orchestration tests |
| `tests/integration/test_recorded_carla_network_smoke.py` | CPU model data-flow contract using a generated mini episode; optional real-episode test |
| `pyproject.toml` | Declare Pillow for PNG-backed replay |
| `configs/README.md` | Canonical ownership and migration instructions |
| `CARLA_INITIAL_COLLECTOR_README.md` | Config-driven collection and new manifest provenance |
| `src/README.md`, `tests/README.md` | New package/test boundaries |
| `System_overview.md`, `docs/技术原理与代码架构.md` | Replay architecture and safety boundary |
| `docs/CARLA_STAGE1B_TASK1_BASELINE_2026-08-31.md` | Superseding CARLA 0.9.16 Canonical note and remaining real-CARLA Gate |
| `docs/ENGINEERING_EXECUTION_ROADMAP.md`, `docs/SmartSteer_Status.md`, `docs/README.md` | Priorities, current evidence and navigation |

---

### Task 1: Structured CARLA 0.9.16 Canonical Configuration

**Files:**
- Modify: `tests/unit/configuration/test_system_config.py`
- Modify: `tests/unit/sim/test_carla_baseline.py`
- Modify: `src/sim/carla_baseline.py`
- Modify: `src/configuration/system.py`
- Modify: `configs/system.yaml`
- Modify: `configs/README.md`
- Modify: `docs/CARLA_STAGE1B_TASK1_BASELINE_2026-08-31.md`

**Interfaces:**
- Produces: `CarlaTransformConfig`, `CarlaCameraConfig`, `CarlaLidarConfig`, `CarlaImuConfig`, extended `CarlaBaselineConfig`.
- Produces: `canonical_yaml_sha256(path: Path) -> str`.
- Produces: `camera_intrinsic_matrix(camera: CarlaCameraConfig, image_size: tuple[int, int]) -> np.ndarray` and `camera_extrinsic_new_orad(camera: CarlaCameraConfig) -> np.ndarray`.
- Preserves: `load_system_stack(path, *, policy_frame=None, occupancy_source=None) -> SystemStackConfig` and `carla_sensor_attributes(...)`.

- [x] **Step 1: Write failing Canonical-value and structured-sensor tests**

Add assertions equivalent to:

```python
stack = load_system_stack(_ROOT_CONFIG)
baseline = stack.carla_baseline
assert baseline.version == "0.9.16"
assert baseline.vehicle_blueprint == "vehicle.lincoln.mkz_2020"
assert baseline.random_seed == 42
assert [camera.name for camera in baseline.cameras] == ["front", "rear", "top"]
assert [camera.fov_degrees for camera in baseline.cameras] == [90.0, 90.0, 100.0]
assert baseline.lidar.points_per_second == 320000
assert baseline.lidar.rotation_frequency_hz == 10.0
assert baseline.imu.history_steps == stack.bev.imu_steps == 10
assert stack.bev.camera_intrinsics is not None
assert stack.bev.camera_extrinsics is not None
```

Also mutate nested camera names, `sensor_tick_seconds`, LiDAR canonical points and IMU history in copied YAML and assert precise `ValueError` messages. Assert the removed `camera_fov_degrees` schema produces a migration error rather than being ignored.

- [x] **Step 2: Run Red tests**

Run:

```bash
python -m pytest -q tests/unit/configuration/test_system_config.py tests/unit/sim/test_carla_baseline.py
```

Expected: failures show the old 0.9.15 scalar baseline and missing structured dataclasses.

Observed: `9 failed / 19 passed`; failures identified the 0.9.15/Tesla/seed/scalar
schema and missing calibration/hash helpers.

- [x] **Step 3: Implement immutable structured dataclasses and calibration helpers**

Implement exact frozen types:

```python
@dataclass(frozen=True)
class CarlaTransformConfig:
    location_m: tuple[float, float, float]
    rotation_degrees: tuple[float, float, float]  # roll, pitch, yaw

@dataclass(frozen=True)
class CarlaCameraConfig:
    name: str
    blueprint: str
    transform: CarlaTransformConfig
    fov_degrees: float
    sensor_tick_seconds: float

@dataclass(frozen=True)
class CarlaLidarConfig:
    blueprint: str
    transform: CarlaTransformConfig
    channels: int
    range_meters: float
    points_per_second: int
    rotation_frequency_hz: float
    upper_fov_degrees: float
    lower_fov_degrees: float
    sensor_tick_seconds: float
    canonical_points: int
    canonical_y_flip: bool

@dataclass(frozen=True)
class CarlaImuConfig:
    blueprint: str
    transform: CarlaTransformConfig
    sensor_tick_seconds: float
    history_steps: int
    channels: int
```

`camera_extrinsic_new_orad()` must compose CARLA transform matrices with both coordinate conversions:

```python
optical_to_carla = np.array([
    [0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]
], dtype=np.float32)
carla_ego_to_new_orad = np.diag([1, -1, 1, 1]).astype(np.float32)
return carla_ego_to_new_orad @ sensor_to_ego_carla @ optical_to_carla
```

Test that the front principal ray maps toward positive ego x, rear toward negative x, and all returned matrices are finite `(4,4)`.

- [x] **Step 4: Replace strict YAML schema and assemble calibrated BEV config**

Use exact simple transforms:

```yaml
front: location [1.5, 0.0, 1.6], rotation [0.0, 0.0, 0.0]
rear:  location [-1.5, 0.0, 1.6], rotation [0.0, 0.0, 180.0]
top:   location [0.0, 0.0, 1.9], rotation [0.0, -15.0, 0.0]
lidar: location [0.0, 0.0, 1.9], rotation [0.0, 0.0, 0.0]
imu:   location [0.0, 0.0, 0.5], rotation [0.0, 0.0, 0.0]
```

Parse nested mappings with explicit allowed/missing-key sets. Pass the three intrinsic and converted extrinsic matrices into `BEVFusionConfig`; validate camera order, sensor ticks equal `control.dt`, canonical points equal `sensors.num_points`, and IMU dimensions equal the shared sensor contract.

- [x] **Step 5: Make Canonical hashing serialization-stable**

Implement `canonical_yaml_sha256()` by parsing YAML, serializing with sorted keys and compact separators, then hashing UTF-8 bytes. Keep `sha256_file()` for byte-level evidence and include both hashes in new manifests.

- [x] **Step 6: Run Green and integration config tests**

Run:

```bash
python -m pytest -q tests/unit/configuration tests/unit/sim/test_carla_baseline.py
python -m pytest -q tests/integration/test_cpu_pipeline_contract.py
```

Expected: all selected tests pass and the CPU pipeline still preserves public model shapes.

Observed: configuration/baseline `28 passed`; CPU pipeline `1 passed`.

- [x] **Step 7: Update Canonical process documentation**

Record the exact new values, the old schema migration error, commands/results and the still-unrun real-CARLA 3-repetition Gate in `configs/README.md` and the dated Task 1 record. Mark the old 0.9.15 values superseded on 2026-09-03 rather than rewriting historical evidence.

- [x] **Step 8: Record checkpoint**

Mark Task 1 complete in this plan with test output. Do not stage/commit while unrelated dirty changes remain; the intended atomic commit paths are the seven Task 1 files above.

---

### Task 2: Make the CARLA Collector Consume the Canonical Stack

**Files:**
- Modify: `tests/sim/test_collect_carla_initial.py`
- Modify: `scripts/collect_carla_initial.py`
- Modify: `CARLA_INITIAL_COLLECTOR_README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `SystemStackConfig.carla_baseline`, `.bev`, `.closed_loop`, `.sensor_health`.
- Produces: `recording_plan_from_stack(stack: SystemStackConfig) -> dict` for CPU testing and CARLA assembly.
- Produces: new episode manifest fields `vehicle_blueprint`, `config_sha256`, `config_canonical_sha256`, `calibration_sha256`, `calibration_version`.

- [x] **Step 1: Write failing collector configuration-ownership tests**

Assert parser defaults include `--config configs/system.yaml` but no `--seed` or `--fixed-delta`. Assert:

```python
plan = recording_plan_from_stack(load_system_stack(_CONFIG))
assert plan["vehicle_blueprint"] == "vehicle.lincoln.mkz_2020"
assert plan["fixed_delta_seconds"] == 0.1
assert plan["camera"]["top"]["fov"] == 100.0
assert plan["lidar"]["points_per_second"] == 320000
assert plan["imu"]["history_steps"] == 10
```

Test `_spawn_vehicle()` with a fake blueprint library where Lincoln is unavailable and assert it raises instead of falling back to `vehicle.*`.

- [x] **Step 2: Run Red collector tests**

Run `python -m pytest -q tests/sim/test_collect_carla_initial.py`.

Expected: parser still exposes duplicated physical arguments and no stack-derived plan exists.

Observed: `3 failed / 7 passed`; missing `--config`, stack-derived recording plan and exact
blueprint failure caused the expected failures.

- [x] **Step 3: Refactor the collector without changing the dataset payload schema**

Load the stack once in `main()`/`collect()`. Generate all CARLA transforms and sensor attributes from `recording_plan_from_stack()`. Change `_spawn_vehicle(world, rng, blueprint_id)` to find exactly the configured blueprint and fail when missing. Keep CARLA lazy imports and retain existing atomic `.incomplete`→`episodes` transaction behavior.

- [x] **Step 4: Add complete provenance to new episodes**

Create calibration JSON from the same structured baseline used to spawn sensors, write it before frames, compute its byte SHA-256, and add all new manifest provenance fields. Validate client/server/map/fixed delta/blueprint with `validate_carla_environment()` before the first recorded frame. Do not alter the 2026-09-02 episode.

- [x] **Step 5: Declare PNG decode dependency and run Green tests**

Add `Pillow>=9.0` to project dependencies because replay treats PNG as a first-class dataset format. Run:

```bash
python -m pytest -q tests/sim/test_collect_carla_initial.py tests/unit/configuration tests/unit/sim/test_carla_baseline.py
```

Expected: all selected tests pass without importing CARLA.

Observed: collector/configuration/baseline selection `39 passed` without CARLA. A direct-script
`--help` regression test also reproduced and closed the missing `src/` bootstrap defect.

- [x] **Step 6: Update collector operations documentation and checkpoint record**

Document the removed physical CLI overrides, exact Canonical command, new manifest fields, legacy provenance limitations and no-overwrite behavior. Mark Task 2 complete here; defer the atomic commit for these four files because of the dirty shared worktree.

---

### Task 3: Build the Read-Only CARLA Episode Loader

**Files:**
- Create: `src/replay/__init__.py`
- Create: `src/replay/carla_dataset.py`
- Create: `tests/unit/replay/test_carla_dataset.py`
- Modify: `src/README.md`

**Interfaces:**
- Produces: `DatasetContractError(code: str, message: str)`.
- Produces: frozen `RecordedCarlaFrame` with raw/canonical point clouds, ordered images, IMU/timing metadata and `VehicleState`.
- Produces: `CarlaRecordedEpisode.open(path: Path, stack: SystemStackConfig, *, allow_legacy_provenance: bool = False) -> CarlaRecordedEpisode`.
- Produces: `CarlaRecordedEpisode.iter_frames(max_frames: int | None = None) -> Iterator[RecordedCarlaFrame]`.
- Produces: properties `provenance_gaps: tuple[str, ...]`, `calibration_sha256: str`, `episode_sha256: str`.

- [x] **Step 1: Generate a minimal valid episode fixture in the test module**

The fixture must write `_SUCCESS`, manifest/calibration JSON, two `192×192` RGB PNGs per named camera, raw/canonical NPY, and JSONL with strictly increasing frame/timestamps. Use `PIL.Image.fromarray(array).save(path)` and never copy from the real dataset.

- [x] **Step 2: Write failing loader contract tests**

Cover:

```python
episode = CarlaRecordedEpisode.open(path, stack, allow_legacy_provenance=True)
frame = next(episode.iter_frames(max_frames=1))
assert frame.images[0].dtype == np.uint8
assert frame.canonical_point_cloud.shape == (stack.bev.num_points, 4)
assert frame.imu_history.shape == (stack.bev.imu_steps, 6)
assert episode.provenance_gaps == (
    "vehicle_blueprint", "config_sha256", "calibration_sha256"
)
```

Add negative tests for missing `_SUCCESS`, unknown schema, sample-count mismatch, duplicate frame, non-monotonic timestamp, missing PNG/NPY, absolute or `../` paths, non-finite NPY, wrong calibration version/axes/camera order/image size, and strict mode rejecting the legacy gaps.

- [x] **Step 3: Run Red loader tests**

Run `python -m pytest -q tests/unit/replay/test_carla_dataset.py`.

Expected: import fails because `replay.carla_dataset` does not exist.

Observed: `7 failed`, all at the missing `replay` package after correcting one test-only
parameterization collection error.

- [x] **Step 4: Implement safe metadata indexing**

`open()` must parse JSON with explicit errors, require `status=complete`, normalize map basename, verify every Canonical field the manifest actually declares, and distinguish absent historical fields from mismatched fields. `_resolve_member(relative)` must reject absolute paths and require `resolved.is_relative_to(episode_root)` using a Python-3.9-compatible parent check.

- [x] **Step 5: Implement lazy frame decoding and ego conversion**

Decode PNG with Pillow into RGB `uint8 HWC`; load NPY with `allow_pickle=False`; copy arrays before returning. Convert world velocity to body velocity using recorded yaw:

```python
vx_body = cos(yaw) * vx_world + sin(yaw) * vy_world
vy_body = -sin(yaw) * vx_world + cos(yaw) * vy_world
```

Map recorded rotation order `[roll,pitch,yaw]` and retain SI units. Reject every wrong shape/dtype/non-finite value with a stable error code containing the frame id.

- [x] **Step 6: Implement provenance hashing without fabricating history**

Hash current `episode.json`, `calibration.json` and `frames.jsonl` into a deterministic episode digest. A missing recorded hash remains in `provenance_gaps`; the computed current calibration hash is a separate property and does not fill the historical manifest field.

- [x] **Step 7: Run Green loader tests and existing dataset verifier**

Run:

```bash
python -m pytest -q tests/unit/replay/test_carla_dataset.py
python scripts/verify_carla_initial_dataset.py datasets/carla_initial
```

Expected: loader tests pass; existing verifier still reports 200 frames and maximum skew `0.000000 s`.

Observed: loader reached `9 passed` after adding Camera transform drift and canonical-config
semantic hash coverage; real episode decoded read-only with gaps
`vehicle_blueprint/config_sha256/calibration_sha256`; legacy verifier reported 200 frames,
range `1044974..1045173`, maximum skew `0.000000 s`.

- [x] **Step 8: Document package boundary and checkpoint**

Add `replay` to `src/README.md`, explicitly distinguishing recorded-data offline verification from CARLA closed-loop. Mark Task 3 complete here and defer its atomic commit.

---

### Task 4: Enforce HealthGate Before Observation and Models

**Files:**
- Create: `src/replay/carla_pipeline.py`
- Create: `tests/unit/replay/test_carla_pipeline.py`

**Interfaces:**
- Consumes: `RecordedCarlaFrame`, `SensorHealthGate`, existing `Observation`.
- Produces: frozen `PreparedReplayFrame(frame, health, observation, health_latency_ms, startup_context)` where `observation` is `None` on rejection.
- Produces: `prepare_recorded_frame(frame: RecordedCarlaFrame, gate: SensorHealthGate, *, calibration_version: str) -> PreparedReplayFrame`.
- Produces: `maximum_brake_record(prepared: PreparedReplayFrame, max_decel: float) -> dict`.

- [x] **Step 1: Write failing health-first tests**

Use a spy model counter around a higher-level callback and assert the first real-shaped frame with IMU accel above 50 is rejected with both stable HealthGate reasons, `prepared.observation is None`, and no callback invocation. Assert maximum-brake output is exactly:

```python
{"steering": 0.0, "speed": 0.0, "accel": -5.0,
 "safety_mode": "emergency_stop", "model_invoked": False}
```

Add an accepted frame test asserting `Observation.frame == "ego"`, synchronized frames, canonical `(256,4)` point cloud and `max_sensor_skew_seconds == 0.0`.

- [x] **Step 2: Run Red health-first tests**

Run `python -m pytest -q tests/unit/replay/test_carla_pipeline.py`.

Expected: `PreparedReplayFrame` and preparation function are missing.

Observed: `2 failed` because `replay.carla_pipeline` did not exist.

- [x] **Step 3: Implement preparation with raw-before-canonical semantics**

Call `gate.evaluate()` with `frame.raw_point_cloud`, ordered decoded images and recorded IMU/timing metadata. Measure only HealthGate wall latency with `time.perf_counter()`. Construct `Observation` only when `report.valid`; use `frame.canonical_point_cloud` for its model point cloud. Set startup context from `sample_index < imu_steps`, but never remove or rename the underlying HealthGate reasons.

- [x] **Step 4: Run Green and the Stage 1A/1B health suites**

Run:

```bash
python -m pytest -q tests/unit/replay/test_carla_pipeline.py tests/unit/utils/test_sensor_health.py tests/unit/sim/test_carla_sensor_health_stream.py
```

Expected: all tests pass and existing thresholds stay unchanged.

Observed: selected replay/Stage 1A/Stage 1B suites passed. Real 200-frame replay produced
`199 valid / 1 rejected`; frame 1044974 retained `imu_accel_range + imu_jump`, produced no
Observation, and frame 1044975 produced an Observation.

- [x] **Step 5: Record checkpoint**

Record Red/Green output in this plan. Defer the atomic commit for the two Task 4 files.

---

### Task 5: Implement Calibration-Aware Network, Safety and Control Replay

**Files:**
- Modify: `src/replay/carla_pipeline.py`
- Modify: `tests/unit/replay/test_carla_pipeline.py`
- Create: `tests/integration/test_recorded_carla_network_smoke.py`

**Interfaces:**
- Produces: `ReplayModelBundle(perception, policy, model_mode, checkpoint_hashes, model_performance_valid, closed_loop_acceptance_valid)`.
- Produces: `CarlaReplayPipeline(stack, health_gate, models, supervisor, safety_filter, controller)`.
- Produces: `CarlaReplayPipeline.process(frame: RecordedCarlaFrame) -> dict`.
- Uses: existing `policy_trajectory_to_waypoints()`, `occupancy_from_points()`, `SafetySupervisor.evaluate()`, `SafetyFilter.filter()`, `PurePursuitController.compute()`.

- [x] **Step 1: Write failing invalid-frame zero-model-call test**

Inject perception and policy spies whose calls raise. Process an invalid sensor frame and assert neither spy was called, safety mode is emergency stop and acceleration equals `stack.controller.max_decel`.

- [x] **Step 2: Write failing valid-frame tensor boundary test**

With tiny deterministic fake models, assert inputs are:

```python
images.shape == (1, 3, 3, 192, 192)
points.shape == (1, 256, 4)
imu.shape == (1, 10, 6)
modality_mask.tolist() == [[True, True]]
```

Assert image tensors are float32 RGB CHW scaled to `[0,1]`, BEV and trajectory are finite, and the final command is finite and within steering/acceleration limits.

- [x] **Step 3: Write failing random-model integration smoke**

Generate a two-frame mini episode with valid nonconstant images, raw LiDAR points inside 50 m and stable IMU near gravity. Build real `BEVFusion(stack.bev).eval()` and `HybridPolicy(stack.policy).eval()` under fixed seeds. Assert one frame reaches both models, reports BEV `(1,32,50,50)`, policy `(1,20,4)`, and reaches either a bounded normal command or explicit maximum-brake supervisor fail-safe. Never assert driving quality.

- [x] **Step 4: Run Red pipeline tests**

Run:

```bash
python -m pytest -q tests/unit/replay/test_carla_pipeline.py tests/integration/test_recorded_carla_network_smoke.py
```

Expected: pipeline/model bundle interfaces are missing.

Observed: `3 failed / 2 passed`; all new failures were the missing `ReplayModelBundle` and
`CarlaReplayPipeline` interfaces.

- [x] **Step 5: Implement tensor conversion and model execution**

Use contiguous tensors, `torch.no_grad()`, CPU by default, `eval()` models and fixed `[[True, True]]` boolean mask. Pass the `BEVFusionConfig` calibration matrices assembled in Task 1. Record stage shapes, finite flags and `perf_counter()` latency. Convert the raw policy output without clipping or fixing invalid random values.

- [x] **Step 6: Implement supervisor/fail-safe routing**

Create policy trajectory with `policy_trajectory_to_waypoints()` and evaluate in the configured frame/timing boundary. If the supervisor rejects random output, compute the controller command from `supervisor.emergency_trajectory(timestamp)`; if accepted, build LiDAR occupancy, apply `SafetyFilter.filter()`, then compute Pure Pursuit. Any caught model/contract exception becomes maximum brake with the failing stage and exception type recorded.

- [x] **Step 7: Run Green and existing perception/deployment contracts**

Run:

```bash
python -m pytest -q tests/unit/replay tests/integration/test_recorded_carla_network_smoke.py
python -m pytest -q tests/integration/test_cpu_pipeline_contract.py tests/unit/deployment/test_onnx_export.py
```

Expected: all selected CPU tests pass and checkpoint parameter keys are unchanged.

Observed: replay selection `12 passed`; existing CPU pipeline/ONNX selection `3 passed`
with pre-existing exporter warnings. A two-frame real dataset probe produced frame 1044974
health reject/model zero-call and frame 1044975 BEV `[1,32,50,50]`, policy `[1,20,4]`,
finite safety/control output.

- [x] **Step 8: Record checkpoint**

Record observed tensor shapes and fail-safe behavior in this plan. Defer the atomic commit for Task 5 files.

---

### Task 6: Add Strict Model Loading, CLI and Traceable Artifacts

**Files:**
- Create: `scripts/replay_carla_pipeline.py`
- Create: `src/replay/artifacts.py`
- Create: `tests/unit/replay/test_replay_artifacts.py`
- Modify: `src/replay/carla_pipeline.py`
- Modify: `tests/integration/test_recorded_carla_network_smoke.py`

**Interfaces:**
- Produces: `load_model_bundle(stack, *, perception_checkpoint: Path | None, policy_checkpoint: Path | None, allow_random_models: bool, seed: int) -> ReplayModelBundle`.
- Produces: `ReplayRunWriter.create(output_root: Path, run_id: str) -> ReplayRunWriter`, `.write_frame(record)`, `.complete(summary)`.
- Produces CLI arguments: `--config`, `--episode`, `--output-root`, `--max-frames`, `--allow-legacy-provenance`, `--allow-random-models`, `--perception-checkpoint`, `--policy-checkpoint`, `--seed`.

- [x] **Step 1: Write failing model-mode tests**

Assert no checkpoints and no random opt-in raises a clear startup error. Assert only one checkpoint raises rather than mixing trained/random models. Assert corrupt, missing-key and unexpected-key checkpoints fail strict loading without fallback. Assert random mode returns deterministic model parameters for the same seed and metadata:

```python
assert bundle.model_mode == "random-model-network-smoke"
assert bundle.model_performance_valid is False
assert bundle.closed_loop_acceptance_valid is False
```

- [x] **Step 2: Write failing exclusive-artifact tests**

Assert writer creation rejects an existing run directory; incomplete runs never contain `_SUCCESS`; `complete()` writes `run.json`, `frames.jsonl`, `summary.json`, then `_SUCCESS`. Assert `run.json` preserves `None` and `provenance_gaps` for absent historical vehicle/config/calibration hashes.

- [x] **Step 3: Run Red artifact tests**

Run `python -m pytest -q tests/unit/replay/test_replay_artifacts.py`.

Expected: model loader and writer do not exist.

Observed: `5 failed`; three missing strict model-loader cases and two missing artifact module
cases. A subsequent direct-CLI RED failed because the script did not yet exist.

- [x] **Step 4: Implement strict model loading**

Seed Python, NumPy and PyTorch before construction. Require both checkpoints or explicit random mode. Load state dictionaries with `strict=True`, verify all tensor values are finite, hash checkpoint bytes, and never catch a loading error to downgrade into random mode.

- [x] **Step 5: Implement transactional run artifacts and summary aggregation**

Write into `<output-root>/.incomplete/<run-id>` using exclusive creation. On success, write `_SUCCESS` last and atomically rename to `<output-root>/<run-id>`. Aggregate total/valid/rejected/model-processed/fail-safe counts, reasons, provenance gaps, shape/finite pass rates and p50/p95/max HealthGate/model/end-to-end latency. Label observed rejection rate; do not call it false rejection without truth labels.

- [x] **Step 6: Implement the thin CLI**

The script loads stack→episode→gate→models→pipeline, iterates `max_frames`, writes each returned record and completes the summary. Exit nonzero on config/dataset/checkpoint/artifact failures and print the new artifact path on success. It must not import or connect to CARLA.

- [x] **Step 7: Run Green artifact/CLI tests**

Run:

```bash
python -m pytest -q tests/unit/replay tests/integration/test_recorded_carla_network_smoke.py
python scripts/replay_carla_pipeline.py --help
```

Expected: all tests pass; help exposes only the specified replay/model/provenance choices.

Observed: artifact/model tests `6 passed`; complete replay selection `18 passed`; CLI help
exposed only the planned config/dataset/model/provenance options.

- [x] **Step 8: Run the actual 200-frame dataset in bounded and full modes**

First run one model-eligible frame while retaining the invalid startup frame:

```bash
python scripts/replay_carla_pipeline.py \
  --config configs/system.yaml \
  --episode datasets/carla_initial/episodes/episode_20260902T110145Z \
  --output-root artifacts/carla_replay \
  --max-frames 2 \
  --allow-legacy-provenance \
  --allow-random-models \
  --seed 42
```

Expected: frame 1044974 is HealthGate rejected/model not invoked; frame 1044975 reaches BEV and policy; both produce a finite bounded or maximum-brake command record. Then run all 200 frames with a new generated run id and record summary counts/latencies. If CPU runtime is impractically long, preserve the successful bounded artifact and report the measured blocker without calling full replay complete.

Observed bounded artifact: `artifacts/carla_replay/replay_20260903T035505Z_dfcbc8bf`.
Observed final full artifact: `artifacts/carla_replay/replay_20260903T035626Z_9ca5f754`.
The full run completed 200 records: 199 valid/model-processed, 1 HealthGate reject/fail-safe,
observed rejection rate `0.005`, network-boundary finite rate `1.0`, maximum-brake frames
`101`, and safety modes `199 normal / 1 emergency_stop`. Latency p50/p95 was health
`6.05/7.11 ms`, model `10.15/14.74 ms`, end-to-end `17.68/23.52 ms`. The artifact retains
all three historical provenance gaps and both performance/closed-loop acceptance flags false.

- [x] **Step 9: Record checkpoint**

Add artifact paths, hashes, exact commands and metrics to this plan. Defer atomic commit of the four Task 6 files.

---

### Task 7: Synchronize Architecture/Status Documents and Verify the Repository

**Files:**
- Modify: `System_overview.md`
- Modify: `docs/技术原理与代码架构.md`
- Modify: `docs/ENGINEERING_EXECUTION_ROADMAP.md`
- Modify: `docs/SmartSteer_Status.md`
- Modify: `docs/README.md`
- Modify: `tests/README.md`
- Modify: this plan

**Interfaces:**
- Consumes: fresh commands and artifacts from Tasks 1–6.
- Produces: evidence-backed current status using only repository-approved maturity labels.

- [x] **Step 1: Update owning architecture and test documents**

Document the structured calibration-aware config, `replay` package dependency direction, health-before-model state machine, random/checkpoint mode boundary, artifact schema and legacy provenance gaps. Add new unit/integration test locations to `tests/README.md`.

- [x] **Step 2: Update roadmap and current status from fresh evidence**

Replace the current Canonical-drift P0 item with the actual implemented/tested state. Preserve these unresolved Gates unless new evidence exists: three real CARLA repetitions, 1000+ normal ticks, expert/hazard/occupancy labels, trained checkpoint, model CARLA closed loop, ORT/TensorRT and ROS2/SIL/HIL. Use “离线验证” only if the reproducible recorded replay artifact exists; random model performance remains invalid.

Observed: the SDD, architecture guide, roadmap, Task 2 evidence, current status, test guide and
document index now describe the structured 0.9.16 baseline, health-first state machine, artifact
schema, three historical provenance gaps and the random-model acceptance boundary. Online Task 1
and Task 2 Gates remain explicitly open.

- [x] **Step 3: Run focused static checks**

Run:

```bash
python -m flake8 src tests scripts --max-line-length=99
git diff --check
```

Expected: both exit zero.

Observed: the first flake8 run found two pre-existing trailing blank lines in the dataset
verifier and its test. After removing only those blank lines, repository-wide flake8 and
`git diff --check` both exited zero.

- [x] **Step 4: Run required branch coverage**

Run:

```bash
python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-branch
```

Expected: repository total remains at least 79%; changed P0 fail-safe branches have complete focused coverage where feasible, and any uncovered branch is listed explicitly.

Observed: `275 passed / 2 skipped / 16 warnings`; total branch coverage `87.93%`, above the
79% gate. `sensor_health.py` remained 98%; new replay modules were dataset 79%, pipeline 83%
and artifacts 90%. Remaining replay misses are primarily malformed-file/I/O variants and
secondary exception reporting branches; health rejection zero-model-call and maximum-brake
P0 behavior have focused tests.

- [x] **Step 5: Run the full regression suite**

Run:

```bash
python -m pytest -ra
```

Expected: no failures; report exact passed/skipped/warning counts from this run and list every environment-dependent skip separately.

Observed: `292 passed / 8 skipped / 16 warnings` in 13.33 s. Skips require CARLA 20-degree
slope, Gazebo suspension, live CARLA 0.9.16 Task 1, live CARLA 0.9.16 Task 2, expert log,
hazard-boundary dataset, CUDA, and the full ROS 2 stack respectively. None are counted as
completed environment evidence.

- [x] **Step 6: Check changed Markdown links**

Extract each changed relative Markdown target, strip optional anchors, and verify the resolved repository path exists. Record the command and result; do not rely only on visual inspection.

Observed: the read-only checker resolved 46 relative links across all changed architecture,
status, operations, design and plan documents; missing count was zero.

- [x] **Step 7: Final self-audit against the design**

Confirm all ten design sections have an implementation or an explicitly retained external Gate. Confirm no existing dataset/artifact file changed, no random result is labelled performance, no historical provenance was fabricated, and `git status --short` separates pre-existing/user changes from this plan’s files.

Observed: all ten design sections map to implemented configuration/collector/replay/artifact
work or an explicitly retained external Gate. The current episode and calibration digests match
the final artifact; config byte/canonical hashes also match. Three historical gaps remain null,
and both random-model acceptance flags remain false. The collector/replay did not overwrite raw
data; every produced report used a new run directory. The normal `main` checkout remains dirty as
approved, with no staging, commit, merge or push performed.

- [x] **Step 8: Record completion checkpoint**

Fill in all task checkboxes/results and provide the user with clickable code, artifact and status-document paths. Do not create a mixed commit; propose exact atomic commit groups only after the user requests repository integration.

Observed: Tasks 1–7 and all Red/Green/verification evidence are recorded in this plan. The
software/data-flow scope is complete; live CARLA Task 1/2, labelled M0 data, trained checkpoints
and deployment Gates remain open and are listed in the current status document.
