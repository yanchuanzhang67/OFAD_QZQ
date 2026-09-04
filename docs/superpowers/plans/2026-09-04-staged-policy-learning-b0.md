# B0 Pure BC Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, trainable and auditable `pure_bc_v1` policy baseline that consumes validated BEV, IMU history and ego dynamics without traversing Affordance, RSSM or Dreamer components.

**Architecture:** Keep the existing `HybridPolicy` intact for checkpoint compatibility and add a separate `BCPolicy`. A frozen, strict-loaded `BEVFusion` produces the shared BEV; the BC policy encodes BEV, IMU and a versioned eight-field ego-dynamics vector, then decodes a fixed ego-frame `(B,20,4)` trajectory. Expert-data loading, masked loss, metrics, checkpoints and CLIs are separate focused modules under a new `training` package.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, PyYAML, Pillow, pytest, pytest-cov, flake8

**Spec:** `docs/superpowers/specs/2026-09-04-staged-policy-learning-b0-design.md`

## Global Constraints

- `configs/system.yaml` remains the only runtime configuration source; unknown, missing and cross-module-inconsistent values fail fast.
- The active B0 checkpoint family is exactly `pure_bc_v1`; legacy `HybridPolicy` imports and state-dict keys remain unchanged.
- Policy input contracts are BEV `(B,32,50,50)`, IMU `(B,10,6)` and `ego-dynamics-v1` `(B,8)`.
- Policy output remains ego-frame `(B,20,4)` ordered as `[x_m,y_m,heading_rad,velocity_mps]`, with waypoint interval equal to `control.dt=0.1 s`.
- Camera and LiDAR must both be valid before BEV inference; missing, malformed or non-finite input is never replaced with an undocumented zero tensor.
- Absolute world `x/y/yaw` never enter the policy ego-dynamics vector.
- Train-only normalization is stored in checkpoint buffers and metadata; validation, test and runtime never recompute it.
- Training requires `expert_label=true` and a strict perception checkpoint; random perception and Traffic Manager smoke data cannot create performance-valid checkpoints.
- Raw datasets, previous artifacts, trained weights and exported engines are never overwritten; every run uses a new explicit directory.
- B0 can reach unit verification without external expert data, but reaches offline verification only with the spec's 1000+ expert-sample, three-seed and frozen-test evidence.
- After every task, update this plan's checkboxes and append the exact Red/Green command outcome before committing.

## Execution Log

- 2026-09-04 pre-change baseline: `python -m pytest -ra` →
  `292 passed / 8 skipped / 16 warnings` in `17.60 s`. The skips are the existing
  CARLA, Gazebo, expert-log, hazard-boundary, CUDA and full ROS 2 environment gates;
  no baseline failure was present.
- 2026-09-04 Task 1 Red: focused config/contract tests stopped at collection with
  the expected `ImportError` for missing `EGO_DYNAMICS_V1_FIELDS`.
- 2026-09-04 Task 1 Green: focused config/contract tests passed (`26 passed`);
  targeted flake8 exited 0.
- 2026-09-04 Task 2 Red: `tests/unit/policy/test_bc_policy.py` stopped at
  collection with the expected `ModuleNotFoundError: policy.bc_policy`.
- 2026-09-04 Task 2 Green: Pure BC policy tests passed (`13 passed`), legacy
  HybridPolicy regression passed (`8 passed`), and targeted flake8 exited 0.
- 2026-09-04 Task 3 Red: replay/expert dataset tests stopped at collection with
  the expected `ModuleNotFoundError: training`.
- 2026-09-04 Task 3 Green: replay/expert dataset tests passed (`17 passed`),
  including explicit non-expert rejection and health rejection evidence;
  targeted flake8 exited 0.
- 2026-09-04 Task 4 Red: metric tests stopped at collection with the expected
  `ModuleNotFoundError: training.bc_metrics`.
- 2026-09-04 Task 4 Green: exact-value open-loop metric tests passed
  (`6 passed`) and targeted flake8 exited 0.
- 2026-09-04 Task 5 Red: artifact tests stopped at collection with the expected
  `ModuleNotFoundError: training.bc_artifacts`.
- 2026-09-04 Task 5 Green: transactional run/checkpoint tests passed
  (`6 passed`) and targeted flake8 exited 0. The non-finite perception fixture
  was corrected to select a floating-point state tensor before injecting NaN.
- 2026-09-04 Task 6 Red: engine tests stopped at collection with the expected
  `ModuleNotFoundError: training.bc_engine`. A supplemental calibration-lineage
  assertion then failed with the expected missing `calibration_sha256` dataset
  attribute.
- 2026-09-04 Task 6 Green: engine plus expert-dataset tests passed
  (`11 passed`); `python scripts/train_bc.py --help` listed the complete B0
  training interface, and targeted flake8 exited 0.
- 2026-09-04 Task 7 Red: evaluation integration tests stopped at collection
  with the expected missing `evaluate_loader` import; the evaluation CLI did
  not yet exist.
- 2026-09-04 Task 7 Green: the CPU fixture train→checkpoint→frozen-test flow
  passed (`2 passed`), including config mismatch rejection and immutable model
  state. The complete training/evaluation focus set passed (`25 passed`), both
  CLI help contracts rendered, and targeted flake8 exited 0.

---

## File Structure

### New files

- `src/policy/config.py` — canonical `BCPolicyConfig` with no Hybrid/RSSM dependency.
- `src/policy/bc_policy.py` — deterministic three-branch BC encoder, trajectory decoder and masked BC objective.
- `src/training/__init__.py` — stable public exports for B0 data, metrics, checkpoint and engine APIs.
- `src/training/bc_dataset.py` — strict expert split manifest, `BCSample`/`BCBatch`, loader and collate logic.
- `src/training/bc_metrics.py` — mask-aware open-loop metrics and grouped accumulation.
- `src/training/bc_artifacts.py` — non-overwriting run writer and versioned strict checkpoint I/O.
- `src/training/bc_engine.py` — frozen-perception train/evaluate loops and deterministic seeding.
- `scripts/train_bc.py` — B0 training CLI.
- `scripts/evaluate_bc.py` — frozen-test evaluation CLI.
- `tests/unit/policy/test_bc_policy.py` — Pure BC architecture, validation, determinism, loss and gradient tests.
- `tests/unit/training/test_bc_dataset.py` — expert manifest, split and sample contract tests.
- `tests/unit/training/test_bc_metrics.py` — exact ADE/FDE/angle/speed/validity metric tests.
- `tests/unit/training/test_bc_artifacts.py` — transaction and checkpoint-lineage tests.
- `tests/unit/training/test_bc_engine.py` — frozen perception and finite gradient/update tests.
- `tests/integration/test_bc_cli.py` — one-step fixture train/evaluate CLI contract.

### Modified files

- `configs/system.yaml` — add the strict `policy_model` B0 section.
- `src/configuration/system.py` — validate/assemble `BCPolicyConfig` alongside the legacy Hybrid config.
- `src/utils/types.py` — add versioned `EgoDynamicsV1` conversion contract.
- `src/utils/contracts.py` — verify BC/perception/control dimensions and waypoint interval.
- `src/policy/__init__.py` — export `BCPolicy` and `BCPolicyConfig` without changing legacy exports.
- `src/replay/carla_dataset.py` — decode optional expert trajectory fields while preserving non-expert replay.
- `src/replay/carla_pipeline.py` — strict-load and invoke `pure_bc_v1`; retain explicit legacy random smoke.
- `scripts/replay_carla_pipeline.py` — record policy family/checkpoint lineage in replay metadata.
- `tests/unit/configuration/test_system_config.py` — strict policy section and consistency tests.
- `tests/unit/replay/test_carla_dataset.py` — additive expert trajectory decode and rejection coverage.
- `tests/unit/replay/test_replay_artifacts.py` — checkpoint-family routing and no-fallback coverage.
- `tests/integration/test_cpu_pipeline_contract.py` — Pure BC sensor-to-control shape/frame contract.
- `README.md`, `System_overview.md`, `docs/技术原理与代码架构.md` — current B0 architecture and commands.
- `docs/ENGINEERING_EXECUTION_ROADMAP.md`, `docs/SmartSteer_Status.md`, `docs/README.md` — evidence, maturity and navigation.
- This plan — real-time Red/Green evidence and completion state.

---

### Task 1: Canonical ego-dynamics and B0 configuration contracts

**Files:**
- Create: `src/policy/config.py`
- Modify: `src/utils/types.py:15-58`
- Modify: `src/policy/__init__.py:1-27`
- Modify: `configs/system.yaml:104-115`
- Modify: `src/configuration/system.py:20-75,82-105,243-345`
- Modify: `src/utils/contracts.py:8-40`
- Test: `tests/unit/utils/test_contracts.py`
- Test: `tests/unit/configuration/test_system_config.py`
- Track: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: `utils.types.VehicleState`, BEV dimensions from `BEVFusionConfig`, IMU dimensions from `sensors`, and `control.dt`.
- Produces: `EGO_DYNAMICS_V1_FIELDS`, `EgoDynamicsV1.from_vehicle_state(state)`, `EgoDynamicsV1.to_array()`, `BCPolicyConfig`, and `SystemStackConfig.bc_policy`.

- [x] **Step 1: Write failing ego-dynamics contract tests**

```python
def test_ego_dynamics_v1_excludes_world_pose_and_preserves_order():
    state = VehicleState(
        x=100.0, y=-20.0, yaw=1.5, speed=3.0, steering=0.1,
        pitch=0.2, roll=-0.3, vx=2.5, vy=-0.4,
        yaw_rate=0.05, accel_z=9.7)
    dynamics = EgoDynamicsV1.from_vehicle_state(state)
    assert EGO_DYNAMICS_V1_FIELDS == (
        "speed", "steering", "pitch", "roll",
        "vx", "vy", "yaw_rate", "accel_z")
    np.testing.assert_array_equal(
        dynamics.to_array(),
        np.array([3.0, 0.1, 0.2, -0.3, 2.5, -0.4, 0.05, 9.7], np.float32))


def test_ego_dynamics_v1_rejects_nonfinite_vehicle_state():
    with pytest.raises(ValueError, match="ego-dynamics-v1.*finite"):
        EgoDynamicsV1.from_vehicle_state(VehicleState(speed=float("nan")))
```

- [x] **Step 2: Write failing strict configuration tests**

```python
def test_system_yaml_builds_pure_bc_config():
    stack = load_system_stack(_ROOT_CONFIG)
    assert stack.bc_policy.family == "pure_bc_v1"
    assert stack.bc_policy.bev_channels == stack.bev.bev_channels == 32
    assert stack.bc_policy.bev_h == stack.bc_policy.bev_w == 50
    assert stack.bc_policy.imu_steps == 10
    assert stack.bc_policy.ego_dim == 8
    assert stack.bc_policy.horizon == 20
    assert stack.bc_policy.traj_dim == 4
    assert stack.bc_policy.waypoint_dt == stack.closed_loop.dt == 0.1


def test_system_config_rejects_unknown_policy_model_key(tmp_path):
    raw = yaml.safe_load(_ROOT_CONFIG.read_text(encoding="utf-8"))
    raw["policy_model"]["hidden_typo"] = 128
    path = tmp_path / "bad-policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown policy_model keys.*hidden_typo"):
        load_system_stack(path)
```

- [x] **Step 3: Run focused tests and record the expected Red result**

Run:

```bash
python -m pytest -q \
  tests/unit/utils/test_contracts.py \
  tests/unit/configuration/test_system_config.py
```

Expected: collection/import failure because `EgoDynamicsV1`, `BCPolicyConfig`,
`policy_model` and `SystemStackConfig.bc_policy` do not exist.

- [x] **Step 4: Implement the minimal shared contracts**

Add the framework-independent state conversion to `utils/types.py`:

```python
EGO_DYNAMICS_V1_FIELDS = (
    "speed", "steering", "pitch", "roll",
    "vx", "vy", "yaw_rate", "accel_z",
)


@dataclass(frozen=True)
class EgoDynamicsV1:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != 8 or not np.isfinite(self.values).all():
            raise ValueError("ego-dynamics-v1 values must be finite with length 8")

    @classmethod
    def from_vehicle_state(cls, state: VehicleState) -> "EgoDynamicsV1":
        values = tuple(float(getattr(state, name))
                       for name in EGO_DYNAMICS_V1_FIELDS)
        if len(values) != 8 or not np.isfinite(values).all():
            raise ValueError("ego-dynamics-v1 values must be finite")
        return cls(values)

    def to_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32).copy()
```

Create the B0 config without importing the Hybrid implementation:

```python
@dataclass(frozen=True)
class BCPolicyConfig:
    family: str = "pure_bc_v1"
    bev_channels: int = 32
    bev_h: int = 50
    bev_w: int = 50
    imu_in_channels: int = 6
    imu_steps: int = 10
    ego_dim: int = 8
    encoder_hidden: int = 256
    ego_hidden: int = 64
    latent_dim: int = 256
    horizon: int = 20
    traj_dim: int = 4
    waypoint_dt: float = 0.1
    bc_xy_weight: float = 1.0
    bc_heading_weight: float = 0.2
    bc_speed_weight: float = 0.2
    bc_smooth_weight: float = 0.05
```

Add this exact YAML section:

```yaml
policy_model:
  family: pure_bc_v1
  encoder_hidden: 256
  ego_hidden: 64
  latent_dim: 256
  horizon: 20
  trajectory_dim: 4
  ego_dynamics_schema: ego-dynamics-v1
```

Extend `_SECTION_KEYS`, derive BEV/IMU/waypoint dimensions rather than duplicating
them, reject a family other than `pure_bc_v1`, and make
`validate_stack_configs(..., bc_policy=None)` backward-compatible for existing callers.

- [x] **Step 5: Run focused tests and record the Green result**

Run:

```bash
python -m pytest -q \
  tests/unit/utils/test_contracts.py \
  tests/unit/configuration/test_system_config.py
```

Expected: PASS with no CPU test skipped.

- [x] **Step 6: Run static checks for the changed Python files**

Run:

```bash
python -m flake8 \
  src/utils/types.py src/utils/contracts.py src/policy/config.py \
  src/configuration/system.py tests/unit/utils/test_contracts.py \
  tests/unit/configuration/test_system_config.py --max-line-length=99
```

Expected: exit code 0.

- [x] **Step 7: Update plan evidence and commit Task 1**

```bash
git add configs/system.yaml src/utils/types.py src/utils/contracts.py \
  src/policy/config.py src/policy/__init__.py src/configuration/system.py \
  tests/unit/utils/test_contracts.py \
  tests/unit/configuration/test_system_config.py \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "feat: define pure BC runtime contracts"
```

---

### Task 2: Deterministic Pure BC network and masked objective

**Files:**
- Create: `src/policy/bc_policy.py`
- Create: `tests/unit/policy/test_bc_policy.py`
- Modify: `src/policy/__init__.py:1-35`
- Track: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: `BCPolicyConfig` from Task 1 and tensors `bev`, `imu`, `ego`.
- Produces: `BCPolicy.forward(bev, imu, ego) -> Tensor[B,N,4]`,
  `BCPolicy.set_ego_normalization(mean, std)`,
  `masked_bc_loss_components(prediction, expert, valid_mask, config)` and
  `BCPolicy.bc_loss_components(bev, imu, ego, expert, valid_mask)`.

- [x] **Step 1: Write failing architecture and determinism tests**

```python
def test_pure_bc_forward_is_deterministic_and_has_no_world_model():
    cfg = _cfg()
    policy = BCPolicy(cfg).eval()
    bev, imu, ego = _inputs(cfg, batch=2)
    with torch.no_grad():
        first = policy(bev, imu, ego)
        second = policy(bev, imu, ego)
    assert first.shape == (2, cfg.horizon, 4)
    assert torch.equal(first, second)
    forbidden = ("rssm", "prior", "posterior", "critic", "reward_head")
    assert not any(any(token in name for token in forbidden)
                   for name, _ in policy.named_parameters())


@pytest.mark.parametrize("input_name", ["bev", "imu", "ego"])
def test_pure_bc_rejects_nonfinite_inputs(input_name):
    cfg = _cfg()
    policy = BCPolicy(cfg)
    bev, imu, ego = _inputs(cfg, batch=1)
    values = {"bev": bev, "imu": imu, "ego": ego}
    values[input_name].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match=f"{input_name}.*finite"):
        policy(values["bev"], values["imu"], values["ego"])
```

- [x] **Step 2: Write failing normalization and loss tests**

```python
def test_ego_normalization_is_a_strict_checkpoint_buffer():
    policy = BCPolicy(_cfg())
    policy.set_ego_normalization(torch.arange(8.0), torch.ones(8))
    state = policy.state_dict()
    assert torch.equal(state["ego_mean"], torch.arange(8.0))
    assert torch.equal(state["ego_std"], torch.ones(8))
    with pytest.raises(ValueError, match="std.*positive"):
        policy.set_ego_normalization(torch.zeros(8), torch.zeros(8))


def test_masked_loss_wraps_heading_and_ignores_padding():
    cfg = _cfg(horizon=3)
    pred = torch.tensor([[[0., 0., 3.13, 1.],
                          [1., 0., 0., 2.],
                          [99., 99., 0., 99.]]])
    target = torch.tensor([[[0., 0., -3.13, 1.],
                            [1., 0., 0., 2.],
                            [0., 0., 0., 0.]]])
    mask = torch.tensor([[True, True, False]])
    losses = masked_bc_loss_components(pred, target, mask, cfg)
    assert losses["heading"] < 0.001
    assert losses["xy"] == 0
    assert losses["speed"] == 0
    assert losses["valid_waypoints"] == 2
```

- [x] **Step 3: Run the new test file and record the Red result**

Run:

```bash
python -m pytest -q tests/unit/policy/test_bc_policy.py
```

Expected: import failure because `policy.bc_policy` does not exist.

- [x] **Step 4: Implement deterministic encoders and decoder**

Define the private MLP helper and deterministic decoder in the same file:

```python
def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, out_dim), nn.ELU())


class TrajectoryDecoder(nn.Module):
    def __init__(self, config: BCPolicyConfig):
        super().__init__()
        self.horizon = config.horizon
        self.projection = _mlp(
            config.latent_dim, config.latent_dim, config.latent_dim)
        self.gru = nn.GRU(
            config.latent_dim, config.latent_dim, batch_first=True)
        self.head = nn.Linear(config.latent_dim, config.traj_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        repeated = self.projection(latent).unsqueeze(1).expand(
            -1, self.horizon, -1)
        decoded, _ = self.gru(repeated)
        return self.head(decoded)
```

Then implement only the three input branches defined by the spec:

```python
class BCPolicy(nn.Module):
    def __init__(self, config: BCPolicyConfig):
        super().__init__()
        self.config = config
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(config.bev_channels, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, config.encoder_hidden), nn.ELU())
        self.imu_encoder = nn.GRU(
            config.imu_in_channels, config.encoder_hidden, batch_first=True)
        self.ego_encoder = _mlp(config.ego_dim, config.ego_hidden,
                                config.ego_hidden)
        self.fusion = _mlp(
            2 * config.encoder_hidden + config.ego_hidden,
            config.latent_dim, config.latent_dim)
        self.decoder = TrajectoryDecoder(config)
        self.register_buffer("ego_mean", torch.zeros(config.ego_dim))
        self.register_buffer("ego_std", torch.ones(config.ego_dim))

    def forward(self, bev, imu, ego):
        self._validate_inputs(bev, imu, ego)
        bev_feature = self.bev_encoder(bev)
        _, imu_hidden = self.imu_encoder(imu)
        ego_feature = self.ego_encoder((ego - self.ego_mean) / self.ego_std)
        latent = self.fusion(torch.cat(
            [bev_feature, imu_hidden.squeeze(0), ego_feature], dim=-1))
        trajectory = self.decoder(latent)
        if not torch.isfinite(trajectory).all():
            raise ValueError("BC trajectory must be finite")
        return trajectory
```

Validate exact rank, shape, floating dtype and finite values. Implement wrapped
heading residual with `atan2(sin(delta), cos(delta))`; calculate smooth loss only
where three consecutive mask values are true; reject a sample with no valid
waypoint. Return scalar tensors `xy`, `heading`, `speed`, `smooth`, `total` and
an integer `valid_waypoints`.

- [x] **Step 5: Run the policy tests and record the Green result**

Run:

```bash
python -m pytest -q tests/unit/policy/test_bc_policy.py
```

Expected: PASS; all policy parameters receiving the supervised path have finite,
non-zero gradients in the gradient test.

- [x] **Step 6: Verify the legacy HybridPolicy regression**

Run:

```bash
python -m pytest -q tests/unit/policy/test_hybrid_policy.py
```

Expected: existing HybridPolicy tests PASS unchanged.

- [x] **Step 7: Update plan evidence and commit Task 2**

```bash
git add src/policy/bc_policy.py src/policy/__init__.py \
  tests/unit/policy/test_bc_policy.py \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "feat: add deterministic pure BC policy"
```

---

### Task 3: Strict expert dataset and episode-level split loader

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/bc_dataset.py`
- Create: `tests/unit/training/test_bc_dataset.py`
- Modify: `src/replay/carla_dataset.py:22-58,330-390`
- Modify: `tests/unit/replay/test_carla_dataset.py`
- Track: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: `CarlaRecordedEpisode`, `RecordedCarlaFrame`, `EgoDynamicsV1`,
  `SystemStackConfig` and a `new-orad-bc-manifest-v1` JSON manifest.
- Produces: additive `RecordedCarlaFrame.expert_trajectory`,
  `RecordedCarlaFrame.trajectory_mask`, `RecordedCarlaFrame.expert_source`,
  `BCSample`, `BCBatch`, `ExpertBCDataset.open(manifest_path, split, stack)` and
  `collate_bc_samples(samples)`.

- [x] **Step 1: Write failing additive replay-schema tests**

```python
def test_recorded_episode_decodes_versioned_expert_trajectory(valid_episode):
    _add_expert_labels(valid_episode, horizon=20, source="human-teleop-v1")
    frame = next(CarlaRecordedEpisode.open(valid_episode, _stack()).iter_frames())
    assert frame.expert_label is True
    assert frame.expert_source == "human-teleop-v1"
    assert frame.expert_trajectory.shape == (20, 4)
    assert frame.trajectory_mask.dtype == np.bool_


def test_non_expert_episode_remains_readable_for_replay(valid_episode):
    frame = next(CarlaRecordedEpisode.open(valid_episode, _stack()).iter_frames())
    assert frame.expert_label is False
    assert frame.expert_trajectory is None
    assert frame.trajectory_mask is None
```

- [x] **Step 2: Write failing manifest, split and rejection tests**

```python
def test_expert_dataset_rejects_current_traffic_manager_episode():
    with pytest.raises(DatasetContractError, match="expert_label_required"):
        ExpertBCDataset.open(_CURRENT_MANIFEST, "train", _stack())


def test_manifest_rejects_episode_leakage_across_splits(tmp_path):
    manifest = _manifest(train=["episode-a"], validation=["episode-a"], test=[])
    path = _write_json(tmp_path / "split.json", manifest)
    with pytest.raises(DatasetContractError, match="split_episode_overlap"):
        ExpertBCDataset.open(path, "train", _stack())


def test_collate_produces_canonical_b0_batch(expert_manifest):
    dataset = ExpertBCDataset.open(expert_manifest, "train", _stack())
    batch = collate_bc_samples([dataset[0], dataset[1]])
    assert batch.images.shape == (2, 3, 3, 192, 192)
    assert batch.lidar.shape == (2, 256, 4)
    assert batch.imu.shape == (2, 10, 6)
    assert batch.ego.shape == (2, 8)
    assert batch.expert.shape == (2, 20, 4)
    assert batch.mask.shape == (2, 20)
```

- [x] **Step 3: Run focused tests and record the Red result**

Run:

```bash
python -m pytest -q \
  tests/unit/replay/test_carla_dataset.py \
  tests/unit/training/test_bc_dataset.py
```

Expected: import/attribute failures for the missing expert fields and training package.

- [x] **Step 4: Implement additive expert decoding**

Append optional fields with defaults so existing frame construction remains compatible:

```python
@dataclass(frozen=True)
class RecordedCarlaFrame:
    # existing fields remain in their existing order
    expert_source: str = "unknown"
    expert_trajectory: Optional[np.ndarray] = None
    trajectory_mask: Optional[np.ndarray] = None
```

When `expert_label` is true, require `expert_source` to be non-empty, parse
`expert_trajectory` as finite float32 `(20,4)`, parse `trajectory_mask` as bool
`(20,)`, and require at least one valid waypoint. When false, leave both tensors
as `None`; do not synthesize labels from `action_raw` or `action_applied`.

- [x] **Step 5: Implement the strict split manifest and dataset**

The manifest contract is:

```json
{
  "schema_version": "new-orad-bc-manifest-v1",
  "dataset_version": "expert-offroad-v1",
  "calibration_version": "carla-default-v1",
  "fixture": false,
  "splits": {
    "train": [{"episode": "episodes/episode-a", "sha256": "<64 hex>"}],
    "validation": [{"episode": "episodes/episode-b", "sha256": "<64 hex>"}],
    "test": [{"episode": "episodes/episode-c", "sha256": "<64 hex>"}]
  }
}
```

Resolve every episode relative to the manifest directory, reject absolute/path-escape
paths, reject duplicate resolved episodes across splits, verify episode hash and
calibration version, then expose only expert frames. Convert images to channel-first
float32 `[0,1]`, use the validated canonical LiDAR `(256,4)`, and use
`EgoDynamicsV1.from_vehicle_state(frame.ego_state)`.

Build the sample index by walking every episode in timestamp order with a fresh
`SensorHealthGate`. Exclude health-invalid frames without calling perception, retain
their frame IDs/reason codes in `dataset.rejections`, and include the rejection count in
every training/evaluation manifest. Contract corruption such as missing files, bad
shape/frame, non-finite values or absent expert fields fails the entire dataset open;
health rejection is never converted to a zero-filled sample.

Use these batch fields so the engine and metric tasks have one stable interface:

```python
@dataclass(frozen=True)
class BCBatch:
    images: torch.Tensor       # (B,3,3,H,W), float32 [0,1]
    lidar: torch.Tensor        # (B,256,4), float32
    imu: torch.Tensor          # (B,10,6), float32
    ego: torch.Tensor          # (B,8), float32
    expert: torch.Tensor       # (B,20,4), float32
    mask: torch.Tensor         # (B,20), bool
    sample_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    tags: tuple[dict[str, str], ...]

    @property
    def size(self) -> int:
        return int(self.images.shape[0])
```

- [x] **Step 6: Run focused tests and record the Green result**

Run:

```bash
python -m pytest -q \
  tests/unit/replay/test_carla_dataset.py \
  tests/unit/training/test_bc_dataset.py
```

Expected: PASS; the repository's current non-expert episode is explicitly rejected
by `ExpertBCDataset` while remaining readable by normal replay.

- [x] **Step 7: Update plan evidence and commit Task 3**

```bash
git add src/replay/carla_dataset.py src/training/__init__.py \
  src/training/bc_dataset.py tests/unit/replay/test_carla_dataset.py \
  tests/unit/training/test_bc_dataset.py \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "feat: add strict expert BC dataset loader"
```

---

### Task 4: Mask-aware open-loop policy metrics

**Files:**
- Create: `src/training/bc_metrics.py`
- Create: `tests/unit/training/test_bc_metrics.py`
- Modify: `src/training/__init__.py`
- Track: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: predicted/target tensors `(B,N,4)`, bool mask `(B,N)`, episode IDs,
  optional categorical tags and configured `max_speed`.
- Produces: `BCMetricAccumulator.update(...)`, `BCMetricAccumulator.compute()` and
  `wrapped_angle_error(prediction, target)`.

- [x] **Step 1: Write failing exact-value metric tests**

```python
def test_metrics_use_last_valid_waypoint_for_fde_and_ignore_padding():
    prediction = torch.tensor([[[0., 0., 0., 1.],
                                [2., 0., 0., 2.],
                                [99., 99., 0., 99.]]])
    target = torch.tensor([[[0., 0., 0., 1.],
                            [1., 0., 0., 1.],
                            [0., 0., 0., 0.]]])
    mask = torch.tensor([[True, True, False]])
    metrics = BCMetricAccumulator(max_speed=12.0)
    metrics.update(prediction, target, mask, episode_ids=["episode-a"])
    result = metrics.compute()
    assert result["ade_m"] == pytest.approx(0.5)
    assert result["fde_m"] == pytest.approx(1.0)
    assert result["velocity_mae_mps"] == pytest.approx(0.5)


def test_heading_metric_wraps_at_pi():
    error = wrapped_angle_error(torch.tensor([3.13]), torch.tensor([-3.13]))
    assert error.item() < 0.03
```

- [x] **Step 2: Write failing validity and grouped-result tests**

```python
def test_metrics_reject_nonfinite_and_empty_masks():
    accumulator = BCMetricAccumulator(max_speed=12.0)
    with pytest.raises(ValueError, match="valid waypoint"):
        accumulator.update(_pred(), _target(), torch.zeros(1, 20, dtype=torch.bool),
                           episode_ids=["episode-a"])


def test_metrics_report_per_episode_groups():
    accumulator = BCMetricAccumulator(max_speed=12.0)
    accumulator.update(_pred(batch=2), _target(batch=2), _mask(batch=2),
                       episode_ids=["episode-a", "episode-b"])
    result = accumulator.compute()
    assert set(result["by_episode"]) == {"episode-a", "episode-b"}
```

- [x] **Step 3: Run the new metric tests and record the Red result**

Run:

```bash
python -m pytest -q tests/unit/training/test_bc_metrics.py
```

Expected: import failure because `training.bc_metrics` does not exist.

- [x] **Step 4: Implement weighted accumulation**

Maintain sums and denominators rather than averaging batch means. ADE uses every valid
XY distance; FDE uses each sample's last valid index; heading uses wrapped absolute
error; speed uses absolute error. Smoothness is the mean norm of XY second differences
where three consecutive waypoints are valid. A negative predicted velocity contributes
to `negative_speed_rate`; a negative change in predicted ego-frame x between two
consecutive valid waypoints contributes to `reverse_waypoint_rate`; velocity above
configured `max_speed` contributes to `overspeed_rate`. Count non-finite trajectories
before rejecting them, and report those rates, `nonfinite_trajectory_count`,
`sample_count`, plus per-episode and supplied tag groups.

```python
def wrapped_angle_error(prediction, target):
    delta = prediction - target
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


class BCMetricAccumulator:
    def __init__(self, max_speed: float): ...
    def update(self, prediction, target, mask, *, episode_ids, tags=None): ...
    def compute(self) -> dict[str, object]: ...
```

- [x] **Step 5: Run metric tests and record the Green result**

Run:

```bash
python -m pytest -q tests/unit/training/test_bc_metrics.py
```

Expected: PASS with exact ADE/FDE tests independent of batch partitioning.

- [x] **Step 6: Update plan evidence and commit Task 4**

```bash
git add src/training/__init__.py src/training/bc_metrics.py \
  tests/unit/training/test_bc_metrics.py \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "feat: add BC open-loop metrics"
```

---

### Task 5: Transactional run artifacts and strict B0 checkpoints

**Files:**
- Create: `src/training/bc_artifacts.py`
- Create: `tests/unit/training/test_bc_artifacts.py`
- Modify: `src/training/__init__.py`
- Track: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: `BCPolicy`, `BCPolicyConfig`, optimizer state, resolved config, metrics
  and code/config/data/calibration/perception provenance.
- Produces: `BCRunWriter.create(root, run_id, metadata)`,
  `save_bc_checkpoint(path, policy, optimizer, metadata)`, and
  `load_bc_checkpoint(path, expected_config) -> tuple[BCPolicy, dict]`,
  `load_frozen_perception_checkpoint(path, perception) -> str`, and
  `resolve_checkpoint_index(path) -> Path`.

- [x] **Step 1: Write failing transactional artifact tests**

```python
def test_bc_run_writer_is_exclusive_and_publishes_success_last(tmp_path):
    writer = BCRunWriter.create(tmp_path, "run-001", {"seed": 41})
    assert writer.working_path.parent.name == ".incomplete"
    assert not (writer.working_path / "_SUCCESS").exists()
    final = writer.complete({"ade_m": 0.2})
    assert (final / "_SUCCESS").read_text() == "complete\n"
    with pytest.raises(FileExistsError):
        BCRunWriter.create(tmp_path, "run-001", {"seed": 41})
```

- [x] **Step 2: Write failing checkpoint family and lineage tests**

```python
def test_checkpoint_roundtrip_requires_pure_bc_family(tmp_path):
    policy = BCPolicy(_cfg())
    path = tmp_path / "policy.pt"
    save_bc_checkpoint(path, policy, None, _lineage())
    loaded, metadata = load_bc_checkpoint(path, _cfg())
    assert loaded.state_dict().keys() == policy.state_dict().keys()
    assert metadata["model_family"] == "pure_bc_v1"
    payload = torch.load(path, weights_only=True)
    payload["model_family"] = "hybrid_legacy"
    torch.save(payload, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="model_family"):
        load_bc_checkpoint(tmp_path / "wrong.pt", _cfg())
```

- [x] **Step 3: Run artifact tests and record the Red result**

Run:

```bash
python -m pytest -q tests/unit/training/test_bc_artifacts.py
```

Expected: import failure because `training.bc_artifacts` does not exist.

- [x] **Step 4: Implement non-overwriting writer and strict checkpoint schema**

Use schema `new-orad-policy-checkpoint-v1` and require these keys:

```python
REQUIRED_LINEAGE = {
    "git", "config_sha256", "dataset_sha256", "split_sha256",
    "calibration_sha256", "perception_checkpoint_sha256", "seed",
    "resolved_command", "epoch", "best_metric",
}

payload = {
    "schema_version": "new-orad-policy-checkpoint-v1",
    "model_family": "pure_bc_v1",
    "config": asdict(policy.config),
    "state_dict": policy.state_dict(),
    "optimizer_state": optimizer.state_dict() if optimizer else None,
    "lineage": metadata,
}
```

Write every `epoch-NNNN.pt` checkpoint with an exclusive temporary file followed by an
atomic rename. Reject an existing destination, missing lineage, non-finite tensors,
config mismatch, missing or unexpected state keys, and normalization buffers with
invalid shape/std. `best.json` and `last.json` contain only filename, SHA-256 and epoch;
update these small indexes atomically while preserving every weight file.

`load_frozen_perception_checkpoint` accepts a tensor state dict (or a payload containing
`state_dict`), checks every tensor is finite, calls `perception.load_state_dict(...,
strict=True)`, freezes/evaluates the model and returns the file SHA-256. Never call
`strict=False` and never construct `HybridPolicy` in this module.

- [x] **Step 5: Run artifact tests and record the Green result**

Run:

```bash
python -m pytest -q tests/unit/training/test_bc_artifacts.py
```

Expected: PASS; wrong family/config/state key and overwrite attempts fail explicitly.

- [x] **Step 6: Update plan evidence and commit Task 5**

```bash
git add src/training/__init__.py src/training/bc_artifacts.py \
  tests/unit/training/test_bc_artifacts.py \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "feat: add auditable BC checkpoints"
```

---

### Task 6: Frozen-perception BC training engine and CLI

**Files:**
- Create: `src/training/bc_engine.py`
- Create: `tests/unit/training/test_bc_engine.py`
- Create: `scripts/train_bc.py`
- Modify: `src/training/__init__.py`
- Modify: `tests/README.md`
- Track: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: `BCBatch`, strict-loaded `BEVFusion`, `BCPolicy`, Adam optimizer,
  dataloaders and `BCRunWriter`.
- Produces: `set_deterministic_seed(seed)`, `fit_ego_normalization(dataset)`,
  `train_one_epoch(perception, policy, loader, optimizer, device)`,
  `validate_one_epoch(...)`, and `scripts/train_bc.py`.

- [x] **Step 1: Write failing frozen-perception and update tests**

```python
def test_train_step_freezes_perception_and_updates_bc_policy():
    perception = _tiny_perception()
    policy = BCPolicy(_tiny_policy_config())
    before_perception = _clone_state(perception)
    before_policy = _clone_state(policy)
    result = train_one_epoch(
        perception, policy, [_batch()],
        torch.optim.Adam(policy.parameters(), lr=1e-3), torch.device("cpu"))
    assert result["steps"] == 1
    assert _state_equal(perception, before_perception)
    assert not _state_equal(policy, before_policy)
    assert all(parameter.grad is None for parameter in perception.parameters())
```

- [x] **Step 2: Write failing finite-gradient and train-only normalization tests**

```python
def test_train_epoch_fails_on_nonfinite_gradient(monkeypatch):
    policy = BCPolicy(_tiny_policy_config())
    monkeypatch.setattr(policy, "bc_loss_components", _nan_loss)
    with pytest.raises(FloatingPointError, match="gradient"):
        train_one_epoch(_tiny_perception(), policy, [_batch()],
                        torch.optim.Adam(policy.parameters()),
                        torch.device("cpu"))


def test_normalization_is_fit_only_from_train_dataset():
    mean, std = fit_ego_normalization(_ego_dataset([[1.] * 8, [3.] * 8]))
    assert torch.equal(mean, torch.full((8,), 2.0))
    assert torch.equal(std, torch.full((8,), 1.0))
```

Compute population standard deviation (`unbiased=False`) in float64, then store finite
float32 mean/std buffers. Reject a field whose std is below `1e-6`; do not replace it
with one because doing so would hide a degenerate training distribution.

- [x] **Step 3: Run engine tests and record the Red result**

Run:

```bash
python -m pytest -q tests/unit/training/test_bc_engine.py
```

Expected: import failure because `training.bc_engine` does not exist.

- [x] **Step 4: Implement the finite deterministic training engine**

The train step must follow this exact ownership:

```python
perception.eval()
perception.requires_grad_(False)
policy.train()
with torch.no_grad():
    feature = perception(
        batch.images, batch.lidar, batch.imu,
        modality_mask=torch.ones(batch.size, 2, dtype=torch.bool, device=device))
losses = policy.bc_loss_components(
    feature.bev, batch.imu, batch.ego, batch.expert, batch.mask)
optimizer.zero_grad(set_to_none=True)
losses["total"].backward()
if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
           for parameter in policy.parameters()):
    raise FloatingPointError("BC gradient must be finite")
optimizer.step()
```

Aggregate every loss component by valid waypoint count. Validation uses `eval` and
`no_grad`. `set_deterministic_seed` sets Python, NumPy and Torch seeds and deterministic
Torch algorithms where supported. The run manifest records when a backend cannot offer
bitwise determinism.

- [x] **Step 5: Implement the training CLI**

`scripts/train_bc.py` accepts exactly:

```text
--config PATH
--data-manifest PATH
--perception-checkpoint PATH
--run-dir PATH
--seed INT
--epochs INT
--batch-size INT
--learning-rate FLOAT
--weight-decay FLOAT
--num-workers INT
--device cpu|cuda
--smoke-test
```

Require seed to be one of `41/42/43` unless `--smoke-test` is present. Smoke mode also
requires manifest `fixture=true`, marks `model_performance_valid=false`, and never writes
a performance-valid `best.json`. Normal mode requires at least 1000 total expert samples,
non-empty train/validation/test splits and a clean, strict perception checkpoint. Create
  the run directory transactionally, save a new `epoch-NNNN.pt` every epoch, update
  `last.json` each epoch and update `best.json` when validation ADE improves. Never
  overwrite an epoch weight file or an existing completed run.

- [x] **Step 6: Run engine and CLI-help tests and record Green**

Run:

```bash
python -m pytest -q tests/unit/training/test_bc_engine.py
python scripts/train_bc.py --help
```

Expected: unit tests PASS and help lists every declared argument.

- [x] **Step 7: Update plan evidence and commit Task 6**

```bash
git add src/training/__init__.py src/training/bc_engine.py \
  tests/unit/training/test_bc_engine.py scripts/train_bc.py tests/README.md \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "feat: add pure BC training workflow"
```

---

### Task 7: Frozen-test evaluation CLI and one-step integration fixture

**Files:**
- Create: `scripts/evaluate_bc.py`
- Create: `tests/integration/test_bc_cli.py`
- Modify: `src/training/bc_engine.py`
- Modify: `tests/fixtures/README.md`
- Track: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: strict perception/policy checkpoints, `ExpertBCDataset` frozen test split,
  `BCMetricAccumulator` and a new output directory.
- Produces: `evaluate_loader(...) -> dict`, an evaluation `metrics.json`, failure-sample
  JSONL and `scripts/evaluate_bc.py`.

- [x] **Step 1: Write a failing evaluation integration test**

```python
def test_fixture_train_then_evaluate_is_explicitly_non_performance(tmp_path):
    fixture = build_bc_fixture(tmp_path, sample_count=4, fixture=True)
    perception_checkpoint = save_tiny_perception(tmp_path)
    train = _run("scripts/train_bc.py", fixture, perception_checkpoint,
                 tmp_path / "train", "--smoke-test")
    assert train.returncode == 0, train.stderr
    last_index = next((tmp_path / "train").glob("*/checkpoints/last.json"))
    checkpoint = resolve_checkpoint_index(last_index)
    evaluate = _run_eval(fixture, perception_checkpoint, checkpoint,
                         tmp_path / "eval", "--smoke-test")
    assert evaluate.returncode == 0, evaluate.stderr
    metrics = json.loads(next((tmp_path / "eval").glob("*/metrics.json")).read_text())
    assert metrics["model_family"] == "pure_bc_v1"
    assert metrics["model_performance_valid"] is False
    assert metrics["sample_count"] > 0
```

- [x] **Step 2: Write failing checkpoint/config mismatch and immutable-test tests**

```python
def test_evaluate_rejects_config_hash_mismatch(tmp_path):
    result = _run_eval_with_modified_system_yaml(tmp_path)
    assert result.returncode != 0
    assert "config" in result.stderr and "mismatch" in result.stderr


def test_evaluate_never_updates_policy_or_normalization(expert_fixture):
    policy = _loaded_policy(expert_fixture)
    before = _clone_state(policy)
    evaluate_loader(_perception(), policy, _loader(expert_fixture),
                    torch.device("cpu"), max_speed=12.0)
    assert _state_equal(policy, before)
```

- [x] **Step 3: Run the integration test and record the Red result**

Run:

```bash
python -m pytest -q tests/integration/test_bc_cli.py
```

Expected: FAIL because `scripts/evaluate_bc.py` and `evaluate_loader` do not exist.

- [x] **Step 4: Implement immutable evaluation**

Add `evaluate_loader` using `perception.eval()`, `policy.eval()` and `torch.no_grad()`.
Write global/per-episode/tag metrics, checkpoint/data/config hashes, threshold results
and failed sample identifiers. Normal evaluation accepts only the manifest's `test`
split and `fixture=false`; smoke evaluation accepts only `fixture=true` and permanently
marks its result non-performance.

The CLI arguments are:

```text
--config PATH
--data-manifest PATH
--perception-checkpoint PATH
--policy-checkpoint PATH
--run-dir PATH
--batch-size INT
--num-workers INT
--device cpu|cuda
--smoke-test
```

Exit non-zero on checkpoint/config/data/calibration mismatch, non-finite output, zero
samples, incomplete metrics or write failure. Do not modify checkpoint or normalization
state during evaluation.

- [x] **Step 5: Run the integration test and record the Green result**

Run:

```bash
python -m pytest -q tests/integration/test_bc_cli.py
```

Expected: PASS; fixture artifacts exist but state `model_performance_valid=false`.

- [x] **Step 6: Update plan evidence and commit Task 7**

```bash
git add scripts/evaluate_bc.py src/training/bc_engine.py \
  tests/integration/test_bc_cli.py tests/fixtures/README.md \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "feat: add pure BC evaluation workflow"
```

---

### Task 8: Pure BC recorded-replay integration without legacy regression

**Files:**
- Modify: `src/replay/carla_pipeline.py:30-105,270-350`
- Modify: `scripts/replay_carla_pipeline.py:70-125`
- Modify: `tests/unit/replay/test_replay_artifacts.py`
- Modify: `tests/unit/replay/test_carla_pipeline.py`
- Modify: `tests/integration/test_cpu_pipeline_contract.py`
- Track: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: `pure_bc_v1` checkpoint loader, `EgoDynamicsV1`, existing health-first
  replay and existing `policy_trajectory_to_waypoints`.
- Produces: `ReplayModelBundle.policy_family`, family-aware policy invocation and replay
  metadata that records the exact family and checkpoint lineage.

- [ ] **Step 1: Write failing strict-routing tests**

```python
def test_replay_loads_pure_bc_checkpoint_and_routes_ego_dynamics(tmp_path):
    stack = load_system_stack(_CONFIG)
    perception_path = _save_perception_checkpoint(tmp_path, stack)
    policy_path = _save_pure_bc_checkpoint(tmp_path, stack)
    bundle = load_model_bundle(
        stack, perception_checkpoint=perception_path,
        policy_checkpoint=policy_path, allow_random_models=False, seed=42)
    assert bundle.policy_family == "pure_bc_v1"
    assert isinstance(bundle.policy, BCPolicy)


def test_replay_never_loads_hybrid_state_into_pure_bc(tmp_path):
    with pytest.raises(ValueError, match="model_family"):
        _load_mislabeled_hybrid_checkpoint(tmp_path)
```

- [ ] **Step 2: Write a failing CPU sensor-to-control Pure BC test**

Extend the CPU pipeline test to call:

```python
ego = torch.from_numpy(
    EgoDynamicsV1.from_vehicle_state(state).to_array()).unsqueeze(0)
raw = policy(feature.bev, imu, ego)
assert raw.shape == (1, cfg.horizon, 4)
assert torch.isfinite(raw).all()
```

Also inject a policy exception and non-finite output into replay, asserting the existing
emergency command and maximum braking behavior remain unchanged.

- [ ] **Step 3: Run replay and CPU integration tests and record the Red result**

Run:

```bash
python -m pytest -q \
  tests/unit/replay/test_replay_artifacts.py \
  tests/unit/replay/test_carla_pipeline.py \
  tests/integration/test_cpu_pipeline_contract.py
```

Expected: failures because model bundles lack `policy_family` and replay invokes every
policy with only `(bev, imu)`.

- [ ] **Step 4: Implement family-aware strict loading and invocation**

Keep random smoke behavior explicit and unchanged:

```python
if perception_checkpoint is None:
    policy = HybridPolicy(stack.policy).eval()
    policy_family = "hybrid_legacy_random_smoke"
else:
    policy, policy_metadata = load_bc_checkpoint(
        policy_checkpoint, stack.bc_policy)
    policy.eval()
    policy_family = "pure_bc_v1"
```

Before Pure BC invocation, convert `frame.ego_state` with `EgoDynamicsV1` and move the
tensor to the BEV device/dtype. Invoke `policy(bev, imu, ego)`. Any conversion, family,
checkpoint, model output or trajectory-contract error follows the existing maximum-brake
path. Record `policy_family`, policy schema and checkpoint lineage in run/summary JSON.

- [ ] **Step 5: Run replay and CPU integration tests and record Green**

Run:

```bash
python -m pytest -q \
  tests/unit/replay/test_replay_artifacts.py \
  tests/unit/replay/test_carla_pipeline.py \
  tests/integration/test_cpu_pipeline_contract.py
```

Expected: PASS; legacy random smoke remains explicitly non-performance and Pure BC uses
three inputs with strict checkpoint provenance.

- [ ] **Step 6: Run the broader model-interface regression**

Run:

```bash
python -m pytest -q \
  tests/unit/policy \
  tests/unit/deployment/test_onnx_export.py \
  tests/integration/test_recorded_carla_network_smoke.py
```

Expected: PASS or only pre-existing environment skips; Hybrid ONNX coverage remains
unchanged because Pure BC deployment is outside this spec.

- [ ] **Step 7: Update plan evidence and commit Task 8**

```bash
git add src/replay/carla_pipeline.py scripts/replay_carla_pipeline.py \
  tests/unit/replay/test_replay_artifacts.py \
  tests/unit/replay/test_carla_pipeline.py \
  tests/integration/test_cpu_pipeline_contract.py \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "feat: route pure BC through recorded replay"
```

---

### Task 9: Documentation synchronization and full verification

**Files:**
- Modify: `README.md`
- Modify: `System_overview.md`
- Modify: `docs/技术原理与代码架构.md`
- Modify: `docs/ENGINEERING_EXECUTION_ROADMAP.md`
- Modify: `docs/SmartSteer_Status.md`
- Modify: `docs/README.md`
- Modify: `docs/superpowers/specs/2026-09-04-staged-policy-learning-b0-design.md`
- Modify: `docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md`

**Interfaces:**
- Consumes: actual Task 1–8 commands, test counts, skips, coverage, code paths and any
  environment/data blockers.
- Produces: one synchronized status stating exactly what is unit verified, what remains
  code skeleton, and why B0 is or is not offline verified.

- [ ] **Step 1: Update architecture and user commands from implemented code**

Document these exact boundaries:

```text
B0 implemented path:
HealthGate → Observation → frozen BEVFusion → BCPolicy(BEV, IMU, EgoDynamicsV1)
→ ego trajectory [B,20,4] → Safety → Control

Not B0 evidence:
Traffic Manager labels, random perception, smoke fixture metrics, Hybrid RSSM output,
CARLA closed-loop behavior, ONNX/ORT/TensorRT, ROS 2 or vehicle behavior
```

Add the real `train_bc.py` and `evaluate_bc.py` help/usage commands. If no formal expert
dataset exists, state B0 maturity as **unit verified** and leave the Stage 3 offline Gate
open. Do not copy test counts from an older report.

- [ ] **Step 2: Run all focused B0 tests**

Run:

```bash
python -m pytest -q \
  tests/unit/utils/test_contracts.py \
  tests/unit/configuration/test_system_config.py \
  tests/unit/policy/test_bc_policy.py \
  tests/unit/training \
  tests/unit/replay/test_carla_dataset.py \
  tests/unit/replay/test_carla_pipeline.py \
  tests/unit/replay/test_replay_artifacts.py \
  tests/integration/test_bc_cli.py \
  tests/integration/test_cpu_pipeline_contract.py
```

Expected: PASS; any skip must name a genuinely missing environment/data dependency.

- [ ] **Step 3: Run unit branch coverage**

Run:

```bash
python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-branch
```

Expected: repository branch coverage at least 79%; changed B0 critical code reaches 95%
where feasible and every finite/family/schema rejection branch has focused coverage.

- [ ] **Step 4: Run static checks**

Run:

```bash
python -m flake8 src tests scripts --max-line-length=99
```

Expected: exit code 0.

- [ ] **Step 5: Run the full regression suite**

Run:

```bash
python -m pytest -ra
```

Expected: all CPU tests PASS. Report CARLA/Gazebo/CUDA/ROS 2/expert-data skips separately;
do not count them as B0 offline or simulation acceptance.

- [ ] **Step 6: Check changed Markdown links and diff formatting**

Run:

```bash
test -f docs/superpowers/specs/2026-09-04-staged-policy-learning-b0-design.md
test -f docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 7: Request code review and address only verified findings**

Review against the design spec, all global constraints, public import compatibility,
checkpoint-key compatibility, fail-safe behavior and actual command output. Re-run the
smallest affected test after each correction, then repeat Steps 2–6.

- [ ] **Step 8: Commit synchronized documentation and final evidence**

```bash
git add README.md System_overview.md docs/技术原理与代码架构.md \
  docs/ENGINEERING_EXECUTION_ROADMAP.md docs/SmartSteer_Status.md docs/README.md \
  docs/superpowers/specs/2026-09-04-staged-policy-learning-b0-design.md \
  docs/superpowers/plans/2026-09-04-staged-policy-learning-b0.md
git commit -m "docs: record pure BC baseline implementation"
```

---

## Final Acceptance Record

At implementation completion, append one dated block here containing:

- every Task 1–8 Red command and the reason it failed before implementation;
- every focused Green command and exact passed/skipped count;
- unit branch coverage percentage;
- flake8 and `git diff --check` outcome;
- full-suite passed/failed/skipped count;
- created run/checkpoint paths, or an explicit statement that no performance-valid
  checkpoint was created;
- remaining external Gate: expert dataset, frozen split, perception checkpoint, B0
  three-seed metrics, CARLA closed loop, deployment and vehicle evidence.
