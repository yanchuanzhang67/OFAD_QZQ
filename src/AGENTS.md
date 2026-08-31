# Source Code Instructions

## Scope

Applies to all files under `src/`. Read [README.md](./README.md) before changing
package ownership or dependencies; detailed algorithms belong in source
docstrings and architecture documents, not here.

## Contracts

- When adding a cross-module payload, define or reuse its canonical schema in
  `utils/types.py`, `utils/schema.py`, or `utils/contracts.py`; do not create a
  second local version. Verify producer and consumer tests.
- When modules need runtime values, obtain them from dataclasses produced by
  `configuration.system.load_system_stack`; do not read another YAML or repeat
  BEV geometry, vehicle limits, IMU dimensions, or control period constants.
- Keep dependencies directed from shared contracts/configuration toward
  perception, policy, safety, simulation, ROS 2, and deployment. Library code
  must not import from `scripts/` or a concrete simulator entry point.
- When changing a model, preserve public imports, input/output shapes, dtypes,
  frames, and existing checkpoint keys. Test old state-dict keys or document a
  versioned migration before accepting incompatibility.
- In perception, Camera and LiDAR encode independently into canonical
  `(B,C,H=y,W=x)` BEV. Online fusion accepts only
  `[camera_valid, lidar_valid] == [True, True]`; empty or non-finite sensor data
  must fail before encoding.
- In safety/control paths, malformed or unavailable inputs must select an
  emergency trajectory or maximum braking. A fallback must not silently resume
  normal control.
- Keep CARLA, ROS 2, CUDA, ONNX Runtime, and TensorRT imports optional where CPU
  contract tests depend on importing the module without those environments.

## Verification

- Run the nearest `tests/unit/<package>/` tests for the package changed.
- For perception/model interface changes, also run
  `tests/integration/test_cpu_pipeline_contract.py` and
  `tests/unit/deployment/test_onnx_export.py`.
- For simulator or fail-safe changes, run `tests/unit/sim/` and the relevant
  safety/controller tests.
- Then apply the root full-regression, coverage, and flake8 requirements.
