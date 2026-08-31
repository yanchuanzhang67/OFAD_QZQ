# Test Suite Instructions

## Scope

Applies to `tests/`. Use [README.md](./README.md) for the test layout and current
environment boundaries.

## TDD and test placement

- For new behavior or a bug fix, first add a test that fails for the intended
  reason, record the Red result, implement the minimum Green change, then
  refactor without weakening assertions.
- Put deterministic algorithm, shape, type, exception, and fail-safe contracts
  in `unit/`. Put CPU-only cross-module workflows in `integration/`. Reserve
  `closed_loop/` for tests that require a real CARLA/Gazebo environment.
- Mirror source packages below `tests/unit/` for new tests. Keep legacy test
  locations stable unless moving them is part of the requested cleanup.
- Assert observable outputs: tensor shape/dtype, finite values, physical units,
  coordinate frame, constraints, exception type, or final control command.
  Avoid assertions that merely reproduce implementation details.

## Mock, Fake, and skip boundaries

- Use Mock/Fake adapters to exercise software orchestration on CPU, including
  timeouts, dependency errors, invalid payloads, and maximum braking.
- Do not use Mock/Fake results as evidence for sensor physics, vehicle dynamics,
  simulator synchronization, dataset quality, GPU kernels, ROS 2 integration,
  or target-hardware performance.
- CPU contract tests must not be skipped. Environment/data tests may skip only
  with a precise missing dependency or fixture reason.
- Do not delete or loosen a failing safety regression to make the suite Green;
  fix the implementation or report the verified blocker.

## Verification

- Focused test: `python -m pytest -q <test-path>`.
- Unit suite with branch coverage:
  `python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-branch`.
- Full suite: `python -m pytest -ra`.
- A completion report must separate passed, failed, and skipped tests and must
  not describe a skip as a completed environment acceptance.
