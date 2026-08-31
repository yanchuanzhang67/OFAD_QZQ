# New_ORAD Repository Instructions

## Scope and navigation

- This file applies to the whole repository. Before editing a path, also read
  the nearest child `AGENTS.md`; child rules add module-specific constraints.
- Use [README.md](./README.md) for project entry points,
  [System_overview.md](./System_overview.md) for the system SDD, and the
  nearest package/document README for detailed architecture. Do not turn this
  file into a project encyclopedia.
- Only uppercase `AGENTS.md` is a project instruction file. If an
  `AGENTS.override.md` is introduced, treat it as replacing the same-directory
  `AGENTS.md`, not extending it.

## Repository-wide workflow

- Before editing, inspect `git status` and the target files. Preserve existing
  uncommitted work and change only files required by the current request.
- When Python behavior changes, add or update a failing pytest first, implement
  the smallest fix, then run focused tests and the full regression suite.
- When only documentation changes, run link/path checks and `git diff --check`;
  do not claim code tests were rerun unless they actually were.
- When a durable architecture, acceptance boundary, or test baseline changes,
  update the owning SDD/TDD document and its navigation index in the same task.
- Do not overwrite raw datasets, trained weights, exported engines, or previous
  experiment outputs. Write generated artifacts to a new explicit directory.

## Shared contracts

- `configs/system.yaml` is the only runtime configuration source. Assemble
  modules with `configuration.system.load_system_stack`; unknown, missing, or
  cross-module-inconsistent values must fail fast.
- Keep `Observation`, `BEVFeature`, `Trajectory`, coordinate frames,
  timestamps, tensor shapes, and SI units explicit across module boundaries.
- Invalid, missing, stale, skewed, or non-finite online data must reach the
  fail-safe path; never hide sensor loss with an undocumented zero tensor.
- Online BEV fusion requires Camera and LiDAR simultaneously valid. The
  `(B, 2)` mask order is `[camera_valid, lidar_valid]`, and every online sample
  must be `[True, True]`.
- Preserve stable public imports, serialized schema, and checkpoint parameter
  keys. If compatibility must break, add a versioned migration and regression
  test, and report the break explicitly.
- Treat external repositories as references. Record provenance and license
  decisions before copying code or introducing a dependency.

## Verification

- Install development dependencies: `python -m pip install -e ".[dev,perception]"`.
- Full regression after Python behavior changes: `python -m pytest -ra`.
- Unit branch coverage after critical-path changes:
  `python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-branch`.
  The repository gate is 79%; changed critical code should reach 95% where
  feasible, and P0 fail-safe branches require complete focused coverage.
- Static checks after Python edits:
  `python -m flake8 src tests scripts --max-line-length=99`.
- Before completion, run `git diff --check`, report commands and outcomes, and
  list environment-dependent tests that remain skipped.

## Child scopes

- `src/AGENTS.md`: source dependencies, runtime contracts, and model safety.
- `tests/AGENTS.md`: TDD, test-layer semantics, skips, Mock/Fake boundaries.
- `configs/AGENTS.md`: canonical configuration ownership and validation.
- `docs/AGENTS.md`: SDD/TDD evidence, status language, and document indexes.
