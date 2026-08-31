# Configuration Instructions

## Scope

Applies to `configs/`. Read [README.md](./README.md) before changing configuration
ownership.

## Contracts

- `system.yaml` is the single runtime configuration source. Do not add a second
  YAML containing the same BEV geometry, IMU shape, vehicle limits, controller
  period, or simulator values.
- When adding a runtime field, add it to the owning dataclass and strict loader,
  reject unknown/missing keys, and add a configuration test before changing the
  YAML.
- Keep CLI overrides limited to explicit run-mode choices such as policy frame
  or occupancy source. Persisted physical/model parameters remain in
  `system.yaml`.
- Training experiment files, if introduced, may contain optimizer, split,
  random seed, and checkpoint-lineage metadata; they must reference the system
  configuration rather than copy it.
- Do not overwrite experiment-specific configurations or checkpoints. Create a
  new named artifact when reproducibility requires a variant.

## Verification

- Run `python -m pytest -q tests/unit/configuration` after configuration or
  loader changes.
- Run `python -m pytest -q tests/integration/test_cpu_pipeline_contract.py`
  when a field affects more than one module.
- Verify `configuration.system.load_system_stack("configs/system.yaml")`
  succeeds for the canonical file and that invalid/unknown keys remain covered.
