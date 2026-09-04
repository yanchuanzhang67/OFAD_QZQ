# Test fixtures

Static regression fixtures used by the integration / closed-loop tests:

* `sample_point_cloud.npy`   – (N, 4) xyz+intensity LiDAR scan
* `sample_rgb/`              – multi-view RGB frames (jpg/png)
* `expert_log.json`          – synchronized (sensor, expert trajectory) pairs

Generate them with:

```bash
python scripts/make_fixtures.py   # TODO: add this generator
```

The B0 Pure BC CLI integration test builds its expert fixture inside pytest's
temporary directory from the canonical recorded-episode schema. It creates
separate train/validation/test episodes (no frame-level split leakage), varies
all eight `ego-dynamics-v1` fields so train-only normalization is valid, and
uses a strict-loadable BEVFusion state dict. The manifest is always
`fixture=true`; resulting checkpoints and metrics must remain
`model_performance_valid=false` and cannot close the offline model gate.
