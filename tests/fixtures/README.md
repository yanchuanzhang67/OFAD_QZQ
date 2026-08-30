# Test fixtures

Static regression fixtures used by the integration / closed-loop tests:

* `sample_point_cloud.npy`   – (N, 4) xyz+intensity LiDAR scan
* `sample_rgb/`              – multi-view RGB frames (jpg/png)
* `expert_log.json`          – synchronized (sensor, expert trajectory) pairs

Generate them with:

```bash
python scripts/make_fixtures.py   # TODO: add this generator
```
