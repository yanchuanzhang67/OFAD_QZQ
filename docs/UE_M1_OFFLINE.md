# Native UE Scout M1 offline consumer

2026-09-10: offline verified. The enclosing Unreal project produced and this
consumer validated a genuine 30-frame rendered UE episode; synthetic fixtures
separately establish software contracts. No BC/closed-loop acceptance is claimed.

## Provenance and boundary

Local reference clone from `C:/Users/29105/AppData/Local/Temp/ofad-review-20260910/model`,
upstream commit `66da2b2cbf6f9de330c32f89a493fba26aaf61ba`, local branch `feature/ue-m1`.
No remote push. Existing repository license declaration is Apache-2.0; this change
adds original adapter code and uses existing numpy/Pillow dependencies. It does not
copy another external implementation. CARLA readers, models and checkpoint keys stay intact.

The frozen producer contract is `Docs/OFAD_M1_Contract.md` in the enclosing Unreal
project. `ue_recording` is a separate numpy/Pillow-only package because importing
the existing `replay` package assembles the Torch/CARLA policy pipeline.
Shared payloads live in `utils.types`.

## Read and validate

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e .
.venv/Scripts/python.exe scripts/verify_ue_episode.py <episode-directory>
```

```python
from ue_recording import UERecordedEpisode

episode = UERecordedEpisode.open("path/to/episode")
frame = next(episode.iter_frames())
assert frame.canonical_point_cloud.shape == (256, 4)
```

`open` validates all metadata and decodes every image/point file before accepting
the episode. Iteration decodes frames on demand without caching all sensor arrays.
Input directories must contain `_SUCCESS` and must not end in `.incomplete`.
Missing, malformed, non-finite or inconsistent input raises `UERecordingError`
(a `ValueError`); the CLI reports it on stderr and returns exit code 1.

`RecordedUEFrame` exposes front/rear/top RGB HWC uint8 images, complete sensor-frame
little-endian float32 xyzi, deterministic 256x4 ego-frame points, 10x6 float32 IMU,
ego measurements, actual applied command and source. Canonical points use
`default_rng(sample_index).choice(N, 256, replace=False)` then the calibrated rigid
transform. Fewer than 256 raw points is an error; there is no padding or duplication.

`ScoutDynamicsV1` is explicitly `scout-dynamics-v1`: signed body-forward speed,
measured yaw rate, pitch, roll, body vx/vy, motion accel_z, commanded yaw rate.
Pitch/roll use right-handed quaternion xyzw with ZYX Euler convention. IMU contains
specific force including gravity, separately from ego motion acceleration.
Commands remain distinct from measurements. Scout commanded yaw rate is never car
steering and these arrays are not passed to old BC weights or `VehicleState`.

Checks cover both schema versions, source and non-expert labels, SHA1 of calibration
bytes, rigid transforms, optical mapping, intrinsics, PNG size/mode, unit quaternion,
finite fields, exact raw byte count, path containment including resolved symlinks,
unique sensor file paths, contiguous sample indices, physical frame/timestamp
relation, frame stride 6, sample interval 0.1 s, and same-frame association.
Ten IMU records are at **0.1 s**, frame stride 6 (physics is 1/60 s); overlapping
histories must agree. Unknown extra JSON keys are tolerated; required keys and
values are strict and duplicate JSON keys are rejected.

CLI JSON reports actual array shapes, raw point count range, latest IMU mean/min/max,
Scout state field names/ranges and non-expert source. These are offline data evidence,
not BC performance, sensor physics or closed-loop driving acceptance.

## TDD and validation evidence

The initial focused run failed with `AssertionError: UE consumer is missing` before
implementation. Current format tests: 56 passed; `ue_recording` statement and branch
coverage 100%. The tests write synthetic PNG/xyzi/metadata files, exercise corruptions,
check coordinate results and deterministic sampling, and run the actual CLI as a subprocess.

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/ue_recording --cov=ue_recording --cov-report=term-missing --cov-branch
.venv/Scripts/python.exe -m pytest -ra
.venv/Scripts/python.exe -m pytest tests/unit --cov=src --cov-report=term-missing --cov-branch
.venv/Scripts/python.exe -m flake8 src tests scripts --max-line-length=99
git diff --check
```

The isolated validation environment uses Python 3.12.13; numpy/Pillow suffice for
the consumer. Repository regression also uses pytest, pytest-cov, flake8, Torch
2.5.1 and ONNX. The first latest-Torch run exposed missing `onnxscript` in historical
ONNX export tests; Torch 2.5.1 matches their legacy exporter API without changing
existing production code. Full regression has an existing Windows path separator
assertion in `tests/sim/test_collect_carla_initial.py` (expects `/` from `str(Path)`).
The test now compares Path objects directly, preserving the intended path check
on both Windows and Linux without changing CARLA production behavior.

Final results in this task: full regression **447 passed, 8 skipped**; unit suite
**428 passed, 2 skipped**, branch-inclusive coverage **90.57%**. Focused UE reader
suite **56 passed**, statement/branch coverage **100%**. Flake8 and git diff --check
passed. Skips require CARLA/Gazebo, real expert/hazard logs, ROS2 or CUDA.

Actual producer acceptance: `episode_20260910T075416Z_449DCA144398A7B91C017698D7A69DF4`
under the enclosing project's `Saved/OFAD/M1Acceptance`. Thirty frames contain
90 RGB images and 6781–6838 raw points per frame, canonical 256x4 and IMU10x6.
The stationary/forward/left-turn sequence has correct measured velocity/yaw signs.
See the enclosing project's `Docs/OFAD_M1_Guide.md` and
`Saved/OFAD/M1Acceptance/verification.json` for actual logs, source hashes and limits.

## Forest V2 offline consumer（2026-09-15）

Forest 单相机协议使用独立的 `UERecordedEpisodeV2`，不放宽本页 V1 loader。验证命令：

```powershell
.venv\Scripts\python.exe scripts\verify_ue_episode_v2.py <forest-v2-episode>
```

V2 严格要求单路 `(180,320,3)` 图像、profile 一致的 `float32 (N,4)` LiDAR、`(10,6)` IMU、同帧 10 Hz 时序和完整 provenance。当前 smoke 数据固定拒绝专家资格。生产端和完整契约见 UE 仓库的 `Docs/OFAD_Forest_V2_Guide.md`。
