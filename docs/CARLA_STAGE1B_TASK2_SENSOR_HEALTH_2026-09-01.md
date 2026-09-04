# Stage 1B Task 2：真实 CARLA Sensor → HealthGate

> 状态日期：2026-09-01  
> 当前成熟度：**单元验证**  
> 真实环境验收：**待 CARLA 1000+ normal ticks**

## 2026-09-03 补充：CARLA 0.9.16 recorded replay 证据

2026-09-01 的在线 Exit Gate 未被改写：仍需连接真实 CARLA 0.9.16，以相同配置在
warm-up 后完成至少 1000 个 normal ticks。2026-09-03 新增的只读回放用于提前检验
当前阈值对已有真实传感器记录的行为，不等同于在线仿真验收。

完整 200 帧回放产物为
[replay_20260903T035626Z_9ca5f754](../artifacts/carla_replay/replay_20260903T035626Z_9ca5f754/summary.json)：

- 199 帧通过 HealthGate，1 帧因启动期 `imu_accel_range + imu_jump` 被拒绝；
- observed rejection rate 为 `0.005`。数据没有 normal/invalid 健康真值，因此不能
  把该数直接称为 false rejection rate；
- health latency p50/p95/max 为 `6.05/7.11/8.89 ms`；
- 199 个有效帧均进入真实 BEVFusion/HybridPolicy 代码边界，network finite rate
  为 `1.0`；模型为固定 seed 的随机初始化 smoke，模型性能和闭环验收标志均为
  `false`；
- 历史 manifest 未记录 vehicle/config/calibration hash，产物原样保留三个
  provenance gaps，未用当前配置反向补写历史事实。

这项证据说明，在这 200 帧记录上，0.20 s age、0.05 s skew、0.11 s IMU gap 等
阈值没有造成持续性拒绝；但样本数不足、缺健康真值且不是在线 stream，所以还不能
回答 Task 2 的最终问题。首帧现象应继续通过明确 warm-up 语义处理，不放宽运行阈值。

## 1. 范围与数据流

Task 2 只验证真实传感器流和健康门禁，不加载 perception/policy：

```text
CARLA 0.9.16 / Town10HD_Opt
        ↓
Camera / LiDAR / IMU raw packet
        ↓
Observation candidate metadata
        ↓
SensorHealthGate
        ↓
per-tick trace + aggregate metrics
```

运行前先完成 20 ticks warm-up，使 IMU history 达到 `(10, 6)`；warm-up 不进入
误拒绝统计。正式阶段使用固定 `throttle=0.15 / steer=0.05 / brake=0.0` 的低速
控制让视觉与点云内容正常变化，但不依赖任何训练模型。

## 2. 时间语义修正

旧 CARLA Runner 使用 `frame * control.dt` 作为 reference timestamp。当 world 的
frame counter 不是从零开始时，它可能与 sensor event timestamp 不在同一起点，
进而把正常帧误判为 stale。

现在 `CarlaSensorStack.tick()` 读取同 frame world snapshot 的
`timestamp.elapsed_seconds`。snapshot frame 不匹配会 fail-fast，HealthGate 和
Observation 均使用该 CARLA simulation time。

## 3. 运行产物与指标

```text
runs/carla_sensor_health/<run-id>/
├── manifest.json
├── ticks.jsonl
└── summary.json
```

Manifest 继承 Task 1 的 commit/config/CARLA/map/vehicle/calibration/sensor
provenance，并增加 warm-up、请求 tick 数、固定控制和健康阈值。逐 tick trace
保存所有模态 frame/timestamp、world reference timestamp、valid/reasons、sensor
age/skew 和 HealthGate latency。

Summary 汇总 false rejection count/rate、原因计数、frame/IMU continuity、sensor
age/skew p50/p95/max、health latency p50/p95/max 和 acquisition errors。任何失败
都会保留证据，不会自动放宽 `system.yaml` 阈值。

## 4. TDD 过程记录

### Red 1：normal-stream metrics

```bash
python -m pytest -q tests/unit/sim/test_carla_sensor_health_stream.py
```

结果：collection error，`sim.carla_sensor_health` 尚不存在。

### Green 1

实现 snapshot time、false rejection/distribution metrics、连续性检查、1000-tick
Exit Gate、acquisition error 和不可覆盖 JSONL 后，新测试 `6 passed`。

### Red 2：非零 world frame 时间起点

```bash
python -m pytest -q \
  tests/unit/sim/test_carla_runner.py::test_runner_uses_world_elapsed_time_when_frame_counter_has_offset
```

结果：失败。旧实现把 frame `107` 推导为 `10.7 s`，真实 sensor timestamp 为
`0.7 s`，产生一次错误 health rejection。

### Green 2 与 focused verification

```bash
python -m pytest -ra tests/unit/sim \
  tests/closed_loop/test_carla_closed_loop.py
python -m flake8 src/sim/carla_sensor_health.py \
  src/sim/carla_closed_loop.py scripts/evaluate_carla_sensor_health.py \
  tests/unit/sim tests/closed_loop/test_carla_closed_loop.py \
  --max-line-length=99
```

结果：`47 passed / 4 skipped`；focused flake8 通过。四项 skip 分别是坡道 CARLA、
Gazebo、Task 1 真实 CARLA 和 Task 2 真实 CARLA 环境验收。

记录时来源 commit 为 `99fc0ea3991cc5e8fc3367529a939b7dee7e54f3`，config
SHA-256 为
`431b9898893141227ad6e7e1589a679a468ec99af51345539e42d8db6461198a`；工作树为
dirty。真实运行 manifest 会重新计算当时 provenance。

## 5. 真实 CARLA 命令与 Exit Gate

```bash
python scripts/evaluate_carla_sensor_health.py \
  --config configs/system.yaml \
  --run-id stage1b-task2-<UTC时间> \
  --warmup-ticks 20 \
  --ticks 1000
```

Exit Gate：

- evaluated ticks `>= 1000`；
- false rejections `== 0`；
- frame/IMU continuity violations `== 0`；
- sensor age max `<= 0.20 s`；
- sensor skew max `<= 0.05 s`；
- HealthGate latency p95 `< 100 ms`；
- acquisition errors 为空。

2026-09-01 记录时的环境没有 CARLA PythonAPI；2026-09-03 本轮也未连接 CARLA
server。只有上述在线命令生成的真实 `summary.json` 才能把本 Task 提升为
仿真闭环验证。
