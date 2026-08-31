# Stage 1A 传感器健康门禁完成记录

## 1. 结论

截至 2026-08-31，Stage 1 的 **CPU 软件门禁**达到 `CPU offline verified`：原始
Camera/LiDAR/IMU 在 `Observation` 和模型计算前经过统一健康检查；任一模态无效
时不调用 perception/policy，直接通过现有安全监督与控制器产生最大制动。

Stage 1 尚不能整体标为车辆或仿真闭环完成。真实 CARLA 传感器物理故障、标定
文件/hash、ROS 2/C++ parity、执行器制动响应和目标 ECU 延迟仍属于 Stage 1B。

## 2. 已实现契约

- `src/utils/sensor_health.py`
  - `SensorHealthConfig`：从唯一 `configs/system.yaml` 派生 shape、标定版本和阈值。
  - `SensorHealthReason`：稳定、可聚合的 Camera/LiDAR/IMU/同步/标定失败原因。
  - `SensorHealthReport`：三模态有效性、frame、reference timestamp、最大 age/skew、
    原因列表及固定 `[camera_valid, lidar_valid]` bool mask。
  - `SensorHealthGate`：NumPy-only、跨帧状态化冻结检查和 episode `reset()`。
- `src/sim/carla_closed_loop.py`
  - Runner 在构造 `Observation` 前强制执行健康门禁。
  - packet timeout 或无效报告均绕过模型并进入 `EMERGENCY_STOP`/最大制动。
  - `EpisodeMetrics` 记录失败次数、原因计数和健康检查 p50/p95 延迟，并支持跨
    episode 聚合。
- `scripts/evaluate_sensor_health_gate.py`
  - 确定性 CPU 故障矩阵；不依赖 CARLA server、ROS 2 或 CUDA。
  - 通过真实 `SafetySupervisor`、`PurePursuitController` 和 CARLA 控制映射验证
    `throttle=0.0, brake=1.0`，不是把“报告无效”直接当成制动成功。

## 3. 数据流

```text
raw Camera/LiDAR/IMU + frame/timestamp/calibration
                         │
                         ▼
                  SensorHealthGate
                  ├─ invalid ─► reason metrics ─► emergency ─► max brake
                  └─ valid   ─► Observation ─► BEVFusion ─► Policy
                                                │
                                                ▼
                                      SafetyFilter/Controller
```

原始 LiDAR 在任何 padding/canonicalization 前检查，空点云不能被零填充伪装成有效
输入。Stage 1A 对 IMU 使用保守策略：无效 IMU 同样急停，尚未启用无 IMU 降级
控制器。

## 4. 当前阈值基线

| 类别 | 配置 | 当前值 | 含义 |
|---|---|---:|---|
| 同步 | `max_sensor_age_seconds` | 0.20 s | 最新模态样本最大年龄 |
| 同步 | `max_sensor_skew_seconds` | 0.05 s | Camera/LiDAR/最新 IMU 最大时间差 |
| Camera | `max_black_ratio` | 0.98 | 黑像素比例上限 |
| Camera | `max_saturation_ratio` | 0.98 | 饱和像素比例上限 |
| Camera/LiDAR | `max_consecutive_identical_frames` | 2 | 连续相同内容容许次数 |
| LiDAR | `min_lidar_points` | 16 | 原始点云最少点数 |
| LiDAR | `min_lidar_valid_ratio` | 0.80 | 坐标范围内点比例下限 |
| LiDAR | `max_lidar_abs_coordinate` | 100 m | XYZ 绝对坐标上限 |
| LiDAR | `max_lidar_duplicate_ratio` | 0.98 | 重复 XYZ 比例上限 |
| IMU | `max_imu_sample_gap_seconds` | 0.11 s | 相邻 IMU 样本最大间隔 |

加速度、角速度及相邻步跳变分别使用 `max_accel_abs`、`max_gyro_abs`、
`max_accel_step_delta` 和 `max_gyro_step_delta`。这些值是 CARLA/CPU 起始基线，
不是车辆级传感器标定结论。

## 5. 原因与动作矩阵

| 故障域 | 已覆盖原因 | Stage 1A 动作 |
|---|---|---|
| 通用 | timeout、非法 frame/reference timestamp、skew | 最大制动 |
| 标定 | calibration missing/mismatch | Camera/LiDAR 无效，最大制动 |
| Camera | count/shape/dtype/non-finite/black/saturated/frozen/frame/time | 最大制动 |
| LiDAR | shape/non-finite/empty/sparse/range/duplicate/frozen/frame/time | 最大制动 |
| IMU | shape/non-finite/incomplete/frame/frequency/range/jump/time | 最大制动 |

单个 packet 可以同时记录多个原因，原因按确定性检查顺序去重。异常对象被收敛为
`packet_malformed` 报告，不允许从健康边界泄漏未分类异常。

## 6. TDD 与验收证据

Red 阶段分别观察到：

- `utils.sensor_health` 不存在导致测试收集失败；
- strict loader 不含 `sensor_health`/派生对象导致 3 个配置测试失败；
- Runner 不接受 `health_gate` 导致 8 个闭环测试失败；
- CPU fault-matrix 模块不存在导致集成测试收集失败；
- episode 聚合缺少 health keys 导致聚合测试失败。

Green/验证结果：

```text
Full suite: 238 collected, 232 passed, 6 skipped, 16 warnings
Unit coverage: 229 passed, 2 skipped, 16 warnings
Total branch coverage: 87.79% (gate: 79%)
sensor_health.py branch coverage: 98.04%
flake8 src/tests/scripts: passed
```

故障矩阵命令：

```bash
python scripts/evaluate_sensor_health_gate.py \
  --samples 1000 --run-id stage1a-20260831-01
```

产物：`runs/sensor_health/stage1a-20260831-01/metrics.json`（Git ignored）。

| 指标 | 结果 |
|---|---:|
| 注入故障 | 1000 |
| 检出/拒绝 | 1000 / 1000 |
| 错误接受 | 0 |
| 检测率 | 100% |
| 最大制动 | 1000 / 1000 |
| fail-safe rate | 100% |
| health latency p50 | 0.680 ms |
| health latency p95 | 0.773 ms |
| 控制周期预算 | 100 ms |

6 个 skip 仍分别对应 CARLA、Gazebo、recorded expert/hazard 数据、CUDA 和完整
ROS 2；它们没有被 CPU/Fake 结果计为完成。

## 7. Stage 1B 待验收

- 在真实 CARLA server 注入冻结、超时、频率下降和标定不匹配，并验证原因日志。
- 使用标定文件内容/hash 校验真实内外参，而不只比较版本字符串。
- 测量仿真/车辆执行器最大制动响应时间与停车距离。
- 在 ROS 2 adapter 和 C++/TensorRT 图外实现同一原因枚举、阈值和动作 parity。
- 在目标 ECU 记录 p50/p95/p99、长期稳定性与资源占用。

在上述证据产生前，Stage 1 的准确状态是“Stage 1A CPU 软件 Gate 已关闭，
Stage 1B 环境 Gate 待验收”。
