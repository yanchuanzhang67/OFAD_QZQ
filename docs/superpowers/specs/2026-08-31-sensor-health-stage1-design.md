# Stage 1 传感器健康门禁设计规格

## 1. 目标与验收边界

本设计关闭 `ENGINEERING_EXECUTION_ROADMAP.md` 中 Stage 1 的 CPU 软件门禁：
Camera、LiDAR 和 IMU 原始数据必须在任何模型计算前完成健康判定；Camera 或
LiDAR 任一无效、或当前 Stage 1 策略下 IMU 无效时，不调用 perception/policy，
直接产生最大制动并记录确定性的失败原因。

本次成熟度上限为 **unit verified / CPU offline verified**。真实 CARLA 传感器频率、
冻结帧物理真实性、车辆实际制动响应、ROS 2/C++ parity 和目标硬件延迟属于
Stage 1B 环境验收，不能由 Fake/Mock 测试宣称完成。

## 2. 方案选择

采用独立的 `utils.sensor_health.SensorHealthGate`，而不是把健康逻辑塞进
`Observation.__post_init__` 或分别复制到 CARLA/ROS/replay adapter：

- `Observation` 继续表示已经通过入口门禁的同步数据，不承载跨帧冻结状态。
- 健康门禁只依赖 NumPy 和公共契约，可被 CARLA、未来 replay、ROS 2 与 C++
  adapter 对齐实现。
- adapter 只负责提供原始数据及元数据；阈值和判定规则集中管理，避免漂移。

## 3. 公共接口

新增 `src/utils/sensor_health.py`：

```python
class SensorHealthReason(str, Enum): ...

@dataclass(frozen=True)
class SensorHealthConfig:
    num_cameras: int
    image_size: tuple[int, int]
    imu_steps: int
    expected_calibration_version: str
    max_sensor_age_seconds: float
    max_sensor_skew_seconds: float
    black_pixel_threshold: int
    saturation_pixel_threshold: int
    max_black_ratio: float
    max_saturation_ratio: float
    max_consecutive_identical_frames: int
    min_lidar_points: int
    min_lidar_valid_ratio: float
    max_lidar_abs_coordinate: float
    max_lidar_duplicate_ratio: float
    max_accel_abs: float
    max_gyro_abs: float
    max_accel_step_delta: float
    max_gyro_step_delta: float
    max_imu_sample_gap_seconds: float

@dataclass(frozen=True)
class SensorHealthReport:
    camera_valid: bool
    lidar_valid: bool
    imu_valid: bool
    frame_id: int
    reference_timestamp: float
    max_sensor_age_seconds: float
    max_sensor_skew_seconds: float
    reasons: tuple[SensorHealthReason, ...]

    @property
    def valid(self) -> bool: ...

    @property
    def modality_mask(self) -> np.ndarray: ...

class SensorHealthGate:
    def evaluate(...raw packet and metadata...) -> SensorHealthReport: ...
    def reset(self) -> None: ...
```

`modality_mask` 的唯一顺序是 `[camera_valid, lidar_valid]`，dtype 为 `bool`，shape
为 `(2,)`。在线 Runner 只在 `report.valid` 为真时继续，因此正常模型入口始终是
`[True, True]`。

失败原因使用稳定字符串枚举，覆盖：

- packet timeout、frame/timestamp 缺失或非有限、sensor skew、sensor stale、
  calibration 缺失或不匹配；
- camera count/shape/dtype/non-finite/black/saturated/frozen；
- LiDAR shape/non-finite/empty/sparse/out-of-range/repeated；
- IMU shape/non-finite/incomplete/accel range/gyro range/jump。

一个数据包可同时返回多个原因；原因顺序按检查顺序固定，并去重，便于测试和日志
聚合。任何校验异常都转换为无效报告，不能从健康门禁泄漏未分类异常。

## 4. 配置所有权

`configs/system.yaml` 保持唯一运行时配置源：

- `sensors` 新增 `calibration_version`；相机数量、图像尺寸、IMU 窗口继续由已有
  字段派生，健康配置中不重复保存。
- 新增严格 `sensor_health` section，仅保存健康阈值。
- `configuration.system.load_system_stack()` 构造并返回
  `SystemStackConfig.sensor_health`；未知、缺失、越界或互相矛盾的值必须 fail-fast。

初始阈值是 CARLA/CPU 基线，不是车辆级标定结论：最大 age `0.20 s`、最大 skew
`0.05 s`、黑/饱和像素比例上限 `0.98`、连续相同帧上限 `2`、最少 LiDAR 点数
`16`、有效点比例下限 `0.80`、坐标绝对值上限 `100 m`、重复点比例上限
`0.98`。IMU 加速度、角速度与步间跳变使用独立阈值，单位分别为 `m/s^2` 和
`rad/s`。

## 5. 数据流与故障安全

```text
raw Camera/LiDAR/IMU + frame/timestamp/calibration
                 │
                 ▼
          SensorHealthGate
          ├─ invalid ─► reason metrics ─► emergency trajectory ─► max brake
          └─ valid   ─► Observation ─► BEVFusion ─► Policy ─► Safety ─► Control
```

`CarlaClosedLoopRunner` 必须接收 `SensorHealthGate`。每个 episode 开始时调用
`reset()`，每个 packet 在 `Observation` 前调用 `evaluate()`。无效报告立即停止
当前 episode，不调用 perception 或 policy，并通过现有
`PurePursuitController + emergency_trajectory` 输出最大制动。

Stage 1 对 IMU 采用显式保守策略：IMU 无效同样 `EMERGENCY_STOP`。未来只有在
独立的降级控制器和风险验收存在后，才允许把某些 IMU 原因改为 `DEGRADED`。

CARLA sensor stack 暴露当前 calibration version。原始 LiDAR 必须先过健康门禁，
之后才允许做固定点数 canonicalization；空点云不能通过 padding 伪装成有效输入。

## 6. 状态化检查

Camera 和 LiDAR 分别维护确定性内容摘要及连续相同计数。只有 frame/timestamp 已
向前推进但内容摘要连续相同时才增加冻结计数；超过
`max_consecutive_identical_frames` 才报告 frozen/repeated。结构错误会清空对应
摘要状态，episode 切换必须统一 `reset()`，避免跨 episode 误报。

Camera 黑屏与过曝基于所有相机像素的比例分别计算。LiDAR 有效点指 XYZ 均在配置
绝对坐标范围内的点；任何 NaN/Inf 立即判无效，范围内比例不足或总点数不足分别
产生独立原因。重复点比例按 XYZ 唯一点数计算。

IMU 必须严格为 `(imu_steps, 6)`，且 `imu_frames`、`imu_timestamps` 也必须完整、
严格递增，相邻时间差不得超过 `max_imu_sample_gap_seconds`。前 3 列是加速度，
后 3 列是角速度；分别检查绝对范围和相邻步差。跨模态 skew 只比较所有 Camera、
LiDAR 与最新 IMU 时间，不把历史 IMU 窗口跨度误算为同步误差。

## 7. 指标与审计产物

`EpisodeMetrics` 新增：

- `sensor_health_failures`；
- `sensor_health_failure_reasons` 原因计数字典；
- health gate 单次延迟样本及 summary 中的 p50/p95。

新增 CPU 评估入口生成新的 `runs/sensor_health/<run_id>/metrics.json`，包含故障
注入总数、每类检测结果、错误接受数、fail-safe 结果和 p50/p95。运行目录是新的
显式目录且保持 Git ignored，不覆盖历史输出。

## 8. TDD 与验收

严格执行 Red → Green：

1. 公共健康 schema/config 校验测试先失败，再实现最小接口。
2. Camera、LiDAR、IMU、时间同步和冻结状态逐类增加失败测试并转 Green。
3. 配置 loader 测试先证明 `sensor_health` 缺失，再实现 YAML/loader。
4. FakeRunner 故障注入先证明坏帧仍会进入旧异常路径或缺少原因，再接入门禁；
   断言 perception/policy 均未调用且最终 brake 为 `1.0`。
5. CPU integration fault matrix 覆盖至少 1000 个确定性样本，错误接受为 0。

最终运行 focused tests、全量 `pytest -ra`、unit branch coverage、flake8 与
`git diff --check`。CPU Gate 只有在所有枚举的确定性故障检测率和最大制动率均为
`100%`、错误接受为 `0`、p95 小于 `control.dt` 时关闭。

## 9. 明确保留的 Stage 1B 项

- 真实 CARLA 摄像头/LiDAR/IMU 频率、超时与冻结故障注入；
- calibration 文件内容/hash 与真实内外参一致性，而非仅版本字符串契约；
- 真实车辆/仿真执行器最大制动响应时间和停车距离；
- ROS 2 adapter、C++/TensorRT 图外健康门禁的原因枚举与阈值 parity；
- 目标 ECU 的 p50/p95/p99 延迟和长期稳定性。

这些项目在本次文档中只能标为环境待验收，不能用 CPU FakeRunner 结果替代。
