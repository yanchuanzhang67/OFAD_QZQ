# New_ORAD CARLA 多模态数据采集系统设计

- 状态：已确认设计，待实现
- 日期：2026-09-01
- 目标版本：CARLA 0.9.16
- 目标平台：Windows CARLA Server + WSL2 Ubuntu 22.04 Python Client
- 数据契约版本：`new-orad-carla-v1`
- 标定版本：`carla-default-v1`

## 1. 背景与核心判断

New_ORAD 已经定义 Camera、LiDAR、IMU、ego state、frame 和 timestamp 的
`Observation` 契约，也已有 CARLA 闭环入口、传感器健康门禁及安全控制骨架；当前缺口
不是再写一个孤立的传感器示例，而是建立可追溯、严格同步、能直接支持 BC、RSSM 和
Dreamer 的 episode 录制子系统。

本设计采用如下主线：

```text
人工/专家控制
→ 同步采集原始传感器
→ 健康与时序校验
→ 保存 Observation_t、Action_t、Observation_t+1
→ 原子提交 episode
→ 只读 replay loader 重建现有 Observation 契约
```

其中最重要的约束是：

1. 三个 Camera、LiDAR 和控制帧必须使用 CARLA `frame_id` 精确对齐。
2. `Action_t` 必须明确表示从 `Observation_t` 到 `Observation_{t+1}` 的动作。
3. 必须同时保存驾驶员的 `action_raw` 和安全处理后的 `action_applied`。
4. 原始 LiDAR 不得在落盘前被不可逆裁剪为 256 点。
5. CARLA 坐标系与 New_ORAD 坐标系必须显式标注并转换，禁止隐式混用。
6. 没有 `_SUCCESS` 的 episode 不得进入训练或正式 replay。

## 2. 目标与非目标

### 2.1 目标

- 采集 3 路 `192×192` RGB：`front`、`rear`、`top`。
- 保存完整 LiDAR `(N,4)=[x,y,z,intensity]` 和 canonical `(256,4)`。
- 保存每个原始 IMU sample，并为每个控制帧构造 `10×6` IMU history。
- 保存 ego 位姿、速度、加速度、角速度、归一化转向量和实际前轮转角。
- 保存 `steer/throttle/brake` 原始动作和实际执行动作。
- 保存 collision、goal、emergency stop 和 safety intervention。
- 保存相机内参、所有传感器外参、地图、天气、版本、配置哈希和控制周期。
- 支持键盘及常见 Pygame 手柄；无手柄时自动回退键盘。
- 支持程序异常后的安全制动、Actor 清理、world settings 恢复和不完整数据隔离。
- 输出可由只读 loader 确定性重建为现有 New_ORAD `Observation`。

### 2.2 非目标

- 本阶段不实现 BC、RSSM 或 Dreamer 训练器。
- 本阶段不把 Traffic Manager 轨迹当作越野 BC 专家数据。
- 本阶段不修改 BEVFusion、HybridPolicy 或现有 checkpoint 契约。
- 本阶段不宣称 Stage 1B 已完成；必须运行真实 CARLA 验收并保存证据后才能升级状态。
- 本阶段不提供开放道路或载人车辆部署能力。

## 3. 方案选择

### 3.1 采用方案

新增独立录制子系统，复用当前配置、公共类型、健康门禁和 frame 约定：

```text
scripts/collect_carla_dataset.py
        │
        ├── HumanControlSource
        ├── CarlaSynchronizedSensors
        ├── SafetyActionTracker
        └── CarlaEpisodeWriter
```

这样可以把“在线推理需要的 canonical Observation”和“研究需要的无损原始数据”分开，
同时避免录制逻辑侵入 `CarlaClosedLoopRunner`。

### 3.2 放弃的方案

- 单文件 `collect_data.py`：会复制严格配置、公共 schema、坐标转换和清理逻辑。
- 直接扩张 `CarlaSensorStack`：现有类偏向在线闭环，只保留最新数据且较早 canonicalize，
  不适合负责长期磁盘事务。
- 将录制直接嵌入 `CarlaClosedLoopRunner`：会把人工 BC、策略 rollout 和评估职责耦合。
- 单一大型 HDF5：单文件吞吐较好，但崩溃恢复、逐帧检查和变长 LiDAR 处理复杂；第一阶段
  优先采用 episode 目录和显式索引。

## 4. 代码边界

计划新增：

```text
src/sim/carla_recording.py
src/sim/human_control.py
scripts/collect_carla_dataset.py
tests/sim/test_carla_recording.py
tests/sim/test_human_control.py
tests/integration/test_carla_recording_contract.py
docs/superpowers/specs/2026-09-01-carla-dataset-collection-design.md
```

计划修改：

```text
configs/system.yaml
src/configuration/*          # 严格 recording 配置及跨模块一致性校验
src/sim/__init__.py          # 稳定公共导出（如项目约定需要）
tests/configuration/*
README.md                    # 采集入口和状态边界
```

模块职责：

| 组件 | 职责 | 不负责 |
|---|---|---|
| `HumanControlSource` | 键盘/手柄事件、连续动作、死区、急停 | CARLA tick、写盘 |
| `CarlaSynchronizedSensors` | 创建传感器、按 frame 聚合、超时与清理 | 训练预处理、磁盘格式 |
| `SafetyActionTracker` | 保存 raw/applied 动作及干预原因 | 策略推理、轨迹规划 |
| `CarlaEpisodeWriter` | 原子写帧、manifest、完成标记 | CARLA Actor 生命周期 |
| `collect_carla_dataset.py` | 装配、episode 状态机、CLI | 定义跨模块 payload |

CARLA 和 Pygame import 必须保持可选；没有这些依赖时，numpy/config/writer 单元测试仍可
在 CPU CI 中导入和运行。

## 5. 运行时数据流与时序语义

### 5.1 初始化

```text
连接 CARLA
→ 读取并验证 system.yaml
→ 保存原始 world settings
→ 设置 synchronous_mode=True
→ fixed_delta_seconds=closed_loop.dt=0.1 s
→ 创建 ego vehicle 和传感器
→ warm-up 10 帧但不录制
→ 收到第一份完整 Observation_0
```

同一个 CARLA Server 只允许该采集 Client 调用 `world.tick()`；其他 Client 只能观察，
不能推进世界。

### 5.2 单步 transition

每一步必须采用以下顺序：

```text
已有 Observation_t
→ 展示 front camera 并读取人工动作 ActionRaw_t
→ 急停/安全适配得到 ActionApplied_t
→ 写入待提交 Sample_t
→ vehicle.apply_control(ActionApplied_t)
→ world.tick() 得到 frame t+1
→ 收齐并校验 Observation_t+1
→ 补全 Sample_t.next_frame_id
→ 原子提交 Sample_t
```

定义：

\[
D_t = \{O_t, A_t^{raw}, A_t^{applied}, E_t, O_{t+1}\}
\]

其中 RSSM 使用：

\[
p(s_{t+1}\mid s_t,a_t)
\]

BC 默认监督 `action_raw`；实际车辆动力学 transition 使用 `action_applied`。当两者不同时，
该样本必须带 `safety_intervention=true`，训练器可选择剔除、单独分层或用于安全学习。

### 5.3 IMU 时间窗口

- IMU `sensor_tick` 与控制周期一致，当前为 `0.1 s`，即 10 Hz。
- 每个 sample 保存最近 10 个有效 IMU sample，覆盖约 1 秒历史。
- 顺序固定为从旧到新，通道为 `[ax, ay, az, gx, gy, gz]`。
- episode 开头不足 10 帧时在前部补零，并保存 `imu_valid_count`；补零不能伪装成真实值。
- 每个 IMU sample 单独保留 `frame_id` 和 `timestamp`。

## 6. 传感器设计

### 6.1 Camera

固定三路 RGB，输出 `uint8 [192,192,3]`：

| 名称 | 位置 `[x,y,z] m` | 旋转 `[pitch,yaw,roll] deg` | FOV |
|---|---|---|---:|
| `front` | `[1.5,0.0,1.6]` | `[0,0,0]` | 90° |
| `rear` | `[-1.5,0.0,1.6]` | `[0,180,0]` | 90° |
| `top` | `[0.0,0.0,1.9]` | `[-15,0,0]` | 100° |

`top` 表示高位、轻微向下俯视的前向相机，不表示垂直向下的鸟瞰相机。三个相机使用
相同 `sensor_tick=0.1 s`，并要求 `image.frame == world.tick()` 返回的 frame。

### 6.2 LiDAR

默认配置：

```text
blueprint: sensor.lidar.ray_cast
channels: 32
range: 50 m
points_per_second: 320000
rotation_frequency: 10 Hz
upper/lower FOV: +15° / -25°
sensor_tick: 0.1 s
```

每帧保存两份：

- `lidar_raw/<frame>.npy`：CARLA 原始 `float32 [N,4]`。
- `lidar_256/<frame>.npy`：转换到 New_ORAD ego 坐标后的 `float32 [256,4]`。

canonical sampling 使用 `dataset_seed + episode_id + frame_id` 生成确定性随机种子。点数
大于 256 时无放回采样，小于 256 时末尾补零；同时记录 `raw_point_count`、
`valid_point_count`、`padding_count` 和 seed。健康检查必须发生在补零/裁剪之前。

### 6.3 IMU 与 collision

- IMU 使用 `sensor.other.imu`，保存加速度 `m/s²` 和角速度 `rad/s`。
- collision 使用 `sensor.other.collision`，事件按其 `frame` 归入对应 sample。
- collision 保存 other actor ID/type、三轴 impulse 和 impulse norm。
- 没有 collision callback 的普通帧显式保存 `collision=false`，不能把“无事件”当缺失数据。

## 7. 坐标系与单位

### 7.1 坐标系

CARLA sensor/ego 坐标使用左手系：

```text
x: forward
y: right
z: up
```

New_ORAD ego 轨迹约定：

```text
x: forward
y: left
z: up
```

因此 canonical LiDAR 使用：

\[
[x,y,z]_{ORAD} = [x,-y,z]_{CARLA}
\]

原始点云不得改写，必须在元数据中标记 `frame="carla_sensor"`；canonical 点云标记
`frame="ego"`。世界位姿保存 CARLA raw 表示供重放，同时由 frame helper 生成
New_ORAD consumer 所需表示。任何 loader 都不得根据文件名猜测坐标系。

### 7.2 单位

| 字段 | 存储单位 |
|---|---|
| position / distance | `m` |
| velocity / speed | `m/s` |
| acceleration | `m/s²` |
| roll/pitch/yaw | `rad` |
| angular velocity / IMU gyro | `rad/s` |
| CARLA normalized steer | `[-1,1]` |
| wheel steer angle | `rad` |
| timestamps | 仿真秒 `s` |

CARLA `get_angular_velocity()` 的 degree/s 输出必须转换为 rad/s；CARLA rotation 的 degree
必须转换为 rad。原始角度只有在带 `_deg` 后缀时才允许保存。

## 8. Ego、Action、Goal 与 Event schema

### 8.1 Ego state

每帧至少包含：

```json
{
  "position_world_m": [0.0, 0.0, 0.0],
  "rotation_world_rad": [0.0, 0.0, 0.0],
  "velocity_world_mps": [0.0, 0.0, 0.0],
  "speed_mps": 0.0,
  "angular_velocity_world_radps": [0.0, 0.0, 0.0],
  "acceleration_world_mps2": [0.0, 0.0, 0.0],
  "steer_normalized": 0.0,
  "front_left_wheel_steer_rad": 0.0,
  "front_right_wheel_steer_rad": 0.0,
  "wheel_steer_available": true
}
```

如果当前车辆 blueprint/API 不提供实际轮角，两个轮角保存 `null` 且
`wheel_steer_available=false`，不能使用 normalized steer 冒充物理轮角。

### 8.2 Action

`action_raw` 和 `action_applied` 使用相同结构：

```json
{
  "steer": 0.0,
  "throttle": 0.0,
  "brake": 0.0,
  "hand_brake": false,
  "reverse": false,
  "gear": 1
}
```

`steer∈[-1,1]`，`throttle/brake∈[0,1]`。第一版安全动作适配器只负责人工急停和
采集 fail-safe；它不把现有 trajectory-level `SafetySupervisor` 假装成已经适配人工
VehicleControl。急停时固定输出：

```text
steer=0.0, throttle=0.0, brake=1.0, hand_brake=false
```

### 8.3 Goal 与事件

goal 使用 world XY 平面距离：

\[
d_t = \sqrt{(x_t-x_g)^2 + (y_t-y_g)^2}
\]

`d_t <= closed_loop.goal_tolerance` 时 `reached_goal=true`。事件字段至少包括：

```text
collision
collision_impulse
reached_goal
distance_to_goal_m
emergency_stop
safety_intervention
safety_reason
terminal
terminal_reason
```

`terminal_reason` 枚举固定为：

```text
goal_reached
collision
emergency_stop
manual_stop
max_steps
sensor_timeout
synchronization_error
write_error
```

## 9. 磁盘格式

### 9.1 数据集目录

```text
datasets/carla/
├── dataset_manifest.json
├── .incomplete/
└── episodes/
    └── episode_000001/
        ├── episode.json
        ├── calibration.json
        ├── frames.jsonl
        ├── camera_front/
        ├── camera_rear/
        ├── camera_top/
        ├── lidar_raw/
        ├── lidar_256/
        └── _SUCCESS
```

### 9.2 Dataset manifest

至少保存：

```text
schema_version
created_at_utc
project_name
carla_client_version
carla_server_version
git_commit
git_dirty
system_config_sha256
dataset_seed
```

无法取得 Git commit 时保存 `null` 并将 `git_commit_available=false`，不得伪造。

### 9.3 Episode manifest

至少保存：

```text
episode_id
schema_version
map
weather
ego_blueprint
spawn_transform
goal_world
control_timestep_seconds
start/end frame
start/end simulation timestamp
sample_count
terminal_reason
collision_count
safety_intervention_count
random_seed
calibration_version
```

### 9.4 Frame index

`frames.jsonl` 每行对应一个 transition，至少包含：

```text
sample_index
frame_id
next_frame_id
timestamp
next_timestamp
各 sensor 的 frame/timestamp/skew
camera 相对路径
lidar raw/canonical 相对路径
IMU history、frame 列表、timestamp 列表、valid_count
ego_state
action_raw
action_applied
event
health_report
```

frame 文件名使用 CARLA frame，而不是 episode 内 sample index，例如
`camera_front/00000124.png`。`sample_index` 只表达 episode 内顺序。

## 10. 原子写入与恢复

每个新 episode 首先创建在：

```text
datasets/carla/.incomplete/episode_<id>_<uuid>/
```

单帧文件写入 `<name>.tmp`，成功 flush 后使用同文件系统原子 rename 变为正式名称；只有
所有模态文件完成后才 append `frames.jsonl`。每 `flush_interval_frames` 执行一次 flush，
episode 正常结束时强制 flush 并生成 manifest。

提交顺序：

```text
完成所有 frame
→ 写 episode.json.tmp
→ rename episode.json
→ 写 _SUCCESS.tmp
→ rename _SUCCESS
→ 将整个 episode 原子移动到 episodes/
```

异常退出时：

1. 立即施加最大制动。
2. 停止所有 sensor listener。
3. 销毁传感器和 ego Actor。
4. 恢复原始 CARLA world settings。
5. 保留 `.incomplete` 供检查，但正式 loader 忽略它。

第一版不续写中断 episode，因为 CARLA world state 无法仅靠磁盘文件可靠恢复；重新运行
创建新 episode。清理不完整目录必须由显式维护命令完成，采集启动时不得自动删除。

## 11. Episode 状态机

```text
CREATED
→ WARMING_UP
→ READY
→ RECORDING
→ FINALIZING
→ COMMITTED
```

任一阶段异常进入：

```text
ABORTING → INCOMPLETE
```

终止条件：

- `goal_reached`
- collision（默认终止）
- emergency stop（默认终止）
- operator manual stop
- `max_steps=1000`
- sensor timeout
- synchronization error
- write error

`sensor_timeout`、`synchronization_error` 和 `write_error` 不产生 `_SUCCESS`，对应 episode
不能用于正常训练。collision、goal、manual stop、max steps 和人工 emergency stop 是有效
终止，可以提交；训练器依据 terminal reason 决定用途。

## 12. 配置设计

`configs/system.yaml` 增加严格顶层节点：

```yaml
recording:
  schema_version: "new-orad-carla-v1"
  calibration_version: "carla-default-v1"

  simulation:
    map: "Town10HD_Opt"
    weather: "ClearNoon"
    warmup_frames: 10
    sensor_timeout_seconds: 2.0
    random_seed: 42

  vehicle:
    blueprint: "vehicle.lincoln.mkz_2020"
    role_name: "hero"
    spawn_point_index: 0

  cameras:
    front:
      location: [1.5, 0.0, 1.6]
      rotation_deg: [0.0, 0.0, 0.0]
      fov_deg: 90.0
    rear:
      location: [-1.5, 0.0, 1.6]
      rotation_deg: [0.0, 180.0, 0.0]
      fov_deg: 90.0
    top:
      location: [0.0, 0.0, 1.9]
      rotation_deg: [-15.0, 0.0, 0.0]
      fov_deg: 100.0

  lidar:
    location: [0.0, 0.0, 1.9]
    rotation_deg: [0.0, 0.0, 0.0]
    channels: 32
    range_m: 50.0
    points_per_second: 320000
    rotation_frequency_hz: 10.0
    upper_fov_deg: 15.0
    lower_fov_deg: -25.0

  imu:
    location: [0.0, 0.0, 0.5]
    rotation_deg: [0.0, 0.0, 0.0]

  episode:
    max_steps: 1000
    goal_world: [80.0, 0.0, 0.0]
    terminate_on_goal: true
    terminate_on_collision: true
    terminate_on_emergency_stop: true

  human_control:
    steering_rate_per_second: 1.5
    throttle_rate_per_second: 1.0
    brake_rate_per_second: 2.0
    steering_return_rate_per_second: 2.0
    joystick_deadzone: 0.08

  writer:
    image_format: "png"
    lidar_format: "npy"
    save_raw_lidar: true
    save_canonical_lidar: true
    flush_interval_frames: 10
```

以下值不在 recording 节点重复：

- Camera 数量和 `192×192` shape：来自 perception 配置。
- canonical LiDAR 点数 256：来自 `ClosedLoopConfig.num_points`。
- IMU history 10：来自 `ClosedLoopConfig.imu_steps`。
- 控制周期 0.1 s：来自 `ClosedLoopConfig.dt`。
- goal tolerance：来自 `ClosedLoopConfig.goal_tolerance`。
- 最大 sensor age/skew 和量程健康阈值：来自 sensor health 配置。

严格 loader 必须拒绝：

- Camera 数量不是 3 或名称不是 `front/rear/top`。
- Camera shape 与 perception 的 `192×192` 不一致。
- `rotation_frequency_hz != 1 / closed_loop.dt`。
- canonical points 与 `num_points=256` 不一致。
- IMU window 与 `imu_steps=10` 不一致。
- `sensor_timeout_seconds <= closed_loop.dt`。
- goal、安全或时序参数与现有模块冲突。
- 未知字段、缺失字段、非有限值、非法范围和重复传感器名称。

## 13. CLI

```bash
python scripts/collect_carla_dataset.py \
  --host "$CARLA_HOST" \
  --port 2000 \
  --config configs/system.yaml \
  --output datasets/carla \
  --episodes 10 \
  --control-device auto
```

CLI 只开放环境选择：

```text
host
port
config path
output root
episode count
control device: auto | keyboard | joystick
```

CLI 不允许覆盖图像尺寸、LiDAR 参数、控制周期、标定和安全阈值。

## 14. 人工控制

默认键位：

| 操作 | 键位 |
|---|---|
| throttle | `W` / Up |
| brake | `S` / Down |
| steer left/right | `A/D` / Left/Right |
| emergency stop | `Space` |
| reverse toggle | `R` |
| finish episode | `N` |
| quit collector | `Esc` |

键盘动作按 `closed_loop.dt` 进行连续积分，并使用独立转向回正速率；手柄输入应用 deadzone
后映射到连续 steer/throttle/brake。实际轴编号和方向在启动时探测并打印，选择结果写入
episode manifest。无法识别的手柄不允许静默产生全零专家动作，`auto` 模式应明确回退键盘。

## 15. 测试设计

### 15.1 无 CARLA 单元测试

- strict config：缺失、未知字段、非法范围和跨模块冲突全部 fail-fast。
- frame synchronizer：精确匹配、旧帧丢弃、未来帧拒绝、重复帧、timeout、skew。
- transition alignment：`Observation_t + Action_t → Observation_t+1` 无 off-by-one。
- LiDAR：相同 episode/frame seed 输出相同 256 点；补零 metadata 正确。
- IMU：顺序、padding、valid count、frame/timestamp 连续性和单位。
- Camera：BGRA→RGB、shape、dtype、文件名和三相机顺序。
- frame conversion：CARLA y-right 到 New_ORAD y-left；角度/角速度单位。
- calibration：K、4×4 sensor-to-ego、逆矩阵和版本字段。
- control：键盘积分、转向回正、手柄 deadzone、急停最大制动。
- event：collision 聚合、goal distance、terminal reason。
- writer：临时文件、原子 rename、flush、`_SUCCESS` 和 incomplete 隔离。

### 15.2 Fake CARLA 集成测试

- 模拟正常、乱序、重复、冻结和缺失 sensor callback。
- 任一 Camera/LiDAR/IMU 无效时，不提交正常 transition。
- timeout、同步错误、写盘错误和人工退出均施加最大制动。
- 任意异常路径停止 listener、销毁 Actor 并恢复原始 world settings。
- 保存的 sample 能由 replay loader 重建并通过现有 `Observation` 和 health gate。
- raw/applied action 不同的 sample 必须带 intervention reason。

### 15.3 真实 CARLA 冒烟验收

固定：

```text
CARLA 0.9.16
Town10HD_Opt
ClearNoon
fixed_delta_seconds=0.1 s
100 recorded transitions
```

通过条件：

- 三 Camera、LiDAR 与控制 frame mismatch 为 0。
- 跨模态 timestamp skew 不超过现有 `0.05 s`。
- 100 个 transition 均可读，且拥有完整 next frame。
- calibration manifest、episode manifest 和 `_SUCCESS` 存在。
- replay 后 Camera/LiDAR/IMU/ego/action 与原记录一致。
- 人工注入急停、collision 和 Camera timeout，终止原因及最大制动正确。

真实测试必须输出证据文件；测试代码存在或本机安装 CARLA 不等于 Stage 1B 已完成。

### 15.4 质量门槛

```text
新增/修改代码 branch coverage ≥ 95%
fail-safe 分支 coverage = 100%
CPU contract tests 不允许 skip
真实 CARLA 测试可使用 carla marker，但必须明确记录未运行原因
```

## 16. 验收标准

实现只有同时满足以下条件才算完成：

1. 严格 loader 能从唯一 `configs/system.yaml` 装配 recorder，拒绝未知/冲突配置。
2. CPU 测试不安装 CARLA/Pygame 也能导入核心模块并达到覆盖率门槛。
3. 真实 CARLA 连续 100 transition 无 frame mismatch，timestamp skew 达标。
4. 终端后无遗留 sensor/vehicle Actor，world settings 恢复。
5. 任一 sensor timeout 或写盘错误都触发最大制动且 episode 不带 `_SUCCESS`。
6. loader 可确定性重建 Camera `[3,192,192,3]`、LiDAR `[256,4]`、IMU `[10,6]`。
7. 每条 transition 都能明确关联 `O_t`、`A_t`、`O_{t+1}`。
8. raw/action applied、安全干预和 terminal reason 均可追溯。
9. 数据集记录 CARLA、配置、Git、标定和随机种子 lineage。

## 17. 实施顺序

```text
1. 配置 dataclass + strict loader 测试
2. numpy-only schema/frame/canonical helpers 测试
3. atomic EpisodeWriter + replay loader 测试
4. HumanControlSource 测试
5. Fake CARLA synchronizer/cleanup 集成测试
6. CLI 装配
7. 真实 CARLA 100-frame smoke test
8. fault injection 和证据归档
```

实现阶段必须遵循 TDD：先提交失败测试，再写最小实现，最后重构。不能先实现全部 CARLA
逻辑再补 mock 测试。

## 18. 风险与控制

| 风险 | 后果 | 控制措施 |
|---|---|---|
| frame/action 偏移一帧 | RSSM 学到错误动力学 | transition test + next_frame 显式字段 |
| CARLA/New_ORAD y 轴混用 | BEV 与轨迹左右颠倒 | raw/canonical 分帧保存 + frame conversion test |
| 先裁剪 LiDAR 再健康检查 | 稀疏/异常被掩盖 | raw health gate 先于 canonicalization |
| 键盘动作过于离散 | BC 标签不平滑 | 连续积分、回正速率、优先支持手柄 |
| 写盘阻塞导致 callback 堆积 | timeout/掉帧 | 队列与 writer 解耦、监控耗时、fail-fast |
| 崩溃留下伪完整数据 | 训练摄入损坏 episode | `.incomplete` + `_SUCCESS` 门禁 |
| 数据量增长过快 | 磁盘耗尽 | 启动前检查空间、每 episode 统计字节、低空间终止 |
| 人工急停被误作专家动作 | BC 学到错误监督 | 同时保存 raw/applied，并标记 intervention |

磁盘不足属于写入故障：采集器必须在开始 episode 前检查可用空间，在运行期间按固定间隔
复查；不足时最大制动并以 `write_error` 中止，不得继续生成半帧。

## 19. 后续阶段

本设计完成后，下一阶段是 Stage 2 recorded replay M0：

- 采集不少于 1000 个同步 transition。
- 建立 hazard-boundary 子集和数据 split manifest。
- 用 replay 数据校准 sensor health confusion matrix。
- 建立 occupancy IoU/FN 和 BC ADE/FDE 基线。
- 将 metrics 与 Git/config/data/checkpoint hash 绑定。

在 recorded replay 和 BC 基线稳定以前，不扩大 Dreamer/RL 训练范围，也不把
learned/fused occupancy 或 affordance 从默认关闭状态提升为正式主链。

