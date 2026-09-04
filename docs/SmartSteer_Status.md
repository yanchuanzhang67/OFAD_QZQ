# SmartSteer / New_ORAD 当前项目状态

> 状态日期：2026-09-04
> 事实来源：当前工作树、`configs/system.yaml`、系统 SDD、工程路线图、Stage 1A/1B
> 记录，以及 `datasets/carla_initial` 的真实 CARLA episode。成熟度只使用
> “代码骨架 / 单元验证 / 离线验证 / 仿真闭环验证 / 车辆验证”；CPU Mock/Fake、
> Traffic Manager smoke、真实模型闭环和车辆验收严格区分。

## 0. 总体结论

New_ORAD 已形成边界较清晰的越野自动驾驶研究原型：

```text
Camera / LiDAR / IMU
        │
        ▼
SensorHealthGate
        ├─ invalid ─► reason metrics ─► EMERGENCY_STOP ─► maximum brake
        │
        ▼ valid
Observation ─► BEVFusion ─► HybridPolicy ─► Safety ─► Pure Pursuit
                    │                              │
                    └─ Occupancy ─────────────────┘
                                                   │
                                                   ▼
                                         CARLA / ROS 2 / deployment
```

当前最成熟的是公共数据契约、严格配置、传感器健康门禁、BEV 输入边界、安全监督和
CPU FakeRunner。Stage 1A 已达到 **离线验证**。CARLA Canonical 已统一为 `0.9.16`，
collector、在线 SensorStack 和 recorded replay 均从 `configs/system.yaml` 读取同一套
车辆、位姿与采样参数。固定基线与正常流工具仍为 **单元验证**：尚未完成 3 次真实
CARLA 固定基线、1000+ normal HealthGate ticks 或在线故障注入，因此 Stage 1B
仍未关闭。

2026-09-03 的 recorded-data 网络 smoke 达到 **离线验证**。不可覆盖产物
[replay_20260903T035626Z_9ca5f754](../artifacts/carla_replay/replay_20260903T035626Z_9ca5f754/summary.json)
包含 200 帧：199 帧通过 HealthGate 并进入 BEVFusion/HybridPolicy，1 帧在模型前因
启动期 `imu_accel_range + imu_jump` 被拒绝。observed rejection rate 为 `0.005`，
network-boundary finite rate 为 `1.0`；health/model/end-to-end latency p50/p95 分别
为 `6.05/7.11`、`10.15/14.74`、`17.68/23.52 ms`。

本次网络使用固定 seed 的随机初始化权重，仅证明 shape/finite/安全控制数据流；
`model_performance_valid=false` 且 `closed_loop_acceptance_valid=false`。历史 episode
没有健康真值，也未记录 vehicle/config/calibration hash，因此不能将 `0.5%` 称为
false rejection rate，不能把当前 Canonical 车辆反推为历史实车事实，也不能替代
Stage 1B/Stage 2 的 1000 样本 Gate。当前测试证据见第 6 节。

### 0.1 总体进度统计

项目进度必须同时看“软件链路是否连通”和“正式工程 Gate 是否关闭”，两者不能混为
一个模糊百分比：

| 统计维度 | 当前结果 | 含义 |
|---|---:|---|
| Recorded 软件链路连通度 | `6/6` 边界已打通 | HealthGate、Observation、BEV、Policy、安全层、控制器均实际执行 |
| 正式里程碑 Gate | `2/8` 已关闭（25%） | 已关闭严格配置/公共契约和 Stage 1A CPU 健康门禁；其余需要真实环境、数据或模型证据 |
| Stage 2 最低样本数量 | `200/1000`（20%） | 只表示数量下限进度，不代表标签、场景覆盖、split 或质量完成度 |
| 当前最高端到端成熟度 | **离线验证** | 真实 recorded sensor 数据 + 随机初始化网络；不是训练模型性能，也不是 CARLA 闭环 |
| 真实模型闭环 | `0` 个正式 episode | 尚无有效 checkpoint 和冻结场景验收 |
| 车辆/SIL/HIL | `0` 项关闭 | ROS 2、TensorRT、台架和车辆级 Gate 均未完成 |

这里的 8 个里程碑依次为：严格配置/公共契约、Stage 1A CPU HealthGate、Stage 1B
Task 1 三次真实基线、Stage 1B Task 2 真实 1000+ ticks、Stage 2 M0 数据、训练模型
离线指标、真实 CARLA 模型闭环、部署/SIL/HIL/车辆。`25%` 是 Gate 数量比，不是
工时或产品可用度估算。

### 0.2 Pipeline 当前到达程度

- 数据与接口层：完整到达控制输出，shape、finite、frame/timestamp、标定和
  fail-safe 路径已有自动化证据。
- 感知与策略层：网络结构能够前向运行，但随机权重只能证明连接；没有 occupancy、
  ADE/FDE 或驾驶性能结论。
- 仿真层：真实 CARLA 数据采集已有 200 帧证据，但本机当前没有 `carla` Python
  模块，三次 baseline、1000+ 在线健康流和训练模型闭环均未运行。
- 部署层：ONNX 导出达到单元验证；ORT/TensorRT、完整 ROS 2、SIL/HIL 和车辆仍是
  代码骨架或未接入状态。

## 1. 已完成内容

### 1.1 公共契约与唯一配置

- `Observation` 明确 Camera/LiDAR/IMU、ego state、frame、timestamp、shape 和同步
  关系，拒绝空、错形、非有限或不同步输入。
- `Trajectory`、`VehicleState`、`OccupancyGrid`、`BEVFeature` 等跨模块类型集中在
  `utils`，并显式区分 `ego/world` frame。
- [system.yaml](../configs/system.yaml) 是正式运行唯一配置源；严格 loader 拒绝
  未知、缺失、越界及跨模块不一致值。
- perception、policy、safety、controller、closed-loop、affordance、domain
  randomization、sensor health 和 CARLA baseline 由同一 stack 装配。

成熟度：**单元验证**；2026-09-04 `configuration/system.py` branch coverage 为
`92%`。

### 1.2 Stage 1A SensorHealthGate

- [sensor_health.py](../src/utils/sensor_health.py) 提供统一 config、稳定原因枚举、
  report 和状态化 gate。
- Camera 检查 count/shape/dtype/finite、黑屏、过曝、冻结、frame 和 timestamp。
- LiDAR 在 padding 前检查 shape/finite/empty、点数、有效比例、范围、重复点、重复
  frame 和 timestamp。
- IMU 检查 `(T,6)`、窗口完整性、frame/timestamp 连续性、频率、量程和跳变。
- 标定版本、sensor age 和跨模态 skew 纳入门禁；任一模态无效都会绕过模型并
  进入最大制动。
- 1000-case CPU 故障矩阵错误接受 `0`、最大制动率 `100%`，历史记录 p95
  `0.773 ms`。

成熟度：**离线验证**。CPU 证据未替代真实 CARLA 故障注入和执行器验收。

### 1.3 Stage 1B 工具与真实 CARLA 初始采集

- [carla_baseline.py](../src/sim/carla_baseline.py) 与
  [verify_carla_baseline.py](../scripts/verify_carla_baseline.py) 可严格校验 CARLA
  版本、地图、车辆、控制周期和传感器属性，并生成 commit/config/environment
  manifest 与 3 次 frame/timestamp trace。
- [carla_sensor_health.py](../src/sim/carla_sensor_health.py) 与
  [evaluate_carla_sensor_health.py](../scripts/evaluate_carla_sensor_health.py) 可在
  不加载 perception/policy 的条件下统计 1000+ normal ticks 的 false rejection、
  age/skew、连续性和 latency。
- CARLA reference timestamp 已改用同 tick world snapshot 的 `elapsed_seconds`，
  避免非零 frame 起点造成错误 stale。
- 新增初始采集器和校验器，可保存 3 路 RGB、raw/canonical LiDAR、10×6 IMU、
  ego state、Traffic Manager action、collision 和 calibration。
- 初始采集器已改为只接受 `--config` 中的物理参数；车辆、传感器位姿/FOV/频率、
  fixed delta 和 seed 全部来自严格 stack。配置的 blueprint 不存在时直接失败，不再
  回退到任意车辆。新 episode 会记录 byte/canonical config hash、calibration hash
  和车辆 blueprint。
- 真实 episode
  [episode_20260902T110145Z](../datasets/carla_initial/episodes/episode_20260902T110145Z/episode.json)
  已采集并通过结构验证：CARLA client/server `0.9.16`、`Town10HD_Opt`、200 帧、
  3×200 PNG、200 raw LiDAR、200 canonical LiDAR、collision `0`，数据约 `87 MB`。

成熟度：固定 baseline/HealthGate 工具为 **单元验证**。初始数据是 CARLA 运行证据，
但只代表 Traffic Manager 驱动的数据采集，不代表 New_ORAD 模型或控制闭环通过。

### 1.4 Recorded CARLA health-first replay

- [carla_dataset.py](../src/replay/carla_dataset.py) 只读加载带 `_SUCCESS` 的 episode，
  校验 schema、已声明 Canonical、frame/timestamp、路径边界、传感器 shape/finite、
  calibration 版本/坐标轴/Camera 外参，并显式暴露历史 provenance gaps。
- [carla_pipeline.py](../src/replay/carla_pipeline.py) 先以 raw LiDAR 执行 HealthGate；
  只有 valid frame 才构造 `Observation`，使用 canonical `(256,4)` LiDAR 调用模型。
- 输入边界固定为 Camera `[1,3,3,192,192]`、LiDAR `[1,256,4]`、IMU
  `[1,10,6]` 和 mask `[[true,true]]`。拒绝帧保证 perception/policy 零调用并记录
  最大制动。
- checkpoint 模式要求 perception/policy 两个权重同时存在并 strict load；随机模式
  必须显式 opt-in，固定种子且永久标记为非性能、非闭环证据。
- [artifacts.py](../src/replay/artifacts.py) 使用 `.incomplete`、exclusive create、
  `_SUCCESS` 最后写入和原子 rename，产物记录 code/config/data/calibration/model
  provenance 与每帧健康、网络、安全、控制结果。

成熟度：**离线验证**，仅限 recorded-data 软件链路。

### 1.5 多模态 BEV 感知

- Camera 与 LiDAR 独立编码到统一 `(B,C,H=y,W=x)` BEV，IMU 作为动力学条件。
- 在线 `(B,2)` mask 顺序固定为 `[camera_valid, lidar_valid]`，只允许两项均真；
  不允许以单模态或补零绕过坏帧门禁。
- BEVFusion 提供 occupancy head；安全层支持显式 `lidar/learned/fused` 路由，默认
  保持可解释的 `lidar` 基线。
- Camera/LiDAR/IMU/batch/attitude 的 shape、空输入和 NaN/Inf 均 fail-fast。

成熟度：**单元验证**。缺 occupancy 真值、IoU/false-negative、真实标定误差和
跨地形泛化指标。

### 1.6 策略、世界模型与辅助任务

- `HybridPolicy` 已有分量加权 BC、RSSM latent dynamics、actor/critic、world-model
  loss/statistics 和 5-step imagination 骨架。
- 输出契约为 ego-frame `[x,y,heading,velocity]`，Runner 负责转换给 world consumer。
- eval latent 使用确定性统计，降低同输入随机漂移。
- `TerrainAffordanceHead` 提供 traversability/roughness 双头、masked loss 与 IMU
  弱标签助手；默认关闭。
- 越野 reward 包含进度、碰撞、姿态和平滑性等项。

成熟度：**单元验证**。当前真实 episode 的 action 来源是 Traffic
Manager 且 `expert_labels=false`，不能直接作为正式 BC 专家标签。

### 1.7 安全、控制与部署骨架

- `SafetySupervisor` 提供 `NORMAL / DEGRADED / EMERGENCY_STOP`；检查 trajectory
  frame/shape/finite/速度、sensor timing 和 model latency。
- `SafetyFilter` 使用 kinematic bicycle 重积分、转向/转向速率/加速度/横向加速度
  约束和 occupancy 碰撞截断。
- Pure Pursuit 输出 Ackermann steering/acceleration；空轨迹和异常路径最大制动。
- CPU FakeRunner 已串通 sensor→health→Observation→BEV→policy→safety→control，
  timeout、模型异常、缺失 occupancy 和坏帧均有 fail-safe 回归。
- perception/policy ONNX 导出与 checker 已有；ROS 2 有 Pure Pursuit/Ackermann
  节点代码；TensorRT 仅有 C++ API 框架。

成熟度：安全与 CPU 编排为 **离线验证**；ONNX 为 **单元验证**；
ROS 2/TensorRT 为 **代码骨架**；无 SIL/HIL 或车辆验证。

## 2. 当前代码架构

### 2.1 模块与依赖方向

```text
utils ───────────────► perception ──► affordance
  │                         │
  ├──────────────► policy / embedded RSSM
  ├──────────────► safety
  ├──────────────► sim
  ├──────────────► orad_ros2
  └──────────────► deployment

configuration ───────► full runtime stack assembly
scripts ─────────────► CARLA collection/evaluation and composition
datasets ────────────► immutable recorded episodes (not runtime source)
replay ──────────────► read-only dataset→health→model→safety/control
```

### 2.2 完整目录与运行职责

```text
New_ORAD/
├── configs/system.yaml              唯一运行配置与 Canonical CARLA 参数
├── src/
│   ├── configuration/system.py      严格 schema、跨模块一致性、全栈装配
│   ├── utils/                       公共类型、坐标系、Observation、HealthGate
│   ├── perception/                  Camera/LiDAR/IMU encoder、BEVFusion、occupancy
│   ├── affordance/                  traversability/roughness 可选辅助头
│   ├── policy/                      BC、RSSM world model、actor/critic、reward
│   ├── world_model/                 目标独立边界；当前实现仍主要位于 policy
│   ├── safety/                      Supervisor、bicycle model、kinematic filter
│   ├── sim/                         CARLA baseline、sensor stack、health/closed-loop
│   ├── replay/                      只读 dataset、health-first pipeline、artifact
│   ├── orad_ros2/                   Pure Pursuit、Ackermann/ROS 2 adapter
│   └── deployment/                  PyTorch→ONNX 导出与检查
├── scripts/                         baseline、采集、校验、健康、replay、闭环 CLI
├── datasets/carla_initial/          不可变历史 CARLA episode
├── artifacts/carla_replay/          不可覆盖的离线回放证据
├── tests/                            unit、integration、sim、closed_loop 分层测试
└── docs/                             SDD、架构、路线、设计、计划与状态证据
```

| 模块 | 当前职责 | 主要入口 |
|---|---|---|
| `utils` | 公共 schema、frame、几何、SensorHealthGate | `schema.py`、`types.py`、`sensor_health.py` |
| `configuration` | 严格读取唯一 YAML 并装配全栈 | `load_system_stack()` |
| `perception` | Camera/LiDAR/IMU encoder、BEV fusion、occupancy | `BEVFusion` |
| `affordance` | traversability/roughness 辅助头 | `TerrainAffordanceHead` |
| `policy` | BC、RSSM、actor/critic、reward | `HybridPolicy` |
| `world_model` | 目标独立包 | 当前为空，RSSM 仍位于 `policy` |
| `safety` | supervisor、运动学约束、碰撞过滤 | `SafetySupervisor`、`SafetyFilter` |
| `sim` | CARLA baseline、sensor stack、health metrics、closed-loop、DR | `CarlaSensorStack`、`CarlaClosedLoopRunner` |
| `replay` | 只读 episode 契约、HealthGate-first 网络回放、事务化证据 | `CarlaRecordedEpisode`、`CarlaReplayPipeline` |
| `orad_ros2` | Pure Pursuit 和 ROS 2 adapter | `PurePursuitController` |
| `deployment` | PyTorch→ONNX | `onnx_export.py` |
| `scripts` | 采集、数据验证、健康评估、闭环 CLI | `collect/evaluate/verify` scripts |
| `datasets` | 当前真实 CARLA 初始 episode | `carla_initial/episodes` |

主运行安全边界仍是：raw sensor 先经过 HealthGate，之后才允许构造 Observation 和
调用模型；SafetySupervisor/SafetyFilter 位于策略和控制之间。collector、在线
SensorStack 和 replay 已复用 `load_system_stack()`；数据集仍是不可变输入，不是
运行配置源。

### 2.3 已打通的数据链路

| 链路 | 状态 | 证据边界 |
|---|---|---|
| synthetic sensors→BEV→policy→safety→control | 已打通 | CPU integration |
| synthetic fault→HealthGate→maximum brake | 已打通 | 1000-case offline verified |
| real CARLA→RGB/LiDAR/IMU/ego/action→disk | 已打通 | 200-frame TM smoke episode |
| recorded initial episode→结构校验 | 已打通 | 200/200 files/frames verified |
| recorded initial episode→当前 HealthGate | 已打通 | 199/200 valid；1 个启动期 IMU reject |
| fixed CARLA baseline×3 | 入口存在 | 尚无真实 `reproducible=true` artifact |
| real CARLA normal→HealthGate×1000+ | 入口存在 | 尚无真实 summary |
| recorded replay→Observation→BEV→policy→safety→control | 已打通 | 200-frame random-model offline smoke；非性能/闭环证据 |
| real CARLA→trained model→closed-loop | 未打通 | 无有效 checkpoint/formal episode |
| PyTorch→ONNX | 基础入口存在 | 无 ORT/TRT 数值和时延对齐 |
| ROS 2→SIL/HIL→vehicle | 局部代码存在 | 无完整节点图和台架证据 |

## 3. 关键参数

正式运行参数来自 [configs/system.yaml](../configs/system.yaml)。collector、在线
SensorStack 和 replay 不提供覆盖物理参数的 CLI；命令行只选择配置、数据、模型和
输出位置。

### 3.1 Canonical 感知、健康与车辆参数

| 参数 | 当前值 |
|---|---:|
| BEV X/Y range | `[-12.5,12.5] m` |
| BEV resolution / shape | `0.5 m/cell` / `50×50` |
| BEV channels | `32` |
| Camera / image | `3` / `192×192` |
| Camera poses `(x,y,z / roll,pitch,yaw)` | front `1.5,0,1.6 / 0,0,0`；rear `-1.5,0,1.6 / 0,0,180°`；top `0,0,1.9 / 0,-15°,0` |
| LiDAR canonical points | `256` |
| LiDAR pose / vertical FOV | `0,0,1.9 / 0,0,0`；`-25°..15°` |
| IMU history | `10×6` |
| IMU pose | `0,0,0.5 / 0,0,0` |
| Calibration version | `carla-default-v1` |
| Max sensor age / skew | `0.20 s / 0.05 s` |
| Max IMU sample gap | `0.11 s` |
| Camera black/saturation ratio | `0.98 / 0.98` |
| LiDAR min points / valid ratio | `16 / 0.80` |
| Accel/gyro max | `50 m/s² / 5 rad/s` |
| Accel/gyro step delta max | `20 m/s² / 2 rad/s` |
| Control period | `0.1 s` |
| Wheelbase / max speed | `2.5 m / 20 m/s` |
| Max steering/rate | `0.5 rad / 0.5 rad/s` |
| Max accel/decel/lateral accel | `+3 / -5 / 4 m/s²` |
| Policy frame / occupancy source | `ego / lidar` |
| Imagination horizon | `5` |
| BC XY/heading/speed/smooth weights | `1.0/0.2/0.2/0.05` |
| Affordance enabled | `false` |

### 3.2 Canonical CARLA baseline 与实际初始采集

| 项目 | Canonical baseline | 实际 episode |
|---|---|---|
| CARLA | `0.9.16` | `0.9.16` |
| Map | `Town10HD_Opt` | `Town10HD_Opt` |
| Vehicle | `vehicle.lincoln.mkz_2020` | manifest 未记录（legacy gap） |
| Seed | `42` | `42` |
| Fixed delta | `0.1 s` | `0.1 s` |
| Camera FOV | front/rear `90°`，top `100°` | front/rear `90°`，top `100°` |
| LiDAR channels/range | `32 / 50 m` | `32 / 50 m` |
| LiDAR points/s | `320000` | calibration 未单独记录；采集代码为 `320000` |
| LiDAR rotation | `10 Hz` | calibration 未单独记录；采集代码为 `10 Hz` |
| Samples | Task 2 要求 `>=1000` | `200` |
| Action source | 固定基线工具 | `traffic_manager_smoke`，非专家 |

当前 `system.yaml` byte SHA-256 为
`3c61c5e9da1773200c58346b7770f88d58a73f378d10ec7b47d2c7c1fa3a761c`，
serialization-stable canonical SHA-256 为
`9786122b446adc2ef5580bea772f190b5e7195cd3106b43c2f83575af6f001d9`。byte hash
用于精确文件追溯；canonical hash 用于判断不受 YAML 排版影响的配置语义一致性。

Domain randomization 配置仍为 friction `0.4–1.4`、mass `1500–1900 kg`、LiDAR
sigma `0–0.1 m`、dropout `0–0.05`、suspension damping `400–1200`；只有采样与
CPU 单测，没有真实物理注入验收。

### 3.3 数据流链路与边界

```text
CARLA online 或 recorded episode
  → raw Camera ×3 / raw LiDAR / IMU history / ego state / timing
  → SensorHealthGate
      ├─ invalid: reason metrics → 跳过 Observation/BEV/Policy
      │           → EMERGENCY_STOP trajectory → maximum brake
      └─ valid:   canonicalize LiDAR → Observation(frame="ego")
                  → tensors + modality mask [[true,true]]
                  → BEVFusion → BEVFeature + occupancy
                  → HybridPolicy → trajectory [x,y,heading,velocity]
                  → SafetySupervisor → SafetyFilter
                  → PurePursuitController → steering/speed/accel
                  → offline record；未来才允许映射为 CARLA/ROS 2 command
```

| 阶段 | 输入 | 输出/固定 shape | 失败行为 |
|---|---|---|---|
| HealthGate | 3×HWC uint8、raw `(N,4)`、IMU `(10,6)`、时序 | `SensorHealthReport` | 无效原因 + 最大制动，模型零调用 |
| Observation | 已通过健康门禁的数据 | ego frame、同步 timestamp、LiDAR `(256,4)` | 构造失败即 fail-safe |
| BEVFusion | Camera `[1,3,3,192,192]`、LiDAR `[1,256,4]`、IMU `[1,10,6]` | BEV `[1,32,50,50]` + occupancy | shape/non-finite 异常进入最大制动 |
| HybridPolicy | BEV + vehicle state | trajectory `[1,20,4]` | strict exception/non-finite fail-safe |
| Safety | trajectory + occupancy + timing | safe/emergency trajectory | 超速、超时、碰撞风险或非法轨迹被拒绝/截断 |
| Control | 过滤后轨迹 + ego state | steering、speed、accel | 空/异常轨迹输出 `-5 m/s²` 最大减速度 |

在线与 recorded 路径共享相同配置、Observation、模型和安全契约。差别是 recorded
replay 只写离线 command，不连接 CARLA 或车辆执行器；因此它能证明网络数据流，不能
证明车辆实际响应。

## 4. 未解决问题

| 优先级 | 问题 | 当前影响 / 关闭条件 |
|---|---|---|
| P0 | Task 1 未验收 | 缺同一 canonical 配置 3 次真实运行及 `reproducible=true` |
| P0 | Task 2 未验收 | 缺 1000+ normal CARLA ticks 的正式 false-rejection/skew/latency summary |
| P0 | 首帧 IMU 启动瞬态 | 200 帧中 1 帧触发 accel range/jump；需定义 warm-up discard/启动状态，不得直接放宽安全阈值 |
| P0 | Stage 2 M0 数据不足 | 仅 200 帧、单 episode、单城市/天气；无 dataset root manifest、hash、split、hazard/occupancy/health 真值 |
| P0 | 当前 action 不是专家标签 | `expert_labels=false`，不能关闭 BC 数据 Gate |
| P0 | 历史 provenance 不完整 | episode 未记录 vehicle/config/calibration hash；只能作为显式 legacy smoke，不能反推或补写采集事实 |
| P0 | 工作树未形成稳定基线 | HEAD 仍是 `99fc0ea` 且 dirty；本轮产物如实记录该状态，正式实验仍需 clean commit |
| P0 | 缺有效 perception/policy checkpoint | 无法进行正式模型 CARLA 闭环 |
| P1 | Perception/Affordance 无真实指标 | 缺 occupancy IoU/FN、traversability F1/IoU、roughness MAE；affordance 保持关闭 |
| P1 | BC/World Model 流水线缺失 | 无正式 dataloader、train/eval CLI、checkpoint lineage、ADE/FDE 或 RL 收敛证据 |
| P1 | World Model 包边界未闭合 | `world_model` 为空，RSSM 仍嵌在 `policy`；迁移需 checkpoint compatibility |
| P1 | 真实 CARLA/Gazebo 闭环缺失 | 无冻结 scenario matrix、正式模型 30 episodes、坡地/低摩擦/故障注入证据 |
| P1 | ORT/TensorRT 未对齐 | 缺 ORT/TRT runtime、FP32/FP16 数值回归与目标硬件延迟 |
| P1 | ROS 2/SIL/HIL 未打通 | 缺强类型消息、launch/node graph、watchdog、完整依赖和台架验收 |
| P2 | CasADi NMPC 未实现 | 当前为 deterministic kinematic projection，不应声明 NMPC 已完成 |
| P2 | 数据存储与许可未定稿 | 原始数据当前位于工作树且约 87 MB；需明确外部数据根、保留/清理和 provenance 策略，禁止覆盖现有 episode |

ONNX exporter/NVML 相关 warning 当前为 16 条；真实 CARLA/Gazebo、recorded
expert/hazard、CUDA 和完整 ROS 2 环境路径的 skip 不能计为已完成。

## 5. 下一步建议

### P0：先形成可追溯基线并关闭真实 CARLA Gate

1. 先整理当前 dirty `main`：按配置/仿真、replay、测试、文档划分原子提交，形成可供
   实验引用的 clean commit；不要把现有原始数据或历史 artifact 混入代码提交。
2. 保留现有 episode 为不可变的 legacy smoke evidence，不覆盖、不补写历史 manifest；
   后续判断继续引用当前 episode/calibration hash 与明确 provenance gaps。
3. 安装与 server 完全匹配的 CARLA 0.9.16 PythonAPI，并确认
   `Town10HD_Opt + vehicle.lincoln.mkz_2020` 可用。当前环境 `import carla` 失败，
   这是执行 Task 1/2 的直接阻塞项。
4. 关闭 Task 1：在 clean commit 上以同一配置运行 3 次，比较 resolved sensor
   attributes、frame/timestamp 增量并要求 `reproducible=true`。
5. 关闭 Task 2：20+ warm-up ticks 后采集至少 1000 normal ticks，要求 false
   rejection `0`、连续性错误 `0`、skew max `<0.05 s`、age max `<0.20 s`、health
   p95 `<100 ms`。启动瞬态应通过显式 warm-up 状态处理，而不是放宽运行阈值。
6. 使用配置驱动 collector 新采至少一个完整 provenance episode，确认 manifest 的
   vehicle、byte/canonical config hash 和 calibration hash 均可在 replay 严格模式
   复验，再扩大数据规模。

### P0/P1：把初始数据升级为 Stage 2 replay M0

7. 在现有只读 loader/runner 上补 versioned dataset root 与 split manifest；新数据的
   所有指标绑定 code/config/data/calibration/model hash。
8. 扩展到不少于 1000 个同步样本和多个 episode，建立 normal/degraded/emergency
   及 hazard-boundary 子集；冻结 test split，禁止相邻帧随机泄漏。
9. 将当前 200 帧作为 loader/Observation/HealthGate 冒烟数据，不作为专家训练集；
   另行采集或标注真正的专家 trajectory/control 标签。
10. 在 replay 上形成 health confusion matrix、occupancy IoU/FN、policy ADE/FDE 和
   safety hazard interception 指标后，才关闭 Stage 2。

### P1/P2：学习、闭环与部署

11. 先建立最小 `BEV→BC trajectory` dataloader、train/eval CLI、3-seed checkpoint
    lineage 和 ADE/FDE 基线，再扩大 RSSM/Dreamer/RL。
    2026-09-04 已确定递进式架构为 B0 `Pure BC` → B1 `Affordance+BC` → B2
    `Affordance+RSSM+BC` → B3 `BC-initialized Dreamer`；当前只完成架构设计确认，
    尚未实现新的 B0，也未改变 Stage 3 成熟度。设计见
    `superpowers/specs/2026-09-04-staged-policy-learning-b0-design.md`。
12. 用正式 checkpoint 重新运行 recorded replay，先确认严格加载、finite、ADE/FDE、
    occupancy 和安全指标，再允许连接 simulator。
13. 使用正式 checkpoint 和冻结 scenario manifest 完成至少 30 个 CARLA episodes，
    达到 success `>90%`、collision/姿态报警 `0`、post-safety risk `0`。
14. 依次完成 PyTorch→ORT→TensorRT FP32/FP16 parity、ROS 2 SIL、HIL 和独立车辆
    安全 Gate；在此之前不进入 INT8 或开放道路测试。

## 6. 2026-09-04 验证结果

- `python -m flake8 src tests scripts --max-line-length=99`：通过；
- `git diff --check`：通过；
- `python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-branch`：
  `275 passed / 2 skipped / 16 warnings`，branch coverage `87.93%`；
- `python -m pytest -ra`：`292 passed / 8 skipped / 16 warnings`；
- 8 项环境 skip：真实 CARLA 20° 坡地、Gazebo suspension、CARLA 0.9.16 Task 1、
  CARLA 0.9.16 Task 2、专家日志、hazard-boundary 数据、CUDA、完整 ROS 2 stack；
- 数据集只读校验：200 帧，frame `1044974..1045173`，最大 timestamp skew
  `0.000000 s`；完整网络 smoke summary 的 `_SUCCESS` 与 200 条 frame record 均存在。
- `python -c "import carla"`：`ModuleNotFoundError`；因此没有执行或宣称真实 CARLA
  Task 1/2 与模型闭环。

覆盖率首次组合运行被执行时间上限中断并留下 0 字节 `.coverage`。确认根因后使用
独立 `/tmp` coverage 数据文件重跑成功，并通过 `python -m coverage erase` 清理该
临时仓库状态；这不是源代码测试失败。

## 7. 相关文档与证据

- [项目入口](../README.md)
- [系统 SDD](../System_overview.md)
- [技术原理与代码架构](./技术原理与代码架构.md)
- [工程执行路线](./ENGINEERING_EXECUTION_ROADMAP.md)
- [Stage 1A 完成记录](./SENSOR_HEALTH_STAGE1A_2026-08-31.md)
- [Stage 1B Task 1 固定基线](./CARLA_STAGE1B_TASK1_BASELINE_2026-08-31.md)
- [Stage 1B Task 2 Sensor Health](./CARLA_STAGE1B_TASK2_SENSOR_HEALTH_2026-09-01.md)
- [初始 CARLA 采集说明](../CARLA_INITIAL_COLLECTOR_README.md)
- [初始 CARLA 采集设计](./superpowers/2026-09-01-carla-dataset-collection-design.md)
- [CARLA 0.9.16 data-first replay 设计](./superpowers/specs/2026-09-03-carla-0916-data-first-replay-design.md)
- [CARLA 0.9.16 data-first replay 实施记录](./superpowers/plans/2026-09-03-carla-0916-data-first-replay.md)
- [分层策略学习架构与 B0 Pure BC 设计](./superpowers/specs/2026-09-04-staged-policy-learning-b0-design.md)
- [真实 episode manifest](../datasets/carla_initial/episodes/episode_20260902T110145Z/episode.json)
- [真实 calibration](../datasets/carla_initial/episodes/episode_20260902T110145Z/calibration.json)
- [200 帧 replay summary](../artifacts/carla_replay/replay_20260903T035626Z_9ca5f754/summary.json)
