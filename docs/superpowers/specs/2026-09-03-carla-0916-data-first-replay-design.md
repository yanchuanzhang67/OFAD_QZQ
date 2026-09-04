# CARLA 0.9.16 数据优先 Canonical 与网络回放设计

- 日期：2026-09-03
- 状态：设计已于 2026-09-03 书面确认并完成本轮实施与验证
- 当前成熟度：离线验证
- 数据契约：`new-orad-carla-v1`
- 目标 Canonical：CARLA `0.9.16`

## 1. 背景与事实基线

设计启动时，仓库存在两套彼此漂移的 CARLA 参数：

- `configs/system.yaml` 声明 CARLA `0.9.15`、`vehicle.tesla.model3`、seed
  `20260831`、单一 Camera FOV `90` 和 LiDAR `32000 points/s @ 20 Hz`；
- 已完成结构校验的真实 episode 明确记录 CARLA client/server `0.9.16`、seed
  `42` 和三路相机独立标定；采集脚本当时以 Lincoln 与 LiDAR
  `320000 points/s @ 10 Hz` 为目标，但历史 manifest 没有记录 vehicle/config/
  calibration hash，因此这些缺失项不能当作已由 episode provenance 证明。

采集脚本当前还通过 CLI 默认值与硬编码创建车辆和传感器，未完全遵守
`configs/system.yaml` 的唯一配置所有权。初始 episode 只有 200 帧、使用 Traffic
Manager 动作且 `expert_labels=false`，因此可以作为真实数据链路 smoke evidence，
不能作为正式 BC 专家数据或驾驶性能证据。

本设计选择以已落盘且通过校验的真实数据为事实锚点，统一 Canonical 配置，并建立从
recorded CARLA 数据到 HealthGate、Observation、BEV、Policy、安全层和控制器的只读
回放路径。

## 2. 目标与非目标

### 2.1 目标

1. 将 CARLA Canonical 统一为 `0.9.16`，固定地图、车辆、随机种子、控制周期及
   Camera/LiDAR/IMU 参数。
2. 让采集、baseline 校验和 recorded replay 都从 `load_system_stack()` 派生配置，
   未知、缺失或跨模块不一致值快速失败。
3. 保留并只读使用现有 episode；为其生成可追溯的校验和与回放报告，并显式记录历史
   manifest 缺失的 provenance 字段，不覆盖原始数据或既有实验输出。
4. 建立正式 replay loader，在模型调用前执行 SensorHealthGate，并只从有效帧构造
   `Observation`。
5. 使用随机初始化的 `BEVFusion` 与 `HybridPolicy` 进行显式 opt-in 的网络数据流
   smoke test，验证 shape、finite、frame、错误处理和 fail-safe。
6. 保存逐帧 JSONL 和汇总 JSON，使结果可追溯到 git commit、config hash、
   calibration hash、dataset schema、episode 和运行模式。
7. 在每个实现里程碑结束时同步更新过程记录、配置说明、架构文档、路线图和当前状态。

### 2.2 非目标

- 不把随机初始化网络输出解释为感知、轨迹或驾驶质量。
- 不用 Traffic Manager action 关闭 BC 专家数据 Gate。
- 不修改或清洗现有 episode 中的原始 Camera、LiDAR、IMU 文件。
- 不因首帧 IMU 启动瞬态而放宽量程、跳变、age 或 skew 安全阈值。
- 不在本轮宣称 1000+ normal CARLA ticks、3 次 baseline 重复或训练模型闭环已通过；
  这些仍需真实 CARLA 环境证据。
- 不变更 checkpoint 参数键或已发布跨模块数据结构。

## 3. 方案比较与决策

### 3.1 采用：数据优先的分层收敛

按 Canonical 配置、recorded replay、网络 smoke 三层推进。每层都有独立测试与报告，
后一层只能消费前一层验证过的输出。该方案工作量适中，能永久消除配置漂移，并为之后
真实 checkpoint 和 Stage 2 数据 Gate 提供稳定入口。

### 3.2 未采用：仅修改版本号和脚本默认值

该方案变更最小，但无法表达三路相机独立外参与 FOV，也不能防止采集器、校验器和模型
入口再次漂移；一条临时 Python 命令不能形成可复现的 replay 资产。

### 3.3 暂缓：直接扩展为完整 Stage 2 数据/训练平台

一次完成专家采集、split、occupancy/hazard 标注、BC/RSSM 训练和 CARLA 闭环会混合
多个尚未满足的 Gate，失败时难以定位是数据、健康门禁、模型还是控制问题。本轮只建立
这些工作的可靠输入边界。

## 4. Canonical 配置设计

### 4.1 固定值

Canonical 的版本、地图、seed、周期和标定以 episode 的 `episode.json` 与
`calibration.json` 为数据优先事实来源；车辆和未在历史 manifest 中解析记录的
LiDAR runtime 属性是本设计明确固定的项目选择，不反向解释为历史 provenance：

| 项目 | Canonical 值 |
|---|---|
| CARLA client/server | `0.9.16` |
| map | `Town10HD_Opt` |
| vehicle blueprint | `vehicle.lincoln.mkz_2020` |
| synchronous fixed delta / control period | `0.1 s` |
| random seed | `42` |
| cameras | `front`、`rear`、`top`，`192×192` |
| camera FOV | front/rear `90°`，top `100°` |
| top camera pitch | `-15°` |
| LiDAR | 32 channels、50 m、320000 points/s、10 Hz、FOV `-25°..15°` |
| canonical LiDAR | `256×4`，显式 CARLA→New_ORAD Y 轴转换 |
| IMU history | `10×6`，通道 `[ax,ay,az,gx,gy,gz]` |

所有精确传感器 transform 直接提升自现有 calibration，不在多个模块中重新手写。逻辑
标定版本保留稳定字符串，同时 manifest 必须保存 calibration 文件内容的 SHA-256；
逻辑版本相同但内容哈希不同必须判为漂移。

### 4.2 配置结构

`configs/system.yaml` 仍是唯一运行配置源。`carla_baseline` 从单一
`camera_fov_degrees` 扩展为具名传感器结构，至少包括：

```yaml
carla_baseline:
  version: 0.9.16
  map_name: Town10HD_Opt
  vehicle_blueprint: vehicle.lincoln.mkz_2020
  random_seed: 42
  repetitions: 3
  cameras:
    - name: front
      blueprint: sensor.camera.rgb
      transform: {...}
      fov_degrees: 90.0
    - name: rear
      blueprint: sensor.camera.rgb
      transform: {...}
      fov_degrees: 90.0
    - name: top
      blueprint: sensor.camera.rgb
      transform: {...}
      fov_degrees: 100.0
  lidar: {...}
  imu: {...}
```

严格 loader 验证传感器名称唯一且顺序固定为 `[front,rear,top]`，image size、
sensor tick、control dt、LiDAR canonical points 和 IMU history 与 BEV/HealthGate 契约
一致。配置哈希按 canonical serialization 计算，不受 YAML 空白或键顺序影响。

为了保留公共 Python API，现有 `CarlaBaselineConfig` 名称继续有效；内部字段扩展为冻结的
具名 Camera/LiDAR/IMU dataclass。旧 YAML schema 不做静默兼容，必须给出指出旧字段的
清晰 migration error，因为双 schema 并存会破坏唯一 Canonical。

### 4.3 采集器约束

采集器必须先调用 `load_system_stack()`。CLI 仅允许配置文件路径、CARLA host/port、
输出根目录、episode 长度和显式运行模式；车辆、地图、dt、seed 和 sensor attribute
不可由默认 CLI 值静默覆盖。连接后的 client/server 版本、world map 和生成 actor 必须
与 Canonical 完全匹配，否则在写入帧数据前失败。

## 5. Recorded replay 架构

### 5.1 代码边界

计划新增独立只读 package，避免 `src` 反向依赖 `scripts`：

```text
src/replay/carla_dataset.py       # manifest、calibration、frame index、文件读取
src/replay/carla_pipeline.py      # HealthGate→Observation→network→safety/control
scripts/replay_carla_pipeline.py  # CLI 与依赖装配
tests/unit/replay/                # loader、校准、错误分支
tests/integration/                # 真实 episode 和网络数据流 smoke
```

公共跨模块 payload 继续使用 `utils` 中已有 `Observation`、`Trajectory`、
`VehicleState` 等类型。replay package 只负责输入边界和编排，不重新定义模型类型。

### 5.2 Loader 状态机

```text
dataset root
  → 选择显式 episode
  → 要求 _SUCCESS
  → 校验 schema/manifest/calibration，并计算现有文件 hashes
  → 区分 strict provenance 与 legacy provenance-gap 模式
  → 建立 frames.jsonl 顺序索引
  → 校验路径位于 episode 内
  → 只读加载原始 Camera/LiDAR/IMU/ego
  → 产生 RawRecordedFrame
```

默认拒绝不完整 episode、路径逃逸、丢失文件、重复或倒退 frame、非单调 timestamp、
错误 dtype/shape、非有限数据、标定或已声明 Canonical 字段的漂移。任何校验失败都带
episode、frame、sensor 和稳定原因码。

当前 200 帧 episode 的历史 manifest 没有保存实际 `vehicle_blueprint`、采集时
`config_sha256` 或 `calibration_sha256`。这些事实无法从现有文件可靠还原，因此 loader
只能在显式 `legacy provenance-gap` 模式下用于网络 smoke，并把缺失字段写入报告；不得
根据采集器当前源码反推并伪造历史值。新采集 episode 必须包含完整字段，缺失时 strict
模式直接失败。现有 calibration 文件可以在回放时计算内容哈希，但该哈希只能证明当前
读取内容，不能伪装成采集时已保存的 provenance。

### 5.3 启动期与 HealthGate

每一帧必须先进入 `SensorHealthGate`：

```text
RawRecordedFrame
  → health report
  ├─ invalid → 不调用 BEV/policy，生成最大制动 fail-safe 记录
  └─ valid   → 构造 Observation → 下游网络
```

当前首帧极端 IMU sample 预期保持 invalid，并记录 `startup/warmup` 上下文以及原始
`imu_accel_range`、`imu_jump` 原因。实现不能删除原始样本、伪造 IMU history，或放宽
阈值使其通过。后续重新采集应在 episode writer 提交正式帧前完成 10 个有效 IMU sample
的 warm-up；旧 episode 只读回放仍保留原事实。

## 6. 网络数据流 smoke

### 6.1 数据流

```text
recorded Camera/LiDAR/IMU
  → SensorHealthGate
  → Observation
  → calibration-aware BEVFusion
  → HybridPolicy
  → trajectory contract validation
  → SafetySupervisor / SafetyFilter
  → PurePursuit command
  → per-frame JSONL + summary JSON
```

Camera tensor 为 `[1,3,3,192,192]`，LiDAR 为 `[1,256,4]`，IMU 为
`[1,10,6]`，online validity mask 固定为 `[[true,true]]`。实际 calibration 的内外参
必须传入 Camera→BEV 映射；禁止依赖模型模块中的合成默认标定来声称真实数据已接通。

### 6.2 随机权重边界

CLI 默认要求 checkpoint。只有显式传入 `--allow-random-models` 才能构建确定性 seed 的
随机 `BEVFusion`/`HybridPolicy`，并在所有输出 artifact 顶层标注：

```json
{
  "mode": "random-model-network-smoke",
  "model_performance_valid": false,
  "closed_loop_acceptance_valid": false
}
```

随机模式仅验证模块调用、shape、finite、frame 和 fail-safe。checkpoint 模式必须严格
加载全部参数，missing/unexpected key、配置不兼容或非有限权重均失败，不得自动回退到
随机网络。

### 6.3 资源与可重复性

CPU smoke 默认允许通过 `--max-frames` 选取少量有效帧以控制运行时间；正式 recorded
replay 报告可覆盖全部 200 帧。随机种子同时固定 Python、NumPy 和 PyTorch。每次运行
创建新的显式结果目录，不覆盖已有报告。

## 7. 输出与追溯

每个 replay run 至少写入：

```text
artifacts/carla_replay/<run-id>/
  run.json
  frames.jsonl
  summary.json
```

`run.json` 保存 git commit、dirty 状态、当前 canonical config hash、calibration
version/hash、CARLA version、map、vehicle、episode schema/id、model mode、checkpoint
hash、随机种子、provenance gaps 和命令参数。历史 manifest 未记录的值必须为 `null` 或
列入 gaps，不能由当前配置代填成历史事实。`frames.jsonl` 保存 frame/timestamp、health
结果、各阶段 shape/finite、延迟、safety state 和 command。`summary.json` 保存：

- total/valid/rejected/model-processed/fail-safe frame 数；
- rejection reason 分布与 false rejection 的证据边界；
- sensor skew、health latency、BEV/policy/end-to-end latency p50/p95；
- 每个网络边界的 shape/finite 通过率；
- 模型异常、安全干预和最大制动次数；
- `model_performance_valid` 与 `closed_loop_acceptance_valid`。

没有 normal ground-truth 标签时，报告只能称 observed rejection rate，不能无条件称
false rejection rate。当前 episode 可基于其 `traffic_manager_smoke`、无 collision 的
有限元数据标注“normal-candidate”，但该推断必须随报告保留。

## 8. 错误与安全策略

| 故障 | 行为 |
|---|---|
| config/schema/calibration 已声明字段漂移 | 启动失败，不进入 replay |
| legacy manifest 缺 provenance 字段 | 仅显式 legacy smoke 可继续，并在所有报告标红 gaps |
| dataset 不完整或路径非法 | 启动或对应帧失败，不读取 episode 外文件 |
| sensor invalid | 跳过所有模型，记录最大制动 fail-safe |
| Observation 构造失败 | 记录契约错误并最大制动 |
| 模型异常/NaN/shape 错误 | 不向后传播坏值，最大制动并记录阶段 |
| checkpoint 不兼容 | 启动失败，不自动随机降级 |
| 单帧报告写入失败 | 停止运行，保留不完整新目录，不伪造 success |

Recorded replay 不向 CARLA 或车辆发送控制，因此这里的 command 只是离线计算结果。
它能证明软件网络数据流，但不能证明真实执行器或仿真闭环控制。

## 9. TDD 与验收

### 9.1 测试顺序

1. 配置 schema 与 cross-module consistency 的失败测试。
2. baseline/collector 从唯一配置源读取固定参数的失败测试。
3. dataset loader 对 happy path、缺文件、路径逃逸、hash/标定漂移的失败测试。
4. 首帧 HealthGate 拒绝且模型零调用的失败测试。
5. 有效真实帧构造 `Observation` 的契约测试。
6. calibration-aware BEV 与随机 policy smoke 的集成测试。
7. 模型 NaN/异常/checkpoint mismatch 的 fail-safe 测试。
8. artifact provenance、原子完成与不覆盖测试。

### 9.2 本轮完成条件

- `configs/system.yaml`、loader、baseline 工具和 collector 对固定 Canonical 一致；
- 现有 episode 在不修改原数据的条件下，以显式 legacy provenance-gap 模式通过正式
  loader，且报告准确列出 `vehicle_blueprint`、历史 config/calibration hash 缺口；
- 首帧 invalid 时 BEV/policy 调用数为零，产生最大制动记录；
- 至少一个后续真实帧完成 Camera/LiDAR/IMU→Observation→BEV→Policy→Safety→Control；
- 全部处理帧的已验收边界 shape 正确且 finite；
- smoke artifacts 包含 commit/config/calibration/dataset/model mode 追溯信息；
- focused tests、全量 pytest、unit branch coverage、flake8、Markdown link check 和
  `git diff --check` 通过，环境相关 skip 单独列出；
- 文档不得把随机模型 smoke 晋升为模型性能或仿真闭环验证。

### 9.3 仍保留的外部 Gate

- 同一 Canonical 配置真实 CARLA 运行 3 次并获得 `reproducible=true`；
- 1000+ normal CARLA ticks 的正式 HealthGate 指标；
- 专家标签、hazard/occupancy 真值、dataset root manifest 与冻结 split；
- 有效 checkpoint 的 recorded-data 指标和真实 CARLA 模型闭环；
- ORT/TensorRT、ROS 2、SIL/HIL 和车辆验收。

## 10. 文档与过程记录

实现开始前生成对应实施计划。每完成一个里程碑，在计划中记录命令、结果、artifact 和
未关闭项，并同步更新其 owning 文档：

- Canonical/config：`configs/README.md`、Task 1 baseline 记录；
- replay/网络边界：系统 SDD、技术架构与工程路线图；
- 测试数据：测试 README 或对应 TDD 记录；
- 当前结论：`docs/SmartSteer_Status.md` 与 `docs/README.md`。

历史审计和旧实验结论只追加 superseded/current note，不回写为仿佛当时已使用新基线。
由于当前工作树包含多组尚未提交的用户与既有任务变更，本设计不会自行创建混合 commit；
commit provenance 在形成用户确认的稳定提交后补齐。
