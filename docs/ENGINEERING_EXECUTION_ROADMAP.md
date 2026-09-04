# New_ORAD 工程执行路线与阶段门禁

> 路线：传感器健康门禁 → recorded replay 数据与指标 → BC 训练基线 →
> 真实 CARLA 闭环 → ONNX Runtime/TensorRT 对齐 → ROS 2/SIL/HIL  
> 使用方式：本文件是当前执行清单，不是历史完成度报告。每个阶段必须以可复现
> 证据关闭 Exit Gate，不能仅凭代码存在或 Mock 结果进入下一阶段。

## 1. 当前起点

截至 2026-08-31，项目已有以下基础：

- Camera/LiDAR/IMU → BEVFusion → HybridPolicy → Safety → Control 的 CPU
  软件链路和 FakeRunner。
- 严格双模态输入契约：在线 Camera/LiDAR 必须同时有效，坏帧进入最大制动。
- 加权 BC loss、RSSM/actor/critic 代码骨架、CARLA 闭环入口、ONNX 导出入口、
  Pure Pursuit/ROS 2 节点代码和 TensorRT C++ 头文件框架。
- 2026-08-31 自动化基线记录为全量 `168 collected / 162 passed / 6 skipped`，
  unit branch coverage `85.92%`。该数字是本路线的起点引用，不代表下面任何环境
  或数据 Gate 已完成。
- Stage 1A 整改后的当前证据为全量 `238 collected / 232 passed / 6 skipped`，
  unit branch coverage `87.79%`，健康关键模块 `98.04%`。1000 个确定性坏帧
  错误接受为 0、最大制动率 100%、p95 `0.773 ms`。
- 2026-09-01 Task 2 软件变更后的当前回归为全量 `267 collected / 259 passed /
  8 skipped`，unit branch coverage `88.34%`；新增 CARLA Task 1/2 环境用例仍为
  skip，不改变 Stage 1B 未关闭结论。
- 2026-09-03 CARLA Canonical 已统一为 0.9.16，collector/CarlaSensorStack 已统一
  消费具名传感器配置；recorded 200 帧网络 smoke artifact 已完成。本轮全量回归为
  `292 passed / 8 skipped / 16 warnings`；unit 为 `275 passed / 2 skipped /
  16 warnings`，branch coverage `87.93%`。
- 2026-09-04 复核：recorded 软件边界连通度为 `6/6`，但正式里程碑只关闭
  `2/8`；Stage 2 数量为 `200/1000` 且缺标签/split/完整 provenance。本机无
  `carla` Python 模块，当前关键路径仍是 clean commit→真实 Task 1→真实 Task 2→
  完整 provenance M0 数据→BC checkpoint，而不是继续扩展随机模型功能。

当前阶段判断：

| 阶段 | 当前成熟度 | 主要缺口 |
|---|---|---|
| 1. 传感器健康门禁 | 离线验证 | Stage 1A 软件门禁、原因指标和最大制动已关闭；真实 CARLA/ROS 2/C++/执行器为 Stage 1B |
| 2. Recorded replay | 离线验证 | 初始 200 帧网络 smoke 已有；缺 1000+ 数据、冻结 split、专家/hazard/occupancy 标签和正式模型指标 |
| 3. BC 训练基线 | 单元验证 | 无 dataloader、训练入口、有效 checkpoint 和 ADE/FDE 报告 |
| 4. 真实 CARLA 闭环 | 代码骨架 | 缺固定环境、场景清单、正式权重和真实 episode 证据 |
| 5. ORT/TensorRT | 代码骨架 | 缺 ORT/TRT 数值回归、C++ runtime、目标硬件延迟和图外健康门禁 |
| 6. ROS 2/SIL/HIL | 代码骨架 | 缺强类型消息、完整节点图、SIL/HIL 台架和车辆级安全验收 |

## 2. 全阶段共同规则

### 2.1 顺序与停止条件

```text
Stage 1 Gate
  └─► Stage 2 Gate
        └─► Stage 3 Gate
              └─► Stage 4 Gate
                    └─► Stage 5 Gate
                          └─► Stage 6 Gate
```

- 上一阶段 Gate 未关闭时，可以搭建下一阶段的代码骨架，但不得使用下一阶段
  结果作为正式基线。
- 任何阶段出现 frame/schema/finite/synchronization/fail-safe 回退，都必须先回到
  Stage 1 或 Stage 2 定位，不能用训练或阈值调参掩盖接口错误。
- BC 未形成稳定开环基线前，不启用 Dreamer/RL 作为本路线的验收对象。
- CARLA 闭环未达标前，不进行 INT8 优化；FP32/FP16 对齐优先。
- SIL/HIL 通过不等于开放道路可用。封闭场地、车辆功能安全和独立安全审查仍是
  后续单独 Gate。

### 2.2 运行与产物管理

大数据放在仓库外部，例如：

```text
${ORAD_DATA_ROOT}/
├── replay/<dataset_version>/
├── expert/<dataset_version>/
└── calibration/<calibration_version>/
```

实验输出写入已被 `.gitignore` 排除的 `runs/`，每次运行使用独立目录：

```text
runs/<stage>/<run_id>/
├── manifest.json
├── resolved_config.yaml
├── metrics.json
├── logs/
├── plots/
└── checkpoints/        # 仅训练阶段需要
```

每个 `manifest.json` 至少记录：

- Git commit、工作树是否干净、运行命令和 UTC 时间。
- `configs/system.yaml` hash、随机种子和依赖版本。
- 数据集/标定版本及 hash。
- perception/policy checkpoint 路径与 hash。
- simulator、CUDA、ONNX Runtime、TensorRT、ROS 2 和硬件版本（适用时）。
- 通过、失败、跳过项及明确原因。

可长期审计的小型摘要进入 `docs/`；原始数据、模型权重、ONNX、TensorRT engine
和大日志不提交到 Git。

---

## 3. Stage 1 — 传感器健康门禁

### 2026-08-31 执行状态

Stage 1A CPU 软件 Gate 已关闭：统一 report/reason、严格 YAML、Camera/LiDAR/IMU
健康诊断、CARLA Runner 前模型门禁、原因/延迟指标和 1000-case 故障矩阵均已
落地。Stage 1B 环境 Gate 仍待真实 CARLA、ROS 2/C++、执行器和目标 ECU 验收。
详细证据见 `SENSOR_HEALTH_STAGE1A_2026-08-31.md`。

Stage 1B Task 1 的固定实验基线代码已落地：2026-09-03 Canonical 为 CARLA
`0.9.16`、`Town10HD_Opt`、Lincoln MKZ 2020、seed `42`、0.1 s control period，
以及 front/rear/top Camera、LiDAR、IMU 的具名位姿与采样属性；collector、baseline
工具和 `CarlaSensorStack` 均由 canonical YAML 派生。三次运行会输出不可覆盖的
manifest/trace/summary。
CPU 配置与证据契约已单元验证，真实 CARLA 三次一致性尚未执行，因此 Task 1 和
Stage 1B 环境 Gate 均保持未关闭。过程见
`CARLA_STAGE1B_TASK1_BASELINE_2026-08-31.md`。

2026-09-01 Stage 1B Task 2 的无模型 Sensor→HealthGate 入口已落地：20 ticks
warm-up 后采集 1000+ normal ticks，输出逐帧 trace、false rejection、age/skew
分布、连续性和 latency。world snapshot timestamp 修正与 1000-tick CPU metrics
contract 已单元验证；真实 CARLA 尚未运行，因此当前阈值的真实误杀率未知，Task 2
保持环境待验收。过程见 `CARLA_STAGE1B_TASK2_SENSOR_HEALTH_2026-09-01.md`。

### 目标

在任何模型计算前判断 Camera/LiDAR/IMU 数据是否可用于当前控制周期，并确保
Camera 或 LiDAR 任一无效时整帧进入故障安全停车。

### 需要完成的工作

1. 定义统一的传感器健康结果，至少包含：
   - `camera_valid`、`lidar_valid`、`imu_valid`。
   - `frame_id`、采集时间、数据年龄、跨传感器 skew。
   - 无效原因枚举，而不是单个模糊布尔值。
2. Camera 健康检查：
   - 相机数量、分辨率、channel、dtype、finite 和同 frame。
   - 空帧、全黑/全白、饱和像素比例、冻结帧和时间陈旧。
   - 内外参与标定版本必须匹配当前配置。
3. LiDAR 健康检查：
   - shape、finite、非空、点数/有效点比例和时间陈旧。
   - 点坐标合理范围、连续重复帧和明显稀疏/遮挡。
   - 不允许先 pad 成固定长度再把空点云声明为有效。
4. IMU 健康检查：
   - `(T,6)` shape、finite、频率、时间窗口完整性和异常跳变。
   - IMU 无效至少进入 `DEGRADED` 或 `EMERGENCY_STOP`，具体策略必须显式。
5. 将健康结果接入 CARLA、未来 ROS 2/C++ adapter：
   - BEV mask 顺序固定为 `[camera_valid, lidar_valid]`。
   - 在线只允许 `[True, True]`。
   - 失败必须记录原因并输出最大制动，不能继续调用正常 policy。
6. 所有阈值进入 `configs/system.yaml` 及严格 loader，不在 adapter 中硬编码。

### 测试与故障注入

- 单元测试覆盖空输入、错误 shape、NaN/Inf、过期 frame、skew 超限、冻结相机、
  全黑/过曝图像、零点/稀疏点云和 IMU 中断。
- FakeRunner 断言每种 P0 故障都不会调用正常策略，并产生最大制动。
- 批量注入测试至少覆盖 batch 内单样本失效，防止只检查整个 batch 聚合值。

建议命令：

```bash
python -m pytest -q tests/unit/perception tests/unit/utils tests/unit/sim
python -m pytest -q tests/integration/test_m0_interface_gate.py
python -m pytest -ra
```

### 交付物

- 健康状态 schema、原因枚举和阈值配置。
- adapter → health gate → fail-safe 的数据流测试。
- `runs/sensor_health/<run_id>/metrics.json`，记录每类故障的检测和制动结果。
- 文档化的正常范围、无效原因及应急动作矩阵。

### Exit Gate

- 枚举的确定性坏帧故障检测率 `100%`，错误接受数为 `0`。
- 所有 P0 故障触发 fail-safe/最大制动的比例 `100%`。
- 健康检查本身不超过控制周期预算，并记录 p50/p95 延迟。
- CPU focused/full regression Green；环境相关项仍需在 Stage 4/6 复验。

Stage 1A 实测：确定性坏帧检测率 `100%`、错误接受 `0/1000`、最大制动
`1000/1000`、健康检查 p50/p95 `0.680/0.773 ms`，小于 `100 ms` 控制周期；
全量 `232 passed / 6 skipped`。因此软件 Gate 关闭，Stage 1B 不随之关闭。

---

## 4. Stage 2 — Recorded replay 数据与指标

### 2026-09-03 初始 smoke 状态

只读 `CarlaRecordedEpisode`、HealthGate-first Observation 构造、calibration-aware
BEVFusion/HybridPolicy、安全/控制 replay 和事务化 artifact 已落地。完整 200 帧
随机模型 smoke 位于
`artifacts/carla_replay/replay_20260903T035626Z_9ca5f754`：199 帧进入模型、1 帧
启动期 IMU 拒绝，network-boundary finite rate `1.0`，health/model/end-to-end p95
分别约 `7.11/14.74/23.52 ms`，最大制动 101 帧。该结果只达到 **离线验证**，且
`model_performance_valid=false`、`closed_loop_acceptance_valid=false`。

历史 episode 未记录 vehicle/config/calibration hash，现有报告明确保留三个
provenance gaps；它不能关闭下面的 M0 数据 Gate。后续新 episode 必须记录完整
provenance，禁止反向修改历史文件。

### 目标

建立可重复、可版本化的离线数据回放，让接口、感知、策略和安全指标基于真实
同步数据，而不是随机 tensor 或 Fake 事件。

### 需要完成的工作

1. 定义 replay 数据 schema：
   - episode/frame/timestamp、各 Camera 图像、LiDAR、IMU、ego state。
   - sensor frame/timestamp/calibration version 和原始健康标签。
   - 专家轨迹、控制量、occupancy/hazard 标签（存在时）。
2. 建立数据采集与版本：
   - 原始数据只读保存，清洗结果生成新 dataset version。
   - 以 episode/路线/地形划分 train/validation/test，禁止相邻帧随机拆分造成泄漏。
   - manifest 记录来源、许可、车辆/传感器配置、天气、地形和校验 hash。
3. 实现 replay runner：
   - 按原 timestamp/frame 顺序生成 `Observation`。
   - 支持确定性重放、速度倍率和指定 frame 范围。
   - 统计丢帧、skew、坏帧原因、模型异常和 fail-safe 动作。
4. 建立指标计算：
   - 健康门禁：误接受、误拒绝、precision/recall 和原因混淆矩阵。
   - Perception：occupancy IoU、false negative、可选 traversability F1/IoU。
   - Policy：ADE、FDE、heading/speed error 和物理约束违反率。
   - Safety：危险边界拦截率、raw/post-safety risk、干预率和急停率。
5. 建立 hazard-boundary 子集，必须包含碰撞边界、传感器故障、空轨迹、NaN/Inf、
   错误 frame 和超时案例。

### 最小数据要求

- M0 正式 replay 至少 `1000` 个同步样本。
- 同时覆盖正常、退化和必须急停三类状态；不能用全部正常帧关闭 Gate。
- test split 在开发期间保持冻结；阈值调节只使用 train/validation。

### 测试与命令

```bash
python -m pytest -q tests/integration/test_open_loop_replay.py
python -m pytest -q tests/integration/test_m0_interface_gate.py
python -m pytest -ra
```

将当前因缺少专家日志/危险数据而 skip 的用例改为真实执行；只有数据未部署的
开发环境可保留 skip，正式 replay job 不允许 skip。

### 交付物

- `dataset_manifest.json`、schema version、split manifest 和 calibration manifest。
- 只读 replay dataset、hazard-boundary 子集和 replay runner。
- `metrics.json`、失败样本清单、混淆矩阵和可视化 overlay。
- `docs/` 中的 replay 版本、命令、指标和已知偏差摘要。

### Exit Gate

- ≥1000 样本 replay 无未分类异常，重复运行指标一致。
- 已标注 P0 坏帧错误接受数为 `0`，fail-safe 触发率 `100%`。
- hazard-boundary 拦截率 `100%`，post-safety risk 为 `0`。
- 每项指标均可追溯到 dataset/config/code hash。
- Gate 未关闭时禁止把 offline replay 用作 BC 正式训练数据基线。

---

## 5. Stage 3 — BC 训练基线

### 2026-09-04 分层策略学习决策

策略研究固定为四级递进实验：B0 `Pure BC`、B1 `Affordance + BC`、B2
`Affordance + RSSM + BC`、B3 `BC-initialized Dreamer`。当前只进入 B0 设计，
现有 `HybridPolicy.bc_loss()` 因前向路径经过 single-step RSSM posterior，继续作为
Hybrid/Dreamer 历史骨架，不能作为 Pure BC 对照组。B0 书面规格见
`superpowers/specs/2026-09-04-staged-policy-learning-b0-design.md`；书面规格已确认，
九任务 TDD 计划位于 `superpowers/plans/2026-09-04-staged-policy-learning-b0.md`，
代码尚未开始。该决策不改变 Stage 3 当前“单元验证”成熟度，也不解除专家数据与
正式 perception checkpoint Gate。

### 目标

以最简单、可解释的 `BEV → BC trajectory` 建立稳定开环基线，然后再考虑
Affordance、RSSM 或 Dreamer 增强。

### 需要完成的工作

1. 定义专家样本：
   - 输入为严格 schema 的同步 Observation/BEV 和 IMU。
   - target 为 ego frame `(N,4) [x,y,heading,velocity]`，带 valid mask。
   - 明确采样频率、预测 horizon、单位和 waypoint 时间间隔。
2. 实现 dataloader：
   - 使用 Stage 2 冻结的 episode 级 split。
   - 支持变长轨迹 mask、数据统计和确定性 seed。
   - 缺失、错 frame、非有限样本必须拒绝，不能在 loader 中静默修复。
3. 建立训练入口：
   - 从 `load_system_stack` 读取系统 shape；训练配置只保存 optimizer、batch、seed、
     split 和 checkpoint lineage。
   - 保存 best/last checkpoint、optimizer state、训练曲线和 resolved config。
   - checkpoint 记录代码、配置、数据集和 perception 权重 hash。
4. 使用现有加权 BC loss：
   - XY、heading、speed、二阶平滑项分别记录。
   - 首个正式 baseline 不接 Affordance/RSSM/RL，先建立可诊断参照。
5. 增加评估：
   - ADE、FDE、heading error、speed error、平滑度和轨迹有效率。
   - 按地形、速度、天气、坡度和 episode 分层，不能只报告全局平均值。
   - 至少 3 个固定 seed，报告 mean/std 和失败案例。

### 测试与命令

需要新增并固定项目训练 CLI；推荐最终形式：

```bash
python scripts/train_bc.py --config configs/system.yaml \
  --data-manifest <expert_manifest> --run-dir runs/bc/<run_id>
python scripts/evaluate_bc.py --checkpoint <best_checkpoint> \
  --data-manifest <frozen_test_manifest>
python -m pytest -q tests/unit/policy tests/integration/test_open_loop_replay.py
```

命令在训练脚本实现前只是目标接口，不能写入完成日志。

### 交付物

- 专家数据 schema、dataloader、训练/评估 CLI 和配置说明。
- best/last checkpoint、checkpoint lineage、loss/metric 曲线。
- 冻结 test split 的 ADE/FDE/heading/speed 报告和失败案例可视化。
- `BEV → BC` baseline 报告，作为后续 Affordance/RSSM/RL 的对照组。

### Exit Gate

- 冻结 open-loop test split 上 ADE `< 0.3 m`（现有系统 SDD 目标）。
- FDE、heading/speed error 和物理约束违反率有固定阈值与报告；阈值一经形成
  baseline，不得为后续模型临时放宽。
- 3 个固定 seed 无 NaN/Inf、无 schema/frame 错误，指标方差可解释。
- checkpoint 可在干净环境严格加载并复现评估指标。
- 未通过本 Gate 前不启动 Dreamer/RL 正式实验。

---

## 6. Stage 4 — 真实 CARLA 闭环

### 目标

使用固定版本 CARLA server、正式 perception/BC checkpoint 和可复现场景清单，
验证真实同步、故障停车、策略闭环和安全约束。

### 需要完成的工作

1. 固定环境：
   - CARLA/PythonAPI、地图、车辆蓝图、fixed delta、传感器参数和 GPU/driver 版本。
   - 记录 server/client 版本匹配和启动命令。
2. 建立 `ScenarioManifest`：
   - 路线、spawn/goal、天气、光照、摩擦、坡度、障碍和随机种子。
   - 至少包含普通越野、20° 坡、乱石/低摩擦、遮挡和传感器故障场景。
3. 正式运行守卫：
   - perception/policy checkpoint 缺一不可，严格加载。
   - 保存 config/checkpoint/scenario hash；随机模型只允许 smoke test。
   - 初始正式 Gate 使用已验证的 `occupancy_source=lidar`；learned/fused 必须先有
     Stage 2 occupancy 指标。
4. 验证真实时序：
   - camera/LiDAR/IMU frame、timestamp、skew、timeout 和健康状态日志。
   - 注入 camera/LiDAR 丢失、延迟、空帧和模型异常，检查实际 brake command。
5. 记录闭环指标：
   - success、collision、route completion、raw/post-safety risk。
   - safety intervention/emergency stop、model/control latency。
   - pitch/roll、侧向加速度、jerk 和 collision impulse。
6. 保存失败 episode 的 sensor/control trace 和最小复现 scenario。

### 最小运行规模

- 正式 Gate 至少 `30` 个 episode，覆盖不少于 `3` 个固定 seed 和全部关键场景。
- 每次模型/配置变更使用同一冻结 scenario manifest 做回归。
- smoke episode 不计入验收统计。

### 运行命令

```bash
python scripts/evaluate_carla_closed_loop.py \
  --config configs/system.yaml --host 127.0.0.1 --port 2000 \
  --episodes 30 --perception-ckpt <perception.pt> \
  --policy-ckpt <bc.pt> --occupancy-source lidar
```

### 交付物

- CARLA environment/scenario manifest、checkpoint/config hash。
- episode-level JSON/CSV、聚合指标、传感器同步和制动证据。
- 碰撞/失败/急停 episode 的日志、截图或录像与复现命令。
- `docs/` 中的 CARLA 验收报告，明确 smoke 与 formal evaluation 的区别。

### Exit Gate

- Success Rate `> 90%`，Collision Rate `= 0`。
- 姿态过载报警次数 `= 0`，危险场景 post-safety risk `= 0`。
- timeout、传感器无效和模型异常实际最大制动成功率 `100%`。
- 端到端 p95 latency `< 0.1 s`，无未解释的控制周期 deadline miss。
- 所有指标绑定环境、scenario、config 和 checkpoint hash。

---

## 7. Stage 5 — ONNX Runtime/TensorRT 对齐

### 目标

证明部署模型不仅“能够导出”，还在 ORT、TensorRT 和目标硬件上保持数值、
安全决策与实时性一致。

### 需要完成的工作

1. 固定部署契约：
   - 输入/输出名称、shape、dtype、动态 batch profile、opset 和 schema version。
   - 明确 ONNX 四输入代表双模态已经通过图外健康门禁。
2. 建立 PyTorch → ORT 回归：
   - batch 1/2、正常 replay、边界 replay 和多次确定性运行。
   - 比较 BEV、occupancy（导出后适用）和 trajectory。
3. 完成 TensorRT C++ runtime：
   - parser/build/load、binding 检查、optimization profile、buffer 生命周期、
     async CUDA stream 和错误返回。
   - engine 与 ONNX/config/hardware hash 绑定，不跨不兼容环境复用。
4. 实现 C++/部署 adapter 健康门禁：
   - shape、finite、空 LiDAR、frame age/skew 和 mask 语义与 Python 对等。
   - 无效输入不得调用 engine，直接生成安全停车结果。
5. 精度顺序：FP32 → FP16 → INT8。
   - FP32/FP16 完成数值和安全回归后才评估 INT8。
   - INT8 calibration 使用独立代表性数据和版本化 cache。
6. 性能测试区分 warm-up、H2D、inference、D2H、安全过滤和完整端到端延迟，
   报告 p50/p95/p99、显存和吞吐。

### 建议初始容差

| 对比 | 初始数值门禁 |
|---|---|
| PyTorch FP32 ↔ ORT FP32 | `rtol <= 1e-4`, `atol <= 1e-4` |
| ORT FP32 ↔ TensorRT FP32 | `rtol <= 1e-4`, `atol <= 1e-4` |
| PyTorch FP32 ↔ TensorRT FP16 | `rtol <= 1e-2`, `atol <= 1e-2` |

这些是本路线新增的初始工程阈值。若模型数值范围要求调整，必须基于 replay
误差分布和安全决策证明重新版本化，不能为单个失败样本临时放宽。

### 测试与命令

```bash
python -m pytest -q tests/unit/deployment
python -m deployment.onnx_export --out-dir runs/deployment/<run_id>/onnx
# 后续新增：PyTorch↔ORT 数值回归、trtexec/build、C++ gtest/ctest
```

### 交付物

- ONNX contract manifest、ORT/TRT engine build log 和数值对齐报告。
- 可复现 C++ runtime、CMake/CTest、engine metadata 和错误处理测试。
- 目标硬件 FP32/FP16 延迟、显存、功耗（可用时）和安全回归报告。
- 图外 sensor validator 的 Python/C++ parity 报告。

### Exit Gate

- 所有冻结 replay 样本满足版本化数值容差，NaN/Inf 数为 `0`。
- Python/ORT/TRT 在 hazard set 上的 fail-safe 和安全决策不一致数为 `0`。
- 目标硬件端到端 p95 `< 0.1 s`，并留有控制、I/O 和 watchdog 预算。
- engine 构建/加载失败、binding 错误和坏输入均进入明确错误或安全停车。
- FP16 Gate 完成后才能将 INT8 纳入下一轮优化。

---

## 8. Stage 6 — ROS 2 / SIL / HIL

### 目标

将已验证的感知、策略、部署和安全链路集成到真实 ROS 2 通信与目标计算平台，
依次完成软件在环和硬件在环；不直接跳到开放道路或载人车辆。

### 需要完成的工作

1. ROS 2 接口：
   - 将裸数组升级为带 `header/frame_id/timestamp/schema_version` 的强类型消息。
   - 明确 sensor observation、BEV/occupancy、trajectory、safety decision、
     Ackermann command 和 diagnostics topic。
   - 为 QoS、频率、queue depth、deadline、liveliness 和 clock source 建立配置。
2. ROS 2 节点图：
   - sensor adapter → health gate → inference → safety → controller → vehicle bridge。
   - 使用 launch 文件统一参数；禁止节点复制系统物理参数。
   - lifecycle、启动顺序、模型加载失败和优雅停机必须可测试。
3. Watchdog 与安全：
   - 传感器/模型/trajectory/control 超时、节点退出、消息错 frame 或非有限值时
     发布最大制动并锁存原因。
   - 设计独立 E-stop 通道，不能依赖神经网络或同一进程存活。
4. SIL：
   - 使用 rosbag/replay 和 CARLA/Gazebo bridge 运行完整节点图。
   - 比较 ROS 2 输出与 Stage 2/4 Python reference 的 trajectory、安全决策和控制。
   - 长时间运行检查内存、queue 堆积、deadline miss 和重启恢复。
5. HIL：
   - 在目标 ECU/GPU、真实 CAN/底盘接口或等价台架上运行部署链路。
   - 先断开实际驱动执行器验证命令，再在受控台架验证制动、转向限幅和 E-stop。
   - 注入掉线、延迟、进程崩溃、过温/降频和电源重启。
6. 车辆前置：
   - 完成 Python/C++ SafetyFilter 和控制器数值对等。
   - 建立车辆参数/标定版本、回滚镜像、操作员 checklist 和事故停止流程。
   - HIL 后仍需独立的封闭场地 Gate，不能直接进入开放道路。

### 交付物

- ROS 2 interfaces、launch、参数、diagnostics、watchdog 和 E-stop 设计。
- rosbag/SIL 测试集、节点级 launch test 和长稳报告。
- HIL 硬件/固件/driver/engine manifest、故障注入和台架测试报告。
- Python/ORT/TRT/ROS/C++ trajectory、安全和控制 parity 报告。
- 封闭场地测试计划、风险清单、回滚及人工接管流程。

### Exit Gate

- ROS 2 强类型消息的 frame/timestamp/schema 检查覆盖所有主链 topic。
- SIL 冻结 replay 的关键输出与 reference 一致，安全决策不一致数为 `0`。
- 长稳测试无未处理进程退出、内存持续增长或 deadline miss。
- HIL 故障注入中 watchdog/E-stop/最大制动成功率 `100%`。
- 目标平台端到端 p95 `< 0.1 s`，过温/降频下仍进入可解释的降级或停车。
- 完成报告只能声明“ROS 2/SIL/HIL verified”；未经封闭场地和车辆级安全审查，
  不得声明 vehicle verified 或开放道路可用。

---

## 9. 阶段完成检查表

| 阶段 | 必须提交的证据 | 进入下一阶段前确认 |
|---|---|---|
| 1. Sensor Health | 健康 schema、故障矩阵、focused/full 测试、制动证据 | 坏帧错误接受 0；P0 最大制动 100% |
| 2. Replay | dataset/split manifest、≥1000 样本指标、hazard 报告 | 可复现；危险拦截 100%；post-risk 0 |
| 3. BC | dataloader、CLI、checkpoint lineage、冻结 test 报告 | ADE <0.3 m；checkpoint 可复现 |
| 4. CARLA | environment/scenario/checkpoint manifest、≥30 episodes | success >90%；collision/姿态报警 0 |
| 5. ORT/TRT | contract、数值/安全 parity、目标硬件性能 | 决策差异 0；p95 <0.1 s |
| 6. ROS 2/SIL/HIL | 强类型接口、launch、watchdog、SIL/HIL 报告 | 故障停车 100%；仍不得越级声明实车完成 |

## 10. 每次 Gate Review 需要回答的问题

1. 使用了哪个代码、配置、数据、标定、模型和环境版本？
2. 哪些命令可以在干净环境复现结果？
3. 哪些测试通过、失败或跳过？skip 是否属于该 Gate 的必需环境？
4. 指标是否来自冻结 test/scenario，而不是调参集？
5. 是否保存了失败案例，而不只保存平均指标？
6. 是否存在 Mock/Fake 被错误当作数据、物理或硬件验收？
7. fail-safe 是否在最终执行边界产生了真实制动结果？
8. 若回退，应该回到哪个最早 Gate，责任文件和负责人是什么？

只有上述问题均有可审计答案，阶段状态才能从“进行中”改为“已验收”。
