# 分层策略学习架构与 B0 Pure BC 设计

- 日期：2026-09-04
- 状态：书面规格已确认；B0 已按实施计划完成软件实现与最终验证
- 本轮实施范围：B0 `BEV + Dynamics → Pure BC trajectory`
- 当前实现成熟度：B0 **单元验证**；既有 `HybridPolicy` 维持兼容与后续骨架
- 外部依赖 Gate：正式专家标签、episode 级冻结 split 和可追溯 perception checkpoint

## 1. 背景与事实基线

New_ORAD 的长期方法链路确定为：

```text
Multi-modal sensing
  → BEV perception
  → Terrain Affordance
  → Terrain-aware RSSM
  → BC-initialized Dreamer policy
  → explicit safety
  → control
```

当前 `HybridPolicy.bc_loss()` 不能充当纯 BC 对照组。它的前向路径是
`PolicyEncoder → single-step RSSM posterior → latent projection → Actor`，训练态还会从
posterior 采样随机 latent。该实现可以继续作为历史 Hybrid/Dreamer 骨架，但它已经引入
RSSM 参数和随机表征，无法回答“不使用 Affordance 和 World Model 时，BC 本身能达到
什么水平”。

设计启动时，真实 CARLA 初始 episode 只有 200 帧，动作来自 Traffic Manager，manifest 明确
记录 `expert_labels=false`。它只能验证数据和网络边界，禁止用于关闭 BC 训练或性能
Gate。当前也没有可作为正式输入的 perception checkpoint、专家轨迹、episode 级冻结
split、BC dataloader、训练/评估 CLI 或 ADE/FDE 报告。B0 软件实施现已补齐 loader、
CLI 和指标能力，但数据与权重依赖仍未满足，所以仍没有正式 ADE/FDE 报告。

因此，本设计先固定可比较的 B0–B3 研究序列，再只实现第一阶段 B0。B1–B3 必须分别
形成后续设计、计划和验收记录，不能在 B0 尚未形成可复现基线时同时接入。

## 2. 设计决策

### 2.1 采用：递进式单变量架构

```text
B0  Pure BC
    BEV + dynamics → deterministic BC → trajectory

B1  Affordance + BC
    BEV + dynamics + affordance representation → BC → trajectory

B2  Affordance + RSSM + BC
    sequential observation + previous action → RSSM posterior → BC actor

B3  Full model
    B2 checkpoint → latent imagination → Dreamer actor/critic fine-tuning
```

每一级只新增一个主要研究能力，能够依次测量 Affordance、时序 World Model 和 Dreamer
长期优化的增益。四级保持相同传感器、BEV backbone、动力学输入、轨迹输出、安全层、
控制器、数据 split 和指标定义。

### 2.2 未采用：继续把现有 `HybridPolicy.bc_loss()` 当作 BC baseline

该方案改动最少，但 BC 梯度会经过 RSSM posterior，训练输出也受 latent sampling 影响。
即使没有调用 world-model loss，也不能把结果解释为不含 RSSM 的 Pure BC。

### 2.3 暂缓：一次实现 B0–B3 或直接训练 Dreamer

该方案会把专家数据质量、Affordance 标签、World Model 预测误差和 RL 优化稳定性混在
同一次实验中。当前数据 Gate 和 checkpoint Gate 尚未关闭，失败时无法进行可信归因。

完整 factorial ablation 中的 `BEV + RSSM + BC` 不是主开发链路；当 B2 稳定且论文需要
拆分 Affordance/RSSM 独立效应时，再作为补充实验实现。

## 3. 目标与非目标

### 3.1 本轮目标

1. 新建不包含 Affordance、RSSM、prior/posterior、reward head 或 critic 的
   `BCPolicy`。
2. 保持策略输出为 ego frame `(B,20,4)`，字段顺序固定为
   `[x_m, y_m, heading_rad, velocity_mps]`，waypoint 间隔 `0.1 s`。
3. 为 BC 输入加入显式动力学条件，并对字段、shape、单位、finite 和 normalization
   provenance 建立版本化契约。
4. 实现支持 valid mask 和角度周期性的可诊断 BC loss。
5. 实现只接受正式专家 manifest 和严格 perception checkpoint 的 dataloader、训练、
   评估与 checkpoint lineage 骨架。
6. 保持现有 `HybridPolicy` 公共导入、参数键和历史随机 replay smoke 可读取；不静默把
   Hybrid checkpoint 加载到 `BCPolicy`。
7. 用 TDD 将 B0 软件边界提升到单元验证；只有真实专家数据和冻结 test 结果存在后，
   才能提升到离线验证。

### 3.2 非目标

- 本轮不接入或训练 `TerrainAffordanceHead`。
- 本轮不迁移、重写或删除现有 `HybridPolicy`、`WorldModel`、RSSM、Actor/Critic。
- 本轮不训练 Dreamer，不报告 RL return 或 CARLA 闭环性能。
- 本轮不把 Traffic Manager 数据重新命名为专家数据。
- 本轮不使用随机 perception 权重生成性能报告。
- 本轮不做 Pure BC 的 ONNX/ORT/TensorRT 导出；部署工作在正式网络和仿真基线形成后
  单独设计。
- 本轮不覆盖原始 dataset、历史 artifact 或既有 checkpoint。

## 4. B0 架构与数据流

### 4.1 运行数据流

```text
valid Camera/LiDAR/IMU
  → SensorHealthGate
  → Observation
  → frozen BEVFusion checkpoint
  → BEV tensor (B,32,50,50)
                         ┐
IMU history (B,10,6)     ├→ deterministic BC encoder
ego dynamics (B,8)       ┘
  → shared latent (B,256)
  → GRU trajectory decoder
  → candidate trajectory (B,20,4), ego frame
  → SafetySupervisor / SafetyFilter
  → controller
```

在线健康契约不变：Camera 和 LiDAR 必须同时有效，IMU 必须有效；任何缺失、陈旧、
skew、shape 或 non-finite 错误必须在模型前进入 fail-safe，禁止用零输入掩盖传感器
故障。

### 4.2 Ego dynamics v1

绝对 world pose 不进入 B0，避免策略记忆地图位置或路线。固定输入顺序为：

```text
[speed, steering, pitch, roll, vx, vy, yaw_rate, accel_z]
```

shape 为 `(B,8)`；速度单位为 `m/s`，角度为 `rad`，角速度为 `rad/s`，加速度为
`m/s²`。转换逻辑放在共享 `utils` 契约中，不在 dataloader、replay 和 simulator 中
复制。所有值必须 finite。

归一化统计只从 train split 计算，作为 checkpoint buffer 和 metadata 保存；validation、
test 和 runtime 只能读取这些统计。任何标准差小于实现 epsilon 的字段必须启动失败，
不能在运行时重新估计或静默替换。

### 4.3 网络组成

B0 使用三个确定性分支：

- BEV encoder：卷积下采样和全局池化，输出 256 维特征；
- IMU encoder：`GRU(6→256)`，输出最后隐状态；
- ego encoder：两层 MLP，将 8 维 `EgoDynamicsV1` 映射到 64 维。

三路特征拼接后通过 fusion MLP 得到 256 维 latent。轨迹 decoder 使用 256 维 latent
初始化固定 horizon 的 GRU 解码器，输出 `(B,20,4)`。B0 中不得出现随机采样、RSSM
state、prior/posterior、action projection、reward head 或 critic。

为了保证后续消融公平，B1–B3 复用相同的 BEV/IMU/ego 输入契约和 trajectory decoder
输出契约。现有 `HybridPolicy` 不做内部搬迁，以避免破坏历史 state-dict 参数键；B0
使用新的 checkpoint family 和 schema version。

### 4.4 配置所有权

`configs/system.yaml` 继续是唯一运行配置源。B0 实现将增加以下严格 section；未知、
缺失或类型错误的字段必须由 `load_system_stack()` 拒绝：

```yaml
policy_model:
  family: pure_bc_v1
  encoder_hidden: 256
  ego_hidden: 64
  latent_dim: 256
  horizon: 20
  trajectory_dim: 4
  ego_dynamics_schema: ego-dynamics-v1
```

BEV/IMU shape 从 `geometry` 和 `sensors` 派生，waypoint 间隔从 `control.dt` 派生，
不得在 `policy_model` 重复配置。最终固定值为：

```text
policy family       pure_bc_v1
BEV channels        32
BEV H/W             50/50（由 geometry 推导）
IMU shape           10 × 6（由 sensors 推导）
ego dynamics dim    8
encoder hidden      256
latent dim          256
trajectory horizon  20
trajectory dim      4
waypoint dt         0.1 s（与 control.dt 一致）
```

BC loss 权重继续由唯一 YAML 提供，初始值为
`xy=1.0, heading=0.2, speed=0.2, smooth=0.05`。optimizer、batch size、epoch、
单次训练 seed 和数据 manifest 属于实验运行参数，必须写入 resolved run config 和
manifest，但不得成为另一个系统运行配置源。

## 5. 专家数据契约

### 5.1 BCSample v1

每个样本必须可追溯到 dataset、episode、frame 和 timestamp，并提供：

```text
images              (3,3,192,192) uint8/declared image encoding
lidar               raw valid point cloud, >= 16 points
imu_history         (10,6) float32
ego_dynamics        (8,) float32, EgoDynamicsV1
expert_trajectory   (20,4) float32, ego frame
trajectory_mask     (20,) bool
calibration_version string
expert_source       string
expert_label        true
```

训练阶段由 strict-loaded、冻结的 BEVFusion checkpoint 在 `eval/no_grad` 下产生 BEV；
该 checkpoint hash 必须进入 BC run manifest。首版不增加持久化 BEV cache，避免同时
维护第二份特征数据；若后续性能需要 cache，必须另行定义 checkpoint/config/hash
绑定和失效规则。

### 5.2 Split 与拒绝规则

- split 必须以 episode/路线/地形为单位，不允许相邻 frame 随机拆分；
- frozen test manifest 在第一次正式评估前写入 hash，训练和调参不得读取；
- `expert_label != true`、来源不明、缺文件、越界路径、错误 frame、错误 shape、空 mask、
  non-finite、标定不匹配或时间不连续的样本必须拒绝并记录原因；
- loader 不允许对缺失传感器、轨迹或动力学字段补零；
- 变长专家轨迹只能通过显式 `trajectory_mask` 表达，padding 值不参与 loss 或 metric。

现有 `datasets/carla_initial` 因 `expert_labels=false` 必须被 B0 loader 拒绝。该失败是
正确的 Gate 行为，不是需要绕过的训练障碍。

## 6. Loss 与指标

### 6.1 Masked BC objective

```text
L_bc = 1.0 L_xy + 0.2 L_heading + 0.2 L_speed + 0.05 L_smooth
```

- `L_xy`：valid waypoint 上的 XY MSE；
- `L_heading`：先以 `atan2(sin Δθ, cos Δθ)` 包装到 `[-π,π]`，再计算 MSE；
- `L_speed`：valid waypoint 上的速度 MSE；
- `L_smooth`：只在连续三个 waypoint 均 valid 时计算预测 XY 二阶差分平方均值。

每个分量、valid waypoint 数、总 loss 和 gradient finite 状态均写入训练日志。batch 中
任一样本没有 valid waypoint，或 loss/gradient non-finite 时，本 step 明确失败；不得
跳过后继续生成成功 checkpoint。

### 6.2 Open-loop 指标

冻结 test split 至少报告：

- ADE、FDE；
- wrapped heading MAE；
- velocity MAE；
- XY 二阶差分平滑度；
- negative-speed、倒退 waypoint、超速和非有限轨迹比例；
- 按地形、速度、天气、坡度、路线和 episode 分层结果；
- 三个 seed 的 mean/std 和失败样本列表。

首个 B0 正式 Gate 固定为：

```text
mean ADE                 < 0.30 m
mean FDE                 < 0.60 m
mean wrapped heading MAE < 0.10 rad
mean velocity MAE        < 0.50 m/s
non-finite trajectory    = 0
schema/frame error       = 0
```

负速度、超速和平滑度先作为必须报告的诊断指标，不通过调节 SafetyFilter 阈值掩盖；
第一次冻结 test 报告会作为 B1–B3 不得临时放宽的比较基线。

## 7. 训练、评估与产物

目标 CLI 为：

```bash
python scripts/train_bc.py --config configs/system.yaml \
  --data-manifest <expert_manifest> \
  --perception-checkpoint <checkpoint> \
  --seed <41|42|43> --run-dir runs/bc/<run_id>

python scripts/evaluate_bc.py --config configs/system.yaml \
  --data-manifest <frozen_test_manifest> \
  --perception-checkpoint <checkpoint> \
  --policy-checkpoint <checkpoint> --run-dir runs/bc_eval/<run_id>
```

CLI 尚未实现前，上述命令只表示目标接口，不构成完成证据。每次运行创建不可覆盖目录：

```text
runs/bc/<run_id>/
├── manifest.json
├── resolved_config.yaml
├── metrics.json
├── logs/
└── checkpoints/
    ├── epoch-0001.pt
    ├── epoch-0002.pt
    ├── best.json
    └── last.json
```

每个 epoch checkpoint 使用新文件名，禁止覆盖已生成权重；`best.json` 和 `last.json`
只保存 checkpoint 文件名、hash 和对应 epoch，并通过原子替换更新。这样既能稳定解析
best/last，又保留所有历史权重。

checkpoint 至少包含：

- `model_family=pure_bc_v1`、schema version 和严格 state dict；
- system config hash、dataset/split/calibration hash；
- perception checkpoint hash；
- train-only normalization statistics；
- git commit、dirty-worktree 标志、依赖版本和随机 seed；
- epoch、optimizer state、best metric 和训练命令。

加载时 family、shape、config、normalization 和 parameter key 任一不匹配都应失败。不得
自动回退为随机权重、部分加载或 legacy HybridPolicy。

## 8. B1–B3 后续 Gate

### 8.1 B1 Affordance + BC

B0 离线 Gate 关闭后才启动。B1 增加经过独立指标验证的 traversability、roughness、
risk/uncertainty representation，同时保留原始 BEV，避免把信息瓶颈误解释为 Affordance
收益。它必须与 B0 使用相同 split、输入、decoder、训练预算和安全链路，并报告参数量
与延迟。

### 8.2 B2 Affordance + RSSM + BC

B1 稳定后启动。B2 必须消费真实序列、previous action、episode boundary 和 burn-in，
不能使用每帧清零的 single-step RSSM 充当时序模型。World Model 先以 reconstruction、
reward/risk、continue 和 balanced KL 独立训练并通过 N-step rollout 指标，再让 BC actor
消费 posterior latent。

### 8.3 B3 Full Dreamer

B2 checkpoint 作为 World Model 和 Actor 初值。Dreamer 仅在固定 World Model 的 latent
imagination 中启动 actor/critic 优化，随后才进行受显式安全层保护的 CARLA 闭环验证。
B3 不是“去掉 BC 的 Dreamer”，而是 `BC initialization + Dreamer fine-tuning`。

## 9. TDD 与验证

### 9.1 Red→Green 顺序

1. `EgoDynamicsV1` 字段顺序、单位、shape 和 non-finite 拒绝测试；
2. strict YAML policy 配置、派生 shape 与跨模块一致性测试；
3. `BCPolicy` 确定性、输出 shape、无 RSSM 参数和梯度边界测试；
4. masked/wrapped BC loss、空 mask 和 non-finite 测试；
5. expert manifest、episode split、路径/hash 和 legacy non-expert 拒绝测试；
6. ADE/FDE/heading/speed 指标测试；
7. checkpoint family、lineage、strict load 和不覆盖测试；
8. train/eval CLI 的小型 CPU fixture 集成测试；
9. recorded replay 的 Pure BC 模型选择、trajectory contract 和 fail-safe 测试；
10. focused、integration、unit branch coverage、flake8 和全量回归。

### 9.2 软件完成条件

- 新测试先因缺少行为失败，再以最小实现通过；Red/Green 命令写入实施计划；
- `BCPolicy` 不含 RSSM/Affordance/Dreamer 参数，eval 同输入 bitwise deterministic；
- 所有 policy 输入、输出、loss、metric 和 checkpoint 张量 finite；
- 现有 `HybridPolicy` 公共导入、state-dict keys 和测试保持兼容；
- 当前 non-expert episode 被训练入口明确拒绝；
- CPU focused/full regression、unit branch coverage 与 flake8 通过；
- `git diff --check` 和修改文档链接检查通过；
- 无专家数据时只能声明 B0 **单元验证**，不能宣称训练基线完成。

### 9.3 离线 Gate

只有以下证据齐备后，B0 才能提升为 **离线验证**：

1. 至少 1000 个同步且带正式专家 trajectory 的样本，并采用 episode 级冻结 split；
2. 三个固定 seed `41/42/43` 均无 NaN/Inf、schema 或 frame 错误；
3. checkpoint 在干净环境 strict load，重复评估得到一致指标；
4. 冻结 test 满足第 6.2 节阈值；
5. 报告绑定 code/config/data/calibration/perception/policy hash。

## 10. 文档与实施边界

本设计已经书面审阅，并已按 B0 实施计划内联完成。每个任务的 Red/Green 证据记录在
计划中；同步更新 `README.md`、`System_overview.md`、
`docs/技术原理与代码架构.md`、`docs/ENGINEERING_EXECUTION_ROADMAP.md`、
`docs/SmartSteer_Status.md` 和 `docs/README.md`。

B0 的软件实现完成不等于专家数据、模型性能或 CARLA 闭环完成；所有状态更新必须继续
使用“代码骨架、单元验证、离线验证、仿真闭环验证、车辆验证”五级术语。

实施计划：`../plans/2026-09-04-staged-policy-learning-b0.md`。

## 11. 2026-09-04 实施结果

- 已交付独立 `BCPolicyConfig`、`EgoDynamicsV1` 与不含 RSSM/Affordance/Dreamer 的
  确定性 `BCPolicy`。
- 已交付严格专家 manifest/episode split/HealthGate loader、masked open-loop 指标、
  冻结 perception 训练评估引擎，以及 `new-orad-policy-checkpoint-v1` 的事务化产物。
- 已交付 `train_bc.py`、`evaluate_bc.py` 与 recorded replay 的 `pure_bc_v1` 严格路由；
  异常、非有限或错误 horizon 进入 fail-safe。
- 已保留 `HybridPolicy` 公共导入和历史随机 smoke 路径，没有迁移或改写其 checkpoint。
- CPU fixture 只证明软件接口，固定标记 `model_performance_valid=false`。现有真实数据
  `expert_labels=false`，未用于训练，仓库未生成正式 B0 checkpoint 或性能 artifact。
- 最终测试/覆盖率命令与数字记录在实施计划和 `SmartSteer_Status.md`；B0 成熟度严格
  保持**单元验证**，第 9.3 节离线 Gate 尚未关闭。
