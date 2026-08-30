# New_ORAD 方法框架 SDD/TDD 跟进记录（2026-08-29）

> 策略输入：`docs/relatetalk.md`  
> 工程原则：Correctness First → Learning Baseline → Robustness；Red → Green → Refactor  
> 状态口径：代码骨架 → 单元验证 → 离线验证 → 仿真闭环 → 实车验证

## 1. 本轮结论

本轮没有提前扩大 Dreamer/RSSM 复杂度，而是先关闭系统接口和失效保护缺口，并落地一个默认关闭、可独立验收的 Terrain Affordance v1。

```text
Full suite: 127 collected, 121 passed, 6 skipped, 16 warnings
Unit suite: 119 passed, 2 skipped, 16 warnings
Unit branch coverage: 80.62% (gate: 79%)
Lint: python -m flake8 src tests scripts --max-line-length=99 → PASS
```

新增关键模块的 branch coverage：

| 模块 | 覆盖率 | 结论 |
|---|---:|---|
| `src/affordance/terrain.py` | 100% | shape/range/mask/loss/异常分支已单元验证 |
| `src/safety/supervisor.py` | 100% | NORMAL/DEGRADED/EMERGENCY_STOP 门禁已单元验证 |
| `src/utils/frames.py` | 100% | ego/world 点和轨迹双向转换已单元验证 |
| `src/utils/schema.py` | 96% | Observation 同步、shape、finite 契约已单元验证 |

## 2. `relatetalk.md` 策略到工程变更的映射

| 文档策略 | 本轮实现 | 验证状态 |
|---|---|---|
| 明确唯一 ego frame/schema | `CoordinateFrame`、`Trajectory.frame/timestamp`、ego↔world 变换 | 单元验证 |
| 传感器严格同步 | `Observation` 记录 simulator frame、各 sensor frame/timestamp；CARLA 各相机独立队列 | CPU 契约完成，真实 CARLA 待验收 |
| timeout/NaN/空输出必须 fail-safe | `SafetySupervisor` 状态机；runner 异常、skew、延迟、非法轨迹转最大制动 | 单元 + 1000 样本 CPU 门禁 |
| 风险应分别统计策略前后 | 新增 raw-policy/post-safety risk、intervention、emergency-stop 指标 | 代码完成，真实闭环待验收 |
| Affordance 独立于通用 BEV | 新增 Traversability/Roughness 双头及 masked multitask loss | 单元验证；默认关闭 |
| 粗糙度可用 IMU 弱监督 | 新增 speed-conditioned vertical-acceleration RMS target | 单元验证；真实标签待验证 |
| BC 应拆分物理量权重 | XY/heading/speed/smooth 四项 loss 与独立日志 | 单元验证；训练基线待建立 |
| 先短 horizon 再扩大 imagination | 默认 imagination horizon 调整为 5 | 配置/代码完成；预测指标待建立 |
| 监控 RSSM 隐变量退化 | world-model loss 返回 prior/posterior std | 单元验证；训练曲线待记录 |

## 3. Stage 1：Correctness 改进

### 3.1 坐标与数据契约

- `utils.frames.CoordinateFrame` 明确 `ego`/`world`，ego 采用 x 前、y 左、z 上，SI 单位和 rad。
- `Trajectory` 增加 `frame` 与 `timestamp`，安全过滤和几何转换保持元数据不丢失。
- 点或轨迹的 ego↔world 转换拒绝错误 shape、NaN/Inf 和未知 frame。
- `Observation` 强制：camera/LiDAR 同 simulator frame、LiDAR `(N,3/4)`、IMU `(T,6)`、数组与 timestamp finite。
- 当前 CARLA safety/controller 仍使用 world trajectory；策略 ego 输出通过唯一显式转换进入 world。完整“安全层全 ego 化”属于较宽接口迁移，需另立 checkpoint/ROS/occupancy 兼容 Red tests 后实施。

### 3.2 SafetySupervisor 与失效路径

```text
NORMAL:
  valid trajectory + frame/shape/finite/speed/time checks pass

DEGRADED:
  deterministic SafetyFilter corrected a valid policy trajectory

EMERGENCY_STOP:
  missing/short trajectory, frame mismatch, NaN/Inf, invalid speed,
  sensor age/skew timeout, model timeout, observation/model exception
```

急停轨迹保留目标 frame/timestamp，由控制器产生 `max_decel`。runner 不再让 schema、预处理、感知或策略异常越过安全边界。

### 3.3 M0 CPU 接口门禁

`tests/integration/test_m0_interface_gate.py` 连续验证 1000 条合成轨迹，每 10 条注入一条 NaN：

- 900 条有效轨迹进入 NORMAL；
- 100 条无效轨迹全部进入 EMERGENCY_STOP；
- 所有急停均映射到最大减速度；
- 无无效 trajectory 到达正常控制路径。

该测试不可 skip。它证明软件接口门禁，但不能替代 recorded replay、CARLA 时序或车辆制动距离验收。

## 4. Stage 2：Learning Baseline 改进

### 4.1 Terrain Affordance v1

新增 `src/affordance/terrain.py`：共享编码器后分别输出：

- `traversability ∈ [0,1]`，使用 masked BCE；
- `roughness ≥ 0`，使用 masked L1；
- loss 权重显式配置，空/非法 mask 和 shape 错配直接拒绝；
- IMU 垂向加速度 RMS 按速度归一化产生 roughness 弱标签，零速度由 epsilon 保护。

该能力尚未获得真实标签、F1/IoU 或跨地形泛化证据，因此 `configs/system.yaml` 中 `affordance.enabled: false`。它不能成为当前 safety hard constraint，也不应被描述为离线验证完成。

### 4.2 BC 与 World Model 可观测性

- BC 从单一整体 MSE 改为 XY、heading、speed、二阶差分 smooth 四项加权 loss。
- 默认权重为 `1.0 / 0.2 / 0.2 / 0.05`，返回组件便于训练日志和消融。
- World Model loss 返回 `prior_std`/`posterior_std`，用于诊断 latent collapse。
- imagination horizon 设为 5；在短期预测误差和 BC ADE 门槛达标前不扩大。

## 5. Gate 完成度矩阵

| Gate | 验收目标 | 当前状态 | 本轮证据 | 未关闭项 |
|---|---|---|---|---|
| M0 接口正确性 | frame/schema/sync/fail-safe，1000 样本无非法下游输出 | 部分完成 | 1000 样本 CPU Green；新 frame/schema/supervisor tests | recorded replay、真实 CARLA timeout/skew/制动 |
| M1 BEV 基线 | 多模态对齐、可视化和稳定离线指标 | 部分完成 | 既有 BEV/occupancy shape 与 CPU pipeline | 真实标定、BEV overlay、离线精度 |
| M2 Affordance | Traversability/Roughness 数据与 F1/IoU | 代码骨架 + 单元验证 | 双头/loss/弱标签 100% branch | dataset、标注策略、F1/IoU、消融 |
| M3 World Model | 短期预测、latent 统计稳定 | 部分完成 | prior/posterior std 可观测，horizon=5 | dataloader、预测误差和训练曲线；独立包迁移 |
| M4 BC | 轨迹 ADE/物理分量门槛 | 部分完成 | 加权 BC loss 单测 | 专家数据、训练入口、checkpoint、ADE |
| M5 Imagination/RL | imagination 稳定且优于 BC | 代码骨架 | 维持短 horizon | 环境训练、回报/安全消融 |
| M6 Robustness | DR、风险、故障注入、闭环阈值 | 部分完成 | DR helper、风险分层指标、fail-safe | CARLA/Gazebo/ROS2/TRT 专用验收 |

## 6. TDD 记录

本轮按以下 Red→Green 顺序执行：

1. 先增加 frame/schema、SafetySupervisor 和 1000 样本 M0 测试；缺少模块时 Red。
2. 实现最小 frame/schema/supervisor，focused tests Green。
3. 先增加 Terrain Affordance shape/range/mask/loss/weak-target 测试；缺少模块时 Red。
4. 实现双头和 loss，focused tests Green。
5. 增加 BC 分量和 RSSM std 断言，完成实现并全量回归。
6. 修复静态检查问题并执行最终 full suite、unit branch coverage、flake8。

仓库快照能保存测试与实现证据，但不能单独证明每个历史提交的 Red/Green 顺序；后续 CI 应保留每阶段运行日志或提交记录。

### 6.1 六个保留 skip

| 数量 | 原因 | 是否可由本轮 CPU Mock 替代 |
|---:|---|---|
| 2 | CARLA 20° 越野地图、Gazebo 悬挂动力学 | 否，属于物理环境验收 |
| 2 | 专家日志、危险边界 recorded dataset | 否，Fake 只能验证读取/指标计算 |
| 1 | CUDA 不可用 | 否，CPU 路径已覆盖 |
| 1 | 完整 ROS 2 消息栈缺失 | adapter 可继续 Fake 化；最终节点验收不能替代 |

## 7. 变更文件

| 范围 | 文件 |
|---|---|
| Frame/schema | `src/utils/frames.py`、`src/utils/schema.py`、`src/utils/types.py` |
| Affordance | `src/affordance/__init__.py`、`src/affordance/terrain.py` |
| Safety | `src/safety/supervisor.py`、`src/safety/kinematic_filter.py`、`src/safety/__init__.py` |
| Policy | `src/policy/hybrid_policy.py` |
| Sim/runtime | `src/sim/carla_closed_loop.py` |
| Config | `configs/system.yaml` |
| TDD | `tests/unit/utils/test_schema_frames.py`、`tests/unit/safety/test_supervisor.py`、`tests/unit/perception/test_terrain_affordance.py`、`tests/integration/test_m0_interface_gate.py`、policy tests |

## 8. 下一阶段 Action Items

### P0：环境正确性验收

- 在固定版本 CARLA server 上记录 camera/LiDAR/IMU frame 与 timestamp，验证 skew/timeout 急停。
- 使用 recorded replay 重跑 M0 ≥1000 样本，保存无效样本分类和 brake 证据。
- 为 runner 增加 FakeVehicle/FakeSensorStack 测试，覆盖 perception/policy exception 和 timeout 的控制输出。
- 建立 hazard-boundary dataset，验证拦截率 100% 且 post-safety risk 为 0。

### P1：离线学习闭环

- 定义专家日志 schema/dataloader/checkpoint metadata，建立 weighted BC ADE 基线。
- 为 Affordance 明确标签、valid mask 与数据划分，建立 traversability F1/IoU 和 roughness MAE。
- 建立 1/3/5-step world-model prediction error、prior/posterior std 与 collapse 告警。
- 在指标达标后再决定是否把 Affordance 接入 policy，不直接接入 Safety hard constraint。

### P2：工程化与部署

- 设计 ROS 2 强类型 Observation/Trajectory 消息，携带 header/frame/schema version。
- 增加 PyTorch→ORT→TensorRT 数值回归、C++ runtime 和目标硬件时延。
- 将 `sim` runner 和 ROS adapter 的纯 CPU 分支覆盖提升，逐步把总 branch coverage 提升至 85%。
- 为完整 ego-frame safety 迁移建立独立 SDD、兼容测试和回滚策略。

## 9. 2026-08-29 日志

- 完成 Stage 1 frame/schema/sync/fail-safe 软件门禁。
- 完成 SafetySupervisor 三态和策略前/后风险指标。
- 新增 M0 1000 样本不可 skip CPU integration gate。
- 完成 Terrain Affordance v1，但保持默认关闭并明确无离线精度声明。
- BC 改为物理分量加权，RSSM 增加方差观测，imagination horizon 收敛到 5。
- 全量回归、unit branch coverage 和 flake8 全部通过；外部环境/数据相关的 6 个 skip 保持诚实记录。
