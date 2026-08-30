# New_ORAD

New_ORAD（Off-Road Autonomous Driving）是面向非结构化越野环境的模块化端到端自动驾驶研究项目，目标涵盖多模态 BEV 感知、隐空间世界模型、模仿学习与强化学习策略、运动学安全过滤、ROS 2 控制及 ONNX/TensorRT 部署。

> 当前状态：核心算法原型与单元测试阶段。项目尚未完成训练数据闭环、CARLA/Gazebo 系统验收和 TensorRT 端侧实现，不应视为可直接用于真实车辆的完整系统。

## 系统架构

```text
RGB / LiDAR / IMU
        │
        ▼
Perception: multi-modal BEV fusion / planned occupancy
        │
        ▼
Affordance: traversability + terrain roughness (optional, disabled by default)
        │
        ▼
World Model: RSSM latent dynamics
        │
        ▼
Policy: behavior cloning + Dreamer-style RL
        │
        ▼
Trajectory [x, y, heading, velocity]
        │
        ▼
Safety: curvature / steering / lateral-accel / collision constraints
        │  SafetySupervisor: NORMAL / DEGRADED / EMERGENCY_STOP
        │
        ▼
ROS 2 / Pure Pursuit / Ackermann control
        │
        ▼
CARLA / Gazebo / ONNX / TensorRT / vehicle deployment
```

设计基线见 [System_overview.md](./System_overview.md)，实际实现状态以审计过程文档和自动化测试为准。

## 目录结构

```text
New_ORAD/
├── src/
│   ├── utils/          # 公共数据类型、Observation schema 与 frame 变换
│   ├── perception/     # Camera/LiDAR/IMU BEV 融合
│   ├── affordance/     # 可通行性与地形粗糙度辅助任务
│   ├── world_model/    # 目标世界模型包；当前主要实现仍位于 policy
│   ├── policy/         # RSSM、BC/RL、actor/critic、越野 reward
│   ├── safety/         # 自行车模型、安全过滤与失效监督状态机
│   ├── sim/            # CARLA 传感器和闭环助手
│   ├── deployment/     # PyTorch → ONNX
│   ├── orad_ros2/      # Pure Pursuit 与 ROS 2 控制节点
│   └── cpp/            # TensorRT C++ API 框架
├── scripts/            # 评估 CLI
├── tests/              # unit / integration / closed_loop
├── configs/            # 系统、训练与部署配置
├── launch/             # 计划中的 ROS 2 launch，当前为空
├── docs/               # Phase 文档、部署指南与 SDD 审计
└── System_overview.md  # 系统级 SDD
```

## 模块与实际状态

| 模块 | 已有能力 | 主要缺口 |
|---|---|---|
| `utils` | 公共类型、`Observation`、ego/world frame 变换、点云 BEV 投影 | schema version 与跨进程消息契约 |
| `perception` | 简化 LSS、PointPillars-lite、IMU GRU 融合、Occupancy head | 真实标定、监督数据与精度指标 |
| `affordance` | Traversability/Roughness 双头、masked loss、IMU 弱标签助手 | 数据标签、F1/IoU、离线泛化；默认未接入主链 |
| `world_model` | SDD 已规划 | 包为空；RSSM/WorldModel 仍在 `policy` |
| `policy` | 分量加权 BC、RSSM、world-model loss/统计、5-step imagination、actor/critic、reward | dataloader、训练脚本、有效 checkpoint、ADE/RL 指标 |
| `safety` | 自行车重积分、运动学限制、碰撞截断、SafetySupervisor/fail-safe | CasADi NMPC、C++ 对等实现、环境验收 |
| `sim` | CARLA 同步传感器、Observation、runner、风险/干预指标 | 真实 CARLA/Gazebo 闭环验收 |
| `deployment` | 感知/策略 ONNX 导出 | ORT/TRT 数值回归、TensorRT engine |
| `orad_ros2` | Pure Pursuit、Ackermann 节点代码 | 完整依赖、节点集成测试与 launch |

## Phase 进度

以下为 2026-08-28 SDD 审计的实证口径：

| Phase | 范围 | 判断 |
|---|---|---|
| Phase 1 | 类型、几何、安全基础 | 部分完成，约 75% |
| Phase 2 | 多模态 BEV / Occupancy | 部分完成，约 60% |
| Phase 3 | IL + RL + World Model | 部分完成，约 45% |
| Phase 4 | Safety + ROS 2 | 部分完成，约 55% |
| Phase 5 | 闭环、Sim-to-Real、ONNX/TensorRT | 部分完成，约 30% |

## 安装

```bash
python -m pip install -e ".[dev,perception]"
```

按需安装可选能力：

```bash
python -m pip install -e ".[deploy]"
python -m pip install -e ".[optimization]"
```

CARLA、ROS 2、Gazebo、CUDA 和 TensorRT 与系统版本高度相关，建议分别在仿真、训练和车载部署环境中安装。

## 测试

```bash
python -m pytest -v
```

2026-08-29 方法框架改进后实测：

```text
127 collected
121 passed
6 skipped
16 warnings
```

仅运行 unit 并启用 branch coverage 的基线：

```text
119 passed, 2 skipped, 16 warnings
Total branch coverage: 80.62%
```

当前仍需重点提升 sim/ROS 2/deployment 的环境路径覆盖。详细边界缺口见 [TDD 审计报告](./docs/TDD_AUDIT_2026-08-28.md)。

分层运行：

```bash
python -m pytest tests/unit -v
python -m pytest -m integration -v
python -m pytest -m closed_loop -v
```

CPU integration 已有两条不可 skip 的 Green contract（全链路和 M0 1000 样本接口门禁）。其余 integration/closed-loop skip 需要专家日志、危险边界数据集、CARLA 越野地图或 Gazebo 悬挂环境。只有这些用例真实执行并达到以下阈值，才可声明系统级完成：

| 层级 | SDD 目标 |
|---|---|
| Unit | 几何、预处理、动力学和安全约束正确，关键分支覆盖 |
| Open-loop | 专家轨迹 ADE < 0.3 m，危险边界拦截覆盖率 100% |
| Closed-loop | 成功率 > 90%，碰撞率 0，姿态过载报警 0 |

## 核心接口

```python
from perception.bev_fusion import BEVFusion, BEVFusionConfig
from policy.hybrid_policy import HybridPolicy, HybridPolicyConfig
from safety.kinematic_filter import SafetyFilter, SafetyFilterConfig

perception = BEVFusion(BEVFusionConfig())
policy = HybridPolicy(HybridPolicyConfig())
safety = SafetyFilter(SafetyFilterConfig())
```

主数据 shape 契约：

```text
images:     (B, N_camera, 3, H, W)
points:     (B, N_point, 4) [x, y, z, intensity]
imu:        (B, T, 6) [ax, ay, az, gx, gy, gz]
BEV:        (B, C, H=y, W=x)
trajectory: (B, N, 4) [x, y, heading, velocity]
```

## ONNX 导出

```bash
PYTHONPATH=src python -m deployment.onnx_export \
  --out-dir exports \
  --num-cameras 3 \
  --image-size 192 \
  --perception-ckpt path/to/bev.pt \
  --policy-ckpt path/to/policy.pt
```

当前导出只证明 ONNX 图可生成并通过 checker，不代表完成 ONNX Runtime/TensorRT 数值与性能验收。

## CARLA 闭环入口

```bash
python scripts/evaluate_carla_closed_loop.py \
  --host 127.0.0.1 --port 2000 \
  --episodes 10 --max-steps 1000 \
  --goal-x 80 --goal-y 0 \
  --policy-ckpt path/to/policy.pt \
  --policy-frame ego --device cuda
```

P0 代码整改已经覆盖 IMU 6D、occupancy/world frame、按 sensor frame 同步和空轨迹制动；正式闭环仍须在 CARLA 环境完成同步、timeout、车辆制动和指标验收。

## 当前最高优先级

- 在真实 CARLA 环境验收同 frame 传感器同步、时间戳 skew、timeout 与最大制动。
- 准备 recorded replay/hazard dataset，关闭 M0 数据门禁和危险边界验收。
- 为 Terrain Affordance 建立监督标签与 F1/IoU 门槛，达标前保持默认关闭。
- 建立 BC dataloader/checkpoint/ADE 基线，再推进更长 horizon 的 world-model imagination。
- 增加 PyTorch→ONNX Runtime→TensorRT 数值一致性与目标硬件时延验收。

完整 P0/P1/P2 清单见最新审计报告。

建议 TDD 质量门槛：

```text
全项目 branch coverage：不得低于 79% 门槛，当前 80.62%，逐步提升至 85%
核心 safety branch coverage：≥95%
新增/修改代码 coverage：≥95%
P0 fail-safe 分支：100%
CPU integration contract tests：不得 skip
```

## 文档索引

- [技术原理与代码架构](./docs/技术原理与代码架构.md)
- [系统 SDD](./System_overview.md)
- [Phase 1/2 过程报告](./docs/PHASE2_PROGRESS_REPORT.md)
- [Phase 3/4 过程报告](./docs/PHASE3_PHASE4_PROGRESS.md)
- [Phase 5 部署指南](./docs/PHASE5_DEPLOYMENT_GUIDE.md)
- [2026-08-28 SDD 深度审计](./docs/SDD_AUDIT_2026-08-28.md)
- [2026-08-28 TDD 严格检验与覆盖率盘点](./docs/TDD_AUDIT_2026-08-28.md)
- [2026-08-28 P0/P1/P2 整改报告](./docs/SDD_REMEDIATION_2026-08-28.md)
- [2026-08-29 方法框架 SDD/TDD 跟进记录](./docs/METHOD_FRAMEWORK_REMEDIATION_2026-08-29.md)

## 定期审计约定

后续报告使用 `docs/SDD_AUDIT_YYYY-MM-DD.md`，记录代码基线、测试实跑、跳过项、接口变化、P0/P1/P2、Phase 成熟度和交付证据。状态统一分为：

```text
代码骨架 → 单元验证 → 离线验证 → 仿真闭环 → 实车验证
```

只有具备对应层级的可复现证据，才将能力标为该层级完成。

## 安全声明

本项目当前用于研究、原型验证和工程开发。未经完整 SIL/HIL、封闭场地测试、车辆级功能安全分析和独立安全审查，不得直接用于开放道路或载人车辆控制。
