# OffTerSim / UniAD 参考架构学习与整改记录（2026-08-30）

> 参考范围：官方仓库 README、运行配置、训练说明和核心调度源码  
> 参考项目：[OffTerSim](https://github.com/dvij542/OffTerSim)、[UniAD](https://github.com/OpenDriveLab/UniAD)  
> 原则：吸收架构思想，不复制外部源码；保持 New_ORAD 的越野场景、安全边界和轻量依赖
>
> 同日 BEVFusion 与四项目仓库结构后续见
> `docs/BEVFUSION_REPOSITORY_REMEDIATION_2026-08-30.md`。

## 1. 结论

本轮从 OffTerSim 吸收“仿真场景参数化、Gym/ROS 2 双接口、传感器与命令边界明确”的思想，从 UniAD 吸收“规划导向的分层任务依赖、显式 occupancy→planning 数据流、分阶段训练和任务指标独立”的思想。

结合 New_ORAD 当前成熟度，本轮只实施两个可被 CPU TDD 证明的改进：

1. `configs/system.yaml` 从说明性文件升级为严格、唯一的运行配置装配源。
2. learned occupancy 不再被 runner 丢弃，安全层可显式选择 `lidar`、`learned` 或保守 `fused` 数据源。

没有引入 UniAD 的检测/跟踪/地图 query 体系，也没有直接绑定 OffTerSim/Unity ML-Agents；当前缺少相应数据、算力和环境证据，贸然照搬会扩大依赖与接口风险。

## 2. 参考项目要点与采用决策

| 来源 | 可学习设计 | New_ORAD 决策 |
|---|---|---|
| OffTerSim | Unity 程序化 heightmap、rocks/trees/grass 随机场景 | 作为后续 terrain scenario profile 输入，不复制 Unity 工程 |
| OffTerSim | `params.yaml` 描述 heightmap、spawn、sensor selection | 采用单一严格 YAML 装配思路；本轮先覆盖现有 CARLA stack |
| OffTerSim | ML-Agents/Gym executable 与 ROS 2 `cmd/pose/twist/accel/scan/pcl` 接口 | 后续建立 backend-neutral adapter；不复制全局命令变量和硬编码 PointCloud2 布局 |
| UniAD | perception→motion→occupancy→planning 分层依赖 | learned occupancy 显式进入 Safety/Planning，而不是生成后被丢弃 |
| UniAD | task loss 命名、权重与分阶段训练 | 保留为训练流水线设计：先 BEV/Affordance，再 BC/WorldModel/E2E |
| UniAD | BEV encoder 可替换，但 `bev_embed/bev_pos` shape 必须一致 | New_ORAD 继续使用 `validate_stack_configs` 和严格 YAML 派生 shape |
| UniAD | planning/occupancy 独立指标和 checkpoint 复现 | 正式闭环同时要求 perception/policy checkpoint；learned occupancy 启用前必须有 IoU/风险证据 |
| UniAD 公开修复经验 | frame 方向、planning cumsum、冻结策略会影响复现 | 不隐式转换 frame；不吞掉 loss NaN；所有迁移必须绑定 Red tests |

## 3. 配置数据流整改

新增 `configuration.system.load_system_stack()`：

```text
configs/system.yaml
        │ strict section/key validation
        ▼
SystemStackConfig
        ├─ BEVFusionConfig
        ├─ HybridPolicyConfig
        ├─ SafetyFilterConfig
        ├─ PurePursuitConfig
        ├─ ClosedLoopConfig
        ├─ TerrainAffordanceConfig
        └─ DomainRandomizationConfig
        │
        ▼
validate_stack_configs()
```

约束：

- 未知 section/key、缺失 section/key、非法二维范围和不存在文件均 fail-fast。
- BEV H/W 由物理范围和 resolution 派生，不允许手工重复配置。
- CLI 只允许覆盖运行模式 `policy_frame` 和 `occupancy_source`；几何、shape、车辆极限仍以 YAML 为唯一来源。
- `evaluate_carla_closed_loop.py` 已使用该装配器，不再自行散落构造模块默认配置。

## 4. Planning-oriented Occupancy 数据流

整改前：

```text
BEVFusion → BEVFeature.bev → Policy
          └→ BEVFeature.occupancy → 丢弃

raw LiDAR → occupancy_from_points → SafetyFilter
```

整改后：

```text
BEVFusion → BEVFeature {bev, learned_occupancy}
        │                    │
        │                    ├─ learned
        │                    └─ max(lidar, learned) = fused
        ▼
HybridPolicy → world Trajectory
        │
        ▼
SafetySupervisor → selected OccupancyGrid → SafetyFilter → Controller
```

运行模式：

| `occupancy_source` | 行为 | 当前使用条件 |
|---|---|---|
| `lidar` | 使用原始 LiDAR 几何占用 | 默认、当前已验证基线 |
| `learned` | 使用 `(1,1,H,W)` learned probability | 必须有有效 perception checkpoint 和离线 IoU/风险证据 |
| `fused` | 对 LiDAR/learned 栅格逐 cell 取最大值 | 保守研究模式；仍需标定和闭环验收 |

learned occupancy 必须为 batch 1、单 channel、与 BEV 几何一致、finite 且范围 `[0,1]`；缺失或非法时 runner 进入最大制动。

## 5. CARLA 正式评估守卫

正式模式现在同时要求：

```text
--perception-ckpt path/to/bev.pt
--policy-ckpt path/to/policy.pt
```

两者均使用严格 state-dict 加载。以前只要求 policy checkpoint，但 perception 可能仍为随机权重，这不满足正式评估定义。只有显式 `--allow-random-policy` 才允许随机 perception/policy 冒烟。

## 6. TDD 记录

Red Stage：

- `configuration.system` 不存在，配置装配测试 collection 失败。
- learned occupancy 转换、三种 occupancy route 和非法值拒绝接口不存在。

Green/Refactor：

- 新增严格配置装配器及 100% branch coverage。
- 新增 learned occupancy→world-aware `OccupancyGrid` 转换和显式 route。
- FakeRunner 连续使用 policy payload 和 learned occupancy，验证 raw risk 被安全层消除并输出最大制动。
- FakeRunner 覆盖 sensor timeout、模型异常和必需 learned occupancy 缺失。

最终证据：

```text
Full suite: 142 collected, 136 passed, 6 skipped, 16 warnings
Unit suite: 134 passed, 2 skipped, 16 warnings
Unit branch coverage: 85.27% (gate: 79%)
configuration/system.py: 100%
sim/carla_closed_loop.py: 70% (此前 49%)
flake8: PASS
```

6 个 skip 仍对应真实 CARLA/Gazebo、专家/危险数据、CUDA 和完整 ROS 2，没有使用 Mock 冒充环境或数据验收。

## 7. 尚未实施

- OffTerSim executable/ML-Agents adapter、heightmap profile 和 ROS 2 topic bridge。
- OffTerSim、CARLA、Gazebo 统一的 backend protocol 与 scenario manifest。
- UniAD 风格的两阶段训练执行器、task loss registry 和 checkpoint lineage。
- Affordance→Policy 接入；在 F1/IoU/MAE 达标前继续默认关闭。
- learned/fused occupancy 的真实标定、IoU、危险边界和 CARLA 闭环验收。
- ONNX occupancy 输出及 TensorRT safety runtime。

## 8. 下一阶段 Action Items

### P0

- 使用训练 perception checkpoint 对 `lidar/learned/fused` 做同一 replay 的 IoU、漏检率和 post-safety risk 对比。
- 在真实 CARLA 验证三种 occupancy route、传感器 timeout 和实际 brake。
- 保存 config hash、两个 checkpoint hash、occupancy source 与 episode metrics。

### P1

- 定义 simulator-neutral `reset/step/close` protocol 和 `ScenarioManifest`。
- 新增 OffTerSim adapter：heightmap、spawn、sensor selection、`cmd/pose/twist/accel/pcl` 映射到 ORAD schema。
- 实现 Stage A（BEV/Affordance）→Stage B（BC）→Stage C（WorldModel/RL）的训练配置与 checkpoint lineage。

### P2

- 导出 occupancy ONNX output，并完成 PyTorch↔ORT 数值对齐。
- 将 ROS 2 裸数组轨迹升级为带 header/frame/schema version 的强类型消息。
- 对 external reference 的 commit/tag、许可证和适用边界建立可复现 manifest。

## 9. 变更文件

- `src/configuration/{__init__,system}.py`
- `configs/system.yaml`
- `src/sim/carla_closed_loop.py`
- `scripts/evaluate_carla_closed_loop.py`
- `src/deployment/onnx_export.py`（同步确定性 eval 导出说明）
- `tests/unit/configuration/test_system_config.py`
- `tests/unit/sim/test_carla_helpers.py`
- `tests/unit/sim/test_carla_runner.py`
- `README.md`、`System_overview.md`、`docs/技术原理与代码架构.md`
- `docs/{SDD_AUDIT,TDD_AUDIT,SDD_REMEDIATION}_2026-08-28.md`
- `docs/PHASE5_DEPLOYMENT_GUIDE.md`、`Agents.md`
