# New_ORAD SDD P0/P1/P2 整改报告（2026-08-28）

> 基线：`docs/SDD_AUDIT_2026-08-28.md`  
> 方法：Red→Green 回归、CPU contract test、全量 pytest、branch coverage  
> 原则：代码完成与 CARLA/Gazebo/ROS2/TensorRT 环境验收分开记录

## 1. 结果

```text
Full suite: 90 collected, 84 passed, 6 skipped, 16 warnings
Unit branch coverage: 79.85% (minimum gate: 79%)
```

本轮新增 21 个有效测试（整改前 69 collected），新增一个不依赖 CARLA 的 perception→policy→safety→control 集成 contract test。

## 2. 已关闭 P0

| 问题 | 整改 | 证据 |
|---|---|---|
| CARLA IMU 仅 `(T,3)` | `imu_samples_to_array` 合并 accel+gyro，输出 padded `(T,6)` | `test_imu_samples_preserve_accel_and_gyro` |
| occupancy/trajectory frame 错配 | OccupancyGrid 支持 origin yaw；局部 LiDAR grid 按 ego world pose 设置旋转原点 | world pose 与负边界测试、CPU pipeline test |
| 空安全轨迹不制动 | Pure Pursuit 对空轨迹/无 target 返回 `max_decel` | `test_controller_empty_trajectory_stops` |
| 多相机身份/帧混淆 | 每相机独立队列，camera/LiDAR 按 CARLA frame 对齐，并使用 timeout 等待 | adapter 代码完成；待真实 CARLA 验收 |
| 随机/无效模型静默评估 | 正式模式强制 checkpoint 存在、`strict=True`；仅显式 `--allow-random-policy` 可冒烟 | CLI 守卫代码 |
| episode 碰撞累计污染 | 每次 runner 开始调用 `reset_episode()` | runner/adapter 代码 |

## 3. 已完成 P1/P2（当前环境可验收部分）

- BEVFusion 新增 Occupancy 概率头，输出 `[B,1,H,W]` 且范围 `[0,1]`。
- 新增空点云、pad、truncate 三条感知边界回归。
- HybridPolicy 在 `eval()` 使用 RSSM 后验均值，同输入输出确定性一致。
- OffRoadReward 将 BEV 网格外区域保守视为占用，与 SafetyFilter 语义一致。
- OccupancyGrid 增加二维、finite、正 resolution、origin 校验，并正确应用 yaw 与 floor 边界。
- 点云 canonicalization 拒绝非法点数、shape 和 NaN/Inf。
- ONNX checker 不再吞掉模型校验错误；checkpoint 加载改为存在性检查和严格匹配。
- 新增跨模块 `validate_stack_configs()`，启动时检查 BEV/IMU/车辆极限/dt 漂移。
- 新增 `configs/system.yaml` 作为系统级几何、传感器、车辆和 DR 参数基线。
- 新增可复现 domain-randomization 采样和 LiDAR noise/dropout 实现及单测。
- 新增 CPU 全链路 integration contract test，integration 不再全部 skip。
- 新增 `.coveragerc`，branch coverage 低于 79% 时失败。
- 修复原非对称点云落点测试依赖随机 MLP 激活的偶发性。

## 4. 尚未关闭或待环境验收

以下事项不能在当前 CPU 工作区内被诚实地标为完成：

| 级别 | 项目 | 状态/所需条件 |
|---|---|---|
| P0 验收 | CARLA frame 同步与 timeout | 代码完成，需 CARLA PythonAPI/server 实跑 |
| P1 | RSSM/WorldModel 物理拆入独立包 | 未实施；涉及训练 checkpoint/API 迁移，需独立 TDD 变更 |
| P1 | SensorPacket dataloader 与 BC/RL 训练入口 | 未实施；缺专家数据 schema/训练目标 |
| P1 | CasADi NMPC | 未实施；当前仍为 deterministic projection fallback |
| P1 | ROS 2 强类型轨迹消息与 launch | 未实施；需 ROS interface package 和完整消息依赖 |
| P1 | TensorRT `.cpp`、CMake、engine/INT8 | 未实施；需 TensorRT/CUDA SDK 与目标硬件 |
| P1 验收 | PyTorch→ORT→TensorRT 数值回归 | ONNX checker 已强化；ORT/TRT 环境验收待办 |
| P1 验收 | CARLA/Gazebo domain randomization | 采样/点云增强已完成；仿真物理注入待环境测试 |
| P2 | 全部公共 dataclass schema version/单位 | OccupancyGrid 已强化；其他类型仍需协议设计 |
| P2 | 完全消除 NumPy/Torch/sim 几何重复 | 未完成；当前用 contract validator 防漂移 |
| P2 | branch coverage ≥85% / safety ≥95% | 当前总计 79.85%，SafetyFilter 89% |

## 5. 主要文件

- `src/utils/types.py`：旋转 occupancy 与输入校验。
- `src/utils/contracts.py`：跨模块配置契约。
- `src/sim/carla_closed_loop.py`：6D IMU、frame 同步、world occupancy、episode reset。
- `src/sim/domain_randomization.py`：DR manifest 与 LiDAR 增强。
- `src/perception/bev_fusion.py`：Occupancy head。
- `src/policy/hybrid_policy.py`：确定性 eval。
- `src/policy/offroad_reward.py`：网格外保守惩罚。
- `scripts/evaluate_carla_closed_loop.py`：正式评估 checkpoint 守卫与配置验证。
- `configs/system.yaml`、`.coveragerc`：配置与质量门槛。

## 6. 下一轮建议

1. 在 CARLA 环境执行 sensor frame/timeout 和 runner 冒烟，关闭 P0 环境验收。
2. 先为 WorldModel 包迁移写 checkpoint compatibility Red 测试，再拆包。
3. 定义带 `header/frame_id/schema_version` 的 ROS 2 trajectory interface。
4. 准备最小专家 fixture，把 ADE 计算和 dataloader 从 skip 转为 CPU Green。
5. 在具备 ONNX Runtime 的 CI 增加 batch 1/2 数值对齐；TensorRT 留给目标硬件 job。
6. 将总 branch coverage 提升至 85%，SafetyFilter 提升至 95%。

## 7. 日志

### 2026-08-28

- 完成两轮 Red→Green：P0 闭环契约；Occupancy/确定性部署与边界条件。
- 全量测试从 `63 passed, 6 skipped` 提升至 `84 passed, 6 skipped`。
- unit branch coverage 从 79% 基线提升至 79.85%，加入 79% 防回退门槛。
- 明确记录 6 个环境/数据依赖 skip，未将其计为完成。

### 2026-08-29 — 第二轮方法框架整改

本轮依据 `docs/relatetalk.md` 执行“Correctness First → Learning Baseline”顺序：

- P0 软件契约：新增 frame/schema、传感器 timestamp/skew、SafetySupervisor 三态、模型异常/非法轨迹最大制动和 1000 样本 M0 gate。
- P1 学习基线：新增默认关闭的 Terrain Affordance v1、masked multitask loss、IMU roughness 弱标签、物理分量 weighted BC、RSSM prior/posterior std 和 5-step imagination。
- P2 可审计性：新增策略前/安全后风险、intervention/emergency-stop 指标，README 和技术架构文档同步到实测状态。
- 回归提升至 `121 passed, 6 skipped`；unit branch coverage 从 `79.85%` 提升至 `80.62%`。
- 未关闭：真实 CARLA/Gazebo/ROS2/TensorRT、专家/危险数据、Affordance 离线指标、BC/World Model 训练闭环、WorldModel 拆包和 CasADi NMPC。
- 完整 Gate 矩阵、设计边界与下一轮清单见 `docs/METHOD_FRAMEWORK_REMEDIATION_2026-08-29.md`。
