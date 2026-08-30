# ORAD Phase 3+4 过程文档：进度与遗留问题

> 生成日期：2026-08-27 · 范围：Phase 3 混合策略网络 + Phase 4 运动学安全过滤器与 ROS 2 接口

本文档记录 ORAD 第三、四阶段实际完成情况与环境/工程发现，作为第五阶段衔接依据。所有结论均经代码审查 + `pytest` 复现。

---

## 1. 总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 | 骨架 `utils/{types,transforms}` + `safety/{bicycle_model,kinematic_filter}` | ✅ 完成 |
| Phase 2 | `perception/bev_fusion.py` 多模态融合 | ✅ 完成 |
| Phase 3 | `policy/{offroad_reward,hybrid_policy}`（BC + Dreamer-lite RSSM + λ-return RL） | ✅ 完成 |
| Phase 4 | `safety/kinematic_filter.py` 增强 + `orad_ros2/vehicle_control_node.py` | ✅ 完成 |

测试：`python -m pytest -v` → **50 passed, 5 skipped**（4 个 CARLA/Gazebo/数据集骨架 + 1 个 ROS 2 节点因 `ackermann_msgs` 未装而 skip），无 failure。

---

## 2. Phase 3 — 混合策略网络（`src/policy/`）

### 2.1 模块
- `offroad_reward.py`（~235 行）：`OffRoadRewardConfig`+`OffRoadReward`，4 项可微奖励 R1 进度 / R2 碰撞（`grid_sample`）/ R3 姿态抖动+侧翻 / R4 加加速度。`forward→(B,)+dict`，`per_step→(B,N)`。
- `hybrid_policy.py`（~444 行）：`PolicyEncoder`、`RSSM`（prior `imagine_step`/posterior `observe_step`，平衡 KL `bal=0.8`）、`BevDecoder`、`WorldModel`、`Actor`、`ActionProj`、`Critic`、`HybridPolicy`（BC forward + `imagine`/`actor_critic_loss` λ-return）。
- `policy/__init__.py` 重导出全部公开类。

### 2.2 关键修复
- **碰撞聚合 `max→sum`**：`OffRoadReward.forward` 碰撞项由 `max(-1)` 改 `sum(-1)`，与 `per_step` 对齐，确保 `reward_head` 监督目标与 RL bootstrap 一致。
- flake8 clean（max-line-length 99）。

### 2.3 测试与约定
- `tests/unit/policy/`：8 reward + 6 policy = 14，全绿。端到端 smoke 形状正确。
- BEV 约定 `(B,C,H=y,W=x)`，reward 碰撞采样器与 policy encoder 一致。
- `imagine` 初始状态 detach → actor 梯度仅流经 imagined actions（世界模型仍在图，**训练须独立优化器或冻结**）。
- 想象奖励用 `reward_head`（学习），非直接 `OffRoadReward`（需每步解码，延后）。

---

## 3. Phase 4 — 运动学安全过滤器 + ROS 2 节点

### 3.1 `src/safety/kinematic_filter.py`（增强而非重写）
保留 Phase 1 接口 `clip_steering/clip_accel/check_collision/project_to_feasible/filter/optimize_casadi`，新增硬约束（SDD Phase-4 spec）：

| 约束 | 实现 |
|---|---|
| 路径曲率 `\|κ\|≤tan(δ_max)/L` | `max_curvature` 属性 + `np.clip(κ,±κmax)` |
| 最大转向角 `\|δ\|≤δ_max` | `δ=atan(κ·L)` 再 `clip_steering`（含速率限幅） |
| 最大侧向加速度 `a_y=v²·\|κ\|≤a_y_max` | `v≤√(a_y_max/\|κ\|)` 后纵向 P 项 |
| 最近可行轨迹投影 | 修正 `(δ,a)` 经 `KinematicBicycleModel.rollout` 重积分 → feasible-by-construction |
| 碰撞硬截断 | `filter` 保留占用网格首命中截断 |

新增：`estimate_curvature`（**steering 可用 `κ=tan(δ)/L`；否则 heading 差分/弧长**——支持端到端 policy 仅给 `(x,y,yaw,v)` 的输出）、`check_limits`（per-waypoint 违规诊断）。

### 3.2 `src/orad_ros2/vehicle_control_node.py`（新建）
- `PurePursuitController`（纯 numpy，可单测）：动态前瞻 `L_d=L0+k·v`，`δ=atan2(2L·sinα,L_d)`，速度被侧向加速度 `a_y=v²·|κ|` 封顶，纵向 P 控制。
- `VehicleControlNode`（rclpy **lazy import**）：订阅 `/orad/safe_trajectory`（`Float32MultiArray` `[x,y,yaw,v]*N`）+ `/carla/ego_vehicle/odometry`（四元数→yaw），定时调控制器，按 `simulator` 参数发 CARLA `AckermannDrive` 或 Gazebo `AckermannDriveStamped`。
- `package.xml`（`ament_python`，包名 `orad_vehicle_control`）+ `setup.py`（console script `orad_vehicle_control`）。

### 3.3 工程发现（重要，影响后续）
1. **`ros2` 包名冲突**：环境已装 ROS 2 Humble（`/opt/ros/humble`），系统 `ros2`（ros2cli）在 pytest11 插件加载阶段被 pre-import 并缓存到 `sys.modules`，使单独 `python -c "import ros2"` 解析到 `src/ros2`（src 优先）但 **pytest 收集时命中系统包** → `ModuleNotFoundError`。**解法**：Python 包重命名 `src/ros2`→`src/orad_ros2`（ROS 2 工程惯例：Python 包不以 `ros2` 命名）。文件 `vehicle_control_node.py` 路径与内容不变。
2. **`launch_testing`/`launch_testing_ros_pytest_entrypoint` 耦合**：ROS 自带 launch_testing hook 对每个测试文件调 `find_launch_test_entrypoint`→`pyimport`。禁用其一即 `PluginValidationError`（hook spec 耦合），**不可禁用**。规避靠包重命名。
3. **测试目录无 `__init__.py`**：`tests/unit/*` 各目录（safety/policy/...）**无** `__init__.py`，pytest 用 prepend 裸名导入。给 `tests/unit/orad_ros2/` 误加 `__init__.py` 反而触发包导入失败（父级 `tests/`、`tests/unit/` 非包）→ 已删除。
4. **`ackermann_msgs` 缺失**：`rclpy`/`nav_msgs`/`std_msgs` 已装，但 `ackermann_msgs` 未装（`ros-humble-ackermann-msgs`）→ `_HAS_RCLPY=False` → 节点测试按整体栈 skip（非 fail）。装该包后节点测试自动运行。

### 3.4 测试
- `tests/unit/safety/test_kinematic_filter_curvature.py`：10 项（曲率派生、steering/heading 双路估计、limit 诊断、曲率 clamp、侧向加速度封顶、安全轨迹保持、空轨迹、policy 风格弧、碰撞截断保留）。
- `tests/unit/orad_ros2/test_vehicle_control_node.py`：9 项（8 controller 全绿 + 1 node skip）。

---

## 4. 关键接口

```python
# SafetyFilter（Phase 4 增强）
SafetyFilter(config).max_curvature                 # = tan(max_steering)/wheelbase
SafetyFilter.estimate_curvature(traj) -> np.ndarray        # per-waypoint κ
SafetyFilter.check_limits(traj) -> dict            # curvature/steer/lat_accel + *_violation
SafetyFilter.project_to_feasible(traj, state) -> Trajectory  # κ-clamp + a_y cap + 重积分
SafetyFilter.filter(traj, state, occ=None) -> Trajectory     # 投影 + 碰撞截断

# ROS 2 控制器（numpy-only，可单测）
PurePursuitController(config)
PurePursuitController.lookahead_distance(speed) -> float
PurePursuitController.select_target(state, traj) -> (idx, dist)|None
PurePursuitController.compute(state, traj, dt) -> ControlCommand(steering, speed, accel)

# rclpy 节点（lazy import, _HAS_RCLPY 守卫）
VehicleControlNode(node_name="vehicle_control")   # 参数: traj/odom/cmd topic, simulator, dt
main(args=None)                                    # console entry `orad_vehicle_control`
```

---

## 5. 遗留与下一阶段

| 级 | 项 | 状态 |
|---|---|---|
| P2 | ISSUE-4 Occupancy 预测头 | 待办（policy reward 已用 grid_sample，可选） |
| P2 | ISSUE-5 CasADi NMPC（`optimize_casadi` 仍 fallback） | 占位，Phase 4 用确定性投影替代 |
| P3 | ISSUE-6 ONNX 实证导出 / ISSUE-7 几何双写重构 | 待办 |
| 🆕 | HybridPolicy 训练脚本（双优化器 / 冻结世界模型） | Phase 3 收尾 |
| 🆕 | Dataloader `SensorPacket→BEVFusion→HybridPolicy` | 须对齐 `OffRoadRewardConfig.bev_*` 与 `BEVFusionConfig` 几何 |
| 🆕 | `ackermann_msgs` 安装 + 节点实跑 | 装 `ros-humble-ackermann-msgs` 后节点测试激活 |

**Phase 5 已完成**（2026-08-27，详见 `docs/PHASE5_DEPLOYMENT_GUIDE.md`）：CARLA 闭环评估脚本（`scripts/evaluate_carla_closed_loop.py` + `src/sim/carla_closed_loop.py`，`carla` 懒加载、纯 numpy 助手可单测）、域随机化（CARLA/Gazebo 地面摩擦·点云噪声·悬挂，含配置片段）、ONNX 导出（`src/deployment/onnx_export.py`，感知+策略 deploy 路径实证成功）、C++ TensorRT 引擎框架（`src/cpp/include/orad_trt_engine.hpp`）。回归：**64 passed, 5 skipped**（新增 15 测试：13 sim 助手 + 2 ONNX 导出），flake8 clean。

---

## 6. 环境与运行

- Python 3.10.12、numpy 1.21、torch 2.x（CUDA）、pytest 6.2.5；ROS 2 Humble 已装（`/opt/ros/humble`），`ackermann_msgs` 缺。
- 单测：`cd /home/qqq/New_ORAD && python -m pytest tests/unit/safety tests/unit/orad_ros2 -v`
- 全套：`cd /home/qqq/New_ORAD && python -m pytest -v`（`-p no:anyio` 固化于 `pyproject.toml`）。
- 关键文件：`src/safety/kinematic_filter.py`（~250 行）、`src/orad_ros2/vehicle_control_node.py`（~237 行）、`src/policy/{offroad_reward,hybrid_policy}.py`。
