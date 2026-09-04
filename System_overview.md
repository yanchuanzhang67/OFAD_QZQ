## 越野非结构化场景下基于 IL + RL 的端到端自动驾驶系统设计文档 (SDD)

> 2026-09-04 实现状态：系统已加入严格 `Observation`/frame/timestamp 契约、
> SafetySupervisor 三态 fail-safe、唯一 YAML 配置装配和显式 occupancy 安全路由；
> 独立 B0 `pure_bc_v1` 数据/训练/评估/checkpoint/replay 边界达到**单元验证**。
> 正式专家数据、训练权重、CARLA 闭环和部署仍待验收；本文包含目标设计，当前证据
> 以 `docs/SmartSteer_Status.md` 为准。

### 1. 系统概述 (System Overview)
1.1 研发背景与目标
非结构化越野环境（Off-Road Unstructured Environments）具有无标准车道线、高低起伏地形、复杂障碍物（如乱石、泥泞、灌木）以及强动态扰动等特征。传统基于规则的规划控制算法在面对复杂越野动力学与非确定性地形时适应性不足。

本系统旨在研发一套结合 模仿学习（Imitation Learning, IL） 与 强化学习（Reinforcement Learning, RL） 的模块化端到端（Modular E2E）自动驾驶系统:
- 模仿学习（IL）：利用专家驾驶数据快速初始化策略，完成基础地形选择与避障能力的构建。
- 强化学习（RL）：在世界模型（World Model）隐空间或仿真环境中，结合车辆平稳性、地形通过性与安全约束，进一步迭代优化策略，解决协变量偏移（Covariate Shift）问题。
- 部署与迁移目标：系统基于算法模块与软硬件解耦设计，首先在 CARLA（算法与传感器验证）和 Gazebo/ROS 2（物理动力学与传感器仿真）中完成闭环验证，最终通过 Sim-to-Real 部署于车载端侧硬件单元
- 可选仿真扩展：后续以 backend-neutral adapter 接入 OffTerSim/Unity 的程序化越野 heightmap 与 Gym/ROS 2 接口；当前仅完成设计研究，不计入已实现或已验收能力。


### 2系统整体架构设计 (System Architecture)
系统采用 “模块化端到端 (Modular E2E)” 方案，分为四大层级：数据感知层、表征与世界模型层、混合策略学习层 以及 安全控制校验层。

```
[ 传感器输入: RGB + PointCloud + IMU ]
                                          │
                                          ▼
                       [ 多模态融合 & BEV / 3D Occupancy 构建 ]
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
 [ 辅助任务 / 占据网格 / Terrain Affordance ]           [ 越野世界模型(World Model)]
                 │                                                 │
                 └────────────────────────┬────────────────────────┘
                                          ▼
                  [ B0 Pure BC → B1 Affordance+BC →
                    B2 Affordance+RSSM+BC → B3 Dreamer ]
                                          │
                                 [ 采样/预测轨迹 (Trajectory) ]
                                          │
                   [ SafetySupervisor: NORMAL/DEGRADED/EMERGENCY_STOP ]
                                          │
                             [ 越野安全过滤与运动学约束 ]
                                          │
                                 [ 底层控制器 (ACKERMANN/PID) ]
```

2.1 各层级详细设计
1) 数据感知与特征融合层 (Perception & Fusion)
- 输入源
    - RGB 图像（前视/多视角）：获取高维语义特征（如植被类型、泥泞区域、障碍物细节）。
    - LiDAR 点云：获取 3D 几何拓扑、地形起伏与距离信息。
    - IMU & 姿态仪：获取车辆实时 Pitch/Roll/Yaw 角速度与 3 轴加速度。
- 特征提取与融合
    - 多模态特征提取：使用 CNN、Transformer 等模型提取 RGB、点云、IMU 等模态的特征。
    - 特征融合：通过concat、拼接等方式将不同模态特征进行融合，构建高维语义特征表示。
    - 利用 Early/Middle Fusion 提取图像与点云特征，将其映射至统一的 Bird's-Eye-View (BEV) 空间或 3D Occupancy Grid（占据网格）。
    - Camera 与 LiDAR 必须独立编码到同一 BEV 几何后再融合；显式 `(B,2)` mask 固定为 `[camera_valid, lidar_valid]`，在线样本必须两项均有效，任一模态为空、错形或含非有限值时整帧无效并进入故障安全路径，禁止以补零伪装可用。
    - 融合 IMU 历史动力学状态，消除 Causal Confusion（因果混乱）。

2) 表征与世界模型层 (Representation & World Model)
- Terrain Affordance 辅助表征：从 BEV 独立预测可通行概率与非负粗糙度。该分支必须先通过真实标签的 F1/IoU/MAE 验收；默认关闭，未达标前不得作为安全硬判据。
- 越野 Latent Dynamics World Model：基于隐空间预测（Dreamer / RSSM 架构），根据当前状态与候选动作，预测未来环境隐状态变迁以及地形冲击度。
- 提供离线与隐空间想象（Imagination Rollouts）能力，大幅提升强化学习样本效率。

3) 混合策略学习层 (Hybrid Policy Network)
- B0 Pure BC：当前已实现独立确定性 `BCPolicy`，输入 frozen BEV `(B,32,50,50)`、
  IMU `(B,10,6)` 与不含 world pose 的 `ego-dynamics-v1` `(B,8)`，输出 ego-frame
  `(B,20,4)` 轨迹。该路径不经过 Affordance、RSSM 或 Dreamer，并具备严格专家
  manifest、masked 指标、冻结 perception、事务化 checkpoint 与 train/eval CLI。
  当前成熟度为**单元验证**；fixture smoke 不构成模型性能证据。
- IL 阶段 (Behavior Cloning / Inverse RL)：采用离线数据预训练，将 BEV 特征与车体姿态作为输入，输出候选行驶轨迹或控制指令。
- RL 阶段 (Model-Based RL / PPO / SAC)：在仿真环境或世界模型中，引入自定义越野 Reward 函数：$$\text{Reward} = R_{\text{progress}} - w_1 \cdot R_{\text{collision}} - w_2 \cdot R_{\text{attitude\_instability}} - w_3 \cdot R_{\text{jerk}}$$

    - $R_{\text{progress}}$：沿目标方向的前进距离。
    - $R_{\text{attitude\_instability}}$：Pitch / Roll 角速度及 Z 轴加速度冲击惩罚（保障平稳性与防翻车）。
    - $R_{\text{jerk}}$：车辆加速度变化率惩罚，减少颠簸感。

4) 安全控制校验与端侧执行层 (Safety & Control)
- SensorHealthGate：在 `Observation` 和模型计算前检查 Camera/LiDAR/IMU 的
  shape、finite、frame、timestamp/age/skew、曝光、冻结、稀疏度、范围、频率、
  跳变与标定版本；任一模态无效时跳过 perception/policy 并进入最大制动。
- SafetySupervisor：在确定性安全过滤前检查 trajectory frame/shape/finite/速度，以及 sensor age/skew 与 model latency；输出 NORMAL、DEGRADED 或 EMERGENCY_STOP。
- Safety Filter（安全过滤层）：针对端到端输出的轨迹，接入基于 Kinematic Bicycle Model 或 CasADi 非线性轨迹优化的二次校验层，硬性截断超限转向角与碰撞危险轨迹。
- Occupancy dependency：默认使用 LiDAR 几何栅格；只有具备有效感知权重和离线指标时才允许选择 learned 或保守 max-fused 栅格。
- 底层控制：输出阿克曼转向角（Steering Angle）、加速度/制动信号，连接 ROS 2 底层驱动；空/非法/超时轨迹必须输出最大安全制动。

## 3.测试驱动开发 (TDD) 规划与实施矩阵

| 测试层级 | 测试目标 / 范围 | 核心测试用例 (Test Cases) | 自动化验证标准 |
| ---- | ---- | ---- | ---- |
| 单元测试 (Unit Test) | 坐标转换、传感器预处理、安全过滤器、动力学约束算法 | 1. 验证 LiDAR 点云至 BEV 投影张量的维度与内存对齐。<br>2. 输入超限转向角，验证 SafetyFilter 截断与重投影逻辑。 | pytest / gtest 测试通过率 100%；算子执行时间满足实时性限制。 |
| 集成测试 (Integration Test) | 感知‑策略‑安全层串联，离线数据回放 (Open‑Loop) | 1. 给定离线专家日志，验证策略输出轨迹与专家轨迹的 ADE / FDE。<br>2. 验证碰撞检测模块对危险边界的 100% 拦截率。 | 开环预测 ADE < 0.3m；离线测试套件安全过滤覆盖率 100%。 |
| 闭环仿真测试 (Closed‑Loop Test) | CARLA / Gazebo 环境下复杂越野地形闭环行驶 | 1. 连续 20° 陡坡及非结构化乱石路段通过性测试。<br>2. 越野行驶过程中的平稳性（Pitch/Roll 抖动均方差值）。 | 场景成功率（Success Rate）> 90%；碰撞率为 0；车体姿态过载报警次数为 0。 |

2026-08-31 Stage 1A 自动化基线：全量 `238 collected / 232 passed / 6 skipped`；
unit `229 passed / 2 skipped`，branch coverage `87.79%`，新增
`sensor_health.py` 为 `98.04%`。CPU 1000-case 故障矩阵实现错误接受 `0`、最大
制动率 `100%`、p95 `0.773 ms`。专家日志、危险数据、真实 CARLA/Gazebo、完整
ROS 2/C++ 与目标硬件验收不得由 Mock/Fake 结果替代。

2026-09-01 当前软件回归：全量 `267 collected / 259 passed / 8 skipped`；unit
`256 passed / 2 skipped`，branch coverage `88.34%`。新增两个 CARLA Stage 1B
环境用例均保持 skip，未替代真实仿真证据。

2026-09-03 Stage 1B Task 1 的当前 Canonical 已统一为 CARLA `0.9.16`、
`Town10HD_Opt`、Lincoln MKZ 2020、seed `42`、0.1 s 控制周期，以及三路具名
Camera/LiDAR/IMU 的位姿与采样属性。配置 loader 同时生成 Camera optical→CARLA→
New_ORAD 标定矩阵，collector 和 `CarlaSensorStack` 均消费同一结构。该能力当前为
单元验证；只有真实
CARLA 生成的 manifest、三个 repetition trace 和 `reproducible=true` summary
才能把本 Task 提升为仿真闭环验证。

2026-09-01 Stage 1B Task 2 已增加无模型的 CARLA Sensor→HealthGate 正常流评估，
对 1000+ ticks 记录 false rejection、sensor age/skew、frame/IMU continuity 和
health latency。CARLA health reference timestamp 已改为同 tick world snapshot 的
`elapsed_seconds`，避免非零 frame 起点造成错误 stale。当前仍缺真实 CARLA
`summary.json`，不能据 CPU 测试确认现有阈值不会误杀真实正常帧。

2026-09-03 新增只读 recorded replay 边界。现有 CARLA 0.9.16 episode 的 200 帧
完整回放中，199 帧通过 HealthGate 并进入 calibration-aware BEVFusion 与
HybridPolicy，1 帧因启动期 `imu_accel_range + imu_jump` 在模型前拒绝；全部已处理
网络边界 finite。随机模型模式明确不构成模型性能或 CARLA 闭环证据。该链路为
离线验证，artifact 位于 `artifacts/carla_replay/replay_20260903T035626Z_9ca5f754`。
历史 episode 未记录 vehicle/config/calibration hash，报告保留三个 provenance gaps。

2026-09-03 当前软件回归：全量 `292 passed / 8 skipped / 16 warnings`；unit
`275 passed / 2 skipped / 16 warnings`，branch coverage `87.93%`，其中
`sensor_health.py` 为 `98%`。八项 skip 分别依赖真实 CARLA 坡地、Gazebo、Task 1
在线基线、Task 2 在线健康流、专家日志、hazard 数据、CUDA 和完整 ROS 2 环境，
均不计为已验收。

2026-09-04 B0 实施前复核：全量为 `292 passed / 8 skipped /
16 warnings`，unit branch coverage 仍为 `87.93%`。当前 recorded 软件链路 6 个
边界全部连通，但按严格配置、Stage 1A、Stage 1B Task 1/2、M0 数据、训练模型离线
指标、真实 CARLA 模型闭环、部署/车辆定义的 8 个里程碑只关闭 `2/8`。本机
`import carla` 仍失败，因此不提升仿真成熟度。

2026-09-04 B0 Pure BC 实施后新增 `BCPolicyConfig`、`EgoDynamicsV1`、确定性
`BCPolicy`、严格专家数据/指标/训练引擎/checkpoint、train/eval CLI 与 Pure BC
recorded replay 路由。现有 200 帧 Traffic Manager episode 明确
`expert_labels=false`，因此未用于训练，也未生成性能有效 checkpoint。最终软件
回归为 `391 passed / 8 skipped / 16 warnings`；unit 为 `372 passed / 2 skipped /
16 warnings`，全仓 branch coverage `90.29%`。环境 skip 明细见
`docs/SmartSteer_Status.md` 第 6 节。B0 成熟度保持**单元验证**，B1–B3 和训练模型
离线 Gate 未关闭。

## 4.仿真平台搭建与 Sim-to-Real 迁移路线

```
[ CARLA 仿真 ] ──(算法逻辑/多视角RGB融合)──┐
                                          ├──► [ ROS 2 中间件协议层 ] ──► [ 端侧 C++ / TensorRT 引擎 ] ──► [ 实车测试 ]
[ Gazebo 仿真 ] ──(物理动力学/LiDAR/IMU)──┘
```
### 4.1仿真平台分工
1.CARLA：侧重于高保真视觉渲染、多相机 RGB 图像获取以及大范围非结构化地图生成。

2.Gazebo / ROS 2：侧重于精确的车辆多连杆悬挂动力学、底盘接触力、LiDAR 仿真以及 IMU 高频噪声模拟

### 4.2 Sim-to-Real 迁移策略
- Domain Randomization（综合域随机化）：
    - 视觉域：在 CARLA/Gazebo 中随机化光照、天气、地面纹理与植被密度。

    - 动力学域：在 Gazebo 中随机化车体质量、地面摩擦系数（泥地/干土/岩石）、悬挂刚度与传感器安装偏差。
    - 传感器噪声域：向 IMU 和点云中叠加高斯噪声、漂移与丢帧模拟。

- 硬件部署规范：
    - 模型基于 PyTorch 训练与验证。
    - 编译导出为 ONNX 格式，并在端侧（如 NVIDIA Jetson 或车载计算单元）利用 TensorRT (C++) 进行 FP16 / INT8 量化与异步多线程 CUDA Stream 部署。
