## 越野非结构化场景下基于 IL + RL 的端到端自动驾驶系统设计文档 (SDD)

> 2026-08-29 实现状态：系统已加入严格 `Observation`/frame/timestamp 契约、SafetySupervisor 三态 fail-safe、1000 样本 M0 CPU 门禁，以及默认关闭的 Terrain Affordance v1。本文描述目标设计；成熟度与验收证据以 `docs/METHOD_FRAMEWORK_REMEDIATION_2026-08-29.md` 为准。

### 1. 系统概述 (System Overview)
1.1 研发背景与目标
非结构化越野环境（Off-Road Unstructured Environments）具有无标准车道线、高低起伏地形、复杂障碍物（如乱石、泥泞、灌木）以及强动态扰动等特征。传统基于规则的规划控制算法在面对复杂越野动力学与非确定性地形时适应性不足。

本系统旨在研发一套结合 模仿学习（Imitation Learning, IL） 与 强化学习（Reinforcement Learning, RL） 的模块化端到端（Modular E2E）自动驾驶系统:
- 模仿学习（IL）：利用专家驾驶数据快速初始化策略，完成基础地形选择与避障能力的构建。
- 强化学习（RL）：在世界模型（World Model）隐空间或仿真环境中，结合车辆平稳性、地形通过性与安全约束，进一步迭代优化策略，解决协变量偏移（Covariate Shift）问题。
- 部署与迁移目标：系统基于算法模块与软硬件解耦设计，首先在 CARLA（算法与传感器验证）和 Gazebo/ROS 2（物理动力学与传感器仿真）中完成闭环验证，最终通过 Sim-to-Real 部署于车载端侧硬件单元


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
                             [ 混合策略网络 (IL + RL) ]
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
    - 融合 IMU 历史动力学状态，消除 Causal Confusion（因果混乱）。

2) 表征与世界模型层 (Representation & World Model)
- Terrain Affordance 辅助表征：从 BEV 独立预测可通行概率与非负粗糙度。该分支必须先通过真实标签的 F1/IoU/MAE 验收；默认关闭，未达标前不得作为安全硬判据。
- 越野 Latent Dynamics World Model：基于隐空间预测（Dreamer / RSSM 架构），根据当前状态与候选动作，预测未来环境隐状态变迁以及地形冲击度。
- 提供离线与隐空间想象（Imagination Rollouts）能力，大幅提升强化学习样本效率。

3) 混合策略学习层 (Hybrid Policy Network)
- IL 阶段 (Behavior Cloning / Inverse RL)：采用离线数据预训练，将 BEV 特征与车体姿态作为输入，输出候选行驶轨迹或控制指令。
- RL 阶段 (Model-Based RL / PPO / SAC)：在仿真环境或世界模型中，引入自定义越野 Reward 函数：$$\text{Reward} = R_{\text{progress}} - w_1 \cdot R_{\text{collision}} - w_2 \cdot R_{\text{attitude\_instability}} - w_3 \cdot R_{\text{jerk}}$$

    - $R_{\text{progress}}$：沿目标方向的前进距离。
    - $R_{\text{attitude\_instability}}$：Pitch / Roll 角速度及 Z 轴加速度冲击惩罚（保障平稳性与防翻车）。
    - $R_{\text{jerk}}$：车辆加速度变化率惩罚，减少颠簸感。

4) 安全控制校验与端侧执行层 (Safety & Control)
- SafetySupervisor：在确定性安全过滤前检查 trajectory frame/shape/finite/速度，以及 sensor age/skew 与 model latency；输出 NORMAL、DEGRADED 或 EMERGENCY_STOP。
- Safety Filter（安全过滤层）：针对端到端输出的轨迹，接入基于 Kinematic Bicycle Model 或 CasADi 非线性轨迹优化的二次校验层，硬性截断超限转向角与碰撞危险轨迹。
- 底层控制：输出阿克曼转向角（Steering Angle）、加速度/制动信号，连接 ROS 2 底层驱动；空/非法/超时轨迹必须输出最大安全制动。

## 3.测试驱动开发 (TDD) 规划与实施矩阵

| 测试层级 | 测试目标 / 范围 | 核心测试用例 (Test Cases) | 自动化验证标准 |
| ---- | ---- | ---- | ---- |
| 单元测试 (Unit Test) | 坐标转换、传感器预处理、安全过滤器、动力学约束算法 | 1. 验证 LiDAR 点云至 BEV 投影张量的维度与内存对齐。<br>2. 输入超限转向角，验证 SafetyFilter 截断与重投影逻辑。 | pytest / gtest 测试通过率 100%；算子执行时间满足实时性限制。 |
| 集成测试 (Integration Test) | 感知‑策略‑安全层串联，离线数据回放 (Open‑Loop) | 1. 给定离线专家日志，验证策略输出轨迹与专家轨迹的 ADE / FDE。<br>2. 验证碰撞检测模块对危险边界的 100% 拦截率。 | 开环预测 ADE < 0.3m；离线测试套件安全过滤覆盖率 100%。 |
| 闭环仿真测试 (Closed‑Loop Test) | CARLA / Gazebo 环境下复杂越野地形闭环行驶 | 1. 连续 20° 陡坡及非结构化乱石路段通过性测试。<br>2. 越野行驶过程中的平稳性（Pitch/Roll 抖动均方差值）。 | 场景成功率（Success Rate）> 90%；碰撞率为 0；车体姿态过载报警次数为 0。 |

2026-08-29 自动化基线：全量 `127 collected / 121 passed / 6 skipped`；unit branch coverage `80.62%`。两条 CPU integration contract 不允许 skip；专家日志、危险数据、CARLA/Gazebo、完整 ROS 2 与目标硬件验收不得由 Mock 结果替代。

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
