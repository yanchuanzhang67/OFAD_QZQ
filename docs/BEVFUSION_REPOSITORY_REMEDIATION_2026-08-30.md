# 四项目架构复核与 BEVFusion/仓库整改记录（2026-08-30）

> 状态说明（2026-08-31）：本文保留 8 月 30 日的历史设计与测试证据；其中
> camera-only/lidar-only 在线降级契约已被
> [`SENSING_BEV_STRICT_VALIDATION_2026-08-31.md`](./SENSING_BEV_STRICT_VALIDATION_2026-08-31.md)
> 取代。当前在线链路要求 Camera 与 LiDAR 同时有效。

> 官方参考：[ADLab BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion)、
> [MIT HAN Lab BEVFusion](https://github.com/mit-han-lab/bevfusion)、
> [OffTerSim](https://github.com/dvij542/OffTerSim)、
> [UniAD](https://github.com/OpenDriveLab/UniAD)  
> 原则：吸收模块边界与验证方法，不复制外部源码，不引入 MMDetection3D/Unity
> 等重型依赖，不把论文指标当作 New_ORAD 的实测指标。

## 1. 结论

New_ORAD 原有 Camera LSS、LiDAR PointPillars-lite、IMU conditioning 和 BEV
concat-conv 方向与两套 BEVFusion 的共同原则一致，但 `bev_fusion.py` 同时承担
encoder、geometry、fusion、head 和 deployment，缺少显式传感器可用性接口，
仓库也没有清晰区分当前文档、历史日志和代码包职责。

本轮完成两类可由 CPU TDD 证明的改进：

1. Camera/LiDAR encoder 继续独立进入统一 BEV，新增显式
   `modality_mask (B,2)` 和可复用 `ModalityAwareBEVFuser`。
2. 新增 `docs/src/tests/configs` 四个导航 README，建立当前基线、历史记录、
   包职责、测试层级和唯一配置源的阅读入口。

## 2. 四项目设计映射

| 项目 | 学习点 | New_ORAD 决策 |
|---|---|---|
| ADLab BEVFusion | Camera stream 不依赖 LiDAR query；通过故障训练验证 LiDAR malfunction robustness | 保持 camera LSS 与 LiDAR scatter 独立；增加显式 modality mask，不用隐式零张量表达故障 |
| ADLab BEVFusion | Camera→LiDAR→Fusion 的分阶段训练与 BEV augmentation | 记录为训练流水线 P1；没有数据与指标前不宣称完成 |
| MIT BEVFusion | encoder/vtransform/fuser/decoder/head 分层；共享 BEV 支持多任务 | 拆出 `encoders.py` 与 `fusion.py`，保留轻量 model facade 和 occupancy head |
| MIT BEVFusion | 简单 concat + Conv fuser、统一 BEV 几何、优化 BEV pooling | 保留 concat-conv 与静态 scatter；不引入自定义 CUDA pooling，继续保证 CPU/ONNX 可测 |
| UniAD | perception→motion/occupancy→planning 的显式任务依赖 | 保持 `BEVFeature {bev, occupancy}`，policy 与 safety 分别消费明确字段 |
| OffTerSim | simulator 参数、sensor/command interface 与算法解耦 | 保持 simulator adapter 位于 `sim`；未来 backend 不得进入 perception 内部 |

MIT 仓库目前是只读归档，因此本轮把它作为架构与部署参考，而不是可持续跟随的
上游依赖。两个 BEVFusion 仓库均为 Apache-2.0，但本轮没有复制其实现。

## 3. BEV fusion 整改

### 3.1 代码边界

```text
perception/
├── encoders.py       ImageEncoder + ImuEncoder
├── fusion.py         ModalityAwareBEVFuser
└── bev_fusion.py     config、frustum/scatter、LiDAR encoder、model facade、head
```

外部稳定入口仍为：

```python
from perception.bev_fusion import BEVFusion, BEVFusionConfig
```

`ImageEncoder`/`ImuEncoder` 的父属性名和 fuser 的 Sequential 参数编号保持不变，
所以历史 key 如 `image_encoder.backbone.*`、`imu_encoder.gru.*`、
`fuser.0.weight` 不因本次拆分变化。

### 3.2 显式缺模态路径

```text
camera images → independent LSS BEV ─┐
                                     ├─ mask → concat/Conv → + IMU → BEV
LiDAR points → independent pillar BEV┘
```

`modality_mask[:,0]` 表示 camera，`[:,1]` 表示 LiDAR。约束：

- shape 必须是 `(B,2)`；
- 值必须 finite 且为 0/1；
- 每个样本至少一个模态可用；
- mask 只门控 BEV feature，不改变 BEV 几何和输出 shape；
- 默认 `None` 保持历史双模态 concat-conv 路径。

该接口用于训练故障注入、camera-only/lidar-only 消融和 adapter 健康状态映射。
它不授权安全层在 LiDAR 故障时继续运行；是否降级或急停仍由 Observation、
SafetySupervisor、occupancy source 和系统安全策略决定。

### 3.3 输入契约

`BEVFusion.forward` 现在 fail-fast 检查：

- images `(B,N,C,H,W)` 的 camera 数、channel 和静态分辨率；
- points `(B,P,lidar_in_channels)`；
- IMU `(B,imu_steps,imu_channels)`；
- 三路 batch 一致；
- attitude `(B,3,3)`。

有限值与传感器同步仍由上游 `Observation` 负责，避免把数据依赖控制流写入静态
ONNX 图。

## 4. 仓库导航整改

| 入口 | 作用 |
|---|---|
| `README.md` | 面向使用者的项目状态、安装、测试和入口 |
| `docs/README.md` | 当前基线、历史审计、Phase 文档与设计输入分类 |
| `src/README.md` | 包职责、单向依赖和 perception 内部边界 |
| `tests/README.md` | unit/integration/closed-loop 的验收含义 |
| `configs/README.md` | 唯一系统配置源与未来实验配置边界 |
| `Agents.md` | DOX 工作契约与质量门禁，不替代产品文档 |

本轮没有大规模移动历史文件，避免破坏 IDE 链接和审计路径。后续只有在链接检查、
文档索引和 Git rename 能同时保证时，才按 `architecture/audits/guides/archive`
物理迁移文档。

## 5. TDD 证据

Red Stage：

- 新测试导入 `perception.encoders` 时 collection 失败，明确证明模块边界不存在。
- modality mask、单模态路径和输入 shape 拒绝接口均不存在。

Green/Refactor：

- encoder/fuser 拆分后原 BEV、CPU pipeline 与 ONNX 测试保持 Green。
- 新增 camera-only/lidar-only 等价性、mask shape/finite/boolean/all-missing、
  feature batch/spatial/channel 和 model camera/point/IMU/batch/attitude 测试。
- 新增模块 coverage：`encoders.py` 100%，`fusion.py` 100%；
  `bev_fusion.py` 99%。

最终基线：

```text
Full suite: 157 collected, 151 passed, 6 skipped, 16 warnings
Unit suite: 149 passed, 2 skipped, 16 warnings
Unit branch coverage: 85.77% (gate: 79%)
flake8: PASS
```

6 个 skip 仍是 CARLA/Gazebo、专家/危险数据、CUDA 和完整 ROS 2 环境验收，
没有因新增单模态软件路径而降低真实验收要求。

## 6. 未实施与风险边界

- 没有引入 MMDetection3D registry、检测 head、radar encoder 或自定义 CUDA
  BEV pooling；它们与当前轻量越野规划目标和 CPU CI 不匹配。
- 没有建立 modality dropout 数据分布、故障概率或训练指标；mask 目前是工程接口，
  不是鲁棒性结果。
- ONNX 当前仍导出历史四输入双模态路径；modality mask 作为第五输入需要 ORT/TRT
  profile、C++ binding 与数值回归后才能改变部署契约。
- learned occupancy、camera-only/fused safety 仍需离线标定、危险边界和真实闭环证据。

## 7. Action Items

### P0

- 定义传感器健康状态如何生成 modality mask，并明确 LiDAR 丢失时不同
  `occupancy_source` 的急停/降级矩阵。
- 使用 recorded replay 对双模态、camera-only、lidar-only 做 occupancy IoU、
  false-negative 和 post-safety risk 对比。

### P1

- 建立 Camera pretrain → LiDAR pretrain → Fusion/Occupancy finetune 的训练配置、
  checkpoint lineage 和 BEV augmentation 一致性测试。
- 将 modality mask 纳入训练 batch schema，增加可复现 sensor dropout manifest。
- 为 ONNX/ORT 增加可选 mask 输入方案与 PyTorch 数值对齐，确认后再变更 C++ 接口。

### P2

- 评估 voxel/near-field priority 取代当前前 N 点截断。
- 在不破坏历史链接的前提下逐步把文档物理归档到稳定子目录。
- 对四个参考仓库记录 commit/tag、许可证和适用边界 manifest。

## 8. 变更文件

- `src/perception/{encoders,fusion,bev_fusion}.py`
- `tests/unit/perception/test_bev_fusion_contracts.py`
- `docs/README.md`、`src/README.md`、`tests/README.md`、`configs/README.md`
- `README.md`、`System_overview.md`、`docs/技术原理与代码架构.md`
- SDD/TDD/参考架构日志与 `Agents.md`
