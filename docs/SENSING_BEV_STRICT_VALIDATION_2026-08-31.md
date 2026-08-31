# Sensing/BEV 严格双模态有效性整改记录（2026-08-31）

## 1. 变更结论

本轮将在线 BEV 感知从“至少一个模态可用”收紧为“Camera 与 LiDAR 必须同时
有效”。`modality_mask` 的稳定契约为 `(B,2)`，列顺序固定为：

```text
[:, 0] = camera_valid
[:, 1] = lidar_valid
```

每个在线样本只接受 `[True, True]`。`[True, False]`、`[False, True]` 和
`[False, False]` 均抛出 `ValueError`，闭环 Runner 捕获后生成紧急轨迹并输出
最大制动。该规则取代 2026-08-30 文档中的单模态在线降级策略。

## 2. 有效性边界

Camera/LiDAR 任一满足下列条件即判定整帧无效：

- `modality_mask` shape 不是 `(B,2)`，包含 NaN/Inf、非 0/1 值或任一项为 0；
- Camera tensor 不是 `(B,N,C,H,W)` 的静态配置 shape，或包含 NaN/Inf；
- LiDAR tensor 不是 `(B,P,lidar_in_channels)`，`P == 0`，或包含 NaN/Inf；
- Camera、LiDAR、IMU batch 不一致。

零填充只用于非空点云对齐静态部署 shape，不再把空 LiDAR payload 解释为有效
点云。`modality_mask=None` 表示 Camera/LiDAR 已通过调用边界校验；它不是缺失
模态旁路。

## 3. 数据流变化

```text
CARLA synchronized sensors
        │
        ▼
Observation schema
  ├─ Camera non-empty / finite / synchronized
  └─ LiDAR non-empty / finite / synchronized
        │ invalid
        └──────────────► Runner exception boundary ─► maximum brake
        │ valid
        ▼
build_perceiver
  └─ modality_mask = [[True, True]]
        │
        ▼
BEVFusion input validation
        │ invalid
        └──────────────► Runner exception boundary ─► maximum brake
        │ valid
        ▼
Camera LSS BEV ─┐
                 ├─ strict ModalityAware Conv Fusion ─► BEV/Occupancy
LiDAR pillar BEV ┘
```

`CarlaClosedLoopRunner` 不再在 `Observation` 与感知模型之间把空点云先补成固定
长度零数组；原始且已验证的点云直接交给感知适配器，由 `BEVFusion` 内部完成
非空点云的 pad/truncate。

## 4. TDD 证据

Red Stage：

- camera-only、lidar-only 和 all-missing mask 未被拒绝；
- Camera/LiDAR NaN/Inf 可进入 encoder；
- `(B,0,4)` 空点云可被补零并生成 occupancy；
- FakeRunner 会调用感知函数处理空 LiDAR。

Green/Refactor：

- `ModalityAwareBEVFuser` 要求每个 mask 样本两项均有效；
- `BEVFusion` 在编码前检查非空 LiDAR 与 Camera/LiDAR finite；
- `Observation` 拒绝空点云，FakeRunner 验证该分支不调用模型并最大制动；
- CARLA perceiver 显式传入 `(1,2)` bool `[True, True]`；
- `bev_fusion.py` 补充坐标、静态 frustum、点云补齐、LSS、mask、IMU 和
  fail-fast 关键注释；公开 import 与 `fuser.0.weight` 等 checkpoint key 未改变。

实测结果：

```text
Focused BEV/schema/runner: 46 passed, 1 skipped
ONNX + CPU pipeline + runner: 7 passed
Full suite: 168 collected, 162 passed, 6 skipped, 16 warnings
Unit suite: 160 passed, 2 skipped, 16 warnings
Unit branch coverage: 85.92% (repository gate: 79%)
bev_fusion.py: 99%; encoders.py/fusion.py: 100%
flake8 --max-line-length=99: PASS
```

6 个 skip 仍对应 CARLA/Gazebo 物理环境、专家/危险数据、CUDA 和完整 ROS 2；
本轮 CPU/Fake 结果不替代这些验收。

## 5. 部署边界与后续事项

- 当前 ONNX 保持历史四输入契约，不新增 mask tensor 输入；导出路径等价于
  双模态已验证。由于 NaN/Inf 判定是数据相关分支，ONNX 图内不执行该 Python
  eager 校验，ORT/TensorRT/C++ adapter 必须提供等价输入健康门禁。
- 当前“合理性”只覆盖 non-empty、shape、batch 与 finite，不判断遮挡、过曝、
  点云稀疏度、时间陈旧度或标定漂移；这些需要传感器健康诊断和数据阈值。
- 如未来需要 camera-only/lidar-only 研究，必须建立独立的离线实验契约、
  checkpoint/指标与安全策略版本，不能复用当前在线 mask 绕过双模态门禁。

下一步优先级：

- P0：在真实 CARLA adapter 与未来 ROS 2/C++ adapter 接入帧龄、频率、点数、
  图像饱和度和标定版本健康状态，并验证任一失败触发最大制动。
- P1：为 ONNX Runtime/TensorRT 外围 validator 增加数值一致性和坏帧回归。
- P2：建立 recorded replay 的传感器故障数据集与误拒绝/漏拒绝统计。

## 6. 2026-08-31 Stage 1A 后续状态

本文件第 5 节中的“图外健康 validator 尚待实现”已由 Stage 1A CPU 软件整改
关闭：`SensorHealthGate` 现已位于 CARLA raw packet 与 `Observation` 之间，覆盖
曝光/冻结、LiDAR 稀疏/范围/重复、IMU 频率/窗口/跳变、age/skew 和标定版本，
并在任一失败时绕过模型、记录原因和最大制动。

最新证据为 `232 passed / 6 skipped`、branch coverage `87.79%`，1000-case
错误接受 0、最大制动率 100%、p95 `0.773 ms`。这只关闭 Stage 1A；ORT/TRT/C++
图外 parity、真实 CARLA 与 recorded replay 仍待验收。完整记录见
`SENSOR_HEALTH_STAGE1A_2026-08-31.md`。
