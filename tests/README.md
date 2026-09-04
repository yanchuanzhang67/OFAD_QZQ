# Test Layout

```text
unit/         纯 CPU/GPU 可选的算法、shape、异常与 fail-safe 契约
integration/  不依赖外部服务的跨模块数据流；不得 skip
closed_loop/  CARLA/Gazebo 等真实物理环境验收
fixtures/     数据 schema 说明与后续最小可复现样本
```

单元测试按源包建立子目录；历史 `unit/test_bev_fusion.py` 暂保留以避免一次性
移动造成审计噪声，新 BEV 契约放在 `unit/perception/`。后续修改遵循：

1. 先运行目标测试并保存基线。
2. 新行为先增加失败测试（Red）。
3. 最小实现转 Green，再执行全量回归和 branch coverage。
4. FakeRunner 只能证明软件编排；真实 CARLA/Gazebo/ROS 2/TRT 测试不得替代。

Stage 1A 新增 `unit/utils/test_sensor_health.py`、Runner 健康故障注入和
`integration/test_sensor_health_gate.py`。1000-case CPU matrix 能证明确定性坏帧
检测与软件最大制动编排，不能替代真实传感器、执行器或车辆验收。

Stage 1B Task 2 新增 `unit/sim/test_carla_sensor_health_stream.py`，只验证正常流
统计、连续性和 Exit Gate 算法。真实 1000-tick CARLA 验收位于
`closed_loop/test_carla_closed_loop.py`，缺少 CARLA server/run-id 时必须明确 skip。

2026-09-03 新增 `unit/replay/`，覆盖只读 episode、路径逃逸、legacy provenance
gap、HealthGate-first 零模型调用、严格 checkpoint 模式、最大制动和事务化 artifact。
`integration/test_recorded_carla_network_smoke.py` 使用 CPU 真实 BEVFusion/HybridPolicy
验证 recorded frame 的 shape/finite 数据流；随机权重不验证驾驶质量。真实 200 帧
artifact 是离线验证，不能替代 `closed_loop/` 中仍需 CARLA server 的 Gate。

2026-09-04 新增 `unit/training/`：B0 Pure BC 采用严格专家 manifest、train-only
动力学归一化、冻结且 strict-loaded 的 BEVFusion、有限 loss/gradient、不可覆盖
checkpoint 与逐 epoch 审计日志。fixture/smoke 只能验证训练软件链路，产物必须记录
`model_performance_valid=false`；没有 1000+ 专家样本、三 seed 和冻结 test 证据时，
不得把 B0 提升为离线验证。
