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
