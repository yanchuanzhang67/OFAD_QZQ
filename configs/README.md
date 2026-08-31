# Configuration Contract

`system.yaml` 是当前唯一运行配置源。代码通过
`configuration.system.load_system_stack()` 派生各模块 dataclass，并执行跨模块
BEV、IMU、车辆极限和控制周期校验。

- 禁止在本目录复制第二套相同物理参数。
- CLI 只允许覆盖运行模式，例如 `policy_frame` 和 `occupancy_source`。
- 未知 section/key、缺失字段和不一致 shape 必须 fail-fast。
- `sensor_health` 只保存健康阈值；Camera 数量、图像尺寸、IMU 窗口和标定版本
  由 `sensors` 派生到 `SensorHealthConfig`，不得复制第二份 shape 配置。
- 训练实验的未来配置应只保存优化器、数据划分、随机种子和 checkpoint lineage，
  并引用系统配置 hash，不复制车辆或 BEV 几何。
