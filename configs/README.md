# Configuration Contract

`system.yaml` 是当前唯一运行配置源。代码通过
`configuration.system.load_system_stack()` 派生各模块 dataclass，并执行跨模块
BEV、IMU、车辆极限和控制周期校验。

- 禁止在本目录复制第二套相同物理参数。
- CLI 只允许覆盖运行模式，例如 `policy_frame` 和 `occupancy_source`。
- 未知 section/key、缺失字段和不一致 shape 必须 fail-fast。
- `sensor_health` 只保存健康阈值；Camera 数量、图像尺寸、IMU 窗口和标定版本
  由 `sensors` 派生到 `SensorHealthConfig`，不得复制第二份 shape 配置。
- `carla_baseline` 固定 CARLA `0.9.16`、`Town10HD_Opt`、
  `vehicle.lincoln.mkz_2020`、seed `42`，以及三路具名 Camera、LiDAR、IMU 的
  blueprint、安装位姿、FOV/采样参数。控制周期、相机数量/图像尺寸和标定版本继续
  分别由 `control`、`sensors` 派生，并由 loader 校验传感器 tick/shape 与其一致。
- Camera 顺序固定为 `[front,rear,top]`；其 FOV 为 `[90,90,100]°`，top pitch
  为 `-15°`。LiDAR 固定 32 channels、50 m、320000 point/s、10 Hz、
  `[-25,15]°`，IMU 固定 `10×6 @ 10 Hz`。
- 旧的 `camera_fov_degrees` 和扁平 `lidar_*` schema 已被显式拒绝；迁移时必须完整
  提供 `cameras`、`lidar`、`imu`，不得静默使用旧默认值。
- `canonical_yaml_sha256()` 对解析后的 YAML 作稳定序列化，用于比较逻辑配置；原始
  文件 SHA-256 仍作为字节级证据，两者不得混用。
- 训练实验的未来配置应只保存优化器、数据划分、随机种子和 checkpoint lineage，
  并引用系统配置 hash，不复制车辆或 BEV 几何。
