# Source Package Map

`src/` 按运行时数据流分层。跨层共享契约只放在 `utils`，系统装配只放在
`configuration`；业务模块不得反向依赖 `scripts` 或具体仿真入口。

```text
utils + configuration
        │
        ├─ perception ── affordance
        ├─ policy / world_model
        ├─ safety
        ├─ sim
        ├─ replay
        ├─ orad_ros2
        └─ deployment / cpp
```

| Package | Responsibility | Stable entry point |
|---|---|---|
| `utils` | schema、frame、公共类型、几何契约、前模型传感器健康门禁 | `utils.types`, `utils.schema`, `utils.sensor_health` |
| `configuration` | 严格读取唯一系统 YAML 并装配全栈 | `load_system_stack` |
| `perception` | 独立传感器编码、BEV fusion、occupancy | `BEVFusion` |
| `affordance` | traversability/roughness 辅助任务 | `TerrainAffordanceHead` |
| `policy` | BC、RSSM、world model、actor/critic | `HybridPolicy` |
| `world_model` | 目标独立边界；当前实现仍在 `policy` | 待迁移 |
| `safety` | supervisor、运动学投影、碰撞过滤 | `SafetySupervisor`, `SafetyFilter` |
| `sim` | simulator adapter、健康门禁编排、闭环 runner、DR | `CarlaClosedLoopRunner` |
| `replay` | recorded CARLA episode 只读校验、HealthGate-first 离线编排与证据 | `CarlaRecordedEpisode` |
| `orad_ros2` | controller 与 ROS 2 adapter | `PurePursuitController` |
| `deployment` / `cpp` | ONNX 与目标 TensorRT runtime | `onnx_export` / 待实现 |

## Perception internal boundary

```text
encoders.py       camera depth lifting backbone + IMU encoder
bev_fusion.py     static geometry, LiDAR scatter, model facade, occupancy head
fusion.py         explicit camera/LiDAR availability and BEV Conv fuser
```

外部代码继续从 `perception.bev_fusion` 导入 `BEVFusion` 和
`BEVFusionConfig`。内部拆分不得破坏该入口或已有 checkpoint key；新 backend
必须先对齐 BEV `(B,C,H=y,W=x)`、frame、resolution 与有限值契约。

## Recorded replay boundary

`replay.carla_dataset` 只读加载带 `_SUCCESS` 的 episode，并在解码前校验 schema、
Canonical、frame/timestamp、路径边界和传感器文件。历史 manifest 缺失的 provenance
必须显式 opt-in 并保留 gaps；不得从当前源码反推历史车辆或配置。recorded replay
属于离线软件证据，不向 CARLA/车辆发送控制，也不替代仿真闭环验收。
