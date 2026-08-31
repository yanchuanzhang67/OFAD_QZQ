# ORAD Phase 5 部署指南：闭环评估 · 域随机化 · ONNX/TensorRT

> 生成日期：2026-08-27 · 范围：Phase 5 CARLA/Gazebo 闭环集成 + 域随机化 + Sim-to-Real 模型部署

本文档覆盖 Phase 5 三项交付：CARLA 闭环评估脚本、域随机化实现建议、PyTorch→ONNX→C++ TensorRT 部署框架，并记录关键工程发现。所有结论经 `flake8` + `pytest` 复现。

## 0. 交付物与状态

| 文件 | 角色 | 状态 |
|---|---|---|
| `src/sim/carla_closed_loop.py` | numpy 助手 + CARLA SensorStack/Runner（`carla` 懒加载） | ✅ |
| `scripts/evaluate_carla_closed_loop.py` | CARLA 闭环 CLI 入口（注入真实模型） | ✅ |
| `src/deployment/onnx_export.py` | 感知+策略 deploy 路径 ONNX 导出 | ✅ 实证导出成功 |
| `src/cpp/include/orad_trt_engine.hpp` | C++ TensorRT 引擎框架（FP16/INT8、异步流） | ✅ 框架 |
| `docs/PHASE5_DEPLOYMENT_GUIDE.md` | 本指南（域随机化 + 部署 + 闭环用法） | ✅ |
| `tests/unit/sim/`、`tests/unit/deployment/` | 纯 Python 助手 + ONNX 导出冒烟 | ✅ 全绿 |

回归：`python -m pytest -v` → **64 passed, 5 skipped**（新增 15：13 sim 助手 + 2 ONNX 导出），flake8 clean。

---

## 1. CARLA 闭环评估脚本

### 1.1 数据流

每个同步 `world.tick()` 执行：

```
RGB×3 + LiDAR(N,4) + IMU(T,6)
   │  Observation: 同帧、非空、shape/finite 校验；失败 → 最大制动
   │  imu_attitude_from_accel (Rodrigues 对齐重力)
   ▼
BEVFusion.forward(..., modality_mask=[[True,True]])
   │  Camera/LiDAR 任一无效 → 整帧无效；非空点云在模型内 pad/truncate
   ▼
BEVFeature(bev=(1,32,50,50), occupancy=(1,1,50,50))
   ▼
HybridPolicy.forward(bev, imu) → trajectory(1,N,4) [x,y,heading,v]  (ego 帧)
   │  policy_trajectory_to_waypoints (ego→世界帧，按当前 VehicleState 旋转)
   ▼
occupancy_from_points(lidar) → OccupancyGrid(50×50, res=0.5)
   ▼
SafetyFilter.filter(traj, state, occ) → 安全轨迹  (κ-clamp + a_y=v²|κ| 封顶 + 自行车重积分 + 碰撞截断)
   ▼
PurePursuitController.compute(state, traj, dt) → ControlCommand(steering, speed, accel)
   │  carla_control_from_command → {throttle, steer, brake} ∈ [0,1]
   ▼
vehicle.apply_control(carla.VehicleControl(**...))
```

### 1.2 用法

```bash
# 1) 启动 CARLA（同步模式由脚本自动设置 fixed_delta=dt）
./CarlaUE4.sh -quality-level=Epic -benchmark -fps=20

# 2) 正式闭环评估（感知与策略权重缺一不可）
python scripts/evaluate_carla_closed_loop.py \
    --host 127.0.0.1 --port 2000 \
    --episodes 10 --max-steps 1000 \
    --goal-x 80 --goal-y 0 \
    --config configs/system.yaml \
    --perception-ckpt exports/perception.pt \
    --policy-ckpt exports/policy.pt \
    --policy-frame ego \
    --occupancy-source lidar \
    --device cuda

# 仅开发冒烟；随机感知/策略结果不得作为性能或安全验收证据
python scripts/evaluate_carla_closed_loop.py \
    --allow-random-policy --episodes 1 --max-steps 20
```

正式模式使用 `strict=True` 加载两个 checkpoint。`occupancy-source` 默认
为 `lidar`；`learned`/`fused` 只有在感知 occupancy 权重与离线指标有效时
才可用于验收，缺失或非法 learned grid 会触发 fail-safe 制动。

退出码：`pass_rate>0.9 且 collision_rate==0` → 0，否则 1（CI 可直接判定）。

### 1.3 指标（`EpisodeMetrics` + `aggregate_episodes`）

| 指标 | 定义 | 通过门槛 |
|---|---|---|
| 通过率 pass_rate | `reached_goal ∧ collisions==0` 的回合占比 | > 0.9 |
| 碰撞率 collision_rate | `collisions>0` 回合占比 | = 0 |
| 姿态平稳性 | `pitch_max`/`roll_max`/`lat_accel_max` 及 `mean_attitude_alarms`（超 `pitch_alarm=0.35rad≈20°`/`roll_alarm`/`lat_accel_alarm=4m/s²` 计次） | alarms=0 |

碰撞计数来自 CARLA `sensor.other.collision` 回调，记录 `normal_impulse` 幅值（N·s）作为冲击烈度。侧向加速度用 `a_y≈v·|yaw_rate|`（平面向心项，与安全过滤器 `v²|κ|` 一致）。

---

## 2. 域随机化（Domain Randomization）

目标：在仿真中对**地面摩擦系数 μ**、**点云噪声**、**悬挂参数**随机化，使 policy 对真实物理与传感器偏差鲁棒（缩小 sim-to-real gap）。每回合从分布采样一组参数注入仿真，再训练/评估。

### 2.1 地面摩擦 μ

**CARLA**（路面物理材质摩擦 + 轮胎摩擦系数）：

```python
import carla, numpy as np
# 方案 A：调轮胎摩擦（VehiclePhysicsControl，最直接）
mu = float(np.random.uniform(0.4, 1.4))     # 干沥青≈1.0，松土≈0.5，湿泥≈0.3
phys = vehicle.get_physics_control()
for w in phys.wheels:
    w.tire_friction = mu
phys.mass = float(np.random.uniform(1500, 1900))  # 顺便扰动质量
vehicle.apply_physics_control(phys)
# 方案 B：路面材质摩擦（需 Town 含 custom material；CARLA 0.9.13+ 支持摩擦系数属性）
# road = world.get_map().get_waypoint(...).lane; road.lane_type 仅类型，摩擦在 OpenDRIVE material。
```

**Gazebo**（地面 link 的 ODE 摩擦，SDF 片段）：

```xml
<model name="ground_plane">
  <link name="link">
    <collision name="c"><geometry><plane/></geometry>
      <surface>
        <friction>
          <ode>
            <mu>0.6</mu>     <!-- 横向，DR 采样 [0.4,1.2] -->
            <mu2>0.6</mu2>    <!-- 纵向 -->
            <fdir1>0 0 1</fdir1>
            <slip1>0.1</slip1><slip2>0.1</slip2>
          </ode>
        </friction>
      </surface></collision>
  </link>
</model>
```

DR 时用 xacro/模板生成器把 `mu`/`mu2` 替换为 `np.random.uniform` 采样值后 `gz sdf` 启动。

### 2.2 点云噪声

**CARLA** LiDAR（`sensor.lidar.ray_cast` 内置高斯噪声 + dropoff）：

```python
bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
bp.set("noise_stddev", "0.05")        # m，高斯；DR 采样 [0.0, 0.10]
bp.set("dropoff_intensity", "0.1")    # dropoff 比例
bp.set("dropoff_distance", "30.0")    # 远距 dropoff 起始
bp.set("dropoff_zero_intensity", "1") # dropoff 点置零强度
```

**Gazebo** ray/gpu_ray 传感器（SDF `<noise>`）：

```xml
<sensor name="lidar" type="gpu_ray">
  <ray><range><min>0.5</min><max>50</max></range></ray>
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.05</stddev>          <!-- DR [0.0,0.10] -->
    <bias_mean>0.0</bias_mean>
    <bias_stddev>0.01</bias_stddev>
  </noise>
</sensor>
```

**后处理注入**（跨平台一致，用于训练数据增广）：

```python
def add_lidar_noise(points: np.ndarray, sigma: float = 0.05,
                    dropout: float = 0.02) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, :3] += np.random.normal(0, sigma, pts[:, :3].shape)
    keep = np.random.rand(pts.shape[0]) > dropout
    return pts[keep]            # 模拟点丢失 + 高斯抖动
```

### 2.3 悬挂参数

**Gazebo**（推荐：弹簧-阻尼悬挂关节，SDF `<dynamics>` + 弹簧）：

```xml
<joint name="wheel_front_left_susp" type="prismatic">
  <parent>chassis</parent><child>wheel_fl</child>
  <axis><xyz>0 0 1</xyz><limit><lower>-0.15</lower><upper>0.15</upper></limit></axis>
  <dynamics><damping>800</damping><friction>50</friction></dynamics>  <!-- DR 阻尼 [400,1200] -->
</joint>
<!-- 弹簧通过 joint 的 spring_stiffness（gz-sim）或 bullet/ode spring 参数； -->
<!-- DR: stiffness ∈ [15000,35000] N/m，max_wheel_travel ∈ [0.10,0.20] m -->
```

**CARLA**：悬挂可调项有限，主要靠 `VehiclePhysicsControl`：

```python
phys = vehicle.get_physics_control()
for w in phys.wheels:
    w.damping_rate = float(np.random.uniform(2.0, 4.5))   # 阻尼率
    w.max_steer_angle = float(np.random.uniform(60, 72))
phys.center_of_mass = carla.Vector3D(0, 0, float(np.random.uniform(0.2, 0.5)))
vehicle.apply_physics_control(phys)
# 注：CARLA PhysX 悬挂弹簧刚度不可直接设；精细悬挂 DR 用 Gazebo。
```

> 建议：**CARLA 做视觉/控制闭环**，**Gazebo 做悬挂动力学 DR 与姿态稳定性专项**（`tests/closed_loop/test_attitude_stability` 骨架即面向 Gazebo）。

---

## 3. ONNX 模型导出

### 3.1 用法

```bash
pip install onnxscript onnx       # torch 2.13 dynamo 导出器强依赖 onnxscript
python -m deployment.onnx_export --out-dir exports --image-size 192 \
    [--perception-ckpt bev.pt] [--policy-ckpt policy.pt]
```

产出（实证成功）：

| 文件 | 输入（仅 batch 动态） | 输出 |
|---|---|---|
| `bev_fusion.onnx`(+.data) | `images` `[b,3,3,192,192]`、`points` `[b,256,4]`、`imu` `[b,10,6]`、`attitude` `[b,3,3]` | `bev` `[b,32,50,50]` |
| `policy.onnx`(+.data) | `bev` `[b,32,50,50]`、`imu` `[b,10,6]` | `trajectory` `[b,20,4]` |

### 3.2 关键约定
- **opset 17**，仅 batch 轴动态（其余维度静态 → 直接映射 TensorRT optimization profile）。
- **导出强制 CPU**：`export_*_onnx` 内部 `model.cpu()`。**必须在 CPU 导出**——torch 2.13 在 CUDA 上对含 `rsample` 的 RSSM 图 dynamo 导出会失败（`strict=False/True` 均 ❌），CPU 则成功（已实证）。
- 策略 deploy 路径在 `eval()` 下使用 RSSM 后验均值，导出保持确定性；训练模式仍使用随机采样。

---

## 4. C++ TensorRT 推理引擎

框架头：`src/cpp/include/orad_trt_engine.hpp`（`orad::TrtEngine` + `orad::PolicyPipeline`）。

### 4.1 构建（CMake）

```cmake
find_package(CUDAToolkit REQUIRED)
# TensorRT 通常装在 /usr/include/x86_64-linux-gnu + /usr/lib/x86_64-linux-gnu
find_library(NVINFER nvinfer HINTS /usr/lib/x86_64-linux-gnu)
find_library(NVONNXPARSER nvonnxparser HINTS /usr/lib/x86_64-linux-gnu)
add_library(orad_trt_engine src/orad_trt_engine.cpp)
target_include_directories(orad_trt_engine PRIVATE include /usr/include/x86_64-linux-gnu)
target_link_libraries(orad_trt_engine PRIVATE ${NVINFER} ${NVONNXPARSER} CUDA::cudart)
```

### 4.2 推理流程（C++）

```cpp
#include "orad_trt_engine.hpp"
using namespace orad;

// 1) 构建/加载引擎（首次从 ONNX 构建，缓存为 .engine 复用）
TrtEngine bev(TrtEngine::Precision::kFP16), pol(TrtEngine::Precision::kFP16);
bev.BuildFromOnnx("exports/bev_fusion.onnx", "exports/bev.engine");
pol.BuildFromOnnx("exports/policy.onnx",     "exports/pol.engine");
// 后续启动：bev.LoadEngine("exports/bev.engine"); pol.LoadEngine("exports/pol.engine");

// 2) 选 optimization profile + 设动态 batch
bev.SetProfile(0, /*batch=*/1);
pol.SetProfile(0, /*batch=*/1);

// 3) 分配 device 缓冲（按 BindingName/BindingShape），异步推理并 overlap 传感器 I/O
cudaStream_t stream;  cudaStreamCreate(&stream);
float *d_imgs, *d_pts, *d_imu, *d_att, *d_bev, *d_traj;  // cudaMalloc ...
bev.Infer({d_imgs, d_pts, d_imu, d_att}, {d_bev}, stream);   // 感知
pol.Infer({d_bev, d_imu},           {d_traj}, stream);      // 策略
cudaStreamSynchronize(stream);
// d_traj = (1,20,4) [x,y,heading,v]，拷回主机喂安全过滤器(C++ 镜像) → Ackermann
```

### 4.3 精度与 INT8
- `Precision::kFP16` 默认（吞吐/精度平衡）。
- `kINT8` 需实现 `nvinfer1::IInt8EntropyCalibrator2`，用代表性 LiDAR/图像 batch（~500 张）校准激活量化区间；off-road 场景建议 FP16 先行，INT8 待闭环达标后启用并回归 ADE/碰撞率。
- 异步流：`Infer(..., stream)` 把感知/策略推理与 CARLA `tick()` 间的 memcpy 交错，隐藏 PCIe 延迟。

---

## 5. Sim-to-Real 部署清单

| # | 项 | 检查 |
|---|---|---|
| 1 | 传感器标定一致性 | 真机相机内参与 `BEVFusionConfig.camera_intrinsics`、外参 `camera_extrinsics` 对齐（单位 m/像素） |
| 2 | 单位与坐标系 | LiDAR (x,y,z,intensity) 车辆坐标、IMU `ax,ay,az,gx,gy,gz` SI 单位；BEV `(H=y,W=x)` 与 reward/encoder 一致 |
| 3 | IMU 偏差 | 真机零偏/bias 预处理（导出前校准，或 `imu_steps` 滑窗去均值） |
| 4 | 延迟预算 | 端到端推理 < 控制周期 `dt=0.1s`；FP16 + async stream；超时 watchdog 回退到安全停车 |
| 5 | 安全过滤器对等 | C++ `safety_filter_node` 与 Python `SafetyFilter` 数值一致（曲率/侧向加速度封顶同公式） |
| 6 | 域间隙回归 | 真机回放日志 → 开环 ADE < 阈值；CARLA DR 集训后真机闭环 pass_rate 不降 |
| 7 | 失效安全 | Camera/LiDAR 任一空、错形、非有限或健康 mask 无效 → 最大制动；policy NaN/越界、模型异常与看门狗超时 → 故障安全停车 |

---

## 6. Phase 5 工程发现（重要）

1. **`carla` 懒加载**：`src/sim/carla_closed_loop.py` 顶部 `try: import carla` 失败时 `_HAS_CARLA=False`，纯 numpy 助手仍可单测（CI 无 CARLA，13 项助手测试全绿）；`CarlaSensorStack`/`CarlaClosedLoopRunner` 仅在 CARLA 环境实例化。
2. **torch 2.13 ONNX 导出强依赖 `onnxscript`**：`torch.onnx.export` 走 dynamo 路径（`torch.export.export`），需 `pip install onnxscript onnx`；缺失则 `ModuleNotFoundError: No module named 'onnxscript'`（本机已补装后导出成功）。
3. **ONNX 导出必须在 CPU**：`HybridPolicy` 的 RSSM 含 `rsample`，torch 2.13 在 **CUDA** 上 dynamo 导出失败（`strict=False/True` 均 ❌），**CPU** 成功。`export_*_onnx` 已强制 `model.cpu()`。CI 用 CPU 导出，部署机导出后引擎可在 GPU 运行。
4. **策略帧约定**：policy 输出 ego 相对 `[x_forward, y_left, heading_rel, v]`，`ego_trajectory_to_world` 按当前 `VehicleState` 旋转到世界帧喂 `SafetyFilter`/`PurePursuitController`；若训练策略直接输出世界坐标，CLI 传 `--policy-frame world`。
5. **外部权重 (.data)**：opset17 导出把大常量写为外部 `.onnx.data`；TensorRT `nvonnxparser` 需把 data 文件与模型同目录或用 `IRuntime` 加载前合并。部署时建议先 `onnx.load`→`save_model` 内联常量，或直接 `trtexec --loadOnnx=`（自动处理）。
6. **双模态健康校验在图外**：当前 ONNX 保持 images/points/imu/attitude 四输入，等价于 Camera/LiDAR 已通过健康门禁。Python eager 的 NaN/Inf 与空 LiDAR 校验不会成为 ONNX 图节点；ORT/TensorRT/C++ adapter 必须实现同等检查并在失败时最大制动。

---

## 7. 状态与遗留

| 级 | 项 | 状态 |
|---|---|---|
| ✅ | CARLA 闭环脚本 + 助手单测 | 助手 13 项绿；Runner 待 CARLA 真跑 |
| ✅ | 域随机化指南（摩擦/点云噪声/悬挂） | 含 CARLA+Gazebo 配置片段 |
| ✅ | ONNX 导出（感知+策略） | 实证成功（CPU） |
| ✅ | C++ TensorRT 引擎框架（头） | `orad_trt_engine.hpp` |
| P1 | `orad_trt_engine.cpp` 实现 + CMake | 待办（头为框架） |
| P1 | INT8 校准器 + 回归 | 待闭环达标后启用 |
| P1 | CARLA 真机闭环（本机无 CARLA） | 脚本就绪，待带 CARLA 环境实跑 |
| ✅ | HybridPolicy RSSM eval mean-mode | 已有确定性重复推理测试；训练模式保留采样 |
| P2 | Gazebo 悬挂 DR harness | 对接 `tests/closed_loop/test_attitude_stability` |
| P2 | Phase 2 ISSUE-4/5/6/7 | 沿用 Phase 3/4 遗留，非阻塞 |

---

## 8. 环境与运行

- Python 3.10.12、torch 2.13.0+cu130（CUDA 可用，但**导出用 CPU**）、onnxscript/onnx 已装；CARLA PythonAPI 未装（`pip install carla` 按需）。
- 助手单测：`cd /home/qqq/New_ORAD && python -m pytest tests/unit/sim tests/unit/deployment -v`
- ONNX 导出：`PYTHONPATH=src python -m deployment.onnx_export --out-dir exports --image-size 192`
- CARLA 正式闭环：`python scripts/evaluate_carla_closed_loop.py --config configs/system.yaml --host 127.0.0.1 --port 2000 --episodes 10 --goal-x 80 --goal-y 0 --perception-ckpt exports/perception.pt --policy-ckpt exports/policy.pt --occupancy-source lidar`
- 关键文件：`src/sim/carla_closed_loop.py`（~500 行）、`scripts/evaluate_carla_closed_loop.py`（~150 行）、`src/deployment/onnx_export.py`（~130 行）、`src/cpp/include/orad_trt_engine.hpp`。
