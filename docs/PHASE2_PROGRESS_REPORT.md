# ORAD Phase 2 过程文档：进度与遗留问题

> 生成日期：2026-08-27 · 范围：Phase 1 骨架 + Phase 2 BEV 融合模块 · 基于实证代码审查与诊断运行

本文档记录 ORAD（off-road IL+RL 模块化自动驾驶）第二阶段**实际**完成情况与**已确认的 bug / 隐患**，作为第三阶段衔接依据。所有问题均经代码审查 + 诊断脚本复现，非推测。

---

## 1. 总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 | 目录骨架、`utils/{types,transforms}`、`safety/{bicycle_model,kinematic_filter}`、`pyproject.toml`、根 `conftest.py` | ✅ 完成 |
| Phase 2 | `perception/bev_fusion.py` 多模态融合（LSS + PointPillars + IMU），TDD 3 测试点 + 2 回归 | ✅ 完成（BUG-1/ISSUE-3 已于 2026-08-27 修复） |

测试现状：裸 `pytest -v` → **18 passed, 4 skipped**（skip 为预期的 CARLA/expert-log 骨架）。BEV 6 个单测全绿（含 2 个非对称范围回归）。**BUG-1、ISSUE-3 已修复**（见 §5、§9 变更日志）。

---

## 2. 已完成进度

### 2.1 Phase 1 — 骨架与安全核心（纯 numpy，可被 C++ 镜像）
- `src/utils/types.py`：`VehicleState`/`Waypoint`/`Trajectory`/`OccupancyGrid`/`BEVFeature`/`SensorPacket` 数据类，含 `to_array`/`from_array` 序列化。
- `src/utils/transforms.py`：`rotation_matrix_2d`、`world_to_bev_index`、`bev_index_to_world`、`point_cloud_to_bev_tensor`（4 通道体素 BEV：max/mean height + density + intensity）。**几何正确**：x→W(cols)、y→H(rows)。
- `src/safety/bicycle_model.py`：后轴参考点运动学自行车模型，`step`/`rollout`/`is_feasible`，纯 numpy Euler 积分。
- `src/safety/kinematic_filter.py`：`SafetyFilter`（steer/accel 限幅 + 自行车重投影 + 占用网格碰撞硬截断），CasADi 可选回退。

### 2.2 Phase 2 — BEV 多模态融合（`src/perception/bev_fusion.py`）
- **三支融合**：Camera（简化 LSS：CNN→特征图×softmax深度分布→`scatter_add` 投射 BEV）、LiDAR（PointPillars-lite：MLP→`scatter_add`）、IMU（GRU→偏置广播）。
- **TensorRT 友好**：全程静态 shape，无 `nonzero`/数据依赖 reshape，点云固定 `num_points`（补零+valid mask），scatter 索引预计算为 buffer。
- **新接口**（TDD 反推）：`forward(images, points, imu, imu_attitude=None)`、`transform_points_to_world(points, R)`（`einsum` 批量旋转）、`BEVFusionConfig.num_points`、`export_onnx`（仅 batch 轴动态）。

---

## 3. 测试现状

| 测试文件 | 用例 | 结果 |
|---|---|---|
| `tests/unit/test_bev_fusion.py` | 输出 shape / IMU 姿态转换 / batch>1 CPU / batch>1 GPU / 非对称 shape / 非对称落点 | 6 passed（GPU 实跑，CUDA 可用） |
| `tests/unit/safety/test_bicycle_model.py` | 5 项 | 5 passed |
| `tests/unit/safety/test_kinematic_filter.py` | 4 项 | 4 passed |
| `tests/unit/utils/test_transforms.py` | 3 项 | 3 passed |
| `tests/integration/*.py` / `tests/closed_loop/*.py` | 4 项 | 4 skipped（依赖 CARLA/数据集） |

---

## 4. 关键接口与架构

```python
@dataclass
class BEVFusionConfig:
    num_cameras:int=2; image_size:(H,W)=(64,64); num_points:int=256
    cam_feat_channels=16; depth_bins=4; depth_min=1.; depth_max=20.
    bev_x_range=(-12.5,12.5); bev_y_range=(-12.5,12.5); bev_resolution=0.5; bev_channels=32
    lidar_in_channels=4; pillar_feat_channels=16
    imu_in_channels=6; imu_hidden=32; imu_steps=10
    camera_intrinsics/extrinsics: Optional[list[np.ndarray]] = None  # None→自动针孔

class BEVFusion(nn.Module):
    transform_points_to_world(points:(B,P,>=3), attitude:(B,3,3)) -> (B,P,>=3)   # einsum 旋转
    encode_lidar(points, attitude=None) -> (B, Cl, H, W)                          # 散射
    encode_images(images:(B,N,C,H,W)) -> (lifted:(B,Cf,P_total), Cf, P_total)     # LSS lift
    _splat_cameras(lifted, Cf) -> (B, Cf, H, W)                                   # 预计算索引散射
    fuse(bev_cam, bev_lidar, imu) -> (B, C, H, W)                                # conv 融合 + IMU 偏置
    forward(images, points, imu, imu_attitude=None) -> BEVFeature                 # bev=(B,C,H,W) float32
    export_onnx(path, sample_images, sample_points, sample_imu, sample_attitude=None)
```

---

## 5. 已确认的 Bug 与隐患（实证，按严重度排序）

### ✅ BUG-1（已修复 2026-08-27）BEV H/W 维度与 x/y 范围错配
**位置**：`src/perception/bev_fusion.py` — `__init__` 的 `self.bev_h / self.bev_w` 及 `encode_lidar`/`_build_frustum` 中的 `col/row/clamp`。

**现状**：
```python
self.bev_h = _grid_n(c.bev_x_range, c.bev_resolution)   # bev_h ← x_range 尺寸 ❌
self.bev_w = _grid_n(c.bev_y_range, c.bev_resolution)   # bev_w ← y_range 尺寸 ❌
# 但语义应为：bev_h ← y_range（行=rows=y），bev_w ← x_range（列=cols=x）
```
`col`（列）由 x 坐标算出却 `clamp(0, bev_w-1=y_size-1)`；`row`（行）由 y 坐标算出却 `clamp(0, bev_h-1=x_size-1)`。

**诊断复现**（已运行）：设 `bev_x_range=(-10,10)`、`bev_y_range=(-5,5)`、res=0.5 → `bev_h=40`、`bev_w=20`。点 `x=8` 本应落到 col=36（<40），却被 `clamp(0, bev_w-1=19)` 截断为 19；输出 `(C,40,20)` 行列语义整体反转。

**为何测试全绿**：`test_bev_fusion.py` 全程使用对称范围 `bev_x_range==bev_y_range=(-12.5,12.5)`，`bev_h==bev_w==50`，错配被掩盖。

**对照**：`src/utils/transforms.py` 的 `_grid_dims`/`world_to_bev_index` 是**正确**的（`W=round((x1-x0)/res)`、`H=round((y1-y0)/res)`，x→col∈[0,W)、y→row∈[0,H)）。两处几何约定不一致。

**影响**：任何非对称 BEV 范围（真实场景常见：前视 50m / 侧向 30m）会导致点云与相机 frustum 落到错误 cell、部分被错误截断，感知输出失真。

**建议修复**：
1. `self.bev_h = _grid_n(c.bev_y_range, res)`、`self.bev_w = _grid_n(c.bev_x_range, res)`；
2. 统一 `col = (x-x0)/res`、`row = (y-y0)/res`、`col` clamp 到 `[0,bev_w-1]`、`row` clamp 到 `[0,bev_h-1]`、`idx = row*bev_w + col`；
3. 在测试中**新增非对称范围用例**（如 x=(-10,10)、y=(-5,5)）作为回归护栏。

**✅ 已修复（2026-08-27）**：`src/perception/bev_fusion.py` `__init__` 对调范围来源 → `bev_h=_grid_n(bev_y_range)`、`bev_w=_grid_n(bev_x_range)`；下游 `col<bev_w`/`row<bev_h`/`idx=row*bev_w+col`/`view(...,bev_h,bev_w)` 自动正确，`_build_frustum`/`encode_lidar`/`_splat_cameras` 无需额外改动。新增 `test_asymmetric_bev_range_shape` + `test_asymmetric_bev_range_point_placement`（单点 (x=8,y=0) 须落 (row=10,col=36)）。输出统一为标准约定 `(B,C,H=y,W=x)`，与 `transforms.py`/`OccupancyGrid` 一致。

---

### 🟠 ISSUE-2（测试盲区）点云 pad/truncate 分支无覆盖
`_canonicalize_points` 有 3 条路径（n==P / n>P / n<P），但 `test_bev_fusion.py` 恒用 `num_points=256` 且输入恰好 256 点，**只覆盖了 n==P**。诊断确认 n<P 补零路径能跑、shape 正确，但无单测。另：n>P 走 `points[:, :P]` **确定性截断末尾点**，密集点云会信息损失（应改体素降采样或随机采样）。建议补 3 个边界用例。

### ✅ ISSUE-3（已修复 2026-08-27）pytest 与 anyio 插件不兼容
系统已装 `anyio`，其 pytest 插件 `import _pytest.scope`（pytest≥8 才存在），与项目 `pytest 6.2.5` 冲突，导致 **pytest 启动即崩溃**（任何不带 `-p no:anyio` 的调用，包括裸 `pytest`、IDE、CI）。
**✅ 已修复（2026-08-27）**：`pyproject.toml` 的 `addopts` 固化为 `"-ra -q -p no:anyio"`（方案 a）。现裸 `python -m pytest -v` 可直接启动，无需手动 flag。注：若后续升级 `pytest>=8` 可移除该 `-p no:anyio`。

### 🟠 ISSUE-4（功能未完成）`BEVFeature.occupancy` 恒为 None
SDD 要求感知输出"BEV + 3D Occupancy"，但 `forward` 仅填 `bev`。需在 `fuse` 后接 `Conv2d(bev_channels,1)→sigmoid` 占用预测头，或独立 LiDAR 占用分支。

### 🟠 ISSUE-5（功能未完成）CasADi NMPC 仅占位
`SafetyFilter.optimize_casadi` 即便 `use_casadi=True` 且装了 CasADi，仍 fallback 到 `project_to_feasible`（IPOPT OCP 未实现，`# TODO`）。当前安全层只有确定性投影，无非线性优化。

### 🟡 ISSUE-6（部署未实证）ONNX 导出无测试且环境未装
`export_onnx` 已写但无单测；`onnx/onnxruntime` 在 `requirements.txt`/`pyproject.toml` 中被注释，环境未装。`scatter_add_`→ONNX `ScatterElements` 在部分 TensorRT 版本可能不支持，**需 Phase 3 实证导出 + `trtexec` 校验**。

### 🟡 ISSUE-7（代码重复 + 约定不一致）几何逻辑双写
`transforms.point_cloud_to_bev_tensor`（numpy 几何，4 通道）与 `bev_fusion.encode_lidar`（可学习 MLP）各自实现"点→BEV cell"投影，且**约定相反**（前者对、后者错，呼应 BUG-1）。建议抽取共享几何常量 `_grid_dims`，让 `bev_fusion` 复用 `transforms` 的正确投影，消除重复与不一致。

### 🟡 ISSUE-8（健壮性）frustum reshape 顺序靠隐式约定
`_build_frustum` 的 `cam_index` 顺序 `(n,d,hf,wf)` 与 `encode_images` 的 `lifted.reshape(B,Cf,N*D*Hf*Wf)` 顺序匹配，但**无断言/注释保护**，未来改动易错。建议加 `assert cam_index.numel()==P_total` 与顺序注释。

---

## 6. 修复优先级

| 级 | 项 | 工作量 | 阻塞第三阶段? |
|---|---|---|---|
| ~~P0~~ | ✅ BUG-1 BEV 维度错配（+ 补非对称测试） | 已修 | — |
| ~~P1~~ | ✅ ISSUE-3 pytest 环境固化 | 已修 | — |
| P1 | ISSUE-2 pad/truncate 测试 + ISSUE-8 frustum 断言 | 0.5h | 否 |
| P2 | ISSUE-4 Occupancy 头 | 1h | 视第三阶段目标 |
| P2 | ISSUE-6 ONNX 实证导出 | 1h | 若 Phase 3 含部署 |
| P3 | ISSUE-5 CasADi NMPC、ISSUE-7 几何重构 | 各 2h+ | 否 |

---

## 7. 第三阶段衔接建议

1. **先修 BUG-1 + ISSUE-3**（P0/P1），再开新功能，避免在错误感知上叠加。
2. **补几何回归测试**：非对称 `bev_x/y_range`、`num_points` ≠ 输入点数、frustum 索引数一致性。
3. 若 Phase 3 = World Model/RSSM：`BEVFeature.bev` 即输入，需先稳定其 shape 与语义（BUG-1 修复后）。
4. 若 Phase 3 = Policy IL：需先打通 dataloader → `BEVFusion.forward` → 监督信号管线；`SensorPacket` 已就绪可作输入。
5. 若 Phase 3 = 部署：优先做 ISSUE-6（ONNX + `trtexec`）并扩 `export_onnx` 测试。

---

## 8. 附录：环境与运行命令

- 环境：Python 3.10.12、numpy 1.21、torch 2.x（CUDA 可用）、pytest 6.2.5；CasADi/onnx 未装（可选）。
- 运行单测：`cd /home/qqq/New_ORAD && python -m pytest tests/unit/test_bev_fusion.py -v`
- 运行全套：`cd /home/qqq/New_ORAD && python -m pytest -v`（`-p no:anyio` 已固化于 `pyproject.toml`，无需手输）
- 复现 BUG-1（已修复，仅供历史参考）：设 `bev_x_range=(-10,10), bev_y_range=(-5,5)`，修复前点 (8,0) 落 (10,19) 而非 (10,36)。
- 关键文件：`src/perception/bev_fusion.py`（279 行）、`tests/unit/test_bev_fusion.py`、`src/utils/{types,transforms}.py`、`src/safety/{bicycle_model,kinematic_filter}.py`。

---

## 9. 变更日志

### 2026-08-27 — 修复 BUG-1 + ISSUE-3

**BUG-1（BEV H/W 维度错配）**
- `src/perception/bev_fusion.py` `__init__`：对调范围来源
  ```python
  self.bev_h = _grid_n(c.bev_y_range, c.bev_resolution)  # H <- y (rows)
  self.bev_w = _grid_n(c.bev_x_range, c.bev_resolution)  # W <- x (cols)
  ```
- 影响范围：下游 `col(x)<bev_w`、`row(y)<bev_h`、`idx=row*bev_w+col`、`view(B,C,bev_h,bev_w)` 全部自动正确；`_build_frustum` / `encode_lidar` / `_splat_cameras` 无需额外改动。
- 输出 shape 语义统一为标准约定 `(B, C, H=y, W=x)`，与 `utils/transforms.py`、`OccupancyGrid` 一致。
- 新增回归测试（`tests/unit/test_bev_fusion.py`）：
  - `test_asymmetric_bev_range_shape`：非对称范围下 shape = (1, 32, y_size=20, x_size=40)。
  - `test_asymmetric_bev_range_point_placement`：单点 (x=8, y=0) 须精确落在 (row=10, col=36)。

**ISSUE-3（pytest 与 anyio 不兼容）**
- `pyproject.toml`：`addopts = "-ra -q -p no:anyio"`，固化禁用 anyio 插件。
- 验证：裸 `python -m pytest -v` 现可直接启动并跑通（修复前崩溃）。

**回归验证**
- `python -m pytest -v` → **18 passed, 4 skipped**（BEV 6 + bicycle 5 + kinematic_filter 4 + transforms 3；skip 为 integration/closed_loop 骨架）。
- GPU 测试 `test_batch_greater_than_one_gpu_alignment` 实跑通过（CUDA 可用）。

**剩余未修复项**：ISSUE-2（pad/truncate 测试覆盖）、ISSUE-4（Occupancy 头）、ISSUE-5（CasADi NMPC）、ISSUE-6（ONNX 实证）、ISSUE-7（几何重构）、ISSUE-8（frustum 断言）——见 §5/§6。

