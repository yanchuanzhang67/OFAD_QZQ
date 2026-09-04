# Stage 1B Task 1：固定 CARLA 实验基线

> 状态日期：2026-08-31  
> 当前成熟度：**单元验证**  
> 环境验收：**待真实 CARLA 三次运行**

## 2026-09-03 Canonical superseding note

2026-08-31 的 `0.9.15 + Tesla + seed 20260831` 初始 Canonical 已被真实采集数据
证据取代。当前唯一 Canonical 为：

| 项目 | 当前固定值 |
|---|---|
| CARLA client/server | `0.9.16` |
| Map | `Town10HD_Opt` |
| Vehicle blueprint | `vehicle.lincoln.mkz_2020` |
| Random seed / repetitions | `42 / 3` |
| Control/sensor period | `0.1 s` |
| Camera | front/rear/top `192×192`，FOV `90/90/100°`，top pitch `-15°` |
| LiDAR | 32 channels、50 m、320000 point/s、10 Hz、FOV `[-25°,15°]` |
| IMU | `10×6 @ 10 Hz` |
| Calibration | `carla-default-v1`，另保存内容 SHA-256 |

配置已扩展为具名传感器结构，并把 Camera optical frame 经 CARLA frame 转换到
New_ORAD `x-forward/y-left/z-up`。旧标量 Camera/LiDAR schema 会给出迁移错误，不会
静默兼容。2026-09-03 TDD 结果：

```text
python -m pytest -q tests/unit/configuration/test_system_config.py \
  tests/unit/sim/test_carla_baseline.py
28 passed

python -m pytest -q tests/integration/test_cpu_pipeline_contract.py
1 passed
```

当前字节级 config SHA-256 为
`3c61c5e9da1773200c58346b7770f88d58a73f378d10ec7b47d2c7c1fa3a761c`，逻辑
canonical SHA-256 为
`9786122b446adc2ef5580bea772f190b5e7195cd3106b43c2f83575af6f001d9`。工作树仍为
dirty，且真实三次 CARLA 运行尚未执行，因此成熟度仍是 **单元验证**。

## 1. 2026-08-31 历史固定基线（已被上述配置取代）

Task 1 将以下运行值收归 `configs/system.yaml` 的严格 `carla_baseline` 配置：

| 项目 | 固定值 |
|---|---|
| CARLA client/server | `0.9.15` |
| Map | `Town10HD_Opt` |
| Vehicle blueprint | `vehicle.tesla.model3` |
| Random seed | `20260831` |
| Repetitions | `3` |
| Control period | `0.1 s`，来自 `control.dt` |
| Camera | `3 × 192 × 192`、FOV `90°` |
| LiDAR | 32 channels、50 m、32000 point/s、20 Hz、FOV `[-25°, 15°]` |
| Calibration | `carla-default-v1` |

未知、缺失、空字符串、非正数和不合法 FOV 会由严格 loader 拒绝。正式 CARLA
入口不再使用硬编码的地图、车辆或传感器属性。

## 2. 证据产物

运行 `scripts/verify_carla_baseline.py` 会创建新的、不可覆盖的目录：

```text
runs/carla_baseline/<run-id>/
├── manifest.json
├── repetition-01.json
├── repetition-02.json
├── repetition-03.json
└── summary.json
```

`manifest.json` 记录 Git commit、dirty 状态、配置绝对路径及 SHA-256、CARLA
client/server version、地图、车辆、随机种子、控制周期、标定版本和完整传感器
属性。三个 repetition 文件保存逐 tick 的 Camera/LiDAR/IMU frame 与 timestamp。

比较时只消除每次运行的绝对 frame/timestamp 起点，保留帧间增量、各模态 frame
对齐、采样间隔和跨模态 timestamp 关系。配置或时序任一漂移都会令
`reproducible=false`，脚本返回非零状态。

## 3. TDD 过程记录

### Red 1：canonical 配置

```bash
python -m pytest -q tests/unit/configuration/test_system_config.py
```

结果：`7 failed / 8 passed`。失败原因是 `SystemStackConfig` 不存在
`carla_baseline`，YAML 不存在对应 section。

### Green 1：canonical 配置

加入 `CarlaBaselineConfig`、严格 section/key 校验和 canonical YAML 后：

```text
15 passed
```

### Red 2：provenance 与三次一致性

```bash
python -m pytest -q tests/unit/sim/test_carla_baseline.py
```

结果：`6 failed`。环境校验、Git/config provenance、三次时序对比和不可覆盖写入
均因尚未实现而失败。

### Green 2：CPU 软件契约

```bash
python -m pytest -q tests/unit/sim/test_carla_baseline.py \
  tests/unit/configuration/test_system_config.py
python -m flake8 src/sim/carla_baseline.py scripts/verify_carla_baseline.py \
  src/configuration/system.py tests/unit/sim/test_carla_baseline.py \
  tests/unit/configuration/test_system_config.py --max-line-length=99
```

结果：`29 passed`；focused flake8 通过。

记录时工作树来源为 commit
`99fc0ea3991cc5e8fc3367529a939b7dee7e54f3`，canonical config SHA-256 为
`431b9898893141227ad6e7e1589a679a468ec99af51345539e42d8db6461198a`。
工作树仍为 dirty，因此这些值只能描述当前开发过程；真实运行时 manifest 会重新
计算并记录当时的 commit、dirty 状态和 config hash。

## 4. 真实 CARLA 验收命令

启动当前 Canonical CARLA 0.9.16，并确保当前地图为 `Town10HD_Opt`，然后执行：

```bash
python scripts/verify_carla_baseline.py \
  --config configs/system.yaml \
  --run-id stage1b-task1-<UTC时间> \
  --steps 100
```

也可以显式运行 closed-loop 验收测试：

```bash
ORAD_CARLA_BASELINE_RUN_ID=stage1b-task1-<UTC时间> \
python -m pytest -q \
  tests/closed_loop/test_carla_closed_loop.py::test_fixed_carla_baseline_is_reproducible_across_three_runs
```

### Exit Gate

- `summary.json.repetitions == 3`；
- `config_consistent == true`；
- `sensor_timing_consistent == true`；
- `reproducible == true`；
- manifest 中 commit/config/CARLA/map/calibration 字段完整；
- 运行目录未被覆盖。

在上述真实环境命令尚未执行前，Task 1 不能标为“仿真闭环验证”，Stage 1B 也不能
关闭。
