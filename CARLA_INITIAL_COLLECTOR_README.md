# CARLA 初始数据采集器

这套脚本用于先跑通 New_ORAD 的原始数据链路。它在当前 CARLA 地图中自动生成一辆
ego 车辆，由 Traffic Manager 驾驶并同步采集 200 帧。该数据用于验证存储、replay、
Observation 和 RSSM 输入管线，不作为最终越野 BC 专家数据。

## 1. 放入 New_ORAD

将压缩包内文件保持目录结构复制到项目根目录：

```text
New_ORAD/
├── scripts/
│   ├── collect_carla_initial.py
│   └── verify_carla_initial_dataset.py
└── tests/sim/
    ├── test_collect_carla_initial.py
    └── test_verify_carla_initial_dataset.py
```

## 2. 启动采集

保持 Windows 中的 CARLA 城市窗口运行。在 WSL2 中执行：

```bash
cd ~/New_ORAD
source ~/venvs/carla0916/bin/activate
python -m pip install -e ".[dev,perception]"

export CARLA_HOST=$(ip route show | awk '/default/ {print $3; exit}')

python scripts/collect_carla_initial.py \
  --config configs/system.yaml \
  --host "$CARLA_HOST" \
  --port 2000 \
  --frames 200 \
  --output datasets/carla_initial
```

WSL mirrored networking 下也可以把 `--host` 改成 `127.0.0.1`。

启动后，CARLA 窗口会出现 ego 车辆，spectator 会自动跟随。WSL 终端持续显示：

```text
采集  120/200 | frame=845 | speed=31.2 km/h | lidar=32000
```

不要同时运行另一个会调用 `world.tick()` 的同步 Client。

车辆、地图、random seed、fixed delta 和全部 Camera/LiDAR/IMU blueprint、位姿及
attribute 只来自 `configs/system.yaml`。CLI 不提供这些物理参数的覆盖入口。当前
Canonical 是 CARLA `0.9.16`、`Town10HD_Opt`、Lincoln MKZ 2020、seed `42`；连接
环境或 blueprint 不一致时会在正式帧写入前失败，不会回退到任意车辆。

## 3. 验证数据

采集结束后执行：

```bash
python scripts/verify_carla_initial_dataset.py datasets/carla_initial
```

正确结果类似：

```text
通过：episode_20260901T050000Z | frames=200 | range=726..925 | max_skew=0.000000s
```

只有包含 `_SUCCESS` 的 episode 才通过验证。

## 4. 输出结构

```text
datasets/carla_initial/
├── .incomplete/
└── episodes/
    └── episode_YYYYMMDDTHHMMSSZ/
        ├── episode.json
        ├── calibration.json
        ├── frames.jsonl
        ├── camera_front/*.png
        ├── camera_rear/*.png
        ├── camera_top/*.png
        ├── lidar_raw/*.npy
        ├── lidar_256/*.npy
        └── _SUCCESS
```

`frames.jsonl` 保存每帧的 IMU `10×6` history、ego state、Traffic Manager action、
collision、goal distance、frame ID 和 timestamp。`expert_label=false` 明确表示这些动作
不能直接作为最终人工 BC 专家标签。

新采集的 `episode.json` 还保存实际 vehicle blueprint、原始/逻辑 config SHA-256、
calibration version/SHA-256。2026-09-02 的历史 200 帧 episode 没有这些字段，不会被
反向修改；后续 replay 必须把它标记为 legacy provenance-gap。

## 5. 常见错误

- `connection timed out`：确认 CARLA 窗口仍在运行、`CARLA_HOST` 正确，并检查
  Windows 防火墙的 `2000/TCP` 和 `2001/TCP`。
- `received future frame`：有另一个同步 Client 调用了 `world.tick()`；关闭它后重试。
- `all vehicle spawn points are occupied`：重启 CARLA Server 或清理当前地图中的车辆。
- `canonical vehicle blueprint unavailable`：当前 CARLA 安装不包含配置指定的
  `vehicle.lincoln.mkz_2020`；不会自动更换车辆。
- `CARLA Python API is missing`：重新激活 `~/venvs/carla0916`，确认
  `python -m pip show carla` 显示版本 `0.9.16`。
- 失败目录留在 `.incomplete/`：这是诊断数据，验证器会自动忽略，不要当作训练集。
