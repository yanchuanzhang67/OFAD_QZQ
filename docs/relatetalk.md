可以。按照你现在更新后的 New_ORAD 方案，我建议三阶段推进时不要平均铺开所有模块，而是遵循一个原则：

$$
\boxed{
先让系统“正确地跑起来”
\rightarrow
再让新方法“有效地学起来”
\rightarrow
最后让系统“稳定地闭环起来”
}
$$

当前项目仍处于核心算法原型和单元测试阶段，README 里明确指出：训练数据闭环、CARLA/Gazebo 系统验收和 TensorRT 端侧实现都还没有完成。 因此，现阶段最重要的不是立刻重构一个复杂的 Terrain-aware RSSM + Dreamer 全系统，而是先清掉会破坏后续实验可信度的基础问题。

---

# 第一阶段：紧急修复与即时执行

## Immediate Action

这一阶段目标只有一个：

$$
\boxed{\text{建立一个“数据正确、坐标正确、异常可控”的最小闭环}}
$$

在此之前，不建议进行正式的 RSSM 或 Dreamer 大规模训练。

---

## 1. 当前最优先问题一：统一数据契约与坐标系

这是第一优先级。

当前 README 已经明确存在两个 P0：

* CARLA IMU 输出 `(T,3)`，而模型要求 `(T,6)`；
* LiDAR occupancy 与 safety trajectory 的坐标系没有统一。

而项目正式定义的接口是：

```text
images:     (B, N_camera, 3, H, W)
points:     (B, N_point, 4)
imu:        (B, T, 6)
BEV:        (B, C, H=y, W=x)
trajectory: (B, N, 4)
```



如果这一层不统一，那么后面 Terrain Affordance、RSSM 和 Safety 的实验结果都不可信。

### 立即执行

先定义一个系统级唯一 frame：

```text
EGO FRAME

x: forward
y: left
z: up

origin:
rear axle center
或 vehicle center
```

一旦选定，整个系统不能再各模块自行解释。

建议统一：

```text
Camera → ego
LiDAR  → ego
Trajectory → ego
Affordance → ego BEV
Safety → ego
```

只有最后需要地图/global trajectory 时才转：

```text
ego ↔ world
```

### 建议新增公共数据定义

例如：

```text
src/utils/schema.py
src/utils/frames.py
```

其中统一：

```python
@dataclass
class EgoState:
    position: np.ndarray
    yaw: float
    velocity: float
    steering: float
    timestamp: float

@dataclass
class IMUState:
    accel: np.ndarray      # ax ay az
    gyro: np.ndarray       # gx gy gz
    timestamp: float

@dataclass
class Trajectory:
    points: np.ndarray     # N x 4
    frame: str             # "ego"
```

任何模块入口都先检查：

```python
assert frame == "ego"
assert trajectory.shape[-1] == 4
assert imu.shape[-1] == 6
```

### Checklist

* [ ] 明确 ego 坐标轴方向；
* [ ] 明确 BEV 中 H/W 分别对应哪个物理轴；
* [ ] 所有 trajectory 明确 frame；
* [ ] IMU 固定为 `[ax, ay, az, gx, gy, gz]`；
* [ ] 明确角度单位全部使用 rad；
* [ ] 明确距离全部使用 m；
* [ ] 明确速度全部使用 m/s；
* [ ] 所有 timestamp 使用同一时间基准；
* [ ] 为 `ego → world → ego` 写 round-trip 单元测试；
* [ ] 为 BEV cell ↔ metric coordinate 写几何单元测试。

---

# 2. 当前最优先问题二：传感器同步

README 已将 Camera、LiDAR 和 IMU 的 frame 同步列为 P0。

这个问题比网络精度更优先。

因为如果：

```text
Camera = t
LiDAR  = t - 100 ms
IMU    = t + 50 ms
```

那么网络看到的实际上不是同一个物理世界状态。

在普通平坦道路上可能暂时没明显问题，但越野车辆存在较大的：

* pitch；
* roll；
* yaw；
* vertical acceleration；

误差会非常明显。

## 建议的数据单位

不要让 DataLoader 返回三个独立 sensor tensor。

建议返回：

```python
Observation(
    timestamp=t,
    images=...,
    lidar=...,
    imu_history=...,
    ego_state=...
)
```

即整个系统基本采样单元是：

$$
o_t
=
\{
I_t,
P_t,
U_{t-k:t},
x_t^{ego}
\}
$$

这也正好符合未来 RSSM 的 observation 定义。

### 第一版同步原则

在 CARLA 中优先使用：

```text
same CARLA frame_id
```

即：

```text
camera.frame == lidar.frame
```

IMU 如果频率更高，则截取：

$$
[t-\Delta t,t]
$$

内最近 \(T\) 个 IMU measurements。

最终：

```text
Observation_t
  camera frame = 1050
  lidar frame  = 1050
  imu history  = frames around 1050
```

### Checklist

* [ ] Camera / LiDAR 都保留 simulator frame id；
* [ ] 不允许使用“最新一帧”这种无约束逻辑；
* [ ] 设置同步 timeout；
* [ ] 缺一个传感器则本帧不进入模型；
* [ ] IMU history 长度固定；
* [ ] 保存时间差统计；
* [ ] 验证 camera-lidar skew；
* [ ] 验证 IMU 与控制周期一致；
* [ ] 在日志中记录每帧 sensor timestamp。

---

# 3. 当前最优先问题三：Fail-safe 与最小集成闭环

README 目前指出：

> 空轨迹、NaN 和 timeout 都应该主动进入安全制动。

并且要求增加一个无需 CARLA 的：

```text
perception
→ policy
→ safety
→ control
```

CPU integration test。

这个应该立刻实现。

### Fail-safe 输入条件

至少检查：

```python
trajectory is None
len(trajectory) == 0
np.isnan(trajectory).any()
np.isinf(trajectory).any()
sensor_timeout
model_timeout
checkpoint_invalid
```

任一成立：

```text
target_speed = 0
brake = 1
steering = safe steering / 0
```

不要继续传给 Pure Pursuit。

---

## 建议建立一个 SafetySupervisor

```text
Policy
  ↓
Trajectory Validator
  ↓
Kinematic Safety
  ↓
Collision Check
  ↓
SafetySupervisor
  ↓
Controller
```

SafetySupervisor 决定：

```text
NORMAL
DEGRADED
EMERGENCY_STOP
```

### Checklist

* [ ] 空轨迹触发 braking；
* [ ] NaN 触发 braking；
* [ ] Inf 触发 braking；
* [ ] sensor timeout 触发 braking；
* [ ] model timeout 触发 braking；
* [ ] trajectory frame 不匹配直接拒绝；
* [ ] trajectory 点数不足拒绝；
* [ ] speed < 0 或极端值拒绝；
* [ ] fail-safe branch coverage = 100%；
* [ ] CPU integration test 不允许 skip。

README 当前建议的工程质量门槛正是 safety branch coverage ≥95%、P0 fail-safe 100%、CPU integration tests 不得 skip。

---

# 第一阶段完成条件

不要用“代码写完了”作为完成条件。

必须同时满足：

```text
Unit Tests
        +
CPU Integration
        +
Recorded Replay
```

第一阶段 Gate：

$$
\boxed{
\text{1000 个 synthetic/replay samples 全链路无 shape/frame error}
}
$$

并且：

```text
0 NaN leakage
0 invalid trajectory entering control
100% fail-safe activation
```

---

# 第二阶段：管线优化与精准微调

## Pipeline Optimization

第一阶段修完后，才开始真正进入你的新版研究方案：

$$
BEV
\rightarrow
Terrain\ Affordance
\rightarrow
RSSM
\rightarrow
BC+Dreamer
$$

这里要分宏观架构和微观参数两部分。

---

# 一、宏观层面：重新整理 Pipeline

我建议代码架构同步修改成：

```text
Sensor Layer
        ↓
Generic BEV Perception
        ↓
Terrain Affordance
        ↓
Ego Dynamics
        ↓
World Model
        ↓
Hybrid Policy
        ↓
Safety
        ↓
Controller
```

而不是现在全部挤在 perception / policy。

目前 README 的实际情况是：`world_model` 包本身还是空的，RSSM / WorldModel 主要位于 `policy` 内。

这个最好尽快重构。

---

## 1. 推荐代码目录

```text
src/
├── perception/
│   ├── camera_encoder.py
│   ├── camera_bev.py
│   ├── lidar_encoder.py
│   └── bev_fusion.py
│
├── affordance/
│   ├── encoder.py
│   ├── traversability_head.py
│   ├── roughness_head.py
│   ├── slope_head.py
│   └── risk_head.py
│
├── dynamics/
│   └── ego_encoder.py
│
├── world_model/
│   ├── rssm.py
│   ├── observation_encoder.py
│   ├── reward_head.py
│   ├── risk_head.py
│   └── imagination.py
│
├── policy/
│   ├── actor.py
│   ├── critic.py
│   ├── trajectory_decoder.py
│   ├── bc_trainer.py
│   └── dreamer_trainer.py
```

这里一个非常重要的原则是：

$$
\boxed{\text{RSSM 不应该再属于 Policy}}
$$

Policy 应该是世界模型的消费者。

---

# 2. 模块间只通过明确接口通信

不要：

```text
Policy import perception internal tensors
```

而应：

```text
PerceptionOutput
      ↓
AffordanceOutput
      ↓
WorldModelState
      ↓
PolicyOutput
```

例如：

```python
@dataclass
class PerceptionOutput:
    bev: Tensor

@dataclass
class AffordanceOutput:
    traversability: Tensor
    roughness: Tensor
    slope: Tensor
    risk: Tensor
    feature: Tensor

@dataclass
class WorldModelState:
    deterministic: Tensor
    stochastic: Tensor

@dataclass
class PolicyOutput:
    trajectory: Tensor
```

这样以后替换 LSS、PointPillars 或 Actor 时，不会牵连整个系统。

---

# 3. 推荐数据流

建议正式定义：

$$
o_t=
(I_t,P_t)
$$

Generic BEV：

$$
F_t^{BEV}=f_{BEV}(I_t,P_t)
$$

Terrain Affordance：

$$
A_t=f_{aff}(F_t^{BEV})
$$

Ego dynamics：

$$
e_t^{ego}
=
f_{dyn}(IMU_{t-k:t},v_t,\delta_t,\dot\psi_t)
$$

World Model：

$$
s_t
=
f_{RSSM}
(
s_{t-1},
a_{t-1},
A_t,
e_t^{ego}
)
$$

Policy：

$$
\tau_t
=
\pi(s_t,g_t)
$$

然后：

$$
\tau_t
\rightarrow
Safety
\rightarrow
Control
$$

---

# 4. BEV 层不要继续重度创新

你上传的 BEVFusion 已经表明，共享 BEV 可以有效同时保持 Camera 语义和 LiDAR 几何，且可以作为多任务、多传感器的通用感知前端。

因此建议：

```text
BEV = strong baseline
```

而不是：

```text
BEV = research battlefield
```

可以固定：

```text
Camera:
ResNet-34/FPN or Swin-T
        ↓
LSS-style BEV

LiDAR:
PointPillars-lite
        ↓
LiDAR BEV
```

融合先用简单：

$$
F^{BEV}
=
Conv([F_{cam},F_{lidar}])
$$

不要一开始做复杂 Cross-Attention。

---

# 二、微观层面：关键参数建议

以下属于“实施建议”，不是 README 已存在的固定参数。

---

# 1. BEV Resolution

第一版不要太大。

推荐：

```text
range:
x ∈ [-10, 40] m
y ∈ [-25, 25] m

resolution:
0.25–0.5 m / cell
```

第一版推荐：

$$
0.5m
$$

得到大约：

```text
100 × 100 BEV
```

如果上来就：

```text
0.1m
```

显存和计算量会非常高。

---

# 2. BEV Channel

建议：

```text
C_bev = 64 or 128
```

第一版：

$$
C=64
$$

等 Affordance + RSSM 全部跑通，再升到 128。

---

# 3. Terrain Affordance

第一版不要五个 head 同时做。

建议最小版本只有：

```text
Traversability
+
Roughness
```

也就是：

$$
A_t=
[T_t,R_t]
$$

因为：

* slope 可以从 LiDAR geometry 自动计算；
* obstacle 可以从 occupancy 派生；
* uncertainty 可以第二轮再做。

这样标签负担小很多。

### Traversability

推荐：

```text
binary:
traversable
non-traversable
```

先不要四分类。

Loss：

$$
L_{trav}
=
BCE
$$

如果类别严重不平衡，再用 Focal Loss：

$$
\gamma\approx2
$$

---

# 4. Roughness

可以利用 IMU 产生 weak supervision。

例如定义局部窗口：

$$
r_t
=
RMS(a_z-\bar a_z)
$$

或者：

$$
r_t
=
\sqrt{
\frac1T
\sum
(a_z-\bar a_z)^2
}
$$

然后把车辆行驶 footprint 投影到 BEV，作为 traversed terrain 的 roughness label。

第一版不要尝试一次性标注整张 BEV roughness。

可以只监督：

```text
vehicle traversed cells
```

并用 mask：

$$
L_{rough}
=
M
\odot
|\hat R-R|
$$

---

# 5. IMU Encoder

第一版参数建议：

```text
IMU frequency:
20–50 Hz

history:
0.5–1.0 s

T:
10–50
```

网络：

```text
6
↓
MLP 32
↓
GRU 64
↓
ego embedding 64
```

不要一开始做 Transformer。

---

# 6. RSSM 参数

第一版建议保持小模型：

```text
deterministic state h:
256

stochastic state z:
32

observation embedding:
256

action embedding:
64
```

如果 latent imagination OOM：

```text
h: 128
z: 16
```

第一目标不是追求最终性能，而是验证：

$$
Prior
\leftrightarrow
Posterior
$$

是否真的在学习。

---

# 7. RSSM Loss

建议起始：

$$
L=
L_{aff}
+
L_{reward}
+
L_{risk}
+
\beta L_{KL}
$$

其中可先：

```text
λ_aff    = 1.0
λ_reward = 1.0
λ_risk   = 1.0
β_KL     = 0.1
```

KL 太大容易：

```text
posterior collapse
```

KL 太小又会导致：

```text
prior rollout completely wrong
```

训练时必须同时监控：

```text
posterior std
prior std
KL divergence
future prediction error
```

---

# 8. Imagination Horizon

不要一开始就 15–50 步。

先：

$$
H=5
$$

验证稳定。

然后：

```text
5 → 10 → 15
```

因为 world model 的误差大致会随 horizon 累积。

---

# 9. BC Trajectory Decoder

第一版用 MLP：

```text
latent 288
+
goal embedding
↓
MLP 256
↓
MLP 256
↓
N × 4
```

例如：

```text
N = 10
dt = 0.2 s
```

即预测未来：

$$
2s
$$

轨迹。

等 baseline 建立后，再换 Transformer decoder。

---

# 10. BC Loss

建议初始：

$$
L_{BC}
=
1.0L_{xy}
+
0.2L_\theta
+
0.2L_v
+
0.05L_{smooth}
$$

不要一开始 smoothness 太大，否则模型会倾向输出过直的轨迹。

---

# 11. Dreamer 不要从第 1 天加入

训练顺序严格控制：

```text
BEV
↓
Affordance
↓
RSSM
↓
BC
↓
Dreamer
```

如果 BC open-loop ADE 都没有达到一个稳定水平，就不要开 RL。

README 当前的系统目标本身给出了 open-loop：

$$
ADE<0.3m
$$

以及 closed-loop：

```text
success > 90%
collision = 0
```

这些可以继续作为远期 gate。

---

# 12. Dreamer Reward

第一版只保留 4 项：

$$
R=
w_pR_{progress}
+
w_tR_{trav}
-
w_cR_{collision}
-
w_aR_{lat}
$$

等策略稳定再加：

```text
roughness
uncertainty
jerk
rollover
```

否则 reward debugging 会非常困难。

---

# 第三阶段：风险预防与进度推演

## Risk Control & Timeline

这里要同时考虑算法风险和工程风险。

---

# 一、主要副作用与隐患

## 1. BEV Resolution 提高 → 显存快速增长

如果从：

```text
100 × 100
```

提高到：

```text
200 × 200
```

二维特征图面积直接：

$$
4\times
$$

如果还有：

```text
Camera BEV
LiDAR BEV
Affordance
RSSM embedding
```

峰值显存可能显著增加。

### 预防

先固定：

```text
0.5 m resolution
C=64
```

监控：

```text
allocated memory
reserved memory
peak memory
```

如果 OOM：

```text
降低 C
降低 BEV resolution
降低 batch
gradient accumulation
AMP
```

---

# 2. 多任务 Affordance → 负迁移

例如：

```text
Traversability
Roughness
Slope
Risk
```

共享 backbone 后，可能出现：

$$
\nabla L_{trav}
$$

和：

$$
\nabla L_{rough}
$$

方向冲突。

表现为：

```text
traversability ↑
roughness ↓
```

或者相反。

### 预防

第一版只做两个 head。

训练时单独记录：

```text
L_trav
L_rough
```

不要只看 total loss。

如果冲突明显：

```text
separate head depth
loss reweight
freeze BEV backbone
```

再考虑复杂的 multi-task balancing。

---

# 3. IMU 弱标签 → 模型学到车辆而不是地形

这是非常重要的风险。

假设：

$$
a_z
$$

很大，可能来自：

* 地形很粗糙；
* 车速过高；
* suspension 参数不同；
* 车辆载荷不同。

所以：

$$
Roughness
\neq
f(a_z)
$$

严格来说：

$$
Vehicle\ Response
=
f(
Terrain,
Speed,
Suspension,
Vehicle
)
$$

### 预防

生成 roughness label 时至少 conditioned on：

```text
speed
```

例如：

$$
r
=
\frac{RMS(a_z)}{v+\epsilon}
$$

这只是简单 proxy。

更好的方式是模型输入：

```text
terrain visual feature
+
speed
+
ego dynamics
```

共同学习 consequence。

---

# 4. RSSM Posterior Collapse

典型表现：

```text
KL → 0
z_t almost constant
```

模型只用 deterministic GRU，不再使用 stochastic latent。

### 预防

可以逐渐采用：

```text
KL warm-up
free nats
KL balancing
```

第一版先监控：

$$
KL_t
$$

如果长期接近 0，就暂停训练检查。

---

# 5. World Model 长 horizon 漂移

可能出现：

```text
t+1 correct
t+3 okay
t+10 nonsense
```

### 预防

所有 world-model 实验都画：

$$
error(k)
$$

即：

```text
1-step
3-step
5-step
10-step
```

如果 5-step 之后明显失效：

不要用 H=15 的 Dreamer imagination。

Imagination horizon 必须：

$$
H_{RL}
\leq
H_{reliable}
$$

---

# 6. Dreamer reward hacking

例如策略可能发现：

```text
慢速不动
```

可以避免：

```text
collision
roughness
lateral acceleration
```

从而得到看似不错的 reward。

### 预防

必须同时记录 reward decomposition：

```text
progress
collision
trav
rough
lat
```

而不能只看 total reward。

并设置：

```text
minimum progress
timeout penalty
stuck penalty
```

---

# 7. BC → RL 后性能退化

Dreamer fine-tuning 可能破坏 BC 已有能力。

表现：

```text
BC ADE = 0.25
Dreamer after training ADE = 0.8
```

但 RL reward 变高。

### 预防

保留行为先验：

$$
L_{actor}
=
L_{RL}
+
\lambda_{BC}L_{BC}
$$

RL 初期：

```text
λ_BC high
```

随后逐渐降低。

也可以采用：

```text
actor initialization from BC
+
small RL learning rate
```

---

# 8. Safety Filter 过强 → 掩盖 Policy 问题

如果每次错误轨迹都被 safety 修复：

```text
collision = 0
```

并不意味着 policy 好。

### 预防

同时统计：

```text
raw policy collision risk
after-safety collision risk
safety intervention rate
```

如果：

```text
safety intervention = 80%
```

说明 policy 本身并不合格。

---

# 二、统一降级策略

建议把系统设计成 4 级模式：

```text
LEVEL 0
Full:
BEV + Affordance + RSSM + Dreamer

LEVEL 1
No Dreamer:
BEV + Affordance + RSSM + BC

LEVEL 2
No World Model:
BEV + Affordance + BC

LEVEL 3
Emergency:
stop / fail-safe
```

这样任何新模块出问题，都不必让整个系统停摆。

---

# 三、验证机制

每次加入新模块，只允许增加一个变量。

例如：

### Baseline A

```text
BEV → BC
```

### Baseline B

```text
BEV → Affordance → BC
```

### Baseline C

```text
BEV → Affordance → RSSM → BC
```

### Final

```text
BEV → Affordance → RSSM → BC+Dreamer
```

不要：

```text
BEV + new fusion + new affordance + RSSM + Dreamer
```

一次全部加入。

否则出问题无法定位。

---

# 四、推荐推进节奏

结合当前 README 的成熟度：Phase 3（IL+RL+World Model）约 45%，Phase 5（闭环与部署）约 30%，说明算法和闭环两端都还需要相当开发量。

我会建议按约 **12–16 周** 的研究周期规划。

| 时间         | 主要目标                   | 必须交付                                        |
| ---------- | ---------------------- | ------------------------------------------- |
| Week 1–2   | P0 修复                  | frame/schema/sync/fail-safe 全部测试通过          |
| Week 3     | CPU 最小集成               | perception→policy→safety→control 无 CARLA 运行 |
| Week 4–5   | Generic BEV baseline   | Camera+LiDAR BEV 离线稳定输出                     |
| Week 6–7   | Terrain Affordance v1  | Traversability + Roughness，完成离线指标           |
| Week 8–9   | RSSM v1                | prior/posterior、1–5 step prediction 可视化     |
| Week 10    | BC baseline            | ADE/FDE 稳定，形成 open-loop baseline            |
| Week 11–12 | Affordance + RSSM + BC | 完整模型但不加入 RL                                 |
| Week 13–14 | Dreamer fine-tuning    | imagined rollout + actor/critic             |
| Week 15    | CARLA/Gazebo 闭环        | success/collision/intervention 指标           |
| Week 16    | 消融与论文结果                | Baseline A/B/C/Final 比较                     |

---

# 五、各里程碑必须有 Gate

### M0 — Interface Gate

必须达到：

```text
shape correct
frame correct
timestamp correct
fail-safe correct
```

否则禁止训练。

---

### M1 — Perception Gate

要求：

```text
BEV 可视化正确
Camera/LiDAR 对齐正确
```

再进入 Affordance。

---

### M2 — Affordance Gate

至少：

```text
Traversability F1 / IoU
明显高于 trivial baseline
```

并对 roughness 做定量验证。

---

### M3 — World Model Gate

必须证明：

$$
Prediction_{t+1:t+H}
$$

比：

```text
copy-current-state baseline
```

更好。

否则 RSSM 没有实际贡献。

---

### M4 — BC Gate

至少达到可接受的：

```text
ADE
FDE
heading error
```

再进入 Dreamer。

---

### M5 — Dreamer Gate

加入 RL 后至少不能破坏：

```text
collision
success
ADE
```

其中关键 closed-loop 指标必须优于 BC-only。

---

### M6 — Closed-loop Gate

README 当前给出的目标是：

```text
Success Rate > 90%
Collision Rate = 0
姿态过载报警 = 0
```

以及 open-loop ADE < 0.3 m。

这些目标可以作为最终理想门槛，但不建议要求第一次闭环就达到。

---

# 六、你现在实际应该从哪里开始

如果明天开始写代码，我建议顺序固定为：

```text
Day 1
统一 frame/schema

        ↓

Day 2
IMU 6D + timestamp

        ↓

Day 3
fail-safe

        ↓

Day 4–5
CPU integration test

        ↓

Week 2
CARLA sensor synchronization

        ↓

然后才开始
Terrain Affordance
```

而不是现在立刻：

```text
开始实现 Dreamer
```

因为你的当前最大风险不是 Dreamer 算法写不出来，而是：

$$
\boxed{
\text{世界模型正在学习一个时间错位、坐标错位甚至数据格式不一致的世界}
}
$$

这种情况下，模型越复杂，越难排查。

---

# 最终三阶段压缩版

整个实施策略可以压缩成：

$$
\boxed{
\textbf{Stage 1 — Correctness}
}
$$

解决：

```text
frame
schema
sync
fail-safe
integration
```

然后：

$$
\boxed{
\textbf{Stage 2 — Learning}
}
$$

逐级构建：

```text
BEV
→ Affordance
→ RSSM
→ BC
→ Dreamer
```

最后：

$$
\boxed{
\textbf{Stage 3 — Robustness}
}
$$

围绕：

```text
ablation
world-model drift
reward hacking
safety intervention
OOM
closed-loop regression
```

建立风险控制。

如果严格按这个节奏推进，你的项目会从现在的“很多模块都已有原型，但系统契约和闭环还不稳”，逐步转成一个**每增加一个研究创新，都有明确 baseline、Gate、消融和闭环证据支撑的研究系统**。这比一开始直接追求完整 Terrain-aware RSSM + Dreamer 更稳，也更利于最终论文结果。
