# New_ORAD 文档导航

本文档是 `docs/` 的入口。状态结论优先阅读“当前基线”；Phase 文档与带日期
的审计报告保留过程证据，不应单独作为当前完成度声明。

## 当前基线

- `SmartSteer_Status.md`：2026-09-04 当前项目状态总览，按软件边界 `6/6`、正式
  里程碑 `2/8` 和 M0 样本 `200/1000` 三种口径统计进度，并汇总完整代码架构、
  Canonical 参数、B0 Pure BC 单元验证、数据流、未解决问题和关键执行顺序。
- `../README.md`：项目入口、安装、运行与当前能力。
- `../System_overview.md`：系统级 SDD 与验收目标。
- `技术原理与代码架构.md`：算法原理、数据契约和代码映射。
- `REFERENCE_ARCHITECTURE_REMEDIATION_2026-08-30.md`：OffTerSim/UniAD
  参考架构整改。
- `BEVFUSION_REPOSITORY_REMEDIATION_2026-08-30.md`：四项目架构复核、
  BEV fusion 与仓库整理记录。
- `SENSING_BEV_STRICT_VALIDATION_2026-08-31.md`：当前 Camera/LiDAR 严格
  双模态有效性契约、TDD 证据与闭环 fail-safe 数据流。
- `ENGINEERING_EXECUTION_ROADMAP.md`：从传感器健康、recorded replay、BC、
  CARLA、ORT/TensorRT 到 ROS 2/SIL/HIL 的当前执行任务与 Exit Gate。
- `SENSOR_HEALTH_STAGE1A_2026-08-31.md`：Stage 1A 统一健康门禁、1000-case
  故障矩阵、最大制动和延迟证据；Stage 1B 环境边界保持待验收。
- `CARLA_STAGE1B_TASK1_BASELINE_2026-08-31.md`：Stage 1B Task 1 固定 CARLA
  版本、地图、车辆、随机种子和传感器配置，记录三次时序一致性验收入口；当前为
  单元验证，真实 CARLA 证据待生成。
- `CARLA_STAGE1B_TASK2_SENSOR_HEALTH_2026-09-01.md`：Stage 1B Task 2 无模型
  CARLA Sensor→HealthGate、1000-tick false rejection/skew/latency 统计与 world
  timestamp 修正；当前为单元验证，真实 CARLA 指标待生成。
- `superpowers/specs/2026-08-31-sensor-health-stage1-design.md` 与
  `superpowers/plans/2026-08-31-sensor-health-stage1.md`：本轮设计规格和 TDD
  实施计划。
- `superpowers/specs/2026-09-03-carla-0916-data-first-replay-design.md`：将真实
  初始 episode 提升为 CARLA 0.9.16 Canonical 事实锚点，统一采集配置，并规定
  recorded SensorHealthGate→BEV→Policy→Safety→Control 网络 smoke 的安全与追溯边界；
  设计已于 2026-09-03 书面确认，当前实现成熟度为离线验证；随机模型结果不具备
  模型性能或闭环验收效力。
- `superpowers/plans/2026-09-03-carla-0916-data-first-replay.md`：上述设计的 TDD
  实施计划，按结构化 Canonical、配置驱动采集、只读 loader、health-first replay、
  网络 smoke、artifact 与全量验证七个任务记录 Red/Green 命令和实时证据；本轮
  七项均已完成，真实 CARLA 与正式模型等外部 Gate 继续保持未关闭。
- `superpowers/specs/2026-09-04-staged-policy-learning-b0-design.md`：固定
  `Pure BC → Affordance+BC → Affordance+RSSM+BC → BC-initialized Dreamer`
  的递进式研究架构，并把当前实施范围限定为不经过 RSSM 的 B0 Pure BC；聊天设计
  与书面规格均已确认，B0 软件实现当前为单元验证；正式专家数据/权重与冻结 test
  性能 Gate 仍未关闭。
- `superpowers/plans/2026-09-04-staged-policy-learning-b0.md`：B0 Pure BC 的九任务
  TDD 实施计划，覆盖动力学/配置契约、确定性网络、专家数据、指标、checkpoint、
  train/eval CLI、recorded replay、全量验证和实时文档证据；已按内联方式执行，
  Task 1–8 已完成，Task 9 记录最终验证与文档同步。

## 审计与整改日志

- `SDD_AUDIT_2026-08-28.md`：SDD 深度审计及后续日志。
- `TDD_AUDIT_2026-08-28.md`：测试覆盖与 Red→Green 日志。
- `SDD_REMEDIATION_2026-08-28.md`：P0/P1/P2 整改状态。
- `METHOD_FRAMEWORK_REMEDIATION_2026-08-29.md`：方法框架整改记录。

## Phase 与部署过程文档

- `PHASE2_PROGRESS_REPORT.md`：Phase 1/2 历史过程记录。
- `PHASE3_PHASE4_PROGRESS.md`：Phase 3/4 历史过程记录。
- `PHASE5_DEPLOYMENT_GUIDE.md`：当前部署与仿真操作指南。

## 设计输入

- `relatetalk.md`：用户提供的方法框架原始输入；保持原文，不作为实现状态。

## 状态使用规则

1. 当前能力以自动化测试、最新日期整改报告和 `README.md` 为准。
2. 历史报告中的测试数字只代表当时基线，不做覆盖式修改。
3. Fake/Mock 只能关闭 CPU 软件契约，不能关闭真实数据、物理仿真或硬件验收。
4. 新增架构变更必须同步 SDD、TDD 日志和本索引。
