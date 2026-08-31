# New_ORAD 文档导航

本文档是 `docs/` 的入口。状态结论优先阅读“当前基线”；Phase 文档与带日期
的审计报告保留过程证据，不应单独作为当前完成度声明。

## 当前基线

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
- `superpowers/specs/2026-08-31-sensor-health-stage1-design.md` 与
  `superpowers/plans/2026-08-31-sensor-health-stage1.md`：本轮设计规格和 TDD
  实施计划。

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
