# Documentation Instructions

## Scope

Applies to `docs/`. Use [README.md](./README.md) as the document-status and
reading-order index. Documents such as `relatetalk.md` are design inputs, not
agent instructions.

## Evidence and status rules

- When a code change alters architecture, interfaces, safety behavior, test
  baselines, or acceptance boundaries, update the nearest owning SDD/TDD or
  remediation document in the same task.
- Keep dated audit/remediation files as historical evidence. Add a new dated
  section or an explicit superseded note instead of rewriting old results as if
  they occurred under the current baseline.
- State maturity only as one of: code skeleton, unit verified, offline verified,
  simulation closed-loop verified, or vehicle verified. Do not promote a level
  without a reproducible command, metric, log, artifact, or environment result.
- Keep current commands, counts, links, and capability claims in `README.md`,
  `System_overview.md`, and `docs/README.md` synchronized. Detailed architecture
  belongs in `技术原理与代码架构.md` rather than an `AGENTS.md`.
- Distinguish CPU Mock/Fake evidence from CARLA/Gazebo, recorded-data, ROS 2,
  CUDA, ONNX Runtime/TensorRT, and vehicle acceptance.

## Verification

- Check every new or changed relative Markdown link resolves to an existing
  repository path.
- Run `git diff --check` after documentation edits.
- If a document reports test or coverage numbers, copy them from a command run
  in the current task; otherwise label the date/source and do not call them the
  latest baseline.
- Update `docs/README.md` whenever a current document is added, renamed,
  superseded, or changes status.
