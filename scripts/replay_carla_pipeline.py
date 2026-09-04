#!/usr/bin/env python3
"""Replay a recorded CARLA episode through health, networks and safety."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from configuration.system import load_system_stack  # noqa: E402
from replay.artifacts import (  # noqa: E402
    ReplayRunWriter,
    summarize_replay_records,
)
from replay.carla_dataset import CarlaRecordedEpisode  # noqa: E402
from replay.carla_pipeline import (  # noqa: E402
    CarlaReplayPipeline,
    load_model_bundle,
)
from sim.carla_baseline import collect_git_metadata  # noqa: E402
from utils.sensor_health import SensorHealthGate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only CARLA dataset replay through HealthGate, networks, "
            "safety and control."))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/system.yaml"))
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/carla_replay"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--allow-legacy-provenance", action="store_true")
    parser.add_argument("--allow-random-models", action="store_true")
    parser.add_argument("--perception-checkpoint", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"replay_{timestamp}_{uuid.uuid4().hex[:8]}"


def run(args: argparse.Namespace) -> Path:
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    stack = load_system_stack(args.config)
    episode = CarlaRecordedEpisode.open(
        args.episode, stack,
        allow_legacy_provenance=args.allow_legacy_provenance)
    models = load_model_bundle(
        stack,
        perception_checkpoint=args.perception_checkpoint,
        policy_checkpoint=args.policy_checkpoint,
        allow_random_models=args.allow_random_models,
        seed=args.seed)
    git = collect_git_metadata(_ROOT)
    manifest = episode.manifest
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "config_path": str(stack.config_path),
        "config_sha256": stack.config_sha256,
        "config_canonical_sha256": stack.config_canonical_sha256,
        "episode_path": str(episode.path),
        "episode_sha256": episode.episode_sha256,
        "episode_schema": manifest.get("schema_version"),
        "carla_client_version": manifest.get("carla_client_version"),
        "carla_server_version": manifest.get("carla_server_version"),
        "map": manifest.get("map"),
        "recorded_vehicle_blueprint": manifest.get("vehicle_blueprint"),
        "recorded_config_sha256": manifest.get("config_sha256"),
        "recorded_calibration_sha256": manifest.get("calibration_sha256"),
        "calibration_version": episode.calibration.get(
            "calibration_version"),
        "calibration_sha256": episode.calibration_sha256,
        "provenance_gaps": list(episode.provenance_gaps),
        "model_mode": models.model_mode,
        "checkpoint_hashes": dict(models.checkpoint_hashes),
        "model_performance_valid": models.model_performance_valid,
        "closed_loop_acceptance_valid": models.closed_loop_acceptance_valid,
        "random_seed": args.seed,
        "max_frames": args.max_frames,
    }
    writer = ReplayRunWriter.create(args.output_root, _run_id(), metadata)
    try:
        pipeline = CarlaReplayPipeline(
            stack, SensorHealthGate(stack.sensor_health), models)
        for frame in episode.iter_frames(max_frames=args.max_frames):
            writer.write_frame(pipeline.process(frame))
        summary = summarize_replay_records(writer.records)
        summary.update({
            "model_mode": models.model_mode,
            "model_performance_valid": models.model_performance_valid,
            "closed_loop_acceptance_valid": (
                models.closed_loop_acceptance_valid),
            "provenance_gaps": list(episode.provenance_gaps),
        })
        return writer.complete(summary)
    except Exception:
        writer.close_incomplete()
        raise


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = run(args)
    except Exception as error:
        print(f"replay failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"replay complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
