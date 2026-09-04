#!/usr/bin/env python3
"""Evaluate one strict Pure BC checkpoint on the manifest test split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import uuid

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from configuration.system import load_system_stack  # noqa: E402
from perception.bev_fusion import BEVFusion  # noqa: E402
from sim.carla_baseline import (  # noqa: E402
    collect_git_metadata,
    sha256_file,
)
from training.bc_artifacts import (  # noqa: E402
    BCRunWriter,
    load_bc_checkpoint,
    load_frozen_perception_checkpoint,
)
from training.bc_dataset import (  # noqa: E402
    ExpertBCDataset,
    collate_bc_samples,
)
from training.bc_engine import (  # noqa: E402
    evaluate_loader,
    set_deterministic_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate pure_bc_v1 on an immutable expert test split.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/system.yaml"))
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--perception-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"bc_eval_{timestamp}_{uuid.uuid4().hex[:8]}"


def _loader(dataset, args) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bc_samples,
        pin_memory=args.device == "cuda",
    )


def _require_matching_lineage(
        payload: dict, *, stack, dataset: ExpertBCDataset,
        perception_sha256: str) -> dict:
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("policy checkpoint lineage is missing")
    expected = {
        "config_sha256": stack.config_sha256,
        "dataset_sha256": dataset.manifest_sha256,
        "calibration_sha256": dataset.calibration_sha256,
        "perception_checkpoint_sha256": perception_sha256,
    }
    for name, value in expected.items():
        if lineage.get(name) != value:
            raise ValueError(
                f"policy checkpoint {name} mismatch: expected {value!r}")
    return lineage


def _write_failure_samples(path: Path, failures: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> Path:
    _validate_arguments(args)
    stack = load_system_stack(args.config)
    dataset = ExpertBCDataset.open(args.data_manifest, "test", stack)
    if len(dataset) == 0:
        raise ValueError("BC test split must contain at least one accepted sample")
    if args.smoke_test and not dataset.fixture:
        raise ValueError("--smoke-test requires manifest fixture=true")
    if not args.smoke_test and dataset.fixture:
        raise ValueError("fixture=true manifest requires --smoke-test")
    if dataset.calibration_sha256 is None:  # pragma: no cover - non-empty split
        raise ValueError("BC test calibration hash is unavailable")

    device = torch.device(args.device)
    perception = BEVFusion(stack.bev)
    perception_hash = load_frozen_perception_checkpoint(
        args.perception_checkpoint, perception)
    policy, checkpoint_payload = load_bc_checkpoint(
        args.policy_checkpoint, stack.bc_policy)
    lineage = _require_matching_lineage(
        checkpoint_payload,
        stack=stack,
        dataset=dataset,
        perception_sha256=perception_hash,
    )
    seed = lineage.get("seed")
    if not isinstance(seed, int):
        raise ValueError("policy checkpoint seed must be an integer")
    determinism = set_deterministic_seed(seed)
    metrics = evaluate_loader(
        perception,
        policy,
        _loader(dataset, args),
        device,
        max_speed=stack.controller.max_speed,
    )
    required_metrics = {
        "ade_m", "fde_m", "heading_mae_rad", "velocity_mae_mps",
        "smoothness_m", "nonfinite_trajectory_count", "sample_count",
        "thresholds", "thresholds_passed", "failed_samples",
    }
    missing = sorted(required_metrics.difference(metrics))
    if missing:
        raise RuntimeError(f"BC evaluation metrics are incomplete: {missing}")
    if metrics["nonfinite_trajectory_count"] != 0:
        raise FloatingPointError("BC evaluation produced non-finite trajectories")

    policy_hash = sha256_file(args.policy_checkpoint)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "pure_bc_v1",
        "split": "test",
        "fixture": dataset.fixture,
        "git": collect_git_metadata(_ROOT),
        "config_sha256": stack.config_sha256,
        "config_canonical_sha256": stack.config_canonical_sha256,
        "dataset_sha256": dataset.manifest_sha256,
        "test_split_sha256": dataset.split_sha256,
        "training_split_sha256": lineage["split_sha256"],
        "calibration_sha256": dataset.calibration_sha256,
        "perception_checkpoint_sha256": perception_hash,
        "policy_checkpoint_sha256": policy_hash,
        "policy_checkpoint_epoch": lineage["epoch"],
        "seed": seed,
        "resolved_command": shlex.join([sys.executable, *sys.argv]),
        "determinism": determinism,
        "rejection_count": len(dataset.rejections),
    }
    performance_valid = bool(
        not args.smoke_test and metrics["thresholds_passed"])
    result = dict(metrics)
    failures = result.pop("failed_samples")
    if not isinstance(failures, list):  # pragma: no cover - engine contract
        raise TypeError("BC failed_samples must be a list")
    result.update({
        "model_family": "pure_bc_v1",
        "model_performance_valid": performance_valid,
        "smoke_test": args.smoke_test,
        "split": "test",
        "rejection_count": len(dataset.rejections),
        "config_sha256": stack.config_sha256,
        "dataset_sha256": dataset.manifest_sha256,
        "test_split_sha256": dataset.split_sha256,
        "calibration_sha256": dataset.calibration_sha256,
        "perception_checkpoint_sha256": perception_hash,
        "policy_checkpoint_sha256": policy_hash,
        "policy_checkpoint_epoch": lineage["epoch"],
        "seed": seed,
        "failure_sample_count": len(failures),
    })
    writer = BCRunWriter.create(args.run_dir, _run_id(), metadata)
    _write_failure_samples(
        writer.working_path / "failure_samples.jsonl", failures)
    return writer.complete(result)


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = run(args)
    except Exception as error:
        print(f"BC evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"BC evaluation complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
