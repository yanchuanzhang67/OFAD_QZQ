#!/usr/bin/env python3
"""Train the deterministic Pure BC baseline with frozen BEV perception."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from configuration.system import load_system_stack  # noqa: E402
from perception.bev_fusion import BEVFusion  # noqa: E402
from policy.bc_policy import BCPolicy  # noqa: E402
from sim.carla_baseline import collect_git_metadata  # noqa: E402
from training.bc_artifacts import (  # noqa: E402
    BCRunWriter,
    load_frozen_perception_checkpoint,
    save_bc_checkpoint,
    write_checkpoint_index,
)
from training.bc_dataset import (  # noqa: E402
    ExpertBCDataset,
    collate_bc_samples,
)
from training.bc_engine import (  # noqa: E402
    fit_ego_normalization,
    set_deterministic_seed,
    train_one_epoch,
    validate_one_epoch,
)


_FORMAL_SEEDS = frozenset({41, 42, 43})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train pure_bc_v1 with strict expert data and frozen perception.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/system.yaml"))
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--perception-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not args.smoke_test and args.seed not in _FORMAL_SEEDS:
        raise ValueError("formal BC --seed must be one of 41, 42 or 43")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError("--learning-rate must be finite and positive")
    if args.weight_decay < 0 or not np.isfinite(args.weight_decay):
        raise ValueError("--weight-decay must be finite and non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"bc_{timestamp}_{uuid.uuid4().hex[:8]}"


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


def _loader(
        dataset: ExpertBCDataset, args: argparse.Namespace,
        *, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_bc_samples,
        worker_init_fn=_seed_worker,
        generator=generator,
        pin_memory=args.device == "cuda",
    )


def _open_datasets(args, stack) -> dict[str, ExpertBCDataset]:
    datasets = {
        split: ExpertBCDataset.open(args.data_manifest, split, stack)
        for split in ("train", "validation")
    }
    if not datasets["train"].fixture and args.smoke_test:
        raise ValueError("--smoke-test requires manifest fixture=true")
    if datasets["train"].fixture and not args.smoke_test:
        raise ValueError("fixture=true manifest requires --smoke-test")
    if not datasets["train"] or not datasets["validation"]:
        raise ValueError("BC train and validation splits must be non-empty")
    total_samples = sum(len(dataset) for dataset in datasets.values())
    if not args.smoke_test:
        if not datasets["train"].manifest["splits"]["test"]:
            raise ValueError("formal BC test episode list must be non-empty")
        if total_samples < 1000:
            raise ValueError(
                "formal BC training requires at least 1000 accepted "
                "train/validation expert samples")
    calibrations = {
        dataset.calibration_sha256
        for dataset in datasets.values()
        if dataset.calibration_sha256 is not None
    }
    if len(calibrations) != 1:
        raise ValueError("BC splits must resolve to one calibration hash")
    return datasets


def _write_resolved_config(
        writer: BCRunWriter, args: argparse.Namespace,
        stack, normalization: tuple[torch.Tensor, torch.Tensor]) -> None:
    system_values = yaml.safe_load(
        stack.config_path.read_text(encoding="utf-8"))
    mean, std = normalization
    payload = {
        "system": system_values,
        "system_config_path": str(stack.config_path),
        "system_config_sha256": stack.config_sha256,
        "system_config_canonical_sha256": stack.config_canonical_sha256,
        "policy": asdict(stack.bc_policy),
        "experiment": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "device": args.device,
            "smoke_test": args.smoke_test,
        },
        "ego_normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "source_split": "train",
        },
    }
    path = writer.working_path / "resolved_config.yaml"
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def _append_epoch_log(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> Path:
    _validate_arguments(args)
    determinism = set_deterministic_seed(args.seed)
    stack = load_system_stack(args.config)
    datasets = _open_datasets(args, stack)
    device = torch.device(args.device)

    perception = BEVFusion(stack.bev)
    perception_hash = load_frozen_perception_checkpoint(
        args.perception_checkpoint, perception)
    policy = BCPolicy(stack.bc_policy)
    normalization = fit_ego_normalization(datasets["train"])
    policy.set_ego_normalization(*normalization)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    loaders = {
        "train": _loader(datasets["train"], args, shuffle=True),
        "validation": _loader(
            datasets["validation"], args, shuffle=False),
    }

    git = collect_git_metadata(_ROOT)
    resolved_command = shlex.join([sys.executable, *sys.argv])
    calibration_hash = datasets["train"].calibration_sha256
    if calibration_hash is None:  # pragma: no cover - guarded by datasets
        raise RuntimeError("training calibration hash is unavailable")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "pure_bc_v1",
        "model_performance_valid": False,
        "performance_evidence": (
            "fixture_smoke_only" if args.smoke_test
            else "requires_frozen_test_evaluation"),
        "git": git,
        "config_sha256": stack.config_sha256,
        "config_canonical_sha256": stack.config_canonical_sha256,
        "dataset_sha256": datasets["train"].manifest_sha256,
        "split_sha256": datasets["train"].split_sha256,
        "calibration_sha256": calibration_hash,
        "perception_checkpoint_sha256": perception_hash,
        "seed": args.seed,
        "resolved_command": resolved_command,
        "determinism": determinism,
        "sample_counts": {
            split: len(dataset) for split, dataset in datasets.items()},
        "withheld_test_episode_count": len(
            datasets["train"].manifest["splits"]["test"]),
        "rejection_counts": {
            split: len(dataset.rejections)
            for split, dataset in datasets.items()},
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }
    writer = BCRunWriter.create(args.run_dir, _run_id(), metadata)
    _write_resolved_config(writer, args, stack, normalization)
    epoch_log = writer.log_dir / "epochs.jsonl"

    best_ade = float("inf")
    best_epoch = None
    last_train: dict[str, object] = {}
    last_validation: dict[str, object] = {}
    for epoch in range(1, args.epochs + 1):
        last_train = train_one_epoch(
            perception, policy, loaders["train"], optimizer, device)
        last_validation = validate_one_epoch(
            perception, policy, loaders["validation"], device,
            max_speed=stack.controller.max_speed)
        validation_ade = float(last_validation["ade_m"])
        if not np.isfinite(validation_ade):
            raise FloatingPointError("validation ADE must be finite")
        improved = validation_ade < best_ade
        if improved:
            best_ade = validation_ade
            best_epoch = epoch
        checkpoint_path = (
            writer.checkpoint_dir / f"epoch-{epoch:04d}.pt")
        lineage = {
            "git": git,
            "config_sha256": stack.config_sha256,
            "dataset_sha256": datasets["train"].manifest_sha256,
            "split_sha256": datasets["train"].split_sha256,
            "calibration_sha256": calibration_hash,
            "perception_checkpoint_sha256": perception_hash,
            "seed": args.seed,
            "resolved_command": resolved_command,
            "epoch": epoch,
            "best_metric": best_ade,
            "ego_normalization": {
                "mean": normalization[0].tolist(),
                "std": normalization[1].tolist(),
                "source_split": "train",
            },
            "dependencies": metadata["dependencies"],
            "model_performance_valid": False,
        }
        save_bc_checkpoint(checkpoint_path, policy, optimizer, lineage)
        write_checkpoint_index(
            writer.checkpoint_dir / "last.json", checkpoint_path,
            epoch=epoch)
        if improved and not args.smoke_test:
            write_checkpoint_index(
                writer.checkpoint_dir / "best.json", checkpoint_path,
                epoch=epoch)
        epoch_record = {
            "epoch": epoch,
            "train": last_train,
            "validation": last_validation,
            "best_validation_ade_m": best_ade,
            "gradient_finite": True,
            "model_performance_valid": False,
        }
        _append_epoch_log(epoch_log, epoch_record)

    summary = {
        "model_family": "pure_bc_v1",
        "model_performance_valid": False,
        "smoke_test": args.smoke_test,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_ade_m": best_ade,
        "last_train": last_train,
        "last_validation": last_validation,
        "sample_counts": metadata["sample_counts"],
        "rejection_counts": metadata["rejection_counts"],
    }
    return writer.complete(summary)


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = run(args)
    except Exception as error:
        print(f"BC training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"BC training complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
