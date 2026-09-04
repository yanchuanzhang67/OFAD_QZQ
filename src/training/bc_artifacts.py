"""Auditable run artifacts and strict checkpoints for Pure BC training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union
import uuid

import torch
import torch.nn as nn

from policy.bc_policy import BCPolicy
from policy.config import BCPolicyConfig
from sim.carla_baseline import sha256_file

__all__ = [
    "BCRunWriter",
    "load_bc_checkpoint",
    "load_frozen_perception_checkpoint",
    "resolve_checkpoint_index",
    "save_bc_checkpoint",
    "write_checkpoint_index",
]


CHECKPOINT_SCHEMA = "new-orad-policy-checkpoint-v1"
MODEL_FAMILY = "pure_bc_v1"
REQUIRED_LINEAGE = frozenset({
    "git",
    "config_sha256",
    "dataset_sha256",
    "split_sha256",
    "calibration_sha256",
    "perception_checkpoint_sha256",
    "seed",
    "resolved_command",
    "epoch",
    "best_metric",
})


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_bytes_exclusive(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_save_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete checkpoint without replacing an existing weight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(path) from None
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def _require_tensor_state(
        state: object, *, context: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError(f"{context} state_dict must be a mapping")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"{context} state_dict must contain named tensors")
        if not torch.isfinite(value).all():
            raise ValueError(
                f"{context} state_dict contains non-finite tensor: {name}")
    return state


def _validate_finite_tree(value: object, *, context: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"{context} contains a non-finite tensor")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite scalar")
    elif isinstance(value, Mapping):
        for nested in value.values():
            _validate_finite_tree(nested, context=context)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_finite_tree(nested, context=context)


@dataclass
class BCRunWriter:
    """Write a run under ``.incomplete`` and publish success atomically."""

    root: Path
    run_id: str
    working_path: Path
    final_path: Path

    @property
    def checkpoint_dir(self) -> Path:
        return self.working_path / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.working_path / "logs"

    @classmethod
    def create(
            cls, root: Union[Path, str], run_id: str,
            metadata: Mapping[str, Any]) -> "BCRunWriter":
        root_path = Path(root)
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be one non-empty path component")
        working_path = root_path / ".incomplete" / run_id
        final_path = root_path / run_id
        if working_path.exists() or final_path.exists():
            raise FileExistsError(final_path)
        working_path.mkdir(parents=True, exist_ok=False)
        writer = cls(root_path, run_id, working_path, final_path)
        writer.checkpoint_dir.mkdir()
        writer.log_dir.mkdir()
        _write_bytes_exclusive(
            working_path / "manifest.json", _json_bytes(metadata))
        return writer

    def complete(self, metrics: Mapping[str, Any]) -> Path:
        if not self.working_path.is_dir():
            raise RuntimeError("BC run working directory is unavailable")
        if self.final_path.exists():
            raise FileExistsError(self.final_path)
        _write_bytes_exclusive(
            self.working_path / "metrics.json", _json_bytes(metrics))
        _write_bytes_exclusive(
            self.working_path / "_SUCCESS", b"complete\n")
        os.replace(self.working_path, self.final_path)
        return self.final_path


def _validate_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(lineage, Mapping):
        raise TypeError("checkpoint lineage must be a mapping")
    missing = sorted(REQUIRED_LINEAGE.difference(lineage))
    if missing:
        raise ValueError(f"checkpoint lineage missing keys: {missing}")
    for name in (
            "config_sha256", "dataset_sha256", "split_sha256",
            "calibration_sha256", "perception_checkpoint_sha256"):
        value = lineage[name]
        try:
            valid_hash = isinstance(value, str) and len(value) == 64
            if valid_hash:
                int(value, 16)
        except ValueError:
            valid_hash = False
        if not valid_hash:
            raise ValueError(f"checkpoint lineage {name} must be SHA-256")
    git = lineage["git"]
    if (not isinstance(git, Mapping)
            or not isinstance(git.get("commit"), str)
            or not git["commit"]
            or not isinstance(git.get("dirty"), bool)):
        raise ValueError("checkpoint lineage git metadata is invalid")
    for name in ("seed", "epoch"):
        value = lineage[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"checkpoint lineage {name} must be a non-negative integer")
    if (not isinstance(lineage["resolved_command"], str)
            or not lineage["resolved_command"].strip()):
        raise ValueError("checkpoint lineage resolved_command must be non-empty")
    best_metric = lineage["best_metric"]
    if (isinstance(best_metric, bool)
            or not isinstance(best_metric, (int, float))
            or not math.isfinite(float(best_metric))):
        raise ValueError("checkpoint lineage best_metric must be finite")
    _validate_finite_tree(lineage, context="checkpoint lineage")
    return dict(lineage)


def save_bc_checkpoint(
        path: Union[Path, str], policy: BCPolicy,
        optimizer: Optional[torch.optim.Optimizer],
        lineage: Mapping[str, Any]) -> None:
    """Exclusively save one immutable Pure BC epoch checkpoint."""
    if not isinstance(policy, BCPolicy):
        raise TypeError("policy must be BCPolicy")
    if policy.config.family != MODEL_FAMILY:
        raise ValueError(f"policy model_family must be {MODEL_FAMILY}")
    state = _require_tensor_state(
        policy.state_dict(), context="BC checkpoint")
    optimizer_state = optimizer.state_dict() if optimizer else None
    _validate_finite_tree(
        optimizer_state, context="BC checkpoint optimizer_state")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_family": MODEL_FAMILY,
        "config": asdict(policy.config),
        "state_dict": dict(state),
        "optimizer_state": optimizer_state,
        "lineage": _validate_lineage(lineage),
    }
    _torch_save_exclusive(Path(path), payload)


def load_bc_checkpoint(
        path: Union[Path, str],
        expected_config: BCPolicyConfig) -> tuple[BCPolicy, dict[str, Any]]:
    """Strictly reconstruct a Pure BC policy from an audited checkpoint."""
    payload = _torch_load(Path(path))
    if not isinstance(payload, Mapping):
        raise TypeError("BC checkpoint payload must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid BC checkpoint schema_version")
    if payload.get("model_family") != MODEL_FAMILY:
        raise ValueError("invalid BC checkpoint model_family")
    if payload.get("config") != asdict(expected_config):
        raise ValueError("BC checkpoint config does not match expected config")
    if "lineage" not in payload:
        raise ValueError("BC checkpoint is missing lineage")
    _validate_lineage(payload["lineage"])
    _validate_finite_tree(
        payload.get("optimizer_state"),
        context="BC checkpoint optimizer_state")
    state = _require_tensor_state(
        payload.get("state_dict"), context="BC checkpoint")

    mean = state.get("ego_mean")
    if mean is not None and mean.shape != (expected_config.ego_dim,):
        raise ValueError("BC checkpoint ego_mean has invalid shape")
    std = state.get("ego_std")
    if std is not None and std.shape != (expected_config.ego_dim,):
        raise ValueError("BC checkpoint ego_std has invalid shape")
    if std is not None and not (std > 1e-6).all():
        raise ValueError("BC checkpoint ego_std must be finite and positive")

    policy = BCPolicy(expected_config)
    policy.load_state_dict(state, strict=True)
    return policy, dict(payload)


def write_checkpoint_index(
        index_path: Union[Path, str], checkpoint_path: Union[Path, str],
        *, epoch: int) -> None:
    """Atomically point ``best.json`` or ``last.json`` to a local weight."""
    index = Path(index_path)
    checkpoint = Path(checkpoint_path)
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("checkpoint index epoch must be a non-negative integer")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.resolve().parent != index.parent.resolve():
        raise ValueError("checkpoint index may reference only a local weight")
    payload = {
        "filename": checkpoint.name,
        "sha256": sha256_file(checkpoint),
        "epoch": epoch,
    }
    _write_bytes_atomic(index, _json_bytes(payload))


def resolve_checkpoint_index(index_path: Union[Path, str]) -> Path:
    """Resolve and hash-check a local checkpoint index."""
    index = Path(index_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint index must contain a JSON object")
    filename = payload.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("checkpoint index filename must be a local basename")
    digest = payload.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("checkpoint index hash must be a SHA-256 string")
    checkpoint = index.parent / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != digest:
        raise ValueError("checkpoint index hash does not match weight file")
    return checkpoint


def load_frozen_perception_checkpoint(
        path: Union[Path, str], perception: nn.Module) -> str:
    """Strictly load finite perception weights, then freeze the module."""
    checkpoint_path = Path(path)
    payload = _torch_load(checkpoint_path)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    tensor_state = _require_tensor_state(
        state, context="perception checkpoint")
    perception.load_state_dict(tensor_state, strict=True)
    perception.eval()
    perception.requires_grad_(False)
    return sha256_file(checkpoint_path)
