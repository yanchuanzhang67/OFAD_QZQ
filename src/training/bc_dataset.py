"""Strict expert dataset and episode-level split loader for B0 BC."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from replay.carla_dataset import (
    CarlaRecordedEpisode,
    DatasetContractError,
    RecordedCarlaFrame,
)
from sim.carla_baseline import sha256_file
from utils.sensor_health import SensorHealthGate
from utils.types import EgoDynamicsV1

if TYPE_CHECKING:
    from configuration.system import SystemStackConfig

_SCHEMA_VERSION = "new-orad-bc-manifest-v1"
_SPLIT_NAMES = ("train", "validation", "test")


def _error(code: str, message: str) -> None:
    raise DatasetContractError(code, message)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _error("bc_manifest_invalid", f"cannot read BC manifest: {error}")
    if not isinstance(value, dict):
        _error("bc_manifest_invalid", "BC manifest must contain a JSON object")
    return value


@dataclass(frozen=True)
class BCSample:
    images: torch.Tensor
    lidar: torch.Tensor
    imu: torch.Tensor
    ego: torch.Tensor
    expert: torch.Tensor
    mask: torch.Tensor
    sample_id: str
    episode_id: str
    tags: dict[str, str]


@dataclass(frozen=True)
class BCBatch:
    images: torch.Tensor
    lidar: torch.Tensor
    imu: torch.Tensor
    ego: torch.Tensor
    expert: torch.Tensor
    mask: torch.Tensor
    sample_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    tags: tuple[dict[str, str], ...]

    @property
    def size(self) -> int:
        return int(self.images.shape[0])


@dataclass(frozen=True)
class _SampleReference:
    episode: CarlaRecordedEpisode
    record_index: int
    episode_id: str
    tags: dict[str, str]


class ExpertBCDataset(Dataset):
    """Lazy expert samples selected by an immutable episode-level manifest."""

    def __init__(
            self, *, manifest_path: Path, manifest: Mapping[str, Any],
            split: str, stack: "SystemStackConfig",
            samples: tuple[_SampleReference, ...],
            rejections: tuple[dict[str, Any], ...], split_sha256: str):
        self.manifest_path = manifest_path
        self.manifest = dict(manifest)
        self.split = split
        self.stack = stack
        self._samples = samples
        self.rejections = rejections
        self.manifest_sha256 = sha256_file(manifest_path)
        self.split_sha256 = split_sha256
        self.dataset_version = str(manifest["dataset_version"])
        self.fixture = bool(manifest["fixture"])

    @classmethod
    def open(
            cls, manifest_path: Path, split: str,
            stack: "SystemStackConfig") -> "ExpertBCDataset":
        path = Path(manifest_path).resolve()
        manifest = _read_manifest(path)
        cls._validate_manifest_header(manifest, split, stack)
        resolved = cls._resolve_all_entries(path.parent, manifest["splits"])
        cls._reject_split_overlap(resolved)

        samples = []
        rejections = []
        selected_entries = resolved[split]
        for episode_path, recorded_hash, tags in selected_entries:
            episode = CarlaRecordedEpisode.open(episode_path, stack)
            if episode.episode_sha256 != recorded_hash:
                _error(
                    "episode_hash_mismatch",
                    f"episode hash differs for {episode_path.name}")
            gate = SensorHealthGate(stack.sensor_health)
            for record_index, frame in enumerate(episode.iter_frames()):
                if not frame.expert_label:
                    _error(
                        "expert_label_required",
                        f"frame {frame.frame_id} is not an expert label")
                if (frame.expert_trajectory is None
                        or frame.trajectory_mask is None):
                    _error(
                        "expert_label_required",
                        f"frame {frame.frame_id} has no expert trajectory")
                report = cls._health_report(frame, gate, stack)
                if not report.valid:
                    rejections.append({
                        "episode_id": episode.path.name,
                        "frame_id": frame.frame_id,
                        "reasons": [reason.value for reason in report.reasons],
                    })
                    continue
                samples.append(_SampleReference(
                    episode=episode,
                    record_index=record_index,
                    episode_id=episode.path.name,
                    tags=tags,
                ))
        split_payload = json.dumps(
            manifest["splits"][split], sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        return cls(
            manifest_path=path,
            manifest=manifest,
            split=split,
            stack=stack,
            samples=tuple(samples),
            rejections=tuple(rejections),
            split_sha256=hashlib.sha256(split_payload).hexdigest(),
        )

    @staticmethod
    def _validate_manifest_header(
            manifest: Mapping[str, Any], split: str,
            stack: "SystemStackConfig") -> None:
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            _error("bc_schema_mismatch", "unsupported BC manifest schema")
        if not str(manifest.get("dataset_version", "")).strip():
            _error("dataset_version_missing", "dataset_version must be non-empty")
        if not isinstance(manifest.get("fixture"), bool):
            _error("fixture_flag_invalid", "fixture must be boolean")
        if (manifest.get("calibration_version")
                != stack.sensor_health.expected_calibration_version):
            _error(
                "calibration_version_mismatch",
                "BC manifest calibration differs from system config")
        splits = manifest.get("splits")
        if not isinstance(splits, Mapping) or tuple(splits) != _SPLIT_NAMES:
            _error(
                "split_schema_invalid",
                "splits must be ordered train/validation/test")
        if split not in _SPLIT_NAMES:
            _error("split_name_invalid", f"unsupported BC split {split!r}")
        for name in _SPLIT_NAMES:
            if not isinstance(splits[name], list):
                _error("split_schema_invalid", f"split {name} must be a list")

    @staticmethod
    def _resolve_all_entries(
            root: Path, splits: Mapping[str, list]
            ) -> dict[str, list[tuple[Path, str, dict[str, str]]]]:
        resolved: dict[str, list[tuple[Path, str, dict[str, str]]]] = {
            name: [] for name in _SPLIT_NAMES}
        for split_name in _SPLIT_NAMES:
            for entry in splits[split_name]:
                if not isinstance(entry, Mapping):
                    _error("split_entry_invalid", "split entry must be an object")
                relative = Path(str(entry.get("episode", "")))
                if relative.is_absolute():
                    _error("unsafe_episode_path", "episode path must be relative")
                episode_path = (root / relative).resolve()
                if episode_path != root and root not in episode_path.parents:
                    _error("unsafe_episode_path", "episode path escapes manifest root")
                recorded_hash = str(entry.get("sha256", ""))
                if len(recorded_hash) != 64:
                    _error("episode_hash_invalid", "episode sha256 must be 64 hex")
                tags_raw = entry.get("tags", {})
                if not isinstance(tags_raw, Mapping):
                    _error("episode_tags_invalid", "episode tags must be an object")
                tags = {str(key): str(value) for key, value in tags_raw.items()}
                resolved[split_name].append(
                    (episode_path, recorded_hash, tags))
        return resolved

    @staticmethod
    def _reject_split_overlap(
            splits: Mapping[str, list[tuple[Path, str, dict[str, str]]]]) -> None:
        owners: dict[Path, str] = {}
        for split_name in _SPLIT_NAMES:
            for path, _, _ in splits[split_name]:
                if path in owners:
                    _error(
                        "split_episode_overlap",
                        f"episode {path.name} appears in {owners[path]} and "
                        f"{split_name}")
                owners[path] = split_name

    @staticmethod
    def _health_report(
            frame: RecordedCarlaFrame, gate: SensorHealthGate,
            stack: "SystemStackConfig"):
        return gate.evaluate(
            images=frame.images,
            point_cloud=frame.raw_point_cloud,
            imu_history=frame.imu_history,
            frame_id=frame.frame_id,
            reference_timestamp=frame.timestamp,
            camera_frames=frame.camera_frames,
            lidar_frame=frame.lidar_frame,
            imu_frames=frame.imu_frames,
            camera_timestamps=frame.camera_timestamps,
            lidar_timestamp=frame.lidar_timestamp,
            imu_timestamps=frame.imu_timestamps,
            calibration_version=(
                stack.sensor_health.expected_calibration_version),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> BCSample:
        reference = self._samples[index]
        frame = reference.episode.frame_at(reference.record_index)
        if frame.expert_trajectory is None or frame.trajectory_mask is None:
            _error(
                "expert_label_required",
                f"frame {frame.frame_id} has no expert trajectory")
        images = np.stack(frame.images, axis=0).transpose(0, 3, 1, 2)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(images)).to(torch.float32).div_(255.0)
        return BCSample(
            images=image_tensor,
            lidar=torch.from_numpy(frame.canonical_point_cloud.copy()),
            imu=torch.from_numpy(frame.imu_history.copy()),
            ego=torch.from_numpy(
                EgoDynamicsV1.from_vehicle_state(
                    frame.ego_state).to_array()),
            expert=torch.from_numpy(frame.expert_trajectory.copy()),
            mask=torch.from_numpy(frame.trajectory_mask.copy()),
            sample_id=f"{reference.episode_id}:{frame.frame_id}",
            episode_id=reference.episode_id,
            tags=dict(reference.tags),
        )


def collate_bc_samples(samples: list[BCSample]) -> BCBatch:
    """Stack validated fixed-shape samples without manufacturing data."""
    if not samples:
        raise ValueError("cannot collate an empty BC sample list")
    return BCBatch(
        images=torch.stack([sample.images for sample in samples]),
        lidar=torch.stack([sample.lidar for sample in samples]),
        imu=torch.stack([sample.imu for sample in samples]),
        ego=torch.stack([sample.ego for sample in samples]),
        expert=torch.stack([sample.expert for sample in samples]),
        mask=torch.stack([sample.mask for sample in samples]),
        sample_ids=tuple(sample.sample_id for sample in samples),
        episode_ids=tuple(sample.episode_id for sample in samples),
        tags=tuple(dict(sample.tags) for sample in samples),
    )
