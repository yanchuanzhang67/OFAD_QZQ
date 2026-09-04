"""Transactional, non-overwriting artifacts for recorded CARLA replay."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _distribution(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": round(float(np.percentile(data, 50)), 12),
        "p95": round(float(np.percentile(data, 95)), 12),
        "max": round(float(np.max(data)), 12),
    }


def summarize_replay_records(records: Iterable[Mapping[str, Any]]) -> dict:
    rows = list(records)
    total = len(rows)
    valid = sum(bool(row.get("health_valid")) for row in rows)
    model_processed = sum(bool(row.get("model_invoked")) for row in rows)
    fail_safe = sum(
        row.get("safety_mode") == "emergency_stop" for row in rows)
    maximum_brake = sum(bool(row.get("maximum_brake")) for row in rows)
    safety_modes = Counter(str(row.get("safety_mode", "unknown")) for row in rows)
    reasons = Counter(
        reason for row in rows for reason in row.get("health_reasons", ()))
    finite_values = [
        bool(value)
        for row in rows
        for value in row.get("finite", {}).values()
    ]
    finite_rate = (
        sum(finite_values) / len(finite_values) if finite_values else 0.0)
    return {
        "total_frames": total,
        "valid_frames": valid,
        "rejected_frames": total - valid,
        "observed_rejection_rate": (total - valid) / max(total, 1),
        "model_processed_frames": model_processed,
        "fail_safe_frames": fail_safe,
        "maximum_brake_frames": maximum_brake,
        "safety_mode_counts": dict(sorted(safety_modes.items())),
        "rejection_reasons": dict(sorted(reasons.items())),
        "health_latency_ms": _distribution(
            row.get("health_latency_ms", 0.0) for row in rows),
        "model_latency_ms": _distribution(
            row.get("model_latency_ms", 0.0) for row in rows),
        "end_to_end_latency_ms": _distribution(
            row.get("end_to_end_latency_ms", 0.0) for row in rows),
        "network_boundary_finite_rate": finite_rate,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


class ReplayRunWriter:
    """Write into .incomplete and atomically publish only completed runs."""

    def __init__(
            self, working_path: Path, final_path: Path,
            frame_stream, metadata: Mapping[str, Any]):
        self.working_path = working_path
        self.final_path = final_path
        self._frame_stream = frame_stream
        self.metadata = dict(metadata)
        self.records: list[dict[str, Any]] = []
        self._completed = False

    @classmethod
    def create(
            cls, output_root: Path, run_id: str,
            metadata: Mapping[str, Any]) -> "ReplayRunWriter":
        root = Path(output_root)
        incomplete = root / ".incomplete"
        working = incomplete / run_id
        final = root / run_id
        if working.exists() or final.exists():
            raise FileExistsError(f"replay run already exists: {run_id}")
        incomplete.mkdir(parents=True, exist_ok=True)
        working.mkdir()
        _write_json_exclusive(working / "run.json", metadata)
        frame_stream = (working / "frames.jsonl").open(
            "x", encoding="utf-8", buffering=1)
        return cls(working, final, frame_stream, metadata)

    def write_frame(self, record: Mapping[str, Any]) -> None:
        if self._completed:
            raise RuntimeError("cannot write after replay completion")
        row = dict(record)
        self._frame_stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.records.append(row)

    def complete(self, summary: Mapping[str, Any]) -> Path:
        if self._completed:
            raise RuntimeError("replay run is already complete")
        self._frame_stream.flush()
        self._frame_stream.close()
        _write_json_exclusive(self.working_path / "summary.json", summary)
        (self.working_path / "_SUCCESS").write_text(
            "complete\n", encoding="utf-8")
        self.working_path.replace(self.final_path)
        self._completed = True
        return self.final_path

    def close_incomplete(self) -> None:
        if not self._frame_stream.closed:
            self._frame_stream.close()
