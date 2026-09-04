"""Health-first orchestration for read-only recorded CARLA frames."""
from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
import time
from typing import Any, Mapping, Optional, TYPE_CHECKING

import numpy as np
import torch

from orad_ros2.vehicle_control_node import PurePursuitController
from perception.bev_fusion import BEVFusion
from policy.hybrid_policy import HybridPolicy
from replay.carla_dataset import RecordedCarlaFrame
from safety.kinematic_filter import SafetyFilter
from safety.supervisor import (
    SafetyMode,
    SafetySupervisor,
    SafetySupervisorConfig,
)
from sim.carla_closed_loop import (
    occupancy_from_points,
    policy_trajectory_to_waypoints,
)
from sim.carla_baseline import sha256_file
from utils.schema import Observation
from utils.sensor_health import SensorHealthGate, SensorHealthReport

if TYPE_CHECKING:
    from configuration.system import SystemStackConfig


@dataclass(frozen=True)
class PreparedReplayFrame:
    frame: RecordedCarlaFrame
    health: SensorHealthReport
    observation: Optional[Observation]
    health_latency_ms: float
    startup_context: str


@dataclass(frozen=True)
class ReplayModelBundle:
    perception: Any
    policy: Any
    model_mode: str
    checkpoint_hashes: Mapping[str, str]
    model_performance_valid: bool
    closed_loop_acceptance_valid: bool


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported PyTorch
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} does not contain a state dict")
    for name, value in payload.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint parameter {name} is not a tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"checkpoint parameter {name} is non-finite")
    return payload


def load_model_bundle(
        stack: "SystemStackConfig", *,
        perception_checkpoint: Optional[Path],
        policy_checkpoint: Optional[Path],
        allow_random_models: bool,
        seed: int) -> ReplayModelBundle:
    """Build either two strict checkpoint models or explicit random smoke models."""
    one_checkpoint = (
        (perception_checkpoint is None) != (policy_checkpoint is None))
    if one_checkpoint:
        raise ValueError("both perception and policy checkpoints are required")
    if perception_checkpoint is None and not allow_random_models:
        raise ValueError(
            "both checkpoints are required unless allow_random_models is true")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    perception = BEVFusion(stack.bev).eval()
    policy = HybridPolicy(stack.policy).eval()
    if perception_checkpoint is None:
        return ReplayModelBundle(
            perception=perception,
            policy=policy,
            model_mode="random-model-network-smoke",
            checkpoint_hashes={},
            model_performance_valid=False,
            closed_loop_acceptance_valid=False,
        )
    perception_path = Path(perception_checkpoint)
    policy_path = Path(policy_checkpoint)
    perception.load_state_dict(_load_state_dict(perception_path), strict=True)
    policy.load_state_dict(_load_state_dict(policy_path), strict=True)
    return ReplayModelBundle(
        perception=perception,
        policy=policy,
        model_mode="checkpoint-recorded-replay",
        checkpoint_hashes={
            "perception": sha256_file(perception_path),
            "policy": sha256_file(policy_path),
        },
        model_performance_valid=False,
        closed_loop_acceptance_valid=False,
    )


def prepare_recorded_frame(
        frame: RecordedCarlaFrame, gate: SensorHealthGate, *,
        calibration_version: str) -> PreparedReplayFrame:
    """Gate raw sensors and construct Observation only after acceptance."""
    started = time.perf_counter()
    report = gate.evaluate(
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
        calibration_version=calibration_version,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    startup_context = (
        "startup" if frame.sample_index < gate.config.imu_steps
        else "steady_state")
    observation = None
    if report.valid:
        timestamps = {
            **{
                f"camera_{name}": timestamp
                for name, timestamp in zip(
                    ("front", "rear", "top"), frame.camera_timestamps)
            },
            "lidar": frame.lidar_timestamp,
            "imu_latest": frame.imu_timestamps[-1],
        }
        observation = Observation(
            timestamp=frame.timestamp,
            simulator_frame=frame.frame_id,
            images=list(frame.images),
            point_cloud=frame.canonical_point_cloud,
            imu_history=frame.imu_history,
            ego_state=frame.ego_state,
            camera_frames=frame.camera_frames,
            lidar_frame=frame.lidar_frame,
            imu_frames=frame.imu_frames,
            sensor_timestamps=timestamps,
        )
    return PreparedReplayFrame(
        frame=frame,
        health=report,
        observation=observation,
        health_latency_ms=latency_ms,
        startup_context=startup_context,
    )


def maximum_brake_record(
        prepared: PreparedReplayFrame, max_decel: float) -> dict:
    """Return the only permitted downstream result for an invalid frame."""
    return {
        "steering": 0.0,
        "speed": 0.0,
        "accel": float(max_decel),
        "safety_mode": "emergency_stop",
        "model_invoked": False,
    }


class CarlaReplayPipeline:
    """Run one recorded frame through health, models, safety and control."""

    def __init__(
            self, stack: "SystemStackConfig", health_gate: SensorHealthGate,
            models: ReplayModelBundle,
            supervisor: Optional[SafetySupervisor] = None,
            safety_filter: Optional[SafetyFilter] = None,
            controller: Optional[PurePursuitController] = None):
        self.stack = stack
        self.health_gate = health_gate
        self.models = models
        self.supervisor = supervisor or SafetySupervisor(
            SafetySupervisorConfig(
                expected_frame="world",
                max_speed=stack.controller.max_speed,
                max_sensor_age=stack.sensor_health.max_sensor_age_seconds,
                max_sensor_skew=stack.sensor_health.max_sensor_skew_seconds,
                max_model_latency=stack.closed_loop.dt,
            ))
        self.safety_filter = safety_filter or SafetyFilter(stack.safety)
        self.controller = controller or PurePursuitController(stack.controller)

    @staticmethod
    def _command_dict(command) -> dict[str, float]:
        return {
            "steering": float(command.steering),
            "speed": float(command.speed),
            "accel": float(command.accel),
        }

    def _emergency_command(self, prepared: PreparedReplayFrame) -> dict[str, float]:
        trajectory = self.supervisor.emergency_trajectory(
            prepared.frame.timestamp)
        command = self.controller.compute(
            prepared.frame.ego_state, trajectory, self.stack.closed_loop.dt)
        return self._command_dict(command)

    def _is_maximum_brake(self, command: Mapping[str, float]) -> bool:
        return bool(
            float(command["accel"]) <= self.stack.controller.max_decel + 1e-9
            and float(command["speed"]) <= self.stack.controller.stop_speed)

    def _base_record(self, prepared: PreparedReplayFrame) -> dict[str, Any]:
        return {
            "sample_index": prepared.frame.sample_index,
            "frame_id": prepared.frame.frame_id,
            "timestamp": prepared.frame.timestamp,
            "startup_context": prepared.startup_context,
            "health_valid": prepared.health.valid,
            "health_reasons": [
                reason.value for reason in prepared.health.reasons],
            "health_latency_ms": prepared.health_latency_ms,
            "sensor_age_seconds": prepared.health.max_sensor_age_seconds,
            "sensor_skew_seconds": prepared.health.max_sensor_skew_seconds,
            "model_mode": self.models.model_mode,
            "model_performance_valid": self.models.model_performance_valid,
            "closed_loop_acceptance_valid": (
                self.models.closed_loop_acceptance_valid),
        }

    @staticmethod
    def _model_tensors(observation: Observation):
        images = np.stack(observation.images, axis=0)
        images = np.ascontiguousarray(
            images.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        points = np.ascontiguousarray(
            observation.point_cloud, dtype=np.float32)
        imu = np.ascontiguousarray(observation.imu_history, dtype=np.float32)
        return (
            torch.from_numpy(images).unsqueeze(0),
            torch.from_numpy(points).unsqueeze(0),
            torch.from_numpy(imu).unsqueeze(0),
            torch.ones((1, 2), dtype=torch.bool),
        )

    def process(self, frame: RecordedCarlaFrame) -> dict[str, Any]:
        end_to_end_started = time.perf_counter()
        prepared = prepare_recorded_frame(
            frame, self.health_gate,
            calibration_version=(
                self.stack.sensor_health.expected_calibration_version))
        record = self._base_record(prepared)
        if prepared.observation is None:
            command = self._emergency_command(prepared)
            record.update({
                "model_invoked": False,
                "safety_mode": SafetyMode.EMERGENCY_STOP.value,
                "safety_reason": "sensor_health_rejected",
                "command": command,
                "maximum_brake": self._is_maximum_brake(command),
                "shapes": {},
                "finite": {"command": True},
                "model_latency_ms": 0.0,
                "end_to_end_latency_ms": (
                    time.perf_counter() - end_to_end_started) * 1000.0,
            })
            return record

        stage = "tensor_conversion"
        model_started = time.perf_counter()
        try:
            images, points, imu, modality_mask = self._model_tensors(
                prepared.observation)
            stage = "perception"
            with torch.no_grad():
                perceived = self.models.perception(
                    images, points, imu, modality_mask=modality_mask)
                bev = getattr(perceived, "bev", perceived)
                if not isinstance(bev, torch.Tensor):
                    raise TypeError("perception BEV output must be a torch.Tensor")
                if not torch.isfinite(bev).all():
                    raise ValueError("perception BEV output must be finite")
                stage = "policy"
                raw_trajectory = self.models.policy(bev, imu)
            if not isinstance(raw_trajectory, torch.Tensor):
                raise TypeError("policy output must be a torch.Tensor")
            if not torch.isfinite(raw_trajectory).all():
                raise ValueError("policy output must be finite")
            model_latency_ms = (time.perf_counter() - model_started) * 1000.0
            stage = "trajectory_contract"
            trajectory = policy_trajectory_to_waypoints(
                raw_trajectory.detach().cpu().numpy(), frame.ego_state,
                frame=self.stack.closed_loop.policy_frame,
                dt=self.stack.closed_loop.dt)
            trajectory.timestamp = frame.timestamp
            decision = self.supervisor.evaluate(
                trajectory,
                sensor_age=prepared.health.max_sensor_age_seconds,
                sensor_skew=prepared.health.max_sensor_skew_seconds,
                model_latency=model_latency_ms / 1000.0)
            if decision.mode is SafetyMode.EMERGENCY_STOP:
                command = self._emergency_command(prepared)
            else:
                stage = "safety_filter"
                canonical = frame.canonical_point_cloud
                non_padding = canonical[np.any(canonical != 0.0, axis=1)]
                occupancy = occupancy_from_points(
                    non_padding,
                    self.stack.closed_loop.bev_x_range,
                    self.stack.closed_loop.bev_y_range,
                    self.stack.closed_loop.bev_resolution,
                    vehicle_state=frame.ego_state)
                safe = self.safety_filter.filter(
                    trajectory, frame.ego_state, occupancy)
                stage = "controller"
                command = self._command_dict(self.controller.compute(
                    frame.ego_state, safe, self.stack.closed_loop.dt))
            command_finite = bool(np.isfinite(list(command.values())).all())
            if not command_finite:
                raise ValueError("controller command must be finite")
            record.update({
                "model_invoked": True,
                "safety_mode": decision.mode.value,
                "safety_reason": decision.reason,
                "command": command,
                "maximum_brake": self._is_maximum_brake(command),
                "shapes": {
                    "images": list(images.shape),
                    "points": list(points.shape),
                    "imu": list(imu.shape),
                    "bev": list(bev.shape),
                    "policy": list(raw_trajectory.shape),
                },
                "finite": {
                    "bev": True, "policy": True, "command": True},
                "model_latency_ms": model_latency_ms,
                "end_to_end_latency_ms": (
                    time.perf_counter() - end_to_end_started) * 1000.0,
            })
            return record
        except Exception as error:
            command = self._emergency_command(prepared)
            record.update({
                "model_invoked": stage not in {"tensor_conversion"},
                "safety_mode": SafetyMode.EMERGENCY_STOP.value,
                "safety_reason": "pipeline_exception",
                "failure_stage": stage,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "command": command,
                "maximum_brake": self._is_maximum_brake(command),
                "shapes": {},
                "finite": {"command": True},
                "model_latency_ms": (
                    time.perf_counter() - model_started) * 1000.0,
                "end_to_end_latency_ms": (
                    time.perf_counter() - end_to_end_started) * 1000.0,
            })
            return record
