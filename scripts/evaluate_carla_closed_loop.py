#!/usr/bin/env python3
"""CARLA closed-loop evaluation entry point (ORAD Phase 5).

Pipeline (per synchronous tick): RGB cameras + LiDAR + IMU -> BEVFusion ->
HybridPolicy -> explicit occupancy routing -> SafetySupervisor/SafetyFilter ->
PurePursuitController -> carla.VehicleControl. Aggregates pass-rate,
collision-rate, safety intervention and attitude-stability metrics.

Usage:
    python scripts/evaluate_carla_closed_loop.py \
        --host 127.0.0.1 --port 2000 --episodes 10 \
        --goal-x 80 --goal-y 0 \
        --perception-ckpt bev.pt --policy-ckpt policy.pt

Requires the CARLA PythonAPI (`pip install carla`) and a running CARLA server
in synchronous mode. Formal evaluation requires perception and policy
checkpoints; random initialization is available only for explicit smoke tests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configuration.system import load_system_stack  # noqa: E402
from perception.bev_fusion import BEVFusion  # noqa: E402
from policy.hybrid_policy import HybridPolicy  # noqa: E402
from safety.kinematic_filter import SafetyFilter  # noqa: E402
from orad_ros2.vehicle_control_node import (  # noqa: E402
    PurePursuitController)
from sim.carla_closed_loop import (  # noqa: E402
    CarlaSensorStack, CarlaClosedLoopRunner,
    aggregate_episodes, _HAS_CARLA)
from utils.sensor_health import SensorHealthGate  # noqa: E402

DEFAULT_CONFIG = os.path.abspath(
    os.path.join(_HERE, "..", "configs", "system.yaml"))


def build_perceiver(model: BEVFusion, device) -> callable:
    """Closure adapting raw numpy sensor arrays to BEVFusion torch inputs."""
    def perceive(images, points, imu, attitude):
        imgs = np.stack([np.asarray(im, dtype=np.float32) / 255.0
                         for im in images], axis=0)            # (N,H,W,3)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).unsqueeze(0)
        pts = torch.from_numpy(points).unsqueeze(0)
        imu_t = torch.from_numpy(imu).unsqueeze(0)
        att = torch.from_numpy(attitude).unsqueeze(0)
        # 闭环入口显式声明 [Camera, LiDAR] 均有效。Observation 与
        # BEVFusion 会分别在 schema/张量边界拒绝空、错形或非有限数据；
        # 因此这里不能用单模态 mask 绕过坏帧。
        modality_mask = torch.ones(
            (imgs.shape[0], 2), dtype=torch.bool, device=device)
        with torch.no_grad():
            out = model(imgs.to(device), pts.to(device),
                        imu_t.to(device), imu_attitude=att.to(device),
                        modality_mask=modality_mask)
        return out
    return perceive


def build_policy_caller(policy: HybridPolicy, device) -> callable:
    def policy_call(bev, imu):
        imu_t = torch.from_numpy(imu).unsqueeze(0).to(device)
        with torch.no_grad():
            return policy(bev.to(device), imu_t)
    return policy_call


def connect_carla(host: str, port: int):
    import carla
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    return client


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ORAD CARLA closed-loop eval")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--goal-x", type=float, default=80.0)
    ap.add_argument("--goal-y", type=float, default=0.0)
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="canonical system YAML")
    ap.add_argument("--perception-ckpt", default=None,
                    help="BEVFusion state_dict (required for formal evaluation)")
    ap.add_argument("--policy-ckpt", default=None,
                    help="HybridPolicy state_dict (required unless --allow-random-policy)")
    ap.add_argument("--allow-random-policy", action="store_true",
                    help="random perception/policy development smoke test only")
    ap.add_argument("--policy-frame", default=None, choices=["ego", "world"])
    ap.add_argument("--occupancy-source", default=None,
                    choices=["lidar", "learned", "fused"],
                    help="override runtime occupancy source from system YAML")
    ap.add_argument("--device", default="cuda"
                    if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    if not _HAS_CARLA:
        raise SystemExit("carla PythonAPI not installed; `pip install carla`")
    device = torch.device(args.device)
    if not args.allow_random_policy:
        missing = [name for name, value in (
            ("--perception-ckpt", args.perception_ckpt),
            ("--policy-ckpt", args.policy_ckpt),
        ) if not value]
        if missing:
            raise SystemExit(
                f"{', '.join(missing)} required for formal evaluation")
    for name, path in (("perception", args.perception_ckpt),
                       ("policy", args.policy_ckpt)):
        if path and not os.path.isfile(path):
            raise SystemExit(f"{name} checkpoint not found: {path}")

    stack = load_system_stack(
        args.config, policy_frame=args.policy_frame,
        occupancy_source=args.occupancy_source)

    bev_model = BEVFusion(stack.bev).to(device).eval()
    policy = HybridPolicy(stack.policy).to(device).eval()
    if args.perception_ckpt:
        state = torch.load(args.perception_ckpt, map_location=device)
        bev_model.load_state_dict(state, strict=True)
    if args.policy_ckpt:
        state = torch.load(args.policy_ckpt, map_location=device)
        policy.load_state_dict(state, strict=True)
    safety = SafetyFilter(stack.safety)
    controller = PurePursuitController(stack.controller)
    cl_cfg = stack.closed_loop

    client = connect_carla(args.host, args.port)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cl_cfg.dt
    world.apply_settings(settings)
    bp_lib = world.get_blueprint_library()
    veh_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_pts = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(veh_bp, spawn_pts[0])

    metrics_all = []
    sensors = None
    try:
        sensors = CarlaSensorStack(
            world, vehicle, cl_cfg, num_cameras=stack.bev.num_cameras,
            image_size=stack.bev.image_size,
            calibration_version=(
                stack.sensor_health.expected_calibration_version))
        perceive = build_perceiver(bev_model, device)
        policy_call = build_policy_caller(policy, device)
        for ep in range(args.episodes):
            vehicle.set_transform(spawn_pts[0])
            runner = CarlaClosedLoopRunner(
                world, vehicle, sensors, perceive, policy_call,
                safety, controller, cl_cfg,
                health_gate=SensorHealthGate(stack.sensor_health),
                goal=(args.goal_x, args.goal_y),
                max_steps=args.max_steps)
            m = runner.run()
            metrics_all.append(m)
            print(f"[episode {ep}] {m.to_summary()}", flush=True)
    finally:
        if sensors is not None:
            sensors.destroy()
        try:
            vehicle.destroy()
        except Exception:
            pass
        settings.synchronous_mode = False
        world.apply_settings(settings)

    summary = aggregate_episodes(metrics_all)
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass_rate"] > 0.9 and summary["collision_rate"] == 0.0 \
        else 1


if __name__ == "__main__":
    raise SystemExit(main())
