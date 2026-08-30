#!/usr/bin/env python3
"""CARLA closed-loop evaluation entry point (ORAD Phase 5).

Pipeline (per synchronous tick): RGB cameras + LiDAR + IMU -> BEVFusion ->
HybridPolicy -> SafetyFilter (curvature / lateral-accel / collision) ->
PurePursuitController -> carla.VehicleControl. Aggregates pass-rate,
collision-rate and attitude-stability metrics across episodes.

Usage:
    python scripts/evaluate_carla_closed_loop.py \
        --host 127.0.0.1 --port 2000 --episodes 10 \
        --goal-x 80 --goal-y 0 [--policy-ckpt policy.pt]

Requires the CARLA PythonAPI (`pip install carla`) and a running CARLA server
in synchronous mode. Without `--policy-ckpt` the policy is random-init (smoke
test only).
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

from perception.bev_fusion import BEVFusion, BEVFusionConfig  # noqa: E402
from policy.hybrid_policy import HybridPolicy, HybridPolicyConfig  # noqa: E402
from safety.kinematic_filter import SafetyFilter, SafetyFilterConfig  # noqa: E402
from orad_ros2.vehicle_control_node import (  # noqa: E402
    PurePursuitController, PurePursuitConfig)
from sim.carla_closed_loop import (  # noqa: E402
    ClosedLoopConfig, CarlaSensorStack, CarlaClosedLoopRunner,
    aggregate_episodes, _HAS_CARLA)
from utils.contracts import validate_stack_configs  # noqa: E402

NUM_CAMERAS = 3
IMAGE_SIZE = (192, 192)


def build_perceiver(model: BEVFusion, device) -> callable:
    """Closure adapting raw numpy sensor arrays to BEVFusion torch inputs."""
    def perceive(images, points, imu, attitude):
        imgs = np.stack([np.asarray(im, dtype=np.float32) / 255.0
                         for im in images], axis=0)            # (N,H,W,3)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).unsqueeze(0)
        pts = torch.from_numpy(points).unsqueeze(0)
        imu_t = torch.from_numpy(imu).unsqueeze(0)
        att = torch.from_numpy(attitude).unsqueeze(0)
        with torch.no_grad():
            out = model(imgs.to(device), pts.to(device),
                        imu_t.to(device), imu_attitude=att.to(device))
        return out.bev
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
    ap.add_argument("--policy-ckpt", default=None,
                    help="HybridPolicy state_dict (required unless --allow-random-policy)")
    ap.add_argument("--allow-random-policy", action="store_true",
                    help="development smoke test only; never use for evaluation")
    ap.add_argument("--policy-frame", default="ego", choices=["ego", "world"])
    ap.add_argument("--device", default="cuda"
                    if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    if not _HAS_CARLA:
        raise SystemExit("carla PythonAPI not installed; `pip install carla`")
    device = torch.device(args.device)
    if not args.allow_random_policy:
        if not args.policy_ckpt:
            raise SystemExit("--policy-ckpt is required for formal evaluation")
        if not os.path.isfile(args.policy_ckpt):
            raise SystemExit(f"policy checkpoint not found: {args.policy_ckpt}")

    bev_model = BEVFusion(
        BEVFusionConfig(num_cameras=NUM_CAMERAS, image_size=IMAGE_SIZE)
    ).to(device).eval()
    policy = HybridPolicy(HybridPolicyConfig()).to(device).eval()
    if args.policy_ckpt:
        sd = torch.load(args.policy_ckpt, map_location=device)
        policy.load_state_dict(sd, strict=True)
    safety_cfg = SafetyFilterConfig()
    controller_cfg = PurePursuitConfig()
    safety = SafetyFilter(safety_cfg)
    controller = PurePursuitController(controller_cfg)
    cl_cfg = ClosedLoopConfig(policy_frame=args.policy_frame)
    validate_stack_configs(bev_model.config, policy.config, safety_cfg,
                           controller_cfg, cl_cfg)

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
            world, vehicle, cl_cfg, num_cameras=NUM_CAMERAS,
            image_size=IMAGE_SIZE)
        perceive = build_perceiver(bev_model, device)
        policy_call = build_policy_caller(policy, device)
        for ep in range(args.episodes):
            vehicle.set_transform(spawn_pts[0])
            runner = CarlaClosedLoopRunner(
                world, vehicle, sensors, perceive, policy_call,
                safety, controller, cl_cfg,
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
