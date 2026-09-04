"""Cross-module runtime contract validation for the ORAD stack."""
from __future__ import annotations


def _grid_n(rng, resolution):
    return max(1, int(round((float(rng[1]) - float(rng[0])) / resolution)))


def validate_stack_configs(
        bev, policy, safety, controller, closed_loop, *, bc_policy=None) -> None:
    """Fail fast when duplicated public dimensions or physical limits drift."""
    errors = []
    expected_h = _grid_n(bev.bev_y_range, bev.bev_resolution)
    expected_w = _grid_n(bev.bev_x_range, bev.bev_resolution)
    checks = [
        (policy.bev_channels, bev.bev_channels, "policy.bev_channels"),
        (policy.bev_h, expected_h, "policy.bev_h"),
        (policy.bev_w, expected_w, "policy.bev_w"),
        (policy.imu_in_channels, bev.imu_in_channels, "policy.imu_in_channels"),
        (policy.imu_steps, bev.imu_steps, "policy.imu_steps"),
        (closed_loop.bev_x_range, bev.bev_x_range, "closed_loop.bev_x_range"),
        (closed_loop.bev_y_range, bev.bev_y_range, "closed_loop.bev_y_range"),
        (closed_loop.bev_resolution, bev.bev_resolution,
         "closed_loop.bev_resolution"),
        (closed_loop.num_points, bev.num_points, "closed_loop.num_points"),
        (closed_loop.imu_steps, bev.imu_steps, "closed_loop.imu_steps"),
        (controller.wheelbase, safety.wheelbase, "controller.wheelbase"),
        (controller.max_steering, safety.max_steering,
         "controller.max_steering"),
        (controller.max_lateral_accel, safety.max_lateral_accel,
         "controller.max_lateral_accel"),
        (controller.max_accel, safety.max_accel, "controller.max_accel"),
        (controller.max_decel, safety.max_decel, "controller.max_decel"),
        (closed_loop.dt, safety.dt, "closed_loop.dt"),
    ]
    if bc_policy is not None:
        checks.extend([
            (bc_policy.bev_channels, bev.bev_channels,
             "bc_policy.bev_channels"),
            (bc_policy.bev_h, expected_h, "bc_policy.bev_h"),
            (bc_policy.bev_w, expected_w, "bc_policy.bev_w"),
            (bc_policy.imu_in_channels, bev.imu_in_channels,
             "bc_policy.imu_in_channels"),
            (bc_policy.imu_steps, bev.imu_steps, "bc_policy.imu_steps"),
            (bc_policy.waypoint_dt, closed_loop.dt,
             "bc_policy.waypoint_dt"),
        ])
    for actual, expected, name in checks:
        if actual != expected:
            errors.append(f"{name}={actual!r}, expected {expected!r}")
    if errors:
        raise ValueError("ORAD stack configuration mismatch: " + "; ".join(errors))
