"""ROS 2 vehicle control node: safe trajectory → Ackermann command.

Phase 4 of ORAD. A safety-filtered trajectory (``x, y, yaw, v`` waypoints)
is consumed by a Pure-Pursuit geometric controller producing a low-level
Ackermann command (steering angle + target speed + accel) and published to the
simulator command topic:

* CARLA:  ``/carla/ego_vehicle/ackermann_cmd`` (ackermann_msgs/AckermannDrive)
* Gazebo: ``/ackermann_cmd``               (ackermann_msgs/AckermannDriveStamped)

The controller (:class:`PurePursuitController`) is numpy-only and fully
unit-testable without rclpy. :class:`VehicleControlNode` wraps it in an rclpy
lifecycle and is imported lazily — when rclpy is unavailable the module still
imports so the controller can be exercised in CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utils.types import Trajectory, VehicleState, Waypoint

__all__ = [
    "PurePursuitConfig",
    "PurePursuitController",
    "ControlCommand",
    "VehicleControlNode",
    "main",
]


def _wrap(angle: float) -> float:
    """Wrap an angle (rad) into ``(-pi, pi]``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class ControlCommand:
    """Low-level Ackermann control produced by the controller."""

    steering: float = 0.0
    speed: float = 0.0
    accel: float = 0.0


@dataclass
class PurePursuitConfig:
    """Pure-pursuit + longitudinal P-controller parameters."""

    wheelbase: float = 2.5
    lookahead: float = 3.0
    lookahead_gain: float = 0.6      # L_d = lookahead + gain*v
    max_steering: float = 0.5
    max_speed: float = 20.0
    max_lateral_accel: float = 4.0   # m/s^2 (off-road stability cap)
    max_accel: float = 3.0
    max_decel: float = -5.0
    kp_speed: float = 1.0
    stop_speed: float = 0.1


class PurePursuitController:
    """Geometric pure-pursuit tracker + speed cap (numpy only)."""

    def __init__(self, config: Optional[PurePursuitConfig] = None):
        self.config = config or PurePursuitConfig()

    @property
    def max_curvature(self) -> float:
        return float(np.tan(self.config.max_steering) / self.config.wheelbase)

    def lookahead_distance(self, speed: float) -> float:
        c = self.config
        ld = c.lookahead + c.lookahead_gain * max(speed, 0.0)
        return float(np.clip(ld, c.lookahead, 12.0))

    def select_target(
        self, state: VehicleState, trajectory: Trajectory
    ) -> Optional[Tuple[int, float]]:
        """Pick the ``(index, distance)`` of the lookahead target waypoint.

        Returns the first waypoint whose distance from the rear-axle pose is
        at least the dynamic lookahead; falls back to the last waypoint for
        short trajectories; ``None`` for an empty trajectory.
        """
        if trajectory.length == 0:
            return None
        ld = self.lookahead_distance(state.speed)
        best_i, best_d = 0, 0.0
        for i, w in enumerate(trajectory.waypoints):
            d = float(np.hypot(w.x - state.x, w.y - state.y))
            if d >= ld:
                return i, d
            best_i, best_d = i, d
        return best_i, best_d

    def compute(
        self, state: VehicleState, trajectory: Trajectory, dt: float = 0.1
    ) -> ControlCommand:
        """Compute the Ackermann command tracking ``trajectory`` from ``state``."""
        c = self.config
        if trajectory.length == 0:
            return ControlCommand(accel=c.max_decel)
        target = self.select_target(state, trajectory)
        if target is None:
            return ControlCommand(accel=c.max_decel)
        ti, _ = target
        w = trajectory.waypoints[ti]
        dx = w.x - state.x
        dy = w.y - state.y
        ld = max(float(np.hypot(dx, dy)), 1e-3)
        alpha = _wrap(np.arctan2(dy, dx) - state.yaw)
        # pure pursuit: delta = atan2(2 L sin(alpha), L_d)
        steer = float(np.arctan2(2.0 * c.wheelbase * np.sin(alpha), ld))
        steer = float(np.clip(steer, -c.max_steering, c.max_steering))
        # speed cap by lateral accel: a_y = v^2 * |kappa|, kappa = tan(steer)/L
        k_abs = max(abs(np.tan(steer)) / c.wheelbase, 1e-9)
        v_lat = float(np.sqrt(c.max_lateral_accel / k_abs))
        v_target = min(float(w.speed), v_lat, c.max_speed)
        # longitudinal P-control on the speed error
        a_raw = c.kp_speed * (v_target - state.speed)
        accel = float(np.clip(a_raw, c.max_decel, c.max_accel))
        if v_target < c.stop_speed:
            v_target = 0.0
            accel = c.max_decel
        return ControlCommand(steering=steer, speed=v_target, accel=accel)


# -- rclpy node (lazy import) ----------------------------------------------
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from ackermann_msgs.msg import AckermannDrive  # type: ignore
    from ackermann_msgs.msg import AckermannDriveStamped  # type: ignore
    from nav_msgs.msg import Odometry  # type: ignore
    from std_msgs.msg import Float32MultiArray  # type: ignore

    _HAS_RCLPY = True
    _BaseNode = Node
except Exception:  # pragma: no cover - CI without ROS 2
    _HAS_RCLPY = False
    _BaseNode = object  # type: ignore[assignment, misc]


class VehicleControlNode(_BaseNode):  # type: ignore[misc]
    """ROS 2 node: safe trajectory + odom → Ackermann command."""

    def __init__(self, node_name: str = "vehicle_control") -> None:
        if not _HAS_RCLPY:
            raise RuntimeError(
                "rclpy not available - install a ROS 2 distribution to run "
                "VehicleControlNode (PurePursuitController is still usable).")
        super().__init__(node_name)
        self.declare_parameter("traj_topic", "/orad/safe_trajectory")
        self.declare_parameter("odom_topic", "/carla/ego_vehicle/odometry")
        self.declare_parameter("cmd_topic", "/carla/ego_vehicle/ackermann_cmd")
        self.declare_parameter("simulator", "carla")  # carla | gazebo
        self.declare_parameter("dt", 0.1)
        traj_topic = self.get_parameter("traj_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        self._sim = str(self.get_parameter("simulator").value)
        self._dt = float(self.get_parameter("dt").value)
        self._controller = PurePursuitController()
        self._state = VehicleState()
        self._trajectory = Trajectory()
        self._create_pub_sub(traj_topic, odom_topic, cmd_topic)
        self._timer = self.create_timer(self._dt, self._tick)

    def _create_pub_sub(
        self, traj_topic: str, odom_topic: str, cmd_topic: str
    ) -> None:
        if self._sim == "gazebo":
            self._pub = self.create_publisher(
                AckermannDriveStamped, cmd_topic, 10)
        else:
            self._pub = self.create_publisher(AckermannDrive, cmd_topic, 10)
        self.create_subscription(
            Float32MultiArray, traj_topic, self._traj_cb, 10)
        self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 10)

    @staticmethod
    def _traj_from_msg(msg) -> Trajectory:
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1, 4)
        wps = [
            Waypoint(x=float(r[0]), y=float(r[1]),
                     yaw=float(r[2]), speed=float(r[3]))
            for r in data
        ]
        return Trajectory(wps)

    def _odom_cb(self, msg) -> None:
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = float(np.arctan2(
            2.0 * (o.w * o.z + o.x * o.y),
            1.0 - 2.0 * (o.y * o.y + o.z * o.z)))
        v = msg.twist.twist.linear.x
        self._state = VehicleState(
            x=float(p.x), y=float(p.y), yaw=yaw, speed=v)

    def _traj_cb(self, msg) -> None:
        self._trajectory = self._traj_from_msg(msg)

    def _tick(self) -> None:
        cmd = self._controller.compute(
            self._state, self._trajectory, self._dt)
        drive = AckermannDrive(
            steering_angle=float(cmd.steering),
            steering_angle_velocity=0.0,
            speed=float(cmd.speed),
            acceleration=float(cmd.accel),
            jerk=0.0,
        )
        if self._sim == "gazebo":
            self._pub.publish(AckermannDriveStamped(drive=drive))
        else:  # CARLA: un-stamped AckermannDrive
            self._pub.publish(drive)


def main(args=None) -> None:
    """Entry point for the ``orad_vehicle_control`` console script."""
    if not _HAS_RCLPY:
        raise SystemExit("rclpy not installed; run in a ROS 2 environment.")
    rclpy.init(args=args)
    node = VehicleControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
