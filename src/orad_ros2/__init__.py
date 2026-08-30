"""ROS 2 integration: trajectory subscriber → control command publisher.

* :class:`orad_ros2.vehicle_control_node.PurePursuitController` - numpy-only
  geometric controller (unit-testable without rclpy)
* :class:`orad_ros2.vehicle_control_node.VehicleControlNode` - rclpy node wiring
  the safe-trajectory topic to the Ackermann command topic (CARLA / Gazebo)
"""
