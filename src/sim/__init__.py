"""Simulation adapters (CARLA / Gazebo-ROS 2).

Modules:

* ``carla_closed_loop`` - CARLA closed-loop eval helpers + synchronous sensor
  stack (lazy ``carla`` import; numpy-only helpers are unit-tested in
  ``tests/unit/sim``). Run via ``scripts/evaluate_carla_closed_loop.py``.

Planned:

* ``gazebo_interface`` - multi-link suspension dynamics, LiDAR / IMU noise
"""
