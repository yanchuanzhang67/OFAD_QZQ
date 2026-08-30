"""ORAD shared utilities.

* :mod:`utils.types`      - framework-agnostic dataclasses (VehicleState, Trajectory, ...)
* :mod:`utils.transforms`  - numpy coordinate transforms & LiDAR→BEV projection geometry
* :mod:`utils.config`       - YAML config loaders (added later)
"""
from utils.types import (  # noqa: F401
    BEVFeature,
    OccupancyGrid,
    SensorPacket,
    Trajectory,
    VehicleState,
    Waypoint,
)
