"""ament_python build glue for the ``orad_vehicle_control`` ROS 2 package.

The control node lives in ``src/orad_ros2/vehicle_control_node.py`` (importable as
``orad_ros2.vehicle_control_node`` thanks to ``src/`` being on ``sys.path`` via the
root ``conftest.py``). This setup script additionally makes it installable /
``colcon build``-able so the node entry point ``orad_vehicle_control`` resolves
inside a ROS 2 workspace.
"""
from pathlib import Path

from setuptools import find_packages, setup

pkg_root = Path(__file__).resolve().parent
long_description = (pkg_root.parent.parent / "README.md").read_text(
    encoding="utf-8"
) if (pkg_root.parent.parent / "README.md").exists() else "ORAD vehicle control"

setup(
    name="orad_vehicle_control",
    version="0.1.0",
    package_dir={"": str(pkg_root.parent)},
    packages=find_packages(where=str(pkg_root.parent)),
    install_requires=["numpy>=1.20"],
    entry_points={
        "console_scripts": [
            "orad_vehicle_control = orad_ros2.vehicle_control_node:main",
        ],
    },
    zip_safe=True,
    maintainer="ORAD",
    maintainer_email="orad@example.com",
    description="ORAD Phase 4 vehicle control node",
    license="Apache-2.0",
    long_description=long_description,
)
