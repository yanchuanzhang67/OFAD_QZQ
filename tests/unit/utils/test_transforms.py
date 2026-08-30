"""Unit tests for utils.transforms (SDD: LiDAR->BEV projection dimension & alignment).

Self-contained: does not rely on shared conftest fixtures.
"""
import numpy as np
import pytest

from utils.transforms import point_cloud_to_bev_tensor, world_to_bev_index
from utils.types import OccupancyGrid

pytestmark = pytest.mark.unit


def test_bev_tensor_shape_dtype_contiguous():
    rng = np.random.default_rng(42)
    pts = rng.uniform(-40, 40, (1000, 4)).astype(np.float32)
    pts[:, 2] = rng.uniform(-2, 2, 1000)
    bev = point_cloud_to_bev_tensor(pts, resolution=0.5)
    assert bev.shape == (4, 200, 200)
    assert bev.dtype == np.float32
    assert bev.flags["C_CONTIGUOUS"]


def test_known_point_lands_in_expected_cell():
    pts = np.array([[0.0, 0.0, 1.0, 0.5]], dtype=np.float32)
    bev = point_cloud_to_bev_tensor(pts, x_range=(-50.0, 50.0),
                                    y_range=(-50.0, 50.0), resolution=1.0)
    assert bev[2, 50, 50] > 0.0
    assert bev[2, 0, 0] == 0.0


def test_world_to_bev_index_validity():
    row, col, valid, shape = world_to_bev_index(
        np.array([[0.0, 0.0], [999.0, 999.0]], np.float32),
        x_range=(-50, 50), y_range=(-50, 50), resolution=1.0,
    )
    assert valid[0] and not valid[1]
    assert shape == (100, 100)


def test_occupancy_grid_negative_boundary_is_outside():
    grid = OccupancyGrid(np.zeros((2, 2), np.float32), resolution=1.0,
                         origin=(0.0, 0.0, 0.0))
    assert grid.is_occupied(-0.01, 0.5)


def test_occupancy_grid_origin_yaw_is_applied():
    data = np.zeros((4, 4), np.float32)
    data[0, 2] = 1.0
    grid = OccupancyGrid(data, resolution=1.0,
                         origin=(10.0, 5.0, np.pi / 2))
    assert grid.is_occupied(10.0, 7.0)


@pytest.mark.parametrize("resolution", [0.0, -1.0, np.nan])
def test_occupancy_grid_rejects_invalid_resolution(resolution):
    with pytest.raises(ValueError, match="resolution"):
        OccupancyGrid(np.zeros((2, 2), np.float32), resolution=resolution)


def test_occupancy_grid_rejects_non_2d_and_non_finite_data():
    with pytest.raises(ValueError, match="2D"):
        OccupancyGrid(np.zeros((1, 2, 2), np.float32))
    with pytest.raises(ValueError, match="finite"):
        OccupancyGrid(np.array([[np.nan]], np.float32))
