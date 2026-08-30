import numpy as np
import pytest

from sim.domain_randomization import (
    DomainRandomizationConfig, augment_lidar, sample_domain)

pytestmark = pytest.mark.unit


def test_sample_domain_is_seeded_and_within_ranges():
    cfg = DomainRandomizationConfig()
    first = sample_domain(cfg, np.random.default_rng(7))
    second = sample_domain(cfg, np.random.default_rng(7))
    assert first == second
    for name, value in first.items():
        low, high = getattr(cfg, name)
        assert low <= value <= high


def test_lidar_augmentation_preserves_input_and_channels():
    points = np.ones((20, 4), np.float32)
    original = points.copy()
    out = augment_lidar(points, sigma=0.05, dropout=0.25,
                        rng=np.random.default_rng(3))
    assert np.array_equal(points, original)
    assert out.ndim == 2 and out.shape[1] == 4
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]
    assert np.all(out[:, 3] == 1.0)


@pytest.mark.parametrize("sigma,dropout", [(-0.1, 0.0), (0.1, -0.1), (0.1, 1.1)])
def test_lidar_augmentation_rejects_invalid_noise(sigma, dropout):
    with pytest.raises(ValueError):
        augment_lidar(np.zeros((1, 4), np.float32), sigma, dropout)
