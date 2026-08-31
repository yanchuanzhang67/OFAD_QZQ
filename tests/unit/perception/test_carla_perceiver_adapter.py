"""Contracts for the CARLA numpy-to-BEVFusion adapter."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.evaluate_carla_closed_loop import build_perceiver  # noqa: E402

pytestmark = pytest.mark.unit


def test_carla_perceiver_passes_strict_camera_lidar_mask():
    class _RecordingModel:
        def __init__(self):
            self.args = None
            self.kwargs = None

        def __call__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return "bev-output"

    model = _RecordingModel()
    perceive = build_perceiver(model, torch.device("cpu"))

    output = perceive(
        [np.zeros((4, 4, 3), dtype=np.uint8)],
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((2, 6), dtype=np.float32),
        np.eye(3, dtype=np.float32),
    )

    mask = model.kwargs["modality_mask"]
    assert output == "bev-output"
    assert mask.shape == (1, 2)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor([[True, True]]))
