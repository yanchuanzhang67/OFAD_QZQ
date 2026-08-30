"""Smoke tests for the ONNX export (requires torch; onnx optional)."""
import os
import sys

import pytest

pytestmark = pytest.mark.unit

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

torch = pytest.importorskip("torch")  # noqa: E402

from deployment.onnx_export import (  # noqa: E402
    export_perception_onnx, export_policy_onnx)
from perception.bev_fusion import BEVFusion, BEVFusionConfig  # noqa: E402
from policy.hybrid_policy import HybridPolicy, HybridPolicyConfig  # noqa: E402


def test_export_perception_onnx(tmp_path):
    cfg = BEVFusionConfig(num_cameras=3, image_size=(64, 64))
    model = BEVFusion(cfg).eval()
    out = tmp_path / "bev_fusion.onnx"
    export_perception_onnx(str(out), model, torch.device("cpu"), cfg)
    assert out.exists() and out.stat().st_size > 0


def test_export_policy_onnx(tmp_path):
    cfg = HybridPolicyConfig()
    policy = HybridPolicy(cfg).eval()
    out = tmp_path / "policy.onnx"
    export_policy_onnx(str(out), policy, torch.device("cpu"), cfg)
    assert out.exists() and out.stat().st_size > 0
