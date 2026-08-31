"""ONNX export of the ORAD perception + policy deploy paths (Phase 5).

Both graphs are static-shape (only the batch axis is dynamic) so they map onto
TensorRT optimization profiles. Requires ``torch``; the ``onnx`` package is used
only for shape/ops checking and is skipped if absent.

Run:
    python -m deployment.onnx_export --out-dir exports \
        [--perception-ckpt bev.pt] [--policy-ckpt policy.pt]

Outputs ``bev_fusion.onnx`` and ``policy.onnx`` in ``--out-dir``.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

from perception.bev_fusion import BEVFusion, BEVFusionConfig  # noqa: E402
from policy.hybrid_policy import HybridPolicy, HybridPolicyConfig  # noqa: E402

__all__ = ["export_perception_onnx", "export_policy_onnx"]


def _check(path: str) -> None:
    try:
        import onnx
        m = onnx.load(path)
        onnx.checker.check_model(m)
        ins = [(i.name, [d.dim_value or d.dim_param
                         for d in i.type.tensor_type.shape.dim])
               for i in m.graph.input]
        print(f"[ok]   {os.path.basename(path)} inputs={ins}")
    except ImportError as e:  # pragma: no cover - onnx optional
        print(f"[skip] {os.path.basename(path)}: onnx unavailable "
              f"({type(e).__name__})")


def export_perception_onnx(path: str, model: BEVFusion, device,
                           cfg: BEVFusionConfig) -> None:
    """Export the BEV fusion tensor path (images, points, imu, attitude -> bev).
    Delegates to :meth:`BEVFusion.export_onnx` (opset 17, dynamic batch).

    Export always runs on CPU for cross-platform portability; the caller's
    model is moved to CPU for the duration of the call."""
    model = model.cpu()
    device = torch.device("cpu")
    b = 1
    images = torch.rand(b, cfg.num_cameras, cfg.image_channels, *cfg.image_size,
                        device=device)
    points = torch.randn(b, cfg.num_points, 4, device=device)
    imu = torch.randn(b, cfg.imu_steps, cfg.imu_in_channels, device=device)
    att = (torch.eye(3, device=device).unsqueeze(0)
           .expand(b, -1, -1).contiguous())
    model.export_onnx(path, images, points, imu, att)
    _check(path)


def export_policy_onnx(path: str, policy: HybridPolicy, device,
                       cfg: HybridPolicyConfig) -> None:
    """Export the deploy path ``(bev, imu) -> trajectory (1, N, 4)``.

    ``eval()`` makes the RSSM posterior use its mean, so the exported deploy
    path is deterministic. This helper exports the forward for graph
    verification (opset 17, dynamic batch). Export runs on CPU (the caller's
    policy is moved to CPU for the call).
    """
    policy = policy.cpu()
    device = torch.device("cpu")
    b = 1

    class _Wrapper(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p

        def forward(self, bev, imu):
            return self.p(bev, imu)

    wrapper = _Wrapper(policy).eval()
    bev = torch.randn(b, cfg.bev_channels, cfg.bev_h, cfg.bev_w, device=device)
    imu = torch.randn(b, cfg.imu_steps, cfg.imu_in_channels, device=device)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (bev, imu), path,
            input_names=["bev", "imu"],
            output_names=["trajectory"],
            dynamic_axes={"bev": {0: "batch"}, "imu": {0: "batch"},
                          "trajectory": {0: "batch"}},
            opset_version=17,
        )
    _check(path)


def _load(model: torch.nn.Module, ckpt: str, device) -> None:
    if not ckpt:
        return
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd, strict=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ORAD ONNX export")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--perception-ckpt", default=None)
    ap.add_argument("--policy-ckpt", default=None)
    ap.add_argument("--num-cameras", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--device", default="cuda"
                    if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    bcfg = BEVFusionConfig(num_cameras=args.num_cameras,
                           image_size=(args.image_size, args.image_size))
    bev_model = BEVFusion(bcfg).to(device).eval()
    _load(bev_model, args.perception_ckpt, device)
    pcfg = HybridPolicyConfig()
    policy = HybridPolicy(pcfg).to(device).eval()
    _load(policy, args.policy_ckpt, device)

    export_perception_onnx(os.path.join(args.out_dir, "bev_fusion.onnx"),
                           bev_model, device, bcfg)
    export_policy_onnx(os.path.join(args.out_dir, "policy.onnx"),
                       policy, device, pcfg)
    print(f"exported to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
