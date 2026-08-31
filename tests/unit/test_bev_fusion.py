"""Unit tests for the multi-modal BEV fusion module (TDD, Phase 2).

Covers the three required test points:
  1. RGB (B, N, C, H, W) + LiDAR (B, N_points, 4) -> BEV (B, C_bev, H_bev, W_bev).
  2. IMU attitude (rotation) matrix -> correct body->world coordinate transform.
  3. Batch size > 1 tensor computation + GPU memory alignment.

Self-contained: does not rely on shared conftest fixtures.
"""
import pytest

torch = pytest.importorskip("torch")  # skip the whole module if torch missing

from perception.bev_fusion import BEVFusion, BEVFusionConfig  # noqa: E402

pytestmark = pytest.mark.unit


def _cfg(**kw):
    base = dict(
        num_cameras=2,
        image_size=(64, 64),
        num_points=256,
        bev_x_range=(-12.5, 12.5),
        bev_y_range=(-12.5, 12.5),
        bev_resolution=0.5,
        bev_channels=32,
    )
    base.update(kw)
    return BEVFusionConfig(**base)


def _inputs(cfg, B=1, device="cpu"):
    H, W = cfg.image_size
    images = torch.rand(B, cfg.num_cameras, cfg.image_channels, H, W, device=device)
    pts = torch.rand(B, cfg.num_points, 4, device=device)
    pts[..., 0] = pts[..., 0] * 20 - 10            # x in [-10, 10]
    pts[..., 1] = pts[..., 1] * 20 - 10            # y in [-10, 10]
    pts[..., 2] = torch.rand(B, cfg.num_points, device=device) * 3 - 1.5  # z
    imu = torch.randn(B, cfg.imu_steps, cfg.imu_in_channels, device=device)
    attitude = torch.eye(3, device=device).repeat(B, 1, 1)   # identity
    return images, pts, imu, attitude


# --- Test point 1: output BEV shape --------------------------------------
def test_bev_output_shape():
    cfg = _cfg()
    model = BEVFusion(cfg).eval()
    with torch.no_grad():
        for layer in model.pillar_mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(0.1)
                layer.bias.fill_(0.1)
    imgs, pts, imu, R = _inputs(cfg, B=1)
    with torch.no_grad():
        out = model(imgs, pts, imu, imu_attitude=R)
    Hb = int(round((cfg.bev_x_range[1] - cfg.bev_x_range[0]) / cfg.bev_resolution))
    Wb = int(round((cfg.bev_y_range[1] - cfg.bev_y_range[0]) / cfg.bev_resolution))
    assert out.bev.shape == (1, cfg.bev_channels, Hb, Wb)
    assert out.bev.dtype == torch.float32


# --- Test point 2: IMU attitude coordinate transform --------------------
def test_imu_attitude_coordinate_transform():
    cfg = _cfg()
    model = BEVFusion(cfg).eval()
    # 90-deg yaw rotation: R @ (1,0,0) == (0,1,0)
    R = torch.tensor([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]]).unsqueeze(0)   # (1, 3, 3)
    points = torch.tensor([[[1.0, 0.0, 0.0, 0.5]]])     # (1, 1, 4) [x,y,z,intensity]
    world = model.transform_points_to_world(points, R)
    assert world.shape == (1, 1, 4)
    assert torch.allclose(world[0, 0, :3], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)
    # intensity channel must be preserved
    assert torch.allclose(world[0, 0, 3], torch.tensor(0.5))


# --- Test point 3a: batch > 1 on CPU -------------------------------------
def test_batch_greater_than_one_cpu():
    cfg = _cfg()
    model = BEVFusion(cfg).eval()
    B = 4
    imgs, pts, imu, R = _inputs(cfg, B=B)
    with torch.no_grad():
        out = model(imgs, pts, imu, imu_attitude=R)
    assert out.bev.shape[0] == B
    assert out.bev.is_contiguous()


# --- Test point 3b: batch > 1 on GPU + memory alignment ------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_batch_greater_than_one_gpu_alignment():
    cfg = _cfg()
    model = BEVFusion(cfg).cuda().eval()
    B = 4
    imgs, pts, imu, R = _inputs(cfg, B=B, device="cuda")
    with torch.no_grad():
        out = model(imgs, pts, imu, imu_attitude=R)
    assert out.bev.shape[0] == B
    assert out.bev.is_cuda
    # contiguous <=> no inter-batch memory padding (alignment check)
    assert out.bev.is_contiguous()
    assert out.bev.stride(0) == out.bev.shape[1] * out.bev.shape[2] * out.bev.shape[3]


# --- BUG-1 regression: asymmetric x/y range dimension mapping -------------
def test_asymmetric_bev_range_shape():
    """H must come from y_range, W from x_range (image (C,H,W) convention).

    Symmetric ranges (the original tests) hide the H/W <-> x/y mapping; this
    uses an asymmetric range so a swapped convention would change the shape.
    """
    cfg = _cfg(bev_x_range=(-10.0, 10.0), bev_y_range=(-5.0, 5.0))
    model = BEVFusion(cfg).eval()
    imgs, pts, imu, R = _inputs(cfg, B=1)
    with torch.no_grad():
        out = model(imgs, pts, imu, imu_attitude=R)
    Hb = int(round((cfg.bev_y_range[1] - cfg.bev_y_range[0]) / cfg.bev_resolution))  # y->H = 20
    Wb = int(round((cfg.bev_x_range[1] - cfg.bev_x_range[0]) / cfg.bev_resolution))  # x->W = 40
    assert out.bev.shape == (1, cfg.bev_channels, Hb, Wb)          # (1, 32, 20, 40)


def test_asymmetric_bev_range_point_placement():
    """A single point at (x=8, y=0) must land in cell (row=10, col=36),
    proving col<-x / row<-y mapping and no erroneous clamp (BUG-1)."""
    cfg = _cfg(num_points=1, bev_x_range=(-10.0, 10.0), bev_y_range=(-5.0, 5.0),
               cam_feat_channels=4, pillar_feat_channels=4, bev_channels=4)
    model = BEVFusion(cfg).eval()
    pts = torch.tensor([[[8.0, 0.0, 0.0, 1.0]]])                   # (1, 1, 4)
    R = torch.eye(3).unsqueeze(0)
    with torch.no_grad():
        for layer in model.pillar_mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(0.1)
                layer.bias.fill_(0.1)
        bev = model.encode_lidar(pts, R)                          # (1, Cl, H, W)
    # x: 20m / 0.5 = 40 cols -> col = (8 - (-10)) / 0.5 = 36
    # y: 10m / 0.5 = 20 rows -> row = (0 - (-5))  / 0.5 = 10
    assert bev.shape == (1, cfg.pillar_feat_channels, 20, 40)
    nz = bev[0].abs().sum(0).nonzero(as_tuple=False)              # (K, 2) [row, col]
    assert nz.shape[0] == 1
    assert tuple(nz[0].tolist()) == (10, 36)


@pytest.mark.parametrize("n_points", [1, 7, 300])
def test_point_count_boundaries_and_occupancy(n_points):
    cfg = BEVFusionConfig(num_points=32)
    model = BEVFusion(cfg).eval()
    images = torch.zeros(1, cfg.num_cameras, cfg.image_channels, *cfg.image_size)
    points = torch.zeros(1, n_points, cfg.lidar_in_channels)
    imu = torch.zeros(1, cfg.imu_steps, cfg.imu_in_channels)
    with torch.no_grad():
        out = model(images, points, imu)
    assert out.bev.shape == (1, cfg.bev_channels, model.bev_h, model.bev_w)
    assert isinstance(out.occupancy, torch.Tensor)
    assert out.occupancy.shape == (1, 1, model.bev_h, model.bev_w)
    assert torch.isfinite(out.occupancy).all()
    assert torch.all((out.occupancy >= 0) & (out.occupancy <= 1))


def test_empty_point_cloud_is_rejected_as_invalid_lidar():
    cfg = BEVFusionConfig(num_points=32)
    model = BEVFusion(cfg).eval()
    images = torch.zeros(
        1, cfg.num_cameras, cfg.image_channels, *cfg.image_size)
    points = torch.empty(1, 0, cfg.lidar_in_channels)
    imu = torch.zeros(1, cfg.imu_steps, cfg.imu_in_channels)

    with pytest.raises(ValueError, match="points must contain at least one"):
        model(images, points, imu)
