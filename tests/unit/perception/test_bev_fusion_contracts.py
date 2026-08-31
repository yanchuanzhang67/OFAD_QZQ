"""BEV fusion contracts derived from the four-project architecture review."""

import pytest

torch = pytest.importorskip("torch")

from perception.bev_fusion import BEVFusion, BEVFusionConfig  # noqa: E402
from perception.encoders import ImageEncoder, ImuEncoder  # noqa: E402
from perception.fusion import ModalityAwareBEVFuser  # noqa: E402

pytestmark = pytest.mark.unit


def _config():
    return BEVFusionConfig(
        num_cameras=2, image_size=(32, 32), num_points=8,
        cam_feat_channels=4, pillar_feat_channels=4, bev_channels=8,
        bev_x_range=(-2.0, 2.0), bev_y_range=(-2.0, 2.0),
        bev_resolution=1.0, imu_hidden=8, imu_steps=3,
    )


def _inputs(config):
    images = torch.rand(
        1, config.num_cameras, config.image_channels, *config.image_size)
    points = torch.tensor([[[0.5, 0.5, 0.0, 1.0]]], dtype=torch.float32)
    imu = torch.zeros(1, config.imu_steps, config.imu_in_channels)
    return images, points, imu


def test_perception_encoders_are_explicit_reusable_modules():
    image = ImageEncoder(3, 4, 2)
    imu = ImuEncoder(6, 8, 4)

    feature, depth = image(torch.zeros(2, 3, 16, 16))
    imu_feature = imu(torch.zeros(2, 3, 6))

    assert feature.shape == (2, 4, 4, 4)
    assert depth.shape == (2, 2, 4, 4)
    assert torch.allclose(depth.sum(dim=1), torch.ones(2, 4, 4))
    assert imu_feature.shape == (2, 4)


def test_modality_mask_accepts_only_dual_modality_availability():
    fuser = ModalityAwareBEVFuser(4, 4, 8).eval()
    camera = torch.rand(2, 4, 3, 5)
    lidar = torch.rand(2, 4, 3, 5)
    availability = torch.ones(2, 2, dtype=torch.bool)

    masked = fuser(camera, lidar, availability)
    unmasked = fuser(camera, lidar)

    assert masked.shape == (2, 8, 3, 5)
    assert torch.allclose(masked, unmasked, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("availability,match", [
    (torch.ones(1, 3, dtype=torch.bool), "shape"),
    (torch.tensor([[True, False]]), "both camera and lidar"),
    (torch.tensor([[False, True]]), "both camera and lidar"),
    (torch.zeros(1, 2, dtype=torch.bool), "both camera and lidar"),
    (torch.tensor([[1.0, float("nan")]]), "finite"),
    (torch.tensor([[1.0, 0.5]]), "boolean"),
])
def test_modality_mask_rejects_ambiguous_or_unsafe_values(availability, match):
    fuser = ModalityAwareBEVFuser(4, 4, 8)
    feature = torch.zeros(1, 4, 3, 3)

    with pytest.raises(ValueError, match=match):
        fuser(feature, feature, availability)


@pytest.mark.parametrize("camera_shape,lidar_shape,match", [
    ((1, 4, 3), (1, 4, 3, 3), "4-D"),
    ((1, 4, 3, 3), (2, 4, 3, 3), "batch/spatial"),
    ((1, 3, 3, 3), (1, 4, 3, 3), "camera BEV channel"),
    ((1, 4, 3, 3), (1, 3, 3, 3), "lidar BEV channel"),
])
def test_fuser_rejects_feature_contract_drift(
        camera_shape, lidar_shape, match):
    fuser = ModalityAwareBEVFuser(4, 4, 8)

    with pytest.raises(ValueError, match=match):
        fuser(torch.zeros(camera_shape), torch.zeros(lidar_shape))


def test_bev_model_rejects_camera_only_and_lidar_only_paths():
    config = _config()
    model = BEVFusion(config).eval()
    keys = set(model.state_dict())
    assert "image_encoder.backbone.0.weight" in keys
    assert "imu_encoder.gru.weight_ih_l0" in keys
    assert "fuser.0.weight" in keys
    images, points, imu = _inputs(config)

    for mask in (
            torch.tensor([[True, False]]),
            torch.tensor([[False, True]])):
        with pytest.raises(ValueError, match="both camera and lidar"):
            model(images, points, imu, modality_mask=mask)


def test_invalid_modality_mask_is_rejected_before_sensor_encoding(
        monkeypatch):
    config = _config()
    model = BEVFusion(config).eval()
    images, points, imu = _inputs(config)

    def unexpected_encode(*args, **kwargs):
        raise AssertionError("sensor encoder must not run for an invalid mask")

    monkeypatch.setattr(model, "encode_lidar", unexpected_encode)
    monkeypatch.setattr(model, "encode_images", unexpected_encode)

    with pytest.raises(ValueError, match="both camera and lidar"):
        model(
            images, points, imu,
            modality_mask=torch.tensor([[True, False]]))


@pytest.mark.parametrize("field,bad_value", [
    ("images", float("nan")),
    ("images", float("inf")),
    ("points", float("nan")),
    ("points", float("inf")),
])
def test_bev_model_rejects_nonfinite_camera_or_lidar_data(field, bad_value):
    config = _config()
    model = BEVFusion(config).eval()
    images, points, imu = _inputs(config)
    if field == "images":
        images[0, 0, 0, 0, 0] = bad_value
    else:
        points[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match=f"{field} must contain only finite"):
        model(images, points, imu)


def test_bev_model_rejects_empty_lidar_point_cloud():
    config = _config()
    model = BEVFusion(config).eval()
    images, _, imu = _inputs(config)
    points = torch.empty(1, 0, config.lidar_in_channels)

    with pytest.raises(ValueError, match="points must contain at least one"):
        model(images, points, imu)


@pytest.mark.parametrize("field", ["images", "points", "imu"])
def test_bev_model_rejects_static_contract_shape_drift(field):
    config = _config()
    model = BEVFusion(config).eval()
    images, points, imu = _inputs(config)
    if field == "images":
        images = images[:, :1]
    elif field == "points":
        points = points[..., :3]
    else:
        imu = imu[:, :-1]

    with pytest.raises(ValueError, match=field):
        model(images, points, imu)


def test_bev_model_rejects_batch_and_attitude_shape_drift():
    config = _config()
    model = BEVFusion(config).eval()
    images, points, imu = _inputs(config)

    with pytest.raises(ValueError, match="batch sizes"):
        model(images.repeat(2, 1, 1, 1, 1), points, imu)
    with pytest.raises(ValueError, match="imu_attitude"):
        model(images, points, imu, imu_attitude=torch.eye(4).unsqueeze(0))
