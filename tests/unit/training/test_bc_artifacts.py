"""Transactional run and strict checkpoint tests for B0 BC."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from perception.bev_fusion import BEVFusion, BEVFusionConfig  # noqa: E402
from policy.bc_policy import BCPolicy  # noqa: E402
from policy.config import BCPolicyConfig  # noqa: E402
from training.bc_artifacts import (  # noqa: E402
    BCRunWriter,
    load_bc_checkpoint,
    load_frozen_perception_checkpoint,
    resolve_checkpoint_index,
    save_bc_checkpoint,
    write_checkpoint_index,
)

pytestmark = pytest.mark.unit


def _cfg(**overrides):
    values = {
        "bev_channels": 8,
        "bev_h": 10,
        "bev_w": 10,
        "imu_steps": 4,
        "encoder_hidden": 16,
        "ego_hidden": 8,
        "latent_dim": 16,
        "horizon": 5,
    }
    values.update(overrides)
    return BCPolicyConfig(**values)


def _lineage():
    return {
        "git": {"commit": "a" * 40, "dirty": False},
        "config_sha256": "1" * 64,
        "dataset_sha256": "2" * 64,
        "split_sha256": "3" * 64,
        "calibration_sha256": "4" * 64,
        "perception_checkpoint_sha256": "5" * 64,
        "seed": 41,
        "resolved_command": "python scripts/train_bc.py",
        "epoch": 1,
        "best_metric": 0.25,
    }


def test_bc_run_writer_is_exclusive_and_publishes_success_last(tmp_path):
    writer = BCRunWriter.create(tmp_path, "run-001", {"seed": 41})

    assert writer.working_path.parent.name == ".incomplete"
    assert writer.checkpoint_dir.is_dir()
    assert writer.log_dir.is_dir()
    assert not (writer.working_path / "_SUCCESS").exists()
    final = writer.complete({"ade_m": 0.2})

    assert final == tmp_path / "run-001"
    assert (final / "_SUCCESS").read_text(encoding="utf-8") == "complete\n"
    assert json.loads((final / "manifest.json").read_text()) == {"seed": 41}
    assert json.loads((final / "metrics.json").read_text()) == {"ade_m": 0.2}
    with pytest.raises(FileExistsError):
        BCRunWriter.create(tmp_path, "run-001", {"seed": 41})
    with pytest.raises(RuntimeError, match="working directory"):
        writer.complete({"ade_m": 0.1})
    with pytest.raises(ValueError, match="run_id"):
        BCRunWriter.create(tmp_path, "../escape", {"seed": 41})
    blocked = BCRunWriter.create(tmp_path, "run-002", {"seed": 41})
    blocked.final_path.mkdir()
    with pytest.raises(FileExistsError):
        blocked.complete({"ade_m": 0.1})


def test_checkpoint_roundtrip_requires_pure_bc_family(tmp_path):
    config = _cfg()
    policy = BCPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    path = tmp_path / "epoch-0001.pt"

    save_bc_checkpoint(path, policy, optimizer, _lineage())
    loaded, metadata = load_bc_checkpoint(path, config)

    assert loaded.state_dict().keys() == policy.state_dict().keys()
    for name, value in policy.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
    assert metadata["model_family"] == "pure_bc_v1"
    assert metadata["lineage"] == _lineage()
    with pytest.raises(FileExistsError):
        save_bc_checkpoint(path, policy, optimizer, _lineage())


def test_checkpoint_rejects_wrong_family_config_and_state(tmp_path):
    config = _cfg()
    policy = BCPolicy(config)
    source = tmp_path / "source.pt"
    save_bc_checkpoint(source, policy, None, _lineage())
    payload = torch.load(source, map_location="cpu", weights_only=True)

    wrong_family = dict(payload)
    wrong_family["model_family"] = "hybrid_legacy"
    torch.save(wrong_family, tmp_path / "wrong-family.pt")
    with pytest.raises(ValueError, match="model_family"):
        load_bc_checkpoint(tmp_path / "wrong-family.pt", config)

    with pytest.raises(ValueError, match="config"):
        load_bc_checkpoint(source, _cfg(latent_dim=32))

    missing_state = dict(payload)
    missing_state["state_dict"] = dict(payload["state_dict"])
    missing_state["state_dict"].pop("ego_mean")
    torch.save(missing_state, tmp_path / "missing-state.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_bc_checkpoint(tmp_path / "missing-state.pt", config)

    invalid_std = dict(payload)
    invalid_std["state_dict"] = dict(payload["state_dict"])
    invalid_std["state_dict"]["ego_std"] = torch.zeros(config.ego_dim)
    torch.save(invalid_std, tmp_path / "invalid-std.pt")
    with pytest.raises(ValueError, match="ego_std"):
        load_bc_checkpoint(tmp_path / "invalid-std.pt", config)

    invalid_shape = dict(payload)
    invalid_shape["state_dict"] = dict(payload["state_dict"])
    invalid_shape["state_dict"]["ego_std"] = torch.ones(config.ego_dim - 1)
    torch.save(invalid_shape, tmp_path / "invalid-std-shape.pt")
    with pytest.raises(ValueError, match="ego_std.*shape"):
        load_bc_checkpoint(tmp_path / "invalid-std-shape.pt", config)
    invalid_mean_shape = dict(payload)
    invalid_mean_shape["state_dict"] = dict(payload["state_dict"])
    invalid_mean_shape["state_dict"]["ego_mean"] = torch.zeros(
        config.ego_dim - 1)
    torch.save(invalid_mean_shape, tmp_path / "invalid-mean-shape.pt")
    with pytest.raises(ValueError, match="ego_mean.*shape"):
        load_bc_checkpoint(tmp_path / "invalid-mean-shape.pt", config)


def test_checkpoint_rejects_invalid_payload_lineage_and_tensors(tmp_path):
    config = _cfg()
    policy = BCPolicy(config)
    source = tmp_path / "source.pt"
    save_bc_checkpoint(source, policy, None, _lineage())
    payload = torch.load(source, map_location="cpu", weights_only=True)

    with pytest.raises(TypeError, match="policy must be BCPolicy"):
        save_bc_checkpoint(tmp_path / "bad-policy.pt", object(), None, _lineage())
    with pytest.raises(TypeError, match="lineage.*mapping"):
        save_bc_checkpoint(tmp_path / "bad-lineage.pt", policy, None, None)
    missing_lineage = _lineage()
    missing_lineage.pop("dataset_sha256")
    with pytest.raises(ValueError, match="missing keys"):
        save_bc_checkpoint(
            tmp_path / "missing-lineage.pt", policy, None, missing_lineage)
    invalid_hash = _lineage()
    invalid_hash["dataset_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="dataset_sha256"):
        save_bc_checkpoint(
            tmp_path / "invalid-hash.pt", policy, None, invalid_hash)
    invalid_hex = _lineage()
    invalid_hex["dataset_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="dataset_sha256"):
        save_bc_checkpoint(
            tmp_path / "invalid-hex.pt", policy, None, invalid_hex)
    invalid_cases = (
        ("git", None, "git metadata"),
        ("seed", -1, "seed"),
        ("resolved_command", "", "resolved_command"),
        ("best_metric", float("nan"), "best_metric"),
    )
    for name, value, message in invalid_cases:
        lineage = _lineage()
        lineage[name] = value
        with pytest.raises(ValueError, match=message):
            save_bc_checkpoint(
                tmp_path / f"invalid-{name}.pt", policy, None, lineage)
    nested_nonfinite = _lineage()
    nested_nonfinite["diagnostics"] = [float("nan")]
    with pytest.raises(ValueError, match="lineage.*non-finite"):
        save_bc_checkpoint(
            tmp_path / "nonfinite-lineage.pt", policy, None,
            nested_nonfinite)

    torch.save([], tmp_path / "not-mapping.pt")
    with pytest.raises(TypeError, match="payload.*mapping"):
        load_bc_checkpoint(tmp_path / "not-mapping.pt", config)
    wrong_schema = dict(payload)
    wrong_schema["schema_version"] = "old"
    torch.save(wrong_schema, tmp_path / "wrong-schema.pt")
    with pytest.raises(ValueError, match="schema_version"):
        load_bc_checkpoint(tmp_path / "wrong-schema.pt", config)
    no_lineage = dict(payload)
    no_lineage.pop("lineage")
    torch.save(no_lineage, tmp_path / "no-lineage.pt")
    with pytest.raises(ValueError, match="missing lineage"):
        load_bc_checkpoint(tmp_path / "no-lineage.pt", config)
    bad_state = dict(payload)
    bad_state["state_dict"] = {"parameter": "not-a-tensor"}
    torch.save(bad_state, tmp_path / "bad-state.pt")
    with pytest.raises(TypeError, match="named tensors"):
        load_bc_checkpoint(tmp_path / "bad-state.pt", config)
    absent_state = dict(payload)
    absent_state.pop("state_dict")
    torch.save(absent_state, tmp_path / "absent-state.pt")
    with pytest.raises(TypeError, match="state_dict.*mapping"):
        load_bc_checkpoint(tmp_path / "absent-state.pt", config)


def test_checkpoint_rejects_nonfinite_optimizer_state(tmp_path):
    policy = BCPolicy(_cfg())
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    parameter = next(policy.parameters())
    optimizer.state[parameter]["step"] = torch.tensor(float("nan"))

    with pytest.raises(ValueError, match="optimizer_state.*non-finite"):
        save_bc_checkpoint(
            tmp_path / "nonfinite-optimizer.pt",
            policy,
            optimizer,
            _lineage(),
        )


def test_checkpoint_index_resolves_only_matching_local_weight(tmp_path):
    checkpoint = tmp_path / "epoch-0001.pt"
    checkpoint.write_bytes(b"weights")
    index = tmp_path / "last.json"

    write_checkpoint_index(index, checkpoint, epoch=1)

    assert resolve_checkpoint_index(index) == checkpoint
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    index.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        resolve_checkpoint_index(index)


def test_checkpoint_index_rejects_unsafe_or_incomplete_references(tmp_path):
    checkpoint = tmp_path / "epoch.pt"
    checkpoint.write_bytes(b"weights")
    with pytest.raises(ValueError, match="epoch"):
        write_checkpoint_index(tmp_path / "bad-epoch.json", checkpoint, epoch=-1)
    with pytest.raises(FileNotFoundError):
        write_checkpoint_index(
            tmp_path / "missing.json", tmp_path / "missing.pt", epoch=1)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="local weight"):
        write_checkpoint_index(other / "last.json", checkpoint, epoch=1)

    index = tmp_path / "last.json"
    index.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        resolve_checkpoint_index(index)
    index.write_text(json.dumps({
        "filename": "../epoch.pt", "sha256": "1" * 64, "epoch": 1,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="basename"):
        resolve_checkpoint_index(index)
    index.write_text(json.dumps({
        "filename": "epoch.pt", "sha256": "short", "epoch": 1,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_checkpoint_index(index)
    index.write_text(json.dumps({
        "filename": "absent.pt", "sha256": "1" * 64, "epoch": 1,
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_index(index)


def test_frozen_perception_checkpoint_is_strict_finite_and_hashed(tmp_path):
    config = BEVFusionConfig(
        num_cameras=2, image_size=(16, 16), num_points=16,
        bev_channels=8, imu_steps=4)
    source = BEVFusion(config)
    path = tmp_path / "perception.pt"
    torch.save(source.state_dict(), path)
    target = BEVFusion(config)

    digest = load_frozen_perception_checkpoint(path, target)

    assert len(digest) == 64
    assert target.training is False
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert all(
        torch.equal(source.state_dict()[name], target.state_dict()[name])
        for name in source.state_dict())

    wrapped_path = tmp_path / "wrapped-perception.pt"
    torch.save({"state_dict": source.state_dict()}, wrapped_path)
    wrapped_target = BEVFusion(config)
    assert len(load_frozen_perception_checkpoint(
        wrapped_path, wrapped_target)) == 64


def test_frozen_perception_checkpoint_rejects_missing_and_nonfinite_state(tmp_path):
    config = BEVFusionConfig(
        num_cameras=2, image_size=(16, 16), num_points=16,
        bev_channels=8, imu_steps=4)
    model = BEVFusion(config)
    missing_path = tmp_path / "missing.pt"
    torch.save({}, missing_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_frozen_perception_checkpoint(missing_path, BEVFusion(config))

    state = model.state_dict()
    first_name = next(
        name for name, value in state.items() if value.is_floating_point())
    state[first_name] = torch.full_like(state[first_name], float("nan"))
    nonfinite_path = tmp_path / "nonfinite.pt"
    torch.save(state, nonfinite_path)
    with pytest.raises(ValueError, match="non-finite"):
        load_frozen_perception_checkpoint(nonfinite_path, BEVFusion(config))
