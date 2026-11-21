"""Latent statistics calibration: compute_latent_stats + fluxflow-calibrate-latents CLI."""

import pytest
import safetensors.torch
import torch
import torch.nn as nn
from fluxflow.models.latent_stats import apply_latent_norm
from fluxflow.models.pipeline import FluxPipeline
from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

from fluxflow_training.scripts import calibrate_latents
from fluxflow_training.scripts.calibrate_latents import compute_latent_stats

_MEAN = torch.tensor([0.0, 1.0, -1.0, 2.0])
_STD = torch.tensor([1.0, 2.0, 0.5, 5.0])


class _FakeComp(torch.nn.Module):
    """Returns packed [B, T+1, 4] with known per-channel stats; identity latent stats."""

    def __init__(self):
        super().__init__()
        self.register_buffer("latent_shift", torch.zeros(4))
        self.register_buffer("latent_scale", torch.ones(4))
        self.calls = 0
        self.seen_shapes: list = []

    def forward(self, imgs, sample=None):
        self.calls += 1
        self.seen_shapes.append(tuple(imgs.shape))
        b = imgs.shape[0]
        tok = torch.randn(b, 64, 4) * _STD + _MEAN
        hw = torch.full((b, 1, 4), 1000.0)  # must be excluded from the stats
        packed = torch.cat([tok, hw], dim=1)
        return apply_latent_norm(packed, self.latent_shift, self.latent_scale)


def test_compute_latent_stats_recovers_mean_std():
    torch.manual_seed(0)
    mean, std = compute_latent_stats(
        _FakeComp(), (torch.zeros(8, 3, 16, 16) for _ in range(50)), 50
    )
    assert torch.allclose(mean, _MEAN, atol=0.1)
    assert torch.allclose(std, _STD, rtol=0.05)


def test_compute_latent_stats_returns_raw_stats_and_is_stateless():
    """Output is already normalized with the current stats; result must be the RAW stats."""
    torch.manual_seed(0)
    comp = _FakeComp()
    comp.latent_shift.copy_(torch.tensor([5.0, -3.0, 0.5, 1.0]))
    comp.latent_scale.copy_(torch.tensor([0.5, 3.0, 2.0, 0.25]))
    before = (comp.latent_shift.clone(), comp.latent_scale.clone())
    mean, std = compute_latent_stats(comp, (torch.zeros(8, 3, 16, 16) for _ in range(50)), 50)
    assert torch.allclose(mean, _MEAN, atol=0.1)
    assert torch.allclose(std, _STD, rtol=0.05)
    assert torch.equal(comp.latent_shift, before[0])
    assert torch.equal(comp.latent_scale, before[1])


def test_compute_latent_stats_feeds_every_resolution_and_honours_max_batches():
    torch.manual_seed(0)
    comp = _FakeComp()
    batches = [[torch.zeros(2, 3, 16, 16), torch.zeros(2, 3, 32, 32)] for _ in range(10)]
    compute_latent_stats(comp, iter([None] + batches), 3)
    assert comp.calls == 6  # 3 batches x 2 resolutions; None (fast-forward) skipped
    assert comp.seen_shapes[:2] == [(2, 3, 16, 16), (2, 3, 32, 32)]


def test_compute_latent_stats_restores_mode_and_checkpointing():
    comp = _FakeComp()
    comp.use_gradient_checkpointing = True
    comp.train()
    seen = {}
    orig = comp.forward

    def _fwd(imgs, sample=None):
        seen["training"] = comp.training
        seen["ckpt"] = comp.use_gradient_checkpointing
        seen["grad"] = torch.is_grad_enabled()
        return orig(imgs, sample=sample)

    comp.forward = _fwd  # type: ignore[method-assign]
    compute_latent_stats(comp, [torch.zeros(2, 3, 8, 8)], 1)
    assert seen == {"training": False, "ckpt": False, "grad": False}
    assert comp.training is True
    assert comp.use_gradient_checkpointing is True


def test_compute_latent_stats_requires_data():
    with pytest.raises(ValueError, match="no batches"):
        compute_latent_stats(_FakeComp(), [], 5)


def test_compute_latent_stats_real_v100_compressor():
    torch.manual_seed(0)
    comp = FluxCompressor_v100(d_model=8, downscales=2, use_gradient_checkpointing=False)
    mean, std = compute_latent_stats(comp, [torch.rand(2, 3, 32, 32) * 2 - 1 for _ in range(2)], 2)
    assert mean.shape == std.shape == comp.latent_shift.shape
    assert torch.isfinite(mean).all() and (std > 0).all()


def test_format_table_lists_every_channel():
    table = calibrate_latents.format_latent_stats_table(_MEAN, _STD, context_dims=1)
    lines = table.splitlines()
    assert sum(1 for line in lines if line.strip().startswith(("0 ", "1 ", "2 ", "3 "))) == 4
    assert "ctx" in table and "z" in table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _tiny_models() -> dict:
    comp = FluxCompressor_v100(d_model=8, downscales=2, use_gradient_checkpointing=False)
    exp = FluxExpander_v100(d_model=8, upscales=2, use_gradient_checkpointing=False)
    flow = nn.Linear(2, 2)
    return {
        "diffuser": FluxPipeline(comp, flow, exp),
        "compressor": comp,
        "expander": exp,
        "flow_processor": flow,
        "text_encoder": nn.Linear(2, 2),
        "model_config": None,
    }


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        calibrate_latents.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--checkpoint", "--config", "--out", "--num-images", "--batch-size", "--force"):
        assert flag in out


def test_cli_refuses_to_overwrite_checkpoint_without_force(tmp_path):
    ckpt = tmp_path / "flxflow_final.safetensors"
    ckpt.write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: {}\n")
    with pytest.raises(SystemExit) as exc:
        calibrate_latents.main(
            ["--checkpoint", str(ckpt), "--config", str(cfg), "--out", str(ckpt)]
        )
    assert exc.value.code != 0
    assert ckpt.read_bytes() == b"x"


def test_cli_end_to_end_writes_calibrated_checkpoint(tmp_path, monkeypatch, capsys):
    torch.manual_seed(0)
    models = _tiny_models()
    batches = [([torch.rand(2, 3, 32, 32) * 2 - 1], torch.zeros(2, 4)) for _ in range(5)]
    captured = {}

    def _fake_build(cli_args):
        captured["batch_size"] = cli_args.batch_size
        return models, batches, None

    monkeypatch.setattr(calibrate_latents, "_build_models_and_data", _fake_build)
    ckpt = tmp_path / "in.safetensors"
    ckpt.write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: {}\n")
    out = tmp_path / "sub" / "calibrated.safetensors"
    calibrate_latents.main(
        [
            "--checkpoint",
            str(ckpt),
            "--config",
            str(cfg),
            "--out",
            str(out),
            "--num-images",
            "6",
            "--batch-size",
            "2",
        ]
    )
    assert captured["batch_size"] == 2
    assert models["diffuser"].latent_stats_calibrated
    sd = safetensors.torch.load_file(str(out))
    for mod in ("compressor", "expander"):
        assert torch.equal(sd[f"diffuser.{mod}.latent_shift"], models[mod].latent_shift.cpu()), mod
        assert torch.equal(sd[f"diffuser.{mod}.latent_scale"], models[mod].latent_scale.cpu())
    # Same key layout as the training export (diffuser.* + bundled text_encoder.*)
    assert any(k.startswith("diffuser.flow_processor.") for k in sd)
    assert any(k.startswith("text_encoder.") for k in sd)
    assert not (out.parent / "text_encoder.safetensors").exists()
    printed = capsys.readouterr().out
    assert "mean" in printed and "std" in printed  # per-channel table


def test_cli_rejects_legacy_compressor(tmp_path, monkeypatch):
    models = _tiny_models()
    legacy = nn.Linear(2, 2)
    models["compressor"] = legacy
    monkeypatch.setattr(calibrate_latents, "_build_models_and_data", lambda a: (models, [], None))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: {}\n")
    with pytest.raises(SystemExit) as exc:
        calibrate_latents.main(
            [
                "--checkpoint",
                str(tmp_path / "a.safetensors"),
                "--config",
                str(cfg),
                "--out",
                str(tmp_path / "b.safetensors"),
            ]
        )
    assert exc.value.code != 0
