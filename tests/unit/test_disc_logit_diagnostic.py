"""Unit tests for discriminator patch-logit diagnostic snapshots.

Verifies the diagnostic is off by default, fires at correct intervals,
writes a JSONL record with all required fields, and creates a PNG heatmap.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.pipeline_config import PipelineStepConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_vae_trainer(
    disc_logit_diagnostic_interval: int = 0,
    diag_dir: Path | None = None,
):
    """Construct a VAETrainer with a mock discriminator and minimal real modules."""
    from fluxflow_training.training.utils import EMA
    from fluxflow_training.training.vae_trainer import VAETrainer

    compressor = nn.Sequential(nn.Linear(4, 4))
    expander = nn.Sequential(nn.Linear(4, 4))
    optimizer = torch.optim.SGD(
        list(compressor.parameters()) + list(expander.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    ema = EMA(nn.ModuleList([compressor, expander]))

    accel = MagicMock()
    accel.backward = lambda loss: loss.backward()
    accel.scaler = None
    accel.clip_grad_norm_ = MagicMock(return_value=1.0)

    # Mock discriminator that returns a known logit tensor
    disc = MagicMock(spec=nn.Module)
    disc.train = Mock()
    disc.eval = Mock()
    disc.named_parameters = Mock(return_value=iter([]))
    disc.parameters = Mock(return_value=iter([nn.Parameter(torch.zeros(1))]))
    # Returns [B=2, 1, H=15, W=15] — PatchGAN-style spatial logits
    disc.side_effect = lambda img, ctx: torch.zeros(img.shape[0], 1, 15, 15)
    disc.__call__ = disc.side_effect
    disc.ctx_dim = 0  # no context projection

    disc_opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1e-3)
    disc_sched = torch.optim.lr_scheduler.ConstantLR(disc_opt)

    trainer = VAETrainer(
        compressor=compressor,
        expander=expander,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        reconstruction_loss_fn=nn.L1Loss(),
        reconstruction_loss_min_fn=nn.MSELoss(),
        use_gan=True,
        discriminator=disc,
        discriminator_optimizer=disc_opt,
        discriminator_scheduler=disc_sched,
        use_lpips=False,
        ctx_input_dim=4,
        lambda_ctx_aux=0.01,
        train_ctx_aux=False,
        accelerator=accel,
        disc_logit_diagnostic_interval=disc_logit_diagnostic_interval,
        disc_logit_diagnostic_dir=str(diag_dir) if diag_dir is not None else None,
    )
    return trainer


# ---------------------------------------------------------------------------
# Tests: config field
# ---------------------------------------------------------------------------


class TestDiscLogitDiagnosticConfigField:
    """disc_logit_diagnostic_interval lives in PipelineStepConfig."""

    def test_default_is_zero(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.disc_logit_diagnostic_interval == 0

    def test_can_be_set(self):
        step = PipelineStepConfig(
            name="s", n_epochs=1, train_vae=True, disc_logit_diagnostic_interval=100
        )
        assert step.disc_logit_diagnostic_interval == 100

    def test_parsed_from_dict(self):
        from fluxflow_training.training.pipeline_config import parse_pipeline_config

        cfg = parse_pipeline_config(
            {
                "steps": [
                    {
                        "name": "gan_step",
                        "n_epochs": 1,
                        "gan_training": True,
                        "disc_logit_diagnostic_interval": 500,
                    }
                ]
            }
        )
        assert cfg.steps[0].disc_logit_diagnostic_interval == 500


# ---------------------------------------------------------------------------
# Tests: diagnostic stored on trainer
# ---------------------------------------------------------------------------


class TestDiscLogitDiagnosticTrainerInit:
    def test_interval_stored_default(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = _make_minimal_vae_trainer(0, Path(d))
            assert trainer.disc_logit_diagnostic_interval == 0

    def test_interval_stored_custom(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = _make_minimal_vae_trainer(1, Path(d))
            assert trainer.disc_logit_diagnostic_interval == 1

    def test_diag_dir_stored(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = _make_minimal_vae_trainer(1, Path(d))
            assert trainer.disc_logit_diagnostic_dir == Path(d)


# ---------------------------------------------------------------------------
# Tests: snapshot written correctly
# ---------------------------------------------------------------------------


class TestDiscLogitDiagnosticSnapshot:
    """JSONL record is written at the requested interval; PNG heatmap created."""

    def _run_fake_discriminator_step(self, trainer, global_step: int):
        """Call _maybe_save_disc_logit_snapshot directly with a known logit tensor."""
        logits = torch.zeros(2, 1, 15, 15)  # known: all-zero spatial map
        trainer._maybe_save_disc_logit_snapshot(logits, global_step)

    def test_jsonl_created_at_step_0(self):
        with tempfile.TemporaryDirectory() as d:
            diag_dir = Path(d)
            trainer = _make_minimal_vae_trainer(1, diag_dir)
            self._run_fake_discriminator_step(trainer, global_step=0)

            jsonl_path = diag_dir / "disc_logits.jsonl"
            assert jsonl_path.exists(), "JSONL not created after first snapshot"

    def test_jsonl_record_has_required_fields(self):
        with tempfile.TemporaryDirectory() as d:
            diag_dir = Path(d)
            trainer = _make_minimal_vae_trainer(1, diag_dir)
            self._run_fake_discriminator_step(trainer, global_step=100)

            jsonl_path = diag_dir / "disc_logits.jsonl"
            record = json.loads(jsonl_path.read_text().strip().splitlines()[0])

            for field in ("step", "shape", "mean", "std", "min", "max", "row_mean", "col_mean"):
                assert field in record, f"Missing field: {field}"

    def test_jsonl_values_match_known_logits(self):
        """All-zero logit map → mean=0, std=0, row_mean all zeros."""
        with tempfile.TemporaryDirectory() as d:
            diag_dir = Path(d)
            trainer = _make_minimal_vae_trainer(1, diag_dir)
            self._run_fake_discriminator_step(trainer, global_step=42)

            record = json.loads((diag_dir / "disc_logits.jsonl").read_text().strip())
            assert record["step"] == 42
            assert record["mean"] == pytest.approx(0.0, abs=1e-6)
            assert record["std"] == pytest.approx(0.0, abs=1e-6)
            assert record["shape"] == [15, 15]
            assert all(v == pytest.approx(0.0, abs=1e-6) for v in record["row_mean"])
            assert all(v == pytest.approx(0.0, abs=1e-6) for v in record["col_mean"])

    def test_png_heatmap_created(self):
        with tempfile.TemporaryDirectory() as d:
            diag_dir = Path(d)
            trainer = _make_minimal_vae_trainer(1, diag_dir)
            self._run_fake_discriminator_step(trainer, global_step=0)

            pngs = list(diag_dir.glob("*.png"))
            assert len(pngs) >= 1, "No PNG heatmap written"

    def test_no_snapshot_when_interval_zero(self):
        """Diagnostic disabled (interval=0) must write nothing."""
        with tempfile.TemporaryDirectory() as d:
            diag_dir = Path(d)
            trainer = _make_minimal_vae_trainer(0, diag_dir)
            self._run_fake_discriminator_step(trainer, global_step=0)

            assert not (diag_dir / "disc_logits.jsonl").exists()
            assert list(diag_dir.glob("*.png")) == []

    def test_interval_skips_non_matching_steps(self):
        """With interval=10, steps 1..9 produce no records."""
        with tempfile.TemporaryDirectory() as d:
            diag_dir = Path(d)
            trainer = _make_minimal_vae_trainer(10, diag_dir)
            for step in range(1, 10):
                self._run_fake_discriminator_step(trainer, global_step=step)

            assert not (diag_dir / "disc_logits.jsonl").exists()

    def test_interval_fires_at_multiples(self):
        """With interval=10, steps 0, 10, 20 each append one line."""
        with tempfile.TemporaryDirectory() as d:
            diag_dir = Path(d)
            trainer = _make_minimal_vae_trainer(10, diag_dir)
            for step in [0, 10, 20]:
                self._run_fake_discriminator_step(trainer, global_step=step)

            jsonl_path = diag_dir / "disc_logits.jsonl"
            lines = jsonl_path.read_text().strip().splitlines()
            assert len(lines) == 3, f"Expected 3 records, got {len(lines)}"

    def test_snapshot_survives_io_error(self, tmp_path):
        """A bad diag_dir must not raise — only log a warning."""
        trainer = _make_minimal_vae_trainer(1, tmp_path / "nonexistent_parent" / "subdir")
        # Should not raise
        logits = torch.zeros(2, 1, 15, 15)
        trainer._maybe_save_disc_logit_snapshot(logits, global_step=0)
