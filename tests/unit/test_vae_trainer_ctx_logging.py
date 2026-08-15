"""Tests for VAE trainer ctx-metric labeling/surfacing and error visibility.

Root cause (see fluxflow-training/CLAUDE.md and prior audits): the "Ctx" console
metric that plateaus near 1.0 regardless of optimizer choice is
``context_alignment_loss`` — a diagnostic probe with both inputs detached
(zero gradient path into SPADE/compressor/expander), trained by its own
separate ``context_predictor_optimizer``. It is now logged as
``ctx_probe_alignment`` to make that distinction visible, alongside the two
gradient-carrying ctx terms (``ctx_shrinkage_loss``, ``ctx_aux_loss``) and the
new SPADE γ/β drift diagnostics.

These tests avoid running a full forward pass through a real compressor (the
existing test suite for this module does the same — see
test_vae_trainer_ctx_shrinkage.py's docstring) and instead assert against the
module source, matching the established pattern used by
test_vae_trainer_bugs.py for similarly deep-in-a-heavy-method regressions.
"""

import importlib.util
import pathlib

import pytest
import torch
import torch.nn as nn
from fluxflow.models.activations import TrainableBezier, WideTrainableBezier

from fluxflow_training.training.vae_trainer import VAETrainer


def _source() -> str:
    spec = importlib.util.find_spec("fluxflow_training.training.vae_trainer")
    return pathlib.Path(spec.origin).read_text()


class TestCtxProbeRelabeling:
    """The disconnected diagnostic probe must not be labeled 'context_alignment'."""

    def test_context_alignment_key_no_longer_returned(self):
        source = _source()
        assert '"context_alignment":' not in source, (
            "vae_trainer.py still returns a 'context_alignment' dict key. It must be "
            "renamed to 'ctx_probe_alignment' so dashboards don't read it as a "
            "training-quality signal for SPADE."
        )

    def test_ctx_probe_alignment_key_present_in_train_generator_return(self):
        source = _source()
        assert '"ctx_probe_alignment": float(context_alignment_loss.detach().item())' in source

    def test_train_step_propagates_ctx_probe_alignment(self):
        source = _source()
        assert 'losses["ctx_probe_alignment"] = gen_losses.get("ctx_probe_alignment"' in source, (
            "train_step's outer losses dict must propagate ctx_probe_alignment from "
            "gen_losses, or it never reaches the orchestrator's console/logger output."
        )


class TestGradientCarryingCtxMetricsSurfaced:
    """ctx_shrinkage_loss and ctx_aux_loss (gradient-carrying) must reach train_step's
    returned dict alongside the probe, so callers can log them side by side."""

    def test_train_step_propagates_ctx_shrinkage_loss(self):
        source = _source()
        assert 'losses["ctx_shrinkage_loss"] = gen_losses.get("ctx_shrinkage_loss"' in source

    def test_train_step_propagates_ctx_shrinkage_alpha(self):
        source = _source()
        assert 'losses["ctx_shrinkage_alpha"] = gen_losses.get("ctx_shrinkage_alpha"' in source

    def test_train_step_already_propagates_ctx_aux_loss(self):
        # Pre-existing wiring; guard against a future regression removing it.
        source = _source()
        assert 'losses["ctx_aux_loss"] = gen_losses.get("ctx_aux_loss"' in source


class TestSpadeDriftSurfaced:
    """SPADE gamma/beta drift must reach train_step's returned dict, clearly labeled."""

    def test_train_step_reports_spade_gamma_drift(self):
        source = _source()
        assert 'losses["spade_gamma_drift"]' in source

    def test_train_step_reports_spade_beta_drift(self):
        source = _source()
        assert 'losses["spade_beta_drift"]' in source

    def test_compute_spade_drift_method_exists(self):
        assert hasattr(VAETrainer, "_compute_spade_drift")

    def test_compute_spade_drift_returns_zero_when_no_spade_blocks(self):
        """Legacy/stub expanders with no SPADE_v100b submodules -> (0.0, 0.0), no crash."""
        stub = type("_Stub", (), {"expander": nn.Linear(1, 1)})()
        gamma_drift, beta_drift = VAETrainer._compute_spade_drift(stub)
        assert gamma_drift == 0.0
        assert beta_drift == 0.0


class TestCtxAuxExceptionSurfaced:
    """Bug: the ctx_aux_loss except-block silently swallowed real exceptions behind
    a generic warning. It must now log the exception type/message loudly while
    staying non-fatal."""

    def test_old_generic_swallow_message_removed(self):
        source = _source()
        assert 'logger.warning(f"ctx_aux_loss computation skipped: {exc}")' not in source

    def test_exception_type_and_message_surfaced(self):
        source = _source()
        assert "type(exc).__name__" in source
        assert "ctx_aux_loss computation failed" in source

    def test_exception_logged_non_fatally_with_traceback(self):
        source = _source()
        assert "exc_info=True" in source


class TestDiscriminatorUpdateFreq:
    """Additive opt-in knob: default 1 must reproduce every-step behavior exactly."""

    @staticmethod
    def _stub(use_gan: bool, discriminator, freq: int):
        return type(
            "_Stub",
            (),
            {
                "use_gan": use_gan,
                "discriminator": discriminator,
                "discriminator_update_freq": freq,
            },
        )()

    def test_default_freq_trains_every_step(self):
        stub = self._stub(True, object(), 1)
        for step in range(5):
            assert VAETrainer._should_train_discriminator(stub, step) is True

    def test_freq_2_skips_odd_steps(self):
        stub = self._stub(True, object(), 2)
        results = [VAETrainer._should_train_discriminator(stub, s) for s in range(4)]
        assert results == [True, False, True, False]

    def test_no_discriminator_never_trains(self):
        stub = self._stub(True, None, 1)
        assert VAETrainer._should_train_discriminator(stub, 0) is False

    def test_use_gan_false_never_trains(self):
        stub = self._stub(False, object(), 1)
        assert VAETrainer._should_train_discriminator(stub, 0) is False

    def test_invalid_freq_raises_value_error(self):
        from unittest.mock import MagicMock

        compressor = nn.Linear(1, 1)
        compressor.use_gradient_checkpointing = False
        compressor.get_context_dims = lambda: 8  # type: ignore[attr-defined]
        expander = nn.Linear(1, 1)
        opt = torch.optim.SGD(expander.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

        with pytest.raises(ValueError, match="discriminator_update_freq"):
            VAETrainer(
                compressor=compressor,
                expander=expander,
                optimizer=opt,
                scheduler=sched,
                ema=MagicMock(),
                reconstruction_loss_fn=nn.L1Loss(),
                reconstruction_loss_min_fn=nn.MSELoss(),
                train_reconstruction=False,
                train_kl=False,
                train_colorstats=False,
                train_histogram=False,
                train_contrast=False,
                train_coarseness=False,
                use_gan=False,
                use_lpips=False,
                ctx_input_dim=16,
                discriminator_update_freq=0,
                accelerator=None,
            )


# ---------------------------------------------------------------------------
# Full train_step integration, reusing the minimal-but-real fixture pattern
# from test_vae_trainer_discriminator_miopen.py (fake compressor/expander/
# discriminator that produce real tensors of the right shapes).
# ---------------------------------------------------------------------------


class _FakeCompressor(nn.Module):
    """Minimal compressor stub returning a (packed, mu, logvar) tuple."""

    def __init__(self, token_dim: int = 8, n_tokens: int = 3):
        super().__init__()
        self.token_dim = token_dim
        self.n_tokens = n_tokens
        self.dummy = nn.Linear(1, 1)
        self.use_gradient_checkpointing = False
        self.mu_activation = TrainableBezier(shape=(4, 2, 2))
        self.logvar_activation = WideTrainableBezier(shape=(4, 2, 2))

    def forward(self, x, training=False):
        B = x.size(0)
        packed = torch.randn(B, self.n_tokens + 1, self.token_dim)
        mu = torch.zeros(B, 4, 2, 2)
        logvar = torch.zeros(B, 4, 2, 2)
        return packed, mu, logvar

    def get_context_dims(self):
        return self.token_dim


class _FakeExpander(nn.Module):
    """Minimal expander stub returning zeros of the expected image shape."""

    def __init__(self, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape
        self.dummy = nn.Linear(1, 1)

    def forward(self, packed, use_context=True):
        B = packed.size(0)
        C, H, W = self.img_shape
        return torch.zeros(B, C, H, W)


class _FakeDiscriminator(nn.Module):
    """Minimal patch discriminator stub with a ctx_proj for dim validation."""

    def __init__(self, ctx_dim: int, img_channels: int = 3):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, 4)
        self.ctx_dim = ctx_dim
        self.conv = nn.Conv2d(img_channels, 1, 3, padding=1)

    def forward(self, x, ctx=None):
        return self.conv(x)


_TOKEN_DIM = 8
_IMG_SHAPE = (3, 8, 8)


def _build_full_trainer(discriminator_update_freq: int = 1) -> VAETrainer:
    from unittest.mock import MagicMock

    compressor = _FakeCompressor(token_dim=_TOKEN_DIM)
    expander = _FakeExpander(_IMG_SHAPE)
    discriminator = _FakeDiscriminator(ctx_dim=_TOKEN_DIM)
    opt = torch.optim.SGD(expander.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    accelerator = MagicMock()
    accelerator.backward = lambda loss: loss.backward()

    return VAETrainer(
        compressor=compressor,
        expander=expander,
        optimizer=opt,
        scheduler=sched,
        ema=MagicMock(),
        reconstruction_loss_fn=nn.L1Loss(),
        reconstruction_loss_min_fn=nn.MSELoss(),
        train_reconstruction=False,
        train_kl=False,
        train_colorstats=False,
        train_histogram=False,
        train_contrast=False,
        train_coarseness=False,
        train_ctx_aux=False,
        use_gan=True,
        discriminator=discriminator,
        discriminator_optimizer=MagicMock(),
        discriminator_scheduler=MagicMock(),
        use_lpips=False,
        ctx_input_dim=_TOKEN_DIM,
        context_channels=2,
        context_height=2,
        context_width=2,
        r1_interval=1000,
        discriminator_update_freq=discriminator_update_freq,
        accelerator=accelerator,
    )


class TestTrainStepCtxMetricsEndToEnd:
    """Real train_step() call: the renamed/new metrics must actually appear."""

    def test_losses_dict_has_new_ctx_and_spade_keys_not_old_one(self):
        trainer = _build_full_trainer()
        real_imgs = torch.randn(2, *_IMG_SHAPE)

        losses = trainer.train_step(real_imgs, global_step=1)

        assert "context_alignment" not in losses
        for key in (
            "ctx_probe_alignment",
            "ctx_shrinkage_loss",
            "ctx_shrinkage_alpha",
            "spade_gamma_drift",
            "spade_beta_drift",
        ):
            assert key in losses, f"train_step losses missing '{key}'"
            assert isinstance(losses[key], float)

    def test_spade_drift_zero_for_expander_without_spade_blocks(self):
        """_FakeExpander has no SPADE_v100b submodules -> drift reads 0.0."""
        trainer = _build_full_trainer()
        real_imgs = torch.randn(2, *_IMG_SHAPE)

        losses = trainer.train_step(real_imgs, global_step=1)

        assert losses["spade_gamma_drift"] == 0.0
        assert losses["spade_beta_drift"] == 0.0


class TestTrainStepDiscriminatorUpdateFreqEndToEnd:
    """discriminator_update_freq must gate the real train_step discriminator path."""

    def test_freq_1_trains_discriminator_every_step(self):
        trainer = _build_full_trainer(discriminator_update_freq=1)
        for step in (0, 1, 2):
            real_imgs = torch.randn(2, *_IMG_SHAPE)
            losses = trainer.train_step(real_imgs, global_step=step)
            assert "discriminator" in losses

    def test_freq_2_skips_discriminator_on_odd_steps(self):
        trainer = _build_full_trainer(discriminator_update_freq=2)

        losses_step0 = trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=0)
        assert "discriminator" in losses_step0

        losses_step1 = trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=1)
        assert "discriminator" not in losses_step1

        losses_step2 = trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=2)
        assert "discriminator" in losses_step2
