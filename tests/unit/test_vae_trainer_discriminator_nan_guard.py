"""Tests for the discriminator NaN/Inf guard in ``VAETrainer._train_discriminator``.

Confirmed bug: ``d_img_loss`` (r1_penalty + d_hinge_loss) was passed straight to
``self.accelerator.backward(...)`` / ``self.discriminator_optimizer.step()`` with
zero NaN/Inf checking, unlike ``_train_generator`` which has a full
``check_for_nan`` diagnostic-and-skip block. This let discriminator weights go
permanently NaN mid-training with no diagnostic ever logged.

``_train_discriminator`` must, immediately after ``d_img_loss`` is fully
composed and scaled by ``gradient_accumulation_steps`` (and before
``accelerator.backward(...)``):

- Detect NaN/Inf in ``d_img_loss`` via the existing ``check_for_nan`` helper.
- Log an error-level diagnostic mirroring ``_train_generator``'s NaN block:
  ``r1`` (only if ``do_r1`` was true this step), ``d_hinge_cond``,
  ``d_hinge_uncond``, and min/max stats for ``real_logits``, ``fake_logits``,
  ``fake_uncond_logits``.
- Skip the batch: never call ``accelerator.backward(...)`` or
  ``discriminator_optimizer.step()``.
- Return ``{"d_loss": 0.0, "_optimizer_stepped": False}`` -- matching the
  existing MIOpen-failure-skip return shape exactly.
- Leave the healthy (finite-loss) path, the MIOpen catch, and the
  gradient-accumulation mid-window skip path unchanged.
"""

import logging
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from fluxflow.models.activations import TrainableBezier, WideTrainableBezier

from fluxflow_training.training.vae_trainer import VAETrainer

TOKEN_DIM = 8
IMG_SHAPE = (3, 8, 8)


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

    def __init__(self, img_shape: tuple[int, int, int]):
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


def _build_trainer(
    discriminator,
    accelerator,
    r1_interval: int = 1000,
    gradient_accumulation_steps: int = 1,
) -> VAETrainer:
    """Build a minimally-configured VAETrainer with GAN enabled.

    Mirrors ``test_vae_trainer_discriminator_miopen.py``'s fixture: bypasses
    reconstruction/KL/LPIPS/regularizer paths via train_* flags so only the
    discriminator (and a cheap generator pass) actually run.
    """
    compressor = _FakeCompressor(token_dim=TOKEN_DIM)
    expander = _FakeExpander(IMG_SHAPE)
    opt = torch.optim.SGD(expander.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

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
        ctx_input_dim=TOKEN_DIM,
        context_channels=2,
        context_height=2,
        context_width=2,
        r1_interval=r1_interval,
        accelerator=accelerator,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


class TestTrainDiscriminatorNanGuardHealthyPath:
    """Regression: finite d_img_loss must behave exactly as before."""

    def test_finite_loss_still_steps_optimizer(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()  # backward is a no-op MagicMock (no exception)
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        result = trainer._train_discriminator(real_imgs, global_step=1)

        accelerator.backward.assert_called_once()
        trainer.discriminator_optimizer.step.assert_called_once()
        assert result["_optimizer_stepped"] is True
        assert isinstance(result["d_loss"], float)
        assert result["d_loss"] == result["d_loss"]  # not NaN


class TestTrainDiscriminatorNanGuardSkip:
    """NaN/Inf d_img_loss must be caught before backward()/step()."""

    def test_nan_d_img_loss_skips_backward_and_step(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        trainer = _build_trainer(discriminator, accelerator, r1_interval=1000)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        with patch(
            "fluxflow_training.training.vae_trainer.d_hinge_loss",
            return_value=torch.tensor(float("nan")),
        ):
            result = trainer._train_discriminator(real_imgs, global_step=1)

        accelerator.backward.assert_not_called()
        trainer.discriminator_optimizer.step.assert_not_called()
        assert result == {"d_loss": 0.0, "_optimizer_stepped": False}

    def test_inf_d_img_loss_skips_backward_and_step(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        trainer = _build_trainer(discriminator, accelerator, r1_interval=1000)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        with patch(
            "fluxflow_training.training.vae_trainer.d_hinge_loss",
            return_value=torch.tensor(float("inf")),
        ):
            result = trainer._train_discriminator(real_imgs, global_step=1)

        accelerator.backward.assert_not_called()
        trainer.discriminator_optimizer.step.assert_not_called()
        assert result == {"d_loss": 0.0, "_optimizer_stepped": False}

    def test_nan_diagnostic_is_logged(self, caplog):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        trainer = _build_trainer(discriminator, accelerator, r1_interval=1000)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        with caplog.at_level(logging.ERROR, logger="fluxflow_training.training.vae_trainer"):
            with patch(
                "fluxflow_training.training.vae_trainer.d_hinge_loss",
                return_value=torch.tensor(float("nan")),
            ):
                trainer._train_discriminator(real_imgs, global_step=1)

        text = caplog.text.lower()
        assert "d_hinge_cond" in text
        assert "d_hinge_uncond" in text
        assert "real_logits" in text
        assert "fake_logits" in text
        assert "fake_uncond_logits" in text


class TestTrainDiscriminatorNanGuardDoR1False:
    """do_r1=False must not raise NameError/UnboundLocalError referencing r1."""

    def test_do_r1_false_nan_does_not_raise(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        # r1_interval huge => do_r1 is False at global_step=1.
        trainer = _build_trainer(discriminator, accelerator, r1_interval=1000)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        with patch(
            "fluxflow_training.training.vae_trainer.d_hinge_loss",
            return_value=torch.tensor(float("nan")),
        ):
            # Must not raise UnboundLocalError/NameError for undefined `r1`.
            result = trainer._train_discriminator(real_imgs, global_step=1)

        assert result == {"d_loss": 0.0, "_optimizer_stepped": False}


class TestTrainDiscriminatorNanGuardDoR1True:
    """do_r1=True: diagnostic must include r1 stats without crashing."""

    def test_do_r1_true_nan_logs_r1_and_does_not_raise(self, caplog):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        # r1_interval=1 => do_r1 is True at global_step=0.
        trainer = _build_trainer(discriminator, accelerator, r1_interval=1)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        with caplog.at_level(logging.ERROR, logger="fluxflow_training.training.vae_trainer"):
            with patch(
                "fluxflow_training.training.vae_trainer.d_hinge_loss",
                return_value=torch.tensor(float("nan")),
            ):
                result = trainer._train_discriminator(real_imgs, global_step=0)

        assert result == {"d_loss": 0.0, "_optimizer_stepped": False}
        accelerator.backward.assert_not_called()
        trainer.discriminator_optimizer.step.assert_not_called()
        assert "r1" in caplog.text.lower()


class TestTrainDiscriminatorNanGuardAccumulation:
    """NaN guard must fire before the should_step accumulation check."""

    def test_nan_guard_fires_regardless_of_accumulation_window(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        # gradient_accumulation_steps=2, _accumulation_step starts at 0 =>
        # should_step would normally be False (mid-window). The NaN guard
        # must still return the NaN-skip shape (d_loss=0.0), not the
        # mid-accumulation-window shape (d_loss=<real value>).
        trainer = _build_trainer(
            discriminator, accelerator, r1_interval=1000, gradient_accumulation_steps=2
        )
        assert trainer._accumulation_step == 0
        real_imgs = torch.randn(2, *IMG_SHAPE)

        with patch(
            "fluxflow_training.training.vae_trainer.d_hinge_loss",
            return_value=torch.tensor(float("nan")),
        ):
            result = trainer._train_discriminator(real_imgs, global_step=1)

        accelerator.backward.assert_not_called()
        trainer.discriminator_optimizer.step.assert_not_called()
        assert result == {"d_loss": 0.0, "_optimizer_stepped": False}
