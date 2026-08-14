"""Tests for the ROCm/MIOpen kernel-compile-failure tolerance in _train_discriminator.

On some runs, ``self.accelerator.backward(d_img_loss)`` raises
``RuntimeError: miopenStatusUnknownError`` due to an upstream MIOpen bug where a
GEMM/Im2Col solver picks a kernel that fails to compile under HIPRTC on this
GPU target. This is a one-time cold-start failure, not a logic bug: once
caught and skipped, subsequent discriminator steps are expected to succeed.

``_train_discriminator`` must:
- Catch only ``RuntimeError`` whose message contains "miopen" (case-insensitive).
- On match: print a diagnostic, skip ``discriminator_optimizer.step()``, and
  signal the caller (via the same ``_optimizer_stepped`` convention used by
  ``_train_generator``'s NaN-skip path) that the step did not happen.
- On any other exception: re-raise unchanged.
- On success: behave exactly as before (dict return, optimizer stepped).

``train_step`` must thread that flag through so a skipped discriminator step
does not pollute ``d_loss_buffer`` or advance ``discriminator_scheduler``.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.vae_trainer import VAETrainer


class _FakeCompressor(nn.Module):
    """Minimal compressor stub returning a (packed, mu, logvar) tuple."""

    def __init__(self, token_dim: int = 8, n_tokens: int = 3):
        super().__init__()
        self.token_dim = token_dim
        self.n_tokens = n_tokens
        self.dummy = nn.Linear(1, 1)
        self.use_gradient_checkpointing = False

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


TOKEN_DIM = 8
IMG_SHAPE = (3, 8, 8)


def _build_trainer(discriminator, accelerator, r1_interval: int = 1000) -> VAETrainer:
    """Build a minimally-configured VAETrainer with GAN enabled.

    Bypasses reconstruction/KL/LPIPS/regularizer paths via train_* flags so
    only the discriminator (and a cheap generator pass) actually run.
    r1_interval is set large so the R1 penalty branch (which needs a real
    autograd graph) is not exercised at the small global_steps used in tests.
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
    )


class TestTrainDiscriminatorMiopenSkip:
    """Direct unit tests on _train_discriminator's MIOpen catch-and-skip path."""

    def test_miopen_error_is_caught_and_step_skipped(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        accelerator.backward.side_effect = RuntimeError(
            "miopenStatusUnknownError: failed to compile kernel"
        )
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        result = trainer._train_discriminator(real_imgs, global_step=1)

        assert result["_optimizer_stepped"] is False
        trainer.discriminator_optimizer.step.assert_not_called()

    def test_miopen_match_is_case_insensitive(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        accelerator.backward.side_effect = RuntimeError("MIOpen: internal error")
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        result = trainer._train_discriminator(real_imgs, global_step=1)

        assert result["_optimizer_stepped"] is False
        trainer.discriminator_optimizer.step.assert_not_called()

    def test_miopen_skip_prints_diagnostic_with_context(self, capsys):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        accelerator.backward.side_effect = RuntimeError("miopenStatusUnknownError")
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        trainer._train_discriminator(real_imgs, global_step=42)

        out = capsys.readouterr().out.lower()
        assert "miopen" in out
        assert "42" in out
        assert "discriminator" in out

    def test_non_miopen_runtime_error_propagates(self):
        """Real bugs (NaN, shape mismatch, OOM) must still crash the run."""
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        accelerator.backward.side_effect = RuntimeError("CUDA out of memory")
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            trainer._train_discriminator(real_imgs, global_step=1)

        trainer.discriminator_optimizer.step.assert_not_called()

    def test_non_runtime_error_propagates(self):
        """Non-RuntimeError exceptions must never be swallowed."""
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        accelerator.backward.side_effect = ValueError("unrelated failure")
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        with pytest.raises(ValueError, match="unrelated failure"):
            trainer._train_discriminator(real_imgs, global_step=1)

    def test_happy_path_unchanged(self):
        """No exception: optimizer steps, dict return carries the loss + stepped=True."""
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()  # backward is a no-op MagicMock (no exception)
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        result = trainer._train_discriminator(real_imgs, global_step=1)

        assert result["_optimizer_stepped"] is True
        assert isinstance(result["d_loss"], float)
        trainer.discriminator_optimizer.step.assert_called_once()


class TestTrainStepDiscriminatorSkipIntegration:
    """train_step must thread the skip flag through without polluting metrics."""

    def test_skip_does_not_pollute_d_loss_buffer_or_step_scheduler(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        # Discriminator backward fails (MIOpen); generator backward succeeds.
        accelerator.backward.side_effect = [
            RuntimeError("miopenStatusUnknownError"),
            None,
        ]
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        losses = trainer.train_step(real_imgs, global_step=1)

        assert "discriminator" not in losses
        assert len(trainer.d_loss_buffer._items) == 0
        trainer.discriminator_scheduler.step.assert_not_called()

    def test_no_crash_propagates_out_of_train_step(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        accelerator.backward.side_effect = [
            RuntimeError("miopenStatusUnknownError"),
            None,
        ]
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        # Must not raise.
        trainer.train_step(real_imgs, global_step=1)

    def test_successful_step_populates_buffer_and_steps_scheduler(self):
        discriminator = _FakeDiscriminator(ctx_dim=TOKEN_DIM)
        accelerator = MagicMock()
        trainer = _build_trainer(discriminator, accelerator)
        real_imgs = torch.randn(2, *IMG_SHAPE)

        losses = trainer.train_step(real_imgs, global_step=1)

        assert "discriminator" in losses
        assert len(trainer.d_loss_buffer._items) == 1
        trainer.discriminator_scheduler.step.assert_called_once()
