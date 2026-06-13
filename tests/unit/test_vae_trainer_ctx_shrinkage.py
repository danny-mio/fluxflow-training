"""Unit tests for the v0.10.0 ctx-shrinkage loss wiring on VAETrainer.

The trainer installs a forward hook on ``compressor.ctx_zinject_norm`` and
adds ``alpha(t) * mean(||ctx||^2)`` to the total loss. The schedule is the
delayed cosine warmup from losses.delayed_cosine_warmup_weight.

We don't run a full training step here - just verify the hook lifecycle
and the schedule wiring. End-to-end behaviour is covered by the existing
integration tests (which exercise the trainer with a v060 compressor that
has no ctx_zinject_norm, so the new path stays a no-op there).
"""

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from fluxflow_training.training.losses import (
    compute_ctx_shrinkage,
    delayed_cosine_warmup_weight,
)
from fluxflow_training.training.vae_trainer import VAETrainer


class _CompressorWithCtxNorm(nn.Module):
    """Minimal compressor stub exposing ``ctx_zinject_norm`` for hook testing."""

    def __init__(self, d_model: int = 8, H: int = 4, W: int = 4):
        super().__init__()
        self.ctx_zinject_norm = nn.GroupNorm(
            num_groups=min(4, d_model), num_channels=d_model, affine=False
        )
        self.d_model = d_model
        self.H = H
        self.W = W
        # One real parameter so .parameters() works.
        self.dummy = nn.Linear(1, 1)
        self.use_gradient_checkpointing = False

    def forward(self, x, training=False):
        # Run ctx_zinject_norm on a real tensor so the hook fires.
        B = x.size(0)
        ctx = torch.ones(B, self.d_model, self.H, self.W)
        _ = self.ctx_zinject_norm(ctx)
        # Return a packed-like tensor of arbitrary shape (not used in these tests).
        return torch.zeros(B, 1, 2 * self.d_model)

    def get_context_dims(self):
        return self.d_model


def _build_trainer(
    compressor,
    ctx_shrinkage_weight=0.001,
    ctx_shrinkage_warmup_start_step=0,
    ctx_shrinkage_warmup_steps=100,
):
    """Build a minimally-configured VAETrainer for hook lifecycle tests.

    We bypass GAN / LPIPS / context_predictor heavy paths via train_* flags
    and pass simple losses.
    """
    expander = nn.Linear(1, 1)
    opt = torch.optim.SGD(expander.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    ema = MagicMock()

    return VAETrainer(
        compressor=compressor,
        expander=expander,
        optimizer=opt,
        scheduler=sched,
        ema=ema,
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
        ctx_shrinkage_weight=ctx_shrinkage_weight,
        ctx_shrinkage_warmup_start_step=ctx_shrinkage_warmup_start_step,
        ctx_shrinkage_warmup_steps=ctx_shrinkage_warmup_steps,
        accelerator=None,
    )


class TestCtxShrinkageHookLifecycle:
    def test_hook_installs_when_weight_positive_and_norm_present(self):
        comp = _CompressorWithCtxNorm()
        trainer = _build_trainer(comp, ctx_shrinkage_weight=0.001)
        assert trainer._ctx_hook_handle is not None

    def test_no_hook_when_weight_zero(self):
        comp = _CompressorWithCtxNorm()
        trainer = _build_trainer(comp, ctx_shrinkage_weight=0.0)
        assert trainer._ctx_hook_handle is None

    def test_no_hook_when_norm_missing(self):
        """Legacy compressors (v060/v070) have no ctx_zinject_norm."""
        comp = nn.Linear(1, 1)
        comp.use_gradient_checkpointing = False  # required by trainer init
        # Compressor's parameters call must work.
        comp.get_context_dims = lambda: 8  # type: ignore[attr-defined]
        trainer = _build_trainer(comp, ctx_shrinkage_weight=0.001)
        assert trainer._ctx_hook_handle is None

    def test_remove_hook_detaches(self):
        comp = _CompressorWithCtxNorm()
        trainer = _build_trainer(comp, ctx_shrinkage_weight=0.001)
        trainer.remove_ctx_shrinkage_hook()
        assert trainer._ctx_hook_handle is None

    def test_hook_populates_cache_on_forward(self):
        comp = _CompressorWithCtxNorm(d_model=8, H=4, W=4)
        trainer = _build_trainer(comp, ctx_shrinkage_weight=0.001)
        # Drive a forward pass on the compressor manually.
        _ = comp(torch.randn(2, 3, 32, 32))
        assert trainer._ctx_features_cache is not None
        assert trainer._ctx_features_cache.shape == (2, 8, 4, 4)


class TestCtxShrinkageSchedule:
    def test_alpha_zero_before_start_step(self):
        # Mirrors trainer behavior; the wiring uses delayed_cosine_warmup_weight.
        assert delayed_cosine_warmup_weight(100, 5000, 5000, 1e-3) == 0.0

    def test_alpha_reaches_max_after_warmup(self):
        v = delayed_cosine_warmup_weight(10_000, 5000, 5000, 1e-3)
        assert abs(v - 1e-3) < 1e-9

    def test_compute_ctx_shrinkage_short_circuits_at_alpha_zero(self):
        out = compute_ctx_shrinkage(torch.randn(4, 8, 4, 4), alpha=0.0)
        assert out.item() == 0.0
        assert not out.requires_grad
