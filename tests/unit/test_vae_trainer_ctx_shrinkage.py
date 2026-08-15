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


class _CompressorWithLearnableCtx(nn.Module):
    """Compressor stub whose pre-norm ctx tensor is a learnable parameter.

    Used to prove the hook restores real gradient flow: optimizing
    ``compute_ctx_shrinkage`` on the *pre-norm* cache should be able to shrink
    even the post-norm (GroupNorm) output magnitude, which a post-hook capture
    cannot do (GroupNorm output is ~scale-invariant to its input).
    """

    def __init__(self, d_model: int = 8, H: int = 4, W: int = 4, ctx_init: float = 1.0):
        super().__init__()
        self.ctx_zinject_norm = nn.GroupNorm(
            num_groups=min(4, d_model), num_channels=d_model, affine=False
        )
        self.ctx_raw = nn.Parameter(
            torch.full((d_model, H, W), ctx_init) + torch.randn(d_model, H, W) * 0.05
        )
        self.d_model = d_model
        self.H = H
        self.W = W
        self.use_gradient_checkpointing = False

    def forward(self, x, training=False):
        B = x.size(0)
        ctx = self.ctx_raw.unsqueeze(0).expand(B, -1, -1, -1)
        _ = self.ctx_zinject_norm(ctx)
        return torch.zeros(B, 1, 2 * self.d_model)

    def get_context_dims(self):
        return self.d_model


class TestCtxShrinkageHookCapturesPreNormInput:
    def test_cache_scales_with_prenorm_input_unlike_postnorm_output(self):
        """The hook must capture the *pre*-norm ctx tensor.

        GroupNorm(affine=False) output is (near) invariant to a uniform scale
        of its input, so a post-hook capture would look identical whether the
        upstream ctx branch is scaled by 1x or 5x. The pre-hook capture must
        differ by exactly that scale factor.
        """
        comp = _CompressorWithCtxNorm(d_model=8, H=4, W=4)
        trainer = _build_trainer(comp, ctx_shrinkage_weight=0.001)

        torch.manual_seed(0)
        base_ctx = torch.randn(2, comp.d_model, comp.H, comp.W) + 2.0  # nonzero variance

        def make_forward(scale):
            def _forward(x, training=False):
                B = x.size(0)
                _ = comp.ctx_zinject_norm(base_ctx * scale)
                return torch.zeros(B, 1, 2 * comp.d_model)

            return _forward

        comp.forward = make_forward(1.0)
        _ = comp(torch.randn(2, 3, 8, 8))
        baseline_cache = trainer._ctx_features_cache.clone()

        comp.forward = make_forward(5.0)
        _ = comp(torch.randn(2, 3, 8, 8))
        scaled_cache = trainer._ctx_features_cache.clone()

        # Pre-norm cache scales linearly with the upstream scale factor.
        assert torch.allclose(scaled_cache, baseline_cache * 5.0, atol=1e-4)

        # The post-norm output itself is ~invariant to the same scale factor -
        # a post-hook capture would NOT show the difference asserted above.
        with torch.no_grad():
            post_baseline = comp.ctx_zinject_norm(base_ctx * 1.0)
            post_scaled = comp.ctx_zinject_norm(base_ctx * 5.0)
        assert torch.allclose(post_baseline, post_scaled, atol=1e-4)


class TestCtxShrinkageRestoresGradientFlow:
    def test_optimizer_steps_under_shrinkage_loss_shrink_postnorm_output(self):
        """N optimizer steps on compute_ctx_shrinkage(pre-norm cache) alone must
        measurably shrink the *post-norm* (GroupNorm) output magnitude.

        This is the test proving the fix restores real gradient flow: under the
        old post-hook bug, optimizing mean(post_norm_output^2) barely moves the
        post-norm output at all (it's already ~unit-variance by construction),
        so this assertion fails against the buggy code.
        """
        torch.manual_seed(0)
        comp = _CompressorWithLearnableCtx(d_model=8, H=4, W=4, ctx_init=1.0)
        trainer = _build_trainer(
            comp,
            ctx_shrinkage_weight=1.0,
            ctx_shrinkage_warmup_start_step=0,
            ctx_shrinkage_warmup_steps=1,
        )
        assert trainer._ctx_hook_handle is not None

        opt = torch.optim.SGD(comp.parameters(), lr=0.5)
        alpha = 1.0
        dummy_x = torch.zeros(2, 3, 8, 8)

        with torch.no_grad():
            initial_post_mag = comp.ctx_zinject_norm(comp.ctx_raw.unsqueeze(0)).pow(2).mean().item()

        for _ in range(2000):
            opt.zero_grad()
            _ = comp(dummy_x)
            loss = compute_ctx_shrinkage(trainer._ctx_features_cache, alpha)
            loss.backward()
            opt.step()

        with torch.no_grad():
            final_post_mag = comp.ctx_zinject_norm(comp.ctx_raw.unsqueeze(0)).pow(2).mean().item()

        assert final_post_mag < initial_post_mag * 0.5, (
            f"post-norm magnitude barely moved ({initial_post_mag:.6g} -> "
            f"{final_post_mag:.6g}); shrinkage loss is not reaching the "
            "pre-norm ctx tensor."
        )


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
