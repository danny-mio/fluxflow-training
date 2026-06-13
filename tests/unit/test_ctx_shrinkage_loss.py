"""Unit tests for v0.10.0 ctx shrinkage loss and KL/ctx warmup helpers."""

import torch

from fluxflow_training.training.losses import (
    compute_ctx_shrinkage,
    cosine_warmup_weight,
    delayed_cosine_warmup_weight,
)


class TestComputeCtxShrinkage:
    """L_ctx_shrink = alpha * mean(||ctx_features||^2)."""

    def test_zero_alpha_returns_zero(self):
        ctx = torch.randn(2, 32, 8, 8)
        loss = compute_ctx_shrinkage(ctx, alpha=0.0)
        assert loss.item() == 0.0

    def test_negative_alpha_returns_zero(self):
        ctx = torch.randn(2, 32, 8, 8)
        loss = compute_ctx_shrinkage(ctx, alpha=-1e-3)
        assert loss.item() == 0.0

    def test_ones_input_gives_alpha(self):
        """For ctx = ones, mean(x^2) = 1 so L = alpha."""
        ctx = torch.ones(2, 32, 8, 8)
        loss = compute_ctx_shrinkage(ctx, alpha=0.001)
        assert abs(loss.item() - 0.001) < 1e-6

    def test_two_constants_gives_4_alpha(self):
        """For ctx = 2*ones, mean(x^2) = 4 so L = 4*alpha."""
        ctx = torch.full((4, 8, 16, 16), 2.0)
        loss = compute_ctx_shrinkage(ctx, alpha=0.5)
        assert abs(loss.item() - 2.0) < 1e-6  # 0.5 * 4 == 2.0

    def test_scalar_output(self):
        ctx = torch.randn(2, 32, 8, 8)
        loss = compute_ctx_shrinkage(ctx, alpha=0.001)
        assert loss.dim() == 0

    def test_gradient_flows_when_alpha_positive(self):
        ctx = torch.randn(2, 32, 8, 8, requires_grad=True)
        loss = compute_ctx_shrinkage(ctx, alpha=0.01)
        loss.backward()
        assert ctx.grad is not None
        assert ctx.grad.abs().sum().item() > 0

    def test_no_gradient_when_alpha_zero(self):
        ctx = torch.randn(2, 32, 8, 8, requires_grad=True)
        loss = compute_ctx_shrinkage(ctx, alpha=0.0)
        # Short-circuit returns a fresh zero tensor with no grad_fn — backward
        # would raise. The contract is: alpha=0 contributes nothing to the
        # gradient graph, which is what training callers depend on.
        assert not loss.requires_grad

    def test_dtype_preserved(self):
        ctx = torch.randn(2, 32, 8, 8, dtype=torch.float64)
        loss = compute_ctx_shrinkage(ctx, alpha=0.0)
        assert loss.dtype == torch.float64

    def test_device_preserved(self):
        ctx = torch.randn(2, 4, 4)
        loss = compute_ctx_shrinkage(ctx, alpha=0.0)
        assert loss.device == ctx.device


class TestCosineWarmupWeight:
    """Cosine warmup helper used by KL_z and ctx shrinkage schedules."""

    def test_zero_at_start(self):
        assert cosine_warmup_weight(0, 1000, 0.5) == 0.0

    def test_max_at_end(self):
        # cosine(pi)=-1 so (1 - cos(pi))/2 = 1.
        assert cosine_warmup_weight(1000, 1000, 0.5) == 0.5

    def test_clamps_beyond_end(self):
        assert cosine_warmup_weight(2000, 1000, 0.5) == 0.5

    def test_midpoint_is_half_max(self):
        # cos(pi/2)=0 so (1 - cos)/2 = 0.5; weight is 0.5 * max_weight.
        w = cosine_warmup_weight(500, 1000, 0.5)
        assert abs(w - 0.25) < 1e-6  # 0.5 * 0.5

    def test_zero_warmup_returns_max(self):
        assert cosine_warmup_weight(0, 0, 0.5) == 0.5

    def test_default_kl_z_target(self):
        # Plan §5: kl_z_weight=0.5, warmup over 10000 steps.
        assert cosine_warmup_weight(10000, 10000, 0.5) == 0.5
        assert cosine_warmup_weight(0, 10000, 0.5) == 0.0


class TestDelayedCosineWarmupWeight:
    """ctx shrinkage schedule: hold at 0 until start_step, then cosine warmup."""

    def test_zero_before_start(self):
        assert delayed_cosine_warmup_weight(0, 5000, 5000, 1e-3) == 0.0
        assert delayed_cosine_warmup_weight(4999, 5000, 5000, 1e-3) == 0.0

    def test_zero_at_start_step(self):
        # At exactly start_step the warmup begins from 0.
        assert delayed_cosine_warmup_weight(5000, 5000, 5000, 1e-3) == 0.0

    def test_max_at_end_of_warmup(self):
        # start_step + warmup_steps -> max_weight.
        w = delayed_cosine_warmup_weight(10000, 5000, 5000, 1e-3)
        assert abs(w - 1e-3) < 1e-9

    def test_held_at_max_after(self):
        w = delayed_cosine_warmup_weight(50000, 5000, 5000, 1e-3)
        assert abs(w - 1e-3) < 1e-9

    def test_midpoint_of_warmup(self):
        # Cosine warmup midpoint: (1 - cos(pi/2))/2 = 0.5, so half of max_weight.
        w = delayed_cosine_warmup_weight(7500, 5000, 5000, 1e-3)
        assert abs(w - 0.5e-3) < 1e-9
