"""Unit tests for the v0.10.0 bezier_reg monotonicity regularizer.

Plan: plans/01-vae-v0.10.0-improvements.md, Fix 2. ``compute_bezier_monotonicity_reg``
is a hinge penalty on a TrainableBezier/WideTrainableBezier module's control
points, enforcing p0<=p1<=p2<=p3.
"""

import torch
from fluxflow.models.activations import TrainableBezier, WideTrainableBezier

from fluxflow_training.training.losses import compute_bezier_monotonicity_reg


class TestComputeBezierMonotonicityReg:
    def test_default_monotonic_init_gives_zero(self):
        """Default control-point init (p0<p1<p2<p3) has zero hinge penalty."""
        module = TrainableBezier(shape=(4,), channel_only=True)
        loss = compute_bezier_monotonicity_reg(module, weight=0.05)
        assert loss.item() == 0.0

    def test_default_wide_bezier_init_gives_zero(self):
        module = WideTrainableBezier(shape=(4,), channel_only=True)
        loss = compute_bezier_monotonicity_reg(module, weight=0.05)
        assert loss.item() == 0.0

    def test_inverted_control_point_gives_positive_penalty(self):
        """Manually invert p1 > p2 -- penalty must be strictly positive."""
        module = TrainableBezier(shape=(4,), channel_only=True)
        with torch.no_grad():
            module.p1.fill_(0.9)
            module.p2.fill_(0.1)
        loss = compute_bezier_monotonicity_reg(module, weight=0.05)
        assert loss.item() > 0.0

    def test_penalty_scales_with_weight(self):
        module = TrainableBezier(shape=(4,), channel_only=True)
        with torch.no_grad():
            module.p1.fill_(0.9)
            module.p2.fill_(0.1)
        loss_low = compute_bezier_monotonicity_reg(module, weight=0.01)
        loss_high = compute_bezier_monotonicity_reg(module, weight=0.5)
        assert loss_high.item() > loss_low.item()

    def test_zero_weight_returns_zero_even_when_inverted(self):
        module = TrainableBezier(shape=(4,), channel_only=True)
        with torch.no_grad():
            module.p1.fill_(0.9)
            module.p2.fill_(0.1)
        loss = compute_bezier_monotonicity_reg(module, weight=0.0)
        assert loss.item() == 0.0

    def test_negative_weight_returns_zero(self):
        module = TrainableBezier(shape=(4,), channel_only=True)
        with torch.no_grad():
            module.p1.fill_(0.9)
            module.p2.fill_(0.1)
        loss = compute_bezier_monotonicity_reg(module, weight=-0.1)
        assert loss.item() == 0.0

    def test_no_gradient_when_weight_zero(self):
        module = TrainableBezier(shape=(4,), channel_only=True)
        loss = compute_bezier_monotonicity_reg(module, weight=0.0)
        assert not loss.requires_grad

    def test_gradient_flows_to_inverted_control_points(self):
        module = TrainableBezier(shape=(4,), channel_only=True)
        with torch.no_grad():
            module.p1.fill_(0.9)
            module.p2.fill_(0.1)
        loss = compute_bezier_monotonicity_reg(module, weight=0.05)
        loss.backward()
        assert module.p1.grad is not None
        assert module.p2.grad is not None
        assert module.p1.grad.abs().sum().item() > 0
        assert module.p2.grad.abs().sum().item() > 0

    def test_scalar_output(self):
        module = TrainableBezier(shape=(4,), channel_only=True)
        loss = compute_bezier_monotonicity_reg(module, weight=0.05)
        assert loss.dim() == 0

    def test_dtype_and_device_preserved(self):
        module = TrainableBezier(shape=(4,), channel_only=True)
        loss = compute_bezier_monotonicity_reg(module, weight=0.0)
        assert loss.dtype == module.p0.dtype
        assert loss.device == module.p0.device
