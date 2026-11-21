"""Unit tests for the cosine warmup helpers in losses (KL, GAN, latent-cycle schedules)."""

from fluxflow_training.training.losses import (
    cosine_warmup_weight,
    delayed_cosine_warmup_weight,
)


class TestCosineWarmupWeight:
    """Cosine warmup helper used by the KL_z schedule."""

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
    """Delayed schedule (GAN, latent cycle): hold at 0 until start_step, then cosine warmup."""

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
