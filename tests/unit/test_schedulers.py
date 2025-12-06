"""Unit tests for training schedulers (src/training/schedulers.py)."""

import math

import pytest
import torch

from fluxflow_training.training.schedulers import cosine_anneal_beta, sample_t


class TestCosineAnnealBeta:
    """Tests for cosine_anneal_beta function."""

    def test_zero_at_start(self):
        """Beta should be 0 at step 0."""
        beta = cosine_anneal_beta(step=0, total_steps=1000, beta_max=0.1)
        assert beta == 0.0

    def test_max_at_end(self):
        """Beta should reach beta_max at total_steps."""
        beta = cosine_anneal_beta(step=1000, total_steps=1000, beta_max=0.1)
        assert abs(beta - 0.1) < 1e-6

    def test_monotonic_increase(self):
        """Beta should monotonically increase."""
        total_steps = 100
        beta_max = 1.0
        betas = [
            cosine_anneal_beta(step=i, total_steps=total_steps, beta_max=beta_max)
            for i in range(total_steps + 1)
        ]

        for i in range(len(betas) - 1):
            assert betas[i] <= betas[i + 1], f"Non-monotonic at step {i}"

    def test_midpoint_value(self):
        """Beta at midpoint should be exactly beta_max/2."""
        beta = cosine_anneal_beta(step=500, total_steps=1000, beta_max=1.0)
        expected = 0.5  # Cosine schedule gives exactly 0.5 at midpoint
        assert abs(beta - expected) < 1e-6

    def test_cosine_curve_shape(self):
        """Verify cosine annealing formula."""
        step = 250
        total_steps = 1000
        beta_max = 2.0

        beta = cosine_anneal_beta(step, total_steps, beta_max)
        frac = step / total_steps
        expected = beta_max * (1 - math.cos(math.pi * frac)) / 2.0

        assert abs(beta - expected) < 1e-6

    def test_different_beta_max(self):
        """Test with various beta_max values."""
        for beta_max in [0.01, 0.1, 1.0, 10.0]:
            beta = cosine_anneal_beta(step=1000, total_steps=1000, beta_max=beta_max)
            assert abs(beta - beta_max) < 1e-6

    def test_zero_total_steps(self):
        """When total_steps=0, should return beta_max immediately."""
        beta = cosine_anneal_beta(step=0, total_steps=0, beta_max=5.0)
        assert beta == 5.0

    def test_step_beyond_total(self):
        """Step beyond total_steps should clamp to beta_max."""
        beta = cosine_anneal_beta(step=2000, total_steps=1000, beta_max=0.5)
        assert abs(beta - 0.5) < 1e-6

    def test_negative_step_clamps_to_zero(self):
        """Negative step should be clamped to 0."""
        beta = cosine_anneal_beta(step=-100, total_steps=1000, beta_max=1.0)
        assert beta == 0.0


class TestSampleT:
    """Tests for sample_t function."""

    def test_output_shape(self):
        """Output should have correct batch size."""
        batch_size = 16
        timesteps = sample_t(batch_size, device=torch.device("cpu"))
        assert timesteps.shape == (batch_size,)

    def test_output_range(self):
        """All timesteps should be in [0, 999]."""
        timesteps = sample_t(100, device=torch.device("cpu"))
        assert (timesteps >= 0).all()
        assert (timesteps <= 999).all()

    def test_output_dtype(self):
        """Output should be long tensor (integer indices)."""
        timesteps = sample_t(10, device=torch.device("cpu"))
        assert timesteps.dtype == torch.long

    def test_device_placement(self):
        """Output should be on specified device."""
        device = torch.device("cpu")
        timesteps = sample_t(10, device=device)
        assert timesteps.device == device

    @pytest.mark.gpu
    def test_gpu_device_placement(self):
        """Output should be on GPU when specified."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            timesteps = sample_t(10, device=device)
            assert timesteps.device.type == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_sampling_distribution(self):
        """Verify cosine-weighted distribution properties."""
        # Sample many timesteps - use larger sample for more stable statistics
        n_samples = 50000
        timesteps = sample_t(n_samples, device=torch.device("cpu"))

        # Convert to histogram
        hist = torch.histc(timesteps.float(), bins=10, min=0, max=999)

        # The cosine weighting: cos((s + 0.008) / 1.008 * pi/2)^2
        # This is a decreasing function from s=0 to s=1, so it favors EARLY timesteps
        # Check that early bins have significantly more samples than late bins
        early_bins = hist[:3].sum()  # bins 0-2 (timesteps 0-299)
        late_bins = hist[7:].sum()  # bins 7-9 (timesteps 700-999)

        assert early_bins > late_bins * 2.0, (
            f"Cosine weighting should favor early timesteps: "
            f"early={early_bins}, late={late_bins}"
        )

    def test_determinism_not_guaranteed(self):
        """Successive calls should produce different samples (stochastic)."""
        t1 = sample_t(100, device=torch.device("cpu"))
        t2 = sample_t(100, device=torch.device("cpu"))

        # Should not be identical (with high probability)
        assert not torch.equal(t1, t2), "Successive calls should produce different samples"

    def test_covers_full_range(self):
        """With enough samples, should cover most of [0, 999] range."""
        # Sample many times
        all_timesteps = []
        for _ in range(100):
            t = sample_t(100, device=torch.device("cpu"))
            all_timesteps.append(t)

        all_t = torch.cat(all_timesteps)
        unique_t = torch.unique(all_t)

        # Should cover at least 80% of possible timesteps
        coverage = len(unique_t) / 1000
        assert coverage > 0.8, f"Only covered {coverage*100:.1f}% of timesteps"

    def test_batch_size_one(self):
        """Should work with batch_size=1."""
        timesteps = sample_t(1, device=torch.device("cpu"))
        assert timesteps.shape == (1,)
        assert 0 <= timesteps.item() <= 999

    def test_large_batch_size(self):
        """Should work with large batch sizes."""
        timesteps = sample_t(10000, device=torch.device("cpu"))
        assert timesteps.shape == (10000,)
        assert (timesteps >= 0).all() and (timesteps <= 999).all()


class TestSchedulersIntegration:
    """Integration tests combining both scheduler functions."""

    def test_beta_schedule_with_sampling(self):
        """Test typical usage: beta scheduling with timestep sampling."""
        total_steps = 1000
        beta_max = 0.1

        # Simulate training loop
        for step in range(0, total_steps, 100):
            beta = cosine_anneal_beta(step, total_steps, beta_max)
            timesteps = sample_t(8, device=torch.device("cpu"))

            # Beta should be in valid range
            assert 0 <= beta <= beta_max

            # Timesteps should be valid
            assert (timesteps >= 0).all() and (timesteps <= 999).all()

    def test_warmup_period(self):
        """Verify beta stays small during warmup."""
        warmup_steps = 100
        total_steps = 1000
        beta_max = 1.0

        # During first 10% of training, beta should be < 0.1
        for step in range(warmup_steps):
            beta = cosine_anneal_beta(step, total_steps, beta_max)
            assert beta < 0.1, f"Beta too high during warmup at step {step}"
