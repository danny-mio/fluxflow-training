"""Unit tests for training loss functions (src/training/losses.py)."""

import torch
import torch.nn as nn

from fluxflow_training.training.losses import (
    _reduce_logits,
    compute_mmd,
    d_hinge_loss,
    g_hinge_loss,
    kl_standard_normal,
    r1_penalty,
)


class TestReduceLogits:
    """Tests for _reduce_logits helper function."""

    def test_scalar_logits(self):
        """Test with already-scalar logits [B]."""
        logits = torch.randn(4)
        reduced = _reduce_logits(logits)
        assert reduced.shape == (4,)

    def test_2d_logits(self):
        """Test with 2D logits [B, C]."""
        logits = torch.randn(4, 10)
        reduced = _reduce_logits(logits)
        assert reduced.shape == (4,)

    def test_4d_patch_logits(self):
        """Test with 4D patch discriminator logits [B, C, H, W]."""
        logits = torch.randn(4, 1, 8, 8)
        reduced = _reduce_logits(logits)
        assert reduced.shape == (4,)

    def test_reduction_correctness_4d(self):
        """Verify 4D reduction computes mean correctly."""
        logits = torch.ones(2, 3, 4, 5)  # All ones
        reduced = _reduce_logits(logits)
        assert torch.allclose(reduced, torch.ones(2))


class TestDHingeLoss:
    """Tests for discriminator hinge loss."""

    def test_perfect_discriminator(self):
        """Loss should be 0 when real > 1 and fake < -1."""
        real_logits = torch.tensor([2.0, 3.0, 1.5])
        fake_logits = torch.tensor([-2.0, -3.0, -1.5])
        loss = d_hinge_loss(real_logits, fake_logits)
        assert loss.item() == 0.0

    def test_bad_discriminator(self):
        """Loss should be positive when boundaries violated."""
        real_logits = torch.tensor([0.5, -0.5])  # Should be > 1
        fake_logits = torch.tensor([0.5, 1.5])  # Should be < -1
        loss = d_hinge_loss(real_logits, fake_logits)
        assert loss.item() > 0.0

    def test_output_scalar(self):
        """Output should be scalar."""
        real_logits = torch.randn(16)
        fake_logits = torch.randn(16)
        loss = d_hinge_loss(real_logits, fake_logits)
        assert loss.dim() == 0

    def test_non_negative(self):
        """Hinge loss should always be non-negative."""
        for _ in range(10):
            real_logits = torch.randn(8)
            fake_logits = torch.randn(8)
            loss = d_hinge_loss(real_logits, fake_logits)
            assert loss.item() >= 0.0

    def test_4d_patch_inputs(self):
        """Should work with 4D patch discriminator outputs."""
        real_logits = torch.randn(4, 1, 8, 8)
        fake_logits = torch.randn(4, 1, 8, 8)
        loss = d_hinge_loss(real_logits, fake_logits)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        """Verify gradients can flow through loss."""
        real_logits = torch.randn(4, requires_grad=True)
        fake_logits = torch.randn(4, requires_grad=True)
        loss = d_hinge_loss(real_logits, fake_logits)
        loss.backward()
        assert real_logits.grad is not None
        assert fake_logits.grad is not None


class TestGHingeLoss:
    """Tests for generator hinge loss."""

    def test_good_generator(self):
        """Loss should be small when fake_logits > 0."""
        fake_logits = torch.tensor([1.0, 2.0, 3.0])
        loss = g_hinge_loss(fake_logits)
        # Loss is negative mean, so higher fake_logits = more negative loss
        assert loss.item() < 0.0

    def test_bad_generator(self):
        """Loss should be positive when fake_logits < 0."""
        fake_logits = torch.tensor([-1.0, -2.0, -3.0])
        loss = g_hinge_loss(fake_logits)
        assert loss.item() > 0.0

    def test_output_scalar(self):
        """Output should be scalar."""
        fake_logits = torch.randn(16)
        loss = g_hinge_loss(fake_logits)
        assert loss.dim() == 0

    def test_opposite_sign_of_mean(self):
        """Loss should be negative of mean fake_logits."""
        fake_logits = torch.tensor([1.0, 2.0, 3.0])
        loss = g_hinge_loss(fake_logits)
        expected = -fake_logits.mean()
        assert torch.allclose(loss, expected)

    def test_4d_patch_inputs(self):
        """Should work with 4D patch discriminator outputs."""
        fake_logits = torch.randn(4, 1, 8, 8)
        loss = g_hinge_loss(fake_logits)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        """Verify gradients can flow through loss."""
        fake_logits = torch.randn(4, requires_grad=True)
        loss = g_hinge_loss(fake_logits)
        loss.backward()
        assert fake_logits.grad is not None


class TestR1Penalty:
    """Tests for R1 gradient penalty."""

    def test_output_scalar(self):
        """Output should be scalar."""
        real_imgs = torch.randn(2, 3, 64, 64, requires_grad=True)
        # Create a simple discriminator to establish computational graph
        discriminator = nn.Linear(3 * 64 * 64, 1)
        d_out = discriminator(real_imgs.view(2, -1))
        penalty = r1_penalty(real_imgs, d_out)
        assert penalty.dim() == 0

    def test_zero_gradients_zero_penalty(self):
        """Near-zero gradients should give small penalty."""
        # Create images and discriminator output
        real_imgs = torch.randn(2, 3, 4, 4, requires_grad=True)
        # Use a very small weight discriminator for near-zero gradients
        discriminator = nn.Linear(3 * 4 * 4, 1)
        with torch.no_grad():
            discriminator.weight.fill_(0.001)
            discriminator.bias.fill_(0.0)
        d_out = discriminator(real_imgs.view(2, -1))
        penalty = r1_penalty(real_imgs, d_out)
        # With tiny gradients, penalty should be small
        assert torch.isfinite(penalty)
        assert penalty.item() < 1.0

    def test_non_negative(self):
        """R1 penalty should always be non-negative (squared gradients)."""
        real_imgs = torch.randn(2, 3, 32, 32, requires_grad=True)
        discriminator = nn.Linear(3 * 32 * 32, 1)
        d_out = discriminator(real_imgs.view(2, -1))
        penalty = r1_penalty(real_imgs, d_out)
        assert penalty.item() >= 0.0

    def test_4d_discriminator_output(self):
        """Should work with 4D patch discriminator outputs."""
        real_imgs = torch.randn(2, 3, 64, 64, requires_grad=True)
        # Create a conv discriminator for 4D output
        discriminator = nn.Conv2d(3, 1, kernel_size=1)
        d_out = discriminator(real_imgs)
        penalty = r1_penalty(real_imgs, d_out)
        assert penalty.dim() == 0
        assert torch.isfinite(penalty)

    def test_requires_grad(self):
        """Should fail gracefully if inputs don't require grad."""
        real_imgs = torch.randn(2, 3, 4, 4)  # No requires_grad
        d_out = torch.randn(2, 1)
        # This should not crash, autograd will handle it
        try:
            penalty = r1_penalty(real_imgs, d_out)
        except RuntimeError:
            # Expected if gradients can't be computed
            pass


class TestKLStandardNormal:
    """Tests for KL divergence to standard normal."""

    def test_zero_mean_unit_var_gives_zero_kl(self):
        """KL(N(0,1) || N(0,1)) should be 0."""
        mu = torch.zeros(2, 4, 8, 8)
        logvar = torch.zeros(2, 4, 8, 8)
        kl = kl_standard_normal(mu, logvar)
        assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_non_zero_mean_positive_kl(self):
        """Non-zero mean should give positive KL."""
        mu = torch.ones(2, 4, 8, 8) * 2.0
        logvar = torch.zeros(2, 4, 8, 8)
        kl = kl_standard_normal(mu, logvar)
        assert kl.item() > 0.0

    def test_output_scalar_with_mean_reduce(self):
        """With reduce='mean', output should be scalar."""
        mu = torch.randn(2, 4, 8, 8)
        logvar = torch.randn(2, 4, 8, 8)
        kl = kl_standard_normal(mu, logvar, reduce="mean")
        assert kl.dim() == 0

    def test_output_batch_with_no_reduce(self):
        """Without reduce, output should be per-sample."""
        mu = torch.randn(2, 4, 8, 8)
        logvar = torch.randn(2, 4, 8, 8)
        kl = kl_standard_normal(mu, logvar, reduce="none")
        assert kl.shape == (2,)

    def test_sum_vs_mean_reduce(self):
        """Sum reduce should be batch_size * mean reduce."""
        mu = torch.randn(4, 2, 8, 8)
        logvar = torch.randn(4, 2, 8, 8)
        kl_mean = kl_standard_normal(mu, logvar, reduce="mean")
        kl_sum = kl_standard_normal(mu, logvar, reduce="sum")
        assert torch.allclose(kl_sum, kl_mean * 4, rtol=1e-4)

    def test_free_bits_constraint(self):
        """Free bits should clamp KL per dimension."""
        mu = torch.zeros(1, 1, 2, 2)
        logvar = torch.zeros(1, 1, 2, 2)

        # Without free bits, KL should be 0
        kl_no_fb = kl_standard_normal(mu, logvar, free_bits_nats=0.0)
        assert torch.allclose(kl_no_fb, torch.tensor(0.0), atol=1e-6)

        # With free bits, KL should be at least free_bits * num_elements
        free_bits = 1.0
        kl_fb = kl_standard_normal(mu, logvar, free_bits_nats=free_bits)
        assert kl_fb.item() >= free_bits * 4  # 1x1x2x2 = 4 elements

    def test_gradient_flow(self):
        """Verify gradients can flow through KL loss."""
        mu = torch.randn(2, 4, 8, 8, requires_grad=True)
        logvar = torch.randn(2, 4, 8, 8, requires_grad=True)
        kl = kl_standard_normal(mu, logvar)
        kl.backward()
        assert mu.grad is not None
        assert logvar.grad is not None


class TestComputeMMD:
    """Tests for Maximum Mean Discrepancy."""

    def test_identical_distributions_zero_mmd(self):
        """MMD between identical distributions should be near 0."""
        z = torch.randn(100, 64)
        mmd = compute_mmd(z, z)
        assert torch.allclose(mmd, torch.tensor(0.0), atol=1e-5)

    def test_different_distributions_positive_mmd(self):
        """MMD between different distributions should be positive."""
        z1 = torch.randn(100, 64)
        z2 = torch.randn(100, 64) + 2.0  # Shifted distribution
        mmd = compute_mmd(z1, z2)
        assert mmd.item() > 0.0

    def test_output_scalar(self):
        """Output should be scalar."""
        z = torch.randn(50, 32)
        z_prior = torch.randn(50, 32)
        mmd = compute_mmd(z, z_prior)
        assert mmd.dim() == 0

    def test_symmetric(self):
        """MMD should be symmetric: MMD(z1, z2) = MMD(z2, z1)."""
        z1 = torch.randn(50, 32)
        z2 = torch.randn(50, 32)
        mmd_12 = compute_mmd(z1, z2)
        mmd_21 = compute_mmd(z2, z1)
        assert torch.allclose(mmd_12, mmd_21)

    def test_different_sigma(self):
        """Different sigma values should give different MMD values."""
        z1 = torch.randn(50, 32)
        z2 = torch.randn(50, 32) + 1.0
        mmd_s1 = compute_mmd(z1, z2, sigma=1.0)
        mmd_s2 = compute_mmd(z1, z2, sigma=2.0)
        assert not torch.allclose(mmd_s1, mmd_s2)

    def test_4d_input(self):
        """Should work with 4D tensors (flattened internally)."""
        z = torch.randn(10, 3, 8, 8)
        z_prior = torch.randn(10, 3, 8, 8)
        mmd = compute_mmd(z, z_prior)
        assert mmd.dim() == 0
        assert torch.isfinite(mmd)

    def test_gradient_flow(self):
        """Verify gradients can flow through MMD loss."""
        z = torch.randn(20, 16, requires_grad=True)
        z_prior = torch.randn(20, 16)
        mmd = compute_mmd(z, z_prior)
        mmd.backward()
        assert z.grad is not None


class TestLossesIntegration:
    """Integration tests combining multiple loss functions."""

    def test_gan_training_losses(self):
        """Test typical GAN training scenario."""
        # Create fake discriminator outputs
        real_logits = torch.randn(8)
        fake_logits_d = torch.randn(8)
        fake_logits_g = torch.randn(8)

        # Discriminator loss
        d_loss = d_hinge_loss(real_logits, fake_logits_d)
        assert torch.isfinite(d_loss)

        # Generator loss
        g_loss = g_hinge_loss(fake_logits_g)
        assert torch.isfinite(g_loss)

    def test_vae_training_losses(self):
        """Test typical VAE training scenario."""
        # KL divergence
        mu = torch.randn(4, 64, 16, 16)
        logvar = torch.randn(4, 64, 16, 16)
        kl_loss = kl_standard_normal(mu, logvar, free_bits_nats=0.5)
        assert torch.isfinite(kl_loss)

        # MMD (alternative to KL)
        z = torch.randn(32, 256)
        z_prior = torch.randn(32, 256)
        mmd_loss = compute_mmd(z, z_prior)
        assert torch.isfinite(mmd_loss)

    def test_combined_losses_gradient_flow(self):
        """Test that gradients flow through combined losses."""
        # Setup
        mu = torch.randn(2, 4, 8, 8, requires_grad=True)
        logvar = torch.randn(2, 4, 8, 8, requires_grad=True)
        fake_logits = torch.randn(2, requires_grad=True)

        # Combined loss
        kl_loss = kl_standard_normal(mu, logvar)
        g_loss = g_hinge_loss(fake_logits)
        total_loss = kl_loss + g_loss

        # Backprop
        total_loss.backward()

        # Check gradients
        assert mu.grad is not None
        assert logvar.grad is not None
        assert fake_logits.grad is not None
