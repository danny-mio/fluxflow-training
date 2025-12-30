"""Integration tests for GAN discriminator training."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from fluxflow.models.discriminators import PatchDiscriminator

from fluxflow_training.training.losses import d_hinge_loss, g_hinge_loss, r1_penalty


class TestGANTrainingIntegration:
    """Integration tests for GAN training workflow."""

    @pytest.fixture
    def small_discriminator(self):
        """Create small discriminator for testing."""
        disc = PatchDiscriminator(in_channels=3, base_ch=32, depth=3, ctx_dim=0)
        return disc

    @pytest.fixture
    def small_generator(self):
        """Create simple generator for testing."""
        # Simple upsampling generator
        gen = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh(),
        )
        return gen

    def test_discriminator_forward_backward(self, small_discriminator):
        """Discriminator should support forward and backward."""
        disc = small_discriminator

        images = torch.randn(4, 3, 64, 64)
        logits = disc(images)

        # Compute loss
        loss = logits.mean()
        loss.backward()

        # Check gradients
        assert any(p.grad is not None for p in disc.parameters() if p.requires_grad)

    def test_gan_discriminator_training_step(self, small_discriminator, small_generator):
        """Complete discriminator training step."""
        disc = small_discriminator
        gen = small_generator

        d_optimizer = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

        disc.train()
        gen.eval()

        # Real images
        real_images = torch.randn(4, 3, 64, 64)

        # Generate fake images
        noise = torch.randn(4, 64, 8, 8)
        with torch.no_grad():
            fake_images = gen(noise)

        # Discriminate
        real_logits = disc(real_images)
        fake_logits = disc(fake_images.detach())

        # Hinge loss
        d_loss = d_hinge_loss(real_logits, fake_logits)

        # Optimize
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        assert d_loss.item() >= 0

    def test_gan_generator_training_step(self, small_discriminator, small_generator):
        """Complete generator training step."""
        disc = small_discriminator
        gen = small_generator

        g_optimizer = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

        disc.eval()
        gen.train()

        # Generate fake images
        noise = torch.randn(4, 64, 8, 8)
        fake_images = gen(noise)

        # Discriminate
        fake_logits = disc(fake_images)

        # Generator loss (wants discriminator to think fakes are real)
        g_loss = g_hinge_loss(fake_logits)

        # Optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # G hinge loss = -mean(fake_logits), which can be any value
        # Just verify it's a valid finite number
        assert torch.isfinite(g_loss)

    def test_gan_alternating_training(self, small_discriminator, small_generator):
        """GAN should support alternating D and G updates."""
        disc = small_discriminator
        gen = small_generator

        d_optimizer = optim.Adam(disc.parameters(), lr=2e-4)
        g_optimizer = optim.Adam(gen.parameters(), lr=2e-4)

        for step in range(3):
            # Train discriminator
            disc.train()
            gen.eval()

            real_images = torch.randn(2, 3, 64, 64)
            noise = torch.randn(2, 64, 8, 8)

            with torch.no_grad():
                fake_images = gen(noise)

            real_logits = disc(real_images)
            fake_logits = disc(fake_images)

            d_loss = d_hinge_loss(real_logits, fake_logits)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            disc.eval()
            gen.train()

            noise = torch.randn(2, 64, 8, 8)
            fake_images = gen(noise)
            fake_logits = disc(fake_images)

            g_loss = g_hinge_loss(fake_logits)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # Should complete without errors
        assert True

    def test_gan_with_r1_penalty(self, small_discriminator):
        """Discriminator training with R1 gradient penalty."""
        disc = small_discriminator
        d_optimizer = optim.Adam(disc.parameters(), lr=2e-4)

        disc.train()

        # Real images (requires grad for R1)
        real_images = torch.randn(4, 3, 64, 64, requires_grad=True)
        fake_images = torch.randn(4, 3, 64, 64)

        # Discriminate
        real_logits = disc(real_images)
        fake_logits = disc(fake_images.detach())

        # Compute R1 penalty (swap argument order: real_imgs first, then d_out)
        r1_loss = r1_penalty(real_images, real_logits)

        # Total discriminator loss
        d_loss = d_hinge_loss(real_logits, fake_logits) + 10.0 * r1_loss

        # Optimize
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        assert r1_loss.item() >= 0

    def test_gan_conditional_with_projection(self):
        """Conditional GAN with projection discriminator."""
        # Discriminator with projection conditioning
        disc = PatchDiscriminator(in_channels=3, base_ch=32, depth=3, ctx_dim=128)
        d_optimizer = optim.Adam(disc.parameters(), lr=2e-4)

        disc.train()

        # Real images and conditioning
        real_images = torch.randn(4, 3, 64, 64)
        real_ctx = torch.randn(4, 128)

        # Fake images with mismatched conditioning
        fake_images = torch.randn(4, 3, 64, 64)
        fake_ctx = torch.randn(4, 128)

        # Discriminate with conditioning
        real_logits = disc(real_images, ctx_vec=real_ctx)
        fake_logits = disc(fake_images, ctx_vec=fake_ctx)

        # Hinge loss
        d_loss = d_hinge_loss(real_logits, fake_logits)

        # Optimize
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        assert d_loss.item() >= 0

    def test_gan_discriminator_eval_mode(self, small_discriminator):
        """Discriminator in eval mode should be deterministic."""
        disc = small_discriminator
        disc.eval()

        images = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            logits1 = disc(images)
            logits2 = disc(images)

        assert torch.allclose(logits1, logits2, atol=1e-5)

    def test_gan_gradient_accumulation(self, small_discriminator):
        """GAN training should support gradient accumulation."""
        disc = small_discriminator
        d_optimizer = optim.Adam(disc.parameters(), lr=2e-4)

        disc.train()

        # Accumulate over 2 mini-batches
        d_optimizer.zero_grad()

        for i in range(2):
            real_images = torch.randn(2, 3, 64, 64)
            fake_images = torch.randn(2, 3, 64, 64)
            real_logits = disc(real_images)
            fake_logits = disc(fake_images)
            d_loss = d_hinge_loss(real_logits, fake_logits)

            (d_loss / 2.0).backward()

        d_optimizer.step()

        assert True

    def test_gan_checkpoint_compatibility(self, small_discriminator):
        """Discriminator state_dict should be saveable."""
        disc = small_discriminator

        # Save state
        state_dict = disc.state_dict()

        # Create new discriminator
        new_disc = PatchDiscriminator(in_channels=3, base_ch=32, depth=3, ctx_dim=0)

        # Load state
        new_disc.load_state_dict(state_dict)

        # Should produce same output
        disc.eval()
        new_disc.eval()

        images = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            logits1 = disc(images)
            logits2 = new_disc(images)

        assert torch.allclose(logits1, logits2, atol=1e-5)


class TestGANLossesIntegration:
    """Tests for GAN loss functions in training."""

    def test_hinge_loss_discriminator(self):
        """Hinge loss should penalize incorrect discriminations."""
        # Real images should have positive logits, fake should have negative
        real_logits = torch.tensor([[1.5], [2.0]])
        fake_logits = torch.tensor([[-1.5], [-2.0]])

        # Combined hinge loss
        loss = d_hinge_loss(real_logits, fake_logits)

        # Loss should be small for correct predictions (real>1, fake<-1)
        assert loss.item() < 0.1

    def test_hinge_loss_generator(self):
        """Generator hinge loss should maximize discriminator score."""
        # Generator wants discriminator to output positive logits for fakes
        fake_logits = torch.tensor([[1.5], [2.0]])
        g_loss = g_hinge_loss(fake_logits)

        # Loss should be small when discriminator is fooled
        assert g_loss.item() < 0.1

    def test_r1_penalty_computation(self):
        """R1 penalty should compute gradient penalty."""
        # Create images that require gradients
        images = torch.randn(4, 3, 64, 64, requires_grad=True)

        # Simple discriminator output
        logits = (images**2).sum(dim=[1, 2, 3], keepdim=True)

        # Compute R1 penalty (real_imgs first, then d_out)
        penalty = r1_penalty(images, logits)

        # Penalty should be non-negative
        assert penalty.item() >= 0


class TestGANMultiScaleIntegration:
    """Tests for multi-scale GAN training."""

    def test_discriminator_different_resolutions(self):
        """Discriminator should handle different resolutions."""
        disc = PatchDiscriminator(in_channels=3, base_ch=32, depth=4, ctx_dim=0)
        disc.eval()

        # Different resolutions
        for size in [64, 128, 256]:
            images = torch.randn(2, 3, size, size)

            with torch.no_grad():
                logits = disc(images)

            # Should produce patch logits
            assert logits.dim() == 4
            assert logits.shape[0] == 2
            assert logits.shape[1] == 1

    def test_multi_scale_discrimination(self):
        """Multi-scale GAN with different discriminator depths."""
        # Shallow discriminator (larger patches)
        disc_shallow = PatchDiscriminator(in_channels=3, base_ch=32, depth=2, ctx_dim=0)

        # Deep discriminator (smaller patches)
        disc_deep = PatchDiscriminator(in_channels=3, base_ch=32, depth=4, ctx_dim=0)

        disc_shallow.eval()
        disc_deep.eval()

        images = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            logits_shallow = disc_shallow(images)
            logits_deep = disc_deep(images)

        # Deep discriminator should have smaller spatial dimensions
        assert logits_deep.shape[2] <= logits_shallow.shape[2]
        assert logits_deep.shape[3] <= logits_shallow.shape[3]


class TestGANStabilityTechniques:
    """Tests for GAN training stability techniques."""

    def test_spectral_normalization_active(self):
        """Discriminator should use spectral normalization."""
        disc = PatchDiscriminator(
            in_channels=3, base_ch=32, depth=3, ctx_dim=0, use_spectral_norm=True
        )

        # Check for spectral norm
        has_spectral_norm = False
        for module in disc.modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "weight_u"):
                    has_spectral_norm = True
                    break

        assert has_spectral_norm

    def test_gradient_clipping(self):
        """GAN training should support gradient clipping."""
        disc = PatchDiscriminator(in_channels=3, base_ch=32, depth=3, ctx_dim=0)
        d_optimizer = optim.Adam(disc.parameters(), lr=2e-4)

        disc.train()

        real_images = torch.randn(4, 3, 64, 64)
        fake_images = torch.randn(4, 3, 64, 64)
        real_logits = disc(real_images)
        fake_logits = disc(fake_images)
        loss = d_hinge_loss(real_logits, fake_logits)

        d_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)

        d_optimizer.step()

        # Should complete without errors
        assert True

    def test_two_timescale_update_rule(self):
        """TTUR: Different learning rates for D and G."""
        disc = PatchDiscriminator(in_channels=3, base_ch=32, depth=3, ctx_dim=0)

        gen = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

        # TTUR: D learning rate > G learning rate
        d_optimizer = optim.Adam(disc.parameters(), lr=4e-4)
        g_optimizer = optim.Adam(gen.parameters(), lr=1e-4)

        # Training step
        disc.train()
        gen.train()

        noise = torch.randn(2, 64, 16, 16)
        fake_images = gen(noise)
        fake_logits = disc(fake_images)

        g_loss = g_hinge_loss(fake_logits)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        assert True
