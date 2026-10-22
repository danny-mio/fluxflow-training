"""Tests for VAE trainer bug fixes.

Covers four bugs identified in systematic debugging:
1. add_instance_noise: inverted requires_grad guard means noise is never added
2. (removed) loss-history adaptive weighting
3. _train_discriminator: uses training=False (deterministic) vs training=True in generator
4. Dead code block after raise in _get_effective_spade_usage
"""

import pytest
import torch

from fluxflow_training.training.vae_trainer import VAETrainer, add_instance_noise


class TestAddInstanceNoise:
    """Bug 1: add_instance_noise guard is inverted - noise is never added."""

    def test_noise_added_to_tensor_without_requires_grad(self):
        """Discriminator inputs (detached/raw) must receive noise.

        real_imgs and out_imgs_for_D.detach() both have requires_grad=False.
        With the old inverted guard, noise was never applied to either.
        """
        torch.manual_seed(42)
        x = torch.ones(2, 3, 8, 8)  # requires_grad=False (raw batch)
        assert not x.requires_grad

        noisy = add_instance_noise(x, noise_std=0.1, step=0)

        # Must differ from original - noise was added
        assert not torch.allclose(x, noisy), (
            "add_instance_noise returned input unchanged for requires_grad=False tensor. "
            "The old `if not x.requires_grad: return x` guard is wrong."
        )

    def test_noise_added_to_detached_tensor(self):
        """Detached fake images (out_imgs_for_D.detach()) must receive noise."""
        torch.manual_seed(0)
        base = torch.randn(2, 3, 8, 8, requires_grad=True)
        detached = base.detach()
        assert not detached.requires_grad

        noisy = add_instance_noise(detached, noise_std=0.1, step=0)

        assert not torch.allclose(
            detached, noisy
        ), "No noise added to detached tensor. Discriminator inputs will never be noised."

    def test_noise_magnitude_matches_std(self):
        """Added noise should have approximately the requested std."""
        torch.manual_seed(1)
        x = torch.zeros(100, 3, 32, 32)  # Large batch for accurate stats
        noisy = add_instance_noise(x, noise_std=0.1, step=0)
        diff = noisy - x
        actual_std = diff.std().item()
        assert abs(actual_std - 0.1) < 0.02, f"Noise std {actual_std:.4f} far from expected 0.1"

    def test_noise_decays_over_steps(self):
        """Noise should decay as step increases."""
        torch.manual_seed(2)
        x = torch.zeros(10, 3, 8, 8)
        noisy_early = add_instance_noise(x, noise_std=0.1, decay_rate=0.999, step=0)
        noisy_late = add_instance_noise(x, noise_std=0.1, decay_rate=0.999, step=10000)
        diff_early = (noisy_early - x).abs().mean().item()
        diff_late = (noisy_late - x).abs().mean().item()
        assert diff_early > diff_late, "Noise should decay over steps"


class TestDiscriminatorReusesGeneratorReconstruction:
    """D trains on the G step's (stochastic, training=True) reconstruction, so the fake
    distribution matches the generator's by construction and no extra forward runs."""

    def test_discriminator_step_runs_no_vae_forward(self):
        import inspect

        source = inspect.getsource(VAETrainer._train_discriminator)
        assert "self.compressor(" not in source
        assert "self.expander(" not in source

    def test_generator_exposes_reconstruction_for_discriminator(self):
        import inspect

        assert "_fake_imgs" in inspect.getsource(VAETrainer._train_generator)


class TestRegularizerLosses:
    def test_contrast_bounded_on_flat_target(self):
        pred = torch.randn(2, 3, 16, 16) * 0.5
        flat = torch.zeros(2, 3, 16, 16)
        loss = VAETrainer._contrast_regularization_loss(None, pred, flat)
        assert torch.isfinite(loss) and loss.item() < 10.0

    def test_contrast_zero_for_identical(self):
        x = torch.randn(2, 3, 16, 16)
        assert VAETrainer._contrast_regularization_loss(None, x, x).item() == pytest.approx(0.0)

    def test_contrast_has_gradient(self):
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        VAETrainer._contrast_regularization_loss(None, pred, torch.randn(2, 3, 16, 16)).backward()
        assert pred.grad.abs().sum() > 0

    def test_histogram_gradient_nonzero(self):
        pred = (torch.rand(2, 3, 16, 16) * 2 - 1).requires_grad_(True)
        target = torch.rand(2, 3, 16, 16) * 0.2
        loss = VAETrainer._histogram_matching_loss(None, pred, target)
        loss.backward()
        assert loss.item() > 0 and pred.grad.abs().sum() > 0

    def test_histogram_zero_for_identical(self):
        x = torch.rand(2, 3, 8, 8) * 2 - 1
        assert VAETrainer._histogram_matching_loss(None, x, x).item() == pytest.approx(0.0)
