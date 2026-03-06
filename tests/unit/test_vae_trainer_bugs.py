"""Tests for VAE trainer bug fixes.

Covers four bugs identified in systematic debugging:
1. add_instance_noise: inverted requires_grad guard means noise is never added
2. _compute_adaptive_weight: unbounded inverse weighting causes explosive GAN gradients
3. _train_discriminator: uses training=False (deterministic) vs training=True in generator
4. Dead code block after raise in _get_effective_spade_usage
"""

import torch

from fluxflow_training.training.vae_trainer import (
    VAETrainer,
    add_instance_noise,
)
from fluxflow_training.training.utils import FloatBuffer


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


class TestAdaptiveWeightBounds:
    """Bug 2: _compute_adaptive_weight returns unbounded values, causing explosive GAN gradients.

    When GAN loss starts very small (e.g. 0.001) and recon loss is larger (e.g. 0.3),
    w_gan = target / 0.001 can reach 300x, overwhelming reconstruction signal.
    """

    def _make_loss_history(self, recon=0.1, kl=1.0, gan=0.001):
        """Build a loss_history dict mimicking early GAN finetuning."""
        history = {
            "recon": FloatBuffer(100),
            "kl": FloatBuffer(100),
            "gan": FloatBuffer(100),
            "lpips": FloatBuffer(100),
        }
        for _ in range(10):
            history["recon"].add_item(recon)
            history["kl"].add_item(kl)
            history["gan"].add_item(gan)
        return history

    def _compute_weight(self, history, loss_type):
        """Call the real _compute_adaptive_weight via a minimal stub object."""
        # Build a minimal object with the two attributes the method uses
        stub = type(
            "_Stub",
            (),
            {
                "adaptive_weights": True,
                "loss_history": history,
            },
        )()
        return VAETrainer._compute_adaptive_weight(stub, loss_type)

    def test_gan_weight_bounded_at_startup(self):
        """w_gan must not spike to 300x when GAN loss transitions from 0 to small positive.

        In early GAN finetuning: recon~0.1, kl~1.0, gan~0.001
        Old formula: w_gan = target/0.001 ≈ 37-300x
        Expected: w_gan <= reasonable maximum (e.g. 10)
        """
        history = self._make_loss_history(recon=0.1, kl=1.0, gan=0.001)
        w_gan = self._compute_weight(history, "gan")
        assert w_gan <= 10.0, (
            f"w_gan = {w_gan:.1f} is dangerously large. "
            "At startup the GAN loss is tiny (0.001), causing the adaptive weight to spike "
            "and overwhelm the reconstruction loss with ~300x amplified GAN gradients."
        )

    def test_all_weights_bounded(self):
        """No adaptive weight should exceed 10x regardless of loss magnitudes."""
        history = self._make_loss_history(recon=0.001, kl=10.0, gan=0.0001)
        for key in ["recon", "kl", "gan"]:
            w = self._compute_weight(history, key)
            assert w <= 10.0, (
                f"weight '{key}' = {w:.1f} exceeds bound of 10. "
                "Unbounded adaptive weights destabilize GAN training."
            )

    def test_weights_still_scale_with_relative_magnitude(self):
        """Smaller losses should still receive larger weights (just bounded)."""
        history = self._make_loss_history(recon=0.01, kl=1.0, gan=0.5)
        w_recon = self._compute_weight(history, "recon")
        w_kl = self._compute_weight(history, "kl")
        # recon is smallest, so it should have the largest weight
        assert w_recon >= w_kl, "Smaller recon loss should have larger weight"


class TestDiscriminatorUsesStochasticLatents:
    """Bug 3: discriminator training calls compressor with training=False (deterministic μ path),
    while generator uses training=True (stochastic μ+ε·σ path). This creates a distribution
    mismatch: discriminator never sees stochastic reconstructions as fake images.

    We verify this by checking the _train_discriminator source text uses training=True.
    """

    @staticmethod
    def _source():
        import pathlib

        return pathlib.Path(
            "/Volumes/DanieleExt/ai/ffnew/fluxflow-training/src/"
            "fluxflow_training/training/vae_trainer.py"
        ).read_text()

    def test_discriminator_compressor_called_with_training_true(self):
        """Discriminator must call compressor(real_imgs, training=True) not training=False.

        With training=False the encoder returns the deterministic mean (μ).
        With training=True it returns a reparameterized sample (μ + ε·σ).
        Discriminator training on deterministic reconstructions while the generator
        trains on stochastic ones creates a persistent distribution mismatch.
        """
        source = self._source()
        assert "compressor(real_imgs, training=False)" not in source, (
            "_train_discriminator still calls compressor(real_imgs, training=False). "
            "This creates a deterministic/stochastic distribution mismatch with the "
            "generator training path which uses training=True."
        )
        # training=True returns (packed, mu, logvar) — must unpack the tuple
        assert "packed, _, _ = self.compressor(real_imgs, training=True)" in source, (
            "_train_discriminator must unpack the (packed, mu, logvar) tuple returned by "
            "compressor(..., training=True), otherwise slicing packed will raise TypeError."
        )


class TestNoDeadCodeAfterRaise:
    """Bug 4: unreachable code block after raise in _get_effective_spade_usage.

    Lines 439-470 in vae_trainer.py contain code that assigns self.use_lpips,
    self.lambda_lpips, self.lpips_fn etc. using undefined variables (suppressed
    with # noqa: F821). These lines are after a raise ValueError and never execute.
    """

    @staticmethod
    def _source():
        import pathlib

        return pathlib.Path(
            "/Volumes/DanieleExt/ai/ffnew/fluxflow-training/src/"
            "fluxflow_training/training/vae_trainer.py"
        ).read_text()

    def test_no_unreachable_code_in_get_effective_spade_usage(self):
        """vae_trainer.py must not contain dead code with noqa: F821 or the old dead block."""
        source = self._source()
        assert "noqa: F821" not in source, (
            "vae_trainer.py still contains `# noqa: F821` markers from the dead code block "
            "after `raise ValueError(...)` in _get_effective_spade_usage."
        )
        assert (
            "# LPIPS perceptual loss\n        self.use_lpips = use_lpips" not in source
        ), "The dead '# LPIPS perceptual loss' block is still present after the raise."
