"""VAE training logic for FluxFlow.

Handles VAE (compressor + expander) training with optional GAN discriminator.
"""

from typing import Optional

import torch
import torch.nn as nn
from fluxflow.utils import get_logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from .losses import d_hinge_loss, g_hinge_loss, kl_standard_normal, r1_penalty
from .schedulers import cosine_anneal_beta
from .utils import EMA, FloatBuffer

logger = get_logger(__name__)


def check_for_nan(tensor, name, logger_inst):
    """Check for NaN/Inf values and log warning."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger_inst.warning(f"NaN/Inf detected in {name}")
        return True
    return False


def add_instance_noise(x, noise_std=0.01, decay_rate=0.9999, step=0):
    """Add decaying Gaussian noise to prevent discriminator overfitting."""
    if not x.requires_grad:  # Only during training
        return x

    # Decay noise over training
    current_std = noise_std * (decay_rate**step)
    noise = torch.randn_like(x) * current_std
    return x + noise


def compute_grad_norm(parameters):
    """Compute total gradient norm across parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


class VAETrainer:
    """
    Handles VAE (Variational Autoencoder) training.

    Manages:
    - VAE reconstruction loss (L1 + MSE)
    - KL divergence with beta annealing
    - Optional GAN training with discriminator
    - EMA (Exponential Moving Average) updates

    Example:
        >>> trainer = VAETrainer(
        ...     compressor=compressor,
        ...     expander=expander,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     use_gan=True,
        ...     discriminator=D_img,
        ...     discriminator_optimizer=opt_D,
        ... )
        >>> loss_dict = trainer.train_step(images, global_step)
    """

    def __init__(
        self,
        compressor: nn.Module,
        expander: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        ema: EMA,
        reconstruction_loss_fn: nn.Module,
        reconstruction_loss_min_fn: nn.Module,
        use_spade: bool = True,
        kl_beta: float = 0.0001,
        kl_warmup_steps: int = 5000,
        kl_free_bits: float = 0.0,
        # GAN settings
        use_gan: bool = True,
        discriminator: Optional[nn.Module] = None,
        discriminator_optimizer: Optional[Optimizer] = None,
        discriminator_scheduler: Optional[_LRScheduler] = None,
        lambda_adv: float = 0.5,
        r1_interval: int = 16,
        r1_gamma: float = 5.0,
        gradient_clip_norm: float = 1.0,
        use_lpips: bool = True,
        lambda_lpips: float = 0.1,
        instance_noise_std: float = 0.01,
        instance_noise_decay: float = 0.9999,
        adaptive_weights: bool = True,
        mse_weight: float = 0.1,
        accelerator=None,
    ):
        """
        Initialize VAE trainer.

        Args:
            compressor: VAE encoder
            expander: VAE decoder
            optimizer: VAE optimizer
            scheduler: VAE learning rate scheduler
            ema: EMA for VAE parameters
            reconstruction_loss_fn: L1 loss
            reconstruction_loss_min_fn: MSE loss
            use_spade: Use SPADE conditioning in decoder
            kl_beta: Final KL divergence weight
            kl_warmup_steps: Steps to warmup KL beta
            kl_free_bits: Free bits for KL divergence
            use_gan: Enable GAN training
            discriminator: Discriminator model (required if use_gan=True)
            discriminator_optimizer: Discriminator optimizer
            discriminator_scheduler: Discriminator scheduler
            lambda_adv: GAN adversarial loss weight
            r1_interval: R1 gradient penalty interval
            r1_gamma: R1 penalty strength
            gradient_clip_norm: Gradient clipping norm
            use_lpips: Enable LPIPS perceptual loss (default: True)
            lambda_lpips: LPIPS loss weight (default: 0.1)
            accelerator: Accelerate accelerator instance
        """
        self.compressor = compressor
        self.expander = expander
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema

        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.reconstruction_loss_min_fn = reconstruction_loss_min_fn
        self.use_spade = use_spade

        # KL settings
        self.kl_beta = kl_beta
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_free_bits = kl_free_bits

        # GAN settings
        self.use_gan = use_gan
        if use_gan and discriminator is None:
            raise ValueError("discriminator is required when use_gan=True")

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_scheduler = discriminator_scheduler
        self.lambda_adv = lambda_adv
        self.r1_interval = r1_interval
        self.r1_gamma = r1_gamma

        self.gradient_clip_norm = gradient_clip_norm
        self.instance_noise_std = instance_noise_std
        self.instance_noise_decay = instance_noise_decay
        self.adaptive_weights = adaptive_weights
        self.mse_weight = mse_weight
        self.accelerator = accelerator

        # LPIPS perceptual loss
        self.use_lpips = use_lpips
        self.lambda_lpips = lambda_lpips
        self.lpips_fn = None
        if use_lpips:
            import warnings

            import lpips

            # Suppress all torchvision/lpips deprecation warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.lpips_fn = lpips.LPIPS(net="vgg").eval()
            # Move to device and freeze
            if accelerator:
                self.lpips_fn = accelerator.prepare(self.lpips_fn)
            for param in self.lpips_fn.parameters():
                param.requires_grad = False

        # Metrics buffers
        self.vae_loss_buffer = FloatBuffer(max_items=20)
        self.kl_loss_buffer = FloatBuffer(max_items=20)
        self.d_loss_buffer = FloatBuffer(max_items=20)
        self.g_loss_buffer = FloatBuffer(max_items=20)
        self.lpips_loss_buffer = FloatBuffer(max_items=20)

        # Loss history for adaptive weighting
        self.loss_history = {
            "recon": FloatBuffer(100),
            "kl": FloatBuffer(100),
            "gan": FloatBuffer(100),
        }

    def _frequency_weighted_loss(self, pred, target, alpha=1.0):
        """
        Frequency-aware reconstruction loss emphasizing high-frequency details.

        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            alpha: Weight for high-frequency term (default: 1.0)

        Returns:
            Weighted L1 loss
        """
        import torch.nn.functional as F

        # Low-frequency (blurred version) - use same padding to preserve dimensions
        # kernel_size=3 with padding=1 keeps same dimensions
        pred_lf = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        target_lf = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

        # High-frequency (difference from blurred)
        pred_hf = pred - pred_lf
        target_hf = target - target_lf

        # Separate losses
        loss_lf = F.l1_loss(pred_lf, target_lf)
        loss_hf = F.l1_loss(pred_hf, target_hf)

        return loss_lf + alpha * loss_hf

    def _compute_adaptive_weight(self, loss_type):
        """Balance losses based on magnitude using inverse weighting."""
        if not self.adaptive_weights:
            return 1.0

        avg = self.loss_history[loss_type].average
        if avg == 0:
            return 1.0

        # Compute total average
        total = sum(h.average for h in self.loss_history.values() if h.average > 0)
        num_losses = sum(1 for h in self.loss_history.values() if h.average > 0)

        if total > 0 and num_losses > 0:
            target = total / num_losses
            return target / (avg + 1e-8)
        return 1.0

    def train_step(
        self,
        real_imgs: torch.Tensor,
        global_step: int,
    ) -> dict[str, float]:
        """
        Perform one VAE training step.

        Args:
            real_imgs: Real images [B, C, H, W]
            global_step: Global training step for KL annealing

        Returns:
            Dictionary with loss values
        """
        self.compressor.train()
        self.expander.train()

        losses = {}

        # Train discriminator first if using GAN
        if self.use_gan and self.discriminator is not None:
            d_loss = self._train_discriminator(real_imgs, global_step)
            losses["discriminator"] = d_loss
            self.d_loss_buffer.add_item(d_loss)

        # Train VAE generator
        gen_losses = self._train_generator(real_imgs, global_step)

        losses["vae"] = gen_losses["vae"]  # recon_loss
        losses["recon"] = gen_losses["recon"]  # same as vae
        losses["kl"] = gen_losses["kl"]
        losses["kl_beta"] = cosine_anneal_beta(global_step, self.kl_warmup_steps, self.kl_beta)

        if self.use_gan:
            losses["generator"] = gen_losses["generator"]

        if self.use_lpips:
            losses["lpips"] = gen_losses["lpips"]

        # Update EMA
        self.ema.update()

        # Step schedulers (ReduceLROnPlateau requires metric, others don't)
        total_loss = gen_losses["vae"]  # Use recon_loss for scheduler

        # Get the underlying scheduler (may be wrapped by accelerator)
        base_scheduler = getattr(self.scheduler, "scheduler", self.scheduler)
        if isinstance(base_scheduler, ReduceLROnPlateau):
            self.scheduler.step(total_loss)
        else:
            self.scheduler.step()

        if self.use_gan and self.discriminator_scheduler is not None:
            base_d_scheduler = getattr(
                self.discriminator_scheduler, "scheduler", self.discriminator_scheduler
            )
            if isinstance(base_d_scheduler, ReduceLROnPlateau):
                self.discriminator_scheduler.step(losses.get("discriminator", 0.0))
            else:
                self.discriminator_scheduler.step()

        # Add comprehensive metrics
        vae_params = list(self.compressor.parameters()) + list(self.expander.parameters())
        losses.update(
            {
                # Gradient norms
                "grad_norm_vae": compute_grad_norm(vae_params),
                "grad_norm_disc": (
                    compute_grad_norm(self.discriminator.parameters()) if self.use_gan else 0.0
                ),
                # Learning rates
                "lr_vae": self.optimizer.param_groups[0]["lr"],
                "lr_disc": (
                    self.discriminator_optimizer.param_groups[0]["lr"] if self.use_gan else 0.0
                ),
                # Adaptive weights (if enabled)
                "weight_recon": (
                    self._compute_adaptive_weight("recon") if self.adaptive_weights else 1.0
                ),
                "weight_kl": self._compute_adaptive_weight("kl") if self.adaptive_weights else 1.0,
                "weight_gan": (
                    self._compute_adaptive_weight("gan")
                    if self.use_gan and self.adaptive_weights
                    else 0.0
                ),
            }
        )

        return losses

    def _train_discriminator(
        self,
        real_imgs: torch.Tensor,
        global_step: int,
    ) -> float:
        """Train discriminator on real and fake images.

        Note: VAE (encoder+decoder) is frozen during discriminator training.
        Only the discriminator learns to distinguish real from fake images.
        """
        self.discriminator.train()

        # Generate fake images with VAE frozen (no gradients to VAE)
        with torch.no_grad():
            packed = self.compressor(real_imgs, training=False)
            img_seq = packed[:, :-1, :].contiguous()
            ctx_vec = img_seq.mean(dim=1)

            # Generate reconstructions (VAE doesn't receive gradients here)
            out_imgs_for_D = self.expander(packed, use_context=self.use_spade)

        # Discriminator step
        self.discriminator_optimizer.zero_grad(set_to_none=True)

        # Add instance noise to inputs
        real_imgs_noisy = add_instance_noise(
            real_imgs, self.instance_noise_std, self.instance_noise_decay, global_step
        )
        fake_imgs_noisy = add_instance_noise(
            out_imgs_for_D.detach(), self.instance_noise_std, self.instance_noise_decay, global_step
        )

        # Real images with gradient for R1 penalty
        real_imgs_noisy.requires_grad_(True)
        real_logits = self.discriminator(real_imgs_noisy, ctx_vec.detach())

        d_img_loss = torch.tensor(0.0, device=real_imgs.device)

        # R1 gradient penalty (periodic)
        if (global_step % self.r1_interval) == 0:
            r1 = r1_penalty(real_imgs_noisy, real_logits)
            d_img_loss = d_img_loss + (self.r1_gamma * 0.5) * r1

        real_imgs_noisy.requires_grad_(False)

        # Fake images
        fake_logits = self.discriminator(fake_imgs_noisy, ctx_vec.detach())
        d_hinge = d_hinge_loss(real_logits, fake_logits)
        d_img_loss = d_img_loss + d_hinge

        self.accelerator.backward(d_img_loss)
        self.discriminator_optimizer.step()

        return float(d_img_loss.detach().item())

    def _train_generator(
        self,
        real_imgs: torch.Tensor,
        global_step: int,
    ) -> dict[str, float]:
        """Train VAE generator (compressor + expander).

        Returns:
            Dictionary with loss values:
            - vae_loss: Total VAE loss
            - recon_loss: Reconstruction loss
            - kl_loss: KL divergence loss
            - g_loss: GAN generator loss (if enabled)
            - lpips_loss: LPIPS perceptual loss (if enabled)
        """
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass with reparameterization
        packed_rec, mu, logvar = self.compressor(real_imgs, training=True)
        out_imgs_rec = self.expander(packed_rec, use_context=self.use_spade)

        # Reconstruction loss with frequency-aware weighting
        recon_l1 = self._frequency_weighted_loss(out_imgs_rec, real_imgs, alpha=1.0)
        recon_mse = self.reconstruction_loss_min_fn(out_imgs_rec, real_imgs)
        recon_loss = recon_l1 + self.mse_weight * recon_mse

        # LPIPS perceptual loss
        perceptual_loss = torch.tensor(0.0, device=real_imgs.device)
        if self.use_lpips and self.lpips_fn is not None:
            # Compute LPIPS WITH gradients so it actually trains the VAE
            perceptual_loss = self.lpips_fn(out_imgs_rec, real_imgs).mean()
            recon_loss = recon_loss + self.lambda_lpips * perceptual_loss

        # KL divergence with beta annealing
        beta = cosine_anneal_beta(global_step, self.kl_warmup_steps, self.kl_beta)
        kl = kl_standard_normal(mu, logvar, free_bits_nats=self.kl_free_bits, reduce="mean")

        # GAN generator loss (decoder only - detach encoder gradients)
        G_img_loss = torch.tensor(0.0, device=real_imgs.device)
        if self.use_gan and self.discriminator is not None:
            # CRITICAL: Detach latent to prevent GAN gradients flowing to encoder
            # The encoder should only learn from reconstruction + KL loss
            # The decoder learns to generate realistic images from any latent
            packed_rec_detached = packed_rec.detach()
            out_imgs_gan = self.expander(packed_rec_detached, use_context=self.use_spade)
            ctx_vec_rec = packed_rec_detached[:, :-1, :].contiguous().mean(dim=1)
            g_real_logits = self.discriminator(out_imgs_gan, ctx_vec_rec)
            G_img_loss = self.lambda_adv * g_hinge_loss(g_real_logits)

        # Update loss history for adaptive weighting
        self.loss_history["recon"].add_item(float(recon_loss.item()))
        self.loss_history["kl"].add_item(float(kl.item()))
        if self.use_gan:
            self.loss_history["gan"].add_item(float(G_img_loss.item()))

        # Compute adaptive weights
        w_recon = self._compute_adaptive_weight("recon")
        w_kl = self._compute_adaptive_weight("kl")
        w_gan = self._compute_adaptive_weight("gan") if self.use_gan else 0.0

        # Total loss with adaptive weighting
        total_loss = w_recon * recon_loss + w_kl * beta * kl
        if self.use_gan:
            total_loss = total_loss + w_gan * G_img_loss

        # Check for NaN/Inf in loss
        if check_for_nan(total_loss, "vae_total_loss", logger):
            logger.error("Skipping batch due to NaN in VAE loss")
            return {
                "vae": 0.0,
                "kl": 0.0,
                "generator": 0.0,
                "lpips": 0.0,
                "recon": 0.0,
            }

        self.accelerator.backward(total_loss)

        # Check gradients for NaN/Inf after backward
        vae_params = list(self.compressor.parameters()) + list(self.expander.parameters())
        if self.accelerator.scaler is not None:
            self.accelerator.scaler.unscale_(self.optimizer)
            for param in vae_params:
                if param.grad is not None and check_for_nan(param.grad, "vae_grad", logger):
                    logger.warning("NaN gradient in VAE, zeroing it")
                    param.grad.zero_()

        # Clip gradients (only VAE parameters)
        self.accelerator.clip_grad_norm_(vae_params, self.gradient_clip_norm)

        self.optimizer.step()

        # Return dict matching original tuple behavior:
        # vae = recon_loss (NOT total_loss which includes adaptive weighting and can be huge/negative)
        return {
            "vae": float(recon_loss.detach().item()),
            "kl": float(kl.detach().item()),
            "generator": float(G_img_loss.detach().item()) if self.use_gan else 0.0,
            "lpips": float(perceptual_loss.detach().item()) if self.use_lpips else 0.0,
            "recon": float(recon_loss.detach().item()),
        }

    def get_average_losses(self) -> dict[str, float]:
        """Get average losses from buffers."""
        losses = {
            "vae_avg": self.vae_loss_buffer.average,
            "kl_avg": self.kl_loss_buffer.average,
        }

        if self.use_gan:
            losses["discriminator_avg"] = self.d_loss_buffer.average
            losses["generator_avg"] = self.g_loss_buffer.average

        if self.use_lpips:
            losses["lpips_avg"] = self.lpips_loss_buffer.average

        return losses
