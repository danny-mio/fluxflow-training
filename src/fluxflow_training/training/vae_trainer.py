"""VAE training logic for FluxFlow.

Handles VAE (compressor + expander) training with optional GAN discriminator.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fluxflow.utils import get_logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from .losses import d_hinge_loss, g_hinge_loss, kl_standard_normal, r1_penalty
from .schedulers import cosine_anneal_beta
from .utils import EMA, FloatBuffer

logger = get_logger(__name__)


def check_for_nan(tensor, name, logger_inst):
    """Check for NaN/Inf values and log warning."""
    # Handle non-tensor inputs (e.g., MagicMock in tests)
    if not isinstance(tensor, torch.Tensor):
        return False
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
        train_reconstruction: bool = True,  # NEW: Control reconstruction loss
        kl_beta: float = 0.0001,
        kl_warmup_steps: int = 5000,
        kl_free_bits: float = 0.0,
        # GAN settings
        use_gan: bool = True,
        discriminator: Optional[nn.Module] = None,
        discriminator_optimizer: Optional[Optimizer] = None,
        discriminator_scheduler: Optional[_LRScheduler] = None,  # type: ignore[type-arg]
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
            train_reconstruction: Compute reconstruction loss (L1+MSE+LPIPS). Set to False for
                GAN-only or SPADE-only training without VAE reconstruction (default: True)
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
        self.train_reconstruction = train_reconstruction

        # KL settings
        self.kl_beta = kl_beta
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_free_bits = kl_free_bits

        # GAN settings
        self.use_gan = use_gan
        if use_gan:
            if discriminator is None:
                raise ValueError("discriminator is required when use_gan=True")
            if discriminator_optimizer is None:
                raise ValueError(
                    "discriminator_optimizer is required when use_gan=True. "
                    "Ensure pipeline config includes optimization.optimizers.discriminator section."
                )

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

    def _bezier_regularization_loss(self):
        """
        Regularize Bezier control points to prevent extreme curves.

        Encourages control points to form a smooth, near-linear curve
        which prevents high contrast and saturation issues.

        Returns:
            Regularization loss for Bezier parameters
        """
        loss = 0.0

        # Find RGB Bezier activation in expander
        if hasattr(self.expander, "rgb_activation"):
            rgb_bezier = self.expander.rgb_activation

            # Regularize control points to stay near linear interpolation
            # For a linear curve from p0=-1 to p3=1:
            # p1 should be near -0.33, p2 should be near 0.33
            linear_p1 = -0.33
            linear_p2 = 0.33

            # L2 penalty for deviation from linear
            if hasattr(rgb_bezier, "p1") and hasattr(rgb_bezier, "p2"):
                loss += torch.mean((rgb_bezier.p1 - linear_p1) ** 2)
                loss += torch.mean((rgb_bezier.p2 - linear_p2) ** 2)

                # Additionally, penalize large differences between p1 and p2
                # (prevents S-curves that cause contrast expansion)
                p_diff = torch.abs(rgb_bezier.p2 - rgb_bezier.p1)
                # Ideal difference for linear curve is 0.66
                loss += torch.mean((p_diff - 0.66) ** 2)

        return loss

    def _histogram_matching_loss(self, pred, target, bins=64):
        """
        Encourage matching color distribution between pred and target.

        Prevents contrast expansion and color shifts by matching
        the histogram of each color channel.

        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            bins: Number of histogram bins

        Returns:
            Histogram matching loss
        """
        loss = 0.0
        for c in range(3):  # R, G, B channels
            # Flatten spatial dimensions for histogram
            pred_c = pred[:, c].reshape(-1)
            target_c = target[:, c].reshape(-1)

            # Compute normalized histograms
            pred_hist = torch.histc(pred_c, bins=bins, min=-1.0, max=1.0)
            target_hist = torch.histc(target_c, bins=bins, min=-1.0, max=1.0)

            # Normalize to probability distribution
            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
            target_hist = target_hist / (target_hist.sum() + 1e-8)

            # Use Earth Mover's Distance (more stable than KL divergence)
            # Compute cumulative distributions
            pred_cdf = torch.cumsum(pred_hist, dim=0)
            target_cdf = torch.cumsum(target_hist, dim=0)

            # L1 distance between CDFs (Wasserstein-1 distance)
            loss += torch.mean(torch.abs(pred_cdf - target_cdf))

        return loss / 3.0  # Average over channels

    def _color_statistics_loss(self, pred, target):
        """
        Match mean and std of each color channel.

        Simple but effective way to prevent contrast/saturation issues.

        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Color statistics matching loss
        """
        loss = 0.0

        for c in range(3):  # R, G, B channels
            # Mean matching
            pred_mean = pred[:, c].mean()
            target_mean = target[:, c].mean()
            loss += (pred_mean - target_mean) ** 2

            # Std matching (prevents contrast expansion)
            pred_std = pred[:, c].std()
            target_std = target[:, c].std()
            loss += (pred_std - target_std) ** 2

        return loss / 3.0  # Average over channels

    def _contrast_regularization_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrast regularization loss to prevent over-saturation.

        Matches both global (per-channel std) and local (per-sample std) contrast.

        Args:
            pred: Predicted images [B, 3, H, W]
            target: Target images [B, 3, H, W]

        Returns:
            Scalar contrast loss
        """
        loss = 0.0

        # Component 1: Per-channel std ratio (global contrast preservation)
        for c in range(3):
            pred_std = pred[:, c].std()
            target_std = target[:, c].std()
            std_ratio = pred_std / (target_std + 1e-8)
            loss += (std_ratio - 1.0) ** 2

        # Component 2: Per-sample std (local contrast preservation)
        pred_std_per_sample = pred.reshape(pred.size(0), -1).std(dim=1)
        target_std_per_sample = target.reshape(target.size(0), -1).std(dim=1)
        loss += F.mse_loss(pred_std_per_sample, target_std_per_sample)

        return loss / 4.0  # Average over 3 channels + 1 local component

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
        discriminator_was_stepped = False

        # Train discriminator first if using GAN
        if self.use_gan and self.discriminator is not None:
            # Extra safety check (should never happen due to __init__ validation)
            if self.discriminator_optimizer is None:
                raise RuntimeError(
                    "VAETrainer.use_gan=True but discriminator_optimizer is None. "
                    "This should have been caught in __init__. Please report this bug."
                )
            d_loss = self._train_discriminator(real_imgs, global_step)
            losses["discriminator"] = d_loss
            self.d_loss_buffer.add_item(d_loss)
            discriminator_was_stepped = True

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

        # Color/contrast regularization metrics
        losses["bezier_reg"] = gen_losses.get("bezier_reg", 0.0)
        losses["color_stats"] = gen_losses.get("color_stats", 0.0)
        losses["hist_loss"] = gen_losses.get("hist_loss", 0.0)
        losses["contrast_loss"] = gen_losses.get("contrast_loss", 0.0)

        # Check if optimizer was actually stepped (could be skipped due to NaN)
        optimizer_was_stepped = gen_losses.pop("_optimizer_stepped", True)

        # Update EMA only if optimizer was stepped
        if optimizer_was_stepped and self.ema is not None:
            self.ema.update()

        # Step schedulers after optimizer step (ReduceLROnPlateau requires metric, others don't)
        # Only step if: 1) optimizer was actually stepped, and 2) not the very first step
        if optimizer_was_stepped and global_step > 0:
            total_loss = gen_losses["vae"]  # Use recon_loss for scheduler

            # Get the underlying scheduler (may be wrapped by accelerator)
            base_scheduler = getattr(self.scheduler, "scheduler", self.scheduler)
            if isinstance(base_scheduler, ReduceLROnPlateau):
                self.scheduler.step(float(total_loss))  # type: ignore[arg-type]
            else:
                self.scheduler.step()  # type: ignore[call-arg]

            # Step discriminator scheduler only if discriminator was actually trained this step
            if discriminator_was_stepped and self.discriminator_scheduler is not None:
                base_d_scheduler = getattr(
                    self.discriminator_scheduler, "scheduler", self.discriminator_scheduler
                )
                if isinstance(base_d_scheduler, ReduceLROnPlateau):
                    self.discriminator_scheduler.step(float(losses.get("discriminator", 0.0)))  # type: ignore[arg-type]
                else:
                    self.discriminator_scheduler.step()  # type: ignore[call-arg]

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

        # Check input for NaN/Inf
        if check_for_nan(real_imgs, "input_images", logger):
            logger.error("NaN detected in input images - skipping batch")
            return {
                "vae": 0.0,
                "kl": 0.0,
                "generator": 0.0,
                "lpips": 0.0,
                "recon": 0.0,
                "_optimizer_stepped": False,
            }

        # Forward pass with reparameterization
        packed_rec, mu, logvar = self.compressor(real_imgs, training=True)

        # Early NaN detection after compression
        if (
            check_for_nan(packed_rec, "packed_rec", logger)
            or check_for_nan(mu, "mu", logger)
            or check_for_nan(logvar, "logvar", logger)
        ):
            logger.error("NaN detected in compressor output - skipping batch")
            logger.error(
                f"  Input image stats: min={real_imgs.min().item():.4f}, max={real_imgs.max().item():.4f}, mean={real_imgs.mean().item():.4f}"
            )
            return {
                "vae": 0.0,
                "kl": 0.0,
                "generator": 0.0,
                "lpips": 0.0,
                "recon": 0.0,
                "_optimizer_stepped": False,
            }

        out_imgs_rec = self.expander(packed_rec, use_context=self.use_spade)

        # Early NaN detection after expansion
        if check_for_nan(out_imgs_rec, "out_imgs_rec", logger):
            logger.error("NaN detected in expander output - skipping batch")
            logger.error(
                f"  packed_rec stats: min={packed_rec.min().item():.4f}, max={packed_rec.max().item():.4f}"
            )
            return {
                "vae": 0.0,
                "kl": 0.0,
                "generator": 0.0,
                "lpips": 0.0,
                "recon": 0.0,
                "_optimizer_stepped": False,
            }

        # Reconstruction loss (skip if train_reconstruction=False)
        recon_loss = torch.tensor(0.0, device=real_imgs.device)
        perceptual_loss = torch.tensor(0.0, device=real_imgs.device)

        if self.train_reconstruction:
            # Reconstruction loss with frequency-aware weighting
            # Reduced alpha from 1.0 to 0.5 to prevent over-sharpening and high contrast
            recon_l1 = self._frequency_weighted_loss(out_imgs_rec, real_imgs, alpha=0.5)
            recon_mse = self.reconstruction_loss_min_fn(out_imgs_rec, real_imgs)
            recon_loss = recon_l1 + self.mse_weight * recon_mse

            # LPIPS perceptual loss
            if self.use_lpips and self.lpips_fn is not None:
                # Compute LPIPS WITH gradients so it actually trains the VAE
                # NOTE: Gradient checkpointing removed - causes recursive checkpointing
                # with VAE decoder, leading to OOM instead of saving memory
                perceptual_loss = self.lpips_fn(out_imgs_rec, real_imgs).mean()
                recon_loss = recon_loss + self.lambda_lpips * perceptual_loss

        # KL divergence with beta annealing (still compute even if not training reconstruction)
        beta = cosine_anneal_beta(global_step, self.kl_warmup_steps, self.kl_beta)
        kl = kl_standard_normal(
            mu, logvar, free_bits_nats=self.kl_free_bits, reduce="mean", normalize_by_dims=True
        )

        # GAN generator loss
        G_img_loss = torch.tensor(0.0, device=real_imgs.device)
        if self.use_gan and self.discriminator is not None:
            # Detach latent only when training reconstruction
            # - When train_reconstruction=True: Encoder learns from recon+KL, decoder from GAN
            # - When train_reconstruction=False: Both encoder+decoder learn from GAN+KL
            if self.train_reconstruction:
                # Detach to prevent GAN gradients flowing to encoder
                packed_rec_for_gan = packed_rec.detach()
            else:
                # Don't detach - encoder needs GAN gradients when no reconstruction loss
                packed_rec_for_gan = packed_rec

            out_imgs_gan = self.expander(packed_rec_for_gan, use_context=self.use_spade)
            ctx_vec_rec = packed_rec_for_gan[:, :-1, :].contiguous().mean(dim=1)

            # Check inputs to discriminator
            if check_for_nan(out_imgs_gan, "out_imgs_gan", logger):
                logger.error("NaN in discriminator input images")
                G_img_loss = torch.tensor(0.0, device=real_imgs.device)
            elif check_for_nan(ctx_vec_rec, "ctx_vec_rec", logger):
                logger.error("NaN in discriminator context vector")
                G_img_loss = torch.tensor(0.0, device=real_imgs.device)
            else:
                g_real_logits = self.discriminator(out_imgs_gan, ctx_vec_rec)

                # Check discriminator output
                if check_for_nan(g_real_logits, "g_real_logits", logger):
                    logger.error("NaN in discriminator output logits")
                    logger.error(
                        f"  out_imgs_gan stats: min={out_imgs_gan.min().item():.4f}, max={out_imgs_gan.max().item():.4f}, mean={out_imgs_gan.mean().item():.4f}"
                    )
                    logger.error(
                        f"  ctx_vec_rec stats: min={ctx_vec_rec.min().item():.4f}, max={ctx_vec_rec.max().item():.4f}, mean={ctx_vec_rec.mean().item():.4f}"
                    )
                    # Check discriminator weights for NaN
                    for name, param in self.discriminator.named_parameters():
                        if torch.isnan(param).any():
                            logger.error(f"  NaN in discriminator weight: {name}")
                            break
                    G_img_loss = torch.tensor(0.0, device=real_imgs.device)
                else:
                    G_img_loss = self.lambda_adv * g_hinge_loss(g_real_logits)

        # Update loss history for adaptive weighting
        if self.train_reconstruction:
            self.loss_history["recon"].add_item(float(recon_loss.item()))
        self.loss_history["kl"].add_item(float(kl.item()))
        if self.use_gan:
            self.loss_history["gan"].add_item(float(G_img_loss.item()))

        # Compute adaptive weights
        w_recon = self._compute_adaptive_weight("recon") if self.train_reconstruction else 0.0
        w_kl = self._compute_adaptive_weight("kl")
        w_gan = self._compute_adaptive_weight("gan") if self.use_gan else 0.0

        # Color/contrast regularization losses
        bezier_reg = self._bezier_regularization_loss()
        color_stats_loss = (
            self._color_statistics_loss(out_imgs_rec, real_imgs)
            if self.train_reconstruction
            else torch.tensor(0.0, device=real_imgs.device)
        )
        hist_loss = (
            self._histogram_matching_loss(out_imgs_rec, real_imgs)
            if self.train_reconstruction
            else torch.tensor(0.0, device=real_imgs.device)
        )
        contrast_loss = (
            self._contrast_regularization_loss(out_imgs_rec, real_imgs)
            if self.train_reconstruction
            else torch.tensor(0.0, device=real_imgs.device)
        )

        # Total loss with adaptive weighting
        total_loss = w_kl * beta * kl
        if self.train_reconstruction:
            total_loss = total_loss + w_recon * recon_loss
        if self.use_gan:
            total_loss = total_loss + w_gan * G_img_loss

        # Add regularization (small weights to not dominate main losses)
        total_loss = total_loss + 0.05 * bezier_reg  # Prevent extreme Bezier curves (increased)
        total_loss = total_loss + 0.05 * color_stats_loss  # Match color statistics
        total_loss = total_loss + 0.02 * hist_loss  # Match color distributions
        total_loss = total_loss + 0.1 * contrast_loss  # Prevent over-saturation (new)

        # Check for NaN/Inf in loss with detailed diagnostics
        if check_for_nan(total_loss, "vae_total_loss", logger):
            logger.error("Skipping batch due to NaN in VAE loss")
            # Detailed diagnostics
            logger.error(
                f"  recon_loss: {recon_loss.item() if not check_for_nan(recon_loss, 'recon', logger) else 'NaN'}"
            )
            logger.error(f"  kl: {kl.item() if not check_for_nan(kl, 'kl', logger) else 'NaN'}")
            logger.error(
                f"  G_img_loss: {G_img_loss.item() if not check_for_nan(G_img_loss, 'gan', logger) else 'NaN'}"
            )
            logger.error(f"  w_recon: {w_recon}, w_kl: {w_kl}, w_gan: {w_gan}, beta: {beta}")
            logger.error(
                f"  mu stats: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}"
            )
            logger.error(
                f"  logvar stats: min={logvar.min().item():.4f}, max={logvar.max().item():.4f}, mean={logvar.mean().item():.4f}"
            )
            if self.train_reconstruction:
                logger.error(
                    f"  out_imgs_rec stats: min={out_imgs_rec.min().item():.4f}, max={out_imgs_rec.max().item():.4f}"
                )
            return {
                "vae": 0.0,
                "kl": 0.0,
                "generator": 0.0,
                "lpips": 0.0,
                "recon": 0.0,
                "_optimizer_stepped": False,  # Signal that optimizer was not stepped
            }

        # CRITICAL: Clear cache before backward to prevent OOM
        # Gradient checkpointing in VAE causes memory spikes during backward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

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
            "bezier_reg": float(bezier_reg.detach().item()),
            "color_stats": float(color_stats_loss.detach().item()),
            "hist_loss": float(hist_loss.detach().item()),
            "contrast_loss": float(contrast_loss.detach().item()),
            "_optimizer_stepped": True,  # Signal that optimizer was stepped
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
