"""VAE training logic for FluxFlow.

Handles VAE (compressor + expander) training with optional GAN discriminator.
"""

from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Context encoder uses SiLU activation
from fluxflow.utils import get_logger
from fluxflow.utils.mps import mps_safe_pool2d
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from .losses import d_hinge_loss, g_hinge_loss, kl_standard_normal, r1_penalty
from .schedulers import cosine_anneal_beta
from .utils import EMA, FloatBuffer


class ContextEncoder(nn.Module):
    """Simple context encoder that handles variable input sizes and MPS compatibility."""

    def __init__(self, context_channels, context_height, context_width):
        super().__init__()
        self.context_channels = context_channels
        self.context_height = context_height
        self.context_width = context_width

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # RGB -> 64ch
            nn.InstanceNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128ch
            nn.InstanceNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, context_channels, 1),  # Reduce to context channels
        )

    def forward(self, x):
        # Encode
        features = self.encoder(x)  # [B, C, H, W]

        pooled = mps_safe_pool2d(features, output_size=(self.context_height, self.context_width))

        return pooled


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


def _empty_cache(device: torch.device) -> None:
    """Free cached memory for the current device."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


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
        spade_training_mode: Literal["full", "alternate"] = "full",
        train_reconstruction: bool = True,  # NEW: Control reconstruction loss
        train_kl: bool = True,
        train_colorstats: bool = True,
        train_histogram: bool = True,
        train_contrast: bool = True,
        train_coarseness: bool = True,
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
        # Context predictor settings
        context_channels: int = 64,
        context_height: int = 16,
        context_width: int = 16,
        context_predictor_path: Optional[str] = None,
        # v0.10.0: explicit input dim for context_predictor.
        # For v0.10.0 compressors ctx_vec = img_seq.mean(dim=1) has dim 2*d_model (256).
        # Pass compressor.get_context_dims() * 2 from the orchestrator.
        # None triggers legacy runtime detection (for v0.8.0 and earlier).
        ctx_input_dim: Optional[int] = None,
        # v0.10.0: auxiliary context reconstruction loss weight (plan §4.1 / DP-1).
        lambda_ctx_aux: float = 0.01,
        train_ctx_aux: bool = True,
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
            spade_training_mode: SPADE training mode ("full" or "alternate")
            train_reconstruction: Compute reconstruction loss (L1+MSE+LPIPS). Set to False for
                GAN-only or SPADE-only training without VAE reconstruction (default: True)
            train_kl: Compute KL divergence loss (default: True)
            train_colorstats: Compute color statistics loss (default: True)
            train_histogram: Compute histogram matching loss (default: True)
            train_contrast: Compute contrast regularization loss (default: True)
            train_coarseness: Compute coarseness distribution loss (default: True)
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
            ctx_input_dim: Explicit input dim for context_predictor (v0.10.0). Pass
                compressor.get_context_dims() * 2 from the orchestrator. None triggers
                legacy runtime detection from a test forward pass (v0.8.0 and earlier).
            lambda_ctx_aux: Weight for auxiliary context reconstruction loss (plan §4.1).
                Applied as MSE(context_tokens, stop_grad(z_tokens)) during VAE training.
                Default 0.01. Only active when train_ctx_aux=True.
            train_ctx_aux: Compute auxiliary context reconstruction loss (default: True).
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
        self.spade_training_mode = spade_training_mode

        self.train_reconstruction = train_reconstruction
        self.train_kl = train_kl
        self.train_colorstats = train_colorstats
        self.train_histogram = train_histogram
        self.train_contrast = train_contrast
        self.train_coarseness = train_coarseness

        # KL settings
        self.kl_beta = kl_beta
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_free_bits = kl_free_bits

        # GAN settings
        self.use_gan = use_gan
        if use_gan:
            if discriminator is None:
                raise ValueError("discriminator must be provided when use_gan=True")
            if discriminator_optimizer is None:
                raise ValueError("discriminator_optimizer must be provided when use_gan=True")
            if discriminator_scheduler is None:
                raise ValueError("discriminator_scheduler must be provided when use_gan=True")

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_scheduler = discriminator_scheduler
        self.lambda_adv = lambda_adv
        self.r1_interval = r1_interval
        self.r1_gamma = r1_gamma

        # Training settings
        self.gradient_clip_norm = gradient_clip_norm
        self.use_lpips = use_lpips
        self.lambda_lpips = lambda_lpips
        self.instance_noise_std = instance_noise_std
        self.instance_noise_decay = instance_noise_decay
        self.adaptive_weights = adaptive_weights
        self.mse_weight = mse_weight
        self.accelerator = accelerator

        # Initialize LPIPS if needed
        if self.use_lpips:
            try:
                import warnings

                import lpips

                # Suppress torchvision pretrained deprecation warning for LPIPS
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
                    # Use spatial=True for detailed perceptual loss maps
                    self.lpips_fn = lpips.LPIPS(net="vgg", spatial=True)

                self.lpips_fn.eval()
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                raise ImportError("LPIPS not available. Install with: pip install lpips")

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
            "lpips": FloatBuffer(100),
        }

        # Context predictor settings
        self.context_channels = context_channels
        self.context_height = context_height
        self.context_width = context_width
        self.context_predictor_path = context_predictor_path
        self.lambda_ctx_aux = lambda_ctx_aux
        self.train_ctx_aux = train_ctx_aux

        # Initialize context predictor with SiLU activation.
        # ctx_input_dim is the dim of ctx_vec = img_seq.mean(dim=1), i.e. the full packed
        # token width.  For v0.10.0 this is 2*d_model (256); for v0.8.0 it is d_model+5 (133).
        # When provided explicitly (from the orchestrator) skip the expensive runtime detection.
        if ctx_input_dim is not None:
            latent_dim = ctx_input_dim
            logger.info(f"Using explicit ctx_input_dim for context_predictor: {latent_dim}")
        else:
            # Legacy runtime detection for v0.8.0 and earlier compressors.
            latent_dim = 27  # conservative fallback
            try:
                with torch.no_grad():
                    test_device = next(self.compressor.parameters()).device
                    test_input = torch.randn(1, 3, 64, 64).to(test_device)
                    compressor_output = self.compressor(test_input, training=True)
                    if isinstance(compressor_output, tuple):
                        packed = compressor_output[0]
                    else:
                        packed = compressor_output
                    # packed: [B, T+1, 2*d_model] — last dim is the full packed token width
                    latent_dim = packed.shape[-1]  # type: ignore
                    logger.info(f"Detected ctx_input_dim from packed output: {latent_dim}")
            except Exception as exc:
                logger.warning(
                    f"Could not detect latent dimension, using fallback {latent_dim}: {exc}"
                )

        context_output_dim = context_channels * context_height * context_width

        self.context_predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),  # Smooth activation for latent->conditioning mapping
            nn.Linear(256, context_output_dim),
        )

        # Load existing context predictor if available
        if context_predictor_path and Path(context_predictor_path).exists():
            try:
                checkpoint = torch.load(context_predictor_path, map_location="cpu")
                self.context_predictor.load_state_dict(checkpoint["context_predictor"])
                logger.info(f"Loaded context predictor from {context_predictor_path}")
            except Exception as e:
                logger.warning(f"Failed to load context predictor: {e}")

        # Context encoder for ideal context (used in KL-context alignment)
        self.context_encoder = ContextEncoder(context_channels, context_height, context_width)

        # Store context size
        self.context_height = context_height
        self.context_width = context_width

        # Move context predictor components to device
        device = next(self.compressor.parameters()).device
        self.context_predictor = self.context_predictor.to(device)
        self.context_encoder = self.context_encoder.to(device)

        # Optimizer for context predictor
        self.context_predictor_optimizer = torch.optim.AdamW(
            list(self.context_predictor.parameters()) + list(self.context_encoder.parameters()),
            lr=1e-4,
            weight_decay=1e-4,
        )

        self._prepare_context_components()

    def save_context_predictor(self, checkpoint_path: str):
        """Save context predictor state for persistence."""
        if self.context_predictor_path:
            checkpoint = {
                "context_predictor": self.context_predictor.state_dict(),
                "context_encoder": self.context_encoder.state_dict(),
                "context_predictor_optimizer": self.context_predictor_optimizer.state_dict(),
            }
            torch.save(checkpoint, self.context_predictor_path)
            logger.info(f"Saved context predictor to {self.context_predictor_path}")

    def load_context_predictor(self, checkpoint_path: str):
        """Load context predictor state."""
        if Path(checkpoint_path).exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.context_predictor.load_state_dict(checkpoint["context_predictor"])
                self.context_encoder.load_state_dict(checkpoint["context_encoder"])
                self.context_predictor_optimizer.load_state_dict(
                    checkpoint["context_predictor_optimizer"]
                )
                logger.info(f"Loaded context predictor from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load context predictor: {e}")

    def _prepare_context_components(self) -> None:
        if self.accelerator is None:
            return
        logger.info(
            "Skipping accelerator.prepare for context components to avoid wrapper "
            "compatibility issues with module indexing."
        )

    def _get_unwrapped_model(self, model: nn.Module) -> nn.Module:
        """
        Get the underlying model, unwrapping from accelerator wrappers like DDP/FSDP.

        Args:
            model: The potentially wrapped model

        Returns:
            The unwrapped model
        """
        if hasattr(model, "module"):
            return model.module
        return model

    def _get_context_input_dim(self) -> int:
        """Get the input dimension for the context predictor."""
        context_predictor = self._get_unwrapped_model(self.context_predictor)
        for module in context_predictor.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise RuntimeError("Context predictor has no Linear layer")

    def _get_effective_spade_usage(self, global_step: int) -> bool:
        """
        Determine whether to use SPADE conditioning for the current training step.

        Args:
            global_step: Current training step number

        Returns:
            True if SPADE should be used, False otherwise
        """
        if not self.use_spade:
            return False

        if self.spade_training_mode == "full":
            return True
        elif self.spade_training_mode == "alternate":
            return global_step % 2 == 0  # Alternate every step (even steps use SPADE)
        else:
            raise ValueError(f"Unknown SPADE training mode: {self.spade_training_mode}")

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
        loss = torch.tensor(0.0, device=pred.device)
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

    def _coarseness_loss(
        self, pred: torch.Tensor, target: torch.Tensor, patch_size: int = 16, bins: int = 32
    ) -> torch.Tensor:
        """
        Match distribution of local patch variances per channel.

        Encourages the decoder to preserve surface texture coarseness by matching
        variance distributions between predicted and target images.

        Args:
            pred: Predicted images [B, 3, H, W]
            target: Target images [B, 3, H, W]
            patch_size: Patch size for local variance calculation
            bins: Number of histogram bins for variance distribution

        Returns:
            Scalar coarseness loss
        """
        if pred.shape[2] < patch_size or pred.shape[3] < patch_size:
            return torch.tensor(0.0, device=pred.device)

        loss = torch.tensor(0.0, device=pred.device)

        for c in range(3):
            pred_c = pred[:, c]
            target_c = target[:, c]

            pred_patches = pred_c.unfold(1, patch_size, patch_size).unfold(
                2, patch_size, patch_size
            )
            target_patches = target_c.unfold(1, patch_size, patch_size).unfold(
                2, patch_size, patch_size
            )

            pred_var = pred_patches.var(dim=(-1, -2), unbiased=False).reshape(-1)
            target_var = target_patches.var(dim=(-1, -2), unbiased=False).reshape(-1)

            pred_var = pred_var.clamp(0.0, 1.0)
            target_var = target_var.clamp(0.0, 1.0)

            bin_centers = torch.linspace(0.0, 1.0, bins, device=pred_var.device)
            sigma = 1.0 / max(bins - 1, 1)

            pred_diff = pred_var[:, None] - bin_centers[None, :]
            target_diff = target_var[:, None] - bin_centers[None, :]

            pred_weights = torch.exp(-0.5 * (pred_diff / sigma) ** 2)
            target_weights = torch.exp(-0.5 * (target_diff / sigma) ** 2)

            pred_hist = pred_weights.sum(dim=0)
            target_hist = target_weights.sum(dim=0)

            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
            target_hist = target_hist / (target_hist.sum() + 1e-8)

            pred_cdf = torch.cumsum(pred_hist, dim=0)
            target_cdf = torch.cumsum(target_hist, dim=0)

            loss += torch.mean(torch.abs(pred_cdf - target_cdf))

        return loss / 3.0

    def _compute_adaptive_weight(self, loss_type, max_weight: float = 5.0):
        """Balance losses based on magnitude using inverse weighting, clamped to max_weight.

        The weight is clamped to prevent any single loss (especially GAN at startup,
        when its magnitude is near zero) from receiving an explosive gradient multiplier
        that overwhelms the reconstruction signal and causes divergence.
        """
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
            return min(target / (avg + 1e-8), max_weight)
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
        losses["coarseness_loss"] = gen_losses.get("coarseness_loss", 0.0)
        # v0.10.0: auxiliary context reconstruction loss
        losses["ctx_aux_loss"] = gen_losses.get("ctx_aux_loss", 0.0)

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

        # Generate fake images using the stochastic (reparameterised) path so the
        # discriminator sees the same distribution of reconstructions as the generator.
        # torch.no_grad() prevents stale encoder gradients from accumulating here;
        # they would be zeroed at the start of _train_generator anyway, but being
        # explicit avoids any subtle interaction with gradient checkpointing.
        with torch.no_grad():
            packed, _, _ = self.compressor(real_imgs, training=True)
        img_seq = packed[:, :-1, :].contiguous()
        ctx_vec = img_seq.mean(dim=1)

        # Validate context vector dimension against discriminator expectation
        if hasattr(self.discriminator, "ctx_proj") and self.discriminator.ctx_dim > 0:
            expected_ctx_dim = self.discriminator.ctx_proj.in_features
            if ctx_vec.shape[-1] != expected_ctx_dim:
                raise RuntimeError(
                    f"Discriminator ctx_dim={expected_ctx_dim} does not match "
                    f"compressor ctx_vec dim={ctx_vec.shape[-1]}. "
                    f"Delete D_img.safetensors and restart to create a fresh discriminator."
                )
        else:
            ctx_vec = None

        # Generate conditional reconstructions (VAE doesn't receive gradients here)
        with torch.no_grad():
            out_imgs_for_D = self.expander(
                packed, use_context=self._get_effective_spade_usage(global_step)
            )

            # Also generate unconditional reconstructions for discriminator training
            out_imgs_uncond_for_D = self.expander(packed, use_context=False)

        # Discriminator step
        self.discriminator_optimizer.zero_grad(set_to_none=True)

        # Add instance noise to inputs
        real_imgs_noisy = add_instance_noise(
            real_imgs, self.instance_noise_std, self.instance_noise_decay, global_step
        )
        fake_imgs_noisy = add_instance_noise(
            out_imgs_for_D.detach(), self.instance_noise_std, self.instance_noise_decay, global_step
        )
        fake_uncond_imgs_noisy = add_instance_noise(
            out_imgs_uncond_for_D.detach(),
            self.instance_noise_std,
            self.instance_noise_decay,
            global_step,
        )

        # Determine if SPADE conditioning is active this step
        spade_active = self._get_effective_spade_usage(global_step)

        # Compute predicted context for SPADE-Aware GAN loss (Step 3)
        B = real_imgs.shape[0]
        predicted_context = None
        if spade_active:
            # Use ctx_vec as latent representation (already padded/truncated)
            latent_repr = ctx_vec.detach()  # [B, expected_ctx_dim]

            context_input_dim = self._get_context_input_dim()
            # Ensure context_predictor matches ctx_vec dimension
            if context_input_dim != latent_repr.shape[-1]:
                logger.warning(
                    "Context predictor dimension mismatch in discriminator: expected %s, got %s",
                    context_input_dim,
                    latent_repr.shape[-1],
                )
                # Skip context prediction if dimensions don't match
                predicted_context = torch.randn(
                    B,
                    self.context_channels,
                    self.context_height,
                    self.context_width,
                    device=latent_repr.device,
                )
            else:
                predicted_context = self.context_predictor(latent_repr)
                predicted_context = predicted_context.view(
                    B, self.context_channels, self.context_height, self.context_width
                )

        # Context vectors: None when SPADE is active (conditioning happens in generator)
        # Actual context when SPADE is inactive (traditional conditional GAN)
        disc_ctx = None if spade_active else (ctx_vec.detach() if ctx_vec is not None else None)

        do_r1 = (global_step % self.r1_interval) == 0

        # Only keep grad graph on real images when R1 penalty is needed
        real_noisy_base = real_imgs_noisy if do_r1 else real_imgs_noisy.detach()
        if do_r1:
            real_noisy_base = real_noisy_base.detach().requires_grad_(True)

        # Real images
        real_logits = self.discriminator(real_noisy_base, disc_ctx)

        d_img_loss = torch.tensor(0.0, device=real_imgs.device)

        # R1 gradient penalty (periodic) - only on real images
        if do_r1:
            r1 = r1_penalty(real_noisy_base, real_logits)
            d_img_loss = d_img_loss + (self.r1_gamma * 0.5) * r1

        # Fake images - use same context logic as real images
        fake_logits = self.discriminator(fake_imgs_noisy, disc_ctx)
        # Unconditional fake images (should also be classified as fake)
        fake_uncond_logits = self.discriminator(fake_uncond_imgs_noisy, disc_ctx)

        # Combine losses - weight unconditional fakes more since they're easier to classify
        d_hinge_cond = d_hinge_loss(real_logits, fake_logits)
        d_hinge_uncond = d_hinge_loss(real_logits, fake_uncond_logits)
        d_hinge = d_hinge_cond + 0.5 * d_hinge_uncond  # Weight conditional loss more
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
        self.context_predictor_optimizer.zero_grad(set_to_none=True)

        # Check input for NaN/Inf
        if check_for_nan(real_imgs, "input_images", logger):
            logger.error("NaN detected in input images - skipping batch")
            return {
                "vae": 0.0,
                "kl": 0.0,
                "generator": 0.0,
                "lpips": 0.0,
                "recon": 0.0,
                "ctx_aux_loss": 0.0,
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
                "ctx_aux_loss": 0.0,
                "_optimizer_stepped": False,
            }

        out_imgs_rec = self.expander(
            packed_rec, use_context=self._get_effective_spade_usage(global_step)
        )

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
                "ctx_aux_loss": 0.0,
                "_optimizer_stepped": False,
            }

        # Reconstruction loss (skip if train_reconstruction=False)
        recon_loss = torch.tensor(0.0, device=real_imgs.device)
        perceptual_loss = torch.tensor(0.0, device=real_imgs.device)

        if self.train_reconstruction:
            # Progressive frequency weighting: start low to prevent over-sharpening, increase over time
            # This allows the model to learn basic structure first, then focus on sharp edges
            total_training_steps = 100000  # Assume ~100k steps for full training
            alpha_progress = min(
                1.0, global_step / (total_training_steps * 0.3)
            )  # Ramp over first 30%
            alpha_start = 0.3  # Start with lower weighting to prevent early over-sharpening
            alpha_end = 1.2  # End with higher weighting for sharp edges
            progressive_alpha = alpha_start + (alpha_end - alpha_start) * alpha_progress

            recon_l1 = self._frequency_weighted_loss(
                out_imgs_rec, real_imgs, alpha=progressive_alpha
            )
            # Remove MSE loss as it can contribute to blur - rely on L1 + LPIPS for reconstruction
            recon_loss = recon_l1

            # LPIPS perceptual loss
            if self.use_lpips and self.lpips_fn is not None:
                # Compute LPIPS WITH gradients so it actually trains the VAE
                # NOTE: Gradient checkpointing removed - causes recursive checkpointing
                # with VAE decoder, leading to OOM instead of saving memory
                # Ensure LPIPS is on the same device as input tensors
                device = out_imgs_rec.device
                lpips_fn = self.lpips_fn.to(device)
                perceptual_loss = lpips_fn(out_imgs_rec, real_imgs).mean()
                recon_loss = recon_loss + self.lambda_lpips * perceptual_loss

        # KL divergence with beta annealing (z branch only; context branch is deterministic).
        beta = 0.0
        if self.train_kl:
            beta = cosine_anneal_beta(global_step, self.kl_warmup_steps, self.kl_beta)
            kl = kl_standard_normal(
                mu,
                logvar,
                free_bits_nats=self.kl_free_bits,
                reduce="mean",
                normalize_by_dims=True,
            )
        else:
            kl = torch.tensor(0.0, device=real_imgs.device)

        # v0.10.0: auxiliary context reconstruction loss (plan §4.1).
        # L_ctx_aux = MSE(context_tokens, sg(z_tokens)) where sg = stop-gradient.
        # Gently encourages context branch to learn a representation aligned with z
        # without forcing them to be identical (independent parameter sets).
        # Active only when train_ctx_aux=True and the packed tensor is wide enough to
        # contain both z and context halves (i.e. v0.10.0 compressor).
        ctx_aux_loss = torch.tensor(0.0, device=real_imgs.device)
        if self.train_ctx_aux:
            try:
                # packed_rec shape: [B, T+1, 2D] for v0.10.0 or [B, T+1, D+5] for earlier.
                # Split at dim // 2 only if total_dim is even (true for v0.10.0 where dim=2D).
                total_dim = packed_rec.size(-1)
                half = total_dim // 2
                if total_dim % 2 == 0 and half > 0:
                    img_seq_rec = packed_rec[:, :-1, :]  # [B, T, 2D]
                    z_tokens_half = img_seq_rec[:, :, :half]  # [B, T, D]
                    ctx_tokens_half = img_seq_rec[:, :, half:]  # [B, T, D]
                    ctx_aux_loss = F.mse_loss(ctx_tokens_half, z_tokens_half.detach())
            except Exception as exc:
                logger.warning(f"ctx_aux_loss computation skipped: {exc}")

        # Context prediction from latents (Step 1 & 2: SiLU activation + KL-context alignment)
        B = real_imgs.shape[0]
        # Use packed_rec ctx_vec — identical to _train_discriminator so context_predictor
        # dimensions stay consistent across both training methods.
        latent_repr = packed_rec[:, :-1, :].contiguous().mean(dim=1)  # [B, 2*d_model]

        context_input_dim = self._get_context_input_dim()
        # Ensure context_predictor matches actual latent dimension (may differ from init detection)
        actual_latent_dim = latent_repr.shape[-1]
        if context_input_dim != actual_latent_dim:
            logger.info(
                "Adjusting context_predictor input dim from %s to %s",
                context_input_dim,
                actual_latent_dim,
            )
            # Create new predictor with correct input dimension
            output_dim = self.context_channels * self.context_height * self.context_width
            new_predictor = nn.Sequential(
                nn.Linear(actual_latent_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, output_dim),
            ).to(latent_repr.device)
            self.context_predictor = new_predictor
            self.context_predictor_optimizer = torch.optim.AdamW(
                list(self.context_predictor.parameters()) + list(self.context_encoder.parameters()),
                lr=1e-4,
                weight_decay=1e-4,
            )

            self._prepare_context_components()

        predicted_context = self.context_predictor(latent_repr.detach())
        predicted_context = predicted_context.view(
            B, self.context_channels, self.context_height, self.context_width
        )

        # Context alignment loss - compare predicted context to ideal context from real images
        ideal_context = self.context_encoder(real_imgs.detach())
        context_alignment_loss = F.mse_loss(predicted_context, ideal_context)

        # GAN generator loss
        G_img_loss = torch.tensor(0.0, device=real_imgs.device)
        spade_active = self._get_effective_spade_usage(global_step)
        if self.use_gan and self.discriminator is not None:
            # Reuse already-decoded images; detach so GAN loss only trains the decoder
            out_imgs_gan = out_imgs_rec.detach()
            ctx_vec_rec = packed_rec[:, :-1, :].contiguous().mean(dim=1).detach()

            # Validate context vector dimension against discriminator expectation
            if hasattr(self.discriminator, "ctx_proj") and self.discriminator.ctx_dim > 0:
                expected_ctx_dim = self.discriminator.ctx_proj.in_features
                if ctx_vec_rec.shape[-1] != expected_ctx_dim:
                    raise RuntimeError(
                        f"Discriminator ctx_dim={expected_ctx_dim} does not match "
                        f"compressor ctx_vec dim={ctx_vec_rec.shape[-1]}. "
                        f"Delete D_img.safetensors and restart to create a fresh discriminator."
                    )
            else:
                ctx_vec_rec = None

            # Check inputs to discriminator
            if check_for_nan(out_imgs_gan, "out_imgs_gan", logger):
                logger.error("NaN in discriminator input images")
                G_img_loss = torch.tensor(0.0, device=real_imgs.device)
            elif ctx_vec_rec is not None and check_for_nan(ctx_vec_rec, "ctx_vec_rec", logger):
                logger.error("NaN in discriminator context vector")
                G_img_loss = torch.tensor(0.0, device=real_imgs.device)
            else:
                gen_ctx = None if spade_active else ctx_vec_rec
                # Discriminator is read-only during generator update
                self.discriminator.eval()
                g_real_logits = self.discriminator(out_imgs_gan, gen_ctx)
                self.discriminator.train()

                # Check discriminator output
                if check_for_nan(g_real_logits, "g_real_logits", logger):
                    logger.error("NaN in discriminator output logits")
                    logger.error(
                        f"  out_imgs_gan stats: min={out_imgs_gan.min().item():.4f}, max={out_imgs_gan.max().item():.4f}, mean={out_imgs_gan.mean().item():.4f}"
                    )
                    if ctx_vec_rec is not None:
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
                    # Raw hinge loss; lambda_adv applied when adding to total_loss
                    G_img_loss = g_hinge_loss(g_real_logits)

        # Update loss history for adaptive weighting (record unscaled G_img_loss)
        if self.train_reconstruction:
            self.loss_history["recon"].add_item(float(recon_loss.item()))
        self.loss_history["kl"].add_item(float(kl.item()))
        if self.use_gan:
            self.loss_history["gan"].add_item(float(G_img_loss.item()))

        # Compute adaptive weights
        w_recon = self._compute_adaptive_weight("recon") if self.train_reconstruction else 0.0
        w_kl = self._compute_adaptive_weight("kl") if self.train_kl else 0.0
        w_gan = self._compute_adaptive_weight("gan") if self.use_gan else 0.0

        # Color/contrast regularization losses
        color_stats_loss = (
            self._color_statistics_loss(out_imgs_rec, real_imgs)
            if self.train_reconstruction and self.train_colorstats
            else torch.tensor(0.0, device=real_imgs.device)
        )
        hist_loss = (
            self._histogram_matching_loss(out_imgs_rec, real_imgs)
            if self.train_reconstruction and self.train_histogram
            else torch.tensor(0.0, device=real_imgs.device)
        )
        contrast_loss = (
            self._contrast_regularization_loss(out_imgs_rec, real_imgs)
            if self.train_reconstruction and self.train_contrast
            else torch.tensor(0.0, device=real_imgs.device)
        )
        coarseness_loss = (
            self._coarseness_loss(out_imgs_rec, real_imgs)
            if self.train_reconstruction and self.train_coarseness
            else torch.tensor(0.0, device=real_imgs.device)
        )

        # Total loss with adaptive weighting
        total_loss = w_kl * beta * kl
        if self.train_reconstruction:
            total_loss = total_loss + w_recon * recon_loss
        if self.use_gan:
            total_loss = total_loss + self.lambda_adv * w_gan * G_img_loss

        # v0.10.0: auxiliary context reconstruction loss (plan §4.1 / DP-1).
        if self.train_ctx_aux:
            total_loss = total_loss + self.lambda_ctx_aux * ctx_aux_loss

        # Add context alignment loss (Step 2)
        total_loss = total_loss + 0.1 * context_alignment_loss

        # Add regularization (small weights to not dominate main losses)
        if self.train_colorstats:
            total_loss = total_loss + 0.05 * color_stats_loss  # Match color statistics
        if self.train_histogram:
            total_loss = total_loss + 0.02 * hist_loss  # Match color distributions
        if self.train_contrast:
            total_loss = total_loss + 0.1 * contrast_loss  # Prevent over-saturation
        if self.train_coarseness:
            total_loss = total_loss + 0.02 * coarseness_loss  # Match texture coarseness

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
                "ctx_aux_loss": 0.0,
                "_optimizer_stepped": False,  # Signal that optimizer was not stepped
            }

        # CRITICAL: Clear cache before backward to prevent OOM
        # Gradient checkpointing in VAE causes memory spikes during backward pass
        _empty_cache(real_imgs.device)

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

        accelerator_step = (
            self.accelerator is not None
            and hasattr(self.accelerator, "step")
            and callable(getattr(self.accelerator, "step"))
        )
        if accelerator_step:
            self.accelerator.step(self.optimizer)
            self.accelerator.step(self.context_predictor_optimizer)
        else:
            self.optimizer.step()
            self.context_predictor_optimizer.step()

        # Return dict matching original tuple behavior:
        # vae = recon_loss (NOT total_loss which includes adaptive weighting and can be huge/negative)
        return {
            "vae": float(recon_loss.detach().item()),
            "kl": float(kl.detach().item()),
            "generator": float(G_img_loss.detach().item()) if self.use_gan else 0.0,
            "lpips": float(perceptual_loss.detach().item()) if self.use_lpips else 0.0,
            "recon": float(recon_loss.detach().item()),
            # v0.10.0: auxiliary context reconstruction loss (plan §4.1)
            "ctx_aux_loss": float(ctx_aux_loss.detach().item()),
            "context_alignment": float(context_alignment_loss.detach().item()),
            "color_stats": float(color_stats_loss.detach().item()),
            "hist_loss": float(hist_loss.detach().item()),
            "contrast_loss": float(contrast_loss.detach().item()),
            "coarseness_loss": float(coarseness_loss.detach().item()),
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
