"""VAE training logic for FluxFlow.

Handles VAE (compressor + expander) training with optional GAN discriminator.
"""

import json
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Context encoder uses SiLU activation
from fluxflow.models.v100.conditioning import SPADE_v100b
from fluxflow.utils import get_logger
from fluxflow.utils.mps import mps_safe_pool2d
from torch.func import functional_call
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from .losses import (
    compute_bezier_monotonicity_reg,
    compute_ctx_shrinkage,
    cosine_warmup_weight,
    d_hinge_loss,
    delayed_cosine_warmup_weight,
    g_hinge_loss,
    kl_standard_normal,
    r1_penalty,
)
from .rocm_compat import r1_conv_context
from .schedulers import cosine_anneal_beta
from .utils import EMA, FloatBuffer

# v0.10.0: fixed weight for the Bezier control-point monotonicity regularizer
# (plan Fix 2). Structural sanity constraint, not a config knob -- matches the
# magnitude of this file's other fixed regularizers (color_stats=0.05,
# contrast=0.1, coarseness=0.02 below).
_BEZIER_REG_WEIGHT = 0.05

# Std floor for the contrast log-ratio (keeps it bounded on flat targets).
_STD_FLOOR = 0.05

# Fallbacks / bounds for the adaptive GAN weight and the KL warning threshold.
_KL_WARN_WEIGHT = 0.05  # kl_z_weight above this is almost certainly too large (per-element KL)
_ADAPTIVE_GAN_EPS = 1e-4
_CTX_DROP_SEED = 0x2D7
_ADAPTIVE_GAN_MAX = 1e4


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


# Free-memory fraction below which _empty_cache_if_low_memory releases the
# CUDA/ROCm allocator cache. Above it, releasing only forces the next backward
# to re-hipMalloc/cudaMalloc every block (~60% of step CPU time on Strix Halo).
_LOW_MEMORY_FREE_FRACTION = 0.15


def _empty_cache_if_low_memory(device: torch.device) -> None:
    """Free cached memory for the current device when it is running low.

    MPS always releases (no cheap free-memory query). CUDA/ROCm only release
    when free device memory drops below ``_LOW_MEMORY_FREE_FRACTION``.
    """
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        if free < _LOW_MEMORY_FREE_FRACTION * total:
            torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


class VAETrainer:
    """
    Handles VAE (Variational Autoencoder) training.

    Manages:
    - VAE reconstruction loss (L1 + frequency-weighted L1)
    - KL divergence with beta annealing
    - Optional GAN training with discriminator
    - EMA (Exponential Moving Average) updates

    Loss flag semantics (each flag is the SOLE gate for its loss):
    - ``train_vae`` / ``train_reconstruction``: L1 reconstruction loss + adaptive weight
    - ``use_lpips``: LPIPS perceptual anchor — independent of train_reconstruction
    - ``train_colorstats``: color statistics regularization — independent
    - ``train_histogram``: histogram matching regularization — independent
    - ``train_contrast``: contrast regularization — independent
    - ``train_coarseness``: coarseness distribution regularization — independent
    - ``train_kl``: KL divergence term — independent
    - ``train_ctx_aux``: auxiliary context reconstruction loss (v0.10.0) — independent
    - ``use_gan`` / ``gan_training``: adversarial loss via PatchDiscriminator

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
        scheduler: Optional[_LRScheduler],
        ema: EMA,
        reconstruction_loss_fn: nn.Module,
        reconstruction_loss_min_fn: nn.Module,
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
        train_ctx_aux: bool = False,
        # v0.10.0: latent cycle loss (image-space G.E idempotence): the compressor
        # re-encodes an expander output and the frozen expander must decode it back
        # to the same image. Compressor-only gradient; fixed (non-adaptive) weight.
        train_latent_cycle: bool = False,
        lambda_latent_cycle: float = 1.0,
        latent_cycle_lpips_weight: float = 0.0,
        latent_cycle_start_step: int = 0,
        latent_cycle_warmup_steps: int = 0,
        latent_cycle_every: int = 1,
        latent_cycle_batch_frac: float = 1.0,
        latent_cycle_crop_frac: float = 1.0,
        latent_cycle_posterior_frac: float = 0.0,
        latent_cycle_reject_range: float = 1.05,
        latent_cycle_reject_frac: float = 0.01,
        # v0.10.0: ctx-features L2 shrinkage (design §5.5). Active only when
        # ``ctx_shrinkage_weight > 0`` AND the underlying compressor exposes the
        # ``ctx_zinject_norm`` submodule (FluxCompressor_v100). The hook captures
        # the pre-norm ctx tensor (ctx_zinject_norm's input) and feeds it to
        # compute_ctx_shrinkage.
        ctx_shrinkage_weight: float = 0.0,
        ctx_shrinkage_warmup_start_step: int = 5000,
        ctx_shrinkage_warmup_steps: int = 5000,
        ctx_shrinkage_max_mean_sq: float = 1000.0,
        # v0.10.0: KL_z asymptotic weight and cosine warmup length. Default 0.0
        # so legacy callers using kl_beta / kl_warmup_steps stay on the old path.
        # When > 0 the trainer uses the wide-range schedule per design §5.5.
        kl_z_weight: float = 0.0,
        kl_z_warmup_steps: int = 10000,
        hf_ramp_steps: int = 30000,
        lambda_lpips: float = 0.1,
        instance_noise_std: float = 0.05,
        instance_noise_decay: float = 0.9999,
        adaptive_weights: bool = True,
        mse_weight: float = 0.1,
        accelerator=None,
        # Diagnostic: save discriminator spatial logit heatmaps every N global steps.
        # 0 = disabled (default). Writes PNG + JSONL to disc_logit_diagnostic_dir.
        disc_logit_diagnostic_interval: int = 0,
        disc_logit_diagnostic_dir: Optional[Union[str, Path]] = None,
        # Additive opt-in efficiency knob: skip discriminator forward+backward on
        # off-cycle steps. 1 (default) reproduces the historical every-step behavior.
        discriminator_update_freq: int = 1,
        # Gradient accumulation: generator (compressor+expander+context_predictor)
        # and discriminator optimizers both zero_grad()/step() only once every
        # gradient_accumulation_steps calls to train_step(), sharing one boundary
        # counter so they stay in lockstep -- mirrors FlowTrainer's
        # _accumulation_step / should_step pattern (flow_trainer.py). Default 1
        # reproduces the historical every-micro-batch-steps behavior.
        gradient_accumulation_steps: int = 1,
        # Adversarial weight is 0 before gan_start_step (D is not trained either),
        # then ramps (cosine) to lambda_adv over gan_warmup_steps.
        gan_start_step: int = 0,
        gan_warmup_steps: int = 1000,
        # v0.10.0 z-vs-ctx balance (training only, v100 compressors only):
        # ctx_dropout_frac: fraction of each batch decoded with use_context=False
        #   (exact GroupNorm, z only) so z must carry content; 0 disables.
        # ctx_noise_sigma: ctx tokens get N(0, (sigma*std_token)^2) noise before decode;
        #   scale-invariant, so shrinking ctx cannot escape it. Capacity of ctx is
        #   ~0.5*log2(1+1/sigma^2) bits/dim. 0 disables.
        ctx_dropout_frac: float = 0.5,
        ctx_noise_sigma: float = 0.5,
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
            train_reconstruction: Compute L1 reconstruction loss and its adaptive weight.
                Set to False for GAN-only training. Does NOT suppress LPIPS, colorstats,
                histogram, contrast, or coarseness — those are controlled by their own flags.
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
            lambda_adv: GAN adversarial loss weight. With ``adaptive_weights`` it scales the
                VQGAN-style ratio ||grad L_rec|| / ||grad L_gan|| at the expander's last layer;
                otherwise it is the fixed weight. Both are multiplied by the warmup factor.
            adaptive_weights: Adaptive GAN weight only; KL and recon are never adaptively weighted.
            kl_z_warmup_steps: Cosine warmup length of ``kl_z_weight`` in OPTIMIZER steps;
                the trainer scales it by ``gradient_accumulation_steps`` since ``global_step``
                counts micro-batches.
            hf_ramp_steps: Length (micro-batch global steps) of the HF-loss alpha ramp
                0.3 -> 1.2. 0 disables the ramp (alpha = 1.2 from the start).
            gan_start_step: Step before which the adversarial loss is off (and D untrained).
            gan_warmup_steps: Cosine ramp length 0 -> lambda_adv after gan_start_step.
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
            train_ctx_aux: Compute auxiliary context reconstruction loss (default: False).
            train_latent_cycle: Compute the latent cycle loss (default: False). The
                compressor re-encodes an expander output x_s; the frozen expander decodes
                it to x_hat; loss = L1(x_hat, x_s) (+ LPIPS). Only the compressor trains.
            lambda_latent_cycle: Fixed weight (outside adaptive weighting). Default 1.0.
            latent_cycle_lpips_weight: LPIPS weight inside the cycle loss (needs lpips_fn).
            latent_cycle_start_step: Weight is 0 before this step.
            latent_cycle_warmup_steps: Cosine ramp 0 -> lambda after start_step.
            latent_cycle_every: Compute only when global_step % every == 0.
            latent_cycle_batch_frac: Use the first ceil(frac * B) samples.
            latent_cycle_crop_frac: Random pixel crop fraction, sizes/offsets aligned to
                2**compressor.downscales (min one latent cell).
            latent_cycle_posterior_frac: Share of the cycle batch sourced from a fresh
                posterior sample of real images (rest reuses the main-path recon).
            latent_cycle_reject_range: |x| threshold of the off-manifold guard.
            latent_cycle_reject_frac: Drop a sample when more than this fraction of its
                pixels exceed reject_range. No clamping anywhere.
            ctx_shrinkage_max_mean_sq: Clamp on mean(ctx_features**2) before scaling by
                the shrinkage alpha (see losses.compute_ctx_shrinkage). Default 1000.0.
            accelerator: Accelerate accelerator instance
            discriminator_update_freq: Run discriminator forward+backward every N
                global steps (default 1 = every step, current behavior). Values > 1
                skip it on off-cycle steps as an opt-in efficiency knob; VAE/generator
                training is unaffected.
            gradient_accumulation_steps: Number of micro-batches to accumulate gradients
                over before zero_grad()/optimizer.step() fire. The generator and the
                discriminator each keep their own counter (the discriminator only counts
                micro-batches on which it runs, see discriminator_update_freq). Default 1
                steps every micro-batch.
        """
        self.compressor = compressor
        self.expander = expander
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema

        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.reconstruction_loss_min_fn = reconstruction_loss_min_fn

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
        # v0.10.0 KL_z schedule (used when kl_z_weight > 0; else legacy kl_beta path)
        self.kl_z_weight = kl_z_weight
        self.kl_z_warmup_steps = kl_z_warmup_steps
        # v0.10.0 ctx shrinkage schedule
        self.ctx_shrinkage_weight = ctx_shrinkage_weight
        self.ctx_shrinkage_warmup_start_step = ctx_shrinkage_warmup_start_step
        self.ctx_shrinkage_warmup_steps = ctx_shrinkage_warmup_steps
        self.ctx_shrinkage_max_mean_sq = ctx_shrinkage_max_mean_sq
        self.ctx_dropout_frac = ctx_dropout_frac
        self.ctx_noise_sigma = ctx_noise_sigma
        # Dedicated CPU generator: B==1 z-only draws are reproducible and leave the
        # global RNG (data order, dropout, noise) untouched.
        self._ctx_drop_gen = torch.Generator().manual_seed(_CTX_DROP_SEED)
        # Hook state: last captured ctx_features (pre-norm, pre-attention).
        self._ctx_features_cache: Optional[torch.Tensor] = None
        self._ctx_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        # Cleared while secondary compressor forwards run so they can't overwrite
        # the real-image ctx features (see _suspend_ctx_capture).
        self._ctx_capture_enabled = True

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
        self.disc_logit_diagnostic_interval = disc_logit_diagnostic_interval
        self.disc_logit_diagnostic_dir = (
            Path(disc_logit_diagnostic_dir) if disc_logit_diagnostic_dir is not None else None
        )
        if discriminator_update_freq < 1:
            raise ValueError(
                f"discriminator_update_freq must be >= 1, got {discriminator_update_freq}"
            )
        self.discriminator_update_freq = discriminator_update_freq
        # Gradient accumulation: independent boundary counters (generator /
        # discriminator), see FlowTrainer._accumulation_step in flow_trainer.py.
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self._accumulation_step = 0
        self._d_accumulation_step = 0
        self.hf_ramp_steps = hf_ramp_steps
        self._kl_warned = False
        self.gan_start_step = gan_start_step
        self.gan_warmup_steps = gan_warmup_steps
        self._last_vae_sched_step = -1
        self._last_d_sched_step = -1
        self._scaler_unscaled = False
        self._adaptive_gan_warned = False
        self._last_w_gan = 0.0

        # Initialize LPIPS if needed
        self.lpips_fn = None
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

        if train_latent_cycle and latent_cycle_lpips_weight > 0 and self.lpips_fn is None:
            logger.warning(
                "latent_cycle_lpips_weight > 0 but use_lpips=False; LPIPS term inactive."
            )

        # Metrics buffers
        self.vae_loss_buffer = FloatBuffer(max_items=20)
        self.kl_loss_buffer = FloatBuffer(max_items=20)
        self.d_loss_buffer = FloatBuffer(max_items=20)
        self.g_loss_buffer = FloatBuffer(max_items=20)
        self.lpips_loss_buffer = FloatBuffer(max_items=20)

        # Context predictor settings
        self.context_channels = context_channels
        self.context_height = context_height
        self.context_width = context_width
        self.context_predictor_path = context_predictor_path
        self.lambda_ctx_aux = lambda_ctx_aux
        self.train_ctx_aux = train_ctx_aux
        # v0.10.0: latent cycle loss (independent VAE loss, fixed weight).
        self.train_latent_cycle = train_latent_cycle
        self.lambda_latent_cycle = lambda_latent_cycle
        self.latent_cycle_lpips_weight = latent_cycle_lpips_weight
        self.latent_cycle_start_step = latent_cycle_start_step
        self.latent_cycle_warmup_steps = latent_cycle_warmup_steps
        self.latent_cycle_every = latent_cycle_every
        self.latent_cycle_batch_frac = latent_cycle_batch_frac
        self.latent_cycle_crop_frac = latent_cycle_crop_frac
        self.latent_cycle_posterior_frac = latent_cycle_posterior_frac
        self.latent_cycle_reject_range = latent_cycle_reject_range
        self.latent_cycle_reject_frac = latent_cycle_reject_frac

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
                checkpoint = torch.load(
                    context_predictor_path, map_location="cpu", weights_only=True
                )
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

        # Install forward pre-hook on ctx_zinject_norm to capture the pre-norm
        # ctx features (the input to compute_ctx_shrinkage) without forcing a
        # signature change on the compressor. Only the v0.10.0 compressor has
        # this submodule, so the hook is installed conditionally.
        self._install_ctx_shrinkage_hook()

    def _install_ctx_shrinkage_hook(self) -> None:
        """Install a forward pre-hook capturing the ctx_zinject_norm input.

        ``ctx_zinject_norm`` is a ``GroupNorm(affine=False)``, so its *output*
        is forced to ~unit variance by construction - a shrinkage loss on the
        output would be a near-zero-gradient no-op. The pre-hook instead
        captures the unconstrained pre-norm ``cx`` tensor (the real input to
        ``ctx_zinject_norm`` inside ``FluxCompressor_v100.forward``), so the
        shrinkage term has real gradient to act on.

        The hook stores the tensor on ``self._ctx_features_cache`` each forward
        pass (unless capture is suspended, see ``_suspend_ctx_capture``).
        Subsequent ``_train_generator`` reads and clears the cache so the
        shrinkage term operates on the real-image ctx features. No-op when the
        compressor doesn't expose ``ctx_zinject_norm``.
        """
        if self.ctx_shrinkage_weight <= 0:
            return
        unwrapped = self._get_unwrapped_model(self.compressor)
        ctx_norm = getattr(unwrapped, "ctx_zinject_norm", None)
        if ctx_norm is None:
            logger.warning(
                "compressor has no 'ctx_zinject_norm' submodule; ctx_shrinkage "
                "loss will be inactive even though ctx_shrinkage_weight > 0."
            )
            return

        def _hook(_module, args):
            if self._ctx_capture_enabled:
                self._ctx_features_cache = args[0]

        self._ctx_hook_handle = ctx_norm.register_forward_pre_hook(_hook)

    @contextmanager
    def _suspend_ctx_capture(self) -> Iterator[None]:
        """Disable ctx-feature capture; wrap every secondary compressor forward."""
        prev = self._ctx_capture_enabled
        self._ctx_capture_enabled = False
        try:
            yield
        finally:
            self._ctx_capture_enabled = prev

    def remove_ctx_shrinkage_hook(self) -> None:
        """Detach the ctx shrinkage forward hook if installed."""
        if self._ctx_hook_handle is not None:
            self._ctx_hook_handle.remove()
            self._ctx_hook_handle = None

    def _autocast(self):
        """Mixed-precision context for the expensive forward passes (compressor,
        expander, discriminator). ``accelerator.autocast()`` is a documented
        standalone Accelerate API -- it does not require ``accelerator.prepare()``
        to have been called on the wrapped models (see Accelerate's own
        ``Accelerator.autocast`` docstring), which matters here because
        VAETrainer's models are never ``.prepare()``'d. No-op (fp32, matching
        historical behavior) when mixed_precision is disabled or no accelerator
        was provided.
        """
        if self.accelerator is None:
            return nullcontext()
        return self.accelerator.autocast()

    def _should_train_discriminator(self, global_step: int) -> bool:
        """Whether discriminator forward+backward should run this step.

        Gated by ``discriminator_update_freq`` (default 1 = every step, the
        historical behavior). Values > 1 skip the discriminator on off-cycle
        steps as an opt-in efficiency knob; VAE/generator training is unaffected.
        """
        return (
            self.use_gan
            and self.discriminator is not None
            and global_step >= self.gan_start_step
            and global_step % self.discriminator_update_freq == 0
        )

    def _gan_factor(self, global_step: int) -> float:
        """Adversarial warmup factor in [0, 1]: 0 before gan_start_step, cosine ramp after."""
        return delayed_cosine_warmup_weight(
            global_step, self.gan_start_step, self.gan_warmup_steps, 1.0
        )

    def _compute_spade_drift(self) -> tuple[float, float]:
        """Mean SPADE γ/β drift from init, averaged across SPADE_v100b blocks
        in the expander.

        Independent of the disconnected ``ctx_probe_alignment`` diagnostic — this
        reads gradient-trained ``gamma_scale``/``beta_scale`` parameters directly,
        so nonzero drift is a real signal that SPADE conditioning is learning.
        Returns ``(0.0, 0.0)`` when the expander has no SPADE_v100b blocks (e.g.
        legacy v060/v070 models).
        """
        gamma_drifts: list[float] = []
        beta_drifts: list[float] = []
        for module in self.expander.modules():
            if isinstance(module, SPADE_v100b):
                gamma_drift, beta_drift = module.scale_drift()
                gamma_drifts.append(gamma_drift)
                beta_drifts.append(beta_drift)
        if not gamma_drifts:
            return 0.0, 0.0
        return sum(gamma_drifts) / len(gamma_drifts), sum(beta_drifts) / len(beta_drifts)

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
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
        """SPADE is always active; retained for compatibility.

        Args:
            global_step: Current training step number (unused)

        Returns:
            Always True
        """
        return True

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

    def _histogram_matching_loss(self, pred, target):
        """
        Encourage matching per-channel color distributions between pred and target.

        Differentiable 1D Wasserstein-1 distance between the channels' empirical
        value distributions (sorted-value matching), normalised by the [-1, 1]
        range. Unlike a hard histogram (``torch.histc``) gradients reach ``pred``.

        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Histogram matching loss
        """
        loss = torch.zeros((), device=pred.device)
        for c in range(3):  # R, G, B channels
            pred_sorted = pred[:, c].reshape(-1).sort().values
            target_sorted = target[:, c].reshape(-1).sort().values
            loss = loss + (pred_sorted - target_sorted).abs().mean() / 2.0
        return loss / 3.0

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
        loss = torch.zeros((), device=pred.device)

        # Component 1: Per-channel std ratio (global contrast preservation)
        for c in range(3):
            pred_std = pred[:, c].std()
            target_std = target[:, c].std()
            # Smoothed log-ratio: bounded on flat targets (std ~ 0), zero when equal.
            loss += torch.log((pred_std + _STD_FLOOR) / (target_std + _STD_FLOOR)) ** 2

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

    def _kl_z_beta(self, global_step: int) -> float:
        """KL_z weight at micro-batch ``global_step``; warmup is in optimizer steps."""
        warmup = self.kl_z_warmup_steps * self.gradient_accumulation_steps
        return cosine_warmup_weight(global_step, warmup, self.kl_z_weight)

    def _warn_large_kl_once(self, packed: torch.Tensor) -> None:
        """One-time warning: v100 KL is a per-element mean with fixed weight."""
        if self._kl_warned or not self.train_kl or self.kl_z_weight <= _KL_WARN_WEIGHT:
            return
        if self._has_ctx_half(packed):
            self._kl_warned = True
            logger.warning(
                f"kl_z_weight={self.kl_z_weight} is large: KL is a per-element mean with a "
                f"fixed weight (no adaptive balancing); values > {_KL_WARN_WEIGHT} can "
                "collapse z toward the prior. Typical: 1e-3."
            )

    def _has_ctx_half(self, packed: torch.Tensor) -> bool:
        """True for v0.10.0 packed tokens (z||ctx, even width) with SPADE context active."""
        model = self._get_unwrapped_model(self.compressor)
        model = getattr(model, "_orig_mod", model)
        return getattr(model, "ctx_zinject_norm", None) is not None and packed.size(-1) % 2 == 0

    def _ctx_dropout_count(self, batch: int) -> int:
        """Number of trailing samples decoded z-only (use_context=False).

        B >= 2: clamp(round(frac*B), 1, B-1), i.e. both halves non-empty. B == 1: all
        (1) with probability frac, drawn from the dedicated generator.
        """
        frac = self.ctx_dropout_frac
        if frac <= 0:
            return 0
        if batch == 1:
            return int(torch.rand(1, generator=self._ctx_drop_gen).item() < frac)
        return min(max(round(frac * batch), 1), batch - 1)

    def _add_ctx_noise(self, packed: torch.Tensor) -> torch.Tensor:
        """ctx += sigma * std_token(ctx) * eps on image tokens; z half and HW row untouched.

        The per-token std is detached: noise stays proportional to ctx's own scale (so
        rescaling ctx cannot escape it) without a gradient path that would reward
        lowering std to shrink the noise itself. Uses the global RNG (no reproducibility
        contract needed for noise).
        """
        half = packed.size(-1) // 2
        tokens = packed[:, :-1]
        z, ctx = tokens[..., :half], tokens[..., half:]
        std = ctx.detach().float().std(dim=-1, keepdim=True).to(ctx.dtype)
        ctx = ctx + self.ctx_noise_sigma * std * torch.randn_like(ctx)
        return torch.cat([torch.cat([z, ctx], dim=-1), packed[:, -1:]], dim=1)

    def _decode_split(self, packed: torch.Tensor, use_context: bool) -> tuple[torch.Tensor, int]:
        """Decode ``packed``; returns ``(images, n_ctx)`` (first n_ctx are ctx-decoded)."""
        B = packed.size(0)
        v100 = use_context and self._has_ctx_half(packed)
        n_z = self._ctx_dropout_count(B) if v100 else 0
        n_ctx = B - n_z
        outs = []
        if n_ctx:
            ctx_in = packed[:n_ctx]
            if self.ctx_noise_sigma > 0 and v100:
                ctx_in = self._add_ctx_noise(ctx_in)
            outs.append(self.expander(ctx_in, use_context=use_context))
        if n_z:
            outs.append(self.expander(packed[n_ctx:], use_context=False))
        return (outs[0] if len(outs) == 1 else torch.cat(outs)), n_ctx

    @staticmethod
    def _recon_l1_halves(out: torch.Tensor, real: torch.Tensor, n_ctx: int) -> dict[str, float]:
        """Detached plain L1 of the ctx-decoded / z-only halves (key absent when empty)."""
        diag = {}
        for key, sl in (("recon_l1_ctx", slice(0, n_ctx)), ("recon_l1_zonly", slice(n_ctx, None))):
            if out[sl].shape[0]:
                diag[key] = float(F.l1_loss(out[sl].detach().float(), real[sl].float()).item())
        return diag

    def _expander_last_weight(self) -> Optional[torch.Tensor]:
        """Weight of the expander's last (RGB) conv, or None when it cannot be found."""
        model = self._get_unwrapped_model(self.expander)
        model = getattr(model, "_orig_mod", model)
        head = getattr(model, "to_rgb_conv", None)
        if head is None:
            head = getattr(model, "to_rgb", model)
        convs = [m for m in head.modules() if isinstance(m, nn.Conv2d)]
        if head is model:
            convs = [m for m in convs if m.out_channels == 3]
        if not convs or not convs[-1].weight.requires_grad:
            return None
        return convs[-1].weight

    def _warn_adaptive_once(self, reason: str) -> None:
        if not self._adaptive_gan_warned:
            self._adaptive_gan_warned = True
            logger.warning(f"Adaptive GAN weight unavailable ({reason}); using fixed lambda_adv.")

    def _adaptive_gan_weight(
        self, rec_loss: torch.Tensor, gan_loss: torch.Tensor, last_weight: Optional[torch.Tensor]
    ) -> Optional[float]:
        """VQGAN weight ||grad_last L_rec|| / (||grad_last L_gan|| + eps), clamped, detached.

        Returns None (one-time warning) when it cannot be computed; callers then use the
        fixed weight. Ratio of grads of unscaled losses, so fp16 loss scaling is irrelevant.
        """
        if last_weight is None:
            self._warn_adaptive_once("last layer not found")
            return None
        if not (rec_loss.requires_grad and gan_loss.requires_grad):
            self._warn_adaptive_once("loss has no grad path")
            return None
        try:
            g_rec = torch.autograd.grad(rec_loss, last_weight, retain_graph=True)[0]
            g_gan = torch.autograd.grad(gan_loss, last_weight, retain_graph=True)[0]
        except RuntimeError as exc:
            self._warn_adaptive_once(str(exc).splitlines()[0])
            return None
        w = (g_rec.norm() / (g_gan.norm() + _ADAPTIVE_GAN_EPS)).detach()
        if not torch.isfinite(w):
            self._warn_adaptive_once("non-finite gradient ratio")
            return None
        return float(w.clamp(0.0, _ADAPTIVE_GAN_MAX).item())

    def _maybe_save_disc_logit_snapshot(self, real_logits: torch.Tensor, global_step: int) -> None:
        """Save discriminator spatial logit heatmap + JSONL record if diagnostic is enabled.

        Args:
            real_logits: Discriminator output [B, 1, H_out, W_out] from real images.
            global_step: Current global training step.
        """
        if self.disc_logit_diagnostic_interval <= 0:
            return
        if global_step % self.disc_logit_diagnostic_interval != 0:
            return

        try:
            diag_dir = self.disc_logit_diagnostic_dir
            if diag_dir is None:
                return
            diag_dir.mkdir(parents=True, exist_ok=True)

            logits_np = real_logits.detach().cpu().float().numpy()  # [B, 1, H, W]
            spatial = logits_np[:, 0, :, :].mean(axis=0)  # [H, W] — mean over batch

            h, w = spatial.shape
            row_mean = spatial.mean(axis=1).tolist()  # length H
            col_mean = spatial.mean(axis=0).tolist()  # length W

            record = {
                "step": global_step,
                "shape": [h, w],
                "mean": float(spatial.mean()),
                "std": float(spatial.std()),
                "min": float(spatial.min()),
                "max": float(spatial.max()),
                "row_mean": row_mean,
                "col_mean": col_mean,
            }
            jsonl_path = diag_dir / "disc_logits.jsonl"
            with open(jsonl_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")

            # PNG heatmap
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5, 5))
                vmin, vmax = record["min"], record["max"]
                im = ax.imshow(spatial, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
                fig.colorbar(im, ax=ax)
                ax.set_title(f"D logits — step {global_step}")
                png_path = diag_dir / f"disc_logits_step{global_step:08d}.png"
                fig.savefig(str(png_path), bbox_inches="tight", dpi=80)
                plt.close(fig)
            except Exception:
                # Fallback: PIL grayscale when matplotlib unavailable
                from PIL import Image

                norm = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
                arr = (norm * 255).astype(np.uint8)
                png_path = diag_dir / f"disc_logits_step{global_step:08d}.png"
                Image.fromarray(arr).save(str(png_path))

        except Exception as exc:
            logger.warning("disc_logit_diagnostic failed at step %d: %s", global_step, exc)

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
        discriminator_was_stepped: bool = False
        self._scaler_unscaled = False

        try:
            # VQGAN order: generator first, then D on the detached G-step reconstruction.
            gen_losses = self._train_generator(real_imgs, global_step)
            fake_imgs = gen_losses.pop("_fake_imgs", None)
            if fake_imgs is not None and self._should_train_discriminator(global_step):
                if self.discriminator_optimizer is None:
                    raise RuntimeError(
                        "VAETrainer.use_gan=True but discriminator_optimizer is None. "
                        "This should have been caught in __init__. Please report this bug."
                    )
                d_result = self._train_discriminator(
                    real_imgs[: fake_imgs.shape[0]], global_step, fake_imgs
                )
                discriminator_was_stepped = bool(d_result["_optimizer_stepped"])
                if discriminator_was_stepped:
                    d_loss = d_result["d_loss"]
                    losses["discriminator"] = d_loss
                    self.d_loss_buffer.add_item(d_loss)
        finally:
            self._finish_scaler()

        losses["vae"] = gen_losses["vae"]  # recon_loss
        losses["recon"] = gen_losses["recon"]  # same as vae
        losses["kl"] = gen_losses["kl"]
        # Report whichever schedule is active so dashboards see the live coefficient.
        if self.kl_z_weight > 0:
            losses["kl_beta"] = self._kl_z_beta(global_step)
        else:
            losses["kl_beta"] = cosine_anneal_beta(global_step, self.kl_warmup_steps, self.kl_beta)

        for key in ("recon_l1_ctx", "recon_l1_zonly"):  # absent when that half is empty
            if key in gen_losses:
                losses[key] = gen_losses[key]

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
        # v0.10.0: latent cycle loss (unweighted) and off-manifold reject fraction
        losses["latent_cycle_loss"] = gen_losses.get("latent_cycle_loss", 0.0)
        losses["latent_cycle_reject_frac"] = gen_losses.get("latent_cycle_reject_frac", 0.0)
        # v0.10.0: gradient-carrying ctx-shrinkage term (see ctx_shrinkage_weight).
        # Reads 0.0 unless the caller explicitly wires a nonzero weight in.
        losses["ctx_shrinkage_loss"] = gen_losses.get("ctx_shrinkage_loss", 0.0)
        losses["ctx_shrinkage_alpha"] = gen_losses.get("ctx_shrinkage_alpha", 0.0)
        # Diagnostic probe only — no gradient into SPADE/compressor/expander. See
        # the comment on context_alignment_loss in _train_generator for why this
        # plateaus near 1.0 regardless of optimizer/training quality.
        losses["ctx_probe_alignment"] = gen_losses.get("ctx_probe_alignment", 0.0)
        # First real, gradient-connected signal of whether SPADE conditioning is
        # learning anything — independent of the disconnected probe above.
        spade_gamma_drift, spade_beta_drift = self._compute_spade_drift()
        losses["spade_gamma_drift"] = spade_gamma_drift
        losses["spade_beta_drift"] = spade_beta_drift

        # Check if optimizer was actually stepped (could be skipped due to NaN)
        optimizer_was_stepped = gen_losses.pop("_optimizer_stepped", True)

        # Update EMA only if optimizer was stepped
        if optimizer_was_stepped and self.ema is not None:
            self.ema.update()

        # Step schedulers at most once per distinct global_step (the orchestrator calls
        # train_step once per resolution with the same step), only after a real optimizer step.
        plateau_metric = gen_losses.pop("_plateau_metric", gen_losses["vae"])
        if optimizer_was_stepped and global_step > 0 and global_step != self._last_vae_sched_step:
            self._last_vae_sched_step = global_step
            self._step_scheduler(self.scheduler, plateau_metric)
        if discriminator_was_stepped and global_step > 0 and global_step != self._last_d_sched_step:
            self._last_d_sched_step = global_step
            self._step_scheduler(self.discriminator_scheduler, losses.get("discriminator", 0.0))

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
                # Loss weights: recon/KL are fixed; weight_gan is the effective
                # multiplier on the generator hinge loss (adaptive x lambda_adv x warmup).
                "weight_recon": 1.0,
                "weight_kl": 1.0,
                "weight_gan": self._last_w_gan if self.use_gan else 0.0,
            }
        )

        return losses

    @staticmethod
    def _step_scheduler(scheduler, metric: float) -> None:
        """Step a scheduler (ReduceLROnPlateau takes a metric); None is a no-op."""
        if scheduler is None:
            return
        base = getattr(scheduler, "scheduler", scheduler)
        if isinstance(base, ReduceLROnPlateau):
            scheduler.step(float(metric))
        else:
            scheduler.step()

    def _unscale_clip(self, optimizer: Optimizer, params: list, name: str) -> bool:
        """Unscale (fp16), clip, and report whether grads are finite (False -> skip step)."""
        self.accelerator.unscale_gradients(optimizer)
        self._scaler_unscaled = True
        norm = torch.nn.utils.clip_grad_norm_(params, self.gradient_clip_norm)
        if not bool(torch.isfinite(norm)):
            logger.warning(f"Non-finite {name} gradients - skipping optimizer step")
            return False
        return True

    def _finish_scaler(self) -> None:
        """Call GradScaler.update() once per iteration, after both optimizers."""
        scaler = getattr(self.accelerator, "scaler", None)
        if self._scaler_unscaled and scaler is not None:
            scaler.update()
        self._scaler_unscaled = False

    def _train_discriminator(
        self,
        real_imgs: torch.Tensor,
        global_step: int,
        fake_imgs: torch.Tensor,
    ) -> dict[str, Any]:
        """Train discriminator on real images and the generator's reconstructions.

        ``fake_imgs`` is the G step's reconstruction (VQGAN order); it is detached
        here, so no extra compressor/expander forward is needed and the VAE gets no
        gradient. The D accumulation counter only counts micro-batches where D runs.

        Returns:
            Dict with ``d_loss`` (float) and ``_optimizer_stepped`` (bool).
            ``_optimizer_stepped`` is ``False`` mid-accumulation, on NaN/Inf loss or
            gradients, and on a ROCm/MIOpen kernel-compile failure (a known upstream
            cold-start bug); in the last cases ``d_loss`` is a placeholder.
        """
        self.discriminator.train()

        accum = self.gradient_accumulation_steps
        should_step = (self._d_accumulation_step + 1) % accum == 0
        if self._d_accumulation_step == 0:
            self.discriminator_optimizer.zero_grad(set_to_none=True)

        # Instance noise on both inputs. SPADE is always active, so D is unconditional.
        real_imgs_noisy = add_instance_noise(
            real_imgs, self.instance_noise_std, self.instance_noise_decay, global_step
        )
        fake_imgs_noisy = add_instance_noise(
            fake_imgs.detach(), self.instance_noise_std, self.instance_noise_decay, global_step
        )
        disc_ctx = None

        do_r1 = (global_step % self.r1_interval) == 0

        # Only keep grad graph on real images when R1 penalty is needed
        real_noisy_base = real_imgs_noisy.detach()
        if do_r1:
            real_noisy_base = real_noisy_base.requires_grad_(True)

        # Real images. On ROCm R1 steps the real forward, R1 term and backward
        # run on native conv kernels (see r1_conv_context).
        with r1_conv_context(do_r1), self._autocast():
            real_logits = self.discriminator(real_noisy_base, disc_ctx)
        self._maybe_save_disc_logit_snapshot(real_logits, global_step)

        d_img_loss = torch.tensor(0.0, device=real_imgs.device)

        # Lazy R1 (every r1_interval steps): scaled by the interval so the expected
        # penalty per step matches applying it every step.
        if do_r1:
            with r1_conv_context(do_r1):
                r1 = r1_penalty(real_noisy_base, real_logits)
            d_img_loss = d_img_loss + (self.r1_gamma * 0.5) * self.r1_interval * r1

        with self._autocast():
            fake_logits = self.discriminator(fake_imgs_noisy, disc_ctx)

        d_hinge = d_hinge_loss(real_logits, fake_logits)
        d_img_loss = d_img_loss + d_hinge

        # Gradient accumulation: scale before backward (standard grad-accum).
        d_img_loss = d_img_loss / accum

        # Check for NaN/Inf in loss with detailed diagnostics (mirrors
        # _train_generator's NaN guard). r1_penalty (double-backward) and
        # d_hinge_loss (unbounded above) can both blow up.
        if check_for_nan(d_img_loss, "d_img_loss", logger):
            logger.error("Skipping discriminator batch due to NaN/Inf in d_img_loss")
            if do_r1:
                logger.error(f"  r1: {r1.item() if not check_for_nan(r1, 'r1', logger) else 'NaN'}")
            logger.error(
                f"  d_hinge: {d_hinge.item() if not check_for_nan(d_hinge, 'd_hinge', logger) else 'NaN'}"
            )
            logger.error(
                f"  real_logits stats: min={real_logits.min().item():.4f}, max={real_logits.max().item():.4f}"
            )
            logger.error(
                f"  fake_logits stats: min={fake_logits.min().item():.4f}, max={fake_logits.max().item():.4f}"
            )
            return {"d_loss": 0.0, "_optimizer_stepped": False}

        try:
            with r1_conv_context(do_r1):
                self.accelerator.backward(d_img_loss)
        except RuntimeError as exc:
            if "miopen" in str(exc).lower():
                print(
                    "Skipped discriminator step: MIOpen kernel-compile failure in "
                    f"backward() | global_step={global_step} | "
                    f"real_imgs shape={tuple(real_imgs.shape)} | "
                    f"real_imgs_noisy shape={tuple(real_imgs_noisy.shape)} | "
                    f"error={exc}"
                )
                return {"d_loss": 0.0, "_optimizer_stepped": False}
            raise

        d_loss = float(d_img_loss.detach().item())
        if not should_step:
            # Mid-accumulation-window: gradients stay accumulated in .grad.
            self._d_accumulation_step += 1
            return {"d_loss": d_loss, "_optimizer_stepped": False}
        self._d_accumulation_step = 0

        # discriminator_optimizer is a plain torch optimizer: unscale explicitly, clip
        # (same max norm as G), and skip the step on non-finite grads (no GradScaler
        # auto-skip here). scaler.update() is called once per iteration in train_step.
        d_params = list(self.discriminator.parameters())
        assert self.discriminator_optimizer is not None
        if not self._unscale_clip(self.discriminator_optimizer, d_params, "discriminator"):
            return {"d_loss": d_loss, "_optimizer_stepped": False}
        self.discriminator_optimizer.step()
        return {"d_loss": d_loss, "_optimizer_stepped": True}

    def _latent_cycle_lambda(self, global_step: int) -> float:
        """Effective cycle weight: 0 when disabled/before start, cosine ramp to lambda."""
        if not self.train_latent_cycle:
            return 0.0
        return delayed_cosine_warmup_weight(
            global_step,
            self.latent_cycle_start_step,
            self.latent_cycle_warmup_steps,
            self.lambda_latent_cycle,
        )

    def _latent_cycle_crop_window(self, H: int, W: int, unit: int) -> tuple[int, int, int, int]:
        """Random ``(top, left, h, w)`` crop; sizes floored and offsets aligned to ``unit``."""

        def _axis(size: int) -> tuple[int, int]:
            length = min(max(int(size * self.latent_cycle_crop_frac) // unit, 1) * unit, size)
            slots = (size - length) // unit + 1
            return int(torch.randint(0, slots, (1,)).item()) * unit, length

        top, h = _axis(H)
        left, w = _axis(W)
        return top, left, h, w

    def _latent_cycle_keep_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample bool mask: False when too many pixels lie outside ``reject_range``."""
        out_frac = (x.detach().float().abs() > self.latent_cycle_reject_range).flatten(1)
        return out_frac.float().mean(dim=1) <= self.latent_cycle_reject_frac

    def _frozen_expander_forward(self, packed: torch.Tensor, use_context: bool) -> torch.Tensor:
        """Expander forward with detached params (frozen, no ``requires_grad`` mutation).

        Grad still flows to ``packed``; gradient checkpointing stays active.
        """
        model = self._get_unwrapped_model(self.expander)
        params = {n: p.detach() for n, p in model.named_parameters()}
        out = functional_call(model, params, (packed,), {"use_context": use_context})
        return cast(torch.Tensor, out)

    def _compute_latent_cycle(
        self, real_imgs: torch.Tensor, out_imgs_rec: torch.Tensor, global_step: int
    ) -> tuple[torch.Tensor, float]:
        """Image-space G.E idempotence loss. Returns ``(loss, reject_frac)``.

        Targets x_s (no grad): C1a = cropped detached main-path recon (``out_imgs_rec`` must
        hold ctx-decoded samples only; a shortfall is filled with C1b), C1b = expander
        decode of a fresh posterior sample of the cropped real images. The compressor
        re-encodes x_s (with grad); the expander, frozen for this call only, decodes it
        to x_hat. Loss = L1(x_hat, x_s) + lpips_weight * LPIPS. Non-finite -> 0.
        """
        device = real_imgs.device
        zero = torch.zeros((), device=device)
        B, _, H, W = real_imgs.shape
        n = min(max(math.ceil(self.latent_cycle_batch_frac * B), 1), B)
        n_post = min(int(self.latent_cycle_posterior_frac * n + 0.5), n)
        n_rec = min(n - n_post, out_imgs_rec.shape[0])  # only ctx-half recons are valid targets
        n_post = n - n_rec
        unit = 2 ** getattr(self._get_unwrapped_model(self.compressor), "downscales", 0)
        top, left, h, w = self._latent_cycle_crop_window(H, W, unit)
        rows, cols = slice(top, top + h), slice(left, left + w)
        spade = self._get_effective_spade_usage(global_step)

        parts = []
        with torch.no_grad(), self._autocast():
            if n_rec:
                parts.append(out_imgs_rec[:n_rec, :, rows, cols].detach())
            if n_post:
                with self._suspend_ctx_capture():
                    packed_b = self.compressor(real_imgs[n_rec:n, :, rows, cols], training=True)[0]
                parts.append(self.expander(packed_b, use_context=spade))
        x_s = torch.cat([p.float() for p in parts])

        keep = self._latent_cycle_keep_mask(x_s)
        reject_frac = 1.0 - float(keep.float().mean().item())
        if not bool(keep.any()):
            return zero, reject_frac
        x_s = x_s[keep]

        with self._autocast(), self._suspend_ctx_capture():
            packed_cycle = self.compressor(x_s, training=True)[0]
        with self._autocast():
            x_hat = self._frozen_expander_forward(packed_cycle, spade)

        x_hat = x_hat.float()
        if not (torch.isfinite(packed_cycle).all() and torch.isfinite(x_hat).all()):
            logger.warning(
                "latent_cycle: non-finite re-encoded latent / decode — "
                "forcing latent_cycle_loss to 0.0 for this step."
            )
            return zero, reject_frac
        loss = F.l1_loss(x_hat, x_s)
        if self.latent_cycle_lpips_weight > 0 and self.lpips_fn is not None:
            lpips_fn = self.lpips_fn.to(device)
            loss = loss + self.latent_cycle_lpips_weight * lpips_fn(x_hat, x_s).mean()
        return loss, reject_frac

    def _train_generator(
        self,
        real_imgs: torch.Tensor,
        global_step: int,
    ) -> dict[str, Any]:
        """Train VAE generator (compressor + expander).

        Each loss component is gated solely by its own flag:
        - recon_loss: gated on ``train_reconstruction``
        - perceptual_loss (LPIPS): gated on ``use_lpips`` — independent of train_reconstruction
        - kl: gated on ``train_kl``
        - G_img_loss: gated on ``use_gan``
        - ctx_aux_loss: gated on ``train_ctx_aux``
        - latent_cycle_loss: gated on ``train_latent_cycle`` (+ start/every/warmup)
        - color_stats_loss: gated on ``train_colorstats``
        - hist_loss: gated on ``train_histogram``
        - contrast_loss: gated on ``train_contrast``
        - coarseness_loss: gated on ``train_coarseness``

        Returns:
            Dictionary with loss values keyed by ``vae``, ``kl``, ``generator``,
            ``lpips``, ``recon``, ``ctx_aux_loss``, ``latent_cycle_loss``,
            ``latent_cycle_reject_frac``, ``ctx_probe_alignment``, ``color_stats``,
            ``hist_loss``, ``contrast_loss``, ``coarseness_loss``.
        """
        # Gradient accumulation: shared boundary counter with _train_discriminator
        # (see __init__ / gradient_accumulation_steps). zero_grad only at the
        # start of a fresh accumulation window so earlier micro-batches'
        # accumulated gradients survive.
        is_accum_boundary_start = self._accumulation_step == 0
        if is_accum_boundary_start:
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
                "latent_cycle_loss": 0.0,
                "latent_cycle_reject_frac": 0.0,
                "_optimizer_stepped": False,
            }

        # Forward pass with reparameterization
        with self._autocast():
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
                "latent_cycle_loss": 0.0,
                "latent_cycle_reject_frac": 0.0,
                "_optimizer_stepped": False,
            }

        self._warn_large_kl_once(packed_rec)
        with self._autocast():
            out_imgs_rec, n_ctx = self._decode_split(
                packed_rec, self._get_effective_spade_usage(global_step)
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
                "latent_cycle_loss": 0.0,
                "latent_cycle_reject_frac": 0.0,
                "_optimizer_stepped": False,
            }

        B_total = real_imgs.shape[0]

        # Reconstruction loss (skip if train_reconstruction=False)
        recon_loss = torch.tensor(0.0, device=real_imgs.device)
        perceptual_loss = torch.tensor(0.0, device=real_imgs.device)
        lpips_per_sample: Optional[torch.Tensor] = None
        progressive_alpha: Any = 1.0

        if self.train_reconstruction:
            # Progressive frequency weighting: start low to prevent over-sharpening, increase over time
            # This allows the model to learn basic structure first, then focus on sharp edges
            alpha_progress = (
                min(1.0, global_step / self.hf_ramp_steps) if self.hf_ramp_steps > 0 else 1.0
            )
            alpha_start = 0.3  # Start with lower weighting to prevent early over-sharpening
            alpha_end = 1.2  # End with higher weighting for sharp edges
            progressive_alpha = alpha_start + (alpha_end - alpha_start) * alpha_progress

            recon_l1 = self._frequency_weighted_loss(
                out_imgs_rec, real_imgs, alpha=progressive_alpha
            )
            # Remove MSE loss as it can contribute to blur - rely on L1 + LPIPS for reconstruction
            recon_loss = recon_l1

        # Plateau-scheduler metric: fixed-alpha L1, detached (no HF ramp, no LPIPS), so
        # the LR scheduler does not see a loss whose definition drifts over training.
        with torch.no_grad():
            n_m = n_ctx or B_total  # ctx half; whole (z-only) batch if it has no ctx samples
            plateau_metric = float(
                self._frequency_weighted_loss(
                    out_imgs_rec[:n_m].detach(), real_imgs[:n_m], alpha=1.0
                ).item()
            )
            recon_diag = self._recon_l1_halves(out_imgs_rec, real_imgs, n_ctx)

        # LPIPS perceptual anchor — independent of train_reconstruction, has its own use_lpips flag.
        # When train_reconstruction is also active, fold into recon_loss (single backward pass).
        # When GAN-only (train_reconstruction=False), add directly to total_loss below.
        if self.use_lpips and self.lpips_fn is not None:
            # Compute LPIPS WITH gradients so it actually trains the VAE
            # NOTE: Gradient checkpointing removed - causes recursive checkpointing
            # with VAE decoder, leading to OOM instead of saving memory
            #
            # Deliberately NOT wrapped in self._autocast(): LPIPS/VGG is numerically
            # unstable in fp16 and its own weights are never cast to half (see
            # lpips.LPIPS.forward, which does no internal dtype handling of its
            # inputs). Under fp16, out_imgs_rec is a genuine half tensor by this
            # point (produced inside the expander's autocast block above) even
            # though we're outside that context now, so it must be explicitly
            # cast back to float32 -- autocast does not retroactively "undo" a
            # tensor's already-realized dtype once you leave the context.
            device = out_imgs_rec.device
            lpips_fn = self.lpips_fn.to(device)
            lpips_per_sample = lpips_fn(out_imgs_rec.float(), real_imgs.float()).flatten()
            perceptual_loss = lpips_per_sample.mean()
            if self.train_reconstruction:
                # Fold into recon_loss so the adaptive weight already covers it
                recon_loss = recon_loss + self.lambda_lpips * perceptual_loss

        # KL divergence with beta annealing (z branch only; context branch is deterministic).
        # v0.10.0 (design §5.5): when kl_z_weight > 0 use the wide-logvar cosine
        # warmup over kl_z_warmup_steps (default 10000 -> 0.5). Otherwise stay on
        # the legacy kl_beta / kl_warmup_steps path for v060 / v070 callers.
        beta = 0.0
        if self.train_kl:
            if self.kl_z_weight > 0:
                beta = self._kl_z_beta(global_step)
            else:
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
                # Non-fatal by design (training must continue), but loud: a broad
                # except here previously swallowed real bugs behind a generic
                # warning. Surface the exception type/message so a regression in
                # this path is actually noticeable instead of silently zeroing.
                logger.error(
                    f"ctx_aux_loss computation failed ({type(exc).__name__}): {exc} "
                    "— forcing ctx_aux_loss to 0.0 for this step.",
                    exc_info=True,
                )

        # v0.10.0: latent cycle loss (independent, fixed weight, outside adaptive
        # weighting). Skipped with no extra forwards unless enabled, past start,
        # on cadence and with effective lambda > 0.
        latent_cycle_loss = torch.tensor(0.0, device=real_imgs.device)
        latent_cycle_reject_frac = 0.0
        latent_cycle_lambda = self._latent_cycle_lambda(global_step)
        if latent_cycle_lambda > 0 and global_step % self.latent_cycle_every == 0:
            try:
                latent_cycle_loss, latent_cycle_reject_frac = self._compute_latent_cycle(
                    real_imgs, out_imgs_rec[:n_ctx], global_step
                )
            except Exception as exc:
                # Non-fatal by design, but loud -- see ctx_aux_loss above.
                logger.error(
                    f"latent_cycle_loss computation failed ({type(exc).__name__}): {exc} "
                    "— forcing latent_cycle_loss to 0.0 for this step.",
                    exc_info=True,
                )
                latent_cycle_loss = torch.tensor(0.0, device=real_imgs.device)

        # Context prediction from latents (Step 1 & 2: SiLU activation + KL-context alignment)
        B = real_imgs.shape[0]
        # Use packed_rec ctx_vec — identical to _train_discriminator so context_predictor
        # dimensions stay consistent across both training methods. context_predictor is
        # deliberately never run under autocast (like LPIPS) -- .float() undoes the half
        # dtype packed_rec may carry from the compressor's autocast'd forward pass above.
        latent_repr = packed_rec[:, :-1, :].contiguous().mean(dim=1).float()  # [B, 2*d_model]

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

        # Diagnostic probe only — NOT a training-quality signal for SPADE/compressor/expander.
        # predicted_context is built from latent_repr.detach() and ideal_context from
        # real_imgs.detach(), so context_alignment_loss has zero gradient path into the VAE;
        # it is trained by its own separate context_predictor_optimizer (see __init__). Its
        # ~1.0 plateau is just the MSE "predict the mean" floor against the InstanceNorm2d
        # target's unit variance, not a sign of anything converging or failing to. Logged as
        # ``ctx_probe_alignment`` (see train_step) to keep that distinction visible.
        ideal_context = self.context_encoder(real_imgs.detach())
        context_alignment_loss = F.mse_loss(predicted_context, ideal_context)

        # GAN generator loss (off, with no D forward, until gan_start_step)
        G_img_loss = torch.tensor(0.0, device=real_imgs.device)
        gan_factor = self._gan_factor(global_step) if self.use_gan else 0.0
        w_gan = 0.0
        spade_active = self._get_effective_spade_usage(global_step)
        # z-only decodes (blurrier by design) never reach D: adversarial terms see ctx-half only.
        gan_on = self.use_gan and gan_factor > 0 and n_ctx > 0
        if gan_on and self.discriminator is not None:
            # Reuse already-decoded images for generator loss (gradients flow into encoder/decoder)
            out_imgs_gan = out_imgs_rec[:n_ctx]
            ctx_vec_rec = packed_rec[:n_ctx, :-1, :].contiguous().mean(dim=1).detach()

            # Validate context vector dimension for ctx-conditioned discriminators only
            # (ctx_dim == 0 / no ctx_proj: unconditional D, nothing to validate).
            if getattr(self.discriminator, "ctx_dim", 0) > 0 and hasattr(
                self.discriminator, "ctx_proj"
            ):
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
                # and its params must not collect G-loss grads (they would leak into
                # the next accumulated D micro-batch).
                d_params = list(self._get_unwrapped_model(self.discriminator).parameters())
                d_flags = [p.requires_grad for p in d_params]
                self.discriminator.eval()
                try:
                    for p in d_params:
                        p.requires_grad_(False)
                    with self._autocast():
                        g_real_logits = self.discriminator(out_imgs_gan, gen_ctx)
                finally:
                    for p, flag in zip(d_params, d_flags):
                        p.requires_grad_(flag)
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

        # GAN weight: VQGAN adaptive ratio at the expander's last layer (or fixed
        # lambda_adv), times the warmup factor. Needs a recon-type loss to compare with.
        if gan_on:
            # Like with like: the GAN gradient comes from ctx samples only, so the rec
            # gradient in the ratio uses the ctx-half loss (LPIPS reused per-sample).
            lp_ctx = None if lpips_per_sample is None else lpips_per_sample[:n_ctx].mean()
            rec_for_weight: Optional[torch.Tensor] = None
            if self.train_reconstruction:
                rec_for_weight = recon_loss
                if n_ctx < B_total:
                    rec_for_weight = self._frequency_weighted_loss(
                        out_imgs_rec[:n_ctx], real_imgs[:n_ctx], alpha=progressive_alpha
                    )
                    if lp_ctx is not None:
                        rec_for_weight = rec_for_weight + self.lambda_lpips * lp_ctx
            elif lp_ctx is not None:
                rec_for_weight = self.lambda_lpips * lp_ctx
            else:
                rec_for_weight = None
            w_adapt = None
            if self.adaptive_weights and rec_for_weight is not None:
                w_adapt = self._adaptive_gan_weight(
                    rec_for_weight, G_img_loss, self._expander_last_weight()
                )
            w_gan = self.lambda_adv * gan_factor * (1.0 if w_adapt is None else w_adapt)
        self._last_w_gan = w_gan

        # Color/contrast regularization losses — each gated on its own flag only.
        # out_imgs_rec and real_imgs are always available (computed above), so these
        # are meaningful in any mode that runs the generator (VAE-only, GAN-only, joint).
        color_stats_loss = (
            self._color_statistics_loss(out_imgs_rec, real_imgs)
            if self.train_colorstats
            else torch.tensor(0.0, device=real_imgs.device)
        )
        hist_loss = (
            self._histogram_matching_loss(out_imgs_rec, real_imgs)
            if self.train_histogram
            else torch.tensor(0.0, device=real_imgs.device)
        )
        contrast_loss = (
            self._contrast_regularization_loss(out_imgs_rec, real_imgs)
            if self.train_contrast
            else torch.tensor(0.0, device=real_imgs.device)
        )
        coarseness_loss = (
            self._coarseness_loss(out_imgs_rec, real_imgs)
            if self.train_coarseness
            else torch.tensor(0.0, device=real_imgs.device)
        )

        # v0.10.0: Bezier control-point monotonicity regularizer (plan Fix 2).
        # Structural sanity constraint (not a config knob, see losses.py
        # docstring) -- gated on activation family, since it only applies to
        # Bezier-shaped mu/logvar activations (p0<=p1<=p2<=p3). Pade
        # activations (activation_type="pade") have no p0-p3 control points
        # and no equivalent monotonicity constraint implemented, so the term
        # is a no-op there. Missing activation_type (pre-v100 compressors)
        # defaults to bezier, since those classes are Bezier-only by
        # construction. Default inits are already monotonic so this is ~0.0
        # in the common bezier case.
        unwrapped_compressor = self._get_unwrapped_model(self.compressor)
        if getattr(unwrapped_compressor, "activation_type", "bezier") == "bezier":
            bezier_reg_loss = compute_bezier_monotonicity_reg(
                unwrapped_compressor.mu_activation, _BEZIER_REG_WEIGHT
            ) + compute_bezier_monotonicity_reg(
                unwrapped_compressor.logvar_activation, _BEZIER_REG_WEIGHT
            )
        else:
            bezier_reg_loss = torch.tensor(0.0, device=real_imgs.device)

        # Total loss: KL and recon use fixed weights (never adaptive).
        total_loss = beta * kl
        if self.train_reconstruction:
            # recon_loss already includes lambda_lpips * perceptual_loss when use_lpips=True
            total_loss = total_loss + recon_loss
        elif self.use_lpips:
            # GAN-only mode: LPIPS not folded into recon_loss, add it standalone here
            total_loss = total_loss + self.lambda_lpips * perceptual_loss
        if self.use_gan and n_ctx > 0:
            total_loss = total_loss + w_gan * G_img_loss

        # v0.10.0: auxiliary context reconstruction loss (plan §4.1 / DP-1).
        if self.train_ctx_aux:
            total_loss = total_loss + self.lambda_ctx_aux * ctx_aux_loss

        # v0.10.0: latent cycle loss, fixed (scheduled) weight, never adaptive.
        if latent_cycle_lambda > 0:
            total_loss = total_loss + latent_cycle_lambda * latent_cycle_loss

        # v0.10.0: ctx-features L2 shrinkage (design §5.5). The forward hook on
        # ctx_zinject_norm populates _ctx_features_cache; if no v0.10.0 compressor
        # is in play or the weight is 0 we just contribute zero.
        ctx_shrinkage_alpha = delayed_cosine_warmup_weight(
            global_step,
            self.ctx_shrinkage_warmup_start_step,
            self.ctx_shrinkage_warmup_steps,
            self.ctx_shrinkage_weight,
        )
        if ctx_shrinkage_alpha > 0 and self._ctx_hook_handle is not None:
            # The pre-hook is installed and the schedule says the term should be
            # active: a missing/malformed cache here means the pre-hook silently
            # stopped firing (e.g. a future refactor of
            # FluxCompressor_v100.forward's ctx_zinject_norm(cx) call convention)
            # rather than an intentional no-op. Surface that loudly instead of
            # silently reverting to a dead-gradient no-op.
            assert (
                isinstance(self._ctx_features_cache, torch.Tensor)
                and self._ctx_features_cache.dim() == 4
            ), (
                "ctx_shrinkage hook is installed and alpha="
                f"{ctx_shrinkage_alpha} > 0 but _ctx_features_cache is "
                f"{self._ctx_features_cache!r} (expected a 4D pre-norm ctx "
                "tensor). The ctx_zinject_norm pre-hook likely stopped firing - "
                "check FluxCompressor_v100.forward's ctx_zinject_norm(cx) call."
            )
        if self._ctx_features_cache is not None and ctx_shrinkage_alpha > 0:
            ctx_shrinkage_loss = compute_ctx_shrinkage(
                self._ctx_features_cache,
                ctx_shrinkage_alpha,
                max_mean_sq=self.ctx_shrinkage_max_mean_sq,
            )
            total_loss = total_loss + ctx_shrinkage_loss
        else:
            ctx_shrinkage_loss = torch.tensor(0.0, device=real_imgs.device)
        # Clear cache so a stale tensor can't be reused in a subsequent step
        # that skips the compressor forward (defensive — train_step always
        # runs the compressor, but the cache lifecycle stays explicit).
        self._ctx_features_cache = None

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
        # Always on -- structural sanity constraint, not gated by a train_* flag.
        total_loss = total_loss + bezier_reg_loss

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
            logger.error(f"  w_gan: {w_gan}, beta: {beta}")
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
                "latent_cycle_loss": 0.0,
                "latent_cycle_reject_frac": 0.0,
                "bezier_reg": 0.0,
                "_optimizer_stepped": False,  # Signal that optimizer was not stepped
            }

        # Clear cache before backward to prevent OOM when memory is tight.
        # Gradient checkpointing in VAE causes memory spikes during backward pass
        _empty_cache_if_low_memory(real_imgs.device)

        # Gradient accumulation: scale before backward (standard grad-accum),
        # matching FlowTrainer's total_loss / gradient_accumulation_steps.
        total_loss = total_loss / self.gradient_accumulation_steps
        self.accelerator.backward(total_loss)

        # Generator counter: only micro-batches that reach backward are counted; the
        # skip paths above return before touching it. The discriminator keeps its own.
        should_step = (self._accumulation_step + 1) % self.gradient_accumulation_steps == 0
        self._accumulation_step += 1

        result: dict[str, Any] = {
            "vae": float(recon_loss.detach().item()),
            "kl": float(kl.detach().item()),
            "generator": float(G_img_loss.detach().item()) if self.use_gan else 0.0,
            "lpips": float(perceptual_loss.detach().item()) if self.use_lpips else 0.0,
            "recon": float(recon_loss.detach().item()),
            "weight_gan": float(w_gan),
            # v0.10.0: auxiliary context reconstruction loss (plan §4.1)
            "ctx_aux_loss": float(ctx_aux_loss.detach().item()),
            # v0.10.0: latent cycle loss (unweighted) + reject fraction
            "latent_cycle_loss": float(latent_cycle_loss.detach().item()),
            "latent_cycle_reject_frac": float(latent_cycle_reject_frac),
            # v0.10.0: ctx-features L2 shrinkage (design §5.5)
            "ctx_shrinkage_loss": float(ctx_shrinkage_loss.detach().item()),
            "ctx_shrinkage_alpha": float(ctx_shrinkage_alpha),
            # Diagnostic probe, no gradient into SPADE/compressor/expander.
            "ctx_probe_alignment": float(context_alignment_loss.detach().item()),
            "color_stats": float(color_stats_loss.detach().item()),
            "hist_loss": float(hist_loss.detach().item()),
            "contrast_loss": float(contrast_loss.detach().item()),
            "coarseness_loss": float(coarseness_loss.detach().item()),
            # v0.10.0: Bezier control-point monotonicity regularizer (plan Fix 2)
            "bezier_reg": float(bezier_reg_loss.detach().item()),
            "_optimizer_stepped": False,
            "_plateau_metric": plateau_metric,
        }
        result.update(recon_diag)
        if self.use_gan and n_ctx > 0:
            result["_fake_imgs"] = out_imgs_rec[:n_ctx].detach()

        if not should_step:
            # Mid-accumulation-window: gradients stay accumulated in .grad,
            # optimizer/scaler untouched until the boundary.
            return result

        # Reset the boundary counter now that this window is complete.
        self._accumulation_step = 0

        # Unscale once (no-op without fp16), clip with the plain torch function (not
        # accelerator.clip_grad_norm_, which would unscale a second time), and skip the
        # step on non-finite grads: self.optimizer is a plain torch optimizer, so no
        # GradScaler auto-skip. scaler.update() is called once per iteration in train_step.
        vae_params = list(self.compressor.parameters()) + list(self.expander.parameters())
        if not self._unscale_clip(self.optimizer, vae_params, "VAE"):
            return result

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

        # "vae" is recon_loss (NOT total_loss, which includes weighting and can be huge/negative)
        result["_optimizer_stepped"] = True
        return result

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
