"""Flow model training logic for FluxFlow.

Handles flow-based diffusion model training with v-prediction.
"""

from typing import Optional

import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler
from fluxflow.utils import get_logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from .schedulers import sample_t

logger = get_logger(__name__)


def check_for_nan(tensor, name, logger_inst):
    """Check for NaN/Inf values and log warning."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger_inst.warning(f"NaN/Inf detected in {name}")
        return True
    return False


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


class FlowTrainer:
    """
    Handles flow-based diffusion model training.

    Uses v-prediction objective for better stability compared to epsilon prediction.

    Example:
        >>> trainer = FlowTrainer(
        ...     flow_processor=flow_model,
        ...     text_encoder=text_encoder,
        ...     compressor=vae_compressor,
        ...     optimizer=optimizer,
        ...     scheduler=lr_scheduler,
        ... )
        >>> loss = trainer.train_step(images, input_ids, attention_mask, global_step)
    """

    def __init__(
        self,
        flow_processor: nn.Module,
        text_encoder: nn.Module,
        compressor: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,  # type: ignore[type-arg]
        text_encoder_optimizer: Optional[Optimizer] = None,
        text_encoder_scheduler: Optional[_LRScheduler] = None,  # type: ignore[type-arg]
        text_encoder_extra_optimizers: Optional[dict] = None,
        text_encoder_extra_schedulers: Optional[dict] = None,
        gradient_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        num_train_timesteps: int = 1000,
        start_step: int = 0,
        ema_decay: float = 0.9999,
        lambda_align: float = 0.0,
        cfg_dropout_prob: float = 0.0,
        ctx_loss_weight: float = 1.0,
        accelerator=None,
    ):
        """
        Initialize Flow trainer.

        Args:
            flow_processor: Flow prediction model (transformer)
            text_encoder: Text encoder for conditioning
            compressor: VAE compressor (frozen during flow training)
            optimizer: Flow processor optimizer
            scheduler: Flow processor learning rate scheduler
            text_encoder_optimizer: Text encoder optimizer (None if frozen)
            text_encoder_scheduler: Text encoder scheduler (None if frozen)
            text_encoder_extra_optimizers: Sub-component optimizers keyed by "backbone"
                and/or "projection". Mutually exclusive with text_encoder_optimizer.
            text_encoder_extra_schedulers: Sub-component schedulers with same keys.
            gradient_clip_norm: Gradient clipping norm
            num_train_timesteps: Number of diffusion timesteps
            start_step: Starting diffusion timestep (default: 0 for noise-to-image)
            ema_decay: EMA decay rate for model parameters (default: 0.9999)
            lambda_align: Text-image alignment loss weight (default: 0.1)
            cfg_dropout_prob: Classifier-free guidance dropout probability (default: 0.0)
                              Set to 0.10 for standard CFG training
            ctx_loss_weight: Scale factor applied to the context-dim v-prediction loss
                             relative to the VAE-dim v-prediction loss. Default 1.0 (equal).
                             Increase if context dims are under-trained; decrease if they
                             dominate early and destabilise the VAE loss.
            accelerator: Accelerate accelerator instance
        """
        self.flow_processor = flow_processor
        self.text_encoder = text_encoder
        self.compressor = compressor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.text_encoder_optimizer = text_encoder_optimizer
        self.text_encoder_scheduler = text_encoder_scheduler
        # Split text encoder optimizers: keys are "backbone" and/or "projection".
        self.text_encoder_extra_optimizers: dict = text_encoder_extra_optimizers or {}
        self.text_encoder_extra_schedulers: dict = text_encoder_extra_schedulers or {}
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._accumulation_step = 0  # Track current accumulation step
        self.lambda_align = lambda_align
        self.cfg_dropout_prob = cfg_dropout_prob
        self.ctx_loss_weight = ctx_loss_weight
        self.accelerator = accelerator

        # Setup EMA for flow processor and text encoder
        from .utils import EMA

        # Create wrapper module for EMA tracking
        class FlowTextWrapper(nn.Module):
            def __init__(self, flow, text):
                super().__init__()
                self.flow_processor = flow
                self.text_encoder = text

        self._ema_wrapper = FlowTextWrapper(flow_processor, text_encoder)
        self.ema = EMA(self._ema_wrapper, decay=ema_decay)

        # Setup diffusion scheduler
        self.num_train_timesteps = num_train_timesteps
        self.start_step = start_step
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="v_prediction",
            use_karras_sigmas=False,
        )
        self.noise_scheduler.set_timesteps(num_train_timesteps)  # type: ignore[arg-type]
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(  # type: ignore[attr-defined]
            next(flow_processor.parameters()).device
        )

        # Track if this is the first training step (to avoid scheduler warning)
        self._first_step = True

    def train_step(
        self,
        real_imgs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_step: int = 0,
    ) -> dict[str, float]:
        """
        Perform one flow training step.

        Args:
            real_imgs: Real images [B, C, H, W]
            input_ids: Tokenized text [B, seq_len]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Dictionary with loss and metric values
        """
        self.flow_processor.train()

        # B1: fine-grained per-sub-module train/eval mode based on active optimizers.
        _has_whole_te_opt = self.text_encoder_optimizer is not None
        _has_backbone_opt = "backbone" in self.text_encoder_extra_optimizers
        _has_projection_opt = "projection" in self.text_encoder_extra_optimizers

        if _has_whole_te_opt or (_has_backbone_opt and _has_projection_opt):
            self.text_encoder.train()
        elif _has_projection_opt and not _has_backbone_opt:
            if hasattr(self.text_encoder, "language_model"):
                self.text_encoder.language_model.eval()
                self.text_encoder.ouput_layer.train()
            else:
                self.text_encoder.train()
        elif _has_backbone_opt and not _has_projection_opt:
            if hasattr(self.text_encoder, "language_model"):
                self.text_encoder.language_model.train()
                self.text_encoder.ouput_layer.eval()
            else:
                self.text_encoder.train()
        else:
            self.text_encoder.eval()

        if self._accumulation_step == 0:
            self.optimizer.zero_grad(set_to_none=True)
            if self.text_encoder_optimizer is not None:
                self.text_encoder_optimizer.zero_grad(set_to_none=True)
            # B4: extra optimizer zero_grad must be inside the accumulation guard
            for opt in self.text_encoder_extra_optimizers.values():
                opt.zero_grad(set_to_none=True)

        # Encode text
        text_embeddings = self.text_encoder(input_ids, attention_mask=attention_mask)
        if check_for_nan(text_embeddings, "text_embeddings", logger):
            logger.error("NaN detected in text embeddings - skipping batch")
            return {"flow_loss": 0.0, "diff_loss": 0.0, "align_loss": 0.0}

        # Encode image to latent (frozen VAE)
        # Temporarily disable gradient checkpointing to avoid issues with torch.no_grad()
        original_checkpoint = self.compressor.use_gradient_checkpointing
        self.compressor.use_gradient_checkpointing = False
        with torch.no_grad():
            latent_packet = self.compressor(real_imgs)
        if check_for_nan(latent_packet, "latent_packet", logger):
            logger.error("NaN detected in compressor output - skipping batch")
            return {"flow_loss": 0.0, "diff_loss": 0.0, "align_loss": 0.0}
        # Restore original setting
        self.compressor.use_gradient_checkpointing = original_checkpoint

        img_seq = latent_packet[:, :-1, :].contiguous()
        hw_vec = latent_packet[:, -1:, :].contiguous()

        # Get context dimensions from model
        context_dims = self.compressor.get_context_dims()

        # Sample timesteps
        device = img_seq.device
        t = sample_t(img_seq.size(0), device, self.start_step, self.num_train_timesteps)

        # Apply progressive classifier-free guidance dropout
        # Higher CFG dropout at high timesteps (more noisy images need more guidance)
        if self.cfg_dropout_prob > 0.0:
            from .cfg_utils import apply_cfg_dropout

            # Progressive CFG: scale dropout probability based on timestep
            # Higher timesteps (more noise) benefit from more guidance
            t_normalized = t.float() / self.num_train_timesteps  # [0, 1]
            progressive_p_uncond = self.cfg_dropout_prob * (
                0.5 + 0.5 * t_normalized
            )  # Scale from 0.5x to 1.5x
            progressive_p_uncond = torch.clamp(
                progressive_p_uncond, 0.0, min(0.5, self.cfg_dropout_prob * 2.0)
            )

            text_embeddings = apply_cfg_dropout(
                text_embeddings, p_uncond=progressive_p_uncond.mean().item()
            )

        # Monitor timestep distribution for training progress (log every 500 steps)
        if global_step % 500 == 0:
            # Create histogram of sampled timesteps
            hist = torch.histc(
                t.float(),
                bins=10,
                min=self.start_step,
                max=self.start_step + self.num_train_timesteps - 1,
            )
            hist = hist / hist.sum()  # Normalize to probabilities

            logger.info(f"Timestep distribution: {hist}")
            logger.info(
                f"Timestep range: {t.min().item()}-{t.max().item()} (mean: {t.float().mean().item():.1f})"
            )

        # Noise all dims uniformly — matches inference exactly.
        # Inference starts from torch.randn over all D dims (VAE + context) and the
        # scheduler denoises them with a single alpha_t/sigma_t schedule.  Training
        # must present the same noisy input distribution so the model learns to
        # denoise context dims from noise, not from their own clean values.
        vae_dims = img_seq.size(-1) - context_dims if context_dims > 0 else img_seq.size(-1)
        noise = torch.randn_like(img_seq)  # [B, T, D+context_dims]
        noised_seq = self.noise_scheduler.add_noise(img_seq, noise, t)  # [B, T, D+context_dims]
        full_input = torch.cat([noised_seq, hw_vec], dim=1)

        # Predict (flow processor expects normalized timesteps in [0, 1]).
        denom = float(max(1, (self.num_train_timesteps - 1) - self.start_step))
        t_model = ((t.float() - float(self.start_step)) / denom).clamp(0.0, 1.0)
        pred = self.flow_processor(full_input, text_embeddings, t_model)

        # Extract predicted sequence (exclude HW vector)
        pred_seq = pred[:, : img_seq.size(1), :]
        if check_for_nan(pred_seq, "flow_prediction", logger):
            logger.error("NaN detected in flow prediction - skipping batch")
            return {"flow_loss": 0.0, "diff_loss": 0.0, "align_loss": 0.0}

        # Compute v-prediction target: v = alpha_t * noise - sigma_t * x0.
        # alphas_cumprod[t] is [B]; reshape to [B, 1, 1] for broadcasting with [B, T, D].
        alpha_cumprod_t = self.alphas_cumprod[t].float()  # [B]
        alpha_t = alpha_cumprod_t.sqrt().view(-1, 1, 1)  # [B, 1, 1]
        sigma_t = (1.0 - alpha_cumprod_t).sqrt().view(-1, 1, 1)  # [B, 1, 1]

        # v-prediction target over all dims: v = alpha_t * noise - sigma_t * x0.
        # All dims (VAE + context) are noised identically above, so the same
        # alpha_t / sigma_t applies to both groups.
        img_seq_fp32 = img_seq.float()
        noise_fp32 = noise.float()
        v_target = alpha_t * noise_fp32 - sigma_t * img_seq_fp32  # [B, T, D+context_dims]

        if context_dims > 0:
            # Split v_target and prediction into VAE and context groups, normalise each
            # separately to equalise loss scale across the two very different dynamic ranges.
            # VAE dims (~unit Gaussian) and context dims (~0.1-0.5 range after mean-pooling)
            # would otherwise produce a ~10x loss imbalance under a shared normalisation.
            vae_v_target = v_target[:, :, :vae_dims].detach()  # [B, T, vae_dims]
            ctx_v_target = v_target[:, :, vae_dims:].detach()  # [B, T, context_dims]

            pred_vae = pred_seq[:, :, :vae_dims].contiguous().float()
            pred_ctx = pred_seq[:, :, vae_dims:].contiguous().float()

            # Normalise by the v-target's own std so the scale is consistent across
            # all timesteps.  Using clean-x0 std (ctx ~0.1-0.5) would over-weight
            # ctx_loss ~11x at high-noise timesteps where v_target ≈ noise (std ~1).
            vae_std = vae_v_target.std() + 1e-8
            ctx_std = ctx_v_target.std() + 1e-8

            vae_loss = nn.functional.mse_loss(pred_vae / vae_std, vae_v_target / vae_std)
            ctx_loss = nn.functional.mse_loss(pred_ctx / ctx_std, ctx_v_target / ctx_std)
            diff_loss = vae_loss + self.ctx_loss_weight * ctx_loss
        else:
            # v0.6 and earlier: no context dims, single normalised loss.
            latent_std = img_seq_fp32.detach().std() + 1e-8
            normalized_v_target = v_target.detach() / latent_std
            normalized_pred = pred_seq.float() / latent_std
            diff_loss = nn.functional.mse_loss(normalized_pred, normalized_v_target)
            ctx_loss = torch.tensor(0.0, device=img_seq.device)

        # Text-image alignment loss (optional, disabled by default due to dimension mismatch issues)
        # Only compute if lambda_align > 0
        if self.lambda_align > 0.0:
            img_features = pred_seq.mean(dim=1)  # [B, T, D] -> [B, D]

            # Pool text embeddings to match image features shape
            if text_embeddings.dim() == 3:
                text_features_pooled = text_embeddings.mean(dim=1)  # [B, seq_len, D] -> [B, D]
            elif text_embeddings.dim() == 2:
                text_features_pooled = text_embeddings  # Already [B, D]
            else:
                logger.warning(
                    f"Unexpected text_embeddings shape: {text_embeddings.shape}, skipping alignment loss"
                )
                align_loss = torch.tensor(0.0, device=pred_seq.device)
                text_features_pooled = None

            # Compute alignment loss if dimensions match
            if text_features_pooled is not None:
                if img_features.shape[-1] == text_features_pooled.shape[-1]:
                    # Normalize and compute cosine similarity
                    text_features = nn.functional.normalize(text_features_pooled, dim=-1)
                    img_features_norm = nn.functional.normalize(img_features, dim=-1)
                    cosine_sim = nn.functional.cosine_similarity(
                        img_features_norm, text_features, dim=-1
                    )
                    align_loss = (1 - cosine_sim).mean()
                else:
                    # Dimension mismatch - skip alignment loss
                    logger.warning(
                        f"Skipping alignment loss: dimension mismatch "
                        f"img_features {img_features.shape} vs text_features {text_features_pooled.shape}"
                    )
                    align_loss = torch.tensor(0.0, device=pred_seq.device)
        else:
            # Alignment loss disabled
            align_loss = torch.tensor(0.0, device=pred_seq.device)

        # Adaptive loss weighting for better training dynamics
        # Gradually increase alignment loss weight to prevent early dominance
        if self.lambda_align > 0.0:
            # Warm up alignment loss over first 10% of training
            warmup_steps = max(1000, int(0.1 * 10000))  # Assume 10k steps for warmup
            align_weight = min(self.lambda_align, self.lambda_align * (global_step / warmup_steps))
        else:
            align_weight = 0.0

        # Combine losses with adaptive weighting
        total_loss = diff_loss + align_weight * align_loss

        # Check for NaN/Inf in loss
        if check_for_nan(total_loss, "flow_total_loss", logger):
            logger.error("Skipping batch due to NaN in flow loss")
            return {"flow_loss": 0.0, "diff_loss": 0.0, "align_loss": 0.0}

        # Gradient accumulation for effective larger batch sizes
        total_loss = total_loss / self.gradient_accumulation_steps
        self.accelerator.backward(total_loss)

        # Get loss value for metrics (defined before accumulation check)
        loss_value = float(total_loss.detach().item())
        grad_norm: float = 0.0

        # Only update weights after accumulating gradients
        self._accumulation_step += 1
        should_step = (self._accumulation_step % self.gradient_accumulation_steps) == 0

        if should_step:
            # Clip gradients to configured norm. Returns the pre-clip total norm.
            clipped = self.accelerator.clip_grad_norm_(
                self.flow_processor.parameters(),
                self.gradient_clip_norm,
            )
            grad_norm = float(clipped) if isinstance(clipped, torch.Tensor) else float(clipped)

            self.optimizer.step()
            if self.text_encoder_optimizer is not None:
                self.text_encoder_optimizer.step()
            for opt in self.text_encoder_extra_optimizers.values():
                opt.step()

            # Update EMA
            self.ema.update()

            # Step schedulers after optimizer step (ReduceLROnPlateau requires metric, others don't)
            # Skip first step to avoid PyTorch warning about calling scheduler before first optimizer step
            if not self._first_step:
                # Get the underlying scheduler (may be wrapped by accelerator)
                base_scheduler = getattr(self.scheduler, "scheduler", self.scheduler)
                if isinstance(base_scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss_value)  # type: ignore[arg-type]
                else:
                    self.scheduler.step()  # type: ignore[call-arg]

            # Reset accumulation step
            self._accumulation_step = 0

            if self.text_encoder_scheduler is not None:
                base_te_scheduler = getattr(
                    self.text_encoder_scheduler, "scheduler", self.text_encoder_scheduler
                )
                if isinstance(base_te_scheduler, ReduceLROnPlateau):
                    self.text_encoder_scheduler.step(loss_value)  # type: ignore[arg-type]
                else:
                    self.text_encoder_scheduler.step()  # type: ignore[call-arg]

            for sched in self.text_encoder_extra_schedulers.values():
                base = getattr(sched, "scheduler", sched)
                if isinstance(base, ReduceLROnPlateau):
                    sched.step(loss_value)
                else:
                    sched.step()

            if self._first_step:
                self._first_step = False

        # Return comprehensive metrics
        metrics = {
            "flow_loss": loss_value,
            "diff_loss": float(diff_loss.detach().item()),
            "ctx_loss": float(ctx_loss.detach().item()),
            "align_loss": float(align_loss.detach().item()),
            "grad_norm_flow": grad_norm,
            "grad_norm_text": compute_grad_norm(self.text_encoder.parameters()),
            "lr_flow": self.optimizer.param_groups[0]["lr"],
            "pred_mean": pred_seq.mean().item(),
            "pred_std": pred_seq.std().item(),
            # Latent statistics for monitoring normalization effectiveness
            "vae_latent_mean": (
                float(img_seq[:, :, :vae_dims].mean().item())
                if context_dims > 0
                else float(img_seq.mean().item())
            ),
            "latent_std": (
                float(img_seq[:, :, :vae_dims].std().item())
                if context_dims > 0
                else float(img_seq.std().item())
            ),
            "latent_max": (
                float(img_seq[:, :, :vae_dims].max().item())
                if context_dims > 0
                else float(img_seq.max().item())
            ),
            "latent_min": (
                float(img_seq[:, :, :vae_dims].min().item())
                if context_dims > 0
                else float(img_seq.min().item())
            ),
        }

        # Add text encoder LR only if optimizer exists
        if self.text_encoder_optimizer is not None:
            metrics["lr_text"] = self.text_encoder_optimizer.param_groups[0]["lr"]

        for key, opt in self.text_encoder_extra_optimizers.items():
            metrics[f"lr_text_{key}"] = opt.param_groups[0]["lr"]

        return metrics

    def move_scheduler_to_device(self, device: torch.device):
        """Move scheduler's alphas_cumprod to the specified device."""
        self.alphas_cumprod = self.alphas_cumprod.to(device)
