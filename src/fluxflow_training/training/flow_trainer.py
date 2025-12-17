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
        gradient_clip_norm: float = 1.0,
        num_train_timesteps: int = 1000,
        ema_decay: float = 0.9999,
        lambda_align: float = 0.0,
        cfg_dropout_prob: float = 0.0,
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
            gradient_clip_norm: Gradient clipping norm
            num_train_timesteps: Number of diffusion timesteps
            ema_decay: EMA decay rate for model parameters (default: 0.9999)
            lambda_align: Text-image alignment loss weight (default: 0.1)
            cfg_dropout_prob: Classifier-free guidance dropout probability (default: 0.0)
                              Set to 0.10 for standard CFG training
            accelerator: Accelerate accelerator instance
        """
        self.flow_processor = flow_processor
        self.text_encoder = text_encoder
        self.compressor = compressor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.text_encoder_optimizer = text_encoder_optimizer
        self.text_encoder_scheduler = text_encoder_scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.lambda_align = lambda_align
        self.cfg_dropout_prob = cfg_dropout_prob
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
        self.noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps)
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
        if self.text_encoder_optimizer is not None:
            self.text_encoder.train()
        else:
            self.text_encoder.eval()

        self.optimizer.zero_grad(set_to_none=True)
        if self.text_encoder_optimizer is not None:
            self.text_encoder_optimizer.zero_grad(set_to_none=True)

        # Encode text
        text_embeddings = self.text_encoder(input_ids, attention_mask=attention_mask)

        # Apply classifier-free guidance dropout
        if self.cfg_dropout_prob > 0.0:
            from .cfg_utils import apply_cfg_dropout

            text_embeddings = apply_cfg_dropout(text_embeddings, p_uncond=self.cfg_dropout_prob)

        # Encode image to latent (frozen VAE)
        with torch.no_grad():
            latent_packet = self.compressor(real_imgs)

        img_seq = latent_packet[:, :-1, :].contiguous()
        hw_vec = latent_packet[:, -1:, :].contiguous()

        # Sample timesteps
        device = img_seq.device
        t = sample_t(img_seq.size(0), device)
        noise = torch.randn_like(img_seq)

        # Add noise using scheduler
        noised_seq = self.noise_scheduler.add_noise(img_seq, noise, t)
        full_input = torch.cat([noised_seq, hw_vec], dim=1)

        # Predict
        pred = self.flow_processor(full_input, text_embeddings, t)
        pred_seq = pred[:, :-1, :]

        # v-prediction loss (more stable than epsilon prediction)
        alpha_t = self.alphas_cumprod[t][:, None, None]
        sigma_t = (1 - alpha_t).sqrt()
        alpha_t = alpha_t.sqrt()
        v_target = alpha_t * noise - sigma_t * img_seq

        diff_loss = nn.functional.mse_loss(pred_seq, v_target)

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

        # Combine losses
        total_loss = diff_loss + self.lambda_align * align_loss

        # Check for NaN/Inf in loss
        if check_for_nan(total_loss, "flow_total_loss", logger):
            logger.error("Skipping batch due to NaN in flow loss")
            return {"flow_loss": 0.0, "diff_loss": 0.0, "align_loss": 0.0}

        self.accelerator.backward(total_loss)

        # Check gradients for NaN/Inf after backward
        if self.accelerator.scaler is not None:
            self.accelerator.scaler.unscale_(self.optimizer)
            for name, param in self.flow_processor.named_parameters():
                if param.grad is not None and check_for_nan(
                    param.grad, f"grad_flow_{name}", logger
                ):
                    logger.warning(f"NaN gradient in flow_processor.{name}, zeroing it")
                    param.grad.zero_()

        # Clip gradients
        self.accelerator.clip_grad_norm_(
            self.flow_processor.parameters(),
            self.gradient_clip_norm,
        )

        self.optimizer.step()
        if self.text_encoder_optimizer is not None:
            self.text_encoder_optimizer.step()

        # Update EMA
        self.ema.update()

        # Get loss value for metrics and schedulers
        loss_value = float(total_loss.detach().item())

        # Step schedulers after optimizer step (ReduceLROnPlateau requires metric, others don't)
        # Skip first step to avoid PyTorch warning about calling scheduler before first optimizer step
        if not self._first_step:
            # Get the underlying scheduler (may be wrapped by accelerator)
            base_scheduler = getattr(self.scheduler, "scheduler", self.scheduler)
            if isinstance(base_scheduler, ReduceLROnPlateau):
                self.scheduler.step(loss_value)  # type: ignore[arg-type]
            else:
                self.scheduler.step()  # type: ignore[call-arg]

            if self.text_encoder_scheduler is not None:
                base_te_scheduler = getattr(
                    self.text_encoder_scheduler, "scheduler", self.text_encoder_scheduler
                )
                if isinstance(base_te_scheduler, ReduceLROnPlateau):
                    self.text_encoder_scheduler.step(loss_value)  # type: ignore[arg-type]
                else:
                    self.text_encoder_scheduler.step()  # type: ignore[call-arg]
        else:
            self._first_step = False

        # Return comprehensive metrics
        metrics = {
            "flow_loss": loss_value,
            "diff_loss": float(diff_loss.detach().item()),
            "align_loss": float(align_loss.detach().item()),
            "grad_norm_flow": compute_grad_norm(self.flow_processor.parameters()),
            "grad_norm_text": compute_grad_norm(self.text_encoder.parameters()),
            "lr_flow": self.optimizer.param_groups[0]["lr"],
            "pred_mean": pred_seq.mean().item(),
            "pred_std": pred_seq.std().item(),
        }

        # Add text encoder LR only if optimizer exists
        if self.text_encoder_optimizer is not None:
            metrics["lr_text"] = self.text_encoder_optimizer.param_groups[0]["lr"]

        return metrics

    def move_scheduler_to_device(self, device: torch.device):
        """Move scheduler's alphas_cumprod to the specified device."""
        self.alphas_cumprod = self.alphas_cumprod.to(device)
