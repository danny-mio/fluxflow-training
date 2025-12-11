"""Classifier-Free Guidance utilities for FluxFlow.

Provides functions for applying CFG dropout during training and CFG-guided
sampling during inference.
"""

import torch
from fluxflow.utils import get_logger

logger = get_logger(__name__)


def apply_cfg_dropout(
    text_embeddings: torch.Tensor,
    p_uncond: float = 0.1,
) -> torch.Tensor:
    """
    Apply classifier-free guidance dropout to text embeddings.

    Randomly replaces text embeddings with zero vectors for a fraction
    of samples in the batch. This enables the model to learn both
    conditional p(x|c) and unconditional p(x) generation.

    Args:
        text_embeddings: Text embeddings [B, D] or [B, seq_len, D]
        p_uncond: Probability of null conditioning (default: 0.1)
                  Set to 0.0 to disable CFG (standard conditional training)
                  Typical values: 0.05-0.20

    Returns:
        Text embeddings with CFG dropout applied [same shape as input]
        NOTE: Modifies input tensor in-place for zero memory overhead.
              If you need to preserve the original, clone before calling.

    Example:
        >>> text_emb = text_encoder(input_ids)  # [B, 1024]
        >>> text_emb_dropped = apply_cfg_dropout(text_emb, p_uncond=0.1)
        >>> # ~10% of batch now has zero embeddings
    """
    if p_uncond < 0.0 or p_uncond > 1.0:
        raise ValueError(f"p_uncond must be in [0, 1], got {p_uncond}")

    if p_uncond == 0.0:
        # CFG disabled, return original embeddings
        return text_embeddings

    batch_size = text_embeddings.size(0)
    device = text_embeddings.device

    # Create null embedding (zero vector, same shape as one sample)
    if text_embeddings.dim() == 2:
        # Shape: [B, D]
        null_emb = torch.zeros(
            1, text_embeddings.size(1), device=device, dtype=text_embeddings.dtype
        )
    elif text_embeddings.dim() == 3:
        # Shape: [B, seq_len, D]
        null_emb = torch.zeros(
            1,
            text_embeddings.size(1),
            text_embeddings.size(2),
            device=device,
            dtype=text_embeddings.dtype,
        )
    else:
        raise ValueError(
            f"text_embeddings must be 2D [B, D] or 3D [B, seq_len, D], "
            f"got shape {text_embeddings.shape}"
        )

    # Create dropout mask [B]
    dropout_mask = torch.rand(batch_size, device=device) < p_uncond

    # Apply null conditioning in-place (zero memory overhead)
    # Note: Modifies input tensor. If you need original, call .clone() before this function.
    text_embeddings[dropout_mask] = null_emb

    # Log statistics (only occasionally to avoid spam)
    if torch.rand(1).item() < 0.01:  # 1% of batches
        num_null = dropout_mask.sum().item()
        logger.debug(
            f"CFG dropout: {num_null}/{batch_size} samples "
            f"({100.0 * num_null / batch_size:.1f}%) set to null conditioning"
        )

    return text_embeddings


def cfg_guided_prediction(
    model_fn,
    z_t: torch.Tensor,
    text_embeddings: torch.Tensor,
    timesteps: torch.Tensor,
    guidance_scale: float = 5.0,
) -> torch.Tensor:
    """
    Perform classifier-free guidance at inference time.

    Computes both conditional and unconditional predictions, then
    extrapolates in the direction of the conditional prediction:

        v_guided = v_uncond + ω * (v_cond - v_uncond)

    where ω (omega) is the guidance_scale.

    Args:
        model_fn: Function that takes (z_t, text_emb, timesteps) and returns prediction
        z_t: Noisy latent [B, T, D]
        text_embeddings: Text embeddings [B, emb_dim] or [B, seq_len, emb_dim]
        timesteps: Timesteps [B]
        guidance_scale: CFG strength (ω)
                        - 0.0: Pure unconditional
                        - 1.0: Standard conditional
                        - >1.0: Over-guided (stronger prompt adherence)
                        Typical range: 3.0-9.0 for flow matching

    Returns:
        Guided prediction [B, T, D]

    Example:
        >>> # During inference
        >>> v_guided = cfg_guided_prediction(
        ...     flow_model,
        ...     z_t=noisy_latent,
        ...     text_embeddings=text_emb,
        ...     timesteps=t,
        ...     guidance_scale=5.0
        ... )
        >>> z_t = z_t + dt * v_guided  # Euler integration step
    """
    if guidance_scale == 1.0:
        # Standard conditional prediction (no guidance)
        return model_fn(z_t, text_embeddings, timesteps)

    if guidance_scale == 0.0:
        # Pure unconditional prediction
        null_emb = torch.zeros_like(text_embeddings)
        return model_fn(z_t, null_emb, timesteps)

    # Create null embedding (zero vector)
    null_emb = torch.zeros_like(text_embeddings)

    # Conditional prediction
    v_cond = model_fn(z_t, text_embeddings, timesteps)

    # Unconditional prediction
    v_uncond = model_fn(z_t, null_emb, timesteps)

    # Classifier-free guidance
    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

    return v_guided


def cfg_guided_prediction_batched(
    model_fn,
    z_t: torch.Tensor,
    text_embeddings: torch.Tensor,
    timesteps: torch.Tensor,
    guidance_scale: float = 5.0,
) -> torch.Tensor:
    """
    Perform CFG with batched conditional/unconditional predictions.

    More memory efficient than separate forward passes - doubles batch
    size instead of doubling forward passes.

    Args:
        Same as cfg_guided_prediction

    Returns:
        Guided prediction [B, T, D]

    Note:
        This is faster and more memory-efficient than cfg_guided_prediction
        for inference, but requires doubling the batch size temporarily.
    """
    if guidance_scale == 1.0:
        return model_fn(z_t, text_embeddings, timesteps)

    if guidance_scale == 0.0:
        null_emb = torch.zeros_like(text_embeddings)
        return model_fn(z_t, null_emb, timesteps)

    # Batch conditional and unconditional together
    null_emb = torch.zeros_like(text_embeddings)

    # Double batch: [cond, uncond]
    z_t_doubled = torch.cat([z_t, z_t], dim=0)
    text_doubled = torch.cat([text_embeddings, null_emb], dim=0)
    timesteps_doubled = torch.cat([timesteps, timesteps], dim=0)

    # Single forward pass for both
    v_doubled = model_fn(z_t_doubled, text_doubled, timesteps_doubled)

    # Split back into conditional and unconditional
    v_cond, v_uncond = v_doubled.chunk(2, dim=0)

    # Guidance
    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

    return v_guided
