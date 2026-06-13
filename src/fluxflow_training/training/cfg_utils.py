"""Classifier-Free Guidance utilities for FluxFlow.

Provides functions for applying CFG dropout during training and CFG-guided
sampling during inference.
"""

from typing import Tuple

import torch
from fluxflow.utils import get_logger

logger = get_logger(__name__)


def apply_cfg_null_substitution(
    text_seq: torch.Tensor,
    text_mask: torch.Tensor,
    text_encoder,
    p_uncond: float = 0.1,
    null_prompt: str = "",
    cache_attr_name: str = "_cfg_null_pair",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample empty-prompt substitution for CFG dropout (v0.10.0 redesign).

    Replaces the legacy ``text_embeddings *= (rand > p)`` zero-out with a real
    encoded empty prompt. The zero-out form is dangerous under per-token text
    because the all-False attention mask on a null sample NaN-out softmax in
    the flow's cross-attention. An encoded empty prompt is a finite, real null
    context and matches the (null_seq, null_mask) used at inference time —
    so train and sample CFG see the *same* unconditional distribution.

    The (null_seq, null_mask) pair is built once via ``build_cfg_null_pair``
    and cached as an attribute on ``text_encoder``. Subsequent calls reuse it.

    Args:
        text_seq: Per-token text embeddings [B, T_txt, E].
        text_mask: Bool mask [B, T_txt].
        text_encoder: The text encoder (used to build the null pair on first call).
        p_uncond: Per-sample probability of substituting the null. 0.0 disables.
        null_prompt: The text used to build the null pair. Default "".
        cache_attr_name: Attribute name on ``text_encoder`` used to cache the
            (null_seq, null_mask) pair across calls.

    Returns:
        (text_seq_out, text_mask_out): tensors with the same shapes as inputs,
        with ``p_uncond`` fraction of samples replaced by the encoded null.
    """
    if p_uncond < 0.0 or p_uncond > 1.0:
        raise ValueError(f"p_uncond must be in [0, 1], got {p_uncond}")
    if p_uncond == 0.0:
        return text_seq, text_mask

    batch_size = text_seq.size(0)
    device = text_seq.device

    # Build or retrieve cached null pair.
    cached = getattr(text_encoder, cache_attr_name, None)
    if cached is None or cached[0].size(1) != text_seq.size(1):
        # Either no cache yet, or T_txt has changed (e.g. a new run with a
        # different max_length). Rebuild and cache.
        null_seq, null_mask = _build_null_pair(
            text_encoder=text_encoder,
            max_length=text_seq.size(1),
            null_prompt=null_prompt,
        )
        setattr(text_encoder, cache_attr_name, (null_seq, null_mask))
        cached = (null_seq, null_mask)

    null_seq, null_mask = cached
    # Move cache to current device/dtype to allow cross-device use.
    null_seq = null_seq.to(device=device, dtype=text_seq.dtype)
    null_mask = null_mask.to(device=device)

    # Per-sample substitution mask.
    drop = torch.rand(batch_size, device=device) < p_uncond  # [B]

    text_seq_out = text_seq.clone()
    text_mask_out = text_mask.clone()
    if drop.any():
        # Broadcast null pair to substitute the rows where drop is True.
        text_seq_out[drop] = null_seq.expand_as(text_seq_out)[drop]
        text_mask_out[drop] = null_mask.expand_as(text_mask_out)[drop]

    return text_seq_out, text_mask_out


def _build_null_pair(
    text_encoder,
    max_length: int,
    null_prompt: str = "",
    tokenizer_name: str = "distilbert-base-uncased",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (null_seq, null_mask) for ``null_prompt`` via the text encoder.

    Prefers ``fluxflow.utils.visualization.build_cfg_null_pair`` when available
    (only handles the empty prompt). For arbitrary ``null_prompt`` strings we
    tokenize and forward locally.
    """
    if null_prompt == "":
        try:
            from fluxflow.utils.visualization import build_cfg_null_pair

            return build_cfg_null_pair(
                text_encoder, max_length=max_length, tokenizer_name=tokenizer_name
            )
        except ImportError:  # pragma: no cover - fluxflow-core too old
            pass

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    enc_in = tok(
        null_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    # Move tokenized inputs to encoder device when possible.
    try:
        enc_device = next(text_encoder.parameters()).device
        input_ids = enc_in["input_ids"].to(enc_device)
        attention_mask = enc_in["attention_mask"].to(enc_device)
    except StopIteration:
        input_ids = enc_in["input_ids"]
        attention_mask = enc_in["attention_mask"]

    with torch.no_grad():
        out = text_encoder(input_ids, attention_mask=attention_mask)
    if isinstance(out, tuple):
        return out
    # Legacy single-tensor encoder: synthesize a mask matching the attention.
    return out, attention_mask.bool()


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
