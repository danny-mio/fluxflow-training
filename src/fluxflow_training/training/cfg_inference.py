"""Classifier-Free Guidance inference utilities for FluxFlow.

Provides high-level functions for generating images with CFG.
"""

import torch
from diffusers import DPMSolverMultistepScheduler
from fluxflow.utils import get_logger

logger = get_logger(__name__)


def generate_with_cfg(
    diffuser,
    text_embeddings: torch.Tensor,
    guidance_scale: float = 5.0,
    img_size: int = 512,
    num_steps: int = 30,
    batch_size: int = 1,
    device: str = "cuda",
    use_batched_cfg: bool = True,
) -> torch.Tensor:
    """
    Generate images using classifier-free guidance.

    Args:
        diffuser: FluxPipeline model with compressor, flow_processor, expander
        text_embeddings: Text conditioning [B, D_text]
        guidance_scale: CFG strength (ω)
                        - 1.0: Standard conditional
                        - 3.0-9.0: Typical range for flow matching
                        - Higher = stronger prompt adherence
        img_size: Generated image resolution (default: 512)
        num_steps: Number of denoising steps (default: 30)
        batch_size: Batch size (default: 1)
        device: Device (default: "cuda")
        use_batched_cfg: Use memory-efficient batched CFG (default: True)

    Returns:
        Generated images [B, 3, img_size, img_size]

    Example:
        >>> from fluxflow.models import FluxPipeline
        >>> from transformers import AutoTokenizer, AutoModel
        >>>
        >>> # Load models
        >>> diffuser = FluxPipeline(...)
        >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        >>> text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        >>>
        >>> # Encode prompt
        >>> prompt = "a beautiful sunset over mountains"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> text_emb = text_encoder(**inputs).last_hidden_state.mean(dim=1)
        >>>
        >>> # Generate with CFG
        >>> images = generate_with_cfg(
        ...     diffuser,
        ...     text_emb,
        ...     guidance_scale=5.0,
        ...     num_steps=30
        ... )
    """
    from .cfg_utils import cfg_guided_prediction, cfg_guided_prediction_batched

    # Initialize latent from random noise
    z_img = (torch.rand((batch_size, 3, img_size, img_size), device=device) * 2) - 1
    latent_z = diffuser.compressor(z_img)

    img_seq = latent_z[:, :-1, :].contiguous()
    hw_vec = latent_z[:, -1:, :].contiguous()

    noise_img = torch.randn_like(img_seq)

    # Setup scheduler
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        algorithm_type="dpmsolver++",
        solver_order=2,
        prediction_type="v_prediction",
        lower_order_final=True,
        timestep_spacing="trailing",
    )
    scheduler.set_timesteps(num_steps, device=device)  # type: ignore[arg-type]

    # Start from full noise
    timesteps_list = scheduler.timesteps  # type: ignore[attr-defined]
    latent = torch.cat([noise_img, hw_vec], dim=1)

    # Denoise with CFG
    logger.info(f"Generating with CFG (guidance_scale={guidance_scale}, steps={num_steps})")

    cfg_fn = cfg_guided_prediction_batched if use_batched_cfg else cfg_guided_prediction

    for i, t in enumerate(timesteps_list):
        # Normalize timestep to [0, 1] range expected by flow model
        t_normalized = t.float() / 1000.0
        t_batch = t_normalized.unsqueeze(0).expand(batch_size).to(device)

        # CFG prediction
        with torch.no_grad():
            pred = cfg_fn(
                model_fn=lambda z, txt, ts: diffuser.flow_processor(z, txt, ts),
                z_t=latent,
                text_embeddings=text_embeddings,
                timesteps=t_batch,
                guidance_scale=guidance_scale,
            )

        # Scheduler step (v-prediction)
        # Note: This is simplified - production code should use scheduler.step()
        latent = scheduler.step(pred, t, latent).prev_sample  # type: ignore[attr-defined]

        if (i + 1) % 10 == 0:
            logger.debug(f"Denoising step {i+1}/{num_steps}")

    # Decode to images
    decoded_images = diffuser.expander(latent)

    return decoded_images


def generate_comparison(
    diffuser,
    text_embeddings: torch.Tensor,
    guidance_scales: list[float],
    img_size: int = 512,
    num_steps: int = 30,
    device: str = "cuda",
) -> dict[float, torch.Tensor]:
    """
    Generate images with multiple guidance scales for comparison.

    Useful for visualizing CFG impact and finding optimal guidance scale.

    Args:
        diffuser: FluxPipeline model
        text_embeddings: Text conditioning [1, D_text]
        guidance_scales: List of guidance scales to try (e.g., [1.0, 3.0, 5.0, 7.0, 9.0])
        img_size: Image resolution
        num_steps: Denoising steps
        device: Device

    Returns:
        Dictionary mapping guidance_scale -> generated image [1, 3, H, W]

    Example:
        >>> results = generate_comparison(
        ...     diffuser,
        ...     text_emb,
        ...     guidance_scales=[1.0, 3.0, 5.0, 7.0],
        ...     num_steps=30
        ... )
        >>>
        >>> # Save comparison
        >>> from torchvision.utils import save_image
        >>> for scale, img in results.items():
        ...     save_image(img, f"output_cfg{scale}.png")
    """
    results = {}

    for scale in guidance_scales:
        logger.info(f"Generating with guidance_scale={scale}")
        img = generate_with_cfg(
            diffuser,
            text_embeddings,
            guidance_scale=scale,
            img_size=img_size,
            num_steps=num_steps,
            batch_size=1,
            device=device,
        )
        results[scale] = img

    return results


def generate_interpolation(
    diffuser,
    text_embeddings_start: torch.Tensor,
    text_embeddings_end: torch.Tensor,
    num_frames: int = 10,
    guidance_scale: float = 5.0,
    img_size: int = 512,
    num_steps: int = 30,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """
    Generate smooth interpolation between two text prompts using CFG.

    Args:
        diffuser: FluxPipeline model
        text_embeddings_start: Starting prompt embeddings [1, D]
        text_embeddings_end: Ending prompt embeddings [1, D]
        num_frames: Number of interpolation frames
        guidance_scale: CFG strength
        img_size: Image resolution
        num_steps: Denoising steps
        device: Device

    Returns:
        List of generated images (length = num_frames)

    Example:
        >>> # Encode prompts
        >>> prompt_a = "sunny beach"
        >>> prompt_b = "snowy mountain"
        >>> emb_a = encode_prompt(prompt_a, text_encoder, tokenizer)
        >>> emb_b = encode_prompt(prompt_b, text_encoder, tokenizer)
        >>>
        >>> # Generate interpolation
        >>> frames = generate_interpolation(
        ...     diffuser, emb_a, emb_b,
        ...     num_frames=20,
        ...     guidance_scale=5.0
        ... )
        >>>
        >>> # Save as video
        >>> import imageio
        >>> imageio.mimsave("interpolation.mp4", [img.cpu() for img in frames], fps=10)
    """
    frames = []

    for i in range(num_frames):
        # Linear interpolation in embedding space
        alpha = i / (num_frames - 1)
        text_emb = (1 - alpha) * text_embeddings_start + alpha * text_embeddings_end

        logger.info(f"Generating frame {i+1}/{num_frames} (alpha={alpha:.2f})")
        img = generate_with_cfg(
            diffuser,
            text_emb,
            guidance_scale=guidance_scale,
            img_size=img_size,
            num_steps=num_steps,
            batch_size=1,
            device=device,
        )
        frames.append(img)

    return frames
