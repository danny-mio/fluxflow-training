"""Schedulers and noise sampling for FluxFlow diffusion training."""

import math

import torch


def cosine_anneal_beta(step: int, total_steps: int, beta_max: float) -> float:
    """
    Cosine annealing schedule for KL divergence weight (β-VAE).

    Gradually increases β from 0 to beta_max using cosine schedule.

    Args:
        step: Current training step
        total_steps: Total warmup steps
        beta_max: Maximum β value

    Returns:
        Current β value in [0, beta_max]
    """
    if total_steps <= 0:
        return beta_max
    frac = min(max(step / total_steps, 0.0), 1.0)
    return float(beta_max * (1 - math.cos(math.pi * frac)) / 2.0)


def sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample diffusion timesteps with cosine-weighted distribution.

    Uses cosine schedule to bias sampling toward middle timesteps.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to place tensor on

    Returns:
        Timestep indices [batch_size] in range [0, 999]
    """
    s = torch.linspace(0, 1, 1000, device=device)
    weights = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    weights /= weights.sum()
    return torch.multinomial(weights, batch_size, replacement=True)
