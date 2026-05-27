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


def sample_t(
    batch_size: int, device: torch.device, start_step: int = 0, num_train_timesteps: int = 1000
) -> torch.Tensor:
    """
    Sample diffusion timesteps with logit-normal distribution (mode at t=500).

    Uses sigmoid(N(0,1)) to bias sampling toward the middle of the timestep range
    while still covering both high-noise and low-noise regimes uniformly.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to place tensor on
        start_step: Starting timestep (default: 0 for noise-to-image)
        num_train_timesteps: Total number of training timesteps (default: 1000)

    Returns:
        Timestep indices [batch_size] in range [start_step, num_train_timesteps-1]
    """
    # Logit-normal distribution: sigmoid(N(0,1)) has mode at 0.5, matching SD3/FLUX practice.
    # Ensures ~uniform coverage of high-noise (t≈999) and low-noise (t≈0) regimes.
    num_steps = num_train_timesteps - start_step
    u = torch.randn(batch_size, device=device)
    t_cont = torch.sigmoid(u)  # [0, 1], mode at 0.5
    indices = (t_cont * num_steps).long().clamp(0, num_steps - 1)
    return indices + start_step
