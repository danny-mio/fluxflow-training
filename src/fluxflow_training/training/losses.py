"""Loss functions for FluxFlow GAN and VAE training."""

import torch
import torch.nn.functional as F


def _reduce_logits(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce patch-based discriminator logits to scalar per sample.

    Args:
        x: Logits tensor [B, ...] (scalar or spatial)

    Returns:
        Reduced logits [B]
    """
    if x.dim() == 4:
        return x.mean(dim=[1, 2, 3])  # patch -> scalar per item
    return x.view(x.size(0), -1).mean(dim=1)


def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for discriminator training.

    Encourages real_logits > 1 and fake_logits < -1.

    Args:
        real_logits: Discriminator output for real samples
        fake_logits: Discriminator output for fake samples

    Returns:
        Scalar hinge loss
    """
    r = F.relu(1.0 - _reduce_logits(real_logits)).mean()
    f = F.relu(1.0 + _reduce_logits(fake_logits)).mean()
    return r + f


def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for generator training.

    Encourages fake_logits > 0.

    Args:
        fake_logits: Discriminator output for generated samples

    Returns:
        Scalar hinge loss
    """
    return -_reduce_logits(fake_logits).mean()


def r1_penalty(real_imgs: torch.Tensor, d_out: torch.Tensor) -> torch.Tensor:
    """
    R1 gradient penalty for discriminator regularization.

    Penalizes gradient magnitude of discriminator output w.r.t. real images.

    Args:
        real_imgs: Real images tensor [B, C, H, W] (must require gradients)
        d_out: Discriminator output for real_imgs (scalar or spatial)

    Returns:
        Scalar R1 penalty
    """
    # Reduce spatial dimensions if patch discriminator
    if d_out.dim() == 4:
        d_out = d_out.mean(dim=[2, 3])

    grad_real = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=real_imgs,
        create_graph=True,
        only_inputs=True,
    )[0]
    return grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()


def kl_standard_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits_nats: float = 0.0,
    reduce: str = "mean",
    normalize_by_dims: bool = False,
) -> torch.Tensor:
    """
    KL divergence between learned Gaussian and standard normal.

    Optionally applies free-bits constraint (minimum KL per dimension).

    Args:
        mu: Mean tensor [B, D, H, W]
        logvar: Log variance tensor [B, D, H, W]
        free_bits_nats: Minimum KL in nats (0.0 = no constraint)
        reduce: 'mean' or 'sum' reduction over batch
        normalize_by_dims: If True, normalize KL by number of dimensions (resolution-invariant).
                          If False, sum over dimensions (legacy behavior).
                          **Recommended: True for new training runs.**

                          Why normalize?
                          - Legacy (False): KL scales with image resolution (e.g., 512x512 → ~150K)
                          - Normalized (True): KL is per-dimension (~1-2), resolution-invariant
                          - With normalize=True, increase kl_beta by ~10× (e.g., 0.0001 → 0.001)

    Returns:
        KL divergence scalar or per-sample tensor
    """
    # Element-wise KL: -0.5 * (1 + log(σ²) - μ² - σ²)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if free_bits_nats > 0.0:
        kl = torch.clamp(kl, min=free_bits_nats)

    if normalize_by_dims:
        # Mean over all dimensions (resolution-invariant, recommended for new training)
        kl = kl.mean(dim=(1, 2, 3))
    else:
        # Sum over spatial and channel dimensions (legacy behavior)
        kl = kl.sum(dim=(1, 2, 3))

    if reduce == "mean":
        return kl.mean()
    elif reduce == "sum":
        return kl.sum()
    return kl


def compute_ctx_shrinkage(ctx_features: torch.Tensor, alpha: float) -> torch.Tensor:
    """β-VAE-style shrinkage on deterministic ctx features.

    Computes ``L = alpha * mean(||ctx_features||^2)``. When ``alpha <= 0``
    the term vanishes and returns a zero scalar on the same device/dtype.

    Used as a regularizer on the pre-attention context features at the
    bottleneck (v0.10.0 redesign §5). Discourages the deterministic ctx
    branch from carrying information that is already representable by z,
    so SPADE residual coupling stays clean.

    Args:
        ctx_features: Deterministic context features tensor of any shape.
        alpha: Shrinkage weight. Non-positive values disable the term.

    Returns:
        Scalar shrinkage loss on the same device/dtype as ``ctx_features``.
    """
    if alpha <= 0:
        return torch.zeros((), device=ctx_features.device, dtype=ctx_features.dtype)
    return alpha * (ctx_features**2).mean()


def cosine_warmup_weight(step: int, warmup_steps: int, max_weight: float) -> float:
    """Cosine warmup from 0 to ``max_weight`` over ``warmup_steps`` steps.

    Identical math to :func:`schedulers.cosine_anneal_beta` but exposed in
    ``losses`` for the v0.10.0 KL / ctx shrinkage schedules so callers can
    import everything loss-related from one module.

    Args:
        step: Current global training step.
        warmup_steps: Number of warmup steps. ``<= 0`` returns ``max_weight``.
        max_weight: Asymptotic weight.

    Returns:
        Current weight in ``[0, max_weight]``.
    """
    import math

    if warmup_steps <= 0:
        return float(max_weight)
    frac = min(max(step / warmup_steps, 0.0), 1.0)
    return float(max_weight * (1 - math.cos(math.pi * frac)) / 2.0)


def delayed_cosine_warmup_weight(
    step: int, start_step: int, warmup_steps: int, max_weight: float
) -> float:
    """Cosine warmup that holds at 0 until ``start_step`` then ramps to ``max_weight``.

    Used by the v0.10.0 ctx shrinkage schedule (§5.5 of the design doc): the
    term is delayed until step 5000 then warms up over the next 5000 steps.

    Args:
        step: Current global training step.
        start_step: Step at which warmup begins. Returns 0 for ``step < start_step``.
        warmup_steps: Warmup length once ``start_step`` is reached.
        max_weight: Asymptotic weight.

    Returns:
        Current weight in ``[0, max_weight]``.
    """
    if step < start_step:
        return 0.0
    return cosine_warmup_weight(step - start_step, warmup_steps, max_weight)


def compute_mmd(
    z: torch.Tensor, z_prior: torch.Tensor, kernel: str = "rbf", sigma: float = 1.0
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with Gaussian (RBF) kernel.

    Measures distance between learned latent distribution and prior.

    Args:
        z: Learned latent samples [B, ...]
        z_prior: Prior samples [B, ...]
        kernel: Kernel type (currently only 'rbf' supported)
        sigma: RBF kernel bandwidth

    Returns:
        Scalar MMD loss
    """

    def gaussian_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        xx = x.unsqueeze(1).expand(x_size, y_size, dim)
        yy = y.unsqueeze(0).expand(x_size, y_size, dim)
        l2_dist = ((xx - yy) ** 2).mean(2)
        return torch.exp(-l2_dist / (2 * sigma**2))

    K_xx = gaussian_kernel(z, z)
    K_yy = gaussian_kernel(z_prior, z_prior)
    K_xy = gaussian_kernel(z, z_prior)
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
