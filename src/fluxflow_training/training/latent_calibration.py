"""Per-channel latent statistics calibration for v0.10.0 VAEs.

The flow's noise schedule assumes ~N(0, I) latents. v0.10.0 compressors/expanders carry
``latent_shift`` / ``latent_scale`` buffers (identity until calibrated); this module
measures the RAW per-channel statistics of the compressor's packed output so they can
be written into both modules via ``FluxPipeline.set_latent_stats``.

Shared by the ``fluxflow-calibrate-latents`` CLI and the pipeline orchestrator's
automatic calibration at the VAE -> flow step transition.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import torch
from fluxflow.models.latent_stats import apply_latent_denorm

from .utils import encode_latent

BatchItem = Union[torch.Tensor, Sequence[torch.Tensor], None]


def has_latent_stats(module: Any) -> bool:
    """True if ``module`` exposes v0.10.0 latent-stat buffers and calibration state."""
    return hasattr(module, "latent_shift") and hasattr(module, "latent_scale")


def _module_device(compressor: Any) -> torch.device:
    """Device of the compressor (buffers first, parameters as fallback)."""
    shift = getattr(compressor, "latent_shift", None)
    if isinstance(shift, torch.Tensor):
        return shift.device
    for p in compressor.parameters():
        return p.device  # type: ignore[no-any-return]
    return torch.device("cpu")


def compute_latent_stats(
    compressor: Any,
    batches: Iterable[BatchItem],
    max_batches: int,
    autocast: Optional[Callable[[], AbstractContextManager]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel RAW (pre-normalization) mean/std of the compressor's token rows.

    Each item of ``batches`` is one image tensor ``[B, 3, H, W]`` in [-1, 1] or a
    sequence of such tensors (one per resolution of a multi-resolution batch; every one
    is encoded). ``None`` items (fast-forward placeholders) are skipped and not counted.
    At most ``max_batches`` items are consumed.

    Latents come from ``encode_latent(compressor, imgs, sample=True)`` -- the same call
    FlowTrainer uses -- and are mapped back to raw space with
    ``apply_latent_denorm(packed, compressor.latent_shift, compressor.latent_scale)``,
    so the result does not depend on (and does not modify) the current stats. The hw
    row (last row of the packed tensor) is excluded. Sums accumulate in float64.

    Runs under ``torch.no_grad()`` and ``autocast()`` (if given) with the compressor in
    eval mode and gradient checkpointing off; both are restored afterwards.

    Returns:
        ``(mean, std)``: float32 CPU tensors of shape ``[C]`` (population std).

    Raises:
        ValueError: if no batch was consumed.
    """
    was_training = bool(getattr(compressor, "training", False))
    had_ckpt = hasattr(compressor, "use_gradient_checkpointing")
    orig_ckpt = getattr(compressor, "use_gradient_checkpointing", None)
    device = _module_device(compressor)
    ctx = autocast or nullcontext

    total: Optional[torch.Tensor] = None
    total_sq: Optional[torch.Tensor] = None
    count = 0
    n_batches = 0
    compressor.eval()
    if had_ckpt:
        compressor.use_gradient_checkpointing = False
    try:
        with torch.no_grad():
            for item in batches:
                if n_batches >= max_batches:
                    break
                if item is None:
                    continue
                tensors = [item] if isinstance(item, torch.Tensor) else list(item)
                for imgs in tensors:
                    with ctx():
                        packed = encode_latent(compressor, imgs.to(device), sample=True)
                    if has_latent_stats(compressor):
                        packed = apply_latent_denorm(
                            packed.float(), compressor.latent_shift, compressor.latent_scale
                        )
                    tokens = packed[:, :-1].reshape(-1, packed.shape[-1]).double()
                    s, sq = tokens.sum(0), (tokens * tokens).sum(0)
                    total = s if total is None else total + s
                    total_sq = sq if total_sq is None else total_sq + sq
                    count += tokens.shape[0]
                n_batches += 1
    finally:
        compressor.train(was_training)
        if had_ckpt:
            compressor.use_gradient_checkpointing = orig_ckpt

    if total is None or total_sq is None or count == 0:
        raise ValueError("compute_latent_stats: no batches were provided")
    mean = total / count
    var = (total_sq / count - mean * mean).clamp(min=0.0)
    return mean.float().cpu(), var.sqrt().float().cpu()


def format_latent_stats_table(mean: torch.Tensor, std: torch.Tensor, context_dims: int = 0) -> str:
    """Human-readable per-channel table; the last ``context_dims`` channels are ctx."""
    n = mean.numel()
    n_z = n - context_dims
    lines = [f"{'ch':>4}  {'part':<4}  {'mean':>10}  {'std':>10}"]
    for i in range(n):
        part = "z" if i < n_z else "ctx"
        lines.append(f"{i:>4}  {part:<4}  {float(mean[i]):>10.4f}  {float(std[i]):>10.4f}")
    for part, sl in (("z", slice(0, n_z)), ("ctx", slice(n_z, n))):
        if sl.stop > sl.start:
            lines.append(
                f"{part:>4} summary: |mean| max {mean[sl].abs().max():.4f}, "
                f"std min {std[sl].min():.4f} / max {std[sl].max():.4f}"
            )
    return "\n".join(lines)


def calibrate_models(
    models: dict,
    batches: Iterable[BatchItem],
    max_batches: int,
    autocast: Optional[Callable[[], AbstractContextManager]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure raw stats with ``models["compressor"]`` and write them into both modules.

    Uses ``models["diffuser"].set_latent_stats`` when a FluxPipeline is present, else
    sets compressor and expander directly.

    Returns:
        The ``(mean, std)`` that were applied.
    """
    compressor = models["compressor"]
    mean, std = compute_latent_stats(compressor, batches, max_batches, autocast=autocast)
    diffuser = models.get("diffuser")
    if diffuser is not None and hasattr(diffuser, "set_latent_stats"):
        diffuser.set_latent_stats(mean, std)
    else:
        for key in ("compressor", "expander"):
            module = models.get(key)
            if module is not None and has_latent_stats(module):
                module.set_latent_stats(mean, std)
    return mean, std


def context_dims_of(compressor: Any) -> int:
    """``compressor.get_context_dims()`` or 0 when unavailable."""
    getter = getattr(compressor, "get_context_dims", None)
    try:
        return int(getter()) if callable(getter) else 0
    except Exception:
        return 0
