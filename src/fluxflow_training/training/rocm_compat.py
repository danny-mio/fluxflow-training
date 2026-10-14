"""ROCm runtime workarounds applied at training startup."""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F

# (backend name, getter, setter) for the fused SDPA kernels that ROCm serves
# through AOTriton. MATH is never disabled: it is the always-working fallback.
_FUSED_SDPA_BACKENDS = (
    (
        "EFFICIENT_ATTENTION",
        torch.backends.cuda.mem_efficient_sdp_enabled,
        torch.backends.cuda.enable_mem_efficient_sdp,
    ),
    (
        "FLASH_ATTENTION",
        torch.backends.cuda.flash_sdp_enabled,
        torch.backends.cuda.enable_flash_sdp,
    ),
)


def _probe_sdpa_backend(backend_name: str, device: torch.device) -> bool:
    """Run a tiny fwd+bwd SDPA pinned to one backend; return True if it works.

    "No available kernel" (backend legitimately unsupported for these inputs)
    also counts as a failure here, but that is harmless: SDPA's dispatcher
    would skip that backend anyway.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = getattr(SDPBackend, backend_name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dtype in (torch.float32, torch.bfloat16):
                q = torch.randn(2, 2, 64, 8, device=device, dtype=dtype, requires_grad=True)
                with sdpa_kernel([backend]):
                    out = F.scaled_dot_product_attention(q, q, q)
                out.sum().backward()
            torch.cuda.synchronize(device)
        return True
    except Exception:
        return False


def disable_broken_rocm_sdpa(device: torch.device) -> list[str]:
    """Disable fused SDPA backends that fail on the current ROCm build.

    Some ROCm torch nightlies (e.g. 2.14+rocm7.16 on gfx1151) ship AOTriton
    flash/mem-efficient kernels that raise ``hipErrorInvalidValue`` at launch.
    The error surfaces asynchronously at the next op (typically
    ``nn.MultiheadAttention``'s ``out_proj``). Probing each backend once and
    disabling the broken ones makes SDPA fall back to the MATH kernel.

    No-op on non-ROCm builds or non-CUDA devices.

    Args:
        device: Training device.

    Returns:
        Names of the backends that were disabled.
    """
    if torch.version.hip is None or device.type != "cuda":
        return []

    disabled = []
    for name, is_enabled, set_enabled in _FUSED_SDPA_BACKENDS:
        if is_enabled() and not _probe_sdpa_backend(name, device):
            set_enabled(False)
            disabled.append(name)
    return disabled


@contextmanager
def r1_conv_context(enabled: bool) -> Iterator[None]:
    """Route convs to PyTorch's native kernels on ROCm R1 steps.

    On ROCm 7.16 nightlies MIOpen's GEMM im2col kernel (``MIOpenIm2d2Col.cpp``)
    fails HIPRTC compilation on gfx1151, and in FAST find mode MIOpen picks it
    for the R1 double-backward through the discriminator convs. Disabling the
    cuDNN/MIOpen backend for the R1 forward *and* the following backward
    sidesteps it. Cheaper than ``MIOPEN_DEBUG_CONV_GEMM=0`` (slows every conv,
    ~2x per batch) or ``MIOPEN_FIND_MODE=NORMAL`` (tunes every bucket shape).

    Toggles ``cudnn.enabled`` directly: ``cudnn.flags()`` also resets the
    benchmark limit, which warns on MIOpen and train.py escalates UserWarnings.

    Args:
        enabled: True on steps that compute the R1 penalty.
    """
    if not enabled or torch.version.hip is None:
        yield
        return
    previous = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = previous
