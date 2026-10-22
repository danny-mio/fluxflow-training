"""ROCm runtime workarounds applied at training startup."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from fluxflow.utils.rocm_compat import disable_broken_rocm_sdpa

__all__ = ["disable_broken_rocm_sdpa", "r1_conv_context"]


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
