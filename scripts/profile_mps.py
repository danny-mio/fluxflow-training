"""
Profile MPS CPU fallbacks during a short VAE forward+backward pass.

Usage:
    python scripts/profile_mps.py

Requires Apple Silicon Mac with macOS 12.3+.
"""

import time
import warnings

import torch


def main():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("MPS not available. Run on Apple Silicon Mac.")
        return

    import os

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    fallback_ops: list[str] = []

    original_warn = warnings.showwarning

    def capture_warn(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if "MPS" in msg or "fallback" in msg.lower():
            fallback_ops.append(msg)
        original_warn(message, category, filename, lineno, file, line)

    warnings.showwarning = capture_warn

    device = torch.device("mps")
    print(f"Running on: {device}")

    from fluxflow.models.v070.vae import FluxCompressor, FluxExpander

    enc = FluxCompressor().to(device)
    dec = FluxExpander().to(device)

    img = torch.randn(1, 3, 64, 64, device=device)
    t0 = time.perf_counter()
    packed, mu, logvar = enc(img, training=True)
    out = dec(packed)
    loss = out.mean()
    loss.backward()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    warnings.showwarning = original_warn

    print(f"\nForward+backward time: {elapsed * 1000:.1f}ms")
    print(f"\nCPU fallback ops detected ({len(fallback_ops)}):")
    for op in set(fallback_ops):
        print(f"  - {op}")
    if not fallback_ops:
        print("  None! ✓")


if __name__ == "__main__":
    main()
