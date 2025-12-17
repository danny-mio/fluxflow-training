#!/usr/bin/env python3
"""Validate a checkpoint for NaN/Inf values."""

import argparse
import sys
from pathlib import Path

import safetensors.torch
import torch


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """
    Validate checkpoint for NaN/Inf values.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if valid, False if NaN/Inf found
    """
    print(f"Validating checkpoint: {checkpoint_path}")

    # Load checkpoint
    if checkpoint_path.suffix == ".safetensors":
        state_dict = safetensors.torch.load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print(f"Loaded {len(state_dict)} parameters")

    # Check each parameter
    nan_params = []
    inf_params = []

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue

        if torch.isnan(param).any():
            nan_params.append(name)
            print(f"  ❌ NaN found in: {name}")
            print(f"     Shape: {param.shape}, NaN count: {torch.isnan(param).sum().item()}")

        if torch.isinf(param).any():
            inf_params.append(name)
            print(f"  ❌ Inf found in: {name}")
            print(f"     Shape: {param.shape}, Inf count: {torch.isinf(param).sum().item()}")

    # Summary
    print("\n" + "=" * 80)
    if nan_params or inf_params:
        print("❌ CHECKPOINT INVALID")
        print(f"   NaN found in {len(nan_params)} parameters")
        print(f"   Inf found in {len(inf_params)} parameters")
        return False
    else:
        print("✅ CHECKPOINT VALID")
        print("   No NaN or Inf values found")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate FluxFlow checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    is_valid = validate_checkpoint(args.checkpoint)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
