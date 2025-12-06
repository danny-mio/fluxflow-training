"""FluxFlow training scripts.

CLI entry points for training and generation.
"""

from .generate import main as generate_main
from .train import main as train_main

__all__ = ["train_main", "generate_main"]
