"""
Learning rate scheduler factory for creating schedulers based on configuration.
Supports per-model scheduler selection with customizable parameters.
"""

from typing import Any, Dict, Union

import torch.optim as optim
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

SUPPORTED_SCHEDULERS = {
    "CosineAnnealingLR": CosineAnnealingLR,
    "LinearLR": LinearLR,
    "ExponentialLR": ExponentialLR,
    "ConstantLR": ConstantLR,
    "StepLR": StepLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
}


def validate_scheduler_config(scheduler_config: Dict[str, Any], total_steps: int) -> None:
    """
    Validate scheduler configuration parameters.

    Args:
        scheduler_config: Dictionary containing scheduler configuration
        total_steps: Total training steps

    Raises:
        ValueError: If any parameter is invalid
    """
    scheduler_type = scheduler_config.get("type", "CosineAnnealingLR")

    # Validate total_steps
    if total_steps <= 0:
        raise ValueError(f"Total steps must be positive, got {total_steps}")

    # Validate eta_min_factor for CosineAnnealingLR
    if scheduler_type == "CosineAnnealingLR":
        eta_min_factor = scheduler_config.get("eta_min_factor", 0.1)
        if not (0 < eta_min_factor <= 1):
            raise ValueError(f"eta_min_factor must be in (0, 1], got {eta_min_factor}")

    # Validate gamma for ExponentialLR and StepLR
    if scheduler_type in ["ExponentialLR", "StepLR"]:
        gamma = scheduler_config.get("gamma", 0.95 if scheduler_type == "ExponentialLR" else 0.1)
        if not (0 < gamma < 1):
            raise ValueError(f"Gamma must be in (0, 1), got {gamma}")

    # Validate step_size for StepLR
    if scheduler_type == "StepLR":
        step_size = scheduler_config.get("step_size", total_steps // 10)
        if step_size <= 0:
            raise ValueError(f"Step size must be positive, got {step_size}")

    # Validate factors for LinearLR
    if scheduler_type == "LinearLR":
        start_factor = scheduler_config.get("start_factor", 1.0)
        end_factor = scheduler_config.get("end_factor", 0.1)
        if start_factor <= 0:
            raise ValueError(f"Start factor must be positive, got {start_factor}")
        if end_factor <= 0:
            raise ValueError(f"End factor must be positive, got {end_factor}")

    # Validate factor for ConstantLR and ReduceLROnPlateau
    if scheduler_type in ["ConstantLR", "ReduceLROnPlateau"]:
        factor = scheduler_config.get("factor", 1.0 if scheduler_type == "ConstantLR" else 0.1)
        if factor <= 0:
            raise ValueError(f"Factor must be positive, got {factor}")
        if scheduler_type == "ReduceLROnPlateau" and factor >= 1:
            raise ValueError(f"ReduceLROnPlateau factor must be < 1, got {factor}")

    # Validate patience for ReduceLROnPlateau
    if scheduler_type == "ReduceLROnPlateau":
        patience = scheduler_config.get("patience", 10)
        if patience < 0:
            raise ValueError(f"Patience must be non-negative, got {patience}")


def create_scheduler(
    optimizer: optim.Optimizer, scheduler_config: Dict[str, Any], total_steps: int
) -> Union[_LRScheduler, ReduceLROnPlateau]:  # type: ignore[type-arg]
    """
    Create a learning rate scheduler based on configuration.

    Args:
        optimizer: Optimizer to schedule
        scheduler_config: Dictionary containing scheduler configuration
            - type: Scheduler type (CosineAnnealingLR, LinearLR, ExponentialLR, etc.)
            - eta_min_factor: Minimum LR as fraction of initial LR (for CosineAnnealingLR)
            - gamma: Decay rate (for ExponentialLR, StepLR)
            - step_size: Period of learning rate decay (for StepLR)
            - start_factor: Starting factor (for LinearLR)
            - end_factor: Ending factor (for LinearLR)
            - total_iters: Total iterations (for LinearLR, ConstantLR)
            - factor: Constant factor (for ConstantLR, ReduceLROnPlateau)
            - patience: Patience for ReduceLROnPlateau
        total_steps: Total training steps (used for schedulers that need it)

    Returns:
        Configured scheduler instance

    Raises:
        ValueError: If scheduler type is not supported
    """
    # Validate configuration before creating scheduler
    validate_scheduler_config(scheduler_config, total_steps)

    scheduler_type = scheduler_config.get("type", "CosineAnnealingLR")

    if scheduler_type not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_type}. "
            f"Supported schedulers: {list(SUPPORTED_SCHEDULERS.keys())}"
        )

    scheduler_class = SUPPORTED_SCHEDULERS[scheduler_type]

    # Get the initial LR from optimizer
    initial_lr = optimizer.param_groups[0]["lr"]

    # Scheduler-specific parameters
    if scheduler_type == "CosineAnnealingLR":
        eta_min_factor = scheduler_config.get("eta_min_factor", 0.1)
        eta_min = initial_lr * eta_min_factor
        return scheduler_class(optimizer, T_max=total_steps, eta_min=eta_min)  # type: ignore[return-value, call-arg, no-any-return]

    elif scheduler_type == "LinearLR":
        start_factor = scheduler_config.get("start_factor", 1.0)
        end_factor = scheduler_config.get("end_factor", 0.1)
        total_iters = scheduler_config.get("total_iters", total_steps)
        return scheduler_class(
            optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters
        )  # type: ignore[return-value, call-arg, no-any-return]

    elif scheduler_type == "ExponentialLR":
        gamma = scheduler_config.get("gamma", 0.95)
        return scheduler_class(optimizer, gamma=gamma)  # type: ignore[return-value, call-arg, no-any-return]

    elif scheduler_type == "ConstantLR":
        factor = scheduler_config.get("factor", 1.0)
        total_iters = scheduler_config.get("total_iters", total_steps)
        return scheduler_class(optimizer, factor=factor, total_iters=total_iters)  # type: ignore[return-value, call-arg, no-any-return]

    elif scheduler_type == "StepLR":
        step_size = scheduler_config.get("step_size", total_steps // 10)
        gamma = scheduler_config.get("gamma", 0.1)
        return scheduler_class(optimizer, step_size=step_size, gamma=gamma)  # type: ignore[return-value, call-arg, no-any-return]

    elif scheduler_type == "ReduceLROnPlateau":
        mode = scheduler_config.get("mode", "min")
        factor = scheduler_config.get("factor", 0.1)
        patience = scheduler_config.get("patience", 10)
        threshold = scheduler_config.get("threshold", 1e-4)
        return scheduler_class(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
        )  # type: ignore[return-value, call-arg, no-any-return]

    else:
        # Fallback to CosineAnnealingLR
        eta_min_factor = scheduler_config.get("eta_min_factor", 0.1)
        eta_min = initial_lr * eta_min_factor
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)  # type: ignore[return-value]


def get_default_scheduler_config(model_name: str) -> Dict[str, Any]:
    """
    Get default scheduler configuration for a specific model.

    Args:
        model_name: Name of the model (flow, vae, text_encoder, discriminator)

    Returns:
        Default scheduler configuration dictionary
    """
    defaults = {
        "flow": {
            "type": "CosineAnnealingLR",
            "eta_min_factor": 0.1,
        },
        "vae": {
            "type": "CosineAnnealingLR",
            "eta_min_factor": 0.1,
        },
        "text_encoder": {
            "type": "CosineAnnealingLR",
            "eta_min_factor": 0.001,  # More aggressive decay for text encoder
        },
        "discriminator": {
            "type": "CosineAnnealingLR",
            "eta_min_factor": 0.1,
        },
    }

    return defaults.get(model_name, defaults["vae"])
