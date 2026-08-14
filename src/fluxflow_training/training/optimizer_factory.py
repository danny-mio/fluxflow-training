"""
Optimizer factory for creating optimizers based on configuration.
Supports per-model optimizer selection with customizable hyperparameters.
"""

from typing import Any, Dict, Iterator, Optional, cast

import torch.nn as nn
import torch.optim as optim
from lion_pytorch import Lion

SUPPORTED_OPTIMIZERS = {
    "Lion": Lion,
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
}


def validate_optimizer_config(optimizer_config: Dict[str, Any]) -> None:
    """
    Validate optimizer configuration parameters.

    Args:
        optimizer_config: Dictionary containing optimizer configuration

    Raises:
        ValueError: If any parameter is invalid
    """
    optimizer_type = optimizer_config.get("type", "AdamW")
    lr = optimizer_config.get("lr", 1e-4)

    # Validate learning rate
    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {lr}")

    # Validate beta values for Adam-like optimizers
    if optimizer_type in ["Lion", "AdamW", "Adam"]:
        betas = optimizer_config.get("betas", [0.9, 0.999])
        if len(betas) != 2:
            raise ValueError(f"Betas must have exactly 2 values, got {len(betas)}")
        for i, beta in enumerate(betas):
            if not (0 <= beta < 1):
                raise ValueError(f"Beta{i+1} must be in [0, 1), got {beta}")

    # Validate weight decay
    weight_decay = optimizer_config.get("weight_decay", 0.0)
    if weight_decay < 0:
        raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")

    # Validate epsilon for Adam-like optimizers
    if optimizer_type in ["AdamW", "Adam", "RMSprop"]:
        eps = optimizer_config.get("eps", 1e-8)
        if eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {eps}")

    # Validate momentum for SGD and RMSprop
    if optimizer_type in ["SGD", "RMSprop"]:
        momentum = optimizer_config.get("momentum", 0.0)
        if not (0 <= momentum < 1):
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")

    # Validate alpha for RMSprop
    if optimizer_type == "RMSprop":
        alpha = optimizer_config.get("alpha", 0.99)
        if not (0 < alpha <= 1):
            raise ValueError(f"Alpha must be in (0, 1], got {alpha}")


def create_optimizer(
    parameters: Iterator[nn.Parameter], optimizer_config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Create an optimizer based on configuration.

    Args:
        parameters: Model parameters to optimize
        optimizer_config: Dictionary containing optimizer configuration
            - type: Optimizer type (Lion, AdamW, Adam, SGD, RMSprop)
            - lr: Learning rate
            - betas: Beta values for Adam-like optimizers (default: [0.9, 0.999])
            - weight_decay: Weight decay factor (default: 0.0)
            - momentum: Momentum for SGD (default: 0.0)
            - amsgrad: AMSGrad for Adam/AdamW (default: False)
            - decoupled_weight_decay: For Lion optimizer (default: True)

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer type is not supported
    """
    # Validate configuration before creating optimizer
    validate_optimizer_config(optimizer_config)

    optimizer_type = optimizer_config.get("type", "AdamW")
    lr = optimizer_config.get("lr", 1e-4)

    if optimizer_type not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. "
            f"Supported optimizers: {list(SUPPORTED_OPTIMIZERS.keys())}"
        )

    optimizer_class = cast(type[optim.Optimizer], SUPPORTED_OPTIMIZERS[optimizer_type])

    # Common parameters
    kwargs = {"lr": lr}

    # Optimizer-specific parameters
    if optimizer_type in ["Lion"]:
        # Lion-specific parameters
        kwargs["betas"] = tuple(optimizer_config.get("betas", [0.9, 0.95]))
        kwargs["weight_decay"] = optimizer_config.get("weight_decay", 0.01)
        kwargs["decoupled_weight_decay"] = optimizer_config.get("decoupled_weight_decay", True)

    elif optimizer_type in ["AdamW", "Adam"]:
        # Adam/AdamW-specific parameters
        kwargs["betas"] = tuple(optimizer_config.get("betas", [0.9, 0.999]))
        kwargs["weight_decay"] = optimizer_config.get("weight_decay", 0.0)
        kwargs["eps"] = optimizer_config.get("eps", 1e-8)

        if optimizer_type == "AdamW":
            kwargs["amsgrad"] = optimizer_config.get("amsgrad", False)

    elif optimizer_type == "SGD":
        # SGD-specific parameters
        kwargs["momentum"] = optimizer_config.get("momentum", 0.0)
        kwargs["weight_decay"] = optimizer_config.get("weight_decay", 0.0)
        kwargs["dampening"] = optimizer_config.get("dampening", 0.0)
        kwargs["nesterov"] = optimizer_config.get("nesterov", False)

    elif optimizer_type == "RMSprop":
        # RMSprop-specific parameters
        kwargs["alpha"] = optimizer_config.get("alpha", 0.99)
        kwargs["eps"] = optimizer_config.get("eps", 1e-8)
        kwargs["weight_decay"] = optimizer_config.get("weight_decay", 0.0)
        kwargs["momentum"] = optimizer_config.get("momentum", 0.0)
        kwargs["centered"] = optimizer_config.get("centered", False)

    result: optim.Optimizer = optimizer_class(parameters, **kwargs)
    return result


def get_default_optimizer_config(
    model_name: str, optimizer_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get default optimizer configuration for a specific model.

    Args:
        model_name: Name of the model (flow, vae, text_encoder, discriminator)
        optimizer_type: If given and equal to "Lion" while the model's native
            default is a different optimizer (currently vae, discriminator,
            text_encoder — flow already defaults to Lion), returns a
            Lion-appropriate config instead of the model's native (AdamW)
            defaults. Lion's update is sign(momentum) * lr — a
            constant-magnitude step every parameter every step — so reusing
            an AdamW-tuned lr/weight_decay causes oscillation instead of
            convergence (the same bug fixed for flow; see the "flow" entry
            below). Any other value (including None, the default) returns
            the model's native default dict unchanged — this parameter only
            rescales the specific Lion-vs-AdamW mismatch, not arbitrary
            optimizer-type swaps.

    Returns:
        Default optimizer configuration dictionary
    """
    # Lion-appropriate overrides for models that default to AdamW. Ratios
    # (~5x lower lr, ~5x higher weight_decay than the model's AdamW default)
    # mirror the same Chen et al. 2023 guidance already applied to flow.
    lion_overrides = {
        "vae": {
            "type": "Lion",
            "lr": 1e-7,  # AdamW default 5e-7 / 5
            "betas": [0.9, 0.95],
            "weight_decay": 0.05,  # AdamW default 0.01 * 5
            "decoupled_weight_decay": True,
        },
        "text_encoder": {
            "type": "Lion",
            "lr": 1e-8,  # AdamW default 5e-8 / 5
            "betas": [0.9, 0.95],
            "weight_decay": 0.05,  # AdamW default 0.01 * 5
            "decoupled_weight_decay": True,
        },
        "discriminator": {
            "type": "Lion",
            "lr": 1e-7,  # AdamW default 5e-7 / 5
            "betas": [0.9, 0.95],
            "weight_decay": 0.005,  # AdamW default 0.001 * 5
            "decoupled_weight_decay": True,
        },
    }

    defaults = {
        "flow": {
            "type": "Lion",
            # Lion's update is sign(momentum) * lr: a constant-magnitude step
            # every parameter every step, unlike AdamW's variance-normalized
            # step which naturally shrinks near a minimum. Per Chen et al.
            # 2023, Lion needs ~3-10x smaller lr and ~3-10x larger
            # weight_decay than an equivalent AdamW setting (was lr=5e-7,
            # weight_decay=0.01 — identical to AdamW's, causing oscillation
            # instead of convergence).
            "lr": 1e-7,
            "betas": [0.9, 0.95],
            "weight_decay": 0.05,
            "decoupled_weight_decay": True,
        },
        "vae": {
            "type": "AdamW",
            "lr": 5e-7,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
        },
        "text_encoder": {
            "type": "AdamW",
            "lr": 5e-8,  # 1/10 of flow lr
            "betas": [0.9, 0.99],
            "weight_decay": 0.01,
        },
        "discriminator": {
            "type": "AdamW",
            "lr": 5e-7,
            "betas": [0.0, 0.9],
            "weight_decay": 0.001,
            "amsgrad": True,
        },
    }

    base = defaults.get(model_name, defaults["vae"])

    if optimizer_type == "Lion" and base["type"] != "Lion":
        override_key = model_name if model_name in lion_overrides else "vae"
        return lion_overrides[override_key]

    return base
