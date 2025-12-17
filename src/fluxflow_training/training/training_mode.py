"""Training mode management for FluxFlow.

Provides enum-based abstractions for determining which components are active
during training, replacing fragile conditional logic with maintainable helpers.
"""

from enum import Flag, auto
from typing import Any, Dict


class TrainingComponent(Flag):
    """Flags for active training components.

    Use bitwise operations to combine multiple components:
        mode = TrainingComponent.VAE | TrainingComponent.GAN
    """

    NONE = 0
    VAE = auto()  # 0b00001 - VAE reconstruction training
    GAN = auto()  # 0b00010 - GAN adversarial training
    SPADE = auto()  # 0b00100 - SPADE conditioning
    FLOW = auto()  # 0b01000 - Flow model training
    FLOW_FULL = auto()  # 0b10000 - Full flow training (includes text encoder)


class TrainingMode:
    """High-level training mode configuration.

    Provides helper methods to determine what samples should be generated
    and which trainers are needed based on active components.
    """

    def __init__(self, components: TrainingComponent):
        """Initialize training mode.

        Args:
            components: Bitwise combination of TrainingComponent flags
        """
        self.components = components

    @classmethod
    def from_args(cls, args: Any) -> "TrainingMode":
        """Create TrainingMode from argparse args object.

        Args:
            args: Argparse namespace with training flags

        Returns:
            TrainingMode instance
        """
        components = TrainingComponent.NONE

        if getattr(args, "train_vae", False):
            components |= TrainingComponent.VAE

        if getattr(args, "gan_training", False):
            components |= TrainingComponent.GAN

        if getattr(args, "train_spade", False):
            components |= TrainingComponent.SPADE

        if getattr(args, "train_diff", False):
            components |= TrainingComponent.FLOW

        if getattr(args, "train_diff_full", False):
            components |= TrainingComponent.FLOW_FULL

        return cls(components=components)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrainingMode":
        """Create TrainingMode from configuration dictionary.

        Args:
            config: Configuration dict with training flags

        Returns:
            TrainingMode instance
        """
        components = TrainingComponent.NONE

        if config.get("train_vae", False):
            components |= TrainingComponent.VAE

        if config.get("gan_training", False):
            components |= TrainingComponent.GAN

        if config.get("train_spade", False):
            components |= TrainingComponent.SPADE

        if config.get("train_diff", False):
            components |= TrainingComponent.FLOW

        if config.get("train_diff_full", False):
            components |= TrainingComponent.FLOW_FULL

        return cls(components=components)

    def needs_vae_samples(self) -> bool:
        """Check if VAE samples should be generated.

        VAE samples show encoder/decoder quality improvements across:
        - VAE mode (reconstruction loss training)
        - GAN-only mode (adversarial loss without reconstruction)
        - SPADE mode (decoder SPADE conditioning)

        Returns:
            True if VAE samples should be generated
        """
        vae_components = TrainingComponent.VAE | TrainingComponent.GAN | TrainingComponent.SPADE
        return bool(self.components & vae_components)

    def needs_flow_samples(self) -> bool:
        """Check if Flow samples should be generated.

        Returns:
            True if Flow samples should be generated
        """
        flow_components = TrainingComponent.FLOW | TrainingComponent.FLOW_FULL
        return bool(self.components & flow_components)

    def is_training(self, component: TrainingComponent) -> bool:
        """Check if a specific component is being trained.

        Args:
            component: Component to check

        Returns:
            True if component is active
        """
        return bool(self.components & component)

    def requires_vae_trainer(self) -> bool:
        """Check if VAE trainer should be initialized.

        Returns:
            True if VAE trainer is needed
        """
        return bool(self.components & (TrainingComponent.VAE | TrainingComponent.GAN))

    def requires_flow_trainer(self) -> bool:
        """Check if Flow trainer should be initialized.

        Returns:
            True if Flow trainer is needed
        """
        return bool(self.components & (TrainingComponent.FLOW | TrainingComponent.FLOW_FULL))

    def __repr__(self) -> str:
        """String representation of training mode."""
        active = []
        if self.is_training(TrainingComponent.VAE):
            active.append("VAE")
        if self.is_training(TrainingComponent.GAN):
            active.append("GAN")
        if self.is_training(TrainingComponent.SPADE):
            active.append("SPADE")
        if self.is_training(TrainingComponent.FLOW):
            active.append("FLOW")
        if self.is_training(TrainingComponent.FLOW_FULL):
            active.append("FLOW_FULL")

        if not active:
            return "TrainingMode(NONE)"

        return f"TrainingMode({' | '.join(active)})"
