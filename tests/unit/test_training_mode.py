"""Unit tests for TrainingMode and TrainingComponent."""

from fluxflow_training.training.training_mode import TrainingComponent, TrainingMode


def test_training_mode_vae_only():
    """Test VAE-only training mode configuration."""

    class Args:
        train_vae = True
        gan_training = False
        train_diff = False
        train_diff_full = False

    mode = TrainingMode.from_args(Args())
    assert mode.needs_vae_samples() is True
    assert mode.needs_flow_samples() is False
    assert mode.requires_vae_trainer() is True
    assert mode.requires_flow_trainer() is False
    assert mode.is_training(TrainingComponent.VAE) is True
    assert mode.is_training(TrainingComponent.GAN) is False


def test_training_mode_gan_only():
    """Test GAN-only training mode configuration."""

    class Args:
        train_vae = False
        gan_training = True
        train_diff = False
        train_diff_full = False

    mode = TrainingMode.from_args(Args())
    assert mode.needs_vae_samples() is True
    assert mode.needs_flow_samples() is False
    assert mode.requires_vae_trainer() is True
    assert mode.requires_flow_trainer() is False
    assert mode.is_training(TrainingComponent.GAN) is True
    assert mode.is_training(TrainingComponent.VAE) is False


def test_training_mode_flow_only():
    """Test Flow-only training mode configuration."""

    class Args:
        train_vae = False
        gan_training = False
        train_diff = True
        train_diff_full = False

    mode = TrainingMode.from_args(Args())
    assert mode.needs_vae_samples() is False
    assert mode.needs_flow_samples() is True
    assert mode.requires_vae_trainer() is False
    assert mode.requires_flow_trainer() is True
    assert mode.is_training(TrainingComponent.FLOW) is True


def test_training_mode_flow_full_only():
    """Test full Flow training mode configuration."""

    class Args:
        train_vae = False
        gan_training = False
        train_diff = False
        train_diff_full = True

    mode = TrainingMode.from_args(Args())
    assert mode.needs_vae_samples() is False
    assert mode.needs_flow_samples() is True
    assert mode.requires_flow_trainer() is True
    assert mode.is_training(TrainingComponent.FLOW_FULL) is True


def test_training_mode_combined():
    """Test combined training mode (VAE + GAN + Flow)."""

    class Args:
        train_vae = True
        gan_training = True
        train_diff = True
        train_diff_full = False

    mode = TrainingMode.from_args(Args())
    assert mode.needs_vae_samples() is True
    assert mode.needs_flow_samples() is True
    assert mode.requires_vae_trainer() is True
    assert mode.requires_flow_trainer() is True
    assert mode.is_training(TrainingComponent.VAE) is True
    assert mode.is_training(TrainingComponent.GAN) is True
    assert mode.is_training(TrainingComponent.FLOW) is True


def test_training_mode_all_components():
    """Test all components enabled."""

    class Args:
        train_vae = True
        gan_training = True
        train_diff = True
        train_diff_full = True

    mode = TrainingMode.from_args(Args())
    assert mode.needs_vae_samples() is True
    assert mode.needs_flow_samples() is True
    assert mode.requires_vae_trainer() is True
    assert mode.requires_flow_trainer() is True


def test_training_mode_from_config():
    """Test creating TrainingMode from configuration dictionary."""
    config = {
        "train_vae": True,
        "gan_training": False,
        "train_diff": False,
        "train_diff_full": False,
    }
    mode = TrainingMode.from_config(config)
    assert mode.needs_vae_samples() is True
    assert mode.needs_flow_samples() is False
    assert mode.is_training(TrainingComponent.VAE) is True


def test_training_mode_from_config_flow():
    """Test creating Flow TrainingMode from configuration dictionary."""
    config = {
        "train_vae": False,
        "gan_training": False,
        "train_diff": True,
        "train_diff_full": False,
    }
    mode = TrainingMode.from_config(config)
    assert mode.needs_vae_samples() is False
    assert mode.needs_flow_samples() is True
    assert mode.is_training(TrainingComponent.FLOW) is True


def test_training_mode_none():
    """Test empty training mode (no components)."""

    class Args:
        train_vae = False
        gan_training = False
        train_diff = False
        train_diff_full = False

    mode = TrainingMode.from_args(Args())
    assert mode.needs_vae_samples() is False
    assert mode.needs_flow_samples() is False
    assert mode.requires_vae_trainer() is False
    assert mode.requires_flow_trainer() is False


def test_training_mode_repr():
    """Test string representation of TrainingMode."""

    class Args:
        train_vae = True
        gan_training = True
        train_diff = False
        train_diff_full = False

    mode = TrainingMode.from_args(Args())
    repr_str = repr(mode)
    assert "TrainingMode" in repr_str
    assert "VAE" in repr_str
    assert "GAN" in repr_str


def test_training_mode_repr_none():
    """Test string representation of empty TrainingMode."""

    class Args:
        train_vae = False
        gan_training = False
        train_diff = False
        train_diff_full = False

    mode = TrainingMode.from_args(Args())
    repr_str = repr(mode)
    assert repr_str == "TrainingMode(NONE)"


def test_training_component_bitwise():
    """Test bitwise operations on TrainingComponent flags."""
    combined = TrainingComponent.VAE | TrainingComponent.GAN
    assert bool(combined & TrainingComponent.VAE)
    assert bool(combined & TrainingComponent.GAN)
    assert not bool(combined & TrainingComponent.FLOW)


def test_spade_not_a_component():
    """SPADE is no longer a TrainingComponent flag."""
    assert not hasattr(TrainingComponent, "SPADE")


def test_needs_vae_samples_requires_vae_or_gan():
    """needs_vae_samples returns True only with VAE or GAN active."""
    vae_mode = TrainingMode(TrainingComponent.VAE)
    gan_mode = TrainingMode(TrainingComponent.GAN)
    flow_mode = TrainingMode(TrainingComponent.FLOW)
    none_mode = TrainingMode(TrainingComponent.NONE)

    assert vae_mode.needs_vae_samples() is True
    assert gan_mode.needs_vae_samples() is True
    assert flow_mode.needs_vae_samples() is False
    assert none_mode.needs_vae_samples() is False


def test_from_args_ignores_train_spade():
    """from_args silently ignores train_spade; no SPADE component exists."""

    class Args:
        train_vae = False
        gan_training = False
        train_spade = True  # should be ignored
        train_diff = False
        train_diff_full = False

    mode = TrainingMode.from_args(Args())
    # No SPADE component, and needs_vae_samples is False because neither VAE nor GAN active
    assert not hasattr(TrainingComponent, "SPADE")
    assert mode.needs_vae_samples() is False
