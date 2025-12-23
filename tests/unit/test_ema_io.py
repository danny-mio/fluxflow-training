"""Tests for EMA state saving and loading."""

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.utils import EMA


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


class TestEMAStateSaving:
    """Tests for EMA state saving and loading."""

    def test_saves_ema_state_to_file(self, temp_dir):
        """Should save EMA state to ema_state.pt."""
        # Create model and EMA
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.999)

        # Update EMA a few times
        for _ in range(5):
            # Modify model parameters
            for p in model.parameters():
                p.data += 0.1
            ema.update()

        # Save EMA state
        ema_path = temp_dir / "ema_state.pt"
        torch.save(ema.state_dict(), ema_path)

        assert ema_path.exists()

    def test_loads_ema_state_correctly(self, temp_dir):
        """Should load EMA state and maintain shadow parameters."""
        # Create model and EMA
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.999)

        # Update EMA
        for _ in range(5):
            for p in model.parameters():
                p.data += 0.1
            ema.update()

        # Save shadow parameters for comparison
        original_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Save EMA state
        ema_path = temp_dir / "ema_state.pt"
        torch.save(ema.state_dict(), ema_path)

        # Create new EMA and load state
        new_model = nn.Linear(10, 10)
        new_ema = EMA(new_model, decay=0.999)

        loaded_state = torch.load(ema_path, weights_only=False)
        new_ema.load_state_dict(loaded_state)

        # Shadow parameters should match
        for key in original_shadow:
            assert torch.allclose(original_shadow[key], new_ema.shadow[key])

        # Decay should match (allow for tensor->float conversion precision loss)
        assert abs(new_ema.decay - 0.999) < 1e-6
