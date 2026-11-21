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


class TestEmaKeepsLiveBuffers:
    """EMA must never overwrite buffers (e.g. calibrated latent stats) with stale values."""

    @staticmethod
    def _pair():
        from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

        return nn.ModuleList(
            [
                FluxCompressor_v100(d_model=8, downscales=2, use_gradient_checkpointing=False),
                FluxExpander_v100(d_model=8, upscales=2, use_gradient_checkpointing=False),
            ]
        )

    def test_apply_shadow_keeps_calibrated_stats(self):
        model = self._pair()
        ema = EMA(model, decay=0.9)  # shadow snapshot has identity stats
        shift, scale = torch.full((16,), 0.3), torch.full((16,), 2.0)
        for m in model:
            m.set_latent_stats(shift, scale)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        raw = next(model.parameters()).detach().clone()
        with ema.applied():
            for m in model:
                assert torch.equal(m.latent_shift, shift)
                assert torch.equal(m.latent_scale, scale)
            # parameters ARE swapped to the shadow
            assert not torch.equal(next(model.parameters()), raw)
        for m in model:
            assert torch.equal(m.latent_shift, shift)

    def test_update_copies_buffers_and_copy_to_uses_live_buffers(self):
        model = self._pair()
        ema = EMA(model, decay=0.999)
        for m in model:
            m.set_latent_stats(torch.full((16,), 0.3), torch.full((16,), 2.0))
        ema.update()
        assert torch.equal(ema.shadow["0.latent_scale"], model[0].latent_scale)
        target = self._pair()
        ema.copy_to(target)
        assert torch.equal(target[0].latent_scale, model[0].latent_scale)
        assert torch.equal(target[1].latent_shift, model[1].latent_shift)
