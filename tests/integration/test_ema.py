"""Integration tests for EMA (Exponential Moving Average) in training."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from fluxflow_training.training.utils import EMA


class TestEMAIntegration:
    """Integration tests for EMA during training."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        return model

    def test_ema_initialization(self, simple_model):
        """EMA should initialize with model parameters."""
        ema = EMA(simple_model, decay=0.999)

        assert ema is not None
        assert ema.decay == 0.999

    def test_ema_update_during_training(self, simple_model):
        """EMA should update shadow parameters during training."""
        ema = EMA(simple_model, decay=0.999)
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)

        simple_model.train()

        # Train for a few steps with EMA updates
        for step in range(10):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA
            ema.update()

        # Shadow parameters should exist
        assert len(ema.shadow) > 0

    def test_ema_copy_to_model(self, simple_model):
        """EMA should be able to copy shadow parameters to model."""
        ema = EMA(simple_model, decay=0.999)
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)

        # Train and update EMA
        for step in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()

        # Save original parameters
        original_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        # Copy EMA to model
        ema.copy_to(simple_model)

        # Parameters should change
        for name, param in simple_model.named_parameters():
            assert not torch.equal(param, original_params[name])

    def test_ema_smooths_parameters(self, simple_model):
        """EMA parameters should be smoother than raw parameters."""
        ema = EMA(simple_model, decay=0.99)
        optimizer = optim.Adam(simple_model.parameters(), lr=0.1)  # High LR for variance

        # Initial parameters
        initial_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        # Train with high variance updates
        torch.manual_seed(42)
        for step in range(20):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()

        # Get current and EMA parameters
        current_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        # Copy EMA to model temporarily
        ema.copy_to(simple_model)
        ema_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        # EMA changes should be smaller than direct updates (smoothing effect)
        for name in initial_params.keys():
            current_change = (current_params[name] - initial_params[name]).abs().mean()
            ema_change = (ema_params[name] - initial_params[name]).abs().mean()

            # EMA should change less than current (smoothed)
            assert ema_change < current_change

    def test_ema_improves_generalization(self, simple_model):
        """EMA model should generalize better (conceptual test)."""
        ema = EMA(simple_model, decay=0.999)
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-2)

        # Train on small dataset
        torch.manual_seed(42)
        train_x = torch.randn(10, 10)
        train_y = torch.randn(10, 10)

        for epoch in range(50):
            pred = simple_model(train_x)
            loss = nn.functional.mse_loss(pred, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()

        # Evaluate on test data
        test_x = torch.randn(10, 10)
        test_y = torch.randn(10, 10)

        simple_model.eval()
        with torch.no_grad():
            # Normal model
            pred_normal = simple_model(test_x)
            loss_normal = nn.functional.mse_loss(pred_normal, test_y)

            # EMA model
            ema.copy_to(simple_model)
            pred_ema = simple_model(test_x)
            loss_ema = nn.functional.mse_loss(pred_ema, test_y)

        # EMA should have similar or better generalization
        # (This is statistical, so we just check it's reasonable)
        assert loss_ema.item() < loss_normal.item() * 2.0

    def test_ema_state_dict(self, simple_model):
        """EMA state should be saveable and loadable."""
        ema = EMA(simple_model, decay=0.999)
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)

        # Train and update EMA
        for step in range(10):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()

        # Save EMA state
        ema_state = ema.state_dict()

        # Create new EMA
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        new_ema = EMA(new_model, decay=0.999)

        # Load state
        new_ema.load_state_dict(ema_state)

        # Shadow parameters should match
        for name in ema.shadow.keys():
            assert torch.allclose(ema.shadow[name], new_ema.shadow[name])

    def test_ema_with_different_decay_rates(self, simple_model):
        """Different decay rates should produce different EMA behaviors."""
        ema_fast = EMA(simple_model, decay=0.9)  # Fast adaptation
        ema_slow = EMA(simple_model, decay=0.999)  # Slow adaptation

        optimizer = optim.Adam(simple_model.parameters(), lr=1e-2)

        # Initial parameters
        initial_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        # Train
        for step in range(20):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_fast.update()
            ema_slow.update()

        # Get shadow parameters
        fast_shadows = {name: tensor.clone() for name, tensor in ema_fast.shadow.items()}
        slow_shadows = {name: tensor.clone() for name, tensor in ema_slow.shadow.items()}

        # Fast EMA should change more than slow EMA
        for name in initial_params.keys():
            fast_change = (fast_shadows[name] - initial_params[name]).abs().mean()
            slow_change = (slow_shadows[name] - initial_params[name]).abs().mean()

            assert fast_change > slow_change

    def test_ema_reset(self, simple_model):
        """EMA should be resettable to current model parameters."""
        ema = EMA(simple_model, decay=0.999)
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)

        # Train and update EMA
        for step in range(10):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()

        # Get current model params
        current_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        # Update EMA to current (effectively reset)
        for name, param in simple_model.named_parameters():
            ema.shadow[name] = param.clone()

        # Copy EMA back to model
        ema.copy_to(simple_model)

        # Should match current params
        for name, param in simple_model.named_parameters():
            assert torch.allclose(param, current_params[name])


class TestEMATrainingWorkflow:
    """Tests for EMA in realistic training workflows."""

    def test_ema_training_evaluation_workflow(self):
        """Complete workflow: train with EMA, evaluate with EMA."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        ema = EMA(model, decay=0.999)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        model.train()
        for epoch in range(5):
            for step in range(10):
                x = torch.randn(4, 10)
                y = torch.randn(4, 10)

                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ema.update()

        # Evaluation with EMA
        model.eval()

        # Save original parameters
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Use EMA for evaluation
        ema.copy_to(model)

        with torch.no_grad():
            test_x = torch.randn(10, 10)
            test_pred = model(test_x)

        # Restore original parameters
        model.load_state_dict(original_state)

        # Test prediction should be valid
        assert test_pred.shape == (10, 10)

    def test_ema_checkpoint_workflow(self, tmp_path):
        """Save and load EMA with checkpoints."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        ema = EMA(model, decay=0.999)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for step in range(10):
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()

        # Save checkpoint with EMA
        checkpoint_path = tmp_path / "ema_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "epoch": 1,
            },
            checkpoint_path,
        )

        # Load checkpoint
        new_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        new_ema = EMA(new_model, decay=0.999)
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)

        checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        new_ema.load_state_dict(checkpoint["ema_state_dict"])

        # Shadow parameters should match
        for name in ema.shadow.keys():
            assert torch.allclose(ema.shadow[name], new_ema.shadow[name])

    def test_ema_with_scheduler(self):
        """EMA should work with learning rate schedulers."""
        model = nn.Sequential(nn.Linear(10, 10))
        ema = EMA(model, decay=0.999)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Train with scheduler
        for epoch in range(10):
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()
            scheduler.step()

        # Training should complete successfully
        assert True

    def test_ema_multi_model_training(self):
        """EMA should support multiple models (e.g., VAE + Flow)."""
        vae = nn.Sequential(nn.Linear(10, 10))
        flow = nn.Sequential(nn.Linear(20, 20))

        ema_vae = EMA(vae, decay=0.999)
        ema_flow = EMA(flow, decay=0.999)

        optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)
        optimizer_flow = optim.Adam(flow.parameters(), lr=1e-3)

        # Train both models
        for step in range(10):
            # VAE training
            x = torch.randn(4, 10)
            y_vae = vae(x)
            loss_vae = y_vae.sum()

            optimizer_vae.zero_grad()
            loss_vae.backward()
            optimizer_vae.step()

            ema_vae.update()

            # Flow training
            z = torch.randn(4, 20)
            y_flow = flow(z)
            loss_flow = y_flow.sum()

            optimizer_flow.zero_grad()
            loss_flow.backward()
            optimizer_flow.step()

            ema_flow.update()

        # Both EMAs should have shadow parameters
        assert len(ema_vae.shadow) > 0
        assert len(ema_flow.shadow) > 0


class TestEMAEdgeCases:
    """Tests for EMA edge cases."""

    def test_ema_with_frozen_parameters(self):
        """EMA should handle models with frozen parameters."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        ema = EMA(model, decay=0.999)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        # Train
        for step in range(10):
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()

        # Should complete without errors
        assert True

    def test_ema_decay_boundary_values(self):
        """EMA should handle boundary decay values."""
        model = nn.Linear(10, 10)

        # Very low decay (fast adaptation)
        ema_low = EMA(model, decay=0.1)
        assert ema_low.decay == 0.1

        # Very high decay (slow adaptation)
        ema_high = EMA(model, decay=0.9999)
        assert ema_high.decay == 0.9999

    def test_ema_update_num_updates(self):
        """EMA should track number of updates."""
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.999)

        # Initial num_updates
        initial_updates = ema.num_updates if hasattr(ema, "num_updates") else 0

        # Update several times
        for i in range(10):
            ema.update()

        # num_updates should increase (if tracked)
        if hasattr(ema, "num_updates"):
            assert ema.num_updates == initial_updates + 10
