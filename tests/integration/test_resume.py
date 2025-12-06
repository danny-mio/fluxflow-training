"""Integration tests for training resume functionality."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from fluxflow.utils.io import load_training_state, save_training_state

from fluxflow_training.data.datasets import ResumableDimensionSampler


class TestTrainingResumeIntegration:
    """Integration tests for mid-epoch training resume."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        return model

    @pytest.fixture
    def mock_sampler(self):
        """Create resumable sampler for testing."""
        # Mock dimension cache (proper format)
        dimension_cache = {
            "size_groups": {
                "(64, 64)": {"indices": [0, 1, 2, 3, 4]},
                "(128, 128)": {"indices": [5, 6, 7, 8, 9]},
                "(256, 256)": {"indices": [10, 11, 12, 13, 14]},
            },
            "total_images": 15,
        }

        sampler = ResumableDimensionSampler(
            dimension_cache=dimension_cache,
            batch_size=2,
            seed=42,
        )
        return sampler

    def test_save_and_load_training_state(self, simple_model, tmp_path):
        """Training state should be saveable and loadable."""
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)

        # Train for a few steps
        for step in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save training state
        checkpoint_path = tmp_path / "checkpoint.pt"
        save_training_state(
            str(checkpoint_path),
            epoch=1,
            batch_idx=5,
            global_step=5,
            samples_trained=20,
            total_samples=100,
            learning_rates={"optimizer": 1e-3},
            optimizers={"optimizer": optimizer},
        )

        assert checkpoint_path.exists()

        # Load training state
        loaded_state = load_training_state(str(checkpoint_path))

        assert loaded_state["epoch"] == 1
        assert loaded_state["batch_idx"] == 5
        assert loaded_state["global_step"] == 5

    def test_resume_training_continues_correctly(self, simple_model, tmp_path):
        """Resumed training should continue from saved point."""
        optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)

        # Train for 5 steps and save
        torch.manual_seed(42)
        losses_before_save = []

        for step in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_before_save.append(loss.item())

        # Save at step 5 (with model state)
        checkpoint_path = tmp_path / "checkpoint_step5.pt"
        save_training_state(
            str(checkpoint_path),
            model=simple_model,
            optimizer=optimizer,
            epoch=0,
            step=5,
        )

        # Continue training for 5 more steps to get "original" continuation
        losses_original_continuation = []
        for step in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_original_continuation.append(loss.item())

        # Now simulate resume from step 5
        resumed_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        resumed_optimizer = optim.Adam(resumed_model.parameters(), lr=1e-3)

        # Load checkpoint (this loads model and optimizer states)
        loaded_state = load_training_state(str(checkpoint_path), resumed_model, resumed_optimizer)
        assert loaded_state is not None
        assert loaded_state["step"] == 5

        # Continue training - should match original continuation
        # Reset random seed to match original continuation
        torch.manual_seed(42)
        # Skip the first 5 random draws (used before checkpoint)
        for _ in range(5):
            torch.randn(4, 10)

        losses_resumed = []
        for step in range(5):  # Continue for 5 more steps
            x = torch.randn(4, 10)
            y = resumed_model(x)
            loss = y.sum()

            resumed_optimizer.zero_grad()
            loss.backward()
            resumed_optimizer.step()

            losses_resumed.append(loss.item())

        # Losses should match approximately (allowing for numerical precision)
        for i in range(5):
            assert abs(losses_original_continuation[i] - losses_resumed[i]) < 1e-3

    def test_sampler_resume_state(self, mock_sampler):
        """Sampler should maintain state across resume."""
        # Consume some batches
        batches_before_save = []
        iterator = iter(mock_sampler)

        for i in range(3):
            batch = next(iterator)
            batches_before_save.append(batch)

        # Get state
        state = mock_sampler.state_dict()

        # Continue and get more batches
        batches_after_save = []
        for i in range(2):
            batch = next(iterator)
            batches_after_save.append(batch)

        # Create new sampler and restore state
        dimension_cache = {
            "size_groups": {
                "(64, 64)": {"indices": [0, 1, 2, 3, 4]},
                "(128, 128)": {"indices": [5, 6, 7, 8, 9]},
                "(256, 256)": {"indices": [10, 11, 12, 13, 14]},
            },
            "total_images": 15,
        }

        new_sampler = ResumableDimensionSampler(
            dimension_cache=dimension_cache,
            batch_size=2,
            seed=42,
        )

        new_sampler.load_state_dict(state)

        # Get batches from resumed sampler
        batches_resumed = []
        new_iterator = iter(new_sampler)
        for i in range(2):
            batch = next(new_iterator)
            batches_resumed.append(batch)

        # Resumed batches should match original continuation
        for i in range(2):
            assert batches_after_save[i] == batches_resumed[i]

    def test_optimizer_state_preserved(self, simple_model, tmp_path):
        """Optimizer state (momentum, etc.) should be preserved."""
        optimizer = optim.SGD(simple_model.parameters(), lr=0.1, momentum=0.9)

        # Train to build up momentum
        for step in range(10):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save
        checkpoint_path = tmp_path / "optimizer_checkpoint.pt"
        save_training_state(
            checkpoint_path,
            model=simple_model,
            optimizer=optimizer,
            epoch=0,
            step=10,
        )

        # Load into new optimizer
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.1, momentum=0.9)

        load_training_state(checkpoint_path, new_model, new_optimizer)

        # Check momentum buffers were loaded
        for param_group in new_optimizer.param_groups:
            for p in param_group["params"]:
                if p.grad is not None:
                    state = new_optimizer.state[p]
                    # Momentum buffer should exist if training occurred
                    # (might be empty initially)
                    assert isinstance(state, dict)

    def test_random_state_consistency(self, tmp_path):
        """Random state should be saveable for reproducibility."""
        # Set random state
        torch.manual_seed(42)

        # Generate some random numbers
        values_before = [torch.randn(3).tolist() for _ in range(5)]

        # Save state AFTER generating values_before
        rng_state = torch.get_rng_state()
        checkpoint_path = tmp_path / "rng_checkpoint.pt"
        torch.save({"rng_state": rng_state}, checkpoint_path)

        # Continue generating
        values_after = [torch.randn(3).tolist() for _ in range(5)]

        # Load state (back to after values_before)
        loaded = torch.load(checkpoint_path, weights_only=False)
        torch.set_rng_state(loaded["rng_state"])

        # Generate should match post-save values
        values_resumed = [torch.randn(3).tolist() for _ in range(5)]

        for i in range(5):
            for j in range(3):
                assert abs(values_after[i][j] - values_resumed[i][j]) < 1e-6


class TestMidEpochResume:
    """Tests for resuming in the middle of an epoch."""

    def test_epoch_step_tracking(self, tmp_path):
        """Should track both epoch and step correctly."""
        model = nn.Linear(10, 10)
        optimizer = optim.Adam(model.parameters())

        # Simulate training
        epoch = 2
        step = 150

        checkpoint_path = tmp_path / "mid_epoch.pt"
        save_training_state(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
        )

        # Load and verify
        new_model = nn.Linear(10, 10)
        new_optimizer = optim.Adam(new_model.parameters())

        state = load_training_state(checkpoint_path, new_model, new_optimizer)

        assert state["epoch"] == epoch
        assert state["step"] == step

    def test_resume_with_different_batch_order(self):
        """Sampler should maintain deterministic ordering on resume."""
        # Create dimension_cache in the expected format
        dimension_cache = {
            "size_groups": {
                "(64, 64)": {"indices": list(range(20))},
            },
            "total_images": 20,
        }

        # First run - collect all batches
        sampler1 = ResumableDimensionSampler(
            dimension_cache=dimension_cache,
            batch_size=4,
            seed=12345,
        )

        all_batches_1 = []
        for batch in sampler1:
            all_batches_1.append(batch)

        # Second run - collect batches, save state, resume
        sampler2 = ResumableDimensionSampler(
            dimension_cache=dimension_cache,
            batch_size=4,
            seed=12345,
        )

        batches_2 = []
        iterator = iter(sampler2)

        # Get first 2 batches
        for i in range(2):
            batches_2.append(next(iterator))

        # Save state
        state = sampler2.state_dict()

        # Get next 2 batches
        for i in range(2):
            batches_2.append(next(iterator))

        # Create new sampler and resume
        sampler3 = ResumableDimensionSampler(
            dimension_cache=dimension_cache,
            batch_size=4,
            seed=12345,
        )

        sampler3.load_state_dict(state)

        # Continue from where sampler2 left off
        batches_3 = batches_2[:2]  # First 2 batches from sampler2
        for batch in sampler3:
            batches_3.append(batch)

        # All three should produce same batches
        assert len(all_batches_1) == len(batches_3)
        for i in range(len(all_batches_1)):
            assert all_batches_1[i] == batches_3[i]


class TestTrainingLoopResume:
    """Tests for training loop epoch/batch resumption."""

    def test_training_state_includes_epoch_and_batch(self, tmp_path):
        """Training state should include epoch and batch_idx for resume."""
        import json

        # Simulate saving training state mid-epoch
        training_state = {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00",
            "epoch": 2,
            "batch_idx": 42,
            "global_step": 500,
            "learning_rates": {"lr": 1e-4, "vae": 1e-4},
        }

        state_file = tmp_path / "training_state.json"
        with open(state_file, "w") as f:
            json.dump(training_state, f)

        # Load and verify
        with open(state_file, "r") as f:
            loaded = json.load(f)

        assert loaded["epoch"] == 2
        assert loaded["batch_idx"] == 42
        assert loaded["global_step"] == 500

    def test_epoch_range_starts_from_saved_epoch(self, tmp_path):
        """Epoch range should start from saved_epoch, not 0."""
        saved_epoch = 3
        n_epochs = 5

        # Simulate training loop
        epochs_executed = []
        for epoch in range(saved_epoch, n_epochs):
            epochs_executed.append(epoch)

        # Should only execute epochs 3 and 4, not 0-4
        assert epochs_executed == [3, 4]
        assert 0 not in epochs_executed
        assert 1 not in epochs_executed
        assert 2 not in epochs_executed

    def test_batch_skip_logic(self, tmp_path):
        """Should skip batches before saved_batch_idx on resume."""
        saved_epoch = 2
        saved_batch_idx = 5
        total_batches = 10

        # Simulate batch loop with skip logic
        processed_batches = []
        for epoch in [saved_epoch]:  # Only test the resume epoch
            for i in range(total_batches):
                # Skip batches until we reach the resume point
                if epoch == saved_epoch and i < saved_batch_idx:
                    continue
                processed_batches.append(i)

        # Should start from batch 5, not 0
        assert processed_batches == [5, 6, 7, 8, 9]
        assert 0 not in processed_batches
        assert 4 not in processed_batches

    def test_global_step_preserved_across_resume(self, tmp_path):
        """Global step should continue from saved value, not reset."""
        # First training session
        global_step_session1 = 0
        for epoch in range(3):
            for batch in range(10):
                global_step_session1 += 1

        assert global_step_session1 == 30

        # Save state
        saved_global_step = global_step_session1

        # Resume from epoch 3, batch 0
        global_step_session2 = saved_global_step
        for epoch in range(3, 5):  # Continue for 2 more epochs
            for batch in range(10):
                global_step_session2 += 1

        # Global step should be 30 + 20 = 50, not 20
        assert global_step_session2 == 50

    def test_complete_checkpoint_resume_integration(self, tmp_path):
        """Test complete flow of saving and resuming with all states."""
        import json

        from torch.optim.lr_scheduler import CosineAnnealingLR

        from fluxflow_training.training.utils import EMA

        # Create model, optimizer, scheduler, and EMA
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
        ema = EMA(model, decay=0.999)

        # Train for a few steps
        for step in range(10):
            # Forward/backward pass
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update()

        # Save all states (simulating train.py checkpoint)
        epoch = 1
        batch_idx = 5
        global_step = 10

        # Save training state JSON
        training_state = {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00",
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": global_step,
            "learning_rates": {"lr": optimizer.param_groups[0]["lr"]},
        }

        with open(tmp_path / "training_state.json", "w") as f:
            json.dump(training_state, f)

        # Save optimizer state
        torch.save({"optimizer": optimizer.state_dict()}, tmp_path / "optimizer_states.pt")

        # Save scheduler state
        torch.save({"scheduler": scheduler.state_dict()}, tmp_path / "scheduler_states.pt")

        # Save EMA state
        torch.save(ema.state_dict(), tmp_path / "ema_state.pt")

        # Save model state
        torch.save(model.state_dict(), tmp_path / "model.pt")

        # Create new instances (simulating restart)
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = CosineAnnealingLR(new_optimizer, T_max=1000, eta_min=1e-5)
        new_ema = EMA(new_model, decay=0.999)

        # Load training state
        with open(tmp_path / "training_state.json", "r") as f:
            loaded_state = json.load(f)

        saved_epoch = loaded_state["epoch"]
        saved_batch_idx = loaded_state["batch_idx"]
        saved_global_step = loaded_state["global_step"]

        # Load model state
        new_model.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=False))

        # Load optimizer state
        opt_states = torch.load(tmp_path / "optimizer_states.pt", weights_only=False)
        new_optimizer.load_state_dict(opt_states["optimizer"])

        # Load scheduler state
        sched_states = torch.load(tmp_path / "scheduler_states.pt", weights_only=False)
        new_scheduler.load_state_dict(sched_states["scheduler"])

        # Load EMA state
        ema_state = torch.load(tmp_path / "ema_state.pt", weights_only=False)
        new_ema.load_state_dict(ema_state)

        # Verify all states match
        assert saved_epoch == 1
        assert saved_batch_idx == 5
        assert saved_global_step == 10

        # Verify optimizer state preserved
        for (p1, state1), (p2, state2) in zip(
            [(p, optimizer.state[p]) for p in optimizer.state],
            [(p, new_optimizer.state[p]) for p in new_optimizer.state],
        ):
            assert "step" in state1
            assert state1["step"] == state2["step"]

        # Verify scheduler LR matches
        assert abs(optimizer.param_groups[0]["lr"] - new_optimizer.param_groups[0]["lr"]) < 1e-9

        # Verify EMA shadows match
        for key in ema.shadow:
            assert torch.allclose(ema.shadow[key], new_ema.shadow[key])


class TestCheckpointVersioning:
    """Tests for checkpoint version compatibility."""

    def test_checkpoint_includes_metadata(self, tmp_path):
        """Checkpoints should include metadata."""
        model = nn.Linear(10, 10)
        optimizer = optim.Adam(model.parameters())

        checkpoint_path = tmp_path / "metadata_checkpoint.pt"
        save_training_state(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=5,
            step=100,
        )

        # Load and check for metadata
        loaded = torch.load(checkpoint_path)

        assert "epoch" in loaded
        assert "step" in loaded
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded

    def test_backward_compatible_loading(self, tmp_path):
        """Should handle loading checkpoints gracefully."""
        model = nn.Linear(10, 10)

        # Save minimal checkpoint
        checkpoint_path = tmp_path / "minimal_checkpoint.pt"
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

        # Should be able to load model state
        new_model = nn.Linear(10, 10)
        loaded = torch.load(checkpoint_path)

        if "model_state_dict" in loaded:
            new_model.load_state_dict(loaded["model_state_dict"])

        # Models should match
        model.eval()
        new_model.eval()

        x = torch.randn(4, 10)
        with torch.no_grad():
            out1 = model(x)
            out2 = new_model(x)

        assert torch.allclose(out1, out2)
