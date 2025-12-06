"""Unit tests for ResumableDimensionSampler (src/data/datasets.py)."""


from fluxflow_training.data.datasets import ResumableDimensionSampler


class TestResumableDimensionSampler:
    """Tests for ResumableDimensionSampler class."""

    def test_initialization(self, mock_dimension_cache):
        """Test sampler initializes correctly."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        assert sampler.batch_size == 8
        assert sampler.seed == 42
        assert sampler.position == 0
        assert sampler.current_epoch == 0

    def test_creates_batches(self, mock_dimension_cache):
        """Test that sampler creates batches from cache."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
        )

        assert len(sampler.epoch_batches) > 0
        assert len(sampler) > 0

    def test_batch_size_consistency(self, mock_dimension_cache):
        """All batches should have exactly batch_size."""
        batch_size = 8
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=batch_size,
        )

        for batch in sampler.epoch_batches:
            assert len(batch) == batch_size

    def test_deterministic_from_seed(self, mock_dimension_cache):
        """Same seed should produce identical batch ordering."""
        sampler1 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        sampler2 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        # Should have identical batches
        assert len(sampler1.epoch_batches) == len(sampler2.epoch_batches)
        for b1, b2 in zip(sampler1.epoch_batches, sampler2.epoch_batches):
            assert b1 == b2

    def test_different_seeds_different_ordering(self, mock_dimension_cache):
        """Different seeds should produce different orderings."""
        sampler1 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        sampler2 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=999,
        )

        # Should have different batch orderings
        assert sampler1.epoch_batches != sampler2.epoch_batches

    def test_iteration(self, mock_dimension_cache):
        """Test __iter__ yields batches."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
        )

        batches = list(sampler)
        assert len(batches) > 0
        assert len(batches) == len(sampler)

    def test_iteration_from_position(self, mock_dimension_cache):
        """Iteration should start from current position."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
        )

        # Manually advance position
        total_batches = len(sampler.epoch_batches)
        start_position = total_batches // 2
        sampler.position = start_position

        # Iterate - should only get remaining batches
        remaining_batches = list(sampler)
        assert len(remaining_batches) == total_batches - start_position

    def test_state_dict(self, mock_dimension_cache):
        """Test state_dict captures current state."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        # Advance position
        sampler.position = 5
        sampler.current_epoch = 2

        state = sampler.state_dict()

        assert state["seed"] == 42
        assert state["position"] == 5
        assert state["current_epoch"] == 2
        assert state["batch_size"] == 8

    def test_state_dict_no_batches(self, mock_dimension_cache):
        """state_dict should not include epoch_batches (too large)."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
        )

        state = sampler.state_dict()

        # Should not contain epoch_batches
        assert "epoch_batches" not in state
        assert "epoch_indices" not in state

    def test_load_state_dict(self, mock_dimension_cache):
        """Test loading state from state_dict."""
        # Create sampler and advance it
        sampler1 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )
        sampler1.position = 5
        sampler1.current_epoch = 2

        # Save state
        state = sampler1.state_dict()

        # Create new sampler from saved state
        sampler2 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            resume_state=state,
        )

        # Should restore position and seed
        assert sampler2.position == 5
        assert sampler2.current_epoch == 2
        assert sampler2.seed == 42
        assert sampler2.batch_size == 8

    def test_resume_produces_same_batches(self, mock_dimension_cache):
        """Resumed sampler should have identical batch ordering."""
        sampler1 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        # Save state at beginning
        state = sampler1.state_dict()

        # Create resumed sampler
        sampler2 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            resume_state=state,
        )

        # Batches should be identical
        assert sampler1.epoch_batches == sampler2.epoch_batches

    def test_resume_from_mid_epoch(self, mock_dimension_cache):
        """Test resuming from middle of epoch."""
        sampler1 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        # Iterate halfway through
        total_batches = len(sampler1.epoch_batches)
        halfway = total_batches // 2

        consumed_batches = []
        for i, batch in enumerate(sampler1):
            consumed_batches.append(batch)
            if i >= halfway - 1:
                break

        # Save state
        state = sampler1.state_dict()

        # Create resumed sampler
        sampler2 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            resume_state=state,
        )

        # Get remaining batches from resumed sampler
        remaining_batches = list(sampler2)

        # Combined batches should equal original full epoch
        combined = consumed_batches + remaining_batches
        assert len(combined) == total_batches

    def test_set_epoch(self, mock_dimension_cache):
        """Test set_epoch updates epoch and re-initializes."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        original_batches = sampler.epoch_batches.copy()
        original_seed = sampler.seed

        # Set new epoch
        sampler.set_epoch(1)

        # Epoch should be updated
        assert sampler.current_epoch == 1

        # Seed should be modified
        assert sampler.seed != original_seed

        # Batches should be re-shuffled (different from epoch 0)
        assert sampler.epoch_batches != original_batches

        # Position should reset
        assert sampler.position == 0

    def test_get_progress_info(self, mock_dimension_cache):
        """Test get_progress_info returns correct progress."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
        )

        # At start
        info = sampler.get_progress_info()
        assert info["batch"] == 0
        assert info["progress_pct"] == 0.0

        # Advance position
        total_batches = len(sampler.epoch_batches)
        sampler.position = total_batches // 2

        info = sampler.get_progress_info()
        assert info["batch"] == total_batches // 2
        assert 40 < info["progress_pct"] < 60  # ~50%
        assert info["total_batches"] == total_batches

    def test_samples_trained_calculation(self, mock_dimension_cache):
        """Test samples_trained is calculated correctly."""
        batch_size = 8
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=batch_size,
        )

        sampler.position = 10

        info = sampler.get_progress_info()
        assert info["samples_trained"] == 10 * batch_size

    def test_no_duplicate_samples_per_epoch(self, mock_dimension_cache):
        """Each sample should appear exactly once per epoch."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
        )

        # Collect all indices
        all_indices = []
        for batch in sampler.epoch_batches:
            all_indices.extend(batch)

        # Check no duplicates
        assert len(all_indices) == len(set(all_indices))

    def test_size_group_batching(self, mock_dimension_cache):
        """Images from same size group should be batched together."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
        )

        # Get size groups from cache
        size_groups = {
            eval(size): set(info["indices"])
            for size, info in mock_dimension_cache["size_groups"].items()
        }

        # Check each batch
        for batch in sampler.epoch_batches:
            # Find which size groups these indices belong to
            groups_in_batch = set()
            for idx in batch:
                for size, indices in size_groups.items():
                    if idx in indices:
                        groups_in_batch.add(size)
                        break

            # In most cases, a batch should come from a single size group
            # (Remainder pool may mix groups, so we allow up to 2)
            assert len(groups_in_batch) <= 2


class TestResumableSamplerIntegration:
    """Integration tests for ResumableDimensionSampler."""

    def test_multi_epoch_training(self, mock_dimension_cache):
        """Test sampler across multiple epochs."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        # Epoch 0 batches
        epoch0_batches = sampler.epoch_batches.copy()

        # Set epoch 1
        sampler.set_epoch(1)
        epoch1_batches = sampler.epoch_batches.copy()

        # Set epoch 2
        sampler.set_epoch(2)
        epoch2_batches = sampler.epoch_batches.copy()

        # Each epoch should have different ordering
        assert epoch0_batches != epoch1_batches
        assert epoch1_batches != epoch2_batches
        assert epoch0_batches != epoch2_batches

    def test_checkpoint_and_resume_workflow(self, mock_dimension_cache):
        """Test realistic checkpoint/resume workflow."""
        # Simulate training for a few batches
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        # Train for 5 batches
        trained_batches = []
        for i, batch in enumerate(sampler):
            trained_batches.append(batch)
            if i >= 4:  # 5 batches (0-4)
                break

        # Checkpoint
        checkpoint_state = sampler.state_dict()

        # Simulate crash/restart - create new sampler from checkpoint
        resumed_sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            resume_state=checkpoint_state,
        )

        # Continue training
        remaining_batches = list(resumed_sampler)

        # Verify no overlap
        trained_indices = set()
        for batch in trained_batches:
            trained_indices.update(batch)

        remaining_indices = set()
        for batch in remaining_batches:
            remaining_indices.update(batch)

        # No duplicates
        assert len(trained_indices & remaining_indices) == 0

    def test_deterministic_resume_multiple_times(self, mock_dimension_cache):
        """Multiple resumes from same checkpoint should be identical."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        # Advance to position 5
        sampler.position = 5
        state = sampler.state_dict()

        # Resume twice
        resumed1 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            resume_state=state,
        )

        resumed2 = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            resume_state=state,
        )

        # Both should produce identical batches
        batches1 = list(resumed1)
        batches2 = list(resumed2)

        assert batches1 == batches2

    def test_epoch_boundary_resume(self, mock_dimension_cache):
        """Test resuming at epoch boundary."""
        sampler = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
        )

        # Consume entire epoch
        list(sampler)

        # Position should be at end
        assert sampler.position == len(sampler.epoch_batches)

        # Save state
        state = sampler.state_dict()

        # Resume - should start new epoch
        resumed = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            resume_state=state,
        )

        # Set new epoch
        resumed.set_epoch(resumed.current_epoch + 1)

        # Should have full epoch of batches again
        new_epoch_batches = list(resumed)
        assert len(new_epoch_batches) == len(sampler.epoch_batches)
