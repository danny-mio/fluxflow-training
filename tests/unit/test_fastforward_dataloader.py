"""Unit tests for FastForwardDataLoader."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from fluxflow_training.training.pipeline_orchestrator import FastForwardDataLoader


class TestFastForwardDataLoader:
    """Test FastForwardDataLoader wrapper for resume optimization."""

    def test_yields_none_for_skip_zone(self):
        """Test that wrapper yields None for skipped batches."""
        # Create dummy dataset
        dataset = TensorDataset(
            torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,))  # images  # labels
        )
        dataloader = DataLoader(dataset, batch_size=4)

        # Wrap with skip_batches=5
        wrapper = FastForwardDataLoader(dataloader, skip_batches=5)

        results = []
        for batch_idx, (imgs, labels) in enumerate(wrapper):
            results.append((batch_idx, imgs is None, labels is None))
            if batch_idx >= 10:  # Only test first 11 batches
                break

        # First 5 batches should be None
        assert results[0] == (0, True, True)
        assert results[1] == (1, True, True)
        assert results[2] == (2, True, True)
        assert results[3] == (3, True, True)
        assert results[4] == (4, True, True)

        # Batches 5+ should have real data
        assert results[5][0] == 5
        assert results[5][1] is False  # imgs is not None
        assert results[5][2] is False  # labels is not None

    def test_consumes_real_data_after_skip(self):
        """Test that wrapper yields real data from dataloader after skip zone."""
        dataset = TensorDataset(
            torch.arange(20).reshape(20, 1),  # Simple sequential data
            torch.arange(20),  # Labels match data
        )
        dataloader = DataLoader(dataset, batch_size=2)  # 10 batches total

        wrapper = FastForwardDataLoader(dataloader, skip_batches=3)

        real_batches = []
        for batch_idx, (data, labels) in enumerate(wrapper):
            if data is not None:
                real_batches.append((batch_idx, data, labels))

        # Should have 10 real batches (dataloader not consumed during skip)
        assert len(real_batches) == 10

        # First real batch should be at batch_idx=3
        assert real_batches[0][0] == 3

        # Verify data is from the beginning of dataloader (not skipped from underlying data)
        first_data = real_batches[0][1]
        assert torch.equal(first_data, torch.tensor([[0], [1]]))

    def test_batch_idx_increments_correctly(self):
        """Test that batch_idx from enumerate increments for all batches including None."""
        dataset = TensorDataset(torch.randn(50, 1), torch.randint(0, 2, (50,)))
        dataloader = DataLoader(dataset, batch_size=5)  # 10 batches

        wrapper = FastForwardDataLoader(dataloader, skip_batches=4)

        batch_indices = []
        for batch_idx, (data, labels) in enumerate(wrapper):
            batch_indices.append(batch_idx)

        # Should have batch_idx from 0 to 13 (4 None + 10 real = 14 batches total)
        assert batch_indices == list(range(14))

    def test_skip_exceeds_dataloader_length(self, caplog):
        """Test warning when skip_batches >= dataloader length."""
        dataset = TensorDataset(torch.randn(10, 1), torch.randint(0, 2, (10,)))
        dataloader = DataLoader(dataset, batch_size=5)  # 2 batches

        # Skip more batches than available
        wrapper = FastForwardDataLoader(dataloader, skip_batches=5)

        # Check warning was logged
        assert "may yield no training data" in caplog.text.lower()

        # Verify behavior: yields 5 Nones, then fetches 2 real batches
        results = []
        for batch_idx, (data, labels) in enumerate(wrapper):
            results.append((batch_idx, data is None))

        # Should yield 5 None batches + 2 real batches = 7 total
        assert len(results) == 7
        assert results[0][1] is True  # Batch 0 is None
        assert results[4][1] is True  # Batch 4 is None
        assert results[5][1] is False  # Batch 5 is real data
        assert results[6][1] is False  # Batch 6 is real data

    def test_zero_skip_batches(self):
        """Test that skip_batches=0 works correctly (no skipping)."""
        dataset = TensorDataset(torch.randn(20, 1), torch.randint(0, 2, (20,)))
        dataloader = DataLoader(dataset, batch_size=4)

        wrapper = FastForwardDataLoader(dataloader, skip_batches=0)

        count = 0
        for batch_idx, (data, labels) in enumerate(wrapper):
            assert data is not None, f"Batch {batch_idx} should not be None with skip_batches=0"
            count += 1

        assert count == 5  # 20 samples / 4 batch_size = 5 batches

    def test_preserves_dataloader_length(self):
        """Test that __len__ returns original dataloader length."""
        dataset = TensorDataset(torch.randn(100, 1), torch.randint(0, 2, (100,)))
        dataloader = DataLoader(dataset, batch_size=10)  # 10 batches

        wrapper = FastForwardDataLoader(dataloader, skip_batches=5)

        assert len(wrapper) == 10  # Should preserve original length

    def test_iter_resets_batch_count(self):
        """Test that __iter__ resets batch_count for multiple iterations."""
        dataset = TensorDataset(torch.randn(20, 1), torch.randint(0, 2, (20,)))
        dataloader = DataLoader(dataset, batch_size=5)

        wrapper = FastForwardDataLoader(dataloader, skip_batches=2)

        # First iteration
        first_none_count = sum(1 for _, (data, _) in enumerate(wrapper) if data is None)
        assert first_none_count == 2

        # Second iteration should also skip 2
        second_none_count = sum(1 for _, (data, _) in enumerate(wrapper) if data is None)
        assert second_none_count == 2


class TestFastForwardDataLoaderEdgeCases:
    """Test edge cases and error conditions."""

    def test_with_iterable_dataset_no_len(self):
        """Test behavior with IterableDataset (no __len__)."""

        class DummyIterableDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for i in range(10):
                    yield (torch.tensor([i]), torch.tensor(i))

        dataset = DummyIterableDataset()
        dataloader = DataLoader(dataset, batch_size=None)

        # Should not crash - hasattr check prevents calling len() on IterableDataset
        # Note: DataLoader with IterableDataset may or may not have __len__ depending on PyTorch version
        wrapper = FastForwardDataLoader(dataloader, skip_batches=3)

        # Verify skip still works
        results = []
        for batch_idx, (data, labels) in enumerate(wrapper):
            results.append((batch_idx, data is None))
            if batch_idx >= 5:
                break

        assert results[0][1] is True  # First 3 are None
        assert results[1][1] is True
        assert results[2][1] is True
        assert results[3][1] is False  # After skip zone

    def test_single_batch_dataloader(self):
        """Test with dataloader that has only 1 batch."""
        dataset = TensorDataset(torch.randn(3, 1), torch.randint(0, 2, (3,)))
        dataloader = DataLoader(dataset, batch_size=5)  # 1 batch

        wrapper = FastForwardDataLoader(dataloader, skip_batches=0)

        count = sum(1 for _ in wrapper)
        assert count == 1
