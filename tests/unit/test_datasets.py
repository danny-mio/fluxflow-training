"""Unit tests for dataset classes (src/data/datasets.py)."""

import os
from unittest.mock import patch

import pytest
import torch

from fluxflow_training.data.datasets import (
    GroupedBatchSampler,
    StreamingGroupedBatchSampler,
    TextImageDataset,
    build_dimension_cache,
    get_or_build_dimension_cache,
)


class TestTextImageDataset:
    """Tests for TextImageDataset class."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_initialization_train_mode(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset
    ):
        """Test dataset initialization in training mode."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        assert len(dataset) == mock_image_dataset["num_images"]
        assert len(dataset.captions) == mock_image_dataset["num_images"]
        assert len(dataset.image_paths) == mock_image_dataset["num_images"]

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_initialization_requires_captions_file(
        self, mock_from_pretrained, mock_tokenizer, temp_dir
    ):
        """Should raise error if captions_file not provided in train mode."""
        mock_from_pretrained.return_value = mock_tokenizer

        with pytest.raises(ValueError, match="captions_file is required"):
            TextImageDataset(
                data_path=str(temp_dir),
                captions_file=None,
                generate_mode=False,
            )

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_caption_uppercasing(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Captions should be converted to uppercase."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        # All captions should be uppercase
        for caption in dataset.captions:
            assert caption.isupper()

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_getitem_train_mode(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Test __getitem__ returns (input_ids, image_path) in train mode."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        input_ids, image_path = dataset[0]

        # Check input_ids shape
        assert input_ids.shape == (128,)  # max_length=128
        assert input_ids.dtype == torch.long

        # Check image_path is string
        assert isinstance(image_path, str)
        assert os.path.exists(image_path)

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_tokenizer_padding_and_truncation(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset
    ):
        """Test that tokenizer applies padding and truncation."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        input_ids, _ = dataset[0]

        # Should be exactly max_length
        assert input_ids.shape[0] == 128

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_get_image_size_class(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Test get_image_size_class returns rounded dimensions."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        size_class = dataset.get_image_size_class(0, multiple=16)

        # Size should be rounded to multiple of 16
        h, w = size_class
        assert h % 16 == 0
        assert w % 16 == 0

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_get_image_size_class_different_multiples(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset
    ):
        """Test size class with different rounding multiples."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        size_32 = dataset.get_image_size_class(0, multiple=32)
        size_64 = dataset.get_image_size_class(0, multiple=64)

        # Check both are properly rounded
        assert size_32[0] % 32 == 0 and size_32[1] % 32 == 0
        assert size_64[0] % 64 == 0 and size_64[1] % 64 == 0

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_generate_mode(self, mock_from_pretrained, mock_tokenizer, temp_dir):
        """Test dataset in generation mode (reading .txt files)."""
        mock_from_pretrained.return_value = mock_tokenizer

        # Create text prompt files
        prompts = ["A beautiful sunset", "A mountain landscape", "Ocean waves"]
        for i, prompt in enumerate(prompts):
            prompt_file = temp_dir / f"prompt_{i}.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)

        dataset = TextImageDataset(
            data_path=str(temp_dir),
            tokenizer_name="distilbert-base-uncased",
            generate_mode=True,
        )

        assert len(dataset) == len(prompts)

        # Test __getitem__ in generate mode
        file_name, input_ids = dataset[0]

        assert isinstance(file_name, str)
        assert input_ids.shape == (128,)
        assert input_ids.dtype == torch.long


class TestGroupedBatchSampler:
    """Tests for GroupedBatchSampler class."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_initialization(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Test sampler initialization."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        sampler = GroupedBatchSampler(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            multiple=32,
        )

        assert sampler.batch_size == 2
        assert len(sampler.grouped_indices) > 0

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_batch_size_consistency(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """All batches should have exactly batch_size samples."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        batch_size = 3
        sampler = GroupedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        for batch in sampler:
            assert len(batch) == batch_size

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_no_duplicate_indices(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Each sample should appear at most once per epoch."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        sampler = GroupedBatchSampler(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
        )

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        # No duplicates
        assert len(all_indices) == len(set(all_indices))

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_shuffle_changes_order(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Shuffle should change batch ordering."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        # Create two samplers with different seeds (via different init)
        sampler1 = GroupedBatchSampler(dataset, batch_size=2, shuffle=True)
        sampler2 = GroupedBatchSampler(dataset, batch_size=2, shuffle=True)

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # With high probability, shuffling should produce different orders
        # (May rarely fail due to random chance, but very unlikely)
        assert batches1 != batches2

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_grouping_by_size(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Images in same batch should have same size class."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        multiple = 32
        sampler = GroupedBatchSampler(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            multiple=multiple,
        )

        for batch in sampler:
            # Get size classes for all images in batch
            size_classes = [dataset.get_image_size_class(idx, multiple=multiple) for idx in batch]

            # All should be the same
            assert len(set(size_classes)) == 1


class TestStreamingGroupedBatchSampler:
    """Tests for StreamingGroupedBatchSampler class."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_initialization(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Test sampler initialization."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        sampler = StreamingGroupedBatchSampler(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
        )

        assert sampler.batch_size == 2

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_yields_batches_immediately(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset
    ):
        """Streaming sampler should yield batches as buckets fill."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        sampler = StreamingGroupedBatchSampler(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
        )

        batches = list(sampler)
        assert len(batches) > 0

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_batch_size_consistency(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """All yielded batches should have exactly batch_size."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        batch_size = 2
        sampler = StreamingGroupedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        for batch in sampler:
            assert len(batch) == batch_size

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_estimated_length(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Test __len__ provides reasonable estimate."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        batch_size = 2
        sampler = StreamingGroupedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
        )

        estimated_len = len(sampler)
        actual_len = len(list(sampler))

        # Estimate should be close (within 50% due to dropped incomplete batches)
        assert abs(estimated_len - actual_len) <= estimated_len * 0.5


class TestBuildDimensionCache:
    """Tests for build_dimension_cache function."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_cache_structure(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Test cache has correct structure."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        cache = build_dimension_cache(dataset, multiple=32, show_progress=False)

        # Check required keys
        assert "dataset_path" in cache
        assert "total_images" in cache
        assert "multiple" in cache
        assert "size_groups" in cache
        assert "statistics" in cache

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_total_images_correct(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Cache should report correct total image count."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        cache = build_dimension_cache(dataset, multiple=32, show_progress=False)

        assert cache["total_images"] == len(dataset)

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_all_images_indexed(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """All images should appear in exactly one size group."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        cache = build_dimension_cache(dataset, multiple=32, show_progress=False)

        # Collect all indices from all groups
        all_indices = []
        for group_data in cache["size_groups"].values():
            all_indices.extend(group_data["indices"])

        # Should have all indices exactly once
        assert len(all_indices) == len(dataset)
        assert set(all_indices) == set(range(len(dataset)))

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_statistics_correctness(self, mock_from_pretrained, mock_tokenizer, mock_image_dataset):
        """Test statistics are computed correctly."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        cache = build_dimension_cache(dataset, multiple=32, show_progress=False)

        stats = cache["statistics"]
        groups = cache["size_groups"]

        # Verify statistics
        group_sizes = [g["count"] for g in groups.values()]
        assert stats["num_groups"] == len(groups)
        assert stats["min_group_size"] == min(group_sizes)
        assert stats["max_group_size"] == max(group_sizes)
        assert stats["avg_group_size"] == sum(group_sizes) // len(group_sizes)


class TestGetOrBuildDimensionCache:
    """Tests for get_or_build_dimension_cache function."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_creates_cache_if_not_exists(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset, temp_dir
    ):
        """Should create cache file if it doesn't exist."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        cache_dir = temp_dir / "cache"
        cache = get_or_build_dimension_cache(dataset, str(cache_dir), multiple=32, rebuild=False)

        # Cache should be returned
        assert cache is not None
        assert cache["total_images"] == len(dataset)

        # Cache file should exist
        cache_files = list(cache_dir.glob("*.dimensions.json"))
        assert len(cache_files) == 1

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_loads_existing_cache(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset, temp_dir
    ):
        """Should load existing cache instead of rebuilding."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        cache_dir = temp_dir / "cache"

        # Build cache first time
        cache1 = get_or_build_dimension_cache(dataset, str(cache_dir), multiple=32)

        # Load cache second time (should be faster, use existing)
        cache2 = get_or_build_dimension_cache(dataset, str(cache_dir), multiple=32)

        # Should return same data
        assert cache1["total_images"] == cache2["total_images"]
        assert cache1["statistics"] == cache2["statistics"]

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_rebuild_flag_forces_rebuild(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset, temp_dir
    ):
        """rebuild=True should force cache rebuild."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        cache_dir = temp_dir / "cache"

        # Build cache
        get_or_build_dimension_cache(dataset, str(cache_dir), multiple=32)

        # Get cache file
        cache_files = list(cache_dir.glob("*.dimensions.json"))
        original_mtime = os.path.getmtime(cache_files[0])

        # Wait a tiny bit
        import time

        time.sleep(0.1)

        # Rebuild
        get_or_build_dimension_cache(dataset, str(cache_dir), multiple=32, rebuild=True)

        # File should be newer
        new_mtime = os.path.getmtime(cache_files[0])
        assert new_mtime > original_mtime

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_cache_invalidation_on_dataset_change(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset, temp_dir
    ):
        """Cache should rebuild if dataset size changes."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        cache_dir = temp_dir / "cache"

        # Build cache
        cache1 = get_or_build_dimension_cache(dataset, str(cache_dir))

        # Modify dataset (add a caption)
        dataset.captions.append("NEW CAPTION")
        dataset.image_paths.append(dataset.image_paths[0])  # Reuse first image

        # Get cache again - should detect mismatch and rebuild
        cache2 = get_or_build_dimension_cache(dataset, str(cache_dir))

        assert cache2["total_images"] == len(dataset)
        assert cache2["total_images"] != cache1["total_images"]


class TestDatasetIntegration:
    """Integration tests combining dataset components."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_dataset_with_grouped_sampler(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset
    ):
        """Test dataset and sampler work together."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        sampler = GroupedBatchSampler(dataset, batch_size=2)

        # Iterate through batches
        batch_count = 0
        for batch in sampler:
            # Get items from dataset
            for idx in batch:
                input_ids, image_path = dataset[idx]
                assert input_ids.shape == (128,)
                assert os.path.exists(image_path)
            batch_count += 1

        assert batch_count > 0

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_dimension_cache_with_sampler(
        self, mock_from_pretrained, mock_tokenizer, mock_image_dataset, temp_dir
    ):
        """Test dimension cache improves sampler performance."""
        mock_from_pretrained.return_value = mock_tokenizer

        dataset = TextImageDataset(
            data_path=mock_image_dataset["data_dir"],
            captions_file=mock_image_dataset["captions_file"],
            tokenizer_name="distilbert-base-uncased",
        )

        # Build cache
        cache = get_or_build_dimension_cache(dataset, str(temp_dir / "cache"), multiple=32)

        # Cache should have grouped images
        assert cache["statistics"]["num_groups"] >= 1

        # Use cache data for sampling (simulated)
        assert cache["total_images"] == len(dataset)
