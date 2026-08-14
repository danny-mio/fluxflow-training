"""Unit tests for shape-keyed dimension caching (aspect_ratio_bucketing feature).

Mirrors tests/unit/test_datasets.py's TestBuildDimensionCache /
TestGetOrBuildDimensionCache style, and tests/unit/test_datasets_sampler.py's
"same shape batched together" invariant style, applied to the new
build_shape_dimension_cache / get_or_build_shape_dimension_cache functions.
"""

import os
from unittest.mock import patch

import pytest
from PIL import Image

from fluxflow_training.data.datasets import (
    ResumableDimensionSampler,
    TextImageDataset,
    build_shape_dimension_cache,
    get_or_build_dimension_cache,
    get_or_build_shape_dimension_cache,
)

# (342, 318) and (352, 305) have different native aspect ratios (1.075 vs
# 1.154) but both resolve to the identical (352, 320) target shape under
# resize_preserving_aspect_min_distortion(image, min(h, w), ceil(max(h, w)/16)*16)
# -- i.e. the exact convention build_shape_dimension_cache uses. Verified
# empirically against transforms.resize_preserving_aspect_min_distortion.
SAME_BUCKET_SIZE_A = (342, 318)
SAME_BUCKET_SIZE_B = (352, 305)
SAME_BUCKET_TARGET = "(352, 320)"
DIFFERENT_BUCKET_SIZE = (200, 200)


@pytest.fixture
def shape_bucket_dataset(temp_dir):
    """Real images: two distinct native aspect ratios sharing one target shape,
    plus one image resolving to a clearly different target shape."""
    data_dir = temp_dir / "images"
    data_dir.mkdir()

    sizes = {
        "same_a.jpg": SAME_BUCKET_SIZE_A,
        "same_b.jpg": SAME_BUCKET_SIZE_B,
        "different.jpg": DIFFERENT_BUCKET_SIZE,
    }
    captions_file = temp_dir / "captions.txt"
    with open(captions_file, "w") as f:
        for name, (w, h) in sizes.items():
            Image.new("RGB", (w, h), color=(10, 20, 30)).save(data_dir / name)
            f.write(f"{name}\tcaption for {name}\n")

    return {
        "data_dir": str(data_dir),
        "captions_file": str(captions_file),
        "sizes": sizes,
    }


def _build_dataset(mock_tokenizer, shape_bucket_dataset):
    return TextImageDataset(
        data_path=shape_bucket_dataset["data_dir"],
        captions_file=shape_bucket_dataset["captions_file"],
        tokenizer_name="distilbert-base-uncased",
    )


class TestBuildShapeDimensionCache:
    """Tests for build_shape_dimension_cache function."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_cache_structure(self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset):
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)

        cache = build_shape_dimension_cache(dataset, show_progress=False)

        assert "dataset_path" in cache
        assert "total_images" in cache
        assert "size_groups" in cache
        assert "statistics" in cache
        assert cache["total_images"] == len(dataset)

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_all_images_indexed(self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset):
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)

        cache = build_shape_dimension_cache(dataset, show_progress=False)

        all_indices = []
        for group_data in cache["size_groups"].values():
            all_indices.extend(group_data["indices"])

        assert len(all_indices) == len(dataset)
        assert set(all_indices) == set(range(len(dataset)))

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_differing_aspect_ratios_share_bucket_when_target_shape_matches(
        self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset
    ):
        """Images with differing native aspect ratio but identical post-resize
        target shape must land in the same bucket."""
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)

        cache = build_shape_dimension_cache(dataset, show_progress=False)

        data_dir = shape_bucket_dataset["data_dir"]
        idx_a = dataset.image_paths.index(os.path.join(data_dir, "same_a.jpg"))
        idx_b = dataset.image_paths.index(os.path.join(data_dir, "same_b.jpg"))
        idx_c = dataset.image_paths.index(os.path.join(data_dir, "different.jpg"))

        group_of = {}
        for size, info in cache["size_groups"].items():
            for i in info["indices"]:
                group_of[i] = size

        assert group_of[idx_a] == SAME_BUCKET_TARGET
        assert group_of[idx_b] == SAME_BUCKET_TARGET
        assert group_of[idx_a] == group_of[idx_b]
        assert group_of[idx_c] != group_of[idx_a]

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_statistics_correctness(
        self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset
    ):
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)

        cache = build_shape_dimension_cache(dataset, show_progress=False)

        stats = cache["statistics"]
        groups = cache["size_groups"]
        group_sizes = [g["count"] for g in groups.values()]
        assert stats["num_groups"] == len(groups)
        assert stats["min_group_size"] == min(group_sizes)
        assert stats["max_group_size"] == max(group_sizes)
        assert stats["avg_group_size"] == sum(group_sizes) // len(group_sizes)


class TestGetOrBuildShapeDimensionCache:
    """Tests for get_or_build_shape_dimension_cache function."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_creates_distinguishable_cache_file(
        self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset, temp_dir
    ):
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)
        cache_dir = temp_dir / "cache"

        cache = get_or_build_shape_dimension_cache(dataset, str(cache_dir), rebuild=False)

        assert cache["total_images"] == len(dataset)
        bucket_files = list(cache_dir.glob("*.bucket_dimensions.json"))
        assert len(bucket_files) == 1

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_does_not_collide_with_plain_dimension_cache(
        self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset, temp_dir
    ):
        """The shape cache and the native-size cache must coexist for the same
        dataset/cache_dir without overwriting each other."""
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)
        cache_dir = temp_dir / "cache"

        get_or_build_dimension_cache(dataset, str(cache_dir), multiple=32)
        get_or_build_shape_dimension_cache(dataset, str(cache_dir))

        plain_files = list(cache_dir.glob("*.dimensions.json"))
        bucket_files = list(cache_dir.glob("*.bucket_dimensions.json"))

        assert len(plain_files) == 1
        assert len(bucket_files) == 1
        assert plain_files[0] != bucket_files[0]

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_second_call_loads_cache(
        self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset, temp_dir
    ):
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)
        cache_dir = temp_dir / "cache"

        cache1 = get_or_build_shape_dimension_cache(dataset, str(cache_dir))
        cache2 = get_or_build_shape_dimension_cache(dataset, str(cache_dir))

        assert cache1["statistics"] == cache2["statistics"]

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_rebuild_detects_dataset_change(
        self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset, temp_dir
    ):
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)
        cache_dir = temp_dir / "cache"

        cache1 = get_or_build_shape_dimension_cache(dataset, str(cache_dir))

        dataset.captions.append("NEW CAPTION")
        dataset.image_paths.append(dataset.image_paths[0])

        cache2 = get_or_build_shape_dimension_cache(dataset, str(cache_dir))

        assert cache1["total_images"] != cache2["total_images"]
        assert cache2["total_images"] == len(dataset)


class TestResumableDimensionSamplerWithShapeCache:
    """The existing sampler must work unmodified against a shape-keyed cache."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_batches_are_shape_uniform(
        self, mock_from_pretrained, mock_tokenizer, shape_bucket_dataset
    ):
        """A full batch drawn from the 'same shape' group must contain only
        indices whose target shape matches (batch_size == 2 == group size)."""
        mock_from_pretrained.return_value = mock_tokenizer
        dataset = _build_dataset(mock_tokenizer, shape_bucket_dataset)
        cache = build_shape_dimension_cache(dataset, show_progress=False)

        data_dir = shape_bucket_dataset["data_dir"]
        idx_a = dataset.image_paths.index(os.path.join(data_dir, "same_a.jpg"))
        idx_b = dataset.image_paths.index(os.path.join(data_dir, "same_b.jpg"))

        sampler = ResumableDimensionSampler(dimension_cache=cache, batch_size=2, seed=42)

        # The "same" group has exactly 2 images (== batch_size), so it must
        # form its own full batch, distinct from the lone "different" image
        # (which alone can't fill a batch and is dropped as an incomplete
        # remainder).
        matching_batches = [
            batch for batch in sampler.epoch_batches if set(batch) == {idx_a, idx_b}
        ]
        assert len(matching_batches) == 1
