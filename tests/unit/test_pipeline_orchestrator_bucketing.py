"""Unit tests for aspect_ratio_bucketing wiring in
TrainingPipelineOrchestrator._create_dataloader_for_dataset.

Verifies the orchestrator picks the shape-keyed cache builder
(get_or_build_shape_dimension_cache) instead of the native-size one
(get_or_build_dimension_cache) exactly when DatasetConfig.aspect_ratio_bucketing
is True, and that default-off behavior is unchanged.
"""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from fluxflow_training.data.datasets import ResumableDimensionSampler
from fluxflow_training.training.pipeline_config import DatasetConfig
from fluxflow_training.training.pipeline_orchestrator import TrainingPipelineOrchestrator


@pytest.fixture
def local_image_dataset(temp_dir):
    """Small real local dataset (4 images + tab-separated captions)."""
    data_dir = temp_dir / "images"
    data_dir.mkdir()
    captions_file = temp_dir / "captions.txt"
    with open(captions_file, "w") as f:
        for i in range(4):
            name = f"img_{i}.jpg"
            Image.new("RGB", (256 + i * 4, 256)).save(data_dir / name)
            f.write(f"{name}\tcaption {i}\n")
    return {"data_dir": str(data_dir), "captions_file": str(captions_file)}


def _make_orchestrator():
    orchestrator = TrainingPipelineOrchestrator()
    orchestrator.accelerator = Mock()
    orchestrator.accelerator.prepare = lambda dl: dl
    return orchestrator


def _make_args(output_path):
    return Mock(
        tokenizer_name="distilbert-base-uncased",
        channels=3,
        img_size=512,
        reduced_min_sizes=None,
        batch_size=2,
        workers=0,
        output_path=output_path,
        fixed_prompt_prefix=None,
    )


class TestAspectRatioBucketingDataloaderWiring:
    """Tests for the aspect_ratio_bucketing branch in _create_dataloader_for_dataset."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_bucketing_enabled_uses_shape_cache(
        self, mock_from_pretrained, mock_tokenizer, local_image_dataset, temp_dir
    ):
        mock_from_pretrained.return_value = mock_tokenizer
        orchestrator = _make_orchestrator()
        cache_dir = temp_dir / "cache"
        args = _make_args(str(cache_dir))

        dataset_config = DatasetConfig(
            type="local",
            image_folder=local_image_dataset["data_dir"],
            captions_file=local_image_dataset["captions_file"],
            aspect_ratio_bucketing=True,
        )

        _dataloader, sampler, dataset_size = orchestrator._create_dataloader_for_dataset(
            dataset_config, "bucketed", args, config={}
        )

        assert isinstance(sampler, ResumableDimensionSampler)
        assert dataset_size == 4
        assert list(cache_dir.glob("*.bucket_dimensions.json")), "shape cache file expected"
        assert not list(
            cache_dir.glob("*.dimensions.json")
        ), "plain dimension cache should not be built when bucketing is enabled"
        assert sampler.group_contiguous is True

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_bucketing_disabled_uses_plain_cache(
        self, mock_from_pretrained, mock_tokenizer, local_image_dataset, temp_dir
    ):
        """Default-off: unchanged behavior still uses the native-size cache."""
        mock_from_pretrained.return_value = mock_tokenizer
        orchestrator = _make_orchestrator()
        cache_dir = temp_dir / "cache"
        args = _make_args(str(cache_dir))

        dataset_config = DatasetConfig(
            type="local",
            image_folder=local_image_dataset["data_dir"],
            captions_file=local_image_dataset["captions_file"],
        )

        _dataloader, sampler, dataset_size = orchestrator._create_dataloader_for_dataset(
            dataset_config, "plain", args, config={}
        )

        assert isinstance(sampler, ResumableDimensionSampler)
        assert dataset_size == 4
        assert list(cache_dir.glob("*.dimensions.json"))
        assert not list(cache_dir.glob("*.bucket_dimensions.json"))
        assert sampler.group_contiguous is False
