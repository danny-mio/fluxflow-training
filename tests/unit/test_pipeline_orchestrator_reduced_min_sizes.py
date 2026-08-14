"""Unit tests for per-dataset reduced_min_sizes wiring in
TrainingPipelineOrchestrator._create_dataloader_for_dataset.

Verifies the orchestrator's collate_fn prefers DatasetConfig.reduced_min_sizes
when set, falling back to the global args.reduced_min_sizes when unset —
mirroring the batch_size/workers fallback convention used in the same method.
"""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

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


def _make_args(output_path, reduced_min_sizes=None):
    return Mock(
        tokenizer_name="distilbert-base-uncased",
        channels=3,
        img_size=512,
        reduced_min_sizes=reduced_min_sizes,
        batch_size=2,
        workers=0,
        output_path=output_path,
        fixed_prompt_prefix=None,
    )


class TestReducedMinSizesDataloaderWiring:
    """Tests for the reduced_min_sizes fallback in _create_dataloader_for_dataset."""

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_per_dataset_reduced_min_sizes_overrides_global(
        self, mock_from_pretrained, mock_tokenizer, local_image_dataset, temp_dir
    ):
        mock_from_pretrained.return_value = mock_tokenizer
        orchestrator = _make_orchestrator()
        args = _make_args(str(temp_dir / "cache"), reduced_min_sizes=[64, 128])

        dataset_config = DatasetConfig(
            type="local",
            image_folder=local_image_dataset["data_dir"],
            captions_file=local_image_dataset["captions_file"],
            reduced_min_sizes=[256, 512],
        )

        dataloader, _sampler, _dataset_size = orchestrator._create_dataloader_for_dataset(
            dataset_config, "override", args, config={}
        )

        assert dataloader.collate_fn.keywords["reduced_min_sizes"] == [256, 512]

    @patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained")
    def test_unset_per_dataset_reduced_min_sizes_falls_back_to_global(
        self, mock_from_pretrained, mock_tokenizer, local_image_dataset, temp_dir
    ):
        """Default behavior: no per-dataset override still uses args.reduced_min_sizes."""
        mock_from_pretrained.return_value = mock_tokenizer
        orchestrator = _make_orchestrator()
        args = _make_args(str(temp_dir / "cache"), reduced_min_sizes=[64, 128])

        dataset_config = DatasetConfig(
            type="local",
            image_folder=local_image_dataset["data_dir"],
            captions_file=local_image_dataset["captions_file"],
        )

        dataloader, _sampler, _dataset_size = orchestrator._create_dataloader_for_dataset(
            dataset_config, "fallback", args, config={}
        )

        assert dataloader.collate_fn.keywords["reduced_min_sizes"] == [64, 128]
