"""Tests for multi-dataset pipeline configuration."""

import pytest

from fluxflow_training.training.pipeline_config import DatasetConfig, parse_pipeline_config


class TestDatasetConfig:
    """Test DatasetConfig dataclass."""

    def test_local_dataset_default_values(self):
        """Test local dataset with default values."""
        dataset = DatasetConfig(
            type="local",
            image_folder="/path/to/images",
            captions_file="/path/to/captions.txt",
        )
        assert dataset.type == "local"
        assert dataset.image_folder == "/path/to/images"
        assert dataset.captions_file == "/path/to/captions.txt"
        assert dataset.batch_size is None
        assert dataset.workers is None

    def test_webdataset_default_values(self):
        """Test webdataset with default values."""
        dataset = DatasetConfig(
            type="webdataset",
            webdataset_url="https://example.com/data.tar",
            webdataset_token="hf_token",
        )
        assert dataset.type == "webdataset"
        assert dataset.webdataset_url == "https://example.com/data.tar"
        assert dataset.webdataset_token == "hf_token"
        assert dataset.webdataset_image_key == "png"
        assert dataset.webdataset_label_key == "json"
        assert dataset.webdataset_caption_key == "prompt"
        assert dataset.webdataset_size == 10000
        assert dataset.webdataset_samples_per_shard == 1000

    def test_dataset_with_overrides(self):
        """Test dataset with batch_size and workers overrides."""
        dataset = DatasetConfig(
            type="local",
            image_folder="/path/to/images",
            captions_file="/path/to/captions.txt",
            batch_size=8,
            workers=4,
        )
        assert dataset.batch_size == 8
        assert dataset.workers == 4


class TestParseDatasetsConfig:
    """Test parsing dataset configurations."""

    def test_parse_single_local_dataset(self):
        """Test parsing a single local dataset."""
        config_dict = {
            "datasets": {
                "local_data": {
                    "type": "local",
                    "image_folder": "/data/images",
                    "captions_file": "/data/captions.txt",
                }
            },
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        config = parse_pipeline_config(config_dict)
        assert "local_data" in config.datasets
        assert config.datasets["local_data"].type == "local"
        assert config.datasets["local_data"].image_folder == "/data/images"

    def test_parse_multiple_datasets(self):
        """Test parsing multiple datasets."""
        config_dict = {
            "datasets": {
                "local1": {
                    "type": "local",
                    "image_folder": "/data1",
                    "captions_file": "/captions1.txt",
                },
                "webdata1": {
                    "type": "webdataset",
                    "webdataset_url": "https://example.com/data.tar",
                    "webdataset_token": "token",
                },
            },
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        config = parse_pipeline_config(config_dict)
        assert len(config.datasets) == 2
        assert "local1" in config.datasets
        assert "webdata1" in config.datasets
        assert config.datasets["local1"].type == "local"
        assert config.datasets["webdata1"].type == "webdataset"

    def test_parse_dataset_with_overrides(self):
        """Test parsing dataset with batch_size and workers."""
        config_dict = {
            "datasets": {
                "my_data": {
                    "type": "local",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                    "batch_size": 16,
                    "workers": 12,
                }
            },
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        config = parse_pipeline_config(config_dict)
        assert config.datasets["my_data"].batch_size == 16
        assert config.datasets["my_data"].workers == 12

    def test_parse_default_dataset(self):
        """Test parsing default_dataset field."""
        config_dict = {
            "datasets": {
                "dataset1": {
                    "type": "local",
                    "image_folder": "/data1",
                    "captions_file": "/captions1.txt",
                },
                "dataset2": {
                    "type": "local",
                    "image_folder": "/data2",
                    "captions_file": "/captions2.txt",
                },
            },
            "default_dataset": "dataset1",
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        config = parse_pipeline_config(config_dict)
        assert config.default_dataset == "dataset1"


class TestStepDatasetAssignment:
    """Test dataset assignment to pipeline steps."""

    def test_step_with_dataset_assignment(self):
        """Test step with explicit dataset assignment."""
        config_dict = {
            "datasets": {
                "my_data": {
                    "type": "local",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                }
            },
            "steps": [{"name": "step1", "n_epochs": 1, "train_vae": True, "dataset": "my_data"}],
        }
        config = parse_pipeline_config(config_dict)
        assert config.steps[0].dataset == "my_data"

    def test_step_without_dataset_assignment(self):
        """Test step without explicit dataset (uses default)."""
        config_dict = {
            "datasets": {
                "my_data": {
                    "type": "local",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                }
            },
            "default_dataset": "my_data",
            "steps": [{"name": "step1", "n_epochs": 1, "train_vae": True}],
        }
        config = parse_pipeline_config(config_dict)
        assert config.steps[0].dataset is None
        assert config.default_dataset == "my_data"

    def test_multiple_steps_different_datasets(self):
        """Test multiple steps with different datasets."""
        config_dict = {
            "datasets": {
                "dataset1": {
                    "type": "local",
                    "image_folder": "/data1",
                    "captions_file": "/captions1.txt",
                },
                "dataset2": {
                    "type": "local",
                    "image_folder": "/data2",
                    "captions_file": "/captions2.txt",
                },
            },
            "steps": [
                {"name": "step1", "n_epochs": 1, "train_vae": True, "dataset": "dataset1"},
                {"name": "step2", "n_epochs": 1, "train_diff": True, "dataset": "dataset2"},
            ],
        }
        config = parse_pipeline_config(config_dict)
        assert config.steps[0].dataset == "dataset1"
        assert config.steps[1].dataset == "dataset2"


class TestDatasetValidation:
    """Test dataset configuration validation."""

    def test_local_dataset_missing_image_folder(self):
        """Test validation fails for local dataset without image_folder."""
        config_dict = {
            "datasets": {"bad_data": {"type": "local", "captions_file": "/captions.txt"}},
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "image_folder" in str(exc_info.value)

    def test_local_dataset_missing_captions_file(self):
        """Test validation fails for local dataset without captions_file."""
        config_dict = {
            "datasets": {"bad_data": {"type": "local", "image_folder": "/data"}},
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "captions_file" in str(exc_info.value)

    def test_webdataset_missing_url(self):
        """Test validation fails for webdataset without URL."""
        config_dict = {
            "datasets": {"bad_data": {"type": "webdataset", "webdataset_token": "token"}},
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "webdataset_url" in str(exc_info.value)

    def test_webdataset_missing_token(self):
        """Test validation fails for webdataset without token."""
        config_dict = {
            "datasets": {"bad_data": {"type": "webdataset", "webdataset_url": "https://data.tar"}},
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "webdataset_token" in str(exc_info.value)

    def test_invalid_dataset_type(self):
        """Test validation fails for invalid dataset type."""
        config_dict = {
            "datasets": {
                "bad_data": {
                    "type": "invalid_type",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                }
            },
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "unknown type" in str(exc_info.value).lower()

    def test_default_dataset_not_found(self):
        """Test validation fails when default_dataset doesn't exist."""
        config_dict = {
            "datasets": {
                "dataset1": {
                    "type": "local",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                }
            },
            "default_dataset": "nonexistent",
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "default_dataset" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_step_dataset_reference_not_found(self):
        """Test validation fails when step references nonexistent dataset."""
        config_dict = {
            "datasets": {
                "dataset1": {
                    "type": "local",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                }
            },
            "steps": [
                {
                    "name": "test",
                    "n_epochs": 1,
                    "train_vae": True,
                    "dataset": "nonexistent",
                }
            ],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "references unknown dataset" in str(exc_info.value)

    def test_default_dataset_without_datasets_dict(self):
        """Test validation fails when default_dataset is set but datasets is empty."""
        config_dict = {
            "default_dataset": "my_data",
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "no datasets configured" in str(exc_info.value)

    def test_step_references_dataset_without_datasets_dict(self):
        """Test validation fails when step references dataset but none configured."""
        config_dict = {
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True, "dataset": "my_data"}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "no datasets configured" in str(exc_info.value)


class TestBackwardCompatibility:
    """Test backward compatibility with single-dataset configs."""

    def test_config_without_datasets(self):
        """Test that configs without datasets field still work."""
        config_dict = {"steps": [{"name": "test", "n_epochs": 1, "train_vae": True}]}
        config = parse_pipeline_config(config_dict)
        assert config.datasets == {}
        assert config.default_dataset is None
        assert config.steps[0].dataset is None

    def test_config_with_empty_datasets(self):
        """Test that configs with empty datasets dict still work."""
        config_dict = {
            "datasets": {},
            "steps": [{"name": "test", "n_epochs": 1, "train_vae": True}],
        }
        config = parse_pipeline_config(config_dict)
        assert config.datasets == {}
        assert config.default_dataset is None
