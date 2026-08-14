"""Unit tests for the ``aspect_ratio_bucketing`` DatasetConfig option.

Mirrors tests/unit/test_pipeline_multi_dataset.py's style (DatasetConfig field
defaults/overrides, parse_pipeline_config end-to-end, validator error-message
assertions).
"""

import pytest

from fluxflow_training.training.pipeline_config import DatasetConfig, parse_pipeline_config


class TestAspectRatioBucketingDefaults:
    """Default-off behavior must be unchanged for existing configs."""

    def test_default_is_false(self):
        dataset = DatasetConfig(
            type="local", image_folder="/data/images", captions_file="/data/captions.txt"
        )
        assert dataset.aspect_ratio_bucketing is False

    def test_reduced_min_sizes_default_is_none(self):
        dataset = DatasetConfig(
            type="local", image_folder="/data/images", captions_file="/data/captions.txt"
        )
        assert dataset.reduced_min_sizes is None


class TestAspectRatioBucketingParsing:
    """Parsing wires the new keys through unchanged otherwise."""

    @staticmethod
    def _config(**dataset_overrides):
        dataset = {
            "type": "local",
            "image_folder": "/data",
            "captions_file": "/captions.txt",
            **dataset_overrides,
        }
        return {
            "datasets": {"d": dataset},
            "steps": [{"name": "s", "n_epochs": 1, "train_vae": True}],
        }

    def test_parses_true(self):
        config = parse_pipeline_config(self._config(aspect_ratio_bucketing=True))
        assert config.datasets["d"].aspect_ratio_bucketing is True

    def test_omitted_key_defaults_false(self):
        """Existing configs without the new key keep working unchanged."""
        config = parse_pipeline_config(self._config())
        assert config.datasets["d"].aspect_ratio_bucketing is False

    def test_parses_reduced_min_sizes(self):
        config = parse_pipeline_config(self._config(reduced_min_sizes=[128, 256]))
        assert config.datasets["d"].reduced_min_sizes == [128, 256]

    def test_omitted_reduced_min_sizes_defaults_none(self):
        config = parse_pipeline_config(self._config())
        assert config.datasets["d"].reduced_min_sizes is None


class TestAspectRatioBucketingValidation:
    """PipelineConfigValidator rejection cases for aspect_ratio_bucketing."""

    def test_rejects_webdataset_with_bucketing(self):
        config_dict = {
            "datasets": {
                "web": {
                    "type": "webdataset",
                    "webdataset_url": "https://example.com/data.tar",
                    "webdataset_token": "token",
                    "aspect_ratio_bucketing": True,
                }
            },
            "steps": [{"name": "s", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        message = str(exc_info.value)
        assert "aspect_ratio_bucketing" in message
        assert "local" in message

    def test_allows_local_with_bucketing(self):
        config_dict = {
            "datasets": {
                "d": {
                    "type": "local",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                    "aspect_ratio_bucketing": True,
                }
            },
            "steps": [{"name": "s", "n_epochs": 1, "train_vae": True}],
        }
        config = parse_pipeline_config(config_dict)
        assert config.datasets["d"].aspect_ratio_bucketing is True

    def test_rejects_bucketing_combined_with_reduced_min_sizes(self):
        config_dict = {
            "datasets": {
                "d": {
                    "type": "local",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                    "aspect_ratio_bucketing": True,
                    "reduced_min_sizes": [128, 256],
                }
            },
            "steps": [{"name": "s", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "reduced_min_sizes" in str(exc_info.value)

    def test_allows_reduced_min_sizes_without_bucketing(self):
        config_dict = {
            "datasets": {
                "d": {
                    "type": "local",
                    "image_folder": "/data",
                    "captions_file": "/captions.txt",
                    "reduced_min_sizes": [128, 256],
                }
            },
            "steps": [{"name": "s", "n_epochs": 1, "train_vae": True}],
        }
        config = parse_pipeline_config(config_dict)
        assert config.datasets["d"].reduced_min_sizes == [128, 256]

    def test_noise_dataset_with_bucketing_rejected(self):
        """aspect_ratio_bucketing is local-only; noise datasets are rejected too."""
        config_dict = {
            "datasets": {
                "n": {
                    "type": "noise",
                    "aspect_ratio_bucketing": True,
                }
            },
            "steps": [{"name": "s", "n_epochs": 1, "train_vae": True}],
        }
        with pytest.raises(ValueError) as exc_info:
            parse_pipeline_config(config_dict)
        assert "aspect_ratio_bucketing" in str(exc_info.value)
