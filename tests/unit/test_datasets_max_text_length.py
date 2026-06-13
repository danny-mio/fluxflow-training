"""Unit tests for the configurable ``max_text_length`` parameter (v0.10.0).

The dataset classes now expose ``max_text_length`` so callers can pin the
tokenized prompt length to the flow processor's T_txt (default 32 in
the redesign). Previously the value was hardcoded at 128.
"""

from unittest.mock import patch

import pytest
import torch

from fluxflow_training.data.datasets import NoiseDataset, TextImageDataset


@pytest.fixture
def mock_tokenizer_factory():
    """Returns a tokenizer whose padded output length tracks the caller's max_length."""
    from unittest.mock import MagicMock

    def _build(max_length: int):
        tok = MagicMock()
        tok.pad_token = "[PAD]"
        tok.pad_token_id = 0

        def _call(text, max_length=max_length, **kwargs):  # noqa: B008
            return {
                "input_ids": torch.zeros(1, max_length, dtype=torch.long),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }

        tok.side_effect = _call
        tok.add_special_tokens = MagicMock()
        return tok

    return _build


class TestTextImageDatasetMaxTextLength:
    def test_default_is_32(self, mock_tokenizer_factory, mock_image_dataset):
        """v0.10.0 default is 32 tokens (matches flow processor T_txt)."""
        with patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained") as patched:
            patched.return_value = mock_tokenizer_factory(max_length=32)
            ds = TextImageDataset(
                data_path=mock_image_dataset["data_dir"],
                captions_file=mock_image_dataset["captions_file"],
                tokenizer_name="distilbert-base-uncased",
            )
            assert ds.max_text_length == 32
            input_ids, _ = ds[0]
            assert input_ids.shape == (32,)

    def test_explicit_override(self, mock_tokenizer_factory, mock_image_dataset):
        """Caller can still request a longer max_length (e.g. for legacy models)."""
        with patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained") as patched:
            patched.return_value = mock_tokenizer_factory(max_length=128)
            ds = TextImageDataset(
                data_path=mock_image_dataset["data_dir"],
                captions_file=mock_image_dataset["captions_file"],
                tokenizer_name="distilbert-base-uncased",
                max_text_length=128,
            )
            assert ds.max_text_length == 128
            input_ids, _ = ds[0]
            assert input_ids.shape == (128,)


class TestNoiseDatasetMaxTextLength:
    def test_default_is_32(self, mock_tokenizer_factory):
        with patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained") as patched:
            patched.return_value = mock_tokenizer_factory(max_length=32)
            ds = NoiseDataset(num_samples=4)
            assert ds.max_text_length == 32
            assert ds.empty_caption_tokens.shape == (32,)

    def test_explicit_override(self, mock_tokenizer_factory):
        with patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained") as patched:
            patched.return_value = mock_tokenizer_factory(max_length=64)
            ds = NoiseDataset(num_samples=4, max_text_length=64)
            assert ds.max_text_length == 64
            assert ds.empty_caption_tokens.shape == (64,)
