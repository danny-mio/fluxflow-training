"""Unit tests for caption-truncation logging (v0.10.0).

Training tokenizes captions with a hardcoded/short ``max_text_length``
(default 32) with silent truncation. These tests cover the
``_TruncationTracker`` helper in ``fluxflow_training.data.datasets`` and its
wiring into ``TextImageDataset``, ``StreamingWebDataset``, and
``NoiseDataset`` so truncation is observable via logging instead of silent.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from fluxflow_training.data.datasets import (
    NoiseDataset,
    StreamingWebDataset,
    TextImageDataset,
    _TruncationTracker,
)


def _tokenizer_stub(lengths_by_text: dict[str, int], max_text_length: int = 32):
    """Tokenizer mock: truncation=False returns the "true" length for the text;
    truncation=True returns a fixed max_text_length-sized padded encoding."""

    def _call(text, truncation=True, max_length=max_text_length, **kwargs):
        if truncation:
            n = max_length
        else:
            n = lengths_by_text[text]
        return {
            "input_ids": (
                torch.zeros(1, n, dtype=torch.long)
                if kwargs.get("return_tensors") == "pt"
                else list(range(n))
            ),
        }

    tok = MagicMock(side_effect=_call)
    return tok


class TestTruncationTracker:
    """Direct unit tests for the sampling/aggregation helper."""

    def test_observe_below_max_never_flags_truncation(self):
        tok = _tokenizer_stub({"short caption": 5}, max_text_length=32)
        logger = MagicMock()
        tracker = _TruncationTracker(
            tok, max_text_length=32, name="X", sample_rate=1, log_every=1, logger=logger
        )
        tracker.observe("short caption")
        assert tracker.truncated_count == 0
        assert tracker.sampled_count == 1

    def test_observe_above_max_flags_truncation(self):
        tok = _tokenizer_stub({"a very long caption": 50}, max_text_length=32)
        logger = MagicMock()
        tracker = _TruncationTracker(
            tok, max_text_length=32, name="X", sample_rate=1, log_every=1, logger=logger
        )
        tracker.observe("a very long caption")
        assert tracker.truncated_count == 1
        assert tracker.sampled_count == 1

    def test_sample_rate_skips_between_samples(self):
        tok = _tokenizer_stub({"c": 10}, max_text_length=32)
        logger = MagicMock()
        tracker = _TruncationTracker(
            tok, max_text_length=32, name="X", sample_rate=3, log_every=100, logger=logger
        )
        for _ in range(7):
            tracker.observe("c")
        # Only samples 3 and 6 (1-indexed, seen % sample_rate == 0) trigger a
        # second, untruncated tokenizer call.
        assert tracker.sampled_count == 2
        assert tracker.seen_count == 7

    def test_logs_summary_every_log_every_samples(self):
        tok = _tokenizer_stub({"c": 10}, max_text_length=32)
        logger = MagicMock()
        tracker = _TruncationTracker(
            tok, max_text_length=32, name="MyDataset", sample_rate=1, log_every=2, logger=logger
        )
        tracker.observe("c")
        logger.info.assert_not_called()
        tracker.observe("c")
        logger.info.assert_called_once()
        (msg, *fmt_args), _ = logger.info.call_args
        rendered = msg % tuple(fmt_args) if fmt_args else msg
        assert "MyDataset" in rendered

    def test_summary_reports_percentiles(self):
        lengths = {f"c{i}": length for i, length in enumerate([10, 20, 30, 40, 50])}
        tok = _tokenizer_stub(lengths, max_text_length=25)
        logger = MagicMock()
        tracker = _TruncationTracker(
            tok, max_text_length=25, name="X", sample_rate=1, log_every=5, logger=logger
        )
        for text in lengths:
            tracker.observe(text)
        assert tracker.truncated_count == 3  # 30, 40, 50 exceed 25
        logger.info.assert_called_once()


class TestTextImageDatasetTruncationLogging:
    def test_getitem_reports_to_tracker(self, mock_image_dataset):
        with patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained") as patched:
            patched.return_value = _tokenizer_stub(
                {c: 40 for c in mock_image_dataset["captions"]}, max_text_length=32
            )
            ds = TextImageDataset(
                data_path=mock_image_dataset["data_dir"],
                captions_file=mock_image_dataset["captions_file"],
                tokenizer_name="distilbert-base-uncased",
                max_text_length=32,
            )
            assert isinstance(ds._truncation_tracker, _TruncationTracker)
            with patch.object(ds._truncation_tracker, "observe") as observe:
                ds[0]
                observe.assert_called_once_with(mock_image_dataset["captions"][0])


class TestStreamingWebDatasetTruncationLogging:
    def test_tracker_constructed_with_max_text_length(self):
        with (
            patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained") as patched,
            patch("fluxflow_training.data.datasets.HfFileSystem") as mock_fs,
            patch("fluxflow_training.data.datasets.hf_hub_url", return_value="http://x"),
        ):
            patched.return_value = _tokenizer_stub({"p": 10}, max_text_length=32)
            mock_fs.return_value.glob.return_value = ["a.tar"]
            mock_fs.return_value.resolve_path.return_value = MagicMock(
                repo_id="repo", path_in_repo="a.tar"
            )
            ds = StreamingWebDataset(
                tokenizer_name="distilbert-base-uncased",
                token="tok",
                max_text_length=32,
            )
            assert ds._truncation_tracker.max_text_length == 32


class TestNoiseDatasetTruncationLogging:
    def test_logs_once_at_construction(self):
        with (
            patch("fluxflow_training.data.datasets.AutoTokenizer.from_pretrained") as patched,
            patch("fluxflow_training.data.datasets.logger") as mock_logger,
        ):
            patched.return_value = _tokenizer_stub({"": 2}, max_text_length=32)
            NoiseDataset(num_samples=4, max_text_length=32)
            assert mock_logger.info.called
