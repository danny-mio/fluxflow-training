"""Unit tests for ResumableDimensionSampler(group_contiguous=True).

Covers the aspect_ratio_bucketing batch-ordering fix: when enabled, batches
from the same size group must stay contiguous in ``epoch_batches`` (only
group *visitation order* is randomized per epoch), instead of being
individually shuffled across groups.
"""

import itertools

import pytest

from fluxflow_training.data.datasets import ResumableDimensionSampler

BATCH_SIZE = 4

# Four groups, indices non-overlapping and contiguous ranges for easy
# membership checks. Sizes chosen so some groups have a remainder and some
# don't, exercising the remainder-pool-appended-last behavior.
_GROUP_RANGES = {
    (64, 64): range(0, 40),  # 40 -> 10 full batches, no remainder
    (64, 128): range(40, 70),  # 30 -> 7 full batches, remainder 2
    (128, 64): range(70, 92),  # 22 -> 5 full batches, remainder 2
    (128, 128): range(92, 142),  # 50 -> 12 full batches, remainder 2
}


@pytest.fixture
def synthetic_dimension_cache():
    """Multi-group dimension cache with varied group sizes for contiguity tests."""
    size_groups = {
        str(tuple(size)): {"indices": list(indices), "count": len(indices)}
        for size, indices in _GROUP_RANGES.items()
    }
    return {
        "dataset_path": "/fake/path",
        "captions_file": "/fake/captions.txt",
        "scan_date": "2025-01-01T00:00:00",
        "total_images": sum(len(r) for r in _GROUP_RANGES.values()),
        "multiple": 32,
        "size_groups": size_groups,
        "statistics": {
            "num_groups": len(_GROUP_RANGES),
            "min_group_size": min(len(r) for r in _GROUP_RANGES.values()),
            "max_group_size": max(len(r) for r in _GROUP_RANGES.values()),
            "avg_group_size": sum(len(r) for r in _GROUP_RANGES.values()) // len(_GROUP_RANGES),
        },
    }


def _label_for_batch(batch: list[int]) -> str:
    """Map a batch to its size-group label, or 'remainder' if mixed/no single match."""
    batch_set = set(batch)
    for size, indices in _GROUP_RANGES.items():
        if batch_set.issubset(set(indices)):
            return str(size)
    return "remainder"


def _label_sequence(epoch_batches: list[list[int]]) -> list[str]:
    return [_label_for_batch(batch) for batch in epoch_batches]


def _first_appearance_order(labels: list[str]) -> list[str]:
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return seen


class TestGroupContiguousDefaultUnchanged:
    """group_contiguous defaults to False and must not alter existing behavior."""

    def test_default_false_explicit(self, mock_dimension_cache):
        sampler_default = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache, batch_size=8, seed=42
        )
        sampler_explicit = ResumableDimensionSampler(
            dimension_cache=mock_dimension_cache,
            batch_size=8,
            seed=42,
            group_contiguous=False,
        )
        assert sampler_default.epoch_batches == sampler_explicit.epoch_batches
        assert sampler_default.group_contiguous is False
        assert sampler_explicit.group_contiguous is False


class TestGroupContiguousBatchOrdering:
    """group_contiguous=True must keep each group's batches as one contiguous block."""

    def test_batches_contiguous_per_group(self, synthetic_dimension_cache):
        sampler = ResumableDimensionSampler(
            dimension_cache=synthetic_dimension_cache,
            batch_size=BATCH_SIZE,
            seed=1,
            group_contiguous=True,
        )

        labels = _label_sequence(sampler.epoch_batches)

        # Every batch has a full batch size, and remainder batches are labeled
        # 'remainder' (mixed groups) since 2+2+2=6 leftovers form 1 full batch.
        assert all(len(batch) == BATCH_SIZE for batch in sampler.epoch_batches)

        # Each label must occupy exactly one contiguous run in the sequence.
        runs = [label for label, _ in itertools.groupby(labels)]
        assert len(runs) == len(set(labels)), (
            f"labels are not contiguous per group: run sequence={runs}, "
            f"unique labels={set(labels)}"
        )

        # Remainder block, if present, must be last.
        if "remainder" in labels:
            assert labels[-1] == "remainder"
            # And it must be a single trailing block.
            remainder_start = labels.index("remainder")
            assert all(label == "remainder" for label in labels[remainder_start:])

    def test_no_batches_dropped_or_duplicated_within_groups(self, synthetic_dimension_cache):
        """Sanity: total indices trained per epoch matches full-batch coverage."""
        sampler = ResumableDimensionSampler(
            dimension_cache=synthetic_dimension_cache,
            batch_size=BATCH_SIZE,
            seed=7,
            group_contiguous=True,
        )
        all_indices = [idx for batch in sampler.epoch_batches for idx in batch]
        assert len(all_indices) == len(set(all_indices))

        # 10 + 7 + 5 + 12 = 34 full-group batches, +1 remainder batch = 35.
        assert len(sampler.epoch_batches) == 35


class TestGroupContiguousVisitationOrderRandomized:
    """Group visitation order (which group's contiguous block comes first) is
    itself randomized via the seeded rng, rather than fixed dict order."""

    def test_order_differs_across_seeds(self, synthetic_dimension_cache):
        orders = []
        for seed in range(1, 7):
            sampler = ResumableDimensionSampler(
                dimension_cache=synthetic_dimension_cache,
                batch_size=BATCH_SIZE,
                seed=seed,
                group_contiguous=True,
            )
            labels = _label_sequence(sampler.epoch_batches)
            real_group_order = [
                label for label in _first_appearance_order(labels) if label != "remainder"
            ]
            orders.append(tuple(real_group_order))

        # Not all seeds should coincidentally produce the same visitation order.
        assert len(set(orders)) > 1, f"visitation order never changed across seeds: {orders}"

    def test_order_differs_across_epochs(self, synthetic_dimension_cache):
        sampler = ResumableDimensionSampler(
            dimension_cache=synthetic_dimension_cache,
            batch_size=BATCH_SIZE,
            seed=42,
            group_contiguous=True,
        )
        labels_epoch0 = _label_sequence(sampler.epoch_batches)
        order0 = tuple(
            label for label in _first_appearance_order(labels_epoch0) if label != "remainder"
        )

        orders_later = set()
        for epoch in range(1, 6):
            sampler.set_epoch(epoch)
            labels = _label_sequence(sampler.epoch_batches)
            order = tuple(
                label for label in _first_appearance_order(labels) if label != "remainder"
            )
            orders_later.add(order)

        assert orders_later != {order0}, "visitation order never changed across epochs"


class TestGroupContiguousResume:
    """Resume must be deterministic and identical when group_contiguous=True."""

    def test_resume_regenerates_identical_epoch_batches(self, synthetic_dimension_cache):
        sampler1 = ResumableDimensionSampler(
            dimension_cache=synthetic_dimension_cache,
            batch_size=BATCH_SIZE,
            seed=42,
            group_contiguous=True,
        )

        # Consume some batches.
        consumed = []
        for i, batch in enumerate(sampler1):
            consumed.append(batch)
            if i >= 9:
                break

        state = sampler1.state_dict()

        sampler2 = ResumableDimensionSampler(
            dimension_cache=synthetic_dimension_cache,
            batch_size=BATCH_SIZE,
            resume_state=state,
            group_contiguous=True,
        )

        assert sampler2.epoch_batches == sampler1.epoch_batches
        assert sampler2.position == sampler1.position

        # Next batch yielded by resumed sampler matches what original would
        # have yielded next.
        expected_next = sampler1.epoch_batches[sampler1.position]
        actual_next = next(iter(sampler2))
        assert actual_next == expected_next
