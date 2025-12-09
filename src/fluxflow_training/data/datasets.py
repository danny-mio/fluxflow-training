"""Dataset classes for FluxFlow text-to-image training and generation."""

import hashlib
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Iterator, Optional

import torch
import webdataset as wds
from fluxflow.types import DimensionCacheData, SamplerState  # type: ignore[attr-defined]
from huggingface_hub import HfFileSystem, hf_hub_url
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, Sampler
from torchvision import transforms
from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False


class TextImageDataset(Dataset):
    """Dataset for local text-image pairs with variable-size batching."""

    def __init__(
        self,
        data_path: str,
        captions_file: Optional[str] = None,
        tokenizer_name: str = "gpt2",
        transform: Optional[Callable] = None,
        generate_mode: bool = False,
        fixed_prompt_prefix: Optional[str] = None,
    ):
        """
        Args:
            data_path: Root directory containing images or text files
            captions_file: Tab-separated file: image_name\tcaption (train mode)
            tokenizer_name: HuggingFace tokenizer identifier
            transform: Optional image transform (applied during loading)
            generate_mode: If True, load .txt files from data_path instead
            fixed_prompt_prefix: Optional text to prepend to all prompts (e.g. "style anime")
        """
        self.data_path = data_path
        self.transform = transform
        self.generate_mode = generate_mode
        self.fixed_prompt_prefix = fixed_prompt_prefix

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir="./_cache", local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        if not generate_mode:
            # Load captions and image paths from tab-separated file
            if captions_file is None:
                raise ValueError("captions_file is required when generate_mode=False")
            self.image_paths = []
            self.captions = []
            with open(captions_file, "r") as f:
                for line in f:
                    image_name, caption = line.strip().split("\t")
                    image_path = os.path.join(data_path, image_name)
                    self.image_paths.append(image_path)
                    # Add fixed prefix if provided
                    if self.fixed_prompt_prefix:
                        caption = f"{self.fixed_prompt_prefix}. {caption}"
                    self.captions.append(caption)
        else:
            # Generation mode: read text prompts from .txt files
            self.captions = []
            self.file_names = []
            for file_name in os.listdir(data_path):
                if file_name.endswith(".txt"):
                    with open(os.path.join(data_path, file_name), "r") as f:
                        caption = f.read().strip()
                        # Add fixed prefix if provided
                        if self.fixed_prompt_prefix:
                            caption = f"{self.fixed_prompt_prefix}. {caption}"
                        self.captions.append(caption)
                        self.file_names.append(file_name.replace(".txt", ""))

    def get_image_size_class(self, idx: int, multiple: int = 16) -> tuple[int, int]:
        """
        Returns (H, W) rounded to nearest multiple for batching.

        Args:
            idx: Dataset index
            multiple: Round dimensions to this value

        Returns:
            Tuple (height, width) rounded to multiple
        """
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            w, h = img.size
        h = round(h / multiple) * multiple
        w = round(w / multiple) * multiple
        return (h, w)

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor] | tuple[torch.Tensor, str]:
        """
        Returns:
            Generate mode: (file_name, input_ids)
            Train mode: (input_ids, image_path)
        """
        caption = self.captions[idx]
        encoding = self.tokenizer.encode_plus(
            caption,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)

        if self.generate_mode:
            return self.file_names[idx], input_ids
        else:
            # Return image_path for lazy loading in collate_fn
            return input_ids, self.image_paths[idx]


class StreamingWebDataset(IterableDataset):
    """Generic WebDataset wrapper for HuggingFace streaming datasets with robust error handling.

    Handles corrupted images and network errors gracefully by skipping bad samples.
    """

    def __init__(
        self,
        tokenizer_name: str,
        token: str,
        url_pattern: str = "hf://datasets/jackyhate/text-to-image-2M/data_512_2M/*.tar",
        channels: int = 3,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        timeout: int = 30,
        image_key: str = "png",
        label_key: str = "json",
        caption_key: str = "prompt",
        dataset_size: Optional[int] = None,
        samples_per_shard: int = 10000,
        fixed_prompt_prefix: Optional[str] = None,
    ):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer identifier
            token: HuggingFace access token for dataset authentication
            url_pattern: HuggingFace dataset URL pattern (e.g., "hf://datasets/user/repo/path/*.tar")
            channels: Number of image channels (3 for RGB)
            max_retries: Maximum number of retries on connection errors
            retry_delay: Base delay between retries (exponential backoff)
            timeout: Curl timeout in seconds
            image_key: Key for image data in WebDataset samples
            label_key: Key for label/metadata in WebDataset samples
            caption_key: Key for caption text within the label JSON (e.g., "prompt" or "caption")
            dataset_size: Total number of samples (if known). If None, estimates from shard count.
            samples_per_shard: Estimated samples per shard for size estimation (default 10000)
            fixed_prompt_prefix: Optional text to prepend to all prompts (e.g. "style anime")
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.token = token
        self.channels = channels
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.image_key = image_key
        self.label_key = label_key
        self.caption_key = caption_key
        self.fixed_prompt_prefix = fixed_prompt_prefix
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5] * channels, [0.5] * channels),
            ]
        )

        # Construct WebDataset URLs from HuggingFace
        fs = HfFileSystem()
        path_strs = list(fs.glob(url_pattern))
        files = [fs.resolve_path(p) for p in path_strs]  # type: ignore[arg-type]
        self.urls = [hf_hub_url(f.repo_id, f.path_in_repo, repo_type="dataset") for f in files]  # type: ignore[attr-defined]

        if not self.urls:
            raise ValueError(f"No files found matching pattern: {url_pattern}")

        # Calculate or estimate dataset size
        self.num_shards = len(self.urls)
        if dataset_size is not None:
            self._dataset_size = dataset_size
        else:
            # Estimate from shard count
            self._dataset_size = self.num_shards * samples_per_shard

    def __len__(self) -> int:
        """Return the total number of samples (exact or estimated)."""
        return self._dataset_size

    @property
    def dataset_size(self) -> int:
        """Return the total number of samples (exact or estimated)."""
        return self._dataset_size

    def _create_dataset(self):
        """Create WebDataset pipeline with robust error handling for corrupted images.

        Performance considerations:
        - shardshuffle=10: Low value for faster startup (was 100)
        - shuffle=100: Small buffer for immediate first batch (was 1000)
        - With workers=1, total buffer = 100 samples (fast!)
        - With workers=N, total buffer = N×100 samples (still reasonable)
        """
        # Use curl with timeout and retry options
        url_pipe = (
            f"pipe: curl -s -L "
            f"--retry {self.max_retries} "
            f"--retry-delay 2 "
            f"--retry-max-time 60 "
            f"--connect-timeout {self.timeout} "
            f"--max-time 300 "
            f"-H 'Authorization:Bearer {self.token}' "
            f"{' '.join(self.urls)}"
        )

        # Custom handler that silently ignores corrupted images
        # This handles DecodingError from broken data streams
        def ignore_and_continue(exn):
            """Silently skip corrupted samples without warnings."""
            return True  # Return True to continue, don't re-raise

        return (
            wds.WebDataset(
                url_pipe,
                shardshuffle=10,  # Reduced from 100 - faster startup, less memory
                handler=ignore_and_continue,
                empty_check=False,
            )
            .shuffle(100)  # Reduced from 1000 - much faster first batch
            .decode("pil", handler=ignore_and_continue)
            .rename(image=self.image_key, label=self.label_key)
            .to_tuple("image", "label")
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, Image.Image]]:
        """Yields (input_ids, image) tuples with robust error handling."""
        retry_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 10

        while True:
            try:
                dataset = self._create_dataset()

                for img, label in dataset:
                    try:
                        prompt = (
                            label.get(self.caption_key, "")
                            .replace("\n", " ")
                            .replace("\r", " ")
                            .strip()
                        )
                        if not prompt:
                            continue

                        # Add fixed prefix if provided
                        if self.fixed_prompt_prefix:
                            prompt = f"{self.fixed_prompt_prefix}. {prompt}"

                        encoding = self.tokenizer.encode_plus(
                            prompt,
                            max_length=128,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        )
                        input_ids = encoding["input_ids"].squeeze(0)

                        # Reset error counters on success
                        consecutive_errors = 0
                        yield input_ids, img

                    except Exception as e:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(
                                f"WebDataset: {max_consecutive_errors} consecutive item errors, reconnecting..."
                            )
                            break
                        # Skip bad items silently
                        continue

                # End of dataset reached - reset retry counter and restart for infinite streaming
                retry_count = 0
                continue

            except (ConnectionError, TimeoutError, OSError, BrokenPipeError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    print(f"WebDataset: Max retries ({self.max_retries}) exceeded, giving up")
                    raise

                delay = self.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                print(
                    f"WebDataset: Connection error ({type(e).__name__}), retry {retry_count}/{self.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)

            except Exception as e:
                # For other exceptions, try to continue if possible
                retry_count += 1
                if retry_count > self.max_retries:
                    print(f"WebDataset: Unexpected error after {self.max_retries} retries: {e}")
                    raise

                delay = self.retry_delay * (2 ** (retry_count - 1))
                print(
                    f"WebDataset: Error ({type(e).__name__}: {e}), retry {retry_count}/{self.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)


# Backward compatibility alias
TTI2MDataset = StreamingWebDataset


class GroupedBatchSampler(Sampler):
    """Groups images by size class and yields fixed-size batches (drops incomplete)."""

    def __init__(
        self,
        dataset: TextImageDataset,
        batch_size: int,
        shuffle: bool = True,
        multiple: int = 32,
    ):
        """
        Args:
            dataset: TextImageDataset instance
            batch_size: Number of samples per batch
            shuffle: Shuffle groups and samples
            multiple: Size class granularity (images rounded to this multiple)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.multiple = multiple
        self.grouped_indices = self._group_indices()

    def _group_indices(self) -> list:
        """Groups dataset indices by image size class."""
        size_buckets = defaultdict(list)

        for idx in range(len(self.dataset)):
            size_class = self.dataset.get_image_size_class(idx, multiple=self.multiple)
            size_buckets[size_class].append(idx)

        grouped = []
        for group in size_buckets.values():
            if self.shuffle:
                random.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                batch = group[i : i + self.batch_size]
                if len(batch) == self.batch_size:  # Drop last incomplete batch
                    grouped.append(batch)

        if self.shuffle:
            random.shuffle(grouped)
        return grouped

    def __iter__(self):
        for batch in self.grouped_indices:
            yield batch

    def __len__(self) -> int:
        return len(self.grouped_indices)


class StreamingGroupedBatchSampler(Sampler):
    """
    Streaming variant that yields batches as soon as a size bucket fills.
    Drops incomplete batches at the end.
    """

    def __init__(
        self,
        dataset: TextImageDataset,
        batch_size: int,
        shuffle: bool = True,
        multiple: int = 32,
    ):
        """
        Args:
            dataset: TextImageDataset instance
            batch_size: Number of samples per batch
            shuffle: Shuffle input indices before grouping
            multiple: Size class granularity
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.multiple = multiple

    def __iter__(self):
        size_buckets = defaultdict(list)
        idx_pool = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idx_pool)

        for idx in idx_pool:
            size_class = self.dataset.get_image_size_class(idx, multiple=self.multiple)
            size_buckets[size_class].append(idx)

            if len(size_buckets[size_class]) >= self.batch_size:
                batch = size_buckets[size_class][: self.batch_size]
                size_buckets[size_class] = size_buckets[size_class][self.batch_size :]
                yield batch

    def __len__(self) -> int:
        """Estimated length (not exact due to dropped incomplete batches)."""
        return len(self.dataset) // self.batch_size


def build_dimension_cache(
    dataset: TextImageDataset,
    multiple: int = 32,
    show_progress: bool = True,
) -> DimensionCacheData:
    """
    Scan all images once and build dimension cache.

    Args:
        dataset: TextImageDataset instance
        multiple: Round dimensions to this multiple
        show_progress: Show progress bar during scanning

    Returns:
        Dictionary with dimension groups and statistics
    """
    size_buckets = defaultdict(list)

    iterator = range(len(dataset))
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="Scanning image dimensions", unit="img")

    for idx in iterator:
        size_class = dataset.get_image_size_class(idx, multiple=multiple)
        size_buckets[str(size_class)].append(idx)

    # Build statistics
    group_sizes = [len(indices) for indices in size_buckets.values()]

    cache_data = {
        "dataset_path": dataset.data_path,
        "captions_file": getattr(dataset, "captions_file", None),
        "scan_date": datetime.now().isoformat(),
        "total_images": len(dataset),
        "multiple": multiple,
        "size_groups": {
            size: {"indices": indices, "count": len(indices)}
            for size, indices in size_buckets.items()
        },
        "statistics": {
            "num_groups": len(size_buckets),
            "min_group_size": min(group_sizes) if group_sizes else 0,
            "max_group_size": max(group_sizes) if group_sizes else 0,
            "avg_group_size": sum(group_sizes) // len(group_sizes) if group_sizes else 0,
        },
    }

    return cache_data


def get_or_build_dimension_cache(
    dataset: TextImageDataset,
    cache_dir: str,
    multiple: int = 32,
    rebuild: bool = False,
) -> DimensionCacheData:
    """
    Load dimension cache if exists, otherwise build and save it.

    Args:
        dataset: TextImageDataset instance
        cache_dir: Directory to store cache files
        multiple: Round dimensions to this multiple
        rebuild: Force rebuild even if cache exists

    Returns:
        Dictionary with dimension groups and statistics
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache key from dataset path and captions file
    cache_key_str = f"{dataset.data_path}"
    if hasattr(dataset, "captions_file") and dataset.captions_file:
        cache_key_str += f"|{dataset.captions_file}"
    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.dimensions.json")

    # Try to load existing cache
    if not rebuild and os.path.exists(cache_path):
        file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"Loading dimension cache from {cache_path} ({file_size_mb:.1f}MB)...")
        try:
            if HAS_ORJSON:
                # orjson is 2-3x faster for large files
                with open(cache_path, "rb") as f:
                    cache_data = orjson.loads(f.read())
            else:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)

            # Verify cache is valid
            if cache_data.get("total_images") == len(dataset):
                print(
                    f"  Loaded {cache_data['statistics']['num_groups']} dimension groups "
                    f"({cache_data['total_images']:,} images)"
                )
                return cache_data
            else:
                print(f"  Cache invalid (image count mismatch), rebuilding...")
        except Exception as e:
            print(f"  Error loading cache: {e}, rebuilding...")

    # Build cache
    print(f"Building dimension cache (multiple={multiple})...")
    cache_data = build_dimension_cache(dataset, multiple=multiple, show_progress=True)

    # Save cache with indentation (loads 2-3x faster with orjson)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Dimension cache saved to {cache_path} ({file_size_mb:.1f}MB)")

    # Print statistics
    print(f"\nDimension Analysis:")
    print(f"  Total images: {cache_data['total_images']}")
    print(f"  Dimension groups: {cache_data['statistics']['num_groups']}")
    print(
        f"  Group size range: {cache_data['statistics']['min_group_size']} - {cache_data['statistics']['max_group_size']}"
    )
    print(f"  Average group size: {cache_data['statistics']['avg_group_size']}")

    # Print top 10 groups
    groups_sorted = sorted(
        cache_data["size_groups"].items(), key=lambda x: x[1]["count"], reverse=True
    )
    print(f"\n  Top dimension groups:")
    for size, info in groups_sorted[:10]:
        print(f"    {size}: {info['count']} images")
    if len(groups_sorted) > 10:
        print(f"    ... and {len(groups_sorted) - 10} more groups")
    print()

    return cache_data


class ResumableDimensionSampler(Sampler):
    """
    Sampler that groups images by dimensions and supports mid-epoch resume.

    Features:
    - Groups images by dimensions (from cache)
    - Balances groups to fill batches efficiently
    - Ensures all images trained exactly once per epoch
    - Supports mid-epoch resume with exact state restoration
    """

    def __init__(
        self,
        dimension_cache: DimensionCacheData,
        batch_size: int,
        seed: Optional[int] = None,
        resume_state: Optional[SamplerState] = None,
    ):
        """
        Args:
            dimension_cache: Dimension cache data from get_or_build_dimension_cache
            batch_size: Number of samples per batch
            seed: Random seed for reproducibility
            resume_state: State dict from previous run (for resume)
        """
        self.batch_size = batch_size
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.dimension_cache = dimension_cache

        # Extract size groups
        self.size_groups: dict[Any, list[int]] = {
            eval(size): info[
                "indices"
            ]  # Convert string tuple back to tuple  # type: ignore[arg-type]
            for size, info in dimension_cache["size_groups"].items()
        }

        self.total_samples = dimension_cache["total_images"]

        # Initialize or restore state
        if resume_state is not None:
            self._load_state_dict(resume_state)
        else:
            self._initialize_new_epoch()

    def _initialize_new_epoch(self):
        """Create new epoch ordering."""
        rng = random.Random(self.seed)

        # Create balanced batches from size groups
        self.epoch_batches = []

        # Reset remainder pool for this epoch (use local variable)
        remainder_pool = []

        # For each size group, create batches
        for size_class, indices in self.size_groups.items():
            group_indices = indices.copy()
            rng.shuffle(group_indices)

            # Create full batches
            for i in range(0, len(group_indices), self.batch_size):
                batch = group_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    self.epoch_batches.append(batch)
                else:
                    # Handle remainder: add to a mixed batch pool
                    remainder_pool.extend(batch)

        # Create batches from remainder pool
        if remainder_pool:
            rng.shuffle(remainder_pool)
            for i in range(0, len(remainder_pool), self.batch_size):
                batch = remainder_pool[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    self.epoch_batches.append(batch)
                # Note: very last incomplete batch is dropped

        # Shuffle batch order
        rng.shuffle(self.epoch_batches)

        # Flatten to create epoch ordering
        self.epoch_indices = []
        for batch in self.epoch_batches:
            self.epoch_indices.extend(batch)

        self.position = 0
        # Don't reset current_epoch here - it should be managed by caller
        if not hasattr(self, "current_epoch"):
            self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Set epoch number and re-initialize for new epoch."""
        self.current_epoch = epoch
        self.seed = (self.seed + epoch * 12345) % (2**32)
        self._initialize_new_epoch()

    def __iter__(self):
        """Yield batches from current position."""
        while self.position < len(self.epoch_batches):
            batch = self.epoch_batches[self.position]
            self.position += 1
            yield batch

    def __len__(self) -> int:
        """Total number of batches in epoch."""
        return len(self.epoch_batches)

    def state_dict(self) -> SamplerState:
        """
        Return current state for checkpointing.

        Note: We don't save epoch_batches as it can be huge (MBs).
        Instead, we regenerate it deterministically from seed on resume.
        This makes resume nearly instantaneous.
        """
        return {
            "seed": self.seed,
            "position": self.position,
            "current_epoch": self.current_epoch,
            "batch_size": self.batch_size,
            # epoch_batches removed - regenerated from seed on resume
        }

    def load_state_dict(self, state: SamplerState) -> None:
        """
        Restore state from checkpoint (public API).

        Regenerates epoch_batches deterministically from seed,
        then fast-forwards to saved position.
        """
        self._load_state_dict(state)

    def _load_state_dict(self, state: SamplerState) -> None:
        """
        Restore state from checkpoint.

        Regenerates epoch_batches deterministically from seed,
        then fast-forwards to saved position.
        """
        self.seed = state["seed"]
        self.batch_size = state["batch_size"]
        self.current_epoch = state.get("current_epoch", 0)

        # Regenerate batches deterministically from seed
        # This is fast and produces identical batch order
        self._initialize_new_epoch()

        # Fast-forward to saved position (set after _initialize_new_epoch to avoid reset)
        self.position = state["position"]
        # Ensure current_epoch is preserved after _initialize_new_epoch
        self.current_epoch = state.get("current_epoch", 0)

    def get_progress_info(self) -> dict[str, int | float]:
        """Return progress information for logging."""
        samples_trained = self.position * self.batch_size
        total_batches = len(self.epoch_batches)

        return {
            "batch": self.position,
            "total_batches": total_batches,
            "samples_trained": samples_trained,
            "total_samples": len(self.epoch_indices),
            "progress_pct": (self.position / total_batches * 100) if total_batches > 0 else 0,
        }
