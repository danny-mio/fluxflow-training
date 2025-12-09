"""Image preprocessing and collate functions for FluxFlow."""

import math
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

try:
    resample_filter: Any = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:
    resample_filter = Image.LANCZOS  # type: ignore[attr-defined, misc]


def resize_preserving_aspect_min_distortion(
    image: Image.Image, min_size: int = 256, max_size: int = 512
) -> Image.Image:
    """
    Resize image to nearest multiple of 16 with minimal aspect ratio distortion.

    Uses a cache to avoid recomputing optimal dimensions for repeated sizes.

    Args:
        image: PIL Image to resize
        min_size: Minimum dimension (rounded to multiple of 16)
        max_size: Maximum dimension (rounded to multiple of 16)

    Returns:
        Resized PIL Image with dimensions as multiples of 16
    """
    orig_w, orig_h = image.size
    key = (orig_w, orig_h, min_size, max_size)

    if not hasattr(resize_preserving_aspect_min_distortion, "_cache"):
        resize_preserving_aspect_min_distortion._cache = {}  # type: ignore

    cache = resize_preserving_aspect_min_distortion._cache  # type: ignore

    if key in cache:
        best_w, best_h = cache[key]
    else:
        orig_ratio = orig_w / orig_h
        min_size_rounded = int(np.ceil(min_size / 16)) * 16
        max_size_rounded = int(np.floor(max_size / 16)) * 16
        if min_size_rounded > max_size_rounded:
            raise ValueError("Invalid size range after rounding to multiples of 16.")

        sizes = np.arange(min_size_rounded, max_size_rounded + 1, 16)
        widths, heights = np.meshgrid(sizes, sizes)
        ratios = widths / heights
        distortions = np.abs(np.log(ratios / orig_ratio))
        min_distortion = distortions.min()

        candidate_idxs = np.where(distortions == min_distortion)
        candidate_ws = widths[candidate_idxs]
        candidate_hs = heights[candidate_idxs]
        candidate_areas = candidate_ws * candidate_hs

        best_idx = np.argmin(candidate_areas)
        best_w = int(candidate_ws[best_idx])
        best_h = int(candidate_hs[best_idx])
        cache[key] = (best_w, best_h)

    if best_h == orig_h and best_w == orig_w:
        return image
    return image.resize((best_w, best_h), Image.LANCZOS)  # type: ignore[attr-defined, misc]


def upscale_image(
    img: Optional[Image.Image] = None, filename: Optional[str] = None
) -> List[Image.Image]:
    """
    Generate multi-scale versions of an image for progressive training.

    Loads image from file or uses provided PIL Image. Checks for cached
    upscaled versions (filename prefix 'ups_').

    Args:
        img: PIL Image (optional if filename provided)
        filename: Path to image file (optional if img provided)

    Returns:
        List of PIL Images at different scales (smallest to largest)
    """
    import os

    img_full: Image.Image
    lr_img: Image.Image

    if img is not None:
        img_full = img.convert("RGB")
        lr_img = img_full
    else:
        if filename is None:
            raise ValueError("Either img or filename must be provided")
        lr_img = Image.open(filename).convert("RGB")
        img_full = lr_img
        dirname, basename = os.path.split(filename)
        upscaled_filename = os.path.join(dirname, f"ups_{os.path.splitext(basename)[0]}.webp")
        if os.path.exists(upscaled_filename):
            img_full = Image.open(upscaled_filename).convert("RGB")

    orig_w, orig_h = img_full.size
    lr_w, lr_h = lr_img.size
    min_size = min(orig_h, orig_w)
    min_size_lr = min(lr_h, lr_w)
    max_size = (math.ceil(max(orig_h, orig_w) / 16)) * 16
    ratio = max_size / min_size

    sizes = [min_size_lr]
    if min_size_lr < min_size:
        sizes.append(min_size)

    imgs = [
        resize_preserving_aspect_min_distortion(
            lr_img if size <= min_size_lr else img_full,
            size,
            max(size, (math.ceil(size * ratio / 16)) * 16),
        )
        for size in sizes
    ]
    return imgs


def generate_reduced_versions(
    image: Image.Image, reduced_min_sizes: List[int]
) -> List[Image.Image]:
    """
    Generate reduced versions of an image based on configured min sizes.

    For each size in reduced_min_sizes that is smaller than the image's minimum
    dimension, creates a Lanczos-downscaled version where the smaller dimension
    matches that size.

    Args:
        image: Original PIL Image
        reduced_min_sizes: List of target minimum sizes (e.g., [128, 256, 512])

    Returns:
        List of reduced PIL Images (sorted by size, smallest first)
        Does not include the original image.

    Example:
        If image is 1024x768 (min=768) and reduced_min_sizes=[128, 256, 512]:
        Returns 3 images with min dimensions 128, 256, 512

        If image is 380x512 (min=380) and reduced_min_sizes=[128, 256, 512]:
        Returns 2 images with min dimensions 128, 256
        (512 is skipped because 380 < 512)
    """
    orig_w, orig_h = image.size
    min_dim = min(orig_w, orig_h)
    aspect_ratio = orig_w / orig_h

    reduced_images = []

    # Sort sizes to ensure consistent ordering (smallest first)
    sorted_sizes = sorted(reduced_min_sizes)

    for target_min in sorted_sizes:
        # Only create reduced version if image is larger than target
        if min_dim > target_min:
            # Calculate new dimensions preserving aspect ratio
            if orig_w <= orig_h:
                # Width is the smaller dimension
                new_w = target_min
                new_h = int(round(target_min / aspect_ratio))
            else:
                # Height is the smaller dimension
                new_h = target_min
                new_w = int(round(target_min * aspect_ratio))

            # Ensure dimensions are multiples of 16 for model compatibility
            new_w = max(16, (new_w // 16) * 16)
            new_h = max(16, (new_h // 16) * 16)

            # Resize using Lanczos for high-quality downscaling
            reduced_img = image.resize((new_w, new_h), resample_filter)
            reduced_images.append(reduced_img)

    return reduced_images


def collate_fn_variable(
    data: List[Tuple], channels: int, img_size: int, reduced_min_sizes: Optional[List[int]] = None
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Collate function for variable-size image batches with multi-scale loading.

    Args:
        data: List of (caption_ids, filename_or_image) tuples
        channels: Number of image channels (3 for RGB)
        img_size: Reserved parameter (currently unused)
        reduced_min_sizes: Optional list of min sizes for reduced versions
            (e.g., [128, 256, 512] to train on smaller versions too)

    Returns:
        Tuple of (multi_scale_images, captions):
            - multi_scale_images: List of batched tensors [B, C, H, W] per scale
            - captions: Padded token IDs [B, seq_len]
    """
    captions, filenames_or_images = zip(*data)

    # Load images
    if isinstance(filenames_or_images[0], str):
        loaded_images = [Image.open(filename).convert("RGB") for filename in filenames_or_images]
    else:
        loaded_images = [img.convert("RGB") for img in filenames_or_images]

    # For each image, collect all versions (reduced + upscaled)
    all_scale_images = []
    for idx, img in enumerate(loaded_images):
        img_versions = []

        # Add reduced versions if configured
        if reduced_min_sizes:
            reduced = generate_reduced_versions(img, reduced_min_sizes)
            img_versions.extend(reduced)

        # Add upscaled versions (existing behavior)
        filename = filenames_or_images[idx] if isinstance(filenames_or_images[idx], str) else None
        upscaled = upscale_image(img, filename=filename)
        img_versions.extend(upscaled)

        all_scale_images.append(img_versions)

    # Group by scale level
    padded_images: list[list[Any]] = []
    for img_versions in all_scale_images:
        for i, img in enumerate(img_versions):
            if len(padded_images) <= i:
                padded_images.append([])
            padded_images[i].append(img)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels),
        ]
    )

    # Convert to tensors - ensure all images at each scale are the same size
    images = []
    for imgs in padded_images:
        # Get sizes of all images at this scale
        sizes = [img.size for img in imgs]

        # If sizes differ, resize all to the most common size
        if len(set(sizes)) > 1:
            # Find most common size
            from collections import Counter

            most_common_size = Counter(sizes).most_common(1)[0][0]

            # Resize any images that don't match
            resized_imgs = []
            for img in imgs:
                if img.size != most_common_size:
                    img = img.resize(most_common_size, resample_filter)
                resized_imgs.append(img)
            imgs = resized_imgs

        tensor_images = [transform(img.convert("RGB")).contiguous() for img in imgs]
        images.append(torch.stack(tensor_images, 0))
    captions_list = [c for c in captions]
    captions_tensor = pad_sequence(captions_list, batch_first=True, padding_value=0)
    return images, captions_tensor


def collate_fn_generate(data: List[Tuple]) -> Tuple[Tuple, torch.Tensor]:
    """
    Collate function for generation mode (text-only batches).

    Args:
        data: List of (file_name, caption_ids) tuples

    Returns:
        Tuple of (file_names, captions):
            - file_names: Tuple of output filenames
            - captions: Padded token IDs [B, seq_len]
    """
    file_names, captions = zip(*data)
    captions_list = [c for c in captions]
    captions_tensor = pad_sequence(captions_list, batch_first=True, padding_value=0)
    return file_names, captions_tensor
