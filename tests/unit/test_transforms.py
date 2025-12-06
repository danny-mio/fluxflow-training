"""Unit tests for data transforms (src/data/transforms.py)."""

import numpy as np
import pytest
import torch
from PIL import Image

from fluxflow_training.data.transforms import (
    collate_fn_generate,
    collate_fn_variable,
    generate_reduced_versions,
    resize_preserving_aspect_min_distortion,
    upscale_image,
)


class TestResizePreservingAspect:
    """Tests for resize_preserving_aspect_min_distortion function."""

    def test_returns_pil_image(self):
        """Should return PIL Image."""
        img = Image.new("RGB", (100, 100))
        result = resize_preserving_aspect_min_distortion(img, min_size=64, max_size=512)
        assert isinstance(result, Image.Image)

    def test_dimensions_multiple_of_16(self):
        """Output dimensions should be multiples of 16."""
        img = Image.new("RGB", (123, 456))
        result = resize_preserving_aspect_min_distortion(img, min_size=128, max_size=512)

        w, h = result.size
        assert w % 16 == 0
        assert h % 16 == 0

    def test_preserves_aspect_ratio(self):
        """Should preserve aspect ratio with minimal distortion."""
        img = Image.new("RGB", (800, 600))  # 4:3 aspect ratio
        result = resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        w, h = result.size
        original_ratio = 800 / 600
        new_ratio = w / h

        # Should be close to original ratio
        assert abs(new_ratio - original_ratio) < 0.1

    def test_respects_min_size(self):
        """Output dimensions should be >= min_size."""
        img = Image.new("RGB", (1000, 1000))
        min_size = 256
        result = resize_preserving_aspect_min_distortion(img, min_size=min_size, max_size=512)

        w, h = result.size
        # After rounding to 16, should be >= min_size rounded
        min_rounded = int(np.ceil(min_size / 16)) * 16
        assert w >= min_rounded or h >= min_rounded

    def test_respects_max_size(self):
        """Output dimensions should be <= max_size."""
        img = Image.new("RGB", (2000, 2000))
        max_size = 512
        result = resize_preserving_aspect_min_distortion(img, min_size=256, max_size=max_size)

        w, h = result.size
        # After rounding to 16, should be <= max_size rounded
        max_rounded = int(np.floor(max_size / 16)) * 16
        assert w <= max_rounded and h <= max_rounded

    def test_uses_cache(self):
        """Should use cache for repeated sizes."""
        img = Image.new("RGB", (300, 200))

        # Clear cache
        if hasattr(resize_preserving_aspect_min_distortion, "_cache"):
            resize_preserving_aspect_min_distortion._cache.clear()  # type: ignore

        # First call
        result1 = resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        # Check cache was populated
        cache = resize_preserving_aspect_min_distortion._cache  # type: ignore
        assert len(cache) > 0

        # Second call with same image size
        result2 = resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        # Should use cached result
        assert result1.size == result2.size

    def test_no_resize_if_already_correct(self):
        """Should return same image if already correct size."""
        # Create image that's already a valid size (multiple of 16)
        img = Image.new("RGB", (256, 256))
        result = resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        # Should be same size (no resize needed)
        assert result.size == img.size

    def test_square_image(self):
        """Should handle square images correctly."""
        img = Image.new("RGB", (500, 500))
        result = resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        w, h = result.size
        # Square should remain square
        assert w == h

    def test_landscape_image(self):
        """Should handle landscape (wide) images."""
        img = Image.new("RGB", (800, 600))
        result = resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        w, h = result.size
        # Should remain landscape
        assert w > h

    def test_portrait_image(self):
        """Should handle portrait (tall) images."""
        img = Image.new("RGB", (600, 800))
        result = resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        w, h = result.size
        # Should remain portrait
        assert h > w

    def test_raises_on_invalid_range(self):
        """Should raise error if min > max after rounding."""
        img = Image.new("RGB", (100, 100))

        # min=500 rounds to 512, max=500 rounds to 496, invalid
        with pytest.raises(ValueError, match="Invalid size range"):
            resize_preserving_aspect_min_distortion(img, min_size=500, max_size=500)


class TestUpscaleImage:
    """Tests for upscale_image function."""

    def test_returns_list_of_images(self):
        """Should return list of PIL Images."""
        img = Image.new("RGB", (256, 256))
        result = upscale_image(img=img)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(i, Image.Image) for i in result)

    def test_loads_from_file(self, temp_dir):
        """Should load image from filename."""
        # Create test image file
        img_path = temp_dir / "test.jpg"
        img = Image.new("RGB", (256, 256))
        img.save(img_path)

        result = upscale_image(filename=str(img_path))

        assert len(result) > 0
        assert isinstance(result[0], Image.Image)

    def test_requires_img_or_filename(self):
        """Should raise error if neither img nor filename provided."""
        with pytest.raises(ValueError, match="Either img or filename must be provided"):
            upscale_image()

    def test_converts_to_rgb(self):
        """Should convert images to RGB mode."""
        # Create RGBA image
        img = Image.new("RGBA", (256, 256), (255, 0, 0, 128))
        result = upscale_image(img=img)

        # All returned images should be RGB
        for img_scale in result:
            assert img_scale.mode == "RGB"

    def test_loads_upscaled_cache_if_exists(self, temp_dir):
        """Should load upscaled version if ups_ file exists."""
        # Create original image
        img_path = temp_dir / "test.jpg"
        img = Image.new("RGB", (256, 256), (255, 0, 0))
        img.save(img_path)

        # Create upscaled version
        upscaled_path = temp_dir / "ups_test.webp"
        upscaled = Image.new("RGB", (512, 512), (0, 255, 0))
        upscaled.save(upscaled_path)

        # Should use upscaled version
        result = upscale_image(filename=str(img_path))

        # Last image should be from upscaled file (larger)
        largest = result[-1]
        assert max(largest.size) >= 512

    def test_multi_scale_output(self):
        """Should return images at multiple scales."""
        img = Image.new("RGB", (512, 512))
        result = upscale_image(img=img)

        # Should have at least one scale
        assert len(result) >= 1

        # Sizes should be in increasing order
        if len(result) > 1:
            sizes = [min(img.size) for img in result]
            assert sizes == sorted(sizes)


class TestCollateFnVariable:
    """Tests for collate_fn_variable function."""

    def test_returns_correct_structure(self, temp_dir):
        """Should return (images, captions) tuple."""
        # Create test images
        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"
        Image.new("RGB", (128, 128)).save(img1_path)
        Image.new("RGB", (128, 128)).save(img2_path)

        # Create test data
        data = [
            (torch.tensor([1, 2, 3]), str(img1_path)),
            (torch.tensor([4, 5]), str(img2_path)),
        ]

        images, captions = collate_fn_variable(data, channels=3, img_size=128)

        # Images should be list of tensors
        assert isinstance(images, list)
        assert len(images) > 0
        assert all(isinstance(img, torch.Tensor) for img in images)

        # Captions should be padded tensor
        assert isinstance(captions, torch.Tensor)
        assert captions.shape[0] == 2  # Batch size

    def test_pads_captions(self, temp_dir):
        """Should pad captions to same length."""
        img_path = temp_dir / "img.jpg"
        Image.new("RGB", (128, 128)).save(img_path)

        # Different length captions
        data = [
            (torch.tensor([1, 2, 3, 4, 5]), str(img_path)),
            (torch.tensor([6, 7]), str(img_path)),
            (torch.tensor([8, 9, 10]), str(img_path)),
        ]

        _, captions = collate_fn_variable(data, channels=3, img_size=128)

        # Should all be same length (longest = 5)
        assert captions.shape == (3, 5)

        # Padding should be 0
        assert captions[1, 2] == 0  # Second caption padded
        assert captions[1, 3] == 0
        assert captions[1, 4] == 0

    def test_normalizes_images(self, temp_dir):
        """Should normalize images to [-1, 1]."""
        img_path = temp_dir / "img.jpg"
        Image.new("RGB", (128, 128), (255, 255, 255)).save(img_path)

        data = [(torch.tensor([1, 2]), str(img_path))]

        images, _ = collate_fn_variable(data, channels=3, img_size=128)

        # Check normalization (white pixels should be ~1.0)
        img_tensor = images[0][0]  # First scale, first batch item
        # Mean should be around 1.0 for white image normalized with mean=0.5, std=0.5
        assert img_tensor.mean().item() > 0.9

    def test_handles_pil_images(self):
        """Should handle PIL Image objects directly."""
        img1 = Image.new("RGB", (128, 128))
        img2 = Image.new("RGB", (128, 128))

        data = [
            (torch.tensor([1, 2, 3]), img1),
            (torch.tensor([4, 5]), img2),
        ]

        images, captions = collate_fn_variable(data, channels=3, img_size=128)

        assert len(images) > 0
        assert captions.shape[0] == 2

    def test_batches_images_correctly(self, temp_dir):
        """Should batch images at each scale."""
        img_path = temp_dir / "img.jpg"
        Image.new("RGB", (256, 256)).save(img_path)

        data = [
            (torch.tensor([1, 2]), str(img_path)),
            (torch.tensor([3, 4]), str(img_path)),
        ]

        images, _ = collate_fn_variable(data, channels=3, img_size=256)

        # Each scale should have batch dimension
        for img_scale in images:
            assert img_scale.shape[0] == 2  # Batch size

    def test_handles_mismatched_sizes(self, temp_dir):
        """Should resize mismatched images to most common size."""
        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"
        Image.new("RGB", (128, 128)).save(img1_path)
        Image.new("RGB", (144, 144)).save(img2_path)  # Slightly different

        data = [
            (torch.tensor([1, 2]), str(img1_path)),
            (torch.tensor([3, 4]), str(img2_path)),
        ]

        # Should not crash
        images, _ = collate_fn_variable(data, channels=3, img_size=128)

        # All images at each scale should have same size
        for img_scale in images:
            batch_size = img_scale.shape[0]
            for i in range(batch_size):
                assert img_scale[i].shape == img_scale[0].shape


class TestCollateFnGenerate:
    """Tests for collate_fn_generate function."""

    def test_returns_correct_structure(self):
        """Should return (file_names, captions) tuple."""
        data = [
            ("output1.png", torch.tensor([1, 2, 3])),
            ("output2.png", torch.tensor([4, 5])),
        ]

        file_names, captions = collate_fn_generate(data)

        # File names should be tuple
        assert isinstance(file_names, tuple)
        assert len(file_names) == 2

        # Captions should be padded tensor
        assert isinstance(captions, torch.Tensor)
        assert captions.shape[0] == 2

    def test_preserves_file_names(self):
        """Should preserve file names in order."""
        data = [
            ("img_a.png", torch.tensor([1])),
            ("img_b.png", torch.tensor([2])),
            ("img_c.png", torch.tensor([3])),
        ]

        file_names, _ = collate_fn_generate(data)

        assert file_names == ("img_a.png", "img_b.png", "img_c.png")

    def test_pads_captions(self):
        """Should pad captions to same length."""
        data = [
            ("out1.png", torch.tensor([1, 2, 3, 4, 5])),
            ("out2.png", torch.tensor([6, 7])),
            ("out3.png", torch.tensor([8])),
        ]

        _, captions = collate_fn_generate(data)

        # Should all be same length (longest = 5)
        assert captions.shape == (3, 5)

        # Check padding
        assert captions[1, 2] == 0  # Padded
        assert captions[2, 1] == 0  # Padded

    def test_batch_first(self):
        """Captions should be batch_first format."""
        data = [
            ("out.png", torch.tensor([1, 2, 3])),
            ("out2.png", torch.tensor([4, 5, 6])),
        ]

        _, captions = collate_fn_generate(data)

        # Shape should be [batch, seq_len]
        assert captions.shape == (2, 3)

    def test_padding_value_zero(self):
        """Padding should use value 0."""
        data = [
            ("out1.png", torch.tensor([1, 2, 3])),
            ("out2.png", torch.tensor([4])),
        ]

        _, captions = collate_fn_generate(data)

        # Second item should be padded with zeros
        assert captions[1, 1] == 0
        assert captions[1, 2] == 0

    def test_single_item_batch(self):
        """Should handle single-item batch."""
        data = [("single.png", torch.tensor([1, 2, 3, 4]))]

        file_names, captions = collate_fn_generate(data)

        assert len(file_names) == 1
        assert captions.shape == (1, 4)

    def test_empty_caption(self):
        """Should handle empty caption tensors."""
        data = [
            ("out1.png", torch.tensor([1, 2])),
            ("out2.png", torch.tensor([])),  # Empty
        ]

        _, captions = collate_fn_generate(data)

        # Should pad to longest (2)
        assert captions.shape == (2, 2)


class TestTransformsIntegration:
    """Integration tests combining multiple transform functions."""

    def test_full_training_pipeline(self, temp_dir):
        """Test complete training data pipeline."""
        # Create test images
        img1_path = temp_dir / "train1.jpg"
        img2_path = temp_dir / "train2.jpg"
        Image.new("RGB", (300, 200)).save(img1_path)
        Image.new("RGB", (400, 300)).save(img2_path)

        # Simulate dataloader batch
        data = [
            (torch.tensor([1, 2, 3, 4]), str(img1_path)),
            (torch.tensor([5, 6]), str(img2_path)),
        ]

        # Process batch
        images, captions = collate_fn_variable(data, channels=3, img_size=256)

        # Verify output
        assert len(images) > 0  # At least one scale
        assert captions.shape[0] == 2  # Batch size
        assert all(img.shape[0] == 2 for img in images)  # All scales batched

    def test_full_generation_pipeline(self):
        """Test complete generation pipeline."""
        # Simulate generation batch
        data = [
            ("generated_a.png", torch.tensor([10, 20, 30])),
            ("generated_b.png", torch.tensor([40, 50])),
            ("generated_c.png", torch.tensor([60, 70, 80, 90])),
        ]

        # Process batch
        file_names, captions = collate_fn_generate(data)

        # Verify output
        assert len(file_names) == 3
        assert captions.shape[0] == 3
        assert captions.shape[1] == 4  # Longest caption

    def test_resize_cache_performance(self):
        """Test that resize cache improves performance."""
        img = Image.new("RGB", (1234, 5678))

        # Clear cache
        if hasattr(resize_preserving_aspect_min_distortion, "_cache"):
            resize_preserving_aspect_min_distortion._cache.clear()  # type: ignore

        # First call (no cache)
        resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        # Second call (with cache) should be faster
        # Just verify it doesn't crash
        resize_preserving_aspect_min_distortion(img, min_size=256, max_size=512)

        # Cache should have entry
        cache = resize_preserving_aspect_min_distortion._cache  # type: ignore
        assert len(cache) > 0


class TestGenerateReducedVersions:
    """Tests for generate_reduced_versions function."""

    def test_returns_list_of_images(self):
        """Should return list of PIL Images."""
        img = Image.new("RGB", (1024, 768))
        result = generate_reduced_versions(img, [128, 256, 512])
        assert isinstance(result, list)
        assert all(isinstance(r, Image.Image) for r in result)

    def test_creates_correct_number_of_versions(self):
        """Should create versions only for sizes smaller than image min dimension."""
        # Image with min dimension 768
        img = Image.new("RGB", (1024, 768))
        result = generate_reduced_versions(img, [128, 256, 512])
        # All 3 sizes are < 768, so 3 versions
        assert len(result) == 3

    def test_skips_sizes_larger_than_image(self):
        """Should skip sizes larger than or equal to image min dimension."""
        # Image with min dimension 380
        img = Image.new("RGB", (512, 380))
        result = generate_reduced_versions(img, [128, 256, 512])
        # Only 128 and 256 are < 380
        assert len(result) == 2

    def test_returns_empty_for_small_image(self):
        """Should return empty list if image is smaller than all target sizes."""
        img = Image.new("RGB", (64, 48))
        result = generate_reduced_versions(img, [128, 256, 512])
        assert len(result) == 0

    def test_preserves_aspect_ratio(self):
        """Should preserve aspect ratio in reduced versions."""
        img = Image.new("RGB", (1024, 512))  # 2:1 aspect ratio
        result = generate_reduced_versions(img, [256])

        w, h = result[0].size
        original_ratio = 1024 / 512
        new_ratio = w / h
        # Allow some tolerance due to rounding to multiples of 16
        assert abs(new_ratio - original_ratio) < 0.2

    def test_dimensions_multiple_of_16(self):
        """Output dimensions should be multiples of 16."""
        img = Image.new("RGB", (1000, 750))
        result = generate_reduced_versions(img, [256])

        w, h = result[0].size
        assert w % 16 == 0
        assert h % 16 == 0

    def test_sorted_by_size(self):
        """Results should be sorted by size (smallest first)."""
        img = Image.new("RGB", (1024, 1024))
        result = generate_reduced_versions(img, [512, 128, 256])

        # Check sizes are increasing
        sizes = [min(r.size) for r in result]
        assert sizes == sorted(sizes)

    def test_portrait_orientation(self):
        """Should handle portrait images correctly."""
        img = Image.new("RGB", (512, 1024))  # Height is larger
        result = generate_reduced_versions(img, [256])

        w, h = result[0].size
        # Width should be the smaller dimension
        assert w < h

    def test_with_real_image_content(self, temp_dir):
        """Test with actual image file to ensure data integrity."""
        # Create a simple test image with some content
        img = Image.new("RGB", (800, 600), color=(255, 128, 64))
        result = generate_reduced_versions(img, [128, 256])

        assert len(result) == 2
        # Verify smallest version
        assert min(result[0].size) <= 128 + 16  # Allow for rounding


class TestCollateWithReducedVersions:
    """Tests for collate_fn_variable with reduced_min_sizes."""

    def test_includes_reduced_versions(self, temp_dir):
        """Should include reduced versions when reduced_min_sizes is provided."""
        # Create a large test image
        img_path = temp_dir / "large.jpg"
        Image.new("RGB", (1024, 1024)).save(img_path)

        data = [(torch.tensor([1, 2, 3]), str(img_path))]

        # With reduced versions
        images, _ = collate_fn_variable(
            data, channels=3, img_size=1024, reduced_min_sizes=[128, 256, 512]
        )

        # Should have reduced versions + original (via upscale_image)
        # 3 reduced + at least 1 from upscale
        assert len(images) >= 4

    def test_without_reduced_versions(self, temp_dir):
        """Should work normally when reduced_min_sizes is None."""
        img_path = temp_dir / "test.jpg"
        Image.new("RGB", (512, 512)).save(img_path)

        data = [(torch.tensor([1, 2]), str(img_path))]

        # Without reduced versions (default behavior)
        images, _ = collate_fn_variable(data, channels=3, img_size=512, reduced_min_sizes=None)

        # Should just have upscale versions
        assert len(images) >= 1

    def test_small_image_no_reduced(self, temp_dir):
        """Small images should not generate reduced versions."""
        img_path = temp_dir / "small.jpg"
        Image.new("RGB", (64, 64)).save(img_path)

        data = [(torch.tensor([1]), str(img_path))]

        images, _ = collate_fn_variable(
            data, channels=3, img_size=64, reduced_min_sizes=[128, 256, 512]
        )

        # No reduced versions (image too small), just upscale
        assert len(images) >= 1
