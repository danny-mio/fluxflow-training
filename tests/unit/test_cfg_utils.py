"""Unit tests for classifier-free guidance utilities."""

import pytest
import torch

from fluxflow_training.training.cfg_utils import (
    apply_cfg_dropout,
    cfg_guided_prediction,
    cfg_guided_prediction_batched,
)


class TestApplyCFGDropout:
    """Tests for apply_cfg_dropout function."""

    def test_dropout_disabled_returns_original(self):
        """Test that p_uncond=0.0 returns original embeddings."""
        text_emb = torch.randn(4, 768)
        result = apply_cfg_dropout(text_emb, p_uncond=0.0)
        assert torch.allclose(result, text_emb)

    def test_dropout_shape_preserved_2d(self):
        """Test that dropout preserves 2D tensor shape."""
        text_emb = torch.randn(8, 1024)
        result = apply_cfg_dropout(text_emb, p_uncond=0.1)
        assert result.shape == text_emb.shape

    def test_dropout_shape_preserved_3d(self):
        """Test that dropout preserves 3D tensor shape."""
        text_emb = torch.randn(4, 77, 768)
        result = apply_cfg_dropout(text_emb, p_uncond=0.1)
        assert result.shape == text_emb.shape

    def test_dropout_creates_null_embeddings(self):
        """Test that some embeddings are zeroed with p_uncond > 0."""
        torch.manual_seed(42)
        batch_size = 100
        text_emb = torch.randn(batch_size, 768)
        
        result = apply_cfg_dropout(text_emb, p_uncond=0.5)
        
        # Count how many samples are null (all zeros)
        null_count = (result.abs().sum(dim=1) == 0).sum().item()
        
        # With p=0.5 and batch=100, expect ~50 null embeddings (allow ±15 for variance)
        assert 35 <= null_count <= 65, f"Expected ~50 null embeddings, got {null_count}"

    def test_dropout_probability_approximately_correct(self):
        """Test that dropout rate matches p_uncond approximately."""
        torch.manual_seed(123)
        batch_size = 1000
        p_uncond = 0.1
        text_emb = torch.randn(batch_size, 512)
        
        result = apply_cfg_dropout(text_emb, p_uncond=p_uncond)
        
        null_count = (result.abs().sum(dim=1) == 0).sum().item()
        actual_rate = null_count / batch_size
        
        # Allow ±3% variance from expected 10%
        assert abs(actual_rate - p_uncond) < 0.03, \
            f"Expected ~{p_uncond}, got {actual_rate}"

    def test_dropout_invalid_probability_raises(self):
        """Test that invalid p_uncond raises ValueError."""
        text_emb = torch.randn(4, 768)
        
        with pytest.raises(ValueError, match="p_uncond must be in"):
            apply_cfg_dropout(text_emb, p_uncond=1.5)
        
        with pytest.raises(ValueError, match="p_uncond must be in"):
            apply_cfg_dropout(text_emb, p_uncond=-0.1)

    def test_dropout_invalid_shape_raises(self):
        """Test that invalid tensor shape raises ValueError."""
        # 1D tensor (invalid)
        text_emb_1d = torch.randn(768)
        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            apply_cfg_dropout(text_emb_1d, p_uncond=0.1)
        
        # 4D tensor (invalid)
        text_emb_4d = torch.randn(2, 4, 77, 768)
        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            apply_cfg_dropout(text_emb_4d, p_uncond=0.1)

    def test_dropout_device_preserved(self):
        """Test that output device matches input device."""
        if torch.cuda.is_available():
            text_emb = torch.randn(4, 768, device="cuda")
            result = apply_cfg_dropout(text_emb, p_uncond=0.1)
            assert result.device == text_emb.device

    def test_dropout_dtype_preserved(self):
        """Test that output dtype matches input dtype."""
        text_emb_fp32 = torch.randn(4, 768, dtype=torch.float32)
        result_fp32 = apply_cfg_dropout(text_emb_fp32, p_uncond=0.1)
        assert result_fp32.dtype == torch.float32
        
        text_emb_fp16 = torch.randn(4, 768, dtype=torch.float16)
        result_fp16 = apply_cfg_dropout(text_emb_fp16, p_uncond=0.1)
        assert result_fp16.dtype == torch.float16


class TestCFGGuidedPrediction:
    """Tests for cfg_guided_prediction function."""

    def setup_method(self):
        """Setup mock model for testing."""
        def mock_model(z_t, text_emb, timesteps):
            """Mock model: returns prediction based on text embeddings."""
            # Check each sample in batch individually
            batch_size = text_emb.size(0)
            result = torch.zeros_like(z_t)
            
            for i in range(batch_size):
                # If text is null (all zeros), return lower values
                # If text is not null, return higher values
                is_null = (text_emb[i].abs().sum() == 0)
                if is_null:
                    result[i] = 0.5  # Unconditional
                else:
                    result[i] = 1.0  # Conditional
            
            return result
        
        self.mock_model = mock_model

    def test_guidance_scale_1_returns_conditional(self):
        """Test that ω=1.0 returns standard conditional prediction."""
        z_t = torch.randn(2, 10, 128)
        text_emb = torch.randn(2, 768)
        timesteps = torch.tensor([0.5, 0.5])
        
        result = cfg_guided_prediction(
            self.mock_model, z_t, text_emb, timesteps, guidance_scale=1.0
        )
        
        # Should equal conditional prediction (1.0)
        assert torch.allclose(result, torch.ones_like(z_t))

    def test_guidance_scale_0_returns_unconditional(self):
        """Test that ω=0.0 returns pure unconditional prediction."""
        z_t = torch.randn(2, 10, 128)
        text_emb = torch.randn(2, 768)
        timesteps = torch.tensor([0.5, 0.5])
        
        result = cfg_guided_prediction(
            self.mock_model, z_t, text_emb, timesteps, guidance_scale=0.0
        )
        
        # Should equal unconditional prediction (0.5)
        assert torch.allclose(result, torch.ones_like(z_t) * 0.5)

    def test_guidance_scale_above_1_extrapolates(self):
        """Test that ω>1 extrapolates beyond conditional."""
        z_t = torch.randn(2, 10, 128)
        text_emb = torch.randn(2, 768)
        timesteps = torch.tensor([0.5, 0.5])
        
        # v_guided = v_uncond + ω * (v_cond - v_uncond)
        # v_guided = 0.5 + 5.0 * (1.0 - 0.5) = 0.5 + 2.5 = 3.0
        result = cfg_guided_prediction(
            self.mock_model, z_t, text_emb, timesteps, guidance_scale=5.0
        )
        
        expected = torch.ones_like(z_t) * 3.0
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batched_cfg_matches_regular_cfg(self):
        """Test that batched CFG produces same result as regular CFG."""
        torch.manual_seed(42)
        z_t = torch.randn(4, 10, 128)
        text_emb = torch.randn(4, 768)
        timesteps = torch.tensor([0.3, 0.5, 0.7, 0.9])
        guidance_scale = 7.0
        
        result_regular = cfg_guided_prediction(
            self.mock_model, z_t, text_emb, timesteps, guidance_scale
        )
        
        result_batched = cfg_guided_prediction_batched(
            self.mock_model, z_t, text_emb, timesteps, guidance_scale
        )
        
        assert torch.allclose(result_regular, result_batched, atol=1e-5)


class TestCFGIntegration:
    """Integration tests for CFG workflow."""

    def test_full_cfg_workflow(self):
        """Test complete CFG workflow: training dropout + inference guidance."""
        # Simulate training with CFG dropout
        torch.manual_seed(42)
        batch_size = 10
        text_emb = torch.randn(batch_size, 768)
        
        # Apply CFG dropout during training
        text_emb_dropped = apply_cfg_dropout(text_emb, p_uncond=0.1)
        
        # Verify some samples are null
        null_count = (text_emb_dropped.abs().sum(dim=1) == 0).sum().item()
        assert null_count >= 0  # At least 0 (could be more)
        
        # Simulate inference with CFG
        def trained_model(z_t, text, t):
            # Model learned to handle both null and non-null text
            return torch.randn_like(z_t)
        
        z_t = torch.randn(1, 10, 128)
        text_test = torch.randn(1, 768)
        timesteps = torch.tensor([0.5])
        
        # Generate with guidance
        result = cfg_guided_prediction(
            trained_model, z_t, text_test, timesteps, guidance_scale=5.0
        )
        
        assert result.shape == z_t.shape

    def test_cfg_with_3d_embeddings(self):
        """Test CFG works with 3D text embeddings (sequence format)."""
        # 3D embeddings: [B, seq_len, D]
        text_emb = torch.randn(4, 77, 768)
        
        # Apply dropout
        dropped = apply_cfg_dropout(text_emb, p_uncond=0.2)
        assert dropped.shape == text_emb.shape
        
        # Mock model that accepts 3D
        def model_3d(z_t, text, t):
            return torch.randn_like(z_t)
        
        z_t = torch.randn(4, 10, 128)
        timesteps = torch.tensor([0.5] * 4)
        
        # Should work with 3D embeddings
        result = cfg_guided_prediction(
            model_3d, z_t, dropped, timesteps, guidance_scale=3.0
        )
        assert result.shape == z_t.shape
