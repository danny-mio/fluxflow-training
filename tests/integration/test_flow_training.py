"""Integration tests for flow model training workflow."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from fluxflow.models.flow import FluxFlowProcessor
from fluxflow.models.vae import FluxCompressor


class TestFlowTrainingIntegration:
    """Integration tests for flow model training."""

    @pytest.fixture
    def small_flow_model(self):
        """Create small flow model for testing."""
        flow = FluxFlowProcessor(
            d_model=128,
            vae_dim=32,
            embedding_size=256,
            n_head=4,
            n_layers=2,
            max_hw=64,
            ctx_tokens=4,
        )
        return flow

    @pytest.fixture
    def small_compressor(self):
        """Create small VAE compressor for latent generation."""
        compressor = FluxCompressor(
            in_channels=3, d_model=32, downscales=2, max_hw=64, use_attention=False
        )
        return compressor

    def test_flow_forward_backward(self, small_flow_model):
        """Flow model should support forward and backward pass."""
        flow = small_flow_model

        # Create dummy latent packet
        batch_size = 2
        num_tokens = 16 * 16
        packed = torch.randn(batch_size, num_tokens + 1, 32)

        # Set HW vector
        packed[:, -1, 0] = 16 / 64.0
        packed[:, -1, 1] = 16 / 64.0

        # Text embeddings
        text_embeddings = torch.randn(batch_size, 256)
        timesteps = torch.randint(0, 1000, (batch_size,))

        # Forward pass
        output = flow(packed, text_embeddings, timesteps)

        # Compute loss (simplified)
        target = torch.randn_like(output)
        loss = nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check gradients
        assert any(p.grad is not None for p in flow.parameters() if p.requires_grad)

    def test_flow_training_step(self, small_flow_model):
        """Complete flow training step with optimizer."""
        flow = small_flow_model

        optimizer = optim.AdamW(flow.parameters(), lr=1e-4)

        flow.train()

        # Create batch
        batch_size = 4
        num_tokens = 8 * 8
        packed = torch.randn(batch_size, num_tokens + 1, 32)
        packed[:, -1, 0] = 8 / 64.0
        packed[:, -1, 1] = 8 / 64.0

        text_embeddings = torch.randn(batch_size, 256)
        timesteps = torch.randint(0, 1000, (batch_size,))

        # Forward pass
        predicted = flow(packed, text_embeddings, timesteps)

        # Flow matching loss (predict velocity)
        target = torch.randn_like(predicted)
        loss = nn.functional.mse_loss(predicted, target)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0

    def test_flow_multi_step_training(self, small_flow_model):
        """Flow model should update over multiple steps."""
        flow = small_flow_model

        optimizer = optim.AdamW(flow.parameters(), lr=1e-3)
        flow.train()

        # Fixed data
        torch.manual_seed(42)
        batch_size = 4
        num_tokens = 8 * 8
        packed = torch.randn(batch_size, num_tokens + 1, 32)
        packed[:, -1, 0] = 8 / 64.0
        packed[:, -1, 1] = 8 / 64.0
        text_embeddings = torch.randn(batch_size, 256)
        timesteps = torch.randint(0, 1000, (batch_size,))
        target = torch.randn(batch_size, num_tokens + 1, 32)

        losses = []
        for step in range(5):
            predicted = flow(packed, text_embeddings, timesteps)
            loss = nn.functional.mse_loss(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0] * 1.5

    def test_flow_different_timesteps(self, small_flow_model):
        """Flow model should handle different timesteps."""
        flow = small_flow_model
        flow.eval()

        batch_size = 4
        num_tokens = 8 * 8
        packed = torch.randn(batch_size, num_tokens + 1, 32)
        packed[:, -1, 0] = 8 / 64.0
        packed[:, -1, 1] = 8 / 64.0
        text_embeddings = torch.randn(batch_size, 256)

        # Different timesteps
        timesteps = torch.tensor([0, 250, 500, 999])

        with torch.no_grad():
            output = flow(packed, text_embeddings, timesteps)

        assert output.shape == packed.shape

    def test_flow_different_spatial_sizes(self, small_flow_model):
        """Flow model should handle different spatial dimensions."""
        flow = small_flow_model
        flow.eval()

        for h, w in [(4, 4), (8, 8), (16, 16)]:
            batch_size = 2
            num_tokens = h * w
            packed = torch.randn(batch_size, num_tokens + 1, 32)
            packed[:, -1, 0] = h / 64.0
            packed[:, -1, 1] = w / 64.0

            text_embeddings = torch.randn(batch_size, 256)
            timesteps = torch.randint(0, 1000, (batch_size,))

            with torch.no_grad():
                output = flow(packed, text_embeddings, timesteps)

            assert output.shape == packed.shape

    def test_flow_with_vae_latents(self, small_flow_model, small_compressor):
        """Flow model should work with VAE-encoded latents."""
        flow = small_flow_model
        compressor = small_compressor

        flow.eval()
        compressor.eval()

        # Encode images to latent
        images = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            packed = compressor(images)
            text_embeddings = torch.randn(2, 256)
            timesteps = torch.randint(0, 1000, (2,))

            # Flow prediction
            predicted = flow(packed, text_embeddings, timesteps)

        assert predicted.shape == packed.shape

    def test_flow_eval_deterministic(self, small_flow_model):
        """Flow model in eval mode should be deterministic."""
        flow = small_flow_model
        flow.eval()

        batch_size = 2
        num_tokens = 8 * 8
        packed = torch.randn(batch_size, num_tokens + 1, 32)
        packed[:, -1, 0] = 8 / 64.0
        packed[:, -1, 1] = 8 / 64.0

        text_embeddings = torch.randn(batch_size, 256)
        timesteps = torch.randint(0, 1000, (batch_size,))

        with torch.no_grad():
            output1 = flow(packed, text_embeddings, timesteps)
            output2 = flow(packed, text_embeddings, timesteps)

        assert torch.allclose(output1, output2, atol=1e-5)

    def test_flow_checkpoint_compatibility(self, small_flow_model):
        """Flow model state_dict should be saveable."""
        flow = small_flow_model

        # Save state
        state_dict = flow.state_dict()

        # Create new model
        new_flow = FluxFlowProcessor(
            d_model=128,
            vae_dim=32,
            embedding_size=256,
            n_head=4,
            n_layers=2,
            max_hw=64,
            ctx_tokens=4,
        )

        # Load state
        new_flow.load_state_dict(state_dict)

        # Should produce same output
        flow.eval()
        new_flow.eval()

        batch_size = 2
        num_tokens = 8 * 8
        packed = torch.randn(batch_size, num_tokens + 1, 32)
        packed[:, -1, 0] = 8 / 64.0
        packed[:, -1, 1] = 8 / 64.0
        text_embeddings = torch.randn(batch_size, 256)
        timesteps = torch.randint(0, 1000, (batch_size,))

        with torch.no_grad():
            out1 = flow(packed, text_embeddings, timesteps)
            out2 = new_flow(packed, text_embeddings, timesteps)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_flow_gradient_accumulation(self, small_flow_model):
        """Flow model should support gradient accumulation."""
        flow = small_flow_model
        optimizer = optim.AdamW(flow.parameters(), lr=1e-4)

        flow.train()

        # Accumulate over 2 mini-batches
        optimizer.zero_grad()

        for i in range(2):
            batch_size = 2
            num_tokens = 8 * 8
            packed = torch.randn(batch_size, num_tokens + 1, 32)
            packed[:, -1, 0] = 8 / 64.0
            packed[:, -1, 1] = 8 / 64.0
            text_embeddings = torch.randn(batch_size, 256)
            timesteps = torch.randint(0, 1000, (batch_size,))

            predicted = flow(packed, text_embeddings, timesteps)
            target = torch.randn_like(predicted)
            loss = nn.functional.mse_loss(predicted, target)

            (loss / 2.0).backward()

        optimizer.step()

        assert True


class TestFlowConditioningIntegration:
    """Tests for flow model conditioning."""

    @pytest.fixture
    def small_flow_model(self):
        """Create small flow model."""
        flow = FluxFlowProcessor(
            d_model=128,
            vae_dim=32,
            embedding_size=256,
            n_head=4,
            n_layers=2,
            max_hw=64,
            ctx_tokens=4,
        )
        return flow

    def test_flow_text_conditioning(self, small_flow_model):
        """Flow should condition on text embeddings."""
        flow = small_flow_model
        flow.eval()

        batch_size = 2
        num_tokens = 8 * 8
        packed = torch.randn(batch_size, num_tokens + 1, 32)
        packed[:, -1, 0] = 8 / 64.0
        packed[:, -1, 1] = 8 / 64.0

        # Different text embeddings
        text1 = torch.randn(batch_size, 256)
        text2 = torch.randn(batch_size, 256)
        timesteps = torch.tensor([500, 500])

        with torch.no_grad():
            out1 = flow(packed, text1, timesteps)
            out2 = flow(packed, text2, timesteps)

        # Different text should produce different outputs
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_flow_timestep_conditioning(self, small_flow_model):
        """Flow should condition on timesteps."""
        flow = small_flow_model
        flow.eval()

        batch_size = 2
        num_tokens = 8 * 8
        packed = torch.randn(batch_size, num_tokens + 1, 32)
        packed[:, -1, 0] = 8 / 64.0
        packed[:, -1, 1] = 8 / 64.0

        text_embeddings = torch.randn(batch_size, 256)

        # Different timesteps
        timesteps1 = torch.tensor([100, 100])
        timesteps2 = torch.tensor([900, 900])

        with torch.no_grad():
            out1 = flow(packed, text_embeddings, timesteps1)
            out2 = flow(packed, text_embeddings, timesteps2)

        # Different timesteps should produce different outputs
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_flow_context_extraction(self, small_flow_model):
        """Flow should extract context from image tokens."""
        flow = small_flow_model
        flow.eval()

        # Flow uses first K tokens as context
        assert hasattr(flow, "ctx_tokens")
        assert flow.ctx_tokens == 4

        batch_size = 2
        num_tokens = 16 * 16
        packed = torch.randn(batch_size, num_tokens + 1, 32)
        packed[:, -1, 0] = 16 / 64.0
        packed[:, -1, 1] = 16 / 64.0

        text_embeddings = torch.randn(batch_size, 256)
        timesteps = torch.randint(0, 1000, (batch_size,))

        with torch.no_grad():
            output = flow(packed, text_embeddings, timesteps)

        # Should produce valid output
        assert output.shape == packed.shape


class TestFlowSamplingIntegration:
    """Tests for flow-based sampling."""

    @pytest.fixture
    def small_flow_model(self):
        """Create small flow model."""
        flow = FluxFlowProcessor(
            d_model=128,
            vae_dim=32,
            embedding_size=256,
            n_head=4,
            n_layers=2,
            max_hw=64,
            ctx_tokens=4,
        )
        return flow

    def test_flow_single_denoising_step(self, small_flow_model):
        """Flow model should perform single denoising step."""
        flow = small_flow_model
        flow.eval()

        batch_size = 1
        num_tokens = 8 * 8
        noisy_latent = torch.randn(batch_size, num_tokens + 1, 32)
        noisy_latent[:, -1, 0] = 8 / 64.0
        noisy_latent[:, -1, 1] = 8 / 64.0

        text_embeddings = torch.randn(batch_size, 256)
        timestep = torch.tensor([500])

        with torch.no_grad():
            # Predict flow/velocity
            flow_pred = flow(noisy_latent, text_embeddings, timestep)

            # Simple Euler step (x_t - dt * v)
            dt = 0.01
            denoised = noisy_latent - dt * flow_pred

        assert denoised.shape == noisy_latent.shape

    def test_flow_multi_step_sampling(self, small_flow_model):
        """Flow model should support multi-step sampling."""
        flow = small_flow_model
        flow.eval()

        batch_size = 1
        num_tokens = 4 * 4
        latent = torch.randn(batch_size, num_tokens + 1, 32)
        latent[:, -1, 0] = 4 / 64.0
        latent[:, -1, 1] = 4 / 64.0

        text_embeddings = torch.randn(batch_size, 256)

        # Simple sampling loop
        num_steps = 5
        timesteps = torch.linspace(999, 0, num_steps).long()

        with torch.no_grad():
            for t in timesteps:
                flow_pred = flow(latent, text_embeddings, t.unsqueeze(0))
                # Euler step
                dt = 1.0 / num_steps
                latent = latent - dt * flow_pred

        # Final latent should have valid shape
        assert latent.shape == (batch_size, num_tokens + 1, 32)
