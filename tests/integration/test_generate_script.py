"""Integration tests for generate.py script.

Tests the generation script setup and basic functionality.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Add scripts to path
scripts_path = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))


@pytest.mark.slow
class TestGenerateScriptSetup:
    """Tests for generate script setup and initialization."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_args(self, temp_output_dir):
        """Create mock arguments for generation."""
        args = MagicMock()
        args.checkpoint = None
        args.prompt = "a beautiful sunset over mountains"
        args.negative_prompt = ""
        args.width = 512
        args.height = 512
        args.num_images = 1
        args.steps = 20  # Fewer steps for testing
        args.guidance_scale = 7.5
        args.seed = 42
        args.output_dir = temp_output_dir
        args.vae_dim = 32  # Small for testing
        args.text_embedding_dim = 64  # Small for testing
        args.feature_maps_dim = 32  # Small for testing
        args.tokenizer_name = "distilbert-base-uncased"
        args.channels = 3
        args.pretrained_bert_model = None
        return args

    def test_generation_pipeline_initialization(self, mock_args):
        """Test generation pipeline components can be initialized."""
        from fluxflow.models import (
            BertTextEncoder,
            FluxCompressor,
            FluxExpander,
            FluxFlowProcessor,
            FluxPipeline,
        )

        device = torch.device("cpu")

        # Create models with small dimensions for testing
        compressor = FluxCompressor(d_model=mock_args.vae_dim, use_attention=False)
        expander = FluxExpander(d_model=mock_args.vae_dim)
        flow_processor = FluxFlowProcessor(
            d_model=mock_args.feature_maps_dim, vae_dim=mock_args.vae_dim
        )
        text_encoder = BertTextEncoder(embed_dim=mock_args.text_embedding_dim)

        # Create pipeline
        pipeline = FluxPipeline(compressor, flow_processor, expander)

        pipeline.to(device)
        text_encoder.to(device)

        # Verify models are on correct device
        assert next(pipeline.parameters()).device == device
        assert next(text_encoder.parameters()).device == device

    def test_tokenizer_initialization(self, mock_args):
        """Test tokenizer can be loaded and used."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            mock_args.tokenizer_name, cache_dir="./_cache", local_files_only=False
        )

        # Handle pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Test tokenization
        prompt = mock_args.prompt
        tokens = tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )

        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 1
        assert tokens["input_ids"].shape[1] <= 77


@pytest.mark.slow
class TestGenerateScriptGeneration:
    """Tests for actual generation functionality."""

    @pytest.fixture
    def small_pipeline(self):
        """Create a small pipeline for testing."""
        from fluxflow.models import FluxCompressor, FluxExpander, FluxFlowProcessor, FluxPipeline

        device = torch.device("cpu")
        compressor = FluxCompressor(d_model=32, use_attention=False).to(device)
        expander = FluxExpander(d_model=32).to(device)
        flow_processor = FluxFlowProcessor(d_model=32, vae_dim=32).to(device)

        pipeline = FluxPipeline(compressor, flow_processor, expander)
        return pipeline.to(device)

    @pytest.fixture
    def small_text_encoder(self):
        """Create a small text encoder for testing."""
        from fluxflow.models import BertTextEncoder

        device = torch.device("cpu")
        text_encoder = BertTextEncoder(embed_dim=64)
        return text_encoder.to(device)

    def test_text_encoding(self, small_text_encoder):
        """Test text encoding produces correct output shape."""
        from transformers import AutoTokenizer

        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Tokenize prompt
        prompt = "a test prompt"
        tokens = tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Encode text
        with torch.no_grad():
            text_embeddings = small_text_encoder(input_ids, attention_mask=attention_mask)

        # Check output shape - BertTextEncoder returns pooled embeddings
        assert text_embeddings.ndim == 2  # [batch, embed_dim]
        assert text_embeddings.shape[0] == 1  # batch size
        assert text_embeddings.shape[1] == 64  # embed_dim

    def test_vae_encode_decode(self, small_pipeline):
        """Test VAE can encode and decode images."""
        device = torch.device("cpu")

        # Create a small test image
        test_image = torch.randn(1, 3, 64, 64).to(device)

        # Encode
        with torch.no_grad():
            latent_packet = small_pipeline.compressor(test_image)

        # Check latent shape
        assert latent_packet.ndim == 3  # [batch, seq_len, dim]
        assert latent_packet.shape[0] == 1  # batch size

        # Decode
        with torch.no_grad():
            reconstructed = small_pipeline.expander(latent_packet)

        # Check output shape matches input
        assert reconstructed.shape == test_image.shape

    @pytest.mark.skip(
        reason="Requires dimension-matched models - complex setup for small test models"
    )
    @pytest.mark.parametrize(
        "width,height",
        [
            (64, 64),
            (64, 128),
            (128, 64),
        ],
    )
    def test_generation_different_sizes(self, small_pipeline, small_text_encoder, width, height):
        """Test generation works with different image sizes."""
        from diffusers import DPMSolverMultistepScheduler
        from transformers import AutoTokenizer

        device = torch.device("cpu")

        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased", cache_dir="./_cache", local_files_only=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Tokenize prompt
        prompt = "test image"
        tokens = tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Encode text
        with torch.no_grad():
            text_embeddings = small_text_encoder(input_ids, attention_mask=attention_mask)

        # Create scheduler
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(5)  # Very few steps for testing

        # Generate initial noise at target size
        # For small models, use reduced resolution
        test_image = torch.randn(1, 3, width, height).to(device)

        # Encode to latent space
        with torch.no_grad():
            latent_packet = small_pipeline.compressor(test_image)

        # Extract components
        img_seq = latent_packet[:, :-1, :].contiguous()
        hw_vec = latent_packet[:, -1:, :].contiguous()

        # Test one denoising step
        noise = torch.randn_like(img_seq)
        timestep = torch.tensor([500], device=device)

        # Add noise
        noisy_latent = scheduler.add_noise(img_seq, noise, timestep)
        full_input = torch.cat([noisy_latent, hw_vec], dim=1)

        # Predict
        with torch.no_grad():
            prediction = small_pipeline.flow_processor(full_input, text_embeddings, timestep)

        # Check prediction shape
        assert prediction.shape[0] == 1  # batch size
        assert prediction.shape[1] == latent_packet.shape[1]  # seq_len (including hw_vec)
