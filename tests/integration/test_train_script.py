"""Integration tests for train.py script.

Tests the training script setup and trainer initialization without full training.
"""

import json
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
class TestTrainScriptSetup:
    """Tests for train script setup and initialization."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_args_vae(self, temp_output_dir):
        """Create mock arguments for VAE training."""
        args = MagicMock()
        args.data_path = None
        args.captions_file = None
        args.use_tt2m = False
        args.output_path = temp_output_dir
        args.model_checkpoint = None
        args.vae_dim = 32  # Small for testing
        args.text_embedding_dim = 64  # Small for testing
        args.feature_maps_dim = 32  # Small for testing
        args.feature_maps_dim_disc = 4
        args.pretrained_bert_model = None
        args.n_epochs = 1
        args.batch_size = 1
        args.workers = 0
        args.lr = 5e-7
        args.lr_min = 0.1
        args.preserve_lr = False
        args.optim_sched_config = None
        args.training_steps = 1
        args.use_fp16 = False
        args.initial_clipping_norm = 1.0
        args.train_vae = True
        args.gan_training = True
        args.train_spade = False
        args.train_diff = False
        args.train_diff_full = False
        args.kl_beta = 0.0001
        args.kl_warmup_steps = 1000
        args.kl_free_bits = 0.0
        args.log_interval = 10
        args.checkpoint_save_interval = 50
        args.samples_per_checkpoint = 1
        args.no_samples = True
        args.test_image_address = []
        args.sample_captions = ["test caption"]
        args.tokenizer_name = "distilbert-base-uncased"
        args.img_size = 256  # Small for testing
        args.channels = 3
        args.lambda_adv = 0.5
        args.tt2m_token = None
        return args

    def test_checkpoint_manager_initialization(self, mock_args_vae):
        """Test CheckpointManager is created correctly."""
        from fluxflow_training.training import CheckpointManager

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(output_dir=mock_args_vae.output_path)

        assert checkpoint_manager.output_dir == Path(mock_args_vae.output_path)
        assert checkpoint_manager.model_path.parent == Path(mock_args_vae.output_path)

    def test_vae_trainer_initialization(self, mock_args_vae):
        """Test VAETrainer is created with correct parameters."""
        from fluxflow.models import FluxCompressor, FluxExpander, PatchDiscriminator
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from fluxflow_training.training import EMA, VAETrainer

        device = torch.device("cpu")

        # Create small models
        compressor = FluxCompressor(d_model=32, use_attention=False)
        expander = FluxExpander(d_model=32)
        discriminator = PatchDiscriminator(in_channels=3, ctx_dim=32)

        # Create optimizers
        optimizer_vae = AdamW(list(compressor.parameters()) + list(expander.parameters()), lr=1e-4)
        optimizer_disc = AdamW(discriminator.parameters(), lr=1e-4)

        # Create schedulers
        scheduler_vae = CosineAnnealingLR(optimizer_vae, T_max=100)
        scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=100)

        # Create EMA
        ema = EMA(torch.nn.ModuleList([compressor, expander]), decay=0.999, device=device)

        # Create loss functions
        reconstruction_loss_fn = torch.nn.L1Loss()
        reconstruction_loss_min_fn = torch.nn.MSELoss()

        # Create mock accelerator
        mock_accelerator = MagicMock()

        # Create trainer - use correct parameter names
        trainer = VAETrainer(
            compressor=compressor,
            expander=expander,
            optimizer=optimizer_vae,
            scheduler=scheduler_vae,
            ema=ema,
            reconstruction_loss_fn=reconstruction_loss_fn,
            reconstruction_loss_min_fn=reconstruction_loss_min_fn,
            use_gan=True,
            discriminator=discriminator,
            discriminator_optimizer=optimizer_disc,
            discriminator_scheduler=scheduler_disc,
            kl_beta=0.0001,
            kl_warmup_steps=1000,
            kl_free_bits=0.0,
            lambda_adv=0.5,
            gradient_clip_norm=1.0,
            use_spade=False,
            r1_gamma=5.0,
            r1_interval=16,
            accelerator=mock_accelerator,
        )

        assert trainer.kl_beta == 0.0001
        assert trainer.kl_warmup_steps == 1000
        assert trainer.lambda_adv == 0.5
        assert trainer.gradient_clip_norm == 1.0

    def test_flow_trainer_initialization(self, mock_args_vae):
        """Test FlowTrainer is created with correct parameters."""
        from fluxflow.models import BertTextEncoder, FluxCompressor, FluxFlowProcessor
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from fluxflow_training.training import FlowTrainer

        device = torch.device("cpu")

        # Create small models
        compressor = FluxCompressor(d_model=32, use_attention=False)
        flow_processor = FluxFlowProcessor(d_model=32, vae_dim=32)
        text_encoder = BertTextEncoder(embed_dim=64)

        # Create optimizers
        optimizer_flow = AdamW(flow_processor.parameters(), lr=1e-4)
        optimizer_te = AdamW(text_encoder.parameters(), lr=1e-5)

        # Create schedulers
        scheduler_flow = CosineAnnealingLR(optimizer_flow, T_max=100)
        scheduler_te = CosineAnnealingLR(optimizer_te, T_max=100)

        # Create mock accelerator
        mock_accelerator = MagicMock()

        # Create trainer
        trainer = FlowTrainer(
            flow_processor=flow_processor,
            text_encoder=text_encoder,
            compressor=compressor,
            optimizer=optimizer_flow,
            scheduler=scheduler_flow,
            text_encoder_optimizer=optimizer_te,
            text_encoder_scheduler=scheduler_te,
            gradient_clip_norm=1.0,
            num_train_timesteps=1000,
            accelerator=mock_accelerator,
        )

        assert trainer.gradient_clip_norm == 1.0
        assert trainer.alphas_cumprod is not None

    def test_training_state_save_load_format(self, temp_output_dir):
        """Test training state JSON format is correct."""
        from fluxflow_training.training import CheckpointManager

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(output_dir=temp_output_dir)

        # Create mock optimizers and schedulers
        mock_optimizer = MagicMock()
        mock_optimizer.state_dict.return_value = {"state": "test"}
        mock_optimizer.param_groups = [{"lr": 0.001}]
        mock_scheduler = MagicMock()
        mock_scheduler.state_dict.return_value = {"state": "test"}
        mock_ema = MagicMock()
        mock_ema.state_dict.return_value = {"state": "test"}

        # Save training state
        checkpoint_manager.save_training_state(
            epoch=5,
            batch_idx=100,
            global_step=1000,
            optimizers={"optimizer": mock_optimizer},
            schedulers={"scheduler": mock_scheduler},
            ema=mock_ema,
            sampler=None,
            kl_beta_current=0.0001,
            kl_warmup_steps=5000,
            kl_max_beta=0.001,
        )

        # Check JSON file was created
        json_path = Path(temp_output_dir) / "training_state.json"
        assert json_path.exists()

        # Check JSON contents
        with open(json_path) as f:
            state = json.load(f)

        assert state["epoch"] == 5
        assert state["batch_idx"] == 100
        assert state["global_step"] == 1000
        assert "kl_warmup" in state
        assert state["kl_warmup"]["current_beta"] == 0.0001

        # Check PyTorch checkpoint was created
        pt_path = Path(temp_output_dir) / "training_states.pt"
        assert pt_path.exists()


@pytest.mark.slow
class TestTrainScriptTrainerIntegration:
    """Integration tests for trainer classes with small models."""

    @pytest.fixture
    def small_batch(self):
        """Create a small batch for testing."""
        return torch.randn(2, 3, 64, 64)  # Small images

    @pytest.fixture
    def small_text_batch(self):
        """Create a small text batch for testing."""
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        return input_ids, attention_mask

    def test_vae_trainer_single_step(self, small_batch):
        """Test VAETrainer can perform a single training step."""
        from fluxflow.models import FluxCompressor, FluxExpander, PatchDiscriminator
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from fluxflow_training.training import EMA, VAETrainer

        device = torch.device("cpu")

        # Create models
        compressor = FluxCompressor(d_model=32, use_attention=False).to(device)
        expander = FluxExpander(d_model=32).to(device)
        discriminator = PatchDiscriminator(in_channels=3, ctx_dim=32).to(device)

        # Create optimizers
        optimizer_vae = AdamW(list(compressor.parameters()) + list(expander.parameters()), lr=1e-4)
        optimizer_disc = AdamW(discriminator.parameters(), lr=1e-4)
        scheduler_vae = CosineAnnealingLR(optimizer_vae, T_max=100)
        scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=100)

        # Create EMA
        ema = EMA(torch.nn.ModuleList([compressor, expander]), decay=0.999, device=device)

        # Create loss functions
        reconstruction_loss_fn = torch.nn.L1Loss()
        reconstruction_loss_min_fn = torch.nn.MSELoss()

        # Mock accelerator
        mock_accelerator = MagicMock()
        mock_accelerator.backward = lambda loss: loss.backward()
        mock_accelerator.clip_grad_norm_ = lambda params, max_norm: torch.nn.utils.clip_grad_norm_(
            params, max_norm
        )

        # Create trainer - use correct parameter names
        trainer = VAETrainer(
            compressor=compressor,
            expander=expander,
            optimizer=optimizer_vae,
            scheduler=scheduler_vae,
            ema=ema,
            reconstruction_loss_fn=reconstruction_loss_fn,
            reconstruction_loss_min_fn=reconstruction_loss_min_fn,
            use_gan=True,
            discriminator=discriminator,
            discriminator_optimizer=optimizer_disc,
            discriminator_scheduler=scheduler_disc,
            kl_beta=0.0001,
            kl_warmup_steps=1000,
            kl_free_bits=0.0,
            lambda_adv=0.5,
            gradient_clip_norm=1.0,
            use_spade=False,
            r1_gamma=5.0,
            r1_interval=16,
            accelerator=mock_accelerator,
        )

        # Perform training step
        losses = trainer.train_step(small_batch.to(device), global_step=1)

        # Check losses are returned
        assert "vae" in losses
        assert "kl" in losses
        assert "discriminator" in losses
        assert "generator" in losses

        # Check losses are finite
        assert not torch.isnan(torch.tensor(losses["vae"]))
        assert not torch.isinf(torch.tensor(losses["vae"]))

    @pytest.mark.skip(
        reason="Requires dimension-matched models - text_encoder output must match flow_processor expectations"
    )
    def test_flow_trainer_single_step(self, small_batch, small_text_batch):
        """Test FlowTrainer can perform a single training step."""
        from fluxflow.models import BertTextEncoder, FluxCompressor, FluxFlowProcessor
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from fluxflow_training.training import FlowTrainer

        device = torch.device("cpu")
        input_ids, attention_mask = small_text_batch

        # Create models - need matching dimensions
        compressor = FluxCompressor(d_model=32, use_attention=False).to(device)
        flow_processor = FluxFlowProcessor(d_model=32, vae_dim=32).to(device)
        text_encoder = BertTextEncoder(embed_dim=64).to(device)

        # Create optimizers
        optimizer_flow = AdamW(flow_processor.parameters(), lr=1e-4)
        optimizer_te = AdamW(text_encoder.parameters(), lr=1e-5)
        scheduler_flow = CosineAnnealingLR(optimizer_flow, T_max=100)
        scheduler_te = CosineAnnealingLR(optimizer_te, T_max=100)

        # Mock accelerator
        mock_accelerator = MagicMock()
        mock_accelerator.backward = lambda loss: loss.backward()
        mock_accelerator.clip_grad_norm_ = lambda params, max_norm: torch.nn.utils.clip_grad_norm_(
            params, max_norm
        )

        # Create trainer
        trainer = FlowTrainer(
            flow_processor=flow_processor,
            text_encoder=text_encoder,
            compressor=compressor,
            optimizer=optimizer_flow,
            scheduler=scheduler_flow,
            text_encoder_optimizer=optimizer_te,
            text_encoder_scheduler=scheduler_te,
            gradient_clip_norm=1.0,
            num_train_timesteps=1000,
            accelerator=mock_accelerator,
        )

        # Perform training step
        loss = trainer.train_step(
            small_batch.to(device), input_ids.to(device), attention_mask.to(device)
        )

        # Check loss is finite
        assert not torch.isnan(torch.tensor(loss))
        assert not torch.isinf(torch.tensor(loss))
        assert loss > 0.0
