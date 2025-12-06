"""Unit tests for training utilities (src/training/utils.py)."""

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.utils import (
    EMA,
    FloatBuffer,
    current_lr,
    get_device,
    img_to_random_packet,
    worker_init_fn,
)


class TestFloatBuffer:
    """Tests for FloatBuffer class."""

    def test_initialization(self):
        """Test buffer initializes with correct max size."""
        buffer = FloatBuffer(max_items=10)
        assert buffer.max_items == 10
        assert buffer.average == 0.0  # Empty buffer

    def test_add_single_item(self):
        """Test adding single item."""
        buffer = FloatBuffer(max_items=10)
        buffer.add_item(5.0)
        assert buffer.average == 5.0

    def test_add_multiple_items(self):
        """Test adding multiple items calculates correct average."""
        buffer = FloatBuffer(max_items=10)
        buffer.add_item(2.0)
        buffer.add_item(4.0)
        buffer.add_item(6.0)

        assert buffer.average == 4.0  # (2 + 4 + 6) / 3

    def test_fifo_behavior(self):
        """Test FIFO (oldest items dropped when buffer full)."""
        buffer = FloatBuffer(max_items=3)

        buffer.add_item(1.0)
        buffer.add_item(2.0)
        buffer.add_item(3.0)
        assert buffer.average == 2.0  # (1 + 2 + 3) / 3

        # Add 4th item, should drop 1.0
        buffer.add_item(4.0)
        assert buffer.average == 3.0  # (2 + 3 + 4) / 3

    def test_max_items_limit(self):
        """Buffer should never exceed max_items."""
        buffer = FloatBuffer(max_items=5)

        for i in range(10):
            buffer.add_item(float(i))

        # Should only keep last 5 items
        assert buffer.average == 7.0  # (5 + 6 + 7 + 8 + 9) / 5

    def test_accepts_tensor_values(self):
        """Should accept torch tensors and convert to float."""
        buffer = FloatBuffer(max_items=3)
        buffer.add_item(torch.tensor(2.5))
        buffer.add_item(torch.tensor(3.5))

        assert buffer.average == 3.0

    def test_accepts_int_values(self):
        """Should accept integers and convert to float."""
        buffer = FloatBuffer(max_items=3)
        buffer.add_item(1)
        buffer.add_item(2)
        buffer.add_item(3)

        assert buffer.average == 2.0

    def test_empty_buffer_returns_zero(self):
        """Empty buffer should return 0.0 average."""
        buffer = FloatBuffer(max_items=10)
        assert buffer.average == 0.0

    def test_single_item_buffer(self):
        """Buffer with max_items=1 should work."""
        buffer = FloatBuffer(max_items=1)
        buffer.add_item(5.0)
        assert buffer.average == 5.0

        buffer.add_item(10.0)
        assert buffer.average == 10.0  # Replaced previous


class TestEMA:
    """Tests for Exponential Moving Average class."""

    def test_initialization(self):
        """Test EMA initializes with model parameters."""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)

        assert ema.decay == 0.999
        assert ema.model is model
        assert len(ema.shadow) > 0

    def test_shadow_params_initialized_correctly(self):
        """Shadow parameters should be copies of model parameters."""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.9)

        # Shadow should match initial model state
        for name, param in model.state_dict().items():
            assert name in ema.shadow
            assert torch.allclose(ema.shadow[name], param)

    def test_update_changes_shadow(self):
        """Calling update() should change shadow parameters."""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.9)

        # Store original shadow
        original_weight = ema.shadow["weight"].clone()

        # Modify model
        with torch.no_grad():
            model.weight.data += 1.0

        # Update EMA
        ema.update()

        # Shadow should have changed
        assert not torch.equal(ema.shadow["weight"], original_weight)

    def test_update_formula(self):
        """Test EMA update formula: shadow = decay * shadow + (1 - decay) * param."""
        model = nn.Linear(2, 2)

        # Set known values
        with torch.no_grad():
            model.weight.data.fill_(1.0)

        ema = EMA(model, decay=0.9)

        # Update model to new value
        with torch.no_grad():
            model.weight.data.fill_(2.0)

        # Calculate expected shadow value
        # shadow = 0.9 * 1.0 + 0.1 * 2.0 = 0.9 + 0.2 = 1.1
        ema.update()

        expected = torch.ones(2, 2) * 1.1
        assert torch.allclose(ema.shadow["weight"], expected, atol=1e-6)

    def test_multiple_updates(self):
        """Test multiple EMA updates."""
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.data.fill_(1.0)

        ema = EMA(model, decay=0.5)

        # Update model
        with torch.no_grad():
            model.weight.data.fill_(2.0)

        # First update: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        ema.update()
        assert torch.allclose(ema.shadow["weight"], torch.ones(2, 2) * 1.5)

        # Second update: 0.5 * 1.5 + 0.5 * 2.0 = 1.75
        ema.update()
        assert torch.allclose(ema.shadow["weight"], torch.ones(2, 2) * 1.75)

    def test_apply_shadow(self):
        """Test apply_shadow replaces model parameters with shadow."""
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.data.fill_(1.0)

        ema = EMA(model, decay=0.9)

        # Modify model and update EMA
        with torch.no_grad():
            model.weight.data.fill_(5.0)
        ema.update()

        # Apply shadow
        ema.apply_shadow()

        # Model should now have shadow values
        assert torch.allclose(model.weight.data, ema.shadow["weight"])

    def test_restore_after_apply(self):
        """Test restore() returns model to original parameters."""
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.data.fill_(1.0)
            original_weight = model.weight.data.clone()

        ema = EMA(model, decay=0.9)

        # Modify and update
        with torch.no_grad():
            model.weight.data.fill_(5.0)
        ema.update()

        # Apply shadow (changes model)
        ema.apply_shadow()
        assert not torch.equal(model.weight.data, original_weight)

        # Restore
        ema.restore()
        assert torch.allclose(model.weight.data, torch.ones(2, 2) * 5.0)

    def test_typical_training_workflow(self):
        """Test typical train/eval workflow with EMA."""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Training step
        x = torch.randn(4, 10, requires_grad=False)
        y = torch.randn(4, 5, requires_grad=False)

        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Step with no_grad to avoid in-place operation errors
        with torch.no_grad():
            optimizer.step()

        # Update EMA after optimizer step
        ema.update()

        # For evaluation, apply shadow
        ema.apply_shadow()
        with torch.no_grad():
            eval_output = model(x)
        ema.restore()

        # Should complete without errors
        assert eval_output.shape == (4, 5)

    def test_device_handling(self):
        """Test EMA handles device correctly."""
        device = torch.device("cpu")
        model = nn.Linear(10, 5).to(device)
        ema = EMA(model, decay=0.9, device=device)

        # Shadow should be on correct device
        assert ema.shadow["weight"].device == device


class TestCurrentLR:
    """Tests for current_lr function."""

    def test_extracts_learning_rate(self):
        """Should extract learning rate from optimizer."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        lr = current_lr(optimizer)
        assert lr == 0.001

    def test_returns_float(self):
        """Should return float, not tensor."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        lr = current_lr(optimizer)
        assert isinstance(lr, float)
        assert lr == 1e-4

    def test_after_lr_change(self):
        """Should reflect changed learning rate."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Change learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = 0.01

        lr = current_lr(optimizer)
        assert lr == 0.01

    def test_multiple_param_groups(self):
        """Should return LR from first param group."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(
            [
                {"params": model.weight, "lr": 0.001},
                {"params": model.bias, "lr": 0.0001},
            ]
        )

        lr = current_lr(optimizer)
        assert lr == 0.001  # First group


class TestWorkerInitFn:
    """Tests for worker_init_fn function."""

    def test_callable(self):
        """Function should be callable."""
        # Should not raise
        worker_init_fn(0)
        worker_init_fn(1)
        worker_init_fn(42)

    def test_accepts_worker_id(self):
        """Should accept worker_id parameter."""
        for worker_id in range(10):
            worker_init_fn(worker_id)


class TestImgToRandomPacket:
    """Tests for img_to_random_packet function."""

    def test_output_shape(self):
        """Output should have shape [B, T+1, d_model]."""
        img = torch.randn(2, 3, 64, 64)
        packet = img_to_random_packet(img, d_model=128, downscales=4)

        # With 4 downscales: 64 / 2^4 = 4, so T = 4 * 4 = 16
        # Output: [B, T+1, d_model] = [2, 17, 128]
        assert packet.shape == (2, 17, 128)

    def test_spatial_token_calculation(self):
        """Should calculate spatial tokens correctly based on downscales."""
        img = torch.randn(1, 3, 128, 128)

        # 4 downscales: 128 / 16 = 8, T = 8 * 8 = 64
        packet = img_to_random_packet(img, d_model=64, downscales=4)
        assert packet.shape == (1, 65, 64)  # 64 tokens + 1 hw token

        # 3 downscales: 128 / 8 = 16, T = 16 * 16 = 256
        packet = img_to_random_packet(img, d_model=64, downscales=3)
        assert packet.shape == (1, 257, 64)

    def test_hw_token_encoding(self):
        """Last token should encode spatial dimensions."""
        img = torch.randn(1, 3, 256, 512)  # H=256, W=512
        max_hw = 1024

        packet = img_to_random_packet(img, d_model=128, downscales=4, max_hw=max_hw)

        # Latent H = 256 / 16 = 16, Latent W = 512 / 16 = 32
        hw_token = packet[:, -1, :]  # Last token

        assert hw_token[0, 0].item() == 16 / max_hw
        assert hw_token[0, 1].item() == 32 / max_hw

    def test_handles_3d_input(self):
        """Should handle 3D input [C, H, W] by adding batch dimension."""
        img = torch.randn(3, 64, 64)
        packet = img_to_random_packet(img, d_model=64, downscales=4)

        # Should add batch dimension
        assert packet.shape[0] == 1

    def test_device_preservation(self):
        """Output should be on same device as input."""
        device = torch.device("cpu")
        img = torch.randn(2, 3, 64, 64, device=device)
        packet = img_to_random_packet(img, d_model=128, downscales=4)

        assert packet.device == device

    def test_dtype_preservation(self):
        """Should preserve float dtype of input."""
        img = torch.randn(2, 3, 64, 64, dtype=torch.float32)
        packet = img_to_random_packet(img, d_model=128, downscales=4)

        assert packet.dtype == torch.float32

    def test_different_d_model_sizes(self):
        """Should work with different d_model values."""
        img = torch.randn(1, 3, 64, 64)

        for d_model in [64, 128, 256, 512]:
            packet = img_to_random_packet(img, d_model=d_model, downscales=4)
            assert packet.shape[-1] == d_model

    def test_minimum_latent_size(self):
        """Should handle very small images (minimum 1x1 latent)."""
        img = torch.randn(1, 3, 8, 8)
        packet = img_to_random_packet(img, d_model=64, downscales=4)

        # 8 / 16 = 0.5, but should be max(1, ...) = 1
        # T = 1 * 1 = 1, shape = [1, 2, 64]
        assert packet.shape == (1, 2, 64)

    def test_different_aspect_ratios(self):
        """Should handle different aspect ratios correctly."""
        img = torch.randn(1, 3, 128, 256)  # 1:2 aspect ratio
        packet = img_to_random_packet(img, d_model=128, downscales=4)

        # H_lat = 128/16 = 8, W_lat = 256/16 = 16
        # T = 8 * 16 = 128
        assert packet.shape == (1, 129, 128)

    def test_random_values(self):
        """Spatial tokens should be random (not all same)."""
        img = torch.randn(1, 3, 64, 64)
        packet = img_to_random_packet(img, d_model=128, downscales=4)

        # Tokens (excluding hw token) should have variance
        tokens = packet[:, :-1, :]
        assert tokens.var().item() > 0.5  # Should be roughly ~1 for standard normal


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_device(self):
        """Should return a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cpu_always_available(self):
        """Should return at least CPU device."""
        device = get_device()
        assert device.type in ["cuda", "mps", "cpu"]

    @pytest.mark.gpu
    def test_prefers_cuda(self):
        """Should prefer CUDA if available."""
        if torch.cuda.is_available():
            device = get_device()
            assert device.type == "cuda"
        else:
            pytest.skip("CUDA not available")


class TestTrainingUtilsIntegration:
    """Integration tests combining multiple utilities."""

    def test_training_loop_with_ema_and_buffer(self):
        """Test typical training loop with EMA and loss buffer."""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_buffer = FloatBuffer(max_items=100)

        # Training loop
        for step in range(5):
            x = torch.randn(4, 10, requires_grad=False)
            y = torch.randn(4, 5, requires_grad=False)

            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss_val = loss.item()
            loss.backward()

            # Step with no_grad to avoid in-place operation errors
            with torch.no_grad():
                optimizer.step()

            # Update EMA and buffer
            ema.update()
            loss_buffer.add_item(loss_val)

        # Should have loss average
        assert loss_buffer.average > 0

        # EMA shadow should exist
        assert len(ema.shadow) > 0

    def test_lr_scheduling_with_current_lr(self):
        """Test learning rate scheduling workflow."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Initial LR
        assert current_lr(optimizer) == 0.1

        # Simulate LR decay
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.5

        assert current_lr(optimizer) == 0.05

    def test_ema_evaluation_mode(self):
        """Test using EMA for evaluation."""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Train
        for _ in range(10):
            x = torch.randn(4, 10, requires_grad=False)
            y = torch.randn(4, 5, requires_grad=False)

            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            # Step with no_grad to avoid in-place operation errors
            with torch.no_grad():
                optimizer.step()

            ema.update()

        # Evaluate with EMA
        model.eval()
        ema.apply_shadow()

        with torch.no_grad():
            test_x = torch.randn(4, 10)
            output = model(test_x)

        ema.restore()
        model.train()

        assert output.shape == (4, 5)

    def test_random_packet_for_discriminator_training(self):
        """Test generating random packets for discriminator."""
        real_img = torch.randn(8, 3, 128, 128)

        # Generate random packet matching real image dimensions
        random_packet = img_to_random_packet(real_img, d_model=128, downscales=4, max_hw=1024)

        # Should match expected shape for discriminator training
        assert random_packet.shape[0] == 8  # Same batch size
        assert random_packet.shape[-1] == 128  # d_model
