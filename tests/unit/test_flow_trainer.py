"""Unit tests for FlowTrainer."""

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_compressor(vae_dim=32, context_dims=5, token_count=16):
    """Mock VAE compressor returning a realistic latent packet."""
    total_dim = vae_dim + context_dims
    compressor = MagicMock()
    compressor.get_context_dims.return_value = context_dims
    compressor.use_gradient_checkpointing = False

    def _forward(imgs):
        B = imgs.shape[0]
        packet = torch.randn(B, token_count + 1, total_dim)
        packet[:, -1, :] = 0.0
        packet[:, -1, 0] = 4 / 64.0
        packet[:, -1, 1] = 4 / 64.0
        return packet.detach()

    compressor.side_effect = _forward
    compressor.parameters = lambda recurse=True: iter([])
    return compressor


class _MinimalFlow(nn.Module):
    """Tiny real flow model covering all dims — produces real gradients."""

    def __init__(self, total_dim):
        super().__init__()
        self.proj = nn.Linear(total_dim, total_dim)

    def forward(self, packet, text_emb, t):
        img_seq = packet[:, :-1, :]
        hw_vec = packet[:, -1:, :]
        return torch.cat([self.proj(img_seq), hw_vec], dim=1)


def _make_trainer(vae_dim=32, context_dims=5, token_count=16):
    """Construct a minimal FlowTrainer for unit testing."""
    from fluxflow_training.training.flow_trainer import FlowTrainer

    total_dim = vae_dim + context_dims
    flow = _MinimalFlow(total_dim)

    class _TextEncoder(nn.Module):
        def forward(self, ids, attention_mask=None):
            return torch.randn(ids.shape[0], 64)

    text_encoder = _TextEncoder()

    compressor = _make_mock_compressor(vae_dim, context_dims, token_count)
    optimizer = AdamW(flow.parameters(), lr=1e-4)
    lr_sched = ConstantLR(optimizer)

    mock_acc = MagicMock()
    mock_acc.backward = lambda loss: loss.backward()
    mock_acc.clip_grad_norm_ = nn.utils.clip_grad_norm_

    trainer = FlowTrainer(
        flow_processor=flow,
        text_encoder=text_encoder,
        compressor=compressor,
        optimizer=optimizer,
        scheduler=lr_sched,
        gradient_clip_norm=1.0,
        num_train_timesteps=100,
        accelerator=mock_acc,
    )
    return trainer, flow, compressor


# ---------------------------------------------------------------------------
# Task 1: v-prediction target
# ---------------------------------------------------------------------------


class TestFlowTrainerVPredLoss:
    """diff_loss must target v_target = alpha_t * noise - sigma_t * x0."""

    def test_loss_target_is_not_x0(self):
        """
        Capture the target passed to smooth_l1_loss.
        Before the fix this equals normalized x0 — after the fix it must not.
        """
        trainer, flow, compressor = _make_trainer(vae_dim=32, context_dims=0, token_count=16)

        captured = {}
        orig = nn.functional.smooth_l1_loss

        def capture(pred, target, **kw):
            if "target" not in captured:
                captured["target"] = target.detach().clone()
            return orig(pred, target, **kw)

        with patch("torch.nn.functional.smooth_l1_loss", side_effect=capture):
            trainer.train_step(
                torch.randn(2, 3, 64, 64),
                torch.zeros(2, 8, dtype=torch.long),
                torch.ones(2, 8, dtype=torch.long),
                global_step=0,
            )

        assert "target" in captured, "smooth_l1_loss was never called"

        # Build the normalized x0 as the buggy code would produce it
        packet = compressor(torch.randn(2, 3, 64, 64))
        img_seq = packet[:, :-1, :].float()
        latent_std = img_seq.detach().std() + 1e-8
        normalized_x0 = (img_seq - img_seq.detach().mean()) / latent_std

        assert not torch.allclose(
            captured["target"], normalized_x0, atol=1e-3
        ), "diff_loss target equals normalized x0 — v-prediction target not computed."

    def test_v_target_at_t0_equals_noise(self):
        """At t=0: alpha_cumprod≈1, sigma≈0 → v_target ≈ noise."""
        trainer, _, _ = _make_trainer()
        ac = trainer.alphas_cumprod
        assert ac[0].item() > 0.99

        t = torch.tensor([0], dtype=torch.long)
        acp = ac[t].float()
        alpha_t = acp.sqrt().view(-1, 1, 1)
        sigma_t = (1.0 - acp).sqrt().view(-1, 1, 1)

        x0 = torch.randn(1, 4, 32)
        noise = torch.randn(1, 4, 32)
        v = alpha_t * noise - sigma_t * x0

        assert torch.allclose(v, noise, atol=0.05)

    def test_v_target_differs_from_x0_and_noise_at_mid_t(self):
        """At mid-timestep v_target is a blend — must differ from both noise and x0."""
        trainer, _, _ = _make_trainer()
        ac = trainer.alphas_cumprod
        mid = len(ac) // 2
        t = torch.tensor([mid], dtype=torch.long)
        acp = ac[t].float()
        alpha_t = acp.sqrt().view(-1, 1, 1)
        sigma_t = (1.0 - acp).sqrt().view(-1, 1, 1)

        torch.manual_seed(0)
        x0 = torch.randn(1, 4, 32)
        noise = torch.randn(1, 4, 32)
        v = alpha_t * noise - sigma_t * x0

        assert not torch.allclose(v, noise, atol=0.01)
        assert not torch.allclose(v, x0, atol=0.01)
