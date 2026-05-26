"""Unit tests for FlowTrainer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        The loss target must be the v-prediction vector, not normalized x0.
        Uses a deterministic compressor so the captured target can be
        meaningfully compared against what the buggy code would produce.
        """
        # Use context_dims=0 so loss goes through the v0.6 branch (simpler to verify)
        from fluxflow_training.training.flow_trainer import FlowTrainer

        vae_dim = 8
        token_count = 4
        B = 2

        # Deterministic fixed latent packet — same tensor every call
        fixed_packet = torch.zeros(B, token_count + 1, vae_dim)
        fixed_packet[:, :-1, :] = 1.0  # all image tokens = 1.0 (x0)
        fixed_packet[:, -1, 0] = 4 / 64.0
        fixed_packet[:, -1, 1] = 4 / 64.0

        flow = _MinimalFlow(vae_dim)
        text_encoder = MagicMock()
        text_encoder.parameters = lambda recurse=True: iter([])
        text_encoder.side_effect = lambda ids, attention_mask=None: torch.randn(ids.shape[0], 64)
        text_encoder.eval = MagicMock()
        text_encoder.train = MagicMock()

        compressor = MagicMock()
        compressor.get_context_dims.return_value = 0  # no context dims → v0.6 branch
        compressor.use_gradient_checkpointing = False
        compressor.side_effect = lambda imgs: fixed_packet.clone()
        compressor.parameters = lambda recurse=True: iter([])

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

        captured = {}
        orig = nn.functional.smooth_l1_loss

        def capture(pred, target, **kw):
            if "target" not in captured:
                captured["target"] = target.detach().clone()
            return orig(pred, target, **kw)

        with patch(
            "fluxflow_training.training.flow_trainer.nn.functional.smooth_l1_loss",
            side_effect=capture,
        ):
            trainer.train_step(
                torch.randn(B, 3, 16, 16),
                torch.zeros(B, 8, dtype=torch.long),
                torch.ones(B, 8, dtype=torch.long),
                global_step=0,
            )

        assert "target" in captured, "smooth_l1_loss was never called"

        # Build what the BUGGY code would have used as target: normalized x0
        x0 = fixed_packet[:, :-1, :].float()  # [B, T, vae_dim] — all ones
        latent_std = x0.detach().std() + 1e-8
        buggy_target = (x0 - x0.detach().mean()) / latent_std

        # After the fix, the captured target must be the v-prediction vector, not x0.
        # v_target depends on sampled noise and alphas_cumprod — it differs from x0.
        assert not torch.allclose(captured["target"], buggy_target, atol=1e-3), (
            "diff_loss target equals normalized x0 — v-prediction target not computed. "
            "The bug is still present."
        )

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


# ---------------------------------------------------------------------------
# Task 2: context dim supervision
# ---------------------------------------------------------------------------


class TestFlowTrainerContextDimsLoss:
    """Context dims must have gradient signal and be included in the loss."""

    def test_context_dims_receive_gradient(self):
        """flow.proj.weight.grad must be non-None and finite after train_step."""
        trainer, flow, _ = _make_trainer(vae_dim=32, context_dims=5, token_count=16)

        captured_grad = {}
        flow.proj.weight.register_hook(lambda g: captured_grad.update({"g": g}))

        trainer.train_step(
            torch.randn(2, 3, 64, 64),
            torch.zeros(2, 8, dtype=torch.long),
            torch.ones(2, 8, dtype=torch.long),
            global_step=0,
        )

        assert "g" in captured_grad, "No gradient reached flow.proj.weight"
        assert torch.isfinite(captured_grad["g"]).all(), "NaN/Inf in gradients"

    def test_ctx_loss_present_in_metrics(self):
        """train_step must return a 'ctx_loss' key in metrics."""
        trainer, _, _ = _make_trainer(vae_dim=32, context_dims=5, token_count=16)
        metrics = trainer.train_step(
            torch.randn(2, 3, 64, 64),
            torch.zeros(2, 8, dtype=torch.long),
            torch.ones(2, 8, dtype=torch.long),
            global_step=0,
        )
        assert "ctx_loss" in metrics, (
            "Metrics must include 'ctx_loss'. " "Missing means context dims are not supervised."
        )
        assert (
            metrics["ctx_loss"] > 0.0
        ), "ctx_loss should be positive (context dims not trivially zero)"

    def test_loss_is_sensitive_to_context_dim_prediction(self):
        """
        ctx_loss formula: zero prediction vs perfect prediction must produce different losses.
        """
        vae_dim, ctx_dim = 32, 5
        B, T = 2, 16

        ctx_target = torch.randn(B, T, ctx_dim) * 3.0  # non-trivial scale
        ctx_std = ctx_target.detach().std() + 1e-8

        # Perfect prediction → zero loss
        loss_perfect = F.smooth_l1_loss(ctx_target / ctx_std, ctx_target / ctx_std, beta=0.01)
        # Zero prediction → non-zero loss
        loss_zero_pred = F.smooth_l1_loss(
            torch.zeros_like(ctx_target) / ctx_std, ctx_target / ctx_std, beta=0.01
        )

        assert loss_perfect.item() < 1e-6, "Perfect prediction should give ~0 loss"
        assert loss_zero_pred.item() > 0.1, "Zero prediction should give significant loss"


# ---------------------------------------------------------------------------
# Task 3: gradient clipping
# ---------------------------------------------------------------------------


class TestFlowTrainerGradientClipping:
    """Gradient clipping must be effective and not over-clip small gradients."""

    def test_large_gradients_clipped_to_clip_norm(self):
        """After clipping, grad norm must be <= gradient_clip_norm."""
        trainer, flow, _ = _make_trainer()
        for p in flow.parameters():
            p.grad = torch.ones_like(p) * 100.0

        pre_norm = sum(p.grad.norm().item() ** 2 for p in flow.parameters()) ** 0.5
        assert pre_norm > trainer.gradient_clip_norm

        nn.utils.clip_grad_norm_(flow.parameters(), trainer.gradient_clip_norm)

        post_norm = sum(p.grad.norm().item() ** 2 for p in flow.parameters()) ** 0.5
        assert (
            post_norm <= trainer.gradient_clip_norm + 1e-3
        ), f"Post-clip norm {post_norm:.4f} exceeds clip_norm {trainer.gradient_clip_norm}"

    def test_small_gradients_not_over_clipped(self):
        """When grad norm < clip_norm, gradients must not be reduced."""
        trainer, flow, _ = _make_trainer()
        n_params = sum(p.numel() for p in flow.parameters())
        small_val = trainer.gradient_clip_norm * 0.1 / (n_params**0.5)
        for p in flow.parameters():
            p.grad = torch.full_like(p, small_val)

        pre_norm = sum(p.grad.norm().item() ** 2 for p in flow.parameters()) ** 0.5
        assert pre_norm < trainer.gradient_clip_norm, "pre_norm should be small for this test"

        nn.utils.clip_grad_norm_(flow.parameters(), trainer.gradient_clip_norm)

        post_norm = sum(p.grad.norm().item() ** 2 for p in flow.parameters()) ** 0.5
        assert (
            abs(post_norm - pre_norm) < 1e-5
        ), f"Small gradients ({pre_norm:.6f}) were incorrectly clipped to {post_norm:.6f}."


# ---------------------------------------------------------------------------
# Task 5: split text-encoder optimizers
# ---------------------------------------------------------------------------


class _SimpleTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, ids, attention_mask=None):
        return torch.randn(ids.shape[0], 4)


def _make_minimal_trainer(text_encoder_extra_optimizers=None):
    from fluxflow_training.training.flow_trainer import FlowTrainer

    flow = _MinimalFlow(4)
    text = _SimpleTextEncoder()
    comp = _make_mock_compressor(vae_dim=4, context_dims=0, token_count=4)

    opt = AdamW(flow.parameters(), lr=1e-4)
    sched = MagicMock()
    sched.scheduler = sched

    accel = MagicMock()
    accel.backward = lambda loss: loss.backward()
    accel.clip_grad_norm_ = lambda params, norm: torch.tensor(0.0)

    return FlowTrainer(
        flow_processor=flow,
        text_encoder=text,
        compressor=comp,
        optimizer=opt,
        scheduler=sched,
        text_encoder_extra_optimizers=text_encoder_extra_optimizers,
        accelerator=accel,
    )


class TestFlowTrainerSplitTextEncoderOptimizers:
    """FlowTrainer accepts split backbone/projection optimizers."""

    def test_init_accepts_extra_optimizers_dict(self):
        enc = nn.Linear(4, 4)
        extras = {
            "backbone": AdamW([{"params": list(enc.parameters())}], lr=5e-8),
            "projection": AdamW([{"params": list(enc.parameters())}], lr=1e-5),
        }
        trainer = _make_minimal_trainer(text_encoder_extra_optimizers=extras)
        assert trainer.text_encoder_extra_optimizers == extras

    def test_init_none_extra_optimizers_is_empty_dict(self):
        trainer = _make_minimal_trainer(text_encoder_extra_optimizers=None)
        assert trainer.text_encoder_extra_optimizers == {}

    def test_metrics_include_lr_backbone_and_projection(self):
        enc = nn.Linear(4, 4)
        extras = {
            "backbone": AdamW([{"params": list(enc.parameters())}], lr=5e-8),
            "projection": AdamW([{"params": list(enc.parameters())}], lr=1e-5),
        }
        trainer = _make_minimal_trainer(text_encoder_extra_optimizers=extras)

        B = 2
        real_imgs = torch.randn(B, 3, 32, 32)
        input_ids = torch.randint(0, 100, (B, 8))
        attn = torch.ones(B, 8)

        metrics = trainer.train_step(real_imgs, input_ids, attn, global_step=0)
        assert "lr_text_backbone" in metrics
        assert "lr_text_projection" in metrics

    def test_extra_optimizer_zero_grad_called_once_per_accumulation_window(self):
        """With gradient_accumulation_steps=2, extra optimizer zero_grad fires once per two micro-steps."""
        from fluxflow_training.training.flow_trainer import FlowTrainer

        flow = _MinimalFlow(4)
        text = _SimpleTextEncoder()
        proj_opt = AdamW(text.parameters(), lr=1e-5)

        accel = MagicMock()
        accel.backward = lambda loss: loss.backward()
        accel.clip_grad_norm_ = lambda params, norm: torch.tensor(0.0)

        trainer = FlowTrainer(
            flow_processor=flow,
            text_encoder=text,
            compressor=_make_mock_compressor(vae_dim=4, context_dims=0, token_count=4),
            optimizer=AdamW(flow.parameters(), lr=1e-4),
            scheduler=MagicMock(scheduler=MagicMock()),
            text_encoder_extra_optimizers={"projection": proj_opt},
            gradient_accumulation_steps=2,
            accelerator=accel,
        )

        B = 2
        real_imgs = torch.randn(B, 3, 32, 32)
        input_ids = torch.randint(0, 100, (B, 8))
        attn = torch.ones(B, 8)

        zero_grad_calls = []
        original_zero_grad = proj_opt.zero_grad
        proj_opt.zero_grad = lambda **kw: zero_grad_calls.append(1) or original_zero_grad(**kw)

        trainer.train_step(real_imgs, input_ids, attn, global_step=0)
        trainer.train_step(real_imgs, input_ids, attn, global_step=1)
        assert (
            len(zero_grad_calls) == 1
        ), f"Expected 1 zero_grad call over 2 micro-steps, got {len(zero_grad_calls)}"
