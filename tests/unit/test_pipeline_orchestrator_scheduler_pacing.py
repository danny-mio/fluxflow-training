"""Unit tests for scheduler-pacing math in TrainingPipelineOrchestrator.

Locks in the fix for a bug where the training-step scheduler was paced
against a massively-inflated total-step count:

1. ``batches_per_epoch`` must resolve batch_size the same way the real
   DataLoader does (``dataset_config.batch_size or args.batch_size`` —
   per-dataset override wins), not the top-level ``args.batch_size`` alone.
2. ``batches_per_epoch`` must reflect what a ``for batch in dataloader``
   loop actually yields (``ceil(dataset_size / batch_size)``), not a
   floor division.
3. The scheduler step budget (``total_steps`` / CosineAnnealingLR's
   ``T_max``) for FlowTrainer-managed schedulers ("flow",
   "text_encoder", "text_encoder_backbone", "text_encoder_projection")
   must divide by ``gradient_accumulation_steps``, because
   ``FlowTrainer`` only calls ``scheduler.step()`` once every
   ``gradient_accumulation_steps`` micro-batches (see
   ``FlowTrainer._accumulation_step`` / ``should_step`` in
   flow_trainer.py). VAETrainer has no such gating -- its scheduler
   steps once per micro-batch regardless of gradient_accumulation_steps
   -- so "vae" and "discriminator" schedulers must NOT be divided.
"""

from unittest.mock import MagicMock

import torch.nn as nn
from torch.optim import AdamW

from fluxflow_training.training.pipeline_config import (
    DatasetConfig,
    OptimizationConfig,
    OptimizerConfig,
    PipelineConfig,
    PipelineStepConfig,
    SchedulerConfig,
)
from fluxflow_training.training.pipeline_orchestrator import (
    TrainingPipelineOrchestrator,
    compute_batches_per_epoch,
    resolve_dataset_batch_size,
)


class TestResolveDatasetBatchSize:
    """Per-dataset batch_size override must win over top-level args.batch_size."""

    def test_dataset_override_wins(self):
        dataset_config = DatasetConfig(batch_size=8)
        args = MagicMock(batch_size=1)
        assert resolve_dataset_batch_size(dataset_config, args) == 8

    def test_falls_back_to_args_when_dataset_batch_size_unset(self):
        dataset_config = DatasetConfig(batch_size=None)
        args = MagicMock(batch_size=4)
        assert resolve_dataset_batch_size(dataset_config, args) == 4

    def test_falls_back_to_args_when_dataset_config_is_none(self):
        args = MagicMock(batch_size=4)
        assert resolve_dataset_batch_size(None, args) == 4

    def test_falls_back_when_dataset_batch_size_is_zero(self):
        dataset_config = DatasetConfig(batch_size=0)
        args = MagicMock(batch_size=4)
        assert resolve_dataset_batch_size(dataset_config, args) == 4


class TestComputeBatchesPerEpoch:
    """batches_per_epoch must equal ceil(dataset_size / batch_size), matching
    what a ``for batch in dataloader`` loop actually yields with drop_last=False.
    """

    def test_exact_division(self):
        assert compute_batches_per_epoch(800, 8) == 100

    def test_partial_last_batch_is_counted(self):
        # 801 samples / 8 per batch -> 100 full batches + 1 partial batch = 101
        assert compute_batches_per_epoch(801, 8) == 101

    def test_top_level_batch_size_would_have_been_wrong(self):
        # Real-world regression case: top-level batch_size=1, per-dataset
        # batch_size=8, dataset_size=800. Old code: 800 // 1 = 800 (8x too
        # many). Real DataLoader yields 800 // 8 = 100 batches/epoch.
        dataset_size = 800
        resolved_batch_size = 8
        assert compute_batches_per_epoch(dataset_size, resolved_batch_size) == 100

    def test_never_returns_less_than_one(self):
        assert compute_batches_per_epoch(0, 8) == 1
        assert compute_batches_per_epoch(3, 100) == 1


class TestCreateStepSchedulersGradientAccumulation:
    """total_steps (CosineAnnealingLR T_max) must divide by
    gradient_accumulation_steps only for FlowTrainer-managed scheduler names.
    """

    def _make_orch(self, step):
        config = PipelineConfig(steps=[step])
        orch = TrainingPipelineOrchestrator.__new__(TrainingPipelineOrchestrator)
        orch.config = config
        orch.device = "cpu"
        orch.accelerator = MagicMock()
        return orch

    def _make_optimizers(self, names):
        optimizers = {}
        for name in names:
            model = nn.Linear(4, 4)
            optimizers[name] = AdamW(model.parameters(), lr=1e-4)
        return optimizers

    def test_flow_scheduler_divided_by_gradient_accumulation_steps(self):
        # Real-world regression case from the bug report: n_epochs=1,
        # batches_per_epoch=3200 (real, resolved), gradient_accumulation_steps=20
        # -> intended real optimizer-step count (T_max) = 160, not 3200.
        step = PipelineStepConfig(
            name="flow_step",
            n_epochs=1,
            train_diff=True,
            gradient_accumulation_steps=20,
            optimization=OptimizationConfig(
                optimizers={"flow": OptimizerConfig(lr=1e-4)},
                schedulers={"flow": SchedulerConfig(type="CosineAnnealingLR")},
            ),
        )
        orch = self._make_orch(step)
        optimizers = self._make_optimizers(["flow"])

        schedulers = orch._create_step_schedulers(step, optimizers, total_steps=3200)

        assert schedulers["flow"].T_max == 160

    def test_text_encoder_schedulers_divided_by_gradient_accumulation_steps(self):
        step = PipelineStepConfig(
            name="flow_step",
            n_epochs=1,
            train_diff=True,
            gradient_accumulation_steps=4,
            optimization=OptimizationConfig(
                optimizers={
                    "text_encoder_backbone": OptimizerConfig(lr=5e-8),
                    "text_encoder_projection": OptimizerConfig(lr=1e-5),
                },
                schedulers={
                    "text_encoder_backbone": SchedulerConfig(type="CosineAnnealingLR"),
                    "text_encoder_projection": SchedulerConfig(type="CosineAnnealingLR"),
                },
            ),
        )
        orch = self._make_orch(step)
        optimizers = self._make_optimizers(["text_encoder_backbone", "text_encoder_projection"])

        schedulers = orch._create_step_schedulers(step, optimizers, total_steps=100)

        assert schedulers["text_encoder_backbone"].T_max == 25
        assert schedulers["text_encoder_projection"].T_max == 25

    def test_vae_scheduler_not_divided_by_gradient_accumulation_steps(self):
        # VAETrainer has no gradient-accumulation gating on scheduler.step() --
        # it steps once per micro-batch. Dividing its total_steps would pace
        # the VAE scheduler wrong (premature decay).
        step = PipelineStepConfig(
            name="vae_step",
            n_epochs=1,
            train_vae=True,
            gradient_accumulation_steps=20,
            optimization=OptimizationConfig(
                optimizers={"vae": OptimizerConfig(lr=1e-4)},
                schedulers={"vae": SchedulerConfig(type="CosineAnnealingLR")},
            ),
        )
        orch = self._make_orch(step)
        optimizers = self._make_optimizers(["vae"])

        schedulers = orch._create_step_schedulers(step, optimizers, total_steps=3200)

        assert schedulers["vae"].T_max == 3200

    def test_discriminator_scheduler_not_divided_by_gradient_accumulation_steps(self):
        step = PipelineStepConfig(
            name="gan_step",
            n_epochs=1,
            train_vae=True,
            gan_training=True,
            gradient_accumulation_steps=10,
            optimization=OptimizationConfig(
                optimizers={"discriminator": OptimizerConfig(lr=1e-4)},
                schedulers={"discriminator": SchedulerConfig(type="CosineAnnealingLR")},
            ),
        )
        orch = self._make_orch(step)
        optimizers = self._make_optimizers(["discriminator"])

        schedulers = orch._create_step_schedulers(step, optimizers, total_steps=500)

        assert schedulers["discriminator"].T_max == 500

    def test_default_gradient_accumulation_steps_of_one_is_a_no_op(self):
        step = PipelineStepConfig(
            name="flow_step",
            n_epochs=1,
            train_diff=True,
            optimization=OptimizationConfig(
                optimizers={"flow": OptimizerConfig(lr=1e-4)},
                schedulers={"flow": SchedulerConfig(type="CosineAnnealingLR")},
            ),
        )
        assert step.gradient_accumulation_steps == 1
        orch = self._make_orch(step)
        optimizers = self._make_optimizers(["flow"])

        schedulers = orch._create_step_schedulers(step, optimizers, total_steps=100)

        assert schedulers["flow"].T_max == 100

    def test_zero_gradient_accumulation_steps_treated_as_one(self):
        # Defensive guard: never divide by zero even if a step is
        # (mis)configured with gradient_accumulation_steps=0.
        step = PipelineStepConfig(
            name="flow_step",
            n_epochs=1,
            train_diff=True,
            gradient_accumulation_steps=0,
            optimization=OptimizationConfig(
                optimizers={"flow": OptimizerConfig(lr=1e-4)},
                schedulers={"flow": SchedulerConfig(type="CosineAnnealingLR")},
            ),
        )
        orch = self._make_orch(step)
        optimizers = self._make_optimizers(["flow"])

        schedulers = orch._create_step_schedulers(step, optimizers, total_steps=100)

        assert schedulers["flow"].T_max == 100


class TestEndToEndScheduleMath:
    """Given a dataset_size, a per-dataset batch_size differing from the
    top-level batch_size, an n_epochs, and a gradient_accumulation_steps,
    the combined batches_per_epoch + total_steps math must land on the real
    number of scheduler.step() calls FlowTrainer will actually make.
    """

    def test_intended_real_optimizer_step_count(self):
        # top-level training.batch_size=1, dataset.batch_size=8,
        # dataset_size=25600, n_epochs=1, gradient_accumulation_steps=20.
        # Real batches/epoch = 25600 / 8 = 3200. Real optimizer/scheduler
        # steps = 3200 * 1 epoch // 20 = 160.
        args = MagicMock(batch_size=1)
        dataset_config = DatasetConfig(batch_size=8)
        n_epochs = 1
        gradient_accumulation_steps = 20

        batch_size = resolve_dataset_batch_size(dataset_config, args)
        batches_per_epoch = compute_batches_per_epoch(25600, batch_size)
        total_steps = max(1, n_epochs * batches_per_epoch)

        step = PipelineStepConfig(
            name="flow_step",
            n_epochs=n_epochs,
            train_diff=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimization=OptimizationConfig(
                optimizers={"flow": OptimizerConfig(lr=1e-4)},
                schedulers={"flow": SchedulerConfig(type="CosineAnnealingLR")},
            ),
        )
        config = PipelineConfig(steps=[step])
        orch = TrainingPipelineOrchestrator.__new__(TrainingPipelineOrchestrator)
        orch.config = config
        orch.device = "cpu"
        orch.accelerator = MagicMock()

        model = nn.Linear(4, 4)
        optimizers = {"flow": AdamW(model.parameters(), lr=1e-4)}

        schedulers = orch._create_step_schedulers(step, optimizers, total_steps=total_steps)

        assert batches_per_epoch == 3200
        assert schedulers["flow"].T_max == 160
