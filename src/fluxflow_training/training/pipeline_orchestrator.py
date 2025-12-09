"""Pipeline orchestrator for multi-step training workflows.

Manages sequential execution of training pipeline steps with model freezing,
loss-threshold transitions, and checkpoint management.
"""

from typing import Any, Optional

import torch.nn as nn
from fluxflow.utils import get_logger

from .checkpoint_manager import CheckpointManager
from .pipeline_config import PipelineConfig, PipelineStepConfig

logger = get_logger(__name__)


class TrainingPipelineOrchestrator:
    """
    Orchestrates multi-step training pipelines.

    Manages:
    - Sequential step execution
    - Model freeze/unfreeze per step
    - Loss-threshold monitoring for transitions
    - Checkpoint save/load with pipeline metadata
    - Resume from any step

    Example:
        >>> config = parse_pipeline_config(config_dict)
        >>> orchestrator = TrainingPipelineOrchestrator(
        ...     config=config,
        ...     models=models_dict,
        ...     checkpoint_manager=checkpoint_mgr,
        ...     accelerator=accelerator,
        ... )
        >>> orchestrator.run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        models: dict[str, nn.Module],
        checkpoint_manager: CheckpointManager,
        accelerator: Any,
        dataloader: Any,
        dataset: Any,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            config: Parsed pipeline configuration
            models: Dictionary of model components
                Required keys: 'compressor', 'expander', 'flow_processor',
                'text_encoder', 'discriminator'
            checkpoint_manager: Checkpoint manager instance
            accelerator: Accelerate accelerator instance
            dataloader: Training dataloader
            dataset: Training dataset
        """
        self.config = config
        self.models = models
        self.checkpoint_manager = checkpoint_manager
        self.accelerator = accelerator
        self.dataloader = dataloader
        self.dataset = dataset

        # Pipeline state
        self.current_step_index = 0
        self.global_step = 0
        self.steps_completed: list[str] = []

        # Metric tracking for loss-threshold transitions
        self.step_metrics: dict[str, dict[str, list[float]]] = {}

        # Validate models dictionary
        self._validate_models()

    def _validate_models(self) -> None:
        """Validate that all required model components are present."""
        required = {"compressor", "expander", "flow_processor", "text_encoder", "discriminator"}
        missing = required - set(self.models.keys())
        if missing:
            raise ValueError(f"Missing required model components: {missing}")

    def freeze_model(self, model_name: str) -> None:
        """
        Freeze model parameters.

        Args:
            model_name: Name of model to freeze (e.g., 'compressor', 'text_encoder')
        """
        if model_name not in self.models:
            logger.warning(f"Cannot freeze '{model_name}': not found in models dict")
            return

        model = self.models[model_name]
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        # Count frozen parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Frozen: {model_name} ({num_params:,} parameters)")

    def unfreeze_model(self, model_name: str) -> None:
        """
        Unfreeze model parameters.

        Args:
            model_name: Name of model to unfreeze
        """
        if model_name not in self.models:
            logger.warning(f"Cannot unfreeze '{model_name}': not found in models dict")
            return

        model = self.models[model_name]
        for param in model.parameters():
            param.requires_grad = True
        model.train()

        # Count unfrozen parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Unfrozen: {model_name} ({num_params:,} parameters)")

    def configure_step_models(self, step: PipelineStepConfig) -> None:
        """
        Configure model freeze/unfreeze state for a pipeline step.

        Args:
            step: Pipeline step configuration
        """
        logger.info(f"Configuring models for step '{step.name}'...")

        # Freeze specified models
        for model_name in step.freeze:
            self.freeze_model(model_name)

        # Unfreeze specified models
        for model_name in step.unfreeze:
            self.unfreeze_model(model_name)

        # Log final state
        trainable_params = sum(
            p.numel() for m in self.models.values() for p in m.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for m in self.models.values() for p in m.parameters())
        frozen_params = total_params - trainable_params

        logger.info(
            f"Model configuration complete: "
            f"{trainable_params:,} trainable, {frozen_params:,} frozen "
            f"({100.0 * trainable_params / total_params:.1f}% trainable)"
        )

    def update_metrics(self, step_name: str, losses: dict[str, float]) -> None:
        """
        Update metric history for transition monitoring.

        Args:
            step_name: Name of current step
            losses: Dictionary of loss values from training step
        """
        if step_name not in self.step_metrics:
            self.step_metrics[step_name] = {}

        for metric_name, value in losses.items():
            if metric_name not in self.step_metrics[step_name]:
                self.step_metrics[step_name][metric_name] = []

            # Keep last 100 values for smoothing
            history = self.step_metrics[step_name][metric_name]
            history.append(float(value))
            if len(history) > 100:
                history.pop(0)

    def get_smoothed_metric(
        self, step_name: str, metric_name: str, window: int = 20
    ) -> Optional[float]:
        """
        Get smoothed metric value using moving average.

        Args:
            step_name: Name of step
            metric_name: Name of metric (e.g., 'vae_loss')
            window: Window size for moving average

        Returns:
            Smoothed metric value, or None if insufficient data
        """
        if step_name not in self.step_metrics:
            return None

        if metric_name not in self.step_metrics[step_name]:
            return None

        history = self.step_metrics[step_name][metric_name]
        if len(history) < window:
            return None

        return sum(history[-window:]) / window

    def should_transition(self, step: PipelineStepConfig, current_epoch: int) -> tuple[bool, str]:
        """
        Check if transition criteria are met for current step.

        Args:
            step: Current pipeline step configuration
            current_epoch: Current epoch number (within this step)

        Returns:
            Tuple of (should_transition, reason_string)
        """
        criteria = step.transition_on

        if criteria.mode == "epoch":
            if current_epoch >= step.n_epochs:
                return True, f"Completed {step.n_epochs} epochs"
            return False, f"Epoch {current_epoch}/{step.n_epochs}"

        elif criteria.mode == "loss_threshold":
            # Get smoothed metric value
            metric_value = self.get_smoothed_metric(step.name, criteria.metric)

            # Check max_epochs upper limit first
            max_epochs = criteria.max_epochs or step.n_epochs
            if current_epoch >= max_epochs:
                if metric_value is not None:
                    return (
                        True,
                        f"Max epochs ({max_epochs}) reached, {criteria.metric}={metric_value:.4f}",
                    )
                return True, f"Max epochs ({max_epochs}) reached"

            # Check if we have enough data for smoothed metric
            if metric_value is None:
                return False, f"Collecting metrics ({criteria.metric})"

            # Check threshold
            if metric_value < criteria.threshold:
                return (
                    True,
                    f"{criteria.metric}={metric_value:.4f} < {criteria.threshold} (threshold met)",
                )

            return (
                False,
                f"{criteria.metric}={metric_value:.4f} "
                f"(target: <{criteria.threshold}, epochs: {current_epoch}/{max_epochs})",
            )

        return False, "Unknown transition mode"

    def get_pipeline_metadata(self, step_index: int, epoch: int) -> dict:
        """
        Get pipeline metadata for checkpoint saving.

        Args:
            step_index: Current step index
            epoch: Current epoch (global, across all steps)

        Returns:
            Dictionary with pipeline state metadata
        """
        current_step = self.config.steps[step_index]

        return {
            "current_step_index": step_index,
            "current_step_name": current_step.name,
            "total_steps": len(self.config.steps),
            "steps_completed": self.steps_completed.copy(),
            "step_start_epoch": epoch,
        }

    def resume_from_checkpoint(self) -> tuple[int, int, int]:
        """
        Resume pipeline from checkpoint if available.

        Returns:
            Tuple of (step_index, epoch, batch_idx)
        """
        training_state = self.checkpoint_manager.load_training_state()

        if not training_state:
            logger.info("No checkpoint found, starting from beginning")
            return 0, 0, 0

        # Check if this is a pipeline checkpoint
        if training_state.get("mode") != "pipeline":
            logger.info("Checkpoint is legacy mode (not pipeline), starting from step 0")
            return 0, 0, 0

        pipeline_meta = training_state.get("pipeline", {})
        step_index = pipeline_meta.get("current_step_index", 0)
        epoch = training_state.get("epoch", 0)
        batch_idx = training_state.get("batch_idx", 0)
        self.global_step = training_state.get("global_step", 0)
        self.steps_completed = pipeline_meta.get("steps_completed", [])

        logger.info(
            f"Resuming from checkpoint: "
            f"step {step_index + 1}/{len(self.config.steps)} "
            f"('{pipeline_meta.get('current_step_name', 'unknown')}'), "
            f"epoch {epoch}, batch {batch_idx}"
        )

        return step_index, epoch, batch_idx

    def print_pipeline_summary(self) -> None:
        """Print pipeline execution plan summary."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION PLAN")
        print("=" * 80)

        total_epochs = sum(step.n_epochs for step in self.config.steps)

        for i, step in enumerate(self.config.steps, 1):
            print(f"\nStep {i}/{len(self.config.steps)}: {step.name} ({step.n_epochs} epochs)")

            if step.description:
                print(f"  Description: {step.description}")

            # Show training modes
            modes = []
            if step.train_vae:
                modes.append("VAE")
            if step.gan_training:
                modes.append("GAN")
            if step.train_spade:
                modes.append("SPADE")
            if step.use_lpips:
                modes.append("LPIPS")
            if step.train_diff or step.train_diff_full:
                modes.append("Flow")
            print(f"  Train: {', '.join(modes)}")

            # Show frozen models
            if step.freeze:
                print(f"  Frozen: {', '.join(step.freeze)}")

            # Show transition criteria
            if step.transition_on.mode == "epoch":
                print(f"  Transition: After {step.n_epochs} epochs")
            elif step.transition_on.mode == "loss_threshold":
                print(
                    f"  Transition: When {step.transition_on.metric} < "
                    f"{step.transition_on.threshold} (max {step.transition_on.max_epochs} epochs)"
                )

        print(f"\nTotal epochs: {total_epochs}")
        print("=" * 80 + "\n")

    def run(self, args, config) -> None:
        """
        Execute the complete training pipeline.

        Args:
            args: Parsed command-line arguments
            config: Loaded YAML config dictionary

        Raises:
            NotImplementedError: Pipeline execution is not yet fully implemented
        """
        logger.info("Starting training pipeline execution...")

        # Print pipeline summary
        self.print_pipeline_summary()

        # Resume from checkpoint if available
        start_step, start_epoch, start_batch = self.resume_from_checkpoint()

        logger.info(
            f"Pipeline has {len(self.config.steps)} steps, " f"starting from step {start_step + 1}"
        )

        # TODO: Complete implementation requires:
        # 1. Initialize models (diffuser, text_encoder, discriminators)
        # 2. Initialize dataset and dataloader
        # 3. For each pipeline step:
        #    a. Apply freeze/unfreeze directives
        #    b. Create optimizers/schedulers from step config
        #    c. Create VAETrainer and/or FlowTrainer based on step settings
        #    d. Run training loop for step duration
        #    e. Monitor transition criteria (loss threshold)
        #    f. Save checkpoints with pipeline metadata
        # 4. Handle resume from checkpoint (restore step, epoch, batch)

        raise NotImplementedError(
            "Full pipeline execution is not yet implemented. "
            "This requires integration with model initialization, "
            "dataloader setup, and trainer creation from train.py. "
            "Current implementation provides validation and planning only."
        )
