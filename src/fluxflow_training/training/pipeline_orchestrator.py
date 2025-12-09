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
        pipeline_config: PipelineConfig = None,
        checkpoint_manager: CheckpointManager = None,
        accelerator: Any = None,
        device: Any = None,
        # Legacy signature support (for tests)
        config: PipelineConfig = None,
        models: dict[str, nn.Module] = None,
        dataloader: Any = None,
        dataset: Any = None,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            pipeline_config: Parsed pipeline configuration (new signature)
            checkpoint_manager: Checkpoint manager instance (new signature)
            accelerator: Accelerate accelerator instance (new signature)
            device: Target device (new signature)
            config: Parsed pipeline configuration (legacy, for tests)
            models: Dictionary of model components (legacy, for tests)
            dataloader: Training dataloader (legacy, for tests)
            dataset: Training dataset (legacy, for tests)
        """
        # Support both new and legacy signatures
        self.config = pipeline_config or config
        self.checkpoint_manager = checkpoint_manager
        self.accelerator = accelerator
        self.device = device

        # Legacy support
        self.models = models or {}
        self.dataloader = dataloader
        self.dataset = dataset

        # Pipeline state
        self.current_step_index = 0
        self.global_step = 0
        self.steps_completed: list[str] = []

        # Metric tracking for loss-threshold transitions
        self.step_metrics: dict[str, dict[str, list[float]]] = {}

        # Validate models dictionary if provided (legacy mode)
        if self.models:
            self._validate_models()

    def _validate_models(self) -> None:
        """Validate that all required model components are present."""
        required = {"compressor", "expander", "flow_processor", "text_encoder", "discriminator"}
        missing = required - set(self.models.keys())
        if missing:
            logger.warning(f"Missing model components (may be provided to run()): {missing}")

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

    def configure_step_models(
        self, step: PipelineStepConfig, models: dict[str, nn.Module] = None
    ) -> None:
        """
        Configure models for pipeline step (freeze/unfreeze).

        Args:
            step: Pipeline step configuration
            models: Dictionary of models (optional, uses self.models if not provided)
        """
        models_dict = models or self.models
        logger.info(f"Configuring models for step '{step.name}'...")

        # Freeze specified models
        for model_name in step.freeze:
            if model_name not in models_dict:
                logger.warning(f"Cannot freeze '{model_name}': not found in models dict")
                continue
            model = models_dict[model_name]
            for param in model.parameters():
                param.requires_grad = False
            logger.info(f"Frozen model: {model_name}")

        # Unfreeze specified models
        for model_name in step.unfreeze:
            if model_name not in models_dict:
                logger.warning(f"Cannot unfreeze '{model_name}': not found in models dict")
                continue
            model = models_dict[model_name]
            for param in model.parameters():
                param.requires_grad = True
            logger.info(f"Unfrozen model: {model_name}")

        # Log final state
        if models_dict:
            trainable_params = sum(
                p.numel() for m in models_dict.values() for p in m.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for m in models_dict.values() for p in m.parameters())
            frozen_params = total_params - trainable_params

            if total_params > 0:
                logger.info(
                    f"Model configuration complete: "
                    f"{trainable_params:,} trainable, {frozen_params:,} frozen "
                    f"({100.0 * trainable_params / total_params:.1f}% trainable)"
                )
            else:
                logger.warning("No model parameters found")

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

    def get_pipeline_metadata(self, step_index: int, step_epoch: int, batch_idx: int) -> dict:
        """
        Get pipeline metadata for checkpoint saving.

        Args:
            step_index: Current step index
            step_epoch: Current epoch within the current step (0-based)
            batch_idx: Current batch index within the current epoch

        Returns:
            Dictionary with pipeline state metadata
        """
        current_step = self.config.steps[step_index]

        return {
            "current_step_index": step_index,
            "current_step_name": current_step.name,
            "current_step_epoch": step_epoch,
            "current_batch_idx": batch_idx,
            "total_steps": len(self.config.steps),
            "steps_completed": self.steps_completed.copy(),
        }

    def resume_from_checkpoint(self) -> tuple[int, int, int]:
        """
        Resume pipeline from checkpoint if available.

        Returns:
            Tuple of (step_index, step_epoch, batch_idx)
                step_epoch: Epoch number within the current step (0-based)
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

        # Use step-local epoch from pipeline metadata (new format)
        # Fall back to global epoch for backward compatibility
        step_epoch = pipeline_meta.get("current_step_epoch", training_state.get("epoch", 0))

        # Use batch_idx from pipeline metadata if available, else from training state
        batch_idx = pipeline_meta.get("current_batch_idx", training_state.get("batch_idx", 0))

        self.global_step = training_state.get("global_step", 0)
        self.steps_completed = pipeline_meta.get("steps_completed", [])

        logger.info(
            f"Resuming from checkpoint: "
            f"step {step_index + 1}/{len(self.config.steps)} "
            f"('{pipeline_meta.get('current_step_name', 'unknown')}'), "
            f"step_epoch {step_epoch + 1}, batch {batch_idx}, global_step {self.global_step}"
        )

        return step_index, step_epoch, batch_idx

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

    def _create_step_optimizers(self, step, models, args):
        """Create optimizers for current step from inline config."""
        from ..training.optimizer_factory import create_optimizer

        optimizers = {}

        if not step.optimization or not step.optimization.optimizers:
            # Use default configs
            logger.info("No optimizer config found, using defaults")
            return optimizers

        for name, opt_config_obj in step.optimization.optimizers.items():
            # Convert OptimizerConfig to dict format expected by create_optimizer
            opt_config = {
                "type": opt_config_obj.type,
                "lr": opt_config_obj.lr,
                "weight_decay": opt_config_obj.weight_decay,
                "eps": opt_config_obj.eps,
            }
            if opt_config_obj.betas:
                opt_config["betas"] = opt_config_obj.betas

            # Get parameters based on optimizer name
            if name == "vae":
                params = list(models["compressor"].parameters()) + list(
                    models["expander"].parameters()
                )
            elif name == "flow":
                params = models["flow_processor"].parameters()
            elif name == "text_encoder":
                params = models["text_encoder"].parameters()
            elif name == "discriminator":
                params = models["D_img"].parameters()
            else:
                logger.warning(f"Unknown optimizer name: {name}")
                continue

            optimizer = create_optimizer(params, opt_config)
            optimizers[name] = optimizer
            logger.info(
                f"Created optimizer '{name}': {opt_config['type']} (lr={opt_config['lr']:.2e})"
            )

        return optimizers

    def _create_step_schedulers(self, step, optimizers, total_steps):
        """Create schedulers for current step from inline config."""
        from ..training.scheduler_factory import create_scheduler

        schedulers = {}

        if not step.optimization or not step.optimization.schedulers:
            logger.info("No scheduler config found, skipping")
            return schedulers

        for name, sched_config_obj in step.optimization.schedulers.items():
            if name not in optimizers:
                logger.warning(f"Scheduler '{name}' has no corresponding optimizer")
                continue

            # Convert SchedulerConfig to dict
            sched_config = {
                "type": sched_config_obj.type,
                "eta_min_factor": sched_config_obj.eta_min_factor,
            }

            scheduler = create_scheduler(optimizers[name], sched_config, total_steps)
            schedulers[name] = scheduler
            logger.info(f"Created scheduler '{name}': {sched_config['type']}")

        return schedulers

    def _create_step_trainers(self, step, models, optimizers, schedulers, ema, args):
        """Create trainers for current step."""
        import torch.nn as nn

        from ..training import FlowTrainer, VAETrainer

        trainers = {}

        if step.train_vae and "vae" in optimizers:
            trainers["vae"] = VAETrainer(
                compressor=models["compressor"],
                expander=models["expander"],
                optimizer=optimizers["vae"],
                scheduler=schedulers.get("vae"),
                ema=ema,
                reconstruction_loss_fn=nn.L1Loss(),
                reconstruction_loss_min_fn=nn.MSELoss(),
                use_spade=step.train_spade,
                kl_beta=step.kl_beta if hasattr(step, "kl_beta") else 0.0001,
                kl_warmup_steps=step.kl_warmup_steps if hasattr(step, "kl_warmup_steps") else 5000,
                kl_free_bits=step.kl_free_bits if hasattr(step, "kl_free_bits") else 0.0,
                use_gan=step.gan_training,
                discriminator=models["D_img"] if step.gan_training else None,
                discriminator_optimizer=optimizers.get("discriminator"),
                discriminator_scheduler=schedulers.get("discriminator"),
                lambda_adv=step.lambda_adv if hasattr(step, "lambda_adv") else 0.5,
                use_lpips=step.use_lpips,
                lambda_lpips=step.lambda_lpips if hasattr(step, "lambda_lpips") else 0.1,
                r1_gamma=5.0,
                r1_interval=16,
                gradient_clip_norm=args.initial_clipping_norm,
                accelerator=self.accelerator,
            )
            logger.info(f"Created VAE trainer (SPADE={'ON' if step.train_spade else 'OFF'})")

        if (step.train_diff or step.train_diff_full) and "flow" in optimizers:
            trainers["flow"] = FlowTrainer(
                flow_processor=models["flow_processor"],
                text_encoder=models["text_encoder"],
                compressor=models["compressor"],
                optimizer=optimizers["flow"],
                scheduler=schedulers.get("flow"),
                text_encoder_optimizer=optimizers.get("text_encoder"),
                text_encoder_scheduler=schedulers.get("text_encoder"),
                gradient_clip_norm=args.initial_clipping_norm,
                num_train_timesteps=1000,
                accelerator=self.accelerator,
            )
            logger.info("Created Flow trainer")

        return trainers

    def _save_checkpoint(
        self, step_idx, step_epoch, batch_idx, models, optimizers, schedulers, ema, args
    ):
        """
        Save checkpoint with pipeline metadata.

        Args:
            step_idx: Current pipeline step index
            step_epoch: Current epoch within the step (0-based)
            batch_idx: Current batch index
            models: Dictionary of models
            optimizers: Dictionary of optimizers
            schedulers: Dictionary of schedulers
            ema: EMA module (if applicable)
            args: Training arguments
        """
        # Get pipeline metadata
        metadata = self.get_pipeline_metadata(step_idx, step_epoch, batch_idx)

        # Save models
        self.checkpoint_manager.save_models(
            diffuser=models["diffuser"],
            text_encoder=models["text_encoder"],
            discriminators={"D_img": models["D_img"]} if models.get("D_img") else None,
        )

        # Save training state with pipeline metadata
        self.checkpoint_manager.save_training_state(
            epoch=step_epoch,  # Use step-local epoch for consistency
            batch_idx=batch_idx,
            global_step=self.global_step,
            optimizers=optimizers,
            schedulers=schedulers,
            ema=ema,
            pipeline_metadata=metadata,
        )

        logger.info(
            f"Checkpoint saved: step {step_idx+1}/{len(self.config.steps)}, "
            f"epoch {step_epoch+1}, batch {batch_idx}, global_step {self.global_step}"
        )

    def run(self, models, dataloader, sampler, tokenizer, progress_logger, args, config) -> None:
        """
        Execute the complete training pipeline.

        This method is the main entry point for pipeline execution. It orchestrates
        multi-step training by configuring models, creating trainers, and managing
        the training loop across pipeline steps.

        Args:
            models: Dict of initialized models:
                - diffuser: FluxPipeline instance
                - compressor: FluxCompressor instance
                - expander: FluxExpander instance
                - flow_processor: FluxFlowProcessor instance
                - text_encoder: BertTextEncoder instance
                - D_img: PatchDiscriminator instance (if GAN training)
            dataloader: Initialized DataLoader for training data
            tokenizer: Tokenizer for text processing
            args: Parsed command-line arguments
            config: Loaded YAML config dictionary

        Raises:
            NotImplementedError: Full implementation deferred to Phase 3b
                                See docs/PIPELINE_ARCHITECTURE.md for design

        Architecture Overview:
            1. Resume from checkpoint (if exists) → get start_step, start_epoch, start_batch
            2. For each step in pipeline:
                a. configure_step_models() - freeze/unfreeze per step config
                b. _create_step_optimizers() - from inline YAML config
                c. _create_step_schedulers() - from inline YAML config
                d. _create_step_trainers() - VAETrainer and/or FlowTrainer
                e. Training loop:
                   - For each epoch in step:
                     - For each batch:
                       - vae_trainer.train_step() if train_vae
                       - flow_trainer.train_step() if train_flow
                       - update_metrics()
                       - log progress
                       - save checkpoint (with pipeline metadata)
                     - Check transition_criteria (epoch or loss_threshold)
                f. Cleanup optimizers/schedulers
            3. Print final summary

        Next Implementation Steps:
            1. Extract initialize_models() and initialize_dataloader() helpers from train_legacy()
            2. Implement _create_step_optimizers() using create_optimizer() factory
            3. Implement _create_step_schedulers() using create_scheduler() factory
            4. Implement _create_step_trainers() using VAETrainer/FlowTrainer
            5. Implement main training loop with transition monitoring
            6. Implement _save_checkpoint() with pipeline metadata
            7. Add integration tests

        For detailed architecture and implementation plan:
            See docs/PIPELINE_ARCHITECTURE.md
        """
        import time

        import torch
        import torch.nn as nn
        from fluxflow.utils import format_duration

        from ..training import EMA, FloatBuffer

        logger.info("Starting training pipeline execution...")

        # Print pipeline summary
        self.print_pipeline_summary()

        # Resume from checkpoint if available
        start_step, start_epoch, start_batch = self.resume_from_checkpoint()

        logger.info(
            f"Pipeline has {len(self.config.steps)} steps, starting from step {start_step + 1}"
        )

        # Get dataset size for progress tracking
        if isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
            dataset_size = getattr(dataloader.dataset, "dataset_size", 1000)
        else:
            dataset_size = len(dataloader.dataset)

        batches_per_epoch = max(1, dataset_size // args.batch_size)

        # Main pipeline loop
        for step_idx in range(start_step, len(self.config.steps)):
            step = self.config.steps[step_idx]

            print(f"\n{'='*80}")
            print(f"PIPELINE STEP {step_idx+1}/{len(self.config.steps)}: {step.name}")
            if step.description:
                print(f"Description: {step.description}")
            print(f"Duration: {step.n_epochs} epochs")
            print(f"{'='*80}\n")

            # Configure models for this step (freeze/unfreeze)
            self.configure_step_models(step, models)

            # Create optimizers and schedulers for this step
            optimizers = self._create_step_optimizers(step, models, args)
            total_steps = step.n_epochs * batches_per_epoch
            schedulers = self._create_step_schedulers(step, optimizers, total_steps)

            # Create EMA if training VAE
            ema = None
            if step.train_vae:
                ema = EMA(
                    nn.ModuleList([models["compressor"], models["expander"]]),
                    decay=0.999,
                    device=self.device,
                )

            # Create trainers for this step
            trainers = self._create_step_trainers(step, models, optimizers, schedulers, ema, args)

            # Training loop for this step
            step_start_time = time.time()

            for epoch in range(start_epoch if step_idx == start_step else 0, step.n_epochs):
                print(
                    f"\nStep {step_idx+1}/{len(self.config.steps)}, Epoch {epoch+1}/{step.n_epochs}"
                )

                epoch_start_time = time.time()

                # Error buffers for logging
                vae_errors = FloatBuffer(max(args.log_interval * 2, 10))
                kl_errors = FloatBuffer(max(args.log_interval * 2, 10))
                flow_errors = FloatBuffer(max(args.log_interval * 2, 10))

                for batch_idx, (imgs, input_ids) in enumerate(dataloader):
                    # Break if max_steps reached (for quick testing)
                    if step.max_steps is not None and batch_idx >= step.max_steps:
                        logger.info(f"Reached max_steps={step.max_steps}, ending epoch early")
                        break

                    # Skip batches if resuming mid-epoch
                    if step_idx == start_step and epoch == start_epoch and batch_idx < start_batch:
                        continue

                    self.global_step += 1
                    input_ids = input_ids.to(self.device)
                    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(self.device)

                    # Train on all resolutions
                    for ri in imgs:
                        real_imgs = ri.to(self.device).detach()

                        # VAE training
                        if step.train_vae and trainers.get("vae"):
                            vae_losses = trainers["vae"].train_step(real_imgs, self.global_step)
                            vae_errors.add_item(vae_losses["vae"])
                            kl_errors.add_item(vae_losses["kl"])

                            # Update metrics for transition monitoring
                            self.update_metrics(step.name, {"vae_loss": vae_losses["vae"]})

                        # Flow training
                        if (step.train_diff or step.train_diff_full) and trainers.get("flow"):
                            flow_losses = trainers["flow"].train_step(
                                real_imgs, input_ids, attention_mask
                            )
                            flow_loss = (
                                flow_losses["flow_loss"]
                                if isinstance(flow_losses, dict)
                                else flow_losses
                            )
                            flow_errors.add_item(flow_loss)

                            # Update metrics for transition monitoring
                            self.update_metrics(step.name, {"flow_loss": flow_loss})

                    # Logging
                    if batch_idx % args.log_interval == 0:
                        elapsed = time.time() - step_start_time
                        elapsed_str = format_duration(elapsed)

                        log_msg = f"[{elapsed_str}] Step {step_idx+1}/{len(self.config.steps)} | Epoch {epoch+1}/{step.n_epochs} | Batch {batch_idx}"

                        if step.train_vae:
                            log_msg += (
                                f" | VAE: {vae_errors.average:.4f} | KL: {kl_errors.average:.4f}"
                            )
                        if step.train_diff or step.train_diff_full:
                            log_msg += f" | Flow: {flow_errors.average:.4f}"

                        print(log_msg)

                        # Log to progress logger
                        metrics = {}
                        if step.train_vae:
                            metrics["vae_loss"] = vae_errors.average
                            metrics["kl_loss"] = kl_errors.average
                        if step.train_diff or step.train_diff_full:
                            metrics["flow_loss"] = flow_errors.average

                        progress_logger.log_metrics(
                            epoch=epoch,
                            batch=batch_idx,
                            global_step=self.global_step,
                            metrics=metrics,
                            learning_rates={},
                        )

                    # Checkpoint saving (mid-epoch)
                    if batch_idx % args.checkpoint_save_interval == 0 and batch_idx > 0:
                        self._save_checkpoint(
                            step_idx, epoch, batch_idx, models, optimizers, schedulers, ema, args
                        )

                # End-of-epoch checkpoint (always save after completing an epoch)
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1} completed in {format_duration(epoch_time)}")

                self._save_checkpoint(
                    step_idx, epoch, batch_idx, models, optimizers, schedulers, ema, args
                )
                logger.info(f"End-of-epoch checkpoint saved")

                # Check transition criteria (after saving checkpoint)
                should_trans, reason = self.should_transition(step, epoch)
                if should_trans:
                    print(f"\nTransition criteria met: {reason}")
                    print(f"Moving to next step...")
                    # Save checkpoint before transitioning
                    self._save_checkpoint(
                        step_idx, epoch, batch_idx, models, optimizers, schedulers, ema, args
                    )
                    logger.info("Pre-transition checkpoint saved")
                    break

            # Save final checkpoint at end of step
            logger.info(f"Step {step_idx+1} complete, saving final checkpoint")
            self._save_checkpoint(
                step_idx, epoch, batch_idx, models, optimizers, schedulers, ema, args
            )

            # Mark step as completed
            self.steps_completed.append(step.name)

            # Cleanup
            del optimizers, schedulers, trainers
            if ema:
                del ema
            torch.cuda.empty_cache()

            # Reset start_epoch and start_batch for next step
            start_epoch = 0
            start_batch = 0

        print(f"\n{'='*80}")
        print("PIPELINE TRAINING COMPLETE")
        print(f"{'='*80}\n")
