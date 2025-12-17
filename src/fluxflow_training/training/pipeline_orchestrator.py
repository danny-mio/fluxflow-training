"""Pipeline orchestrator for multi-step training workflows.

Manages sequential execution of training pipeline steps with model freezing,
loss-threshold transitions, and checkpoint management.
"""

import gc
from dataclasses import asdict
from typing import Any, Optional

import torch.nn as nn
from fluxflow.utils import get_logger, safe_vae_sample, save_sample_images

from .checkpoint_manager import CheckpointManager
from .pipeline_config import PipelineConfig, PipelineStepConfig

logger = get_logger(__name__)


class FastForwardDataLoader:
    """
    Wrapper that yields None for initial batches to enable fast resume without data processing.

    When resuming training mid-epoch, this wrapper returns (None, None) for the first
    `skip_batches` iterations without consuming from the underlying dataloader iterator.
    This avoids downloading tar files and decoding images for batches that will be skipped.

    CRITICAL Requirements for Training Loop:
        1. MUST use enumerate() to preserve batch_idx semantics
        2. MUST skip None batches: if imgs is None: continue
        3. Only use for resume scenarios, not general-purpose iteration

    Thread Safety: Not thread-safe. Do not share across processes.

    Example:
        >>> dataloader = DataLoader(dataset, batch_size=4)
        >>> wrapper = FastForwardDataLoader(dataloader, skip_batches=1000)
        >>> for batch_idx, (imgs, labels) in enumerate(wrapper):
        ...     if imgs is None:
        ...         continue  # Skip without processing
        ...     # Training starts at batch_idx=1000
    """

    def __init__(self, dataloader, skip_batches=0):
        """
        Args:
            dataloader: PyTorch DataLoader to wrap
            skip_batches: Number of batches to skip (return None for these)
                         If this exceeds dataloader length, no real data will be yielded.
        """
        self.dataloader = dataloader
        self.skip_batches = skip_batches
        self.batch_count = 0

        # Warn if skip might exceed dataloader length
        # Use try/except to handle IterableDatasets which may have __len__ but raise TypeError
        try:
            if hasattr(dataloader, "__len__") and skip_batches >= len(dataloader):
                logger.warning(
                    f"FastForwardDataLoader: skip_batches ({skip_batches}) >= dataloader length "
                    f"({len(dataloader)}). This epoch may yield no training data."
                )
        except TypeError:
            # IterableDataset - no length available, skip warning
            pass

    def __iter__(self):
        self.batch_count = 0
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        # For skipped batches, return None without consuming from dataloader
        if self.batch_count < self.skip_batches:
            self.batch_count += 1
            return None, None

        # Past skip point - yield real data from dataloader
        self.batch_count += 1
        return next(self.iterator)

    def __len__(self):
        """Preserve original dataloader length."""
        return len(self.dataloader) if hasattr(self.dataloader, "__len__") else 0


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

    def _create_dataloader_for_dataset(self, dataset_config, dataset_name, args, config):
        """
        Create a dataloader for a specific dataset configuration.

        Args:
            dataset_config: DatasetConfig object with dataset settings
            dataset_name: Name of the dataset (for logging)
            args: Command-line arguments
            config: Full config dictionary

        Returns:
            Tuple of (dataloader, sampler, dataset_size)
        """
        from functools import partial

        import torch
        from torch.utils.data import DataLoader

        from ..data import (
            ResumableDimensionSampler,
            StreamingWebDataset,
            TextImageDataset,
            collate_fn_variable,
            get_or_build_dimension_cache,
        )
        from ..training.utils import worker_init_fn

        # Prepare collate function
        collate_fn = partial(
            collate_fn_variable,
            channels=args.channels,
            img_size=args.img_size,
            reduced_min_sizes=args.reduced_min_sizes,
        )

        # Get batch_size and workers (with fallback priority)
        batch_size = dataset_config.batch_size or args.batch_size
        workers = dataset_config.workers or args.workers

        logger.info(f"  Batch size: {batch_size}, Workers: {workers}")

        if dataset_config.type == "webdataset":
            # WebDataset
            logger.info(f"  WebDataset URL: {dataset_config.webdataset_url}")
            dataset = StreamingWebDataset(
                tokenizer_name=args.tokenizer_name,
                token=dataset_config.webdataset_token,
                url_pattern=dataset_config.webdataset_url,
                channels=args.channels,
                image_key=dataset_config.webdataset_image_key or "png",
                label_key=dataset_config.webdataset_label_key or "json",
                caption_key=dataset_config.webdataset_caption_key or "prompt",
                dataset_size=dataset_config.webdataset_size or 10000,
                samples_per_shard=dataset_config.webdataset_samples_per_shard or 1000,
                fixed_prompt_prefix=getattr(args, "fixed_prompt_prefix", None),
            )
            sampler = None
            dataset_size = len(dataset)

        elif dataset_config.type == "local":
            # Local dataset
            logger.info(f"  Image folder: {dataset_config.image_folder}")
            logger.info(f"  Captions file: {dataset_config.captions_file}")
            dataset = TextImageDataset(
                data_path=dataset_config.image_folder,
                captions_file=dataset_config.captions_file,
                tokenizer_name=args.tokenizer_name,
                transform=None,
                fixed_prompt_prefix=getattr(args, "fixed_prompt_prefix", None),
            )

            # Build dimension cache
            dimension_cache = get_or_build_dimension_cache(
                dataset,
                cache_dir=args.output_path,
                multiple=32,
                rebuild=False,
            )

            # Create sampler
            sampler = ResumableDimensionSampler(
                dimension_cache=dimension_cache,
                batch_size=batch_size,
                seed=42,
            )
            dataset_size = len(dataset)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_config.type}")

        # Build DataLoader
        dataloader_kwargs = {
            "dataset": dataset,
            "num_workers": workers,
            "pin_memory": (not torch.backends.mps.is_available()),
            "collate_fn": collate_fn,
            "worker_init_fn": worker_init_fn,
            "persistent_workers": workers > 0,
        }

        if sampler is not None:
            dataloader_kwargs["batch_sampler"] = sampler
        else:
            dataloader_kwargs["batch_size"] = batch_size
            dataloader_kwargs["shuffle"] = False

        dataloader = DataLoader(**dataloader_kwargs)
        dataloader = self.accelerator.prepare(dataloader)

        return dataloader, sampler, dataset_size

    def _create_step_optimizers(self, step, models, args):
        """Create optimizers for current step from inline config or defaults."""
        from ..training.optimizer_factory import create_optimizer

        optimizers = {}

        if not step.optimization or not step.optimization.optimizers:
            # Create default optimizers based on training modes
            logger.info("No optimizer config found, creating defaults based on training modes")

            # Default optimizer config
            default_opt_config = {
                "type": "AdamW",
                "lr": (
                    step.lr
                    if hasattr(step, "lr") and step.lr
                    else (args.lr if hasattr(args, "lr") else 0.0001)
                ),
                "weight_decay": 0.01,
                "eps": 1e-8,
                "betas": (0.9, 0.999),
            }

            # Create VAE optimizer if training VAE/GAN/SPADE/LPIPS
            needs_vae_trainer = (
                step.train_vae or step.gan_training or step.train_spade or step.use_lpips
            )
            if needs_vae_trainer:
                vae_params = list(models["compressor"].parameters()) + list(
                    models["expander"].parameters()
                )
                optimizers["vae"] = create_optimizer(vae_params, default_opt_config)
                logger.info(
                    f"✓ Created default VAE optimizer: AdamW (lr={default_opt_config['lr']:.2e})"
                )

            # Create discriminator optimizer if GAN enabled
            # CRITICAL: Must check both gan_training AND that discriminator model exists
            if step.gan_training:
                if not models.get("D_img"):
                    raise ValueError(
                        f"Step '{step.name}' requires GAN training (gan_training=true) "
                        f"but discriminator model (D_img) is not available"
                    )
                optimizers["discriminator"] = create_optimizer(
                    models["D_img"].parameters(), default_opt_config
                )
                logger.info(
                    f"✓ Created default discriminator optimizer: AdamW (lr={default_opt_config['lr']:.2e})"
                )

            # Create flow optimizer if training flow
            if (step.train_diff or step.train_diff_full) and models.get("flow_processor"):
                optimizers["flow"] = create_optimizer(
                    models["flow_processor"].parameters(), default_opt_config
                )
                logger.info(
                    f"✓ Created default flow optimizer: AdamW (lr={default_opt_config['lr']:.2e})"
                )

            # Create text encoder optimizer if specified
            if (
                hasattr(step, "train_text_encoder")
                and step.train_text_encoder
                and models.get("text_encoder")
            ):
                optimizers["text_encoder"] = create_optimizer(
                    models["text_encoder"].parameters(), default_opt_config
                )
                logger.info(
                    f"✓ Created default text_encoder optimizer: AdamW (lr={default_opt_config['lr']:.2e})"
                )

            return optimizers

        # Explicit optimizer config provided
        for name, opt_config_obj in step.optimization.optimizers.items():
            # Convert dataclass to dict, filtering out None values
            if hasattr(opt_config_obj, "__dataclass_fields__"):
                opt_config = {k: v for k, v in asdict(opt_config_obj).items() if v is not None}
            else:
                opt_config = opt_config_obj

            # Determine which parameters to optimize
            if name == "vae":
                params = list(models["compressor"].parameters()) + list(
                    models["expander"].parameters()
                )
            elif name == "discriminator":
                params = models["D_img"].parameters()
            elif name == "flow":
                params = models["flow_processor"].parameters()
            elif name == "text_encoder":
                params = models["text_encoder"].parameters()
            else:
                logger.warning(f"Unknown optimizer name: {name}, skipping")
                continue

            optimizers[name] = create_optimizer(params, opt_config)
            logger.info(f"Created optimizer for {name}: {opt_config.get('type', 'AdamW')}")

        # CRITICAL: Check if discriminator optimizer is missing when GAN training is enabled
        # This handles the case where some optimizers are defined in config but discriminator is missing
        if step.gan_training and "discriminator" not in optimizers:
            logger.warning(
                f"Step '{step.name}' has gan_training=true but no discriminator optimizer in config. "
                f"Creating default discriminator optimizer."
            )
            if not models.get("D_img"):
                raise ValueError(
                    f"Step '{step.name}' requires GAN training (gan_training=true) "
                    f"but discriminator model (D_img) is not available"
                )

            # Use same config as VAE optimizer if available, else use defaults
            if "vae" in optimizers and step.optimization.optimizers.get("vae"):
                vae_config_obj = step.optimization.optimizers["vae"]
                if hasattr(vae_config_obj, "__dataclass_fields__"):
                    disc_opt_config = {
                        k: v for k, v in asdict(vae_config_obj).items() if v is not None
                    }
                else:
                    disc_opt_config = (
                        vae_config_obj.copy() if isinstance(vae_config_obj, dict) else {}
                    )
                logger.info(f"Using VAE optimizer config for discriminator")
            else:
                disc_opt_config = {
                    "type": "AdamW",
                    "lr": 0.0001,
                    "weight_decay": 0.01,
                    "betas": (0.9, 0.999),
                }
                logger.info(f"Using default AdamW config for discriminator")

            optimizers["discriminator"] = create_optimizer(
                models["D_img"].parameters(), disc_opt_config
            )
            logger.info(
                f"✓ Created discriminator optimizer: {disc_opt_config.get('type', 'AdamW')} "
                f"(lr={disc_opt_config.get('lr', 0.0001):.2e})"
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

            # Convert SchedulerConfig dataclass to dict, filtering out None values
            if hasattr(sched_config_obj, "__dataclass_fields__"):
                sched_config = {k: v for k, v in asdict(sched_config_obj).items() if v is not None}
            else:
                sched_config = sched_config_obj

            scheduler = create_scheduler(optimizers[name], sched_config, total_steps)
            schedulers[name] = scheduler
            logger.info(f"Created scheduler '{name}': {sched_config['type']}")

        return schedulers

    def _create_step_trainers(self, step, models, optimizers, schedulers, ema, args):
        """Create trainers for current step."""
        import torch.nn as nn

        from ..training import FlowTrainer, VAETrainer

        trainers = {}

        # Create VAE trainer if training VAE, GAN, SPADE, or LPIPS
        # (these all require the VAE trainer even if train_vae=false)
        needs_vae_trainer = (
            step.train_vae or step.gan_training or step.train_spade or step.use_lpips
        )

        if needs_vae_trainer and "vae" in optimizers:
            # Auto-add missing VAE models if they don't exist
            if "compressor" not in models or models["compressor"] is None:
                from fluxflow import FluxCompressor

                logger.warning(
                    "Compressor not found in models dict, creating new FluxCompressor. "
                    "This may not load from checkpoint - ensure models are initialized properly."
                )
                # Get vae_dim from args or use default
                vae_dim = getattr(args, "vae_dim", 128)
                models["compressor"] = FluxCompressor(
                    d_model=vae_dim,
                    use_attention=True,
                    use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", False),
                ).to(self.device)

            if "expander" not in models or models["expander"] is None:
                from fluxflow import FluxExpander

                logger.warning(
                    "Expander not found in models dict, creating new FluxExpander. "
                    "This may not load from checkpoint - ensure models are initialized properly."
                )
                vae_dim = getattr(args, "vae_dim", 128)
                models["expander"] = FluxExpander(
                    d_model=vae_dim,
                    use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", False),
                ).to(self.device)

            # Auto-add discriminator if GAN training enabled
            if step.gan_training and ("D_img" not in models or models["D_img"] is None):
                from fluxflow import PatchDiscriminator

                logger.warning(
                    "Discriminator 'D_img' not found but GAN training enabled, creating new PatchDiscriminator. "
                    "This may not load from checkpoint - ensure models are initialized properly."
                )
                channels = getattr(args, "channels", 3)
                vae_dim = getattr(args, "vae_dim", 128)
                models["D_img"] = PatchDiscriminator(in_channels=channels, ctx_dim=vae_dim).to(
                    self.device
                )

            trainers["vae"] = VAETrainer(
                compressor=models["compressor"],
                expander=models["expander"],
                optimizer=optimizers["vae"],
                scheduler=schedulers.get("vae"),
                ema=ema,
                reconstruction_loss_fn=nn.L1Loss(),
                reconstruction_loss_min_fn=nn.MSELoss(),
                use_spade=step.train_spade,
                train_reconstruction=step.train_vae,  # Only compute recon loss if train_vae=True
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
                # GAN-specific parameters (read from config, with safe defaults)
                r1_gamma=step.r1_gamma if hasattr(step, "r1_gamma") else 5.0,
                r1_interval=step.r1_interval if hasattr(step, "r1_interval") else 16,
                instance_noise_std=(
                    step.instance_noise_std if hasattr(step, "instance_noise_std") else 0.01
                ),
                instance_noise_decay=(
                    step.instance_noise_decay if hasattr(step, "instance_noise_decay") else 0.9999
                ),
                adaptive_weights=(
                    step.adaptive_weights if hasattr(step, "adaptive_weights") else True
                ),
                mse_weight=step.mse_weight if hasattr(step, "mse_weight") else 0.1,
                gradient_clip_norm=args.initial_clipping_norm,
                accelerator=self.accelerator,
            )

            # Build descriptive message about what's being trained
            modes = []
            if step.train_vae:
                modes.append("VAE")
            if step.train_spade:
                modes.append("SPADE")
            if step.gan_training:
                modes.append("GAN")
            if step.use_lpips:
                modes.append("LPIPS")

            logger.info(f"Created VAE trainer ({', '.join(modes)})")

        if (step.train_diff or step.train_diff_full) and "flow" in optimizers:
            # Auto-add missing Flow models if they don't exist
            if "flow_processor" not in models or models["flow_processor"] is None:
                from fluxflow import FluxFlowProcessor

                logger.warning(
                    "FlowProcessor not found in models dict, creating new FluxFlowProcessor. "
                    "This may not load from checkpoint - ensure models are initialized properly."
                )
                feature_maps_dim = getattr(args, "feature_maps_dim", 512)
                vae_dim = getattr(args, "vae_dim", 128)
                models["flow_processor"] = FluxFlowProcessor(
                    d_model=feature_maps_dim, vae_dim=vae_dim
                ).to(self.device)

            if "text_encoder" not in models or models["text_encoder"] is None:
                from fluxflow import BertTextEncoder

                logger.warning(
                    "TextEncoder not found in models dict, creating new BertTextEncoder. "
                    "This may not load from checkpoint - ensure models are initialized properly."
                )
                text_embedding_dim = getattr(args, "text_embedding_dim", 768)
                pretrained_bert = getattr(args, "pretrained_bert_model", "bert-base-uncased")
                models["text_encoder"] = BertTextEncoder(
                    embed_dim=text_embedding_dim, pretrain_model=pretrained_bert
                ).to(self.device)

            if "compressor" not in models or models["compressor"] is None:
                from fluxflow import FluxCompressor

                logger.warning(
                    "Compressor not found in models dict for Flow trainer, creating new FluxCompressor. "
                    "This may not load from checkpoint - ensure models are initialized properly."
                )
                vae_dim = getattr(args, "vae_dim", 128)
                models["compressor"] = FluxCompressor(
                    d_model=vae_dim,
                    use_attention=True,
                    use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", False),
                ).to(self.device)

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

    def _generate_samples(
        self, step, step_idx, epoch, batch_idx, models, tokenizer, args, parsed_sample_sizes
    ):
        """
        Generate sample images for monitoring training progress.

        Args:
            step: Current pipeline step config
            step_idx: Current step index
            epoch: Current epoch within step
            batch_idx: Current batch index
            models: Dictionary of models
            tokenizer: Tokenizer instance
            args: Training arguments
            parsed_sample_sizes: List of sample size tuples
        """
        import glob
        import os

        if args.no_samples:
            return

        diffuser = models.get("diffuser")
        text_encoder = models.get("text_encoder")
        if not diffuser:
            logger.warning("Cannot generate samples: diffuser model not found")
            return

        # Sample epoch identifier (use global_step for legacy compatibility)
        sample_epoch = self.global_step

        logger.info(
            f"Generating samples for step {step_idx+1}/{len(self.config.steps)}, "
            f"epoch {epoch+1}, batch {batch_idx}, global_step {sample_epoch}"
        )

        # Create prefix for new naming: stepname_step_epoch_batch
        step_name_short = step.name[:20]  # Limit step name length
        # Include batch for mid-epoch samples, omit for end-of-epoch (batch=-1 or max batch)
        if batch_idx >= 0 and batch_idx < 999999:  # Mid-epoch
            sample_prefix = f"{step_name_short}_{step_idx+1:03d}_{epoch+1:03d}_{batch_idx:05d}"
        else:  # End-of-epoch or initial
            sample_prefix = f"{step_name_short}_{step_idx+1:03d}_{epoch+1:03d}"

        # VAE reconstruction samples (if test images provided)
        if args.test_image_address and len(args.test_image_address) > 0:
            for img_addr in args.test_image_address:
                try:
                    # Generate samples with epoch number (legacy naming)
                    safe_vae_sample(
                        diffuser,
                        img_addr,
                        args.channels if hasattr(args, "channels") else 3,
                        args.output_path,
                        sample_epoch,
                        self.device,
                    )

                    # Rename files to include step/epoch info
                    # Pattern: vae_epoch_0031-{hash}-{suffix}.webp → {prefix}_{hash}-{suffix}.webp
                    pattern = os.path.join(args.output_path, f"vae_epoch_{sample_epoch:04d}-*.webp")
                    for old_file in glob.glob(pattern):
                        old_filename = os.path.basename(old_file)
                        # Extract hash and suffix: "vae_epoch_0031-cd7a10...-ctx.webp"
                        # Split on first '-' after epoch number
                        parts = old_filename.replace(f"vae_epoch_{sample_epoch:04d}-", "")
                        # parts = "cd7a10...-ctx.webp"
                        hash_suffix = parts.replace(".webp", "")

                        # New filename: stepname_step_epoch_hash-suffix.webp
                        new_filename = f"{sample_prefix}_{hash_suffix}.webp"
                        new_file = os.path.join(args.output_path, new_filename)

                        os.rename(old_file, new_file)
                        logger.debug(f"Renamed: {old_filename} → {new_filename}")

                except Exception as e:
                    logger.warning(f"Failed to generate VAE sample from {img_addr}: {e}")

        # Flow-based text-to-image samples (if flow training active)
        if (step.train_diff or step.train_diff_full) and text_encoder:
            if args.sample_captions and len(args.sample_captions) > 0:
                try:
                    # Generate samples with epoch number (legacy naming)
                    save_sample_images(
                        diffuser,
                        text_encoder,
                        tokenizer,
                        args.output_path,
                        sample_epoch,
                        self.device,
                        args.sample_captions,
                        args.batch_size,
                        sample_sizes=parsed_sample_sizes,
                    )

                    # Rename files to include step/epoch/batch info
                    # Pattern: sample_epoch_<N>_<caption_idx>_<size>.png
                    pattern = os.path.join(args.output_path, f"sample_epoch_{sample_epoch}_*.png")
                    for old_file in glob.glob(pattern):
                        old_filename = os.path.basename(old_file)
                        # Extract caption index and size: "sample_epoch_247_0_512x512.png"
                        parts = old_filename.replace(f"sample_epoch_{sample_epoch}_", "").replace(
                            ".png", ""
                        )
                        # parts = "0_512x512" or just "0"

                        # New filename: stepname_step_epoch_batch_captionidx_size.png
                        new_filename = f"{sample_prefix}_{parts}.png"
                        new_file = os.path.join(args.output_path, new_filename)

                        os.rename(old_file, new_file)
                        logger.debug(f"Renamed: {old_filename} → {new_filename}")

                except Exception as e:
                    logger.warning(f"Failed to generate flow samples: {e}")

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

        # Parse sample sizes for sample generation
        from ..scripts.train import parse_sample_sizes

        parsed_sample_sizes = parse_sample_sizes(
            config.get("output", {}).get("sample_sizes", [512])
        )

        # Get dataset size for progress tracking
        if isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
            dataset_size = getattr(dataloader.dataset, "dataset_size", 1000)
        else:
            dataset_size = len(dataloader.dataset)

        batches_per_epoch = max(1, dataset_size // args.batch_size)

        # Generate initial samples (before training starts)
        if start_step == 0 and start_epoch == 0:
            logger.info("Generating initial samples before training...")
            self._generate_samples(
                self.config.steps[0], 0, -1, 0, models, tokenizer, args, parsed_sample_sizes
            )

        # Main pipeline loop
        for step_idx in range(start_step, len(self.config.steps)):
            step = self.config.steps[step_idx]

            print(f"\n{'='*80}")
            print(f"PIPELINE STEP {step_idx+1}/{len(self.config.steps)}: {step.name}")
            if step.description:
                print(f"Description: {step.description}")
            print(f"Duration: {step.n_epochs} epochs")
            print(f"{'='*80}\n")

            # Switch dataset if needed for this step
            current_dataset_name = step.dataset or self.config.default_dataset
            if current_dataset_name and self.config.datasets:
                dataset_config = self.config.datasets.get(current_dataset_name)
                if dataset_config:
                    logger.info(f"Dataset: {current_dataset_name} ({dataset_config.type})")
                    print(f"Dataset: {current_dataset_name} ({dataset_config.type})")

                    # Recreate dataloader for this dataset if it's different from previous step
                    needs_new_dataloader = step_idx == start_step or (
                        step_idx > 0
                        and (self.config.steps[step_idx - 1].dataset or self.config.default_dataset)
                        != current_dataset_name
                    )

                    if needs_new_dataloader:
                        logger.info(f"Creating dataloader for dataset: {current_dataset_name}")
                        dataloader, sampler, dataset_size = self._create_dataloader_for_dataset(
                            dataset_config, current_dataset_name, args, config
                        )
                        batches_per_epoch = max(1, dataset_size // args.batch_size)
                        logger.info(
                            f"Dataset size: {dataset_size:,} samples, {batches_per_epoch} batches/epoch"
                        )
            else:
                logger.info("Using default dataloader (no dataset specified for this step)")

            # Configure models for this step (freeze/unfreeze)
            self.configure_step_models(step, models)

            # Update progress logger for this step (step-specific files)
            progress_logger.set_step(step.name)
            logger.info(f"Progress logging to: {progress_logger.metrics_file}")

            # Create optimizers and schedulers for this step
            optimizers = self._create_step_optimizers(step, models, args)
            total_steps = step.n_epochs * batches_per_epoch
            schedulers = self._create_step_schedulers(step, optimizers, total_steps)

            # Create EMA if training VAE
            # Create EMA if we need VAE trainer (for VAE, GAN, SPADE, or LPIPS)
            needs_vae_trainer = step.train_vae or step.train_spade or step.use_lpips
            ema = None
            if needs_vae_trainer and step.use_ema:
                ema = EMA(
                    nn.ModuleList([models["compressor"], models["expander"]]),
                    decay=0.999,
                    device=self.device,
                )
                logger.info("✓ EMA enabled (adds 2x model VRAM)")
            elif needs_vae_trainer and not step.use_ema:
                logger.info("⚠ EMA disabled to save VRAM (~14GB for vae_dim=128)")

            # Load optimizer/scheduler/EMA states when resuming
            # CRITICAL: Only load for the step we're resuming from
            if step_idx == start_step and (start_epoch > 0 or start_batch > 0):
                logger.info(f"Resuming from step {step_idx}, loading optimizer/scheduler states...")
                loaded = self.checkpoint_manager.load_optimizer_scheduler_ema_states(
                    optimizers=optimizers,
                    schedulers=schedulers,
                    ema=ema,
                )
                if loaded:
                    logger.info("✓ Restored optimizer, scheduler, and EMA states from checkpoint")
                else:
                    logger.warning(
                        "⚠ Could not load optimizer states, starting with fresh optimizers"
                    )

            # Create trainers for this step
            trainers = self._create_step_trainers(step, models, optimizers, schedulers, ema, args)

            # Training loop for this step
            step_start_time = time.time()

            # Initialize epoch and batch_idx in case loop exits early (e.g., max_steps < batches_per_epoch)
            epoch = start_epoch if step_idx == start_step else 0
            batch_idx = 0

            for epoch in range(start_epoch if step_idx == start_step else 0, step.n_epochs):
                # Calculate total batches for this epoch (considering max_steps)
                epoch_total_batches = (
                    min(batches_per_epoch, step.max_steps) if step.max_steps else batches_per_epoch
                )

                # Show which dataset is being used
                current_dataset_name = step.dataset or self.config.default_dataset
                dataset_info = f", Dataset: {current_dataset_name}" if current_dataset_name else ""

                print(
                    f"\nStep {step.name} ({step_idx+1}/{len(self.config.steps)}), "
                    f"Epoch {epoch+1}/{step.n_epochs}, "
                    f"Batches 0/{epoch_total_batches}{dataset_info}"
                )

                # Wrap dataloader with fast-forward if resuming mid-epoch
                # This allows batch_idx to increment correctly without downloading/processing skipped batches
                dataloader_for_epoch = dataloader
                if step_idx == start_step and epoch == start_epoch and start_batch > 0:
                    logger.info(
                        f"Fast-forwarding to batch {start_batch} "
                        f"(skipping without downloading/processing)"
                    )
                    dataloader_for_epoch = FastForwardDataLoader(
                        dataloader, skip_batches=start_batch
                    )

                epoch_start_time = time.time()

                # Error buffers for logging
                vae_errors = FloatBuffer(max(args.log_interval * 2, 10))
                kl_errors = FloatBuffer(max(args.log_interval * 2, 10))
                flow_errors = FloatBuffer(max(args.log_interval * 2, 10))
                g_errors = FloatBuffer(max(args.log_interval * 2, 10))  # GAN generator loss
                d_errors = FloatBuffer(max(args.log_interval * 2, 10))  # GAN discriminator loss
                lpips_errors = FloatBuffer(max(args.log_interval * 2, 10))  # LPIPS loss
                bezier_reg_errors = FloatBuffer(
                    max(args.log_interval * 2, 10)
                )  # Bezier regularization
                color_stats_errors = FloatBuffer(
                    max(args.log_interval * 2, 10)
                )  # Color statistics loss
                hist_loss_errors = FloatBuffer(max(args.log_interval * 2, 10))  # Histogram loss
                contrast_errors = FloatBuffer(max(args.log_interval * 2, 10))  # Contrast loss
                batch_times = FloatBuffer(max(args.log_interval * 2, 10))  # Batch timing

                for batch_idx, (imgs, input_ids) in enumerate(dataloader_for_epoch):
                    # Skip None batches from FastForwardDataLoader (fast-forward mode)
                    if imgs is None:
                        continue

                    batch_start_time = time.time()
                    # Break if max_steps reached (for quick testing)
                    if step.max_steps is not None and batch_idx >= step.max_steps:
                        logger.info(f"Reached max_steps={step.max_steps}, ending epoch early")
                        break

                    self.global_step += 1
                    input_ids = input_ids.to(self.device)
                    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(self.device)

                    # Train on all resolutions
                    for ri in imgs:
                        real_imgs = ri.to(self.device).detach()

                        # VAE/GAN/SPADE training (runs if trainer exists, even with train_vae=false)
                        if trainers.get("vae"):
                            vae_losses = trainers["vae"].train_step(real_imgs, self.global_step)
                            vae_errors.add_item(vae_losses["vae"])
                            kl_errors.add_item(vae_losses["kl"])

                            # Track GAN losses if available
                            if "generator" in vae_losses:
                                g_errors.add_item(vae_losses["generator"])
                            if "discriminator" in vae_losses:
                                d_errors.add_item(vae_losses["discriminator"])
                            if "lpips" in vae_losses:
                                lpips_errors.add_item(vae_losses["lpips"])

                            # Track VAE regularization losses if available
                            if "bezier_reg" in vae_losses:
                                bezier_reg_errors.add_item(vae_losses["bezier_reg"])
                            if "color_stats" in vae_losses:
                                color_stats_errors.add_item(vae_losses["color_stats"])
                            if "hist_loss" in vae_losses:
                                hist_loss_errors.add_item(vae_losses["hist_loss"])
                            if "contrast_loss" in vae_losses:
                                contrast_errors.add_item(vae_losses["contrast_loss"])

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

                        # Critical: Delete tensors immediately after use to prevent accumulation
                        del real_imgs

                    # Delete batch tensors after processing all resolutions
                    del imgs, input_ids, attention_mask

                    # Track batch time
                    batch_time = time.time() - batch_start_time
                    batch_times.add_item(batch_time)

                    # Periodic cache clearing to prevent fragmentation (every 10 batches)
                    # Works for both CUDA and MPS backends
                    if batch_idx % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                        # Force Python garbage collection periodically

                        gc.collect()

                    # Deep memory cleanup every 100 batches to prevent gradual accumulation
                    # This is critical for long training runs (hours/days)
                    if batch_idx % 100 == 0 and batch_idx > 0:

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            # Force synchronization to ensure all GPU operations complete
                            torch.cuda.synchronize()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                            # MPS-specific: force synchronization
                            torch.mps.synchronize()

                        logger.info(
                            f"Deep memory cleanup at batch {batch_idx} "
                            f"(global_step={self.global_step})"
                        )

                    # Logging
                    if batch_idx % args.log_interval == 0:
                        elapsed = time.time() - step_start_time
                        elapsed_str = format_duration(elapsed)

                        log_msg = (
                            f"[{elapsed_str}] Step {step.name} ({step_idx+1}/{len(self.config.steps)}) | "
                            f"Epoch {epoch+1}/{step.n_epochs} | "
                            f"Batch {batch_idx}/{epoch_total_batches}"
                        )

                        # Console logging (show metrics if VAE trainer ran)
                        if len(vae_errors._items) > 0:
                            log_msg += (
                                f" | VAE: {vae_errors.average:.4f} | KL: {kl_errors.average:.4f}"
                            )
                            # Add GAN losses if active
                            if step.gan_training and len(g_errors._items) > 0:
                                log_msg += (
                                    f" | G: {g_errors.average:.4f} | D: {d_errors.average:.4f}"
                                )
                            # Add LPIPS if active
                            if step.use_lpips and len(lpips_errors._items) > 0:
                                log_msg += f" | LPIPS: {lpips_errors.average:.4f}"
                            # Add VAE regularization losses if available
                            if len(bezier_reg_errors._items) > 0:
                                log_msg += f" | Bezier: {bezier_reg_errors.average:.4f}"
                            if len(color_stats_errors._items) > 0:
                                log_msg += f" | ColorStats: {color_stats_errors.average:.4f}"
                            if len(hist_loss_errors._items) > 0:
                                log_msg += f" | Hist: {hist_loss_errors.average:.4f}"
                            if len(contrast_errors._items) > 0:
                                log_msg += f" | Contrast: {contrast_errors.average:.4f}"

                        if step.train_diff or step.train_diff_full:
                            log_msg += f" | Flow: {flow_errors.average:.4f}"

                        # Add average batch time
                        if len(batch_times._items) > 0:
                            log_msg += f" | {batch_times.average:.2f}s/batch"

                        print(log_msg)

                        # Log to progress logger (include metrics if VAE trainer ran)
                        metrics = {}
                        if len(vae_errors._items) > 0:
                            metrics["vae_loss"] = vae_errors.average
                            metrics["kl_loss"] = kl_errors.average
                            # Add GAN metrics
                            if step.gan_training and len(g_errors._items) > 0:
                                metrics["g_loss"] = g_errors.average
                                metrics["d_loss"] = d_errors.average
                            # Add LPIPS metrics
                            if step.use_lpips and len(lpips_errors._items) > 0:
                                metrics["lpips_loss"] = lpips_errors.average
                            # Add VAE regularization metrics
                            if len(bezier_reg_errors._items) > 0:
                                metrics["bezier_reg"] = bezier_reg_errors.average
                            if len(color_stats_errors._items) > 0:
                                metrics["color_stats"] = color_stats_errors.average
                            if len(hist_loss_errors._items) > 0:
                                metrics["hist_loss"] = hist_loss_errors.average
                            if len(contrast_errors._items) > 0:
                                metrics["contrast_loss"] = contrast_errors.average

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

                        # Clear cache after checkpoint save to prevent fragmentation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                        # Force garbage collection after checkpoint

                        gc.collect()

                        # Generate samples at checkpoint intervals if requested
                        if args.samples_per_checkpoint > 0:
                            self._generate_samples(
                                step,
                                step_idx,
                                epoch,
                                batch_idx,
                                models,
                                tokenizer,
                                args,
                                parsed_sample_sizes,
                            )

                            # Clear cache after sample generation
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            elif torch.backends.mps.is_available():
                                torch.mps.empty_cache()

                # End-of-epoch checkpoint (always save after completing an epoch)
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1} completed in {format_duration(epoch_time)}")

                self._save_checkpoint(
                    step_idx, epoch, batch_idx, models, optimizers, schedulers, ema, args
                )
                logger.info(f"End-of-epoch checkpoint saved")

                # Generate samples after epoch completes (use last batch_idx)
                self._generate_samples(
                    step, step_idx, epoch, batch_idx, models, tokenizer, args, parsed_sample_sizes
                )

                # Aggressive memory cleanup at end of epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                gc.collect()

                # Clear error buffers to prevent accumulation
                del (
                    vae_errors,
                    kl_errors,
                    flow_errors,
                    g_errors,
                    d_errors,
                    lpips_errors,
                    batch_times,
                )

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
