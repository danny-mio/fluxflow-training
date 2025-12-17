"""FluxFlow unified training script."""

import argparse
import json
import os
import random
import sys
import time
import warnings
from functools import partial
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from accelerate import Accelerator
from fluxflow.models import (
    BertTextEncoder,
    FluxCompressor,
    FluxExpander,
    FluxFlowProcessor,
    FluxPipeline,
    ImageEncoder,
    PatchDiscriminator,
)
from fluxflow.utils import (
    format_duration,
    safe_vae_sample,
    save_sample_images,
)
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from fluxflow_training.data import (
    ResumableDimensionSampler,
    StreamingWebDataset,
    TextImageDataset,
    collate_fn_variable,
    get_or_build_dimension_cache,
)
from fluxflow_training.training import (
    EMA,
    CheckpointManager,
    FloatBuffer,
    FlowTrainer,
    RobustDataLoaderIterator,
    TrainingPipelineOrchestrator,
    TrainingProgressLogger,
    VAETrainer,
    cosine_anneal_beta,
    current_lr,
    parse_pipeline_config,
    worker_init_fn,
)
from fluxflow_training.training.optimizer_factory import (
    create_optimizer,
    get_default_optimizer_config,
)
from fluxflow_training.training.scheduler_factory import (
    create_scheduler,
    get_default_scheduler_config,
)

# Environment setup
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast(args...)` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.custom_fwd(args...)` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings("error", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def parse_sample_sizes(sample_sizes_arg):
    """
    Parse sample sizes from CLI argument or config.

    Args:
        sample_sizes_arg: List of size specs (int, str "WxH", list/tuple [W, H])

    Returns:
        List of parsed sizes (int for square, tuple for WxH)
        Returns None if input is None (use defaults)

    Examples:
        ["512", "768x512", "1024"] -> [512, (768, 512), 1024]
        [512, [768, 512]] -> [512, (768, 512)]
    """
    if sample_sizes_arg is None:
        return None

    parsed = []
    for spec in sample_sizes_arg:
        if isinstance(spec, (list, tuple)):
            # Already a tuple/list [W, H]
            parsed.append(tuple(spec))
        elif isinstance(spec, int):
            # Integer for square image
            parsed.append(spec)
        elif isinstance(spec, str):
            if "x" in spec.lower():
                # "WxH" format
                parts = spec.lower().split("x")
                parsed.append((int(parts[0]), int(parts[1])))
            else:
                # String number for square image
                parsed.append(int(spec))
        else:
            raise ValueError(f"Invalid sample size spec: {spec}")

    return parsed


def load_optimizer_scheduler_config(args, lr):
    """
    Load or create optimizer and scheduler configurations.

    Args:
        args: Parsed command-line arguments
        lr: Dictionary with learning rates {"lr": flow_lr, "vae": vae_lr}

    Returns:
        Tuple of (optimizer_configs, scheduler_configs) dictionaries
    """
    # Check if config file is provided
    if hasattr(args, "optim_sched_config") and args.optim_sched_config:
        config_path = args.optim_sched_config
        if os.path.exists(config_path):
            print(f"Loading optimizer/scheduler config from {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)
            optimizer_configs = config.get("optimizers", {})
            scheduler_configs = config.get("schedulers", {})
        else:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            optimizer_configs = {}
            scheduler_configs = {}
    else:
        # Use defaults
        optimizer_configs = {}
        scheduler_configs = {}

    # Fill in defaults for missing configurations
    model_names = ["flow", "vae", "text_encoder", "discriminator"]

    for model_name in model_names:
        # Check if we need to use defaults for this model
        use_default_lr = model_name not in optimizer_configs

        if use_default_lr:
            optimizer_configs[model_name] = get_default_optimizer_config(model_name)

        # Override LR with command-line arguments only if:
        # 1. No custom config was provided for this model (use_default_lr), OR
        # 2. Custom config exists but doesn't specify LR
        should_override_lr = use_default_lr or "lr" not in optimizer_configs[model_name]

        if should_override_lr:
            if model_name == "flow":
                optimizer_configs[model_name]["lr"] = lr["lr"]
            elif model_name == "vae":
                optimizer_configs[model_name]["lr"] = lr["vae"]
            elif model_name == "text_encoder":
                optimizer_configs[model_name]["lr"] = lr["lr"] / 10
            elif model_name == "discriminator":
                optimizer_configs[model_name]["lr"] = lr["vae"]

        # Handle scheduler configuration
        use_default_scheduler = model_name not in scheduler_configs

        if use_default_scheduler:
            scheduler_configs[model_name] = get_default_scheduler_config(model_name)

        # Use lr_min CLI argument as eta_min_factor if not specified in config
        # This connects the --lr_min CLI argument to the scheduler configuration
        if "eta_min_factor" not in scheduler_configs[model_name]:
            lr_min = getattr(args, "lr_min", 0.1)
            scheduler_configs[model_name]["eta_min_factor"] = lr_min

    return optimizer_configs, scheduler_configs


def initialize_models(args, config, device, checkpoint_manager):
    """
    Initialize FluxFlow models from configuration.

    Args:
        args: Parsed command-line arguments
        config: Loaded YAML config dictionary
        device: Target device (CPU/CUDA/MPS)
        checkpoint_manager: CheckpointManager instance for loading checkpoints

    Returns:
        Dict containing initialized models
    """
    channels = args.channels

    # Initialize models
    text_encoder = BertTextEncoder(
        embed_dim=args.text_embedding_dim, pretrain_model=args.pretrained_bert_model
    )
    image_encoder = ImageEncoder(
        channels,
        text_embedding_dim=args.text_embedding_dim,
        feature_maps=args.feature_maps_dim_disc,
    )

    compressor = FluxCompressor(
        d_model=args.vae_dim,
        use_attention=True,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    expander = FluxExpander(
        d_model=args.vae_dim, use_gradient_checkpointing=args.use_gradient_checkpointing
    )
    flow_processor = FluxFlowProcessor(d_model=args.feature_maps_dim, vae_dim=args.vae_dim)
    diffuser = FluxPipeline(compressor, flow_processor, expander)

    # Discriminators
    D_img = PatchDiscriminator(in_channels=args.channels, ctx_dim=args.vae_dim)

    # Load checkpoints if resuming
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        loaded_states = checkpoint_manager.load_models_parallel(
            checkpoint_path=args.model_checkpoint
        )

        if loaded_states.get("diffuser.compressor"):
            compressor.load_state_dict(loaded_states["diffuser.compressor"], strict=False)  # type: ignore[arg-type]
            print("✓ Loaded compressor checkpoint")
        if loaded_states.get("diffuser.flow_processor"):
            flow_processor.load_state_dict(loaded_states["diffuser.flow_processor"], strict=False)  # type: ignore[arg-type]
            print("✓ Loaded flow_processor checkpoint")
        if loaded_states.get("diffuser.expander"):
            expander.load_state_dict(loaded_states["diffuser.expander"], strict=False)  # type: ignore[arg-type]
            print("✓ Loaded expander checkpoint")
        if loaded_states.get("text_encoder"):
            text_encoder.load_state_dict(loaded_states["text_encoder"], strict=False)  # type: ignore[arg-type]
            print("✓ Loaded text_encoder checkpoint")
        if loaded_states.get("image_encoder"):
            image_encoder.load_state_dict(loaded_states["image_encoder"], strict=False)  # type: ignore[arg-type]
            print("✓ Loaded image_encoder checkpoint")
        if loaded_states.get("D_img"):
            D_img.load_state_dict(loaded_states["D_img"], strict=False)  # type: ignore[arg-type]
            print("✓ Loaded D_img checkpoint")

            # Validate discriminator weights for NaN/Inf
            nan_found = False
            for name, param in D_img.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"  ⚠️  WARNING: NaN/Inf in D_img parameter: {name}")
                    nan_found = True
            if nan_found:
                print("  ⚠️  Reinitializing discriminator due to NaN/Inf values")
                D_img = PatchDiscriminator(in_channels=args.channels, ctx_dim=args.vae_dim)

    # Move to device
    diffuser.to(device)
    text_encoder.to(device)
    image_encoder.to(device)
    D_img.to(device)

    return {
        "diffuser": diffuser,
        "compressor": compressor,
        "expander": expander,
        "flow_processor": flow_processor,
        "text_encoder": text_encoder,
        "image_encoder": image_encoder,
        "D_img": D_img,
    }


def initialize_tokenizer(args):
    """
    Initialize tokenizer from configuration.

    Args:
        args: Parsed command-line arguments

    Returns:
        Initialized tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, cache_dir="./_cache", local_files_only=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def initialize_dataloader(args, accelerator):
    """
    Initialize dataset and dataloader from configuration.

    Args:
        args: Parsed command-line arguments
        accelerator: Accelerator instance

    Returns:
        Tuple of (dataloader, sampler, dataset_size)
    """
    collate_fn = partial(
        collate_fn_variable,
        channels=args.channels,
        img_size=args.img_size,
        reduced_min_sizes=args.reduced_min_sizes,
    )

    if args.use_webdataset:
        print(f"Using WebDataset streaming from: {args.webdataset_url}")
        dataset = StreamingWebDataset(
            tokenizer_name=args.tokenizer_name,
            token=args.webdataset_token,
            url_pattern=args.webdataset_url,
            channels=args.channels,
            image_key=args.webdataset_image_key,
            label_key=args.webdataset_label_key,
            caption_key=args.webdataset_caption_key,
            dataset_size=args.webdataset_size,
            samples_per_shard=args.webdataset_samples_per_shard,
            fixed_prompt_prefix=args.fixed_prompt_prefix,
        )
        print(f"  Shards: {dataset.num_shards}, Estimated size: {dataset.dataset_size:,} samples")
        if args.fixed_prompt_prefix:
            print(f"  Fixed prompt prefix: '{args.fixed_prompt_prefix}'")
        sampler = None
        dataset_size = len(dataset)
    else:
        dataset = TextImageDataset(
            data_path=args.data_path,
            captions_file=args.captions_file,
            tokenizer_name=args.tokenizer_name,
            transform=None,
            fixed_prompt_prefix=args.fixed_prompt_prefix,
        )
        if args.fixed_prompt_prefix:
            print(f"  Fixed prompt prefix: '{args.fixed_prompt_prefix}'")

        # Build dimension cache
        dimension_cache = get_or_build_dimension_cache(
            dataset, cache_dir=args.output_path, multiple=32, rebuild=False
        )

        # Create sampler
        sampler = ResumableDimensionSampler(
            dimension_cache=dimension_cache,
            batch_size=args.batch_size,
            seed=42,
        )
        dataset_size = len(dataset)

    # Build DataLoader
    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": args.workers,
        "pin_memory": (not torch.backends.mps.is_available()),
        "collate_fn": collate_fn,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": args.workers > 0,
        # prefetch_factor removed - with pin_memory=True, prefetching uses GPU-pinned memory
        # which caused OOM with large batches (user reported 47GB usage vs 48GB capacity)
    }

    if sampler is not None:
        dataloader_kwargs["batch_sampler"] = sampler
    else:
        dataloader_kwargs["batch_size"] = args.batch_size
        dataloader_kwargs["shuffle"] = False

    dataloader = DataLoader(**dataloader_kwargs)
    dataloader = accelerator.prepare(dataloader)

    return dataloader, sampler, dataset_size


def train_pipeline(args, config):
    """
    Pipeline-based training loop for FluxFlow.

    Args:
        args: Parsed command-line arguments
        config: Loaded YAML config dictionary
    """
    print("\n" + "=" * 80)
    print("PIPELINE MODE - Multi-step training enabled")
    print("=" * 80 + "\n")

    # Parse pipeline configuration
    pipeline_config = parse_pipeline_config(config["training"]["pipeline"])

    # Set random seed
    manualSeed = random.randint(1, sys.maxsize)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Initialize accelerator
    if torch.backends.mps.is_available():
        print("Using MPS backend.")
        accelerator = Accelerator(cpu=False, device_placement=True)
    elif torch.cuda.is_available():
        print("Using CUDA backend.")
        accelerator = Accelerator(
            cpu=False,
            device_placement=True,
            mixed_precision="fp16" if args.use_fp16 else "no",
        )
    else:
        print("Using CPU")
        accelerator = Accelerator(cpu=True, mixed_precision="fp16" if args.use_fp16 else "no")

    device = accelerator.device

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=args.output_path, generate_diagrams=args.generate_diagrams
    )

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(args)

    # Initialize models
    print("\nInitializing models...")
    models = initialize_models(args, config, device, checkpoint_manager)

    # Initialize dataloader
    print("\nInitializing dataset and dataloader...")
    dataloader, sampler, dataset_size = initialize_dataloader(args, accelerator)
    print(f"Dataset size: {dataset_size:,} samples")

    # Initialize pipeline orchestrator
    orchestrator = TrainingPipelineOrchestrator(
        pipeline_config=pipeline_config,
        checkpoint_manager=checkpoint_manager,
        accelerator=accelerator,
        device=device,
    )

    # Initialize progress logger
    progress_logger = TrainingProgressLogger(args.output_path)
    print(f"Training progress will be logged to: {progress_logger.get_graph_dir()}")

    # Run pipeline
    print("\nStarting pipeline training...")
    orchestrator.run(
        models=models,
        dataloader=dataloader,
        sampler=sampler,
        tokenizer=tokenizer,
        progress_logger=progress_logger,
        args=args,
        config=config,
    )

    print("\n" + "=" * 80)
    print("Pipeline training complete!")
    print("=" * 80 + "\n")


def train_legacy(args):
    """Legacy single-mode training loop for FluxFlow (backward compatibility)."""
    print("\n" + "=" * 80)
    print("LEGACY MODE - Single-step training (consider migrating to pipeline mode)")
    print("=" * 80 + "\n")
    # Load/save learning rates and global step
    LR_SAVE_FILE = os.path.join(args.output_path, "lr_sav.json")
    TRAINING_STATE_FILE = os.path.join(args.output_path, "training_state.json")

    lr = {"lr": args.lr, "vae": args.lr}
    saved_global_step = 0
    saved_epoch = 0
    saved_batch_idx = 0

    if args.preserve_lr and os.path.exists(LR_SAVE_FILE):
        with open(LR_SAVE_FILE, "r") as f:
            loaded_lr = json.load(f)
            # Handle both old format {"lr": ..., "vae": ...} and new format from CheckpointManager
            if "lr" in loaded_lr and "vae" in loaded_lr:
                # Old format - use as is
                lr = loaded_lr
            elif "optimizer_FDiff" in loaded_lr or "optimizer_Diff" in loaded_lr:
                # New CheckpointManager format - convert to expected format
                lr = {
                    "lr": loaded_lr.get("optimizer_FDiff", args.lr),
                    "vae": loaded_lr.get("optimizer_Diff", args.lr),
                }
            else:
                # Unknown format - use defaults
                print(f"Warning: Unexpected lr_sav.json format, using defaults")
                lr = {"lr": args.lr, "vae": args.lr}

    # Load training state from checkpoint if available
    saved_last_sample_step = 0
    if os.path.exists(TRAINING_STATE_FILE):
        try:
            with open(TRAINING_STATE_FILE, "r") as f:
                training_state = json.load(f)
                saved_global_step = training_state.get("global_step", 0)
                saved_epoch = training_state.get("epoch", 0)
                saved_batch_idx = training_state.get("batch_idx", 0)
                saved_last_sample_step = training_state.get("last_sample_step", 0)
                print(
                    f"Resuming from epoch {saved_epoch}, batch {saved_batch_idx}, global step: {saved_global_step}"
                )
        except Exception as e:
            print(f"Warning: Could not load training state: {e}")

    # Initialize training progress logger
    progress_logger = TrainingProgressLogger(args.output_path)
    print(f"Training progress will be logged to: {progress_logger.get_graph_dir()}")

    # Initialize accelerator
    if torch.backends.mps.is_available():
        print("Using MPS backend.")
        accelerator = Accelerator(cpu=False, device_placement=True)
    elif torch.cuda.is_available():
        print("Using CUDA backend.")
        accelerator = Accelerator(
            cpu=False,
            device_placement=True,
            mixed_precision="fp16" if args.use_fp16 else "no",
        )
    else:
        print("Using CPU")
        accelerator = Accelerator(cpu=True, mixed_precision="fp16" if args.use_fp16 else "no")

    # Set random seed
    manualSeed = random.randint(1, sys.maxsize)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = accelerator.device
    channels = args.channels

    # Load tokenizer (uses cache if present, otherwise downloads)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, cache_dir="./_cache", local_files_only=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)

    # Initialize models
    text_encoder = BertTextEncoder(
        embed_dim=args.text_embedding_dim, pretrain_model=args.pretrained_bert_model
    )
    image_encoder = ImageEncoder(
        channels,
        text_embedding_dim=args.text_embedding_dim,
        feature_maps=args.feature_maps_dim_disc,
    )

    compressor = FluxCompressor(
        d_model=args.vae_dim,
        use_attention=True,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    expander = FluxExpander(
        d_model=args.vae_dim, use_gradient_checkpointing=args.use_gradient_checkpointing
    )
    flow_processor = FluxFlowProcessor(d_model=args.feature_maps_dim, vae_dim=args.vae_dim)
    diffuser = FluxPipeline(compressor, flow_processor, expander)

    # Discriminators
    D_img = PatchDiscriminator(in_channels=args.channels, ctx_dim=args.vae_dim)

    # Initialize checkpoint manager for easier model management
    checkpoint_manager = CheckpointManager(
        output_dir=args.output_path, generate_diagrams=args.generate_diagrams
    )

    # Load checkpoints in parallel for faster resume
    loaded_states = checkpoint_manager.load_models_parallel(checkpoint_path=args.model_checkpoint)

    # Apply loaded state dicts to models if they exist
    if loaded_states.get("diffuser.compressor"):
        compressor.load_state_dict(loaded_states["diffuser.compressor"], strict=False)  # type: ignore[arg-type]
        print("✓ Loaded compressor checkpoint")
    if loaded_states.get("diffuser.flow_processor"):
        flow_processor.load_state_dict(loaded_states["diffuser.flow_processor"], strict=False)  # type: ignore[arg-type]
        print("✓ Loaded flow_processor checkpoint")
    if loaded_states.get("diffuser.expander"):
        expander.load_state_dict(loaded_states["diffuser.expander"], strict=False)  # type: ignore[arg-type]
        print("✓ Loaded expander checkpoint")
    if loaded_states.get("text_encoder"):
        text_encoder.load_state_dict(loaded_states["text_encoder"], strict=False)  # type: ignore[arg-type]
        print("✓ Loaded text_encoder checkpoint")
    if loaded_states.get("image_encoder"):
        image_encoder.load_state_dict(loaded_states["image_encoder"], strict=False)  # type: ignore[arg-type]
        print("✓ Loaded image_encoder checkpoint")
    if loaded_states.get("D_img"):
        D_img.load_state_dict(loaded_states["D_img"], strict=False)  # type: ignore[arg-type]
        print("✓ Loaded D_img checkpoint")

        # Validate discriminator weights for NaN/Inf
        nan_found = False
        for name, param in D_img.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ⚠️  WARNING: NaN/Inf in D_img parameter: {name}")
                nan_found = True
        if nan_found:
            print("  ⚠️  Reinitializing discriminator due to NaN/Inf values")
            D_img = PatchDiscriminator(in_channels=args.channels, ctx_dim=args.vae_dim)

    diffuser.to(device)
    text_encoder.to(device)
    image_encoder.to(device)
    D_img.to(device)

    # Dataset
    collate_fn = partial(
        collate_fn_variable,
        channels=args.channels,
        img_size=args.img_size,
        reduced_min_sizes=args.reduced_min_sizes,
    )
    if args.use_webdataset:
        print(f"Using WebDataset streaming from: {args.webdataset_url}")
        dataset = StreamingWebDataset(
            tokenizer_name=args.tokenizer_name,
            token=args.webdataset_token,
            url_pattern=args.webdataset_url,
            channels=args.channels,
            image_key=args.webdataset_image_key,
            label_key=args.webdataset_label_key,
            caption_key=args.webdataset_caption_key,
            dataset_size=args.webdataset_size,
            samples_per_shard=args.webdataset_samples_per_shard,
            fixed_prompt_prefix=args.fixed_prompt_prefix,
        )
        print(f"  Shards: {dataset.num_shards}, Estimated size: {dataset.dataset_size:,} samples")
        if args.fixed_prompt_prefix:
            print(f"  Fixed prompt prefix: '{args.fixed_prompt_prefix}'")
        sampler = None  # IterableDataset doesn't use sampler
    else:
        dataset = TextImageDataset(
            data_path=args.data_path,
            captions_file=args.captions_file,
            tokenizer_name=args.tokenizer_name,
            transform=None,
            fixed_prompt_prefix=args.fixed_prompt_prefix,
        )
        if args.fixed_prompt_prefix:
            print(f"  Fixed prompt prefix: '{args.fixed_prompt_prefix}'")

        # Build dimension cache for efficient batching
        dimension_cache = get_or_build_dimension_cache(
            dataset, cache_dir=args.output_path, multiple=32, rebuild=False
        )

        # Load sampler state if resuming
        sampler_state = None
        if saved_global_step > 0:
            sampler_state_path = os.path.join(args.output_path, "sampler_state.pt")
            if os.path.exists(sampler_state_path):
                try:
                    sampler_state = torch.load(sampler_state_path, weights_only=False)
                    print(
                        f"Loaded sampler state: epoch {sampler_state.get('current_epoch', 0)}, position {sampler_state.get('position', 0)}"
                    )
                except Exception as e:
                    print(f"Warning: Could not load sampler state: {e}")

        # Create resumable sampler
        sampler = ResumableDimensionSampler(
            dimension_cache=dimension_cache,
            batch_size=args.batch_size,
            seed=42,  # Fixed seed for reproducibility
            resume_state=sampler_state,
        )

    # Build DataLoader kwargs conditionally based on whether we're using a batch_sampler
    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": args.workers,
        "pin_memory": (not torch.backends.mps.is_available()),
        "collate_fn": collate_fn,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": True,
        # prefetch_factor removed - with pin_memory=True, prefetching uses GPU-pinned memory
        # which caused OOM with large batches (user reported 47GB usage vs 48GB capacity)
    }

    if sampler is not None:
        # Using batch_sampler - cannot specify batch_size, shuffle, sampler, or drop_last
        dataloader_kwargs["batch_sampler"] = sampler
    else:
        # Not using batch_sampler - specify batch_size and shuffle
        dataloader_kwargs["batch_size"] = args.batch_size
        dataloader_kwargs["shuffle"] = False

    dataloader = DataLoader(**dataloader_kwargs)
    dataloader = accelerator.prepare(dataloader)

    # Get reference to the actual sampler after accelerator wrapping
    # Accelerate may wrap the dataloader, but we need to access the original sampler
    if sampler is not None:
        # For batch_sampler, it's typically accessible as dataloader.batch_sampler
        # But with accelerator, we need to check if it's wrapped
        if hasattr(dataloader, "batch_sampler"):
            actual_sampler = dataloader.batch_sampler
        else:
            # Fallback to original sampler if accelerator doesn't expose it
            actual_sampler = sampler
    else:
        actual_sampler = None

    # For WebDataset streaming, create robust iterator with error recovery
    use_robust_iterator = args.use_webdataset
    robust_iterator = None
    if use_robust_iterator:

        def dataloader_factory():
            new_loader = DataLoader(**dataloader_kwargs)
            return accelerator.prepare(new_loader)

        robust_iterator = RobustDataLoaderIterator(
            dataloader=dataloader,
            dataloader_factory=dataloader_factory,
            max_retries=args.dataloader_max_retries,
            retry_delay=args.dataloader_retry_delay,
            max_consecutive_errors=10,
        )
        print(
            f"Using robust DataLoader iterator (max_retries={args.dataloader_max_retries}, retry_delay={args.dataloader_retry_delay}s)"
        )

    # Load optimizer and scheduler configurations
    optimizer_configs, scheduler_configs = load_optimizer_scheduler_config(args, lr)

    # Calculate total training steps
    if isinstance(dataset, IterableDataset):
        dt_items = len(dataset)  # Uses dataset_size property
    else:
        dt_items = len(dataloader)
    steps_per_epoch = dt_items * args.training_steps
    total_steps = args.n_epochs * steps_per_epoch

    # Create optimizers using factory
    print("\n=== Creating Optimizers ===")
    optimizer_FDiff = create_optimizer(
        diffuser.flow_processor.parameters(), optimizer_configs["flow"]
    )
    print(f"Flow: {optimizer_configs['flow']['type']} (lr={optimizer_configs['flow']['lr']:.2e})")

    optimizer_Diff = create_optimizer(
        list(diffuser.compressor.parameters()) + list(diffuser.expander.parameters()),
        optimizer_configs["vae"],
    )
    print(f"VAE: {optimizer_configs['vae']['type']} (lr={optimizer_configs['vae']['lr']:.2e})")

    optimizer_te = create_optimizer(text_encoder.parameters(), optimizer_configs["text_encoder"])
    print(
        f"Text Encoder: {optimizer_configs['text_encoder']['type']} (lr={optimizer_configs['text_encoder']['lr']:.2e})"
    )

    optimizer_Dimg = create_optimizer(D_img.parameters(), optimizer_configs["discriminator"])
    print(
        f"Discriminator: {optimizer_configs['discriminator']['type']} (lr={optimizer_configs['discriminator']['lr']:.2e})"
    )

    # Create schedulers using factory
    print("\n=== Creating Schedulers ===")
    scheduler_FDiff = create_scheduler(optimizer_FDiff, scheduler_configs["flow"], total_steps)
    print(f"Flow: {scheduler_configs['flow']['type']}")

    scheduler_Diff = create_scheduler(optimizer_Diff, scheduler_configs["vae"], total_steps)
    print(f"VAE: {scheduler_configs['vae']['type']}")

    scheduler_te = create_scheduler(optimizer_te, scheduler_configs["text_encoder"], total_steps)
    print(f"Text Encoder: {scheduler_configs['text_encoder']['type']}")

    scheduler_Dimg = create_scheduler(
        optimizer_Dimg, scheduler_configs["discriminator"], total_steps
    )
    print(f"Discriminator: {scheduler_configs['discriminator']['type']}")
    print(f"Total steps: {total_steps}\n")

    # Prepare with accelerator
    objs = [
        diffuser,
        text_encoder,
        image_encoder,
        optimizer_Diff,
        optimizer_FDiff,
        optimizer_te,
        optimizer_Dimg,
        scheduler_Diff,
        scheduler_FDiff,
        scheduler_te,
        scheduler_Dimg,
        dataloader,
    ]
    prepared = accelerator.prepare(*objs)
    (
        diffuser,
        text_encoder,
        image_encoder,
        optimizer_Diff,
        optimizer_FDiff,
        optimizer_te,
        optimizer_Dimg,
        scheduler_Diff,
        scheduler_FDiff,
        scheduler_te,
        scheduler_Dimg,
        dataloader,
    ) = prepared

    # EMA for VAE
    ema = EMA(
        nn.ModuleList([diffuser.compressor, diffuser.expander]),
        decay=0.999,
        device=device,
    )

    # Load optimizer, scheduler, and EMA states if resuming
    if saved_global_step > 0:
        checkpoint_manager.load_optimizer_scheduler_ema_states(
            optimizers={
                "optimizer_FDiff": optimizer_FDiff,
                "optimizer_Diff": optimizer_Diff,
                "optimizer_te": optimizer_te,
                "optimizer_Dimg": optimizer_Dimg,
            },
            schedulers={
                "scheduler_FDiff": scheduler_FDiff,
                "scheduler_Diff": scheduler_Diff,
                "scheduler_te": scheduler_te,
                "scheduler_Dimg": scheduler_Dimg,
            },
            ema=ema,
        )

    n_critic = args.training_steps
    trn_steps = range(n_critic)
    total_batches = dt_items * args.n_epochs
    start_time = time.time()

    model_saved = False
    vae_errors = FloatBuffer(max(args.log_interval * 2, 10))
    kl_errors = FloatBuffer(max(args.log_interval * 2, 10))
    lpips_errors = FloatBuffer(max(args.log_interval * 2, 10))
    diff_errors = FloatBuffer(max(args.log_interval * 2, 10))
    adv_img_errors = FloatBuffer(max(args.log_interval * 2, 10))

    # Create trainers for modular training
    vae_trainer = None
    flow_trainer = None

    if args.train_vae:
        vae_trainer = VAETrainer(
            compressor=diffuser.compressor,
            expander=diffuser.expander,
            optimizer=optimizer_Diff,
            scheduler=scheduler_Diff,
            ema=ema,
            reconstruction_loss_fn=nn.L1Loss(),
            reconstruction_loss_min_fn=nn.MSELoss(),
            use_spade=args.train_spade,
            kl_beta=args.kl_beta,
            kl_warmup_steps=args.kl_warmup_steps,
            kl_free_bits=args.kl_free_bits,
            use_gan=args.gan_training,
            discriminator=D_img if args.gan_training else None,
            discriminator_optimizer=optimizer_Dimg if args.gan_training else None,
            discriminator_scheduler=scheduler_Dimg if args.gan_training else None,
            lambda_adv=args.lambda_adv,
            use_lpips=args.use_lpips,
            lambda_lpips=args.lambda_lpips,
            r1_gamma=5.0,
            r1_interval=16,
            gradient_clip_norm=args.initial_clipping_norm,
            accelerator=accelerator,
        )

    if args.train_diff or args.train_diff_full:
        flow_trainer = FlowTrainer(
            flow_processor=diffuser.flow_processor,
            text_encoder=text_encoder,
            compressor=diffuser.compressor,
            optimizer=optimizer_FDiff,
            scheduler=scheduler_FDiff,
            text_encoder_optimizer=optimizer_te,
            text_encoder_scheduler=scheduler_te,
            gradient_clip_norm=args.initial_clipping_norm,
            num_train_timesteps=1000,
            accelerator=accelerator,
        )

    # Parse sample sizes for generation
    parsed_sample_sizes = parse_sample_sizes(args.sample_sizes)

    # Initial sampling
    if not args.no_samples:
        for img_addr in args.test_image_address:
            safe_vae_sample(diffuser, img_addr, channels, args.output_path, 0, device)
        if args.train_diff or args.train_diff_full:
            save_sample_images(
                diffuser,
                text_encoder,
                tokenizer,
                args.output_path,
                0,
                device,
                args.sample_captions,
                args.batch_size,
                sample_sizes=parsed_sample_sizes,
                use_cfg=True,
                guidance_scale=5.0,
            )

    global_step = saved_global_step
    last_sample_step = saved_last_sample_step
    sample_interval = args.checkpoint_save_interval * args.samples_per_checkpoint

    print(f"\nStarting training for {args.n_epochs} epochs...")
    print(f"Total batches: {total_batches}, Steps per epoch: {steps_per_epoch}")
    print(f"Training modes: VAE={args.train_vae}, FLOW={args.train_diff or args.train_diff_full}")
    if saved_global_step > 0:
        print(f"Resuming from global step: {global_step}")
        if args.train_vae:
            # Show current KL warmup state when resuming
            current_beta = cosine_anneal_beta(global_step, args.kl_warmup_steps, args.kl_beta)
            warmup_progress = min(100.0 * global_step / args.kl_warmup_steps, 100.0)
            print(f"  KL warmup: {warmup_progress:.1f}% complete (β={current_beta:.6f})")
            print(f"  KL will reach max (β={args.kl_beta}) at step {args.kl_warmup_steps}")

    # Training loop - resume from saved epoch and batch
    for epoch in range(saved_epoch, args.n_epochs):
        # Update sampler for new epoch (if using ResumableDimensionSampler)
        if actual_sampler is not None and epoch > saved_epoch:
            actual_sampler.set_epoch(epoch)

        # Get initial batch index from sampler (for proper resume)
        batch_start_idx = 0
        if actual_sampler is not None and epoch == saved_epoch:
            # When resuming the saved epoch, start from the sampler's current position
            batch_start_idx = actual_sampler.position
            print(f"  Resuming epoch {epoch} from batch {batch_start_idx}")

        # Use robust iterator for streaming WebDataset, regular dataloader for local datasets
        data_source = robust_iterator if use_robust_iterator else dataloader

        for enum_idx, (imgs, input_ids) in enumerate(data_source):
            # Calculate actual batch index (enumerate always starts from 0)
            i = batch_start_idx + enum_idx

            # For IterableDataset, break after dt_items batches to advance epoch
            if i >= dt_items:
                break

            # Sampler handles resume automatically - no need to skip batches
            model_saved = False
            batch_start_time = time.time()
            input_ids = input_ids.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            avg_diff_loss = 0.0
            avg_vae_loss = 0.0
            avg_rev_vae_loss = 0.0
            avg_D_img_loss = 0.0
            avg_G_img_loss = 0.0

            try:
                resolutions = len(imgs)

                # Training steps
                for _ in trn_steps:
                    # Train on all resolutions
                    for ri in imgs:
                        real_imgs = ri.to(device).detach()

                        # VAE training using trainer
                        if vae_trainer:
                            vae_losses = vae_trainer.train_step(real_imgs, global_step)
                            avg_vae_loss += vae_losses["vae"] / resolutions
                            vae_errors.add_item(vae_losses["vae"])
                            kl_errors.add_item(vae_losses["kl"])
                            if "lpips" in vae_losses and vae_losses["lpips"] > 0:
                                lpips_errors.add_item(vae_losses["lpips"])
                            if "discriminator" in vae_losses:
                                avg_D_img_loss += vae_losses["discriminator"] / resolutions
                                adv_img_errors.add_item(vae_losses["discriminator"])
                            if "generator" in vae_losses:
                                avg_G_img_loss += vae_losses["generator"] / resolutions

                        # Flow/Diffusion training using trainer
                        if flow_trainer:
                            diff_metrics = flow_trainer.train_step(
                                real_imgs, input_ids, attention_mask
                            )
                            # Handle dict return type from train_step
                            diff_loss = (
                                diff_metrics["flow_loss"]
                                if isinstance(diff_metrics, dict)
                                else diff_metrics
                            )
                            avg_diff_loss += diff_loss / resolutions
                            diff_errors.add_item(diff_loss)

                # Increment global step once per batch (after all training steps)
                global_step += 1

                # Logging
                if i % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    elapsed_str = format_duration(elapsed)
                    batch_time = time.time() - batch_start_time

                    # Calculate ETA
                    batches_done = epoch * dt_items + i
                    batches_total = args.n_epochs * dt_items
                    batches_remaining = batches_total - batches_done

                    if batches_done > 0 and batch_time > 0:
                        # Use recent batch time as estimate
                        eta_seconds = batches_remaining * batch_time
                        eta_str = format_duration(eta_seconds)
                    else:
                        eta_str = "calculating..."

                    # Add memory monitoring
                    mem_str = ""
                    if torch.cuda.is_available():
                        mem_allocated_gb = torch.cuda.memory_allocated() / 1e9
                        mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
                        mem_str = f" | GPU: {mem_allocated_gb:.1f}GB"

                        # Warn if approaching memory limit
                        if i % (args.log_interval * 10) == 0:
                            max_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                            if mem_allocated_gb > max_memory_gb * 0.85:
                                print(
                                    f"⚠️  High memory usage: {mem_allocated_gb:.1f}/{max_memory_gb:.1f}GB (85%+ used)"
                                )

                    log_msg = f"[{elapsed_str}] Epoch {epoch}/{args.n_epochs} | Batch {i}/{dt_items} | {batch_time:.2f}s/batch{mem_str} | ETA: {eta_str}"

                    # Store current beta for logging
                    current_beta = 0.0
                    if args.train_vae:
                        # Show current KL beta for warmup monitoring
                        current_beta = cosine_anneal_beta(
                            global_step, args.kl_warmup_steps, args.kl_beta
                        )
                        log_msg += f" | VAE: {vae_errors.average:.4f} | KL: {kl_errors.average:.4f} (β={current_beta:.6f})"
                        if lpips_errors.average > 0:
                            log_msg += f" | LPIPS: {lpips_errors.average:.4f}"
                        if args.gan_training:
                            log_msg += (
                                f" | D: {adv_img_errors.average:.4f} | G: {avg_G_img_loss:.4f}"
                            )
                    if args.train_diff or args.train_diff_full:
                        log_msg += f" | Flow: {avg_diff_loss:.4f}"

                    # Show appropriate LR based on what's being trained
                    if args.train_diff or args.train_diff_full:
                        log_msg += f" | LR_F: {current_lr(optimizer_FDiff):.2e}"
                    elif args.train_vae:
                        log_msg += f" | LR_V: {current_lr(optimizer_Diff):.2e}"

                    print(log_msg)

                    # Log metrics to progress logger (with all available metrics)
                    metrics = {}
                    extras = {
                        "batch_time": batch_time,
                        "eta_seconds": eta_seconds if batches_done > 0 and batch_time > 0 else 0,
                    }

                    if args.train_vae:
                        metrics["vae_loss"] = vae_errors.average
                        metrics["kl_loss"] = kl_errors.average
                        metrics["kl_beta"] = current_beta  # Log beta value for graphs
                        if lpips_errors.average > 0:
                            metrics["lpips_loss"] = lpips_errors.average
                        if args.gan_training:
                            metrics["discriminator_loss"] = adv_img_errors.average
                            metrics["generator_loss"] = avg_G_img_loss
                    if args.train_diff or args.train_diff_full:
                        metrics["flow_loss"] = avg_diff_loss

                    progress_logger.log_metrics(
                        epoch=epoch,
                        batch=i,
                        global_step=global_step,
                        metrics=metrics,
                        learning_rates={
                            "flow_lr": current_lr(optimizer_FDiff),
                            "vae_lr": current_lr(optimizer_Diff),
                        },
                        extras=extras,
                    )

                # Checkpoint saving
                if i % args.checkpoint_save_interval == 0 and i > 0:
                    # Save models using checkpoint manager
                    checkpoint_manager.save_models(
                        diffuser=diffuser,
                        text_encoder=text_encoder,
                        discriminators=(
                            {"D_img": D_img} if args.train_vae and args.gan_training else None
                        ),
                    )

                    # Save learning rates
                    with open(LR_SAVE_FILE, "w") as f:
                        json.dump(
                            {
                                "lr": current_lr(optimizer_FDiff),
                                "vae": current_lr(optimizer_Diff),
                            },
                            f,
                        )

                    # Save training state using checkpoint manager
                    checkpoint_manager.save_training_state(
                        epoch=epoch,
                        batch_idx=i,
                        global_step=global_step,
                        optimizers={
                            "optimizer_FDiff": optimizer_FDiff,
                            "optimizer_Diff": optimizer_Diff,
                            "optimizer_te": optimizer_te,
                            "optimizer_Dimg": optimizer_Dimg,
                        },
                        schedulers={
                            "scheduler_FDiff": scheduler_FDiff,
                            "scheduler_Diff": scheduler_Diff,
                            "scheduler_te": scheduler_te,
                            "scheduler_Dimg": scheduler_Dimg,
                        },
                        ema=ema,
                        sampler=actual_sampler,
                        kl_beta_current=cosine_anneal_beta(
                            global_step, args.kl_warmup_steps, args.kl_beta
                        ),
                        kl_warmup_steps=args.kl_warmup_steps,
                        kl_max_beta=args.kl_beta,
                        last_sample_step=last_sample_step,
                    )

                    # Clear CUDA cache to prevent memory fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Generate samples based on step interval (decoupled from checkpoint count)
                    if (
                        not args.no_samples
                        and global_step > 0
                        and (global_step - last_sample_step) >= sample_interval
                    ):
                        for img_addr in args.test_image_address:
                            safe_vae_sample(
                                diffuser,
                                img_addr,
                                channels,
                                args.output_path,
                                epoch,
                                device,
                            )
                        if args.train_diff or args.train_diff_full:
                            save_sample_images(
                                diffuser,
                                text_encoder,
                                tokenizer,
                                args.output_path,
                                epoch,
                                device,
                                args.sample_captions,
                                args.batch_size,
                                sample_sizes=parsed_sample_sizes,
                                use_cfg=True,
                                guidance_scale=5.0,
                            )
                        last_sample_step = global_step

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Save at end of epoch
        print(f"\nEpoch {epoch} complete. Saving models...")
        checkpoint_manager.save_models(
            diffuser=diffuser,
            text_encoder=text_encoder,
            discriminators={"D_img": D_img} if args.train_vae and args.gan_training else None,
        )

        # Save learning rates
        with open(LR_SAVE_FILE, "w") as f:
            json.dump(
                {"lr": current_lr(optimizer_FDiff), "vae": current_lr(optimizer_Diff)},
                f,
            )

        # Save training state using checkpoint manager
        checkpoint_manager.save_training_state(
            epoch=epoch,
            batch_idx=i,
            global_step=global_step,
            optimizers={
                "optimizer_FDiff": optimizer_FDiff,
                "optimizer_Diff": optimizer_Diff,
                "optimizer_te": optimizer_te,
                "optimizer_Dimg": optimizer_Dimg,
            },
            schedulers={
                "scheduler_FDiff": scheduler_FDiff,
                "scheduler_Diff": scheduler_Diff,
                "scheduler_te": scheduler_te,
                "scheduler_Dimg": scheduler_Dimg,
            },
            ema=ema,
            sampler=actual_sampler,
            kl_beta_current=cosine_anneal_beta(global_step, args.kl_warmup_steps, args.kl_beta),
            kl_warmup_steps=args.kl_warmup_steps,
            kl_max_beta=args.kl_beta,
            last_sample_step=last_sample_step,
        )

        model_saved = True

    # Print robust iterator stats if used
    if robust_iterator is not None:
        stats = robust_iterator.get_stats()
        if stats["total_errors"] > 0:
            print(
                f"\nDataLoader recovery stats: {stats['total_errors']} errors recovered, {stats['batches_yielded']} batches yielded"
            )

    print("\nTraining complete!")


def detect_config_mode(config):
    """
    Detect whether config uses pipeline mode or legacy mode.

    Args:
        config: Loaded YAML config dictionary

    Returns:
        str: "pipeline" if config has training.pipeline, "legacy" otherwise
    """
    if config and "training" in config and "pipeline" in config["training"]:
        return "pipeline"
    return "legacy"


def validate_and_show_plan(config, args):
    """
    Validate pipeline configuration and show execution plan (dry-run).

    Args:
        config: Loaded YAML config dictionary
        args: Parsed command-line arguments

    Raises:
        ValueError: If pipeline configuration is invalid
    """
    print("\n" + "=" * 80)
    print("PIPELINE VALIDATION - DRY RUN MODE")
    print("=" * 80)

    # Parse and validate pipeline config
    pipeline_config = parse_pipeline_config(config["training"]["pipeline"])

    print(f"\n✓ Pipeline configuration is valid")
    print(f"  Total steps: {len(pipeline_config.steps)}")

    print("\n" + "-" * 80)
    print("EXECUTION PLAN")
    print("-" * 80)

    for idx, step in enumerate(pipeline_config.steps, 1):
        print(f"\nStep {idx}: {step.name}")
        print(f"  Duration: {step.n_epochs} epochs")
        if step.max_steps is not None:
            print(f"  Max steps per epoch: {step.max_steps} (for quick testing)")

        # Training modes
        modes = []
        if step.train_vae:
            modes.append(f"VAE (SPADE={'ON' if step.train_spade else 'OFF'})")
        if step.train_diff or step.train_diff_full:
            modes.append("Flow")
        print(f"  Training: {', '.join(modes) if modes else 'None'}")

        # Frozen/unfrozen modules
        frozen = step.freeze
        unfrozen = step.unfreeze

        if frozen:
            print(f"  Frozen: {', '.join(frozen)}")
        if unfrozen:
            print(f"  Active: {', '.join(unfrozen)}")

        # Optimizers
        if step.optimization and step.optimization.optimizers:
            print(f"  Optimizers:")
            for name, opt_cfg in step.optimization.optimizers.items():
                print(f"    - {name}: {opt_cfg.type} (lr={opt_cfg.lr})")

        # Schedulers
        if step.optimization and step.optimization.schedulers:
            print(f"  Schedulers:")
            for name, sched_cfg in step.optimization.schedulers.items():
                print(f"    - {name}: {sched_cfg.type}")

        # Transition criteria
        if step.transition_on and step.transition_on.mode != "epoch":
            tc = step.transition_on
            criteria = []
            if tc.metric:
                criteria.append(f"metric={tc.metric}")
            if tc.threshold is not None:
                criteria.append(f"threshold<{tc.threshold}")
            if tc.max_epochs is not None:
                criteria.append(f"max_epochs={tc.max_epochs}")
            if criteria:
                print(f"  Transition: {', '.join(criteria)}")

    print("\n" + "=" * 80)
    print("Validation complete. No training will be performed.")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FluxFlow text-to-image model")

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file (overrides CLI args)")
    parser.add_argument(
        "--validate-pipeline",
        action="store_true",
        help="Validate pipeline configuration and show execution plan (dry-run mode)",
    )

    # Data
    parser.add_argument("--data_path", type=str, help="Path to training images")
    parser.add_argument("--captions_file", type=str, help="Tab-separated captions file")
    parser.add_argument(
        "--fixed_prompt_prefix",
        type=str,
        default=None,
        help="Optional fixed text to prepend to all prompts (e.g., 'style anime')",
    )
    # New generic WebDataset arguments
    parser.add_argument("--use_webdataset", action="store_true", help="Use WebDataset streaming")
    parser.add_argument("--webdataset_token", type=str, default=None, help="HuggingFace token")
    parser.add_argument(
        "--webdataset_url",
        type=str,
        default="hf://datasets/jackyhate/text-to-image-2M/data_512_2M/*.tar",
        help="WebDataset URL pattern",
    )
    parser.add_argument("--webdataset_image_key", type=str, default="png", help="Image key in tar")
    parser.add_argument("--webdataset_label_key", type=str, default="json", help="Label key in tar")
    parser.add_argument(
        "--webdataset_caption_key", type=str, default="prompt", help="Caption key within JSON"
    )
    parser.add_argument(
        "--webdataset_size",
        type=int,
        default=None,
        help="Total dataset size (if known). If not set, estimates from shard count.",
    )
    parser.add_argument(
        "--webdataset_samples_per_shard",
        type=int,
        default=10000,
        help="Estimated samples per shard for size estimation",
    )
    # Legacy aliases for backward compatibility
    parser.add_argument("--use_tt2m", action="store_true", help="(Deprecated) Use --use_webdataset")
    parser.add_argument(
        "--tt2m_token", type=str, default=None, help="(Deprecated) Use --webdataset_token"
    )

    # Model
    parser.add_argument("--model_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--vae_dim", type=int, default=128, help="VAE latent dimension")
    parser.add_argument(
        "--text_embedding_dim", type=int, default=1024, help="Text embedding dimension"
    )
    parser.add_argument("--feature_maps_dim", type=int, default=128, help="Flow feature dimension")
    parser.add_argument(
        "--feature_maps_dim_disc",
        type=int,
        default=8,
        help="Discriminator feature dimension",
    )
    parser.add_argument(
        "--pretrained_bert_model",
        type=str,
        default=None,
        help="Pretrained BERT checkpoint",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing in VAE (saves VRAM during forward, costs VRAM during backward)",
    )

    # Training
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate for flow")
    parser.add_argument("--lr_min", type=float, default=1e-1, help="Minimum LR multiplier")
    parser.add_argument("--preserve_lr", action="store_true", help="Load saved learning rates")
    parser.add_argument(
        "--optim_sched_config",
        type=str,
        default=None,
        help="Path to JSON file with optimizer/scheduler configurations",
    )
    parser.add_argument(
        "--training_steps", type=int, default=1, help="Inner training steps per batch"
    )
    parser.add_argument("--use_fp16", action="store_true", help="Use mixed precision")
    parser.add_argument(
        "--initial_clipping_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )

    # Training modes
    parser.add_argument("--train_vae", action="store_true", help="Train VAE (compressor+expander)")
    parser.add_argument("--gan_training", action="store_true", help="Enable GAN training for VAE")
    parser.add_argument(
        "--use_lpips", action="store_true", help="Enable LPIPS perceptual loss for VAE"
    )
    parser.add_argument("--train_spade", action="store_true", help="Use SPADE conditioning")
    parser.add_argument("--train_diff", action="store_true", help="Train flow model")
    parser.add_argument(
        "--train_diff_full", action="store_true", help="Train flow with full schedule"
    )

    # KL divergence
    parser.add_argument(
        "--kl_beta", type=float, default=0.0001, help="Final KL weight (reduced from 1.0)"
    )
    parser.add_argument(
        "--kl_warmup_steps", type=int, default=5000, help="KL warmup steps (reduced from 100000)"
    )
    parser.add_argument("--kl_free_bits", type=float, default=0.0, help="Free bits (nats)")

    # Output
    parser.add_argument("--output_path", type=str, default="outputs", help="Output directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging frequency")
    parser.add_argument(
        "--checkpoint_save_interval",
        type=int,
        default=50,
        help="Checkpoint saving frequency (batches)",
    )
    parser.add_argument(
        "--samples_per_checkpoint",
        type=int,
        default=1,
        help="Generate samples every N checkpoint saves",
    )
    parser.add_argument(
        "--no_samples", action="store_true", help="Disable sampling during training"
    )
    parser.add_argument(
        "--generate_diagrams",
        action="store_true",
        help="Automatically generate training diagrams on each checkpoint save",
    )
    parser.add_argument(
        "--test_image_address",
        nargs="+",
        default=[],
        help="Test images for VAE sampling",
    )
    parser.add_argument(
        "--sample_captions",
        nargs="+",
        default=["A sample caption"],
        help="Captions for sampling",
    )

    # Misc
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="distilbert-base-uncased",
        help="Tokenizer name",
    )
    parser.add_argument("--img_size", type=int, default=1024, help="Image size")
    parser.add_argument("--channels", type=int, default=3, help="Image channels")
    parser.add_argument(
        "--sample_sizes",
        nargs="+",
        default=None,
        help="Sample image sizes. Can be integers (square) or WxH pairs (e.g., 512 768x512)",
    )
    parser.add_argument(
        "--reduced_min_sizes",
        nargs="+",
        type=int,
        default=None,
        help="Min sizes for reduced training images (e.g., 128 256 512)",
    )
    parser.add_argument(
        "--dataloader_max_retries",
        type=int,
        default=5,
        help="Max retries for dataloader errors (WebDataset)",
    )
    parser.add_argument(
        "--dataloader_retry_delay",
        type=float,
        default=5.0,
        help="Base delay between dataloader retries in seconds",
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=0.5, help="GAN loss weight (increased from 0.1)"
    )
    parser.add_argument(
        "--lambda_lpips",
        type=float,
        default=0.1,
        help="LPIPS perceptual loss weight (lower=sharper, higher=smoother)",
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Map config to args (config overrides defaults, but CLI args override config)
        cli_provided = set()
        for action in parser._actions:
            if action.dest in sys.argv or f"--{action.dest}" in sys.argv:
                cli_provided.add(action.dest)

        # Apply config values if not provided via CLI
        if "model" in config:
            if "vae_dim" in config["model"] and "vae_dim" not in cli_provided:
                args.vae_dim = config["model"]["vae_dim"]
            if "feature_maps_dim" in config["model"] and "feature_maps_dim" not in cli_provided:
                args.feature_maps_dim = config["model"]["feature_maps_dim"]
            if (
                "feature_maps_dim_disc" in config["model"]
                and "feature_maps_dim_disc" not in cli_provided
            ):
                args.feature_maps_dim_disc = config["model"]["feature_maps_dim_disc"]
            if "text_embedding_dim" in config["model"] and "text_embedding_dim" not in cli_provided:
                args.text_embedding_dim = config["model"]["text_embedding_dim"]
            if (
                "use_gradient_checkpointing" in config["model"]
                and "use_gradient_checkpointing" not in cli_provided
            ):
                args.use_gradient_checkpointing = config["model"]["use_gradient_checkpointing"]
            if (
                "pretrained_bert_model" in config["model"]
                and "pretrained_bert_model" not in cli_provided
            ):
                args.pretrained_bert_model = config["model"]["pretrained_bert_model"]

        if "data" in config:
            if "data_path" in config["data"] and "data_path" not in cli_provided:
                args.data_path = config["data"]["data_path"]
            if "captions_file" in config["data"] and "captions_file" not in cli_provided:
                args.captions_file = config["data"]["captions_file"]
            if (
                "fixed_prompt_prefix" in config["data"]
                and "fixed_prompt_prefix" not in cli_provided
            ):
                args.fixed_prompt_prefix = config["data"]["fixed_prompt_prefix"]
            # New WebDataset config options
            if "use_webdataset" in config["data"] and "use_webdataset" not in cli_provided:
                args.use_webdataset = config["data"]["use_webdataset"]
            if "webdataset_token" in config["data"] and "webdataset_token" not in cli_provided:
                args.webdataset_token = config["data"]["webdataset_token"]
            if "webdataset_url" in config["data"] and "webdataset_url" not in cli_provided:
                args.webdataset_url = config["data"]["webdataset_url"]
            if (
                "webdataset_image_key" in config["data"]
                and "webdataset_image_key" not in cli_provided
            ):
                args.webdataset_image_key = config["data"]["webdataset_image_key"]
            if (
                "webdataset_label_key" in config["data"]
                and "webdataset_label_key" not in cli_provided
            ):
                args.webdataset_label_key = config["data"]["webdataset_label_key"]
            if (
                "webdataset_caption_key" in config["data"]
                and "webdataset_caption_key" not in cli_provided
            ):
                args.webdataset_caption_key = config["data"]["webdataset_caption_key"]
            if "webdataset_size" in config["data"] and "webdataset_size" not in cli_provided:
                args.webdataset_size = config["data"]["webdataset_size"]
            if (
                "webdataset_samples_per_shard" in config["data"]
                and "webdataset_samples_per_shard" not in cli_provided
            ):
                args.webdataset_samples_per_shard = config["data"]["webdataset_samples_per_shard"]
            # Legacy aliases (deprecated)
            if "use_tt2m" in config["data"] and "use_tt2m" not in cli_provided:
                args.use_tt2m = config["data"]["use_tt2m"]
            if "tt2m_token" in config["data"] and "tt2m_token" not in cli_provided:
                args.tt2m_token = config["data"]["tt2m_token"]
            if "img_size" in config["data"] and "img_size" not in cli_provided:
                args.img_size = config["data"]["img_size"]
            if "channels" in config["data"] and "channels" not in cli_provided:
                args.channels = config["data"]["channels"]
            if "tokenizer_name" in config["data"] and "tokenizer_name" not in cli_provided:
                args.tokenizer_name = config["data"]["tokenizer_name"]
            if (
                "dataloader_max_retries" in config["data"]
                and "dataloader_max_retries" not in cli_provided
            ):
                args.dataloader_max_retries = config["data"]["dataloader_max_retries"]
            if (
                "dataloader_retry_delay" in config["data"]
                and "dataloader_retry_delay" not in cli_provided
            ):
                args.dataloader_retry_delay = config["data"]["dataloader_retry_delay"]

        if "training" in config:
            if "n_epochs" in config["training"] and "n_epochs" not in cli_provided:
                args.n_epochs = config["training"]["n_epochs"]
            if "batch_size" in config["training"] and "batch_size" not in cli_provided:
                args.batch_size = config["training"]["batch_size"]
            if "workers" in config["training"] and "workers" not in cli_provided:
                args.workers = config["training"]["workers"]
            if "lr" in config["training"] and "lr" not in cli_provided:
                args.lr = config["training"]["lr"]
            if "lr_min" in config["training"] and "lr_min" not in cli_provided:
                args.lr_min = config["training"]["lr_min"]
            if "preserve_lr" in config["training"] and "preserve_lr" not in cli_provided:
                args.preserve_lr = config["training"]["preserve_lr"]
            if "training_steps" in config["training"] and "training_steps" not in cli_provided:
                args.training_steps = config["training"]["training_steps"]
            if "use_fp16" in config["training"] and "use_fp16" not in cli_provided:
                args.use_fp16 = config["training"]["use_fp16"]
            if (
                "initial_clipping_norm" in config["training"]
                and "initial_clipping_norm" not in cli_provided
            ):
                args.initial_clipping_norm = config["training"]["initial_clipping_norm"]
            if "train_vae" in config["training"] and "train_vae" not in cli_provided:
                args.train_vae = config["training"]["train_vae"]
            if "gan_training" in config["training"] and "gan_training" not in cli_provided:
                args.gan_training = config["training"]["gan_training"]
            if "use_lpips" in config["training"] and "use_lpips" not in cli_provided:
                args.use_lpips = config["training"]["use_lpips"]
            if "train_spade" in config["training"] and "train_spade" not in cli_provided:
                args.train_spade = config["training"]["train_spade"]
            if "train_diff" in config["training"] and "train_diff" not in cli_provided:
                args.train_diff = config["training"]["train_diff"]
            if "train_diff_full" in config["training"] and "train_diff_full" not in cli_provided:
                args.train_diff_full = config["training"]["train_diff_full"]
            if "kl_beta" in config["training"] and "kl_beta" not in cli_provided:
                args.kl_beta = config["training"]["kl_beta"]
            if "kl_warmup_steps" in config["training"] and "kl_warmup_steps" not in cli_provided:
                args.kl_warmup_steps = config["training"]["kl_warmup_steps"]
            if "kl_free_bits" in config["training"] and "kl_free_bits" not in cli_provided:
                args.kl_free_bits = config["training"]["kl_free_bits"]
            if "lambda_adv" in config["training"] and "lambda_adv" not in cli_provided:
                args.lambda_adv = config["training"]["lambda_adv"]
            if "lambda_lpips" in config["training"] and "lambda_lpips" not in cli_provided:
                args.lambda_lpips = config["training"]["lambda_lpips"]

        if "optimization" in config:
            if (
                "optim_sched_config" in config["optimization"]
                and "optim_sched_config" not in cli_provided
            ):
                args.optim_sched_config = config["optimization"]["optim_sched_config"]

        if "output" in config:
            if "output_path" in config["output"] and "output_path" not in cli_provided:
                args.output_path = config["output"]["output_path"]
            if "log_interval" in config["output"] and "log_interval" not in cli_provided:
                args.log_interval = config["output"]["log_interval"]
            if (
                "checkpoint_save_interval" in config["output"]
                and "checkpoint_save_interval" not in cli_provided
            ):
                args.checkpoint_save_interval = config["output"]["checkpoint_save_interval"]
            if (
                "samples_per_checkpoint" in config["output"]
                and "samples_per_checkpoint" not in cli_provided
            ):
                args.samples_per_checkpoint = config["output"]["samples_per_checkpoint"]
            if "no_samples" in config["output"] and "no_samples" not in cli_provided:
                args.no_samples = config["output"]["no_samples"]
            if (
                "test_image_address" in config["output"]
                and "test_image_address" not in cli_provided
            ):
                args.test_image_address = config["output"]["test_image_address"]
            if "sample_captions" in config["output"] and "sample_captions" not in cli_provided:
                args.sample_captions = config["output"]["sample_captions"]
            if "sample_sizes" in config["output"] and "sample_sizes" not in cli_provided:
                args.sample_sizes = config["output"]["sample_sizes"]

        if "data" in config:
            if "reduced_min_sizes" in config["data"] and "reduced_min_sizes" not in cli_provided:
                args.reduced_min_sizes = config["data"]["reduced_min_sizes"]

        if "model_checkpoint" in config and "model_checkpoint" not in cli_provided:
            args.model_checkpoint = config["model_checkpoint"]

    return args


def main():
    """Main entry point for the training script."""
    args = parse_args()

    # Load config if provided
    config = None
    if args.config:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Handle --validate-pipeline flag
    if args.validate_pipeline:
        if not config:
            print("Error: --validate-pipeline requires --config <path/to/config.yaml>")
            sys.exit(1)
        if detect_config_mode(config) != "pipeline":
            print("Error: Config file does not contain training.pipeline section")
            sys.exit(1)

        try:
            validate_and_show_plan(config, args)
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ Pipeline validation failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Handle backward compatibility for legacy tt2m arguments
    if args.use_tt2m and not args.use_webdataset:
        args.use_webdataset = True
    if args.tt2m_token and not args.webdataset_token:
        args.webdataset_token = args.tt2m_token

    # Detect if we're using pipeline mode (skip validation if so)
    is_pipeline_mode = config and detect_config_mode(config) == "pipeline"

    # Validate arguments (only if NOT using pipeline mode with datasets defined)
    if not is_pipeline_mode:
        has_local_data = args.data_path and args.captions_file
        has_webdataset = args.use_webdataset and args.webdataset_token

        if not has_local_data and not has_webdataset:
            print(
                "Error: Please provide --data_path and --captions_file, "
                "or use --use_webdataset with --webdataset_token"
            )
            sys.exit(1)

    # Detect training mode
    if config and detect_config_mode(config) == "pipeline":
        # Pipeline mode
        train_pipeline(args, config)
    else:
        # Legacy mode
        if not (args.train_vae or args.train_diff or args.train_diff_full):
            print(
                "Warning: No training mode enabled. Use --train_vae, --train_diff, or --train_diff_full"
            )
        train_legacy(args)


if __name__ == "__main__":
    main()
