# Pipeline Training Architecture

## Overview

Multi-step training pipelines allow sequential training phases with different configurations, enabling hypothesis testing (e.g., SPADE OFF → SPADE ON) and complex training strategies.

## Current Status

- ✅ **Phase 1**: Pipeline config parsing and validation
- ✅ **Phase 2**: train.py integration and dry-run validation
- ⏸️ **Phase 3**: Full pipeline execution (architecture documented, implementation deferred)

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                        train.py                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ main()                                                 │  │
│  │  ├─ detect_config_mode()                              │  │
│  │  ├─ validate_and_show_plan() [if --validate-pipeline]│  │
│  │  └─ train_pipeline() or train_legacy()               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ├─ Legacy Mode
                               │  └─ train_legacy() - existing training loop
                               │
                               └─ Pipeline Mode
                                  └─ train_pipeline()
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────┐
│              TrainingPipelineOrchestrator                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ __init__(pipeline_config, checkpoint_manager,           │  │
│  │          accelerator, device)                           │  │
│  │                                                          │  │
│  │ run(models, dataloader, tokenizer, args, config)        │  │
│  │  ├─ resume_from_checkpoint() [if checkpoint exists]    │  │
│  │  ├─ for each step in pipeline:                         │  │
│  │  │    ├─ configure_step_models() [freeze/unfreeze]     │  │
│  │  │    ├─ create_step_optimizers()                      │  │
│  │  │    ├─ create_step_schedulers()                      │  │
│  │  │    ├─ create_step_trainers() [VAE/Flow]            │  │
│  │  │    ├─ for epoch in range(step.n_epochs):           │  │
│  │  │    │    ├─ train_epoch()                           │  │
│  │  │    │    ├─ update_metrics()                        │  │
│  │  │    │    ├─ should_transition()                     │  │
│  │  │    │    └─ save_checkpoint() [with pipeline meta]  │  │
│  │  │    └─ cleanup optimizers/schedulers                │  │
│  │  └─ print_pipeline_summary()                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Config (YAML)
    │
    ├─ parse_pipeline_config() → PipelineConfig
    │                              │
    │                              ├─ steps: [PipelineStepConfig, ...]
    │                              └─ defaults: PipelineStepConfig
    │
    ▼
train_pipeline(args, config)
    │
    ├─ Initialize models (diffuser, text_encoder, discriminator)
    ├─ Initialize dataset/dataloader
    ├─ Initialize checkpoint_manager, accelerator
    │
    ▼
TrainingPipelineOrchestrator.run(models, dataloader, ...)
    │
    ├─ Resume from checkpoint [optional]
    │   └─ Restore: step_index, epoch, batch_idx
    │
    └─ For each step in pipeline:
        │
        ├─ Configure step
        │   ├─ Freeze/unfreeze models per step config
        │   ├─ Create optimizers (inline YAML config)
        │   ├─ Create schedulers (inline YAML config)
        │   └─ Create trainers (VAETrainer, FlowTrainer)
        │
        ├─ Train step
        │   └─ For epoch in range(step.n_epochs):
        │       ├─ For batch in dataloader:
        │       │   ├─ vae_trainer.train_step() [if step.train_vae]
        │       │   ├─ flow_trainer.train_step() [if step.train_flow]
        │       │   ├─ update_metrics({"vae_loss": ..., "flow_loss": ...})
        │       │   └─ log progress
        │       │
        │       ├─ Check transition_criteria
        │       │   ├─ Mode: "epoch" → transition after n_epochs
        │       │   └─ Mode: "loss_threshold" → transition if loss < threshold
        │       │
        │       └─ Save checkpoint
        │           └─ Include pipeline metadata:
        │               ├─ current_step_index
        │               ├─ current_step_name
        │               ├─ step_epoch
        │               ├─ total_pipeline_progress
        │               └─ pipeline_config (for resume)
        │
        └─ Cleanup
            └─ Delete optimizers/schedulers (free memory)
```

## Implementation Plan

### Phase 3b.1: Extract Helper Functions (HIGH PRIORITY)

Create reusable initialization helpers in `train.py`:

```python
def initialize_models(args, device):
    """Initialize FluxFlow models from config."""
    text_encoder = BertTextEncoder(...)
    compressor = FluxCompressor(...)
    expander = FluxExpander(...)
    flow_processor = FluxFlowProcessor(...)
    diffuser = FluxPipeline(compressor, flow_processor, expander)
    D_img = PatchDiscriminator(...)
    
    # Move to device
    diffuser.to(device)
    text_encoder.to(device)
    D_img.to(device)
    
    return {
        "diffuser": diffuser,
        "compressor": compressor,
        "expander": expander,
        "flow_processor": flow_processor,
        "text_encoder": text_encoder,
        "D_img": D_img,
    }

def initialize_dataloader(args, tokenizer, device):
    """Initialize dataset and dataloader from config."""
    if args.use_webdataset:
        dataset = StreamingWebDataset(...)
        sampler = None
    else:
        dataset = TextImageDataset(...)
        dimension_cache = get_or_build_dimension_cache(...)
        sampler = ResumableDimensionSampler(...)
    
    dataloader = DataLoader(dataset, ...)
    return dataloader, sampler
```

### Phase 3b.2: Implement Orchestrator.run() (CORE LOGIC)

```python
def run(self, models, dataloader, tokenizer, args, config):
    """Execute complete training pipeline.
    
    Args:
        models: Dict of initialized models
        dataloader: Initialized DataLoader
        tokenizer: Tokenizer for text processing
        args: CLI arguments
        config: Full YAML config dict
    """
    # Resume from checkpoint
    start_step, start_epoch, start_batch = self.resume_from_checkpoint()
    
    # Initialize progress logger
    progress_logger = TrainingProgressLogger(args.output_path)
    
    # Main pipeline loop
    for step_idx, step_config in enumerate(self.config.steps):
        if step_idx < start_step:
            continue  # Skip completed steps
        
        print(f"\n{'='*80}")
        print(f"PIPELINE STEP {step_idx+1}/{len(self.config.steps)}: {step_config.name}")
        print(f"{'='*80}\n")
        
        # Configure models for this step
        self.configure_step_models(step_config)
        
        # Create optimizers
        optimizers = self._create_step_optimizers(step_config, models)
        
        # Create schedulers
        schedulers = self._create_step_schedulers(step_config, optimizers)
        
        # Prepare with accelerator
        prepared = self.accelerator.prepare(*optimizers.values(), *schedulers.values())
        # Unpack prepared objects...
        
        # Create trainers
        trainers = self._create_step_trainers(step_config, models, optimizers, schedulers)
        
        # Training loop for this step
        for epoch in range(start_epoch if step_idx == start_step else 0, step_config.n_epochs):
            for batch_idx, (imgs, input_ids) in enumerate(dataloader):
                # Skip batches if resuming mid-epoch
                if step_idx == start_step and epoch == start_epoch and batch_idx < start_batch:
                    continue
                
                # Train VAE
                if step_config.train_vae and trainers.get("vae"):
                    vae_losses = trainers["vae"].train_step(imgs)
                    self.update_metrics("vae_loss", vae_losses["vae"])
                
                # Train Flow
                if (step_config.train_diff or step_config.train_diff_full) and trainers.get("flow"):
                    flow_losses = trainers["flow"].train_step(imgs, input_ids, attention_mask)
                    self.update_metrics("flow_loss", flow_losses["flow_loss"])
                
                # Log progress
                if batch_idx % args.log_interval == 0:
                    self._log_progress(step_idx, epoch, batch_idx, args)
                
                # Save checkpoint
                if batch_idx % args.checkpoint_save_interval == 0 and batch_idx > 0:
                    self._save_checkpoint(step_idx, epoch, batch_idx, models, optimizers, schedulers)
            
            # Check transition criteria
            if self.should_transition(step_config, epoch):
                print(f"Transition criteria met, moving to next step")
                break
        
        # Cleanup
        del optimizers, schedulers, trainers
        torch.cuda.empty_cache()
```

### Phase 3b.3: Helper Methods

```python
def _create_step_optimizers(self, step_config, models):
    """Create optimizers for current step from inline config."""
    optimizers = {}
    
    if not step_config.optimization or not step_config.optimization.optimizers:
        return optimizers  # Use defaults or skip
    
    for name, opt_config in step_config.optimization.optimizers.items():
        if name == "vae":
            params = list(models["compressor"].parameters()) + list(models["expander"].parameters())
        elif name == "flow":
            params = models["flow_processor"].parameters()
        elif name == "text_encoder":
            params = models["text_encoder"].parameters()
        elif name == "discriminator":
            params = models["D_img"].parameters()
        else:
            continue
        
        # Create optimizer from config
        optimizer = create_optimizer(params, opt_config)
        optimizers[name] = optimizer
    
    return optimizers

def _create_step_schedulers(self, step_config, optimizers):
    """Create schedulers for current step from inline config."""
    schedulers = {}
    
    if not step_config.optimization or not step_config.optimization.schedulers:
        return schedulers
    
    # Calculate total steps for this step
    total_steps = step_config.n_epochs * len(self.dataloader)
    
    for name, sched_config in step_config.optimization.schedulers.items():
        if name not in optimizers:
            continue
        
        scheduler = create_scheduler(optimizers[name], sched_config, total_steps)
        schedulers[name] = scheduler
    
    return schedulers

def _create_step_trainers(self, step_config, models, optimizers, schedulers):
    """Create trainers for current step."""
    trainers = {}
    
    if step_config.train_vae and "vae" in optimizers:
        trainers["vae"] = VAETrainer(
            compressor=models["compressor"],
            expander=models["expander"],
            optimizer=optimizers["vae"],
            scheduler=schedulers.get("vae"),
            use_spade=step_config.train_spade,
            discriminator=models["D_img"] if step_config.gan_training else None,
            discriminator_optimizer=optimizers.get("discriminator"),
            discriminator_scheduler=schedulers.get("discriminator"),
            accelerator=self.accelerator,
            # ... other params from step_config
        )
    
    if (step_config.train_diff or step_config.train_diff_full) and "flow" in optimizers:
        trainers["flow"] = FlowTrainer(
            flow_processor=models["flow_processor"],
            text_encoder=models["text_encoder"],
            compressor=models["compressor"],
            optimizer=optimizers["flow"],
            scheduler=schedulers.get("flow"),
            text_encoder_optimizer=optimizers.get("text_encoder"),
            text_encoder_scheduler=schedulers.get("text_encoder"),
            accelerator=self.accelerator,
            # ... other params
        )
    
    return trainers

def _save_checkpoint(self, step_idx, epoch, batch_idx, models, optimizers, schedulers):
    """Save checkpoint with pipeline metadata."""
    # Get pipeline metadata
    metadata = self.get_pipeline_metadata(step_idx, epoch, batch_idx)
    
    # Save models
    self.checkpoint_manager.save_models(
        diffuser=models["diffuser"],
        text_encoder=models["text_encoder"],
        discriminators={"D_img": models["D_img"]} if models.get("D_img") else None,
    )
    
    # Save training state with pipeline metadata
    self.checkpoint_manager.save_training_state(
        epoch=epoch,
        batch_idx=batch_idx,
        global_step=metadata["global_step"],
        optimizers=optimizers,
        schedulers=schedulers,
        ema=None,  # TODO: Add EMA support
        pipeline_metadata=metadata,
    )
```

### Phase 3b.4: Update train_pipeline()

```python
def train_pipeline(args, config):
    """Pipeline-based training loop for FluxFlow."""
    # ... existing initialization code ...
    
    # Initialize models
    models = initialize_models(args, device)
    
    # Load checkpoints if resuming
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        loaded_states = checkpoint_manager.load_models_parallel(checkpoint_path=args.model_checkpoint)
        # Apply state dicts...
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, ...)
    
    # Initialize dataloader
    dataloader, sampler = initialize_dataloader(args, tokenizer, device)
    
    # Run pipeline
    orchestrator.run(
        models=models,
        dataloader=dataloader,
        tokenizer=tokenizer,
        args=args,
        config=config,
    )
```

## Testing Strategy

### Unit Tests (Existing ✅)
- ✅ Pipeline config parsing
- ✅ Pipeline config validation
- ✅ Orchestrator initialization
- ✅ Model freeze/unfreeze
- ✅ Metric tracking
- ✅ Transition criteria evaluation

### Integration Tests (TODO)
- [ ] Model initialization helper
- [ ] Dataloader initialization helper
- [ ] Single-step training (minimal)
- [ ] Multi-step training (minimal)
- [ ] Checkpoint save/resume with pipeline metadata
- [ ] Transition criteria (loss threshold)

### End-to-End Tests (TODO)
- [ ] Full 2-step pipeline on toy dataset
- [ ] Resume from mid-step checkpoint
- [ ] Validate SPADE ON/OFF hypothesis

## Design Decisions

### Why separate initialization from orchestrator?

**Pros:**
- Orchestrator focuses on orchestration logic
- Initialization can be reused between pipeline and legacy modes
- Easier to test in isolation
- Models can be pre-loaded with different strategies

**Cons:**
- More parameters to pass to `run()`
- Less self-contained

**Decision**: Separate initialization for flexibility and testability.

### Why inline YAML for optimizers/schedulers?

**Pros:**
- Single source of truth (no external JSON files)
- Step-specific configurations visible in one place
- Easier to version control
- No file path management

**Cons:**
- More verbose YAML
- No config reuse across steps (but defaults handle this)

**Decision**: Inline YAML for clarity and maintainability.

### Why NotImplementedError for run()?

**Pros:**
- Honest about current state
- Clear signal that work is needed
- Validation and planning work is still valuable
- Allows incremental implementation

**Cons:**
- Feature not usable yet
- May disappoint users

**Decision**: Pragmatic approach - document architecture thoroughly, implement incrementally.

## Next Steps (Priority Order)

1. **Extract initialization helpers** from train_legacy() - enables code reuse
2. **Implement orchestrator._create_step_optimizers()** - critical for per-step configs
3. **Implement orchestrator._create_step_schedulers()** - critical for per-step configs
4. **Implement orchestrator._create_step_trainers()** - critical for per-step training
5. **Implement orchestrator training loop** - core execution logic
6. **Add integration tests** - verify end-to-end flow
7. **Update documentation** - user guide and migration path

## References

- [Pipeline Config](../src/fluxflow_training/training/pipeline_config.py)
- [Pipeline Orchestrator](../src/fluxflow_training/training/pipeline_orchestrator.py)
- [Train Script](../src/fluxflow_training/scripts/train.py)
- [Config Example](../config.example.yaml)
