# Pipeline Training Architecture

**Status**: ✅ **FULLY IMPLEMENTED** (v0.2.0+)

**Implementation**: 1035 lines in `src/fluxflow_training/training/pipeline_orchestrator.py`

## Overview

Multi-step training pipelines allow sequential training phases with different configurations, enabling:
- **Hypothesis testing** (e.g., SPADE OFF → SPADE ON, GAN-only → VAE+GAN)
- **Staged training** (VAE warmup → Flow training)
- **Per-step optimization** (different learning rates, schedulers per phase)
- **Selective freezing** (freeze encoder in one step, unfreeze in next)

## Quick Start

### Minimal Pipeline Example

```yaml
training:
  pipeline:
    steps:
      - name: "vae_warmup"
        n_epochs: 10
        train_vae: true
        gan_training: false
        
      - name: "vae_gan"
        n_epochs: 20
        train_vae: true
        gan_training: true
```

### Run Pipeline Training

```bash
fluxflow-train --config config.yaml
```

Pipeline mode is automatically detected when `training.pipeline.steps` is present in config.

---

## Configuration Reference

### Pipeline Structure

```yaml
training:
  # Global training settings
  batch_size: 4
  workers: 8
  
  pipeline:
    # Optional: Global defaults for all steps
    defaults:
      freeze:
        - text_encoder
      optimization:
        optimizers:
          vae:
            type: "AdamW"
            lr: 0.0001
    
    # Required: List of training steps
    steps:
      - name: "step1"
        n_epochs: 10
        # ... step config
      
      - name: "step2"
        n_epochs: 5
        # ... step config
```

### Step Configuration

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique step identifier (used in checkpoints, samples) |
| `n_epochs` | `int` | Number of epochs for this step |

#### Training Modes

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_vae` | `bool` | `false` | Train VAE encoder/decoder with reconstruction loss |
| `train_spade` | `bool` | `false` | Use SPADE conditioning in decoder |
| `gan_training` | `bool` | `false` | Train GAN discriminator |
| `use_lpips` | `bool` | `false` | Add LPIPS perceptual loss (adds ~6-8GB VRAM) |
| `train_diff` | `bool` | `false` | Train flow processor (partial) |
| `train_diff_full` | `bool` | `false` | Train flow processor (full) |

#### Special Training Modes

**GAN-Only Mode** (train encoder/decoder with adversarial loss, no reconstruction):
```yaml
train_vae: false          # Don't compute reconstruction loss  
gan_training: true        # Train GAN discriminator
train_spade: true         # Optional: SPADE conditioning
```

**Memory savings**: Skipping reconstruction loss saves ~8-12GB VRAM.

**Reconstruction Training** (VAE only, no GAN):
```yaml
train_vae: true
gan_training: false
use_lpips: true           # Optional: perceptual loss (adds ~6-8GB VRAM)
```

#### Model Freezing

```yaml
freeze:
  - compressor
  - expander
  - flow_processor
  - text_encoder
  - discriminator

unfreeze:  # Explicit unfreeze (overrides freeze)
  - compressor
```

#### Loss Weights

```yaml
kl_beta: 0.0001           # KL divergence weight
kl_warmup_steps: 5000     # Steps to reach full kl_beta
kl_free_bits: 0.0         # Free bits threshold (nats)
lambda_adv: 0.5           # GAN adversarial loss weight
lambda_lpips: 0.1         # LPIPS perceptual loss weight
```

#### Optimization (Per-Step)

```yaml
optimization:
  optimizers:
    vae:
      type: "AdamW"
      lr: 0.0001
      betas: [0.9, 0.999]
      weight_decay: 0.01
    discriminator:
      type: "AdamW"
      lr: 0.0001
    flow:
      type: "AdamW"
      lr: 0.00005
    text_encoder:
      type: "SGD"
      lr: 0.00001
      momentum: 0.9
  
  schedulers:
    vae:
      type: "CosineAnnealingLR"
      eta_min_factor: 0.1
    discriminator:
      type: "StepLR"
      step_size: 10
      gamma: 0.5
```

**Supported Optimizers**: `AdamW`, `Adam`, `SGD`, `RMSprop`  
**Supported Schedulers**: `CosineAnnealingLR`, `StepLR`, `ExponentialLR`, `ReduceLROnPlateau`

#### Transition Criteria

Control when to move to the next step:

**Epoch-based (default)**:
```yaml
transition_on:
  mode: "epoch"
  # Automatically transitions after n_epochs completes
```

**Loss threshold**:
```yaml
transition_on:
  mode: "loss_threshold"
  metric: "vae_loss"       # Metric to monitor
  threshold: 0.05          # Transition when metric < threshold
  max_epochs: 50           # Safety limit (prevent infinite training)
```

**Supported metrics**: `vae_loss`, `kl_loss`, `flow_loss`, `g_loss`, `d_loss`

#### Testing/Debugging

```yaml
max_steps: 30             # Limit batches per epoch (for quick testing)
```

---

## Architecture Components

### TrainingPipelineOrchestrator

Main class coordinating multi-step training.

**Location**: `src/fluxflow_training/training/pipeline_orchestrator.py` (1035 lines)

**Key Methods**:
- `run()` - Execute complete pipeline
- `configure_step_models()` - Freeze/unfreeze models per step
- `should_transition()` - Check transition criteria
- `update_metrics()` - Track smoothed metrics for loss thresholds

### PipelineConfig

Dataclass representing pipeline configuration.

**Location**: `src/fluxflow_training/training/pipeline_config.py`

**Validation**: `PipelineConfigValidator` checks:
- Step names are unique
- Freeze/unfreeze don't conflict
- Training modes are valid
- Optimizer/scheduler types are supported
- Transition criteria are properly configured

---

## Checkpoint Format

Pipeline checkpoints include step-local state:

```python
{
  "epoch": 42,              # Global epoch counter
  "global_step": 12847,     # Global step counter
  "models": {...},          # Model state dicts
  "optimizers": {...},      # Optimizer state
  "schedulers": {...},      # Scheduler state
  "ema": {...},            # EMA state
  
  # Pipeline-specific metadata
  "pipeline_metadata": {
    "current_step": 2,                  # Step index (0-based)
    "current_step_name": "vae_gan",     # Step name
    "step_start_epoch": 30,             # When this step started
    "current_step_epoch": 12,           # Epoch within current step
    "step_metrics": {                   # Per-step metric history
      "vae_warmup": {"vae_loss": [...], "kl_loss": [...]},
      "vae_gan": {"vae_loss": [...], "g_loss": [...]}
    }
  }
}
```

### Resume Behavior

When resuming from a pipeline checkpoint:
1. Loads step index, epoch, and batch from metadata
2. Skips completed steps
3. Resumes from mid-step if training was interrupted
4. Recreates optimizers/schedulers for current step
5. Applies correct freeze/unfreeze configuration

**Example**:
```bash
# Training interrupted at step 2, epoch 5, batch 127
fluxflow-train --config config.yaml

# Output:
# Resuming from checkpoint: step=2 (vae_gan), epoch=5, batch=127
# Skipping steps: vae_warmup, vae_spade_off
# Starting step: vae_gan, epoch 5, batch 127
```

---

## Sample Naming Convention

Pipeline mode uses structured naming for samples:

### Format

**Mid-epoch samples** (generated every `checkpoint_save_interval` batches):
```
{stepname}_{step:03d}_{epoch:03d}_{batch:05d}_{hash}-{suffix}.webp
```

**End-of-epoch samples**:
```
{stepname}_{step:03d}_{epoch:03d}_{hash}-{suffix}.webp
```

### Examples

```
vae_warmup_001_001_00010_abc123-original.webp    # Step 1, epoch 1, batch 10
vae_warmup_001_001_00020_abc123-ctx.webp         # Step 1, epoch 1, batch 20
vae_warmup_001_001_abc123-nr_o.webp              # Step 1, epoch 1, end-of-epoch
vae_gan_002_005_def456-original.webp             # Step 2, epoch 5, end-of-epoch
```

### Suffix Meanings

| Suffix | Description |
|--------|-------------|
| `-original` or `_ns_i` | Original input image |
| `-nr_o` | VAE reconstruction output (no random sampling) |
| `_ns_o` | VAE reconstruction with random sampling |
| `-ctx` | Context vector visualization |
| `-nc` | Non-context (image without context) |

---

## Logging Output

### Console Output Format

```
[HH:MM:SS] Step {name} ({idx}/{total}) | Epoch {e}/{n} | Batch {b}/{total} | VAE: X.XX | KL: X.XX | G: X.XX | D: X.XX | LPIPS: X.XX | Xs/batch
```

**Example**:
```
[00:15:23] Step vae_gan (2/4) | Epoch 5/20 | Batch 127/500 | VAE: 0.0234 | KL: 12.45 | G: 0.156 | D: 0.089 | LPIPS: 0.0812 | 3.2s/batch
```

### Metrics Shown by Training Mode

| Mode | Metrics Displayed |
|------|-------------------|
| `train_vae: true` | VAE, KL, (G, D if gan_training), (LPIPS if use_lpips) |
| `train_vae: false, gan_training: true` | VAE (0.0), KL, G, D |
| `train_diff: true` | Flow |

### JSONL Metrics File

Pipeline mode creates step-specific metrics files:

```
outputs/flux/graph/training_metrics_vae_warmup.jsonl
outputs/flux/graph/training_metrics_vae_gan.jsonl
outputs/flux/graph/training_metrics_flow.jsonl
```

**Format**:
```json
{
  "timestamp": "2025-12-09T17:30:45.123456",
  "session_id": "20251209_173000",
  "epoch": 5,
  "batch": 127,
  "global_step": 2847,
  "metrics": {
    "vae_loss": 0.0234,
    "kl_loss": 12.45,
    "g_loss": 0.156,
    "d_loss": 0.089,
    "lpips_loss": 0.0812
  }
}
```

---

## Validation

Validate pipeline config before training:

```bash
fluxflow-train --config pipeline.yaml --validate-pipeline
```

**Checks**:
- ✅ YAML syntax valid
- ✅ Required fields present (`name`, `n_epochs`)
- ✅ Step names unique
- ✅ Training modes valid (at least one enabled)
- ✅ Freeze/unfreeze no conflicts
- ✅ Optimizer types supported
- ✅ Scheduler types supported
- ✅ Transition criteria valid
- ✅ Loss threshold metrics exist

**Example Output**:
```
✅ Pipeline validation passed!

Pipeline Summary:
  Total steps: 4
  Total epochs: 85
  
Step 1: vae_warmup (10 epochs)
  Training: VAE
  Freeze: flow_processor, text_encoder
  
Step 2: vae_gan (20 epochs)
  Training: VAE, GAN
  Transition: loss_threshold (vae_loss < 0.05, max 50 epochs)
  
...
```

---

## Complete Example

See `test_pipeline_minimal.yaml` for a working minimal example:

```yaml
training:
  batch_size: 2
  workers: 1
  
  pipeline:
    steps:
      - name: "vae_only"
        n_epochs: 2
        max_steps: 2              # Quick test: 2 batches only
        train_vae: true
        gan_training: false
        freeze:
          - flow_processor
          - text_encoder
        optimization:
          optimizers:
            vae:
              type: "AdamW"
              lr: 0.0001
        transition_on:
          mode: "epoch"
      
      - name: "vae_gan"
        n_epochs: 2
        max_steps: 2
        train_vae: true
        gan_training: true
        freeze:
          - flow_processor
          - text_encoder
        optimization:
          optimizers:
            vae:
              type: "AdamW"
              lr: 0.00005
            discriminator:
              type: "AdamW"
              lr: 0.00005
        transition_on:
          mode: "epoch"
```

Run with:
```bash
fluxflow-train --config test_pipeline_minimal.yaml
```

---

## Implementation Notes

### Training Flow

1. **Initialize**: Parse config, create checkpoint manager
2. **Resume** (if checkpoint exists): Load step/epoch/batch from metadata
3. **For each step**:
   - Configure models (freeze/unfreeze)
   - Create optimizers from inline config
   - Create schedulers from inline config
   - Create trainers (VAE, Flow)
   - For each epoch:
     - Training loop
     - Update metrics
     - Save checkpoint (mid-epoch + end-of-epoch)
     - Generate samples
   - Check transition criteria
   - If threshold not met and `max_epochs` reached, log warning and transition
4. **Complete**: Save final checkpoint

### Gradient Flow in GAN-Only Mode

When `train_vae: false` but `gan_training: true`:

- **Encoder gradients**: KL divergence + GAN generator loss
- **Decoder gradients**: GAN generator loss
- **Discriminator gradients**: Hinge loss + R1 penalty

The latent is **NOT detached** when `train_reconstruction=false`, allowing encoder to learn from GAN gradients.

### EMA Handling

EMA (Exponential Moving Average) is created when any of these are true:
- `train_vae: true`
- `gan_training: true`
- `train_spade: true`
- `use_lpips: true`

This ensures EMA exists even in GAN-only mode.

### Auto-Create Missing Models

**New in Unreleased** (PR #13): Pipeline orchestrator automatically creates missing models when transitioning between steps.

#### How It Works

When transitioning from VAE training to Flow training, the following models are auto-created if missing:

| Model | Created When | Default Parameters |
|-------|--------------|-------------------|
| `flow_processor` | `train_diff: true` or `train_diff_full: true` | `feature_maps_dim` from args |
| `text_encoder` | Flow training enabled | `text_embedding_dim` from args |
| `compressor` (for Flow) | Flow training enabled | `vae_dim` from args |
| `expander` | VAE training with GAN | `vae_dim` from args |
| `D_img` (discriminator) | `gan_training: true` | `vae_dim` from args |

#### Example Scenario

```yaml
steps:
  - name: vae_warmup
    train_vae: true
    gan_training: false
    # Models: compressor, decoder (auto-created by VAE trainer)
  
  - name: flow_training
    train_diff: true
    freeze: [compressor, expander]
    # Auto-creates: flow_processor, text_encoder, compressor (for Flow)
    # Logs: "Auto-created flow_processor with feature_maps_dim=128"
```

#### User Impact

- **Before (v0.3.0 and earlier)**: Crash with `AttributeError: 'NoneType' object has no attribute 'forward'`
- **After (Unreleased)**: Models created automatically with default parameters, training continues

#### When Auto-Creation Happens

Models are created in `_ensure_required_models()` method (lines 579-713 in `pipeline_orchestrator.py`):

1. **Before creating trainers** for each step
2. **Only if missing** from `models` dict
3. **With warnings logged**: `"Auto-created {model_name} with default parameters"`
4. **On correct device**: Models moved to `args.device`

#### Disabling Auto-Creation

Auto-creation cannot be disabled (by design). If you need custom model parameters:

**Option 1**: Pre-create models before pipeline:
```python
from fluxflow.models import FlowProcessor
flow_processor = FlowProcessor(
    feature_maps_dim=256,  # Custom value
    # ... other custom params
)
models = {"flow_processor": flow_processor}
```

**Option 2**: Use first step to initialize:
```yaml
steps:
  - name: init_models
    n_epochs: 1
    train_diff: true  # Creates flow_processor with defaults
  
  - name: actual_training
    n_epochs: 100
    train_diff_full: true
```

---

## Troubleshooting

### "AttributeError: 'list' object has no attribute 'get'"

**Cause**: Missing `steps:` wrapper in config.

**Fix**:
```yaml
# Wrong
pipeline:
  - name: "step1"

# Correct
pipeline:
  steps:
    - name: "step1"
```

### "AttributeError: 'NoneType' object has no attribute 'update'" (EMA)

**Cause**: Bug in versions < 0.2.0 where EMA wasn't created for GAN-only mode.

**Fix**: Upgrade to v0.2.0+ or set `train_vae: true`.

### Empty metrics in JSONL file

**Cause**: Bug in versions < 0.2.0 where VAE trainer wasn't called when `train_vae: false`.

**Fix**: Upgrade to v0.2.0+.

### Samples not updating during training

**Cause**: Mid-epoch sample generation was disabled in early versions.

**Fix**: Upgrade to v0.2.0+ where mid-epoch samples respect `checkpoint_save_interval`.

### Out of Memory (OOM) at 47GB+ on 48GB GPU

**Cause**: LPIPS + GAN + SPADE combined can exceed 48GB VRAM.

**Fix**:
- Reduce batch size: `batch_size: 2` or `1`
- Disable LPIPS: `use_lpips: false` (saves ~6-8GB)
- Reduce image size: `img_size: 512` (saves ~10-15GB)
- Disable SPADE: `train_spade: false` (saves ~3-5GB)
- Use GAN-only mode: `train_vae: false, gan_training: true` (saves ~8-12GB)

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) "Limited VRAM Strategy" for more.

---

## Related Documentation

- [README.md](../README.md) - Quick start and features
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training strategies and memory optimization
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide

---

**Last Updated**: 2025-12-11  
**Version**: 0.2.1  
**Implementation**: ✅ Fully operational (1035 lines in pipeline_orchestrator.py)
