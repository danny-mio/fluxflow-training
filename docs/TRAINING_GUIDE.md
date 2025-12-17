# FluxFlow Training Guide

A comprehensive guide to configuring and running training for FluxFlow text-to-image models.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Training Command Reference](#training-command-reference)
- [Parameter Details](#parameter-details)
- [Training Strategies](#training-strategies)
- [Configuration Examples](#configuration-examples)
- [Troubleshooting](#troubleshooting)

## Memory Requirements & OOM Prevention

**Critical Information** (empirically measured, December 2025):

### VRAM Usage by Configuration

**VAE Training** (batch_size=4, vae_dim=128, img_size=1024):
- **VAE only** (no GAN): ~18-22GB VRAM
- **VAE + GAN**: ~25-30GB VRAM
- **VAE + GAN + LPIPS**: ~28-35GB VRAM
- **VAE + GAN + LPIPS + SPADE**: ~35-42GB VRAM
- **Peak observed**: 47.4GB on A6000 48GB → **triggered OOM**

**Flow Training** (batch_size=1, feature_maps_dim=128):
- ~24-30GB VRAM

**Minimum Viable** (reduced dimensions):
- 16GB VRAM: `batch_size=2, vae_dim=64, img_size=512, gan_training=false`
- 24GB VRAM: `batch_size=1, vae_dim=128, img_size=1024, use_lpips=false`

### OOM Prevention Strategies

If you hit **47GB+ on 48GB GPU** (or equivalent):

**1. Reduce Batch Size** (most effective):
```yaml
batch_size: 2  # or 1
```

**2. Disable LPIPS** (saves ~6-8GB):
```yaml
use_lpips: false
```

**3. Reduce Image Size** (saves ~10-15GB):
```yaml
img_size: 512  # instead of 1024
```

**4. Use GAN-Only Mode** (saves ~8-12GB by skipping reconstruction):
```yaml
train_vae: false
gan_training: true
train_spade: true
```

**5. Disable SPADE** (saves ~3-5GB):
```yaml
train_spade: false
```

**6. Reduce Dimensions** (saves ~5-10GB):
```yaml
vae_dim: 64              # instead of 128
feature_maps_dim: 64
feature_maps_dim_disc: 8
```

**7. Use FP16** (saves ~20-30% if GPU supports Tensor Cores):
```yaml
use_fp16: true  # RTX 3090/4090 recommended
```

### Recent Optimizations (v0.2.1)

FluxFlow v0.2.1 includes critical memory optimizations:
- Removed LPIPS gradient checkpointing (caused OOM spikes)
- Removed dataloader prefetch_factor (reduced memory overhead)
- CUDA cache clearing between batches
- R1 gradient penalty fix (prevented memory leaks)

**If still hitting OOM after v0.2.1**, apply strategies 1-7 above.

### Hardware Recommendations

| GPU VRAM | Recommended Config | Max Quality Config |
|----------|-------------------|-------------------|
| 8GB | batch=1, dim=32, img=512, no_gan | Not recommended |
| 12GB | batch=2, dim=64, img=512, gan | batch=1, dim=64, img=512, gan+lpips |
| 16GB | batch=2, dim=64, img=1024, gan | batch=1, dim=128, img=512, gan+lpips |
| 24GB | batch=2, dim=128, img=1024, gan | batch=1, dim=128, img=1024, gan+lpips+spade |
| 48GB | batch=4, dim=128, img=1024, gan+lpips | batch=2, dim=256, img=1024, gan+lpips+spade |

**Note**: 48GB configs may still OOM if LPIPS+GAN+SPADE all enabled. Monitor with `nvidia-smi`.

---

## Overview

FluxFlow uses a unified training script (`packages/training/src/fluxflow_training/scripts/train.py`) that supports:
- **VAE Training**: Train the autoencoder (compressor + expander)
- **Flow Model Training**: Train the diffusion model for text-to-image generation
- **Joint Training**: Train both VAE and flow simultaneously (advanced)

The training process is highly configurable with parameters for data, model architecture, training behavior, and output.

## Configuration Methods

FluxFlow supports two configuration approaches:

### 🎯 YAML Config Files (Recommended for Production)

**Use when:**
- Running multi-step training pipelines
- Need reproducible, version-controlled configs
- Want inline optimizer/scheduler customization per step
- Training production models

**Example:**
```yaml
# config.yaml
data:
  data_path: "/path/to/images"
  captions_file: "/path/to/captions.txt"

model:
  vae_dim: 128
  feature_maps_dim: 128

training:
  batch_size: 4
  pipeline:
    steps:
      - name: "vae_training"
        n_epochs: 50
        train_vae: true
        train_spade: true
        optimizers:
          vae:
            optimizer_type: "AdamW"
            lr: 0.0001
            weight_decay: 0.01
        schedulers:
          vae:
            scheduler_type: "CosineAnnealingLR"
            eta_min_factor: 0.1

output:
  output_path: "outputs/vae"
```

**Run:** `fluxflow-train --config config.yaml`

---

### 🔧 CLI Arguments (For Quick Tests Only)

**Use when:**
- Running quick experiments
- Testing single training modes (VAE-only or Flow-only)
- Debugging or prototyping

**Example:**
```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --train_vae \
  --n_epochs 50 \
  --batch_size 2 \
  --lr 1e-5
```

**Limitations:**
- ❌ No multi-step pipelines
- ❌ No per-step optimizer customization
- ❌ No inline YAML optimizer config
- ❌ Hard to version control
- ❌ Complex commands become unwieldy

---

### 📊 Feature Comparison

| Feature | YAML Config | CLI Args |
|---------|-------------|----------|
| Multi-step pipelines | ✅ | ❌ |
| Inline optimizer config | ✅ | ❌ (requires external JSON) |
| Per-step freezing | ✅ | ❌ |
| Loss-based transitions | ✅ | ❌ |
| Version control friendly | ✅ | ⚠️ (must maintain scripts) |
| Quick experiments | ⚠️ (requires YAML file) | ✅ |

**Recommendation:**
- **Beginners / Quick Tests:** Start with CLI args (see Quick Start below)
- **Production / Serious Training:** Use YAML config + pipeline mode (see "Pipeline Training Mode" section)

---

## Quick Start

Choose your path based on your needs:

### Path A: CLI Quick Test (5 Minutes) ⚡

For quick experiments and learning, use CLI arguments:

**Step 1: Prepare Data**
```bash
# Your directory structure:
# /path/to/images/
#   ├── image1.jpg
#   ├── image2.png
#   └── ...
# /path/to/captions.txt (tab-separated: filename\tcaption)

# Example captions.txt:
# image1.jpg	a photo of a cat sitting on a couch
# image2.png	an illustration of mountains at sunset
```

**Step 2: Run Quick VAE Training**
```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/quick_test \
  --train_vae \
  --n_epochs 5 \
  --batch_size 2 \
  --lr 1e-5 \
  --vae_dim 64  # Reduced for speed
```

**⚠️ When you're ready for serious training, switch to Path B (YAML Config).**

---

### Path B: YAML Config (Production-Ready) 🚀

For reproducible, production-quality training with multi-step pipelines:

**Step 1: Prepare Data** (same as Path A)

**Step 2: Create `config.yaml`**
```yaml
data:
  data_path: "/path/to/images"
  captions_file: "/path/to/captions.txt"

model:
  vae_dim: 128
  feature_maps_dim: 128

training:
  batch_size: 2
  workers: 8
  
  pipeline:
    steps:
      # Step 1: VAE training with GAN
      - name: "vae_training"
        n_epochs: 50
        train_vae: true
        train_spade: true
        gan_training: true
        
        optimizers:
          vae:
            optimizer_type: "AdamW"
            lr: 0.00001  # Same as 1e-5
            betas: [0.9, 0.999]
            weight_decay: 0.01
          discriminator:
            optimizer_type: "AdamW"
            lr: 0.00001
            betas: [0.0, 0.9]
            amsgrad: true
        
        schedulers:
          vae:
            scheduler_type: "CosineAnnealingLR"
            eta_min_factor: 0.1
          discriminator:
            scheduler_type: "CosineAnnealingLR"
            eta_min_factor: 0.1

output:
  output_path: "outputs/production"
```

**Step 3: Run Training**
```bash
fluxflow-train --config config.yaml
```

**✅ Benefits:**
- Reproducible configs (version control friendly)
- Multi-step pipeline support
- Per-step optimizer customization
- No complex CLI commands

**See "Pipeline Training Mode" section below for multi-stage training (VAE → Flow → Fine-tune).**

## Training Command Reference

### Complete Parameter List

#### Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_path` | str | - | Path to directory containing training images |
| `--captions_file` | str | - | Path to tab-separated captions file (filename\tcaption) |
| `--fixed_prompt_prefix` | str | None | Optional fixed text to prepend to all prompts (e.g., "style anime") |
| `--use_webdataset` | flag | False | Use WebDataset streaming instead of local files |
| `--webdataset_token` | str | None | HuggingFace token for accessing streaming datasets |
| `--webdataset_url` | str | (default) | WebDataset URL pattern (e.g., "hf://datasets/user/repo/*.tar") |
| `--webdataset_image_key` | str | "png" | Image field in tar samples (jpg, png, etc.) |
| `--webdataset_label_key` | str | "json" | Metadata field in tar samples |
| `--webdataset_caption_key` | str | "prompt" | Caption key within JSON (e.g., "prompt", "caption") |
| `--use_tt2m` | flag | False | (Deprecated) Use `--use_webdataset` instead |
| `--tt2m_token` | str | None | (Deprecated) Use `--webdataset_token` instead |

**Example:**
```bash
# Local dataset
--data_path /data/images --captions_file /data/captions.txt

# Local dataset with fixed prefix for style-specific training
--data_path /data/anime --captions_file /data/captions.txt --fixed_prompt_prefix "style anime"

# WebDataset streaming (default: TTI-2M)
--use_webdataset --webdataset_token hf_your_actual_token

# Custom WebDataset with specific field mappings
--use_webdataset --webdataset_token hf_token \
  --webdataset_url "hf://datasets/user/dataset/*.tar" \
  --webdataset_image_key "png" \
  --webdataset_caption_key "caption"

# Legacy (still works but deprecated):
# --use_tt2m --tt2m_token hf_your_actual_token
```

**Fixed Prompt Prefix** (added to captions at training time):

The `--fixed_prompt_prefix` parameter allows you to prepend consistent text to all prompts during training. This is useful for:
- Style-specific fine-tuning (e.g., "style anime", "oil painting style")
- Content-type training (e.g., "photo realistic", "digital art")
- Domain-specific models (e.g., "medical diagram", "architectural rendering")

Example: With `--fixed_prompt_prefix "style anime"`, the prompt "a girl running" becomes "style anime. a girl running"

If not set, prompts are used exactly as provided in the dataset.

#### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_checkpoint` | str | - | Path to checkpoint for resuming training |
| `--vae_dim` | int | 128 | VAE latent dimension (higher = more detail, more VRAM) |
| `--text_embedding_dim` | int | 1024 | Text embedding dimension from BERT |
| `--feature_maps_dim` | int | 128 | Flow model feature dimension |
| `--feature_maps_dim_disc` | int | 8 | Discriminator feature dimension |
| `--pretrained_bert_model` | str | - | Path to pretrained BERT checkpoint (optional) |

**Dimension Guidelines:**
- **Limited VRAM (8GB)**: `vae_dim=32, feature_maps_dim=32, feature_maps_dim_disc=32`
- **Mid VRAM (12-16GB)**: `vae_dim=64, feature_maps_dim=64, feature_maps_dim_disc=8`
- **High VRAM (24GB+)**: `vae_dim=128, feature_maps_dim=128, feature_maps_dim_disc=8`
- **Maximum Quality**: `vae_dim=256, feature_maps_dim=256, feature_maps_dim_disc=16`

**Example:**
```bash
# Resume from checkpoint
--model_checkpoint outputs/flux/flxflow_final.safetensors

# Custom dimensions for limited VRAM
--vae_dim 32 --feature_maps_dim 32 --feature_maps_dim_disc 32
```

#### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_epochs` | int | 1 | Number of epochs to train |
| `--batch_size` | int | 2 | Batch size (reduce if out of memory) |
| `--workers` | int | 1 | Number of data loading workers |
| `--lr` | float | 5e-7 | Learning rate for flow model |
| `--lr_min` | float | 0.1 | Minimum LR multiplier for cosine annealing |
| `--preserve_lr` | flag | False | Load saved learning rates from checkpoint |
| `--optim_sched_config` | str | - | Path to JSON file with optimizer/scheduler configurations |
| `--training_steps` | int | 1 | Inner training steps per batch (gradient accumulation) |
| `--use_fp16` | flag | False | Use mixed precision training (FP16) |
| `--initial_clipping_norm` | float | 1.0 | Gradient clipping norm for stability |

**Learning Rate Guidelines:**
- **VAE Training**: `1e-5` to `5e-5`
- **Flow Training**: `5e-7` to `1e-6`
- **Fine-tuning**: `1e-6` to `5e-6`

**Optimizer/Scheduler Configuration:**

FluxFlow supports advanced per-model optimizer and scheduler configuration via JSON file. This allows you to use different optimizers (Lion, AdamW, Adam, SGD, RMSprop) and schedulers (CosineAnnealingLR, LinearLR, ExponentialLR, etc.) for each model component (flow, vae, text_encoder, discriminator).

**Example optimizer/scheduler config file:**
```json
{
  "optimizers": {
    "flow": {
      "type": "Lion",
      "lr": 5e-7,
      "betas": [0.9, 0.95],
      "weight_decay": 0.01,
      "decoupled_weight_decay": true
    },
    "vae": {
      "type": "AdamW",
      "lr": 5e-7,
      "betas": [0.9, 0.95],
      "weight_decay": 0.01
    },
    "text_encoder": {
      "type": "AdamW",
      "lr": 5e-8,
      "betas": [0.9, 0.99],
      "weight_decay": 0.01
    },
    "discriminator": {
      "type": "AdamW",
      "lr": 5e-7,
      "betas": [0.0, 0.9],
      "weight_decay": 0.001,
      "amsgrad": true
    }
  },
  "schedulers": {
    "flow": {
      "type": "CosineAnnealingLR",
      "eta_min_factor": 0.1
    },
    "vae": {
      "type": "CosineAnnealingLR",
      "eta_min_factor": 0.1
    },
    "text_encoder": {
      "type": "CosineAnnealingLR",
      "eta_min_factor": 0.001
    },
    "discriminator": {
      "type": "CosineAnnealingLR",
      "eta_min_factor": 0.1
    }
  }
}
```

**Supported Optimizers:**
- **Lion**: Memory-efficient optimizer, great for flow models
- **AdamW**: Adam with decoupled weight decay, excellent for VAE and discriminator
- **Adam**: Standard Adam optimizer
- **SGD**: Stochastic gradient descent with momentum
- **RMSprop**: Root mean square propagation

**Supported Schedulers:**
- **CosineAnnealingLR**: Cosine annealing (default, recommended)
- **LinearLR**: Linear learning rate decay
- **ExponentialLR**: Exponential decay
- **ConstantLR**: Constant learning rate
- **StepLR**: Step-wise decay
- **ReduceLROnPlateau**: Reduce on plateau (metric-based)

---

### Optimizer & Scheduler Configuration

FluxFlow supports per-model optimizer and scheduler configuration for fine-grained control.

**Supported Optimizers:** Lion (recommended for flow), AdamW, Adam, SGD, RMSprop

**Supported Schedulers:** CosineAnnealingLR, LinearLR, ExponentialLR, ConstantLR, StepLR, ReduceLROnPlateau

**📚 Detailed References:**
- **[OPTIMIZERS.md](OPTIMIZERS.md)** - Complete optimizer parameter reference with examples
- **[SCHEDULERS.md](SCHEDULERS.md)** - Complete scheduler parameter reference with examples

**Quick Example:**

```yaml
optimizer_config:
  flow_processor:
    type: "Lion"
    lr: 5e-7
    betas: [0.9, 0.95]
    weight_decay: 0.01

scheduler_config:
  flow_processor:
    type: "CosineAnnealingLR"
    T_max: 100
    eta_min_factor: 0.1
```

---

## Parameter Details

### Understanding VAE Dimensions

The `vae_dim` parameter controls the size of the latent space:
- **Higher values** (128, 256): More detail preserved, better quality, more VRAM
- **Lower values** (32, 64): Less detail, faster training, less VRAM

### Understanding Feature Map Dimensions

- `feature_maps_dim`: Controls the flow model's capacity
- `feature_maps_dim_disc`: Controls the discriminator's capacity

For best results, keep `feature_maps_dim` equal to `vae_dim`.

### Mixed Precision Training

Using `--use_fp16` enables mixed precision:
- **Pros**: ~40% faster, ~40% less VRAM, same quality
- **Cons**: Requires NVIDIA GPU with Tensor Cores (RTX series)
- **Recommendation**: Always use on RTX 3090/4090 for 2-4x speedup

### Gradient Accumulation

Using `--training_steps N` accumulates gradients over N batches:
- Effective batch size = `batch_size * training_steps`
- Useful when `batch_size=1` is too small for stable training
- Example: `--batch_size 1 --training_steps 4` = effective batch size of 4

### Resume Training

To resume from a checkpoint:
```bash
--model_checkpoint outputs/flux/flxflow_final.safetensors --preserve_lr
```

The training script automatically saves:
- `training_state.json`: Epoch, batch, global step
- `lr_sav.json`: Learning rates
- `training_states.pt`: Optimizer, scheduler, EMA states
- `sampler_state.pt`: Data sampler state

### KL Divergence Normalization (v0.3.0+)

**BREAKING CHANGE**: KL divergence is now normalized by dimensions (resolution-invariant).

#### What Changed?

**Before (v0.2.x and earlier)**:
- KL divergence was computed by **summing** over all spatial and channel dimensions
- This caused KL to scale with image resolution:
  - 512×512 images: KL ≈ 150,000 (weighted: 0.0001 × 150K = 15.0)
  - 1024×1024 images: KL ≈ 600,000 (weighted: 0.0001 × 600K = 60.0)
- The weighted KL dominated training, causing latent collapse and high contrast reconstructions
- Different `kl_beta` values needed for different resolutions

**After (v0.3.0+)**:
- KL divergence is computed by **averaging** over all dimensions
- KL is now resolution-invariant:
  - Any resolution: KL ≈ 1.2 (weighted: 0.001 × 1.2 = 0.0012)
- Same `kl_beta` works across all resolutions
- Better balance with reconstruction and perceptual losses

#### Migration Guide

If you're upgrading from v0.2.x to v0.3.0+:

1. **Increase `kl_beta` by 10×**:
   ```yaml
   # OLD (v0.2.x)
   kl_beta: 0.0001
   
   # NEW (v0.3.0+)
   kl_beta: 0.001  # Increased 10× to compensate for normalized scale
   ```

2. **Start training from scratch** (do NOT resume from old checkpoints):
   - Old checkpoints were trained with unnormalized KL
   - Resuming with normalized KL will cause training instability
   - You must train fresh VAE weights from scratch

3. **Update monitoring expectations**:
   ```
   # OLD: Expected KL values
   KL (raw): 100,000 - 200,000
   KL (weighted): 10.0 - 20.0
   
   # NEW: Expected KL values
   KL (raw): 1.0 - 2.0
   KL (weighted): 0.001 - 0.002
   ```

#### New Contrast Regularization Loss

v0.3.0 also adds explicit contrast regularization to prevent over-saturation:

```python
# Component 1: Global contrast (per-channel std matching)
# Component 2: Local contrast (per-sample std preservation)
contrast_loss = 0.0

for c in [R, G, B]:
    std_ratio = pred_std / target_std
    contrast_loss += (std_ratio - 1.0)²

contrast_loss += MSE(pred_std_per_sample, target_std_per_sample)
```

**Expected values**: 0.001 - 0.01 (logged as `Contrast`)

**Training logs before**:
```
VAE: 0.0523 | KL: 126562.6031 | Bezier: 0.5020 | ColorStats: 0.0015 | Hist: 0.0199
```

**Training logs after**:
```
VAE: 0.0523 | KL: 1.2665 | Bezier: 0.0520 | ColorStats: 0.0015 | Hist: 0.0199 | Contrast: 0.0012
```

#### Why This Change Matters

The old KL computation caused **high contrast** and **over-saturation** because:
1. Massive KL loss (15.0) vs tiny reconstruction loss (0.05) → 300× imbalance
2. Encoder learned to output near-zero latents (minimizing KL)
3. Decoder forced to use extreme Bezier curves to extract any signal
4. Result: High contrast, oversaturated reconstructions

With normalized KL:
- KL (0.0012) is properly balanced with reconstruction (0.05)
- Encoder learns meaningful latent representations
- Decoder uses moderate Bezier curves
- Result: Natural contrast and saturation

#### Backward Compatibility

If you need legacy behavior (e.g., comparing with old experiments):

```python
# In src/fluxflow_training/training/vae_trainer.py, line ~615
kl = kl_standard_normal(
    mu, logvar,
    free_bits_nats=self.kl_free_bits,
    reduce="mean",
    normalize_by_dims=False  # Set to False for legacy behavior
)
```

**Note**: Legacy behavior will be removed in v0.4.0.

## Pipeline Training Mode

**New in v0.2.0**: Multi-step pipeline training allows you to define sequential training phases with different configurations.

### What is Pipeline Training?

Pipeline training breaks your training workflow into multiple sequential steps, each with its own:
- Training mode (VAE-only, GAN-only, Flow-only, or combinations)
- Learning rate and scheduler
- Freeze/unfreeze configurations
- Loss threshold transitions

### When to Use Pipeline Training

**Use pipeline training for:**
- **Hypothesis testing**: Compare different training strategies (e.g., SPADE OFF → SPADE ON)
- **Staged training**: VAE warmup → GAN training → Flow training
- **Selective freezing**: Train components independently
- **Loss-based transitions**: Automatically move to next step when loss threshold is met
- **Multi-dataset training**: Train different steps on different datasets (see `docs/MULTI_DATASET_TRAINING.md`)

**Use standard training for:**
- Simple single-mode training (VAE-only or Flow-only)
- Quick experiments
- Resume training with same configuration

### Pipeline Resilience Features

**Auto-Create Missing Models** (New in Unreleased):

When transitioning between pipeline steps (e.g., VAE → Flow), required models are automatically created:

```yaml
steps:
  - name: vae_warmup
    train_vae: true
    # Only creates: compressor, decoder
  
  - name: flow_training
    train_diff: true
    # Auto-creates: flow_processor, text_encoder ✨
    # No manual initialization needed!
```

**Auto-created models:**
- `flow_processor` - Created when `train_diff: true` or `train_diff_full: true`
- `text_encoder` - Created for flow training
- `compressor` (for Flow) - Created for flow training if missing
- `expander` - Created for VAE with GAN
- `D_img` (discriminator) - Created when `gan_training: true`

**What you see:**
```
⚠️  Auto-created flow_processor with feature_maps_dim=128
⚠️  Auto-created text_encoder with text_embedding_dim=512
```

This prevents crashes and makes pipeline mode more robust. See `docs/PIPELINE_ARCHITECTURE.md` for details.

### Quick Start: Pipeline Training

**Example 1: VAE Warmup → GAN Training**

```yaml
# config.yaml
data:
  data_path: "/path/to/images"
  captions_file: "/path/to/captions.txt"

training:
  batch_size: 4
  output_path: "outputs/pipeline_training"
  
  pipeline:
    steps:
      - name: "vae_warmup"
        n_epochs: 10
        train_vae: true
        gan_training: false
        lr: 1e-5
        
      - name: "vae_with_gan"
        n_epochs: 40
        train_vae: true
        gan_training: true
        lr: 1e-5
        stop_condition:
          loss_name: "loss_recon"
          threshold: 0.01
```

**Run:**
```bash
fluxflow-train --config config.yaml
```

**Example 2: Multi-Stage with Different Optimizers**

```yaml
training:
  batch_size: 2
  
  pipeline:
    steps:
      # Step 1: VAE warmup with Adam
      - name: "vae_warmup"
        n_epochs: 10
        train_vae: true
        gan_training: false
        optim_sched_config: "configs/adam_warmup.json"
        
      # Step 2: GAN training with Lion
      - name: "gan_training"
        n_epochs: 30
        train_vae: true
        gan_training: true
        train_spade: true
        optim_sched_config: "configs/lion_gan.json"
        
      # Step 3: Flow training
      - name: "flow_training"
        n_epochs: 100
        train_diff_full: true
        freeze_vae: true  # Freeze VAE, train flow only
        optim_sched_config: "configs/lion_flow.json"
```

**Optimizer config example (lion_gan.json):**
```json
{
  "optimizers": {
    "vae": {
      "type": "Lion",
      "lr": 5e-6,
      "weight_decay": 0.01
    },
    "discriminator": {
      "type": "AdamW",
      "lr": 5e-6,
      "betas": [0.0, 0.9],
      "amsgrad": true
    }
  },
  "schedulers": {
    "vae": {"type": "CosineAnnealingLR", "eta_min_factor": 0.1},
    "discriminator": {"type": "CosineAnnealingLR", "eta_min_factor": 0.1}
  }
}
```

### Pipeline-Specific Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `steps[].name` | str | Unique step identifier (used in logs, checkpoints) |
| `steps[].n_epochs` | int | Epochs for this step only |
| `steps[].max_steps` | int | Optional: max batches (for testing) |
| `steps[].freeze_vae` | bool | Freeze VAE encoder/decoder |
| `steps[].freeze_flow` | bool | Freeze flow model |
| `steps[].freeze_text_encoder` | bool | Freeze text encoder |
| `steps[].optim_sched_config` | str | Path to optimizer/scheduler config JSON |
| `steps[].stop_condition.loss_name` | str | Loss to monitor (e.g., "loss_recon", "loss_flow") |
| `steps[].stop_condition.threshold` | float | Exit step when loss < threshold |

### Pipeline Features

#### ✅ Per-Step Checkpoints
- Each step saves its own checkpoints: `flxflow_step_vae_warmup_final.safetensors`
- Resume from any step: automatically loads the last completed step
- Step-specific metrics files: `outputs/graph/training_metrics_vae_warmup.jsonl`

#### ✅ Selective Freezing
- Freeze any combination of models between steps
- Example: Train VAE in step 1, freeze it in step 2 for flow training
- Gradients automatically disabled for frozen models

#### ✅ Loss-Threshold Transitions
- Automatically exit step when loss reaches target
- Useful for adaptive training (exit VAE warmup when reconstruction is good enough)
- Example: `stop_condition: {loss_name: "loss_recon", threshold: 0.01}`

#### ✅ Inline Optimizer/Scheduler Configs
- Different optimizers per step (e.g., Adam warmup → Lion training)
- Different schedulers per step
- Full control over per-model hyperparameters

#### ✅ GAN-Only Training Mode
- `train_reconstruction: false` - Train encoder/decoder with adversarial loss only
- No pixel-level reconstruction loss computed
- Use case: SPADE conditioning without reconstruction overhead
- Example:
  ```yaml
  - name: "gan_only"
    train_vae: true
    gan_training: true
    train_spade: true
    train_reconstruction: false  # GAN-only mode
  ```

#### ✅ Full Resume Support
- Resumes from last completed step
- Preserves optimizer/scheduler/EMA states
- Mid-step resume: continues from exact batch within step

### Complete Pipeline Example

```yaml
# config.yaml - Complete 3-stage training pipeline
data:
  data_path: "/data/images"
  captions_file: "/data/captions.txt"

model:
  vae_dim: 128
  feature_maps_dim: 128
  feature_maps_dim_disc: 8

training:
  batch_size: 4
  workers: 8
  output_path: "outputs/full_pipeline"
  checkpoint_save_interval: 100
  
  pipeline:
    steps:
      # Step 1: VAE reconstruction warmup (no GAN)
      - name: "vae_warmup"
        n_epochs: 10
        train_vae: true
        gan_training: false
        train_spade: false
        lr: 2e-5
        kl_beta: 0.0001
        stop_condition:
          loss_name: "loss_recon"
          threshold: 0.02  # Exit when reconstruction is good
        
      # Step 2: Add SPADE and GAN
      - name: "vae_spade_gan"
        n_epochs: 40
        train_vae: true
        gan_training: true
        train_spade: true
        lr: 1e-5
        lambda_adv: 0.9
        kl_beta: 0.001
        optim_sched_config: "configs/lion_gan.json"
        
      # Step 3: Flow training (freeze VAE)
      - name: "flow_training"
        n_epochs: 100
        train_diff_full: true
        train_vae: false
        freeze_vae: true
        lr: 5e-7
        sample_captions:
          - "a photo of a cat sitting on a couch"
          - "an illustration of mountains at sunset"
        optim_sched_config: "configs/lion_flow.json"
```

**Run:**
```bash
fluxflow-train --config config.yaml
```

**Output structure:**
```
outputs/full_pipeline/
├── flxflow_step_vae_warmup_final.safetensors
├── flxflow_step_vae_spade_gan_final.safetensors
├── flxflow_step_flow_training_final.safetensors
├── graph/
│   ├── training_metrics_vae_warmup.jsonl
│   ├── training_metrics_vae_spade_gan.jsonl
│   ├── training_metrics_flow_training.jsonl
│   ├── training_losses_vae_warmup.png
│   ├── training_losses_vae_spade_gan.png
│   └── training_losses_flow_training.png
└── samples/
    ├── sample_vae_warmup_epoch_5_batch_100.png
    ├── sample_vae_spade_gan_epoch_20_batch_500.png
    └── sample_flow_training_epoch_50_batch_1000.png
```

### Pipeline vs. Standard Training

| Feature | Standard Training | Pipeline Training |
|---------|------------------|------------------|
| Configuration | CLI args | YAML config |
| Stages | Single mode | Multiple sequential steps |
| Per-step checkpoints | ❌ | ✅ |
| Per-step optimizers | ❌ | ✅ |
| Selective freezing | Manual | Per-step config |
| Loss-based transitions | Manual | Automatic |
| Hypothesis testing | Requires multiple runs | Single run |
| Resume mid-pipeline | ❌ | ✅ |

**Recommendation**: Use pipeline mode for production training, standard mode for quick tests.

### Troubleshooting Pipeline Training

**Issue**: Pipeline doesn't start / "No pipeline steps defined"
- **Solution**: Ensure `training.pipeline.steps` exists in YAML config
- **Check**: `steps` must be a list with at least one entry

**Issue**: Step checkpoint not found when resuming
- **Solution**: Pipeline automatically loads last completed step
- **Check**: Look for `flxflow_step_<name>_final.safetensors` in output directory

**Issue**: Loss-based stop condition never triggers
- **Solution**: Check `loss_name` matches actual logged loss key
- **Valid keys**: `loss_recon`, `loss_kl`, `loss_flow`, `loss_gen`, `loss_disc`
- **Check logs**: See current loss values in console output

**Issue**: Optimizer config not loading
- **Solution**: Verify JSON file path is correct and valid
- **Check**: Run `python -m json.tool <config.json>` to validate JSON

**Issue**: Models not freezing
- **Solution**: Ensure `freeze_vae`, `freeze_flow`, or `freeze_text_encoder` is set to `true` (not `True`)
- **Check logs**: Should see "Freezing <model_name>" in output

---

## Training Strategies

### Recommended 3-Stage Training

#### Stage 1: VAE Pretraining (50-100 epochs)

**Goal**: Train a high-quality autoencoder

```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/stage1_vae \
  --train_vae \
  --train_spade \
  --n_epochs 50 \
  --batch_size 2 \
  --lr 1e-5 \
  --lr_min 0.1 \
  --vae_dim 128 \
  --feature_maps_dim 128 \
  --feature_maps_dim_disc 8 \
  --lambda_adv 0.9 \
  --kl_beta 0.0001 \
  --kl_warmup_steps 5000 \
  --checkpoint_save_interval 50 \
  --gan_training \
  --workers 8
```

**Monitoring:**
- Watch `loss_recon` (reconstruction loss): Should decrease to < 0.01
- Watch `loss_kl` (KL divergence): Should stabilize around 1.0-2.0 (v0.3.0+ normalized)
- Watch `contrast_loss`: Should stabilize around 0.001-0.01
- Check sample images: Should look similar to input images

**Checkpoint**: `outputs/stage1_vae/flxflow_final.safetensors`

#### Stage 2: Flow Training (100-200 epochs)

**Goal**: Train the diffusion model using frozen VAE

```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/stage2_flow \
  --model_checkpoint outputs/stage1_vae/flxflow_final.safetensors \
  --train_diff_full \
  --n_epochs 100 \
  --batch_size 2 \
  --lr 5e-7 \
  --lr_min 0.1 \
  --vae_dim 128 \
  --feature_maps_dim 128 \
  --sample_captions \
    "a photo of a cat sitting on a couch" \
    "an illustration of mountains at sunset" \
    "abstract geometric shapes on a blue background" \
  --checkpoint_save_interval 100 \
  --workers 8
```

**Monitoring:**
- Watch `loss_flow` (flow matching loss): Should decrease to < 0.1
- Check sample images: Should match the captions

**Checkpoint**: `outputs/stage2_flow/flxflow_final.safetensors`

#### Stage 3: Joint Fine-tuning (Optional, 20-50 epochs)

**Goal**: Fine-tune both VAE and flow together

```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/stage3_joint \
  --model_checkpoint outputs/stage2_flow/flxflow_final.safetensors \
  --train_vae \
  --train_spade \
  --train_diff_full \
  --n_epochs 20 \
  --batch_size 2 \
  --lr 1e-6 \
  --lr_min 0.1 \
  --vae_dim 128 \
  --feature_maps_dim 128 \
  --feature_maps_dim_disc 8 \
  --lambda_adv 0.5 \
  --kl_beta 0.001 \
  --sample_captions \
    "a photo of a cat sitting on a couch" \
    "an illustration of mountains at sunset" \
  --checkpoint_save_interval 50 \
  --gan_training \
  --workers 8
```

**Monitoring:**
- Watch all losses: Should remain stable or slightly improve
- Check sample quality: Should be better than Stage 2

**Checkpoint**: `outputs/stage3_joint/flxflow_final.safetensors`

### Limited VRAM Strategy (8GB)

For GPUs with limited VRAM:

```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/low_vram \
  --train_vae \
  --train_spade \
  --n_epochs 100 \
  --batch_size 1 \
  --training_steps 4 \
  --lr 1e-5 \
  --vae_dim 32 \
  --feature_maps_dim 32 \
  --feature_maps_dim_disc 32 \
  --img_size 512 \
  --use_fp16 \
  --workers 4
```

**Key settings:**
- Small dimensions (32)
- Smaller image size (512)
- Batch size 1 with gradient accumulation
- FP16 mixed precision

### WebDataset Streaming

For training on streaming datasets from HuggingFace:

```bash
# Example 1: TTI-2M (default dataset with jpg images and "prompt" field)
fluxflow-train \
  --use_webdataset \
  --webdataset_token hf_your_actual_token_here \
  --webdataset_image_key jpg \
  --webdataset_caption_key prompt \
  --output_path outputs/webdataset_training \
  --train_vae \
  --train_spade \
  --n_epochs 1 \
  --batch_size 4 \
  --lr 1e-5 \
  --vae_dim 128 \
  --feature_maps_dim 128 \
  --use_fp16 \
  --workers 8

# Example 2: Custom dataset with png images and "caption" field
fluxflow-train \
  --use_webdataset \
  --webdataset_token hf_token \
  --webdataset_url "hf://datasets/nyuuzyou/textureninja/dataset_*.tar" \
  --webdataset_image_key png \
  --webdataset_caption_key caption \
  --output_path outputs/custom_dataset \
  --train_vae \
  --n_epochs 10 \
  --batch_size 2
```

**Benefits:**
- No need to download dataset
- Streams data on-the-fly
- Works with any HuggingFace WebDataset
- Configurable field mappings for different dataset formats

**Requirements:**
- HuggingFace account with dataset access
- Stable internet connection
- Token from: https://huggingface.co/settings/tokens

**Note:** The old `--use_tt2m` and `--tt2m_token` flags still work but are deprecated.

## Configuration Examples

### Example 1: Quick Test Run (5 epochs)

```bash
fluxflow-train \
  --data_path data/test_images \
  --captions_file data/test_captions.txt \
  --output_path outputs/test \
  --train_vae \
  --n_epochs 5 \
  --batch_size 2 \
  --lr 1e-5 \
  --vae_dim 32 \
  --feature_maps_dim 32 \
  --log_interval 5 \
  --checkpoint_save_interval 10
```

### Example 2: High-Quality VAE Training

```bash
fluxflow-train \
  --data_path /data/high_quality_images \
  --captions_file /data/captions.txt \
  --output_path outputs/high_quality_vae \
  --train_vae \
  --train_spade \
  --n_epochs 100 \
  --batch_size 4 \
  --lr 2e-5 \
  --lr_min 0.05 \
  --vae_dim 256 \
  --feature_maps_dim 256 \
  --feature_maps_dim_disc 16 \
  --lambda_adv 0.9 \
  --kl_beta 0.001 \
  --kl_warmup_steps 10000 \
  --use_fp16 \
  --workers 16 \
  --checkpoint_save_interval 100 \
  --gan_training
```

### Example 3: Flow Training with Custom Samples

```bash
fluxflow-train \
  --data_path /data/images \
  --captions_file /data/captions.txt \
  --output_path outputs/flow_training \
  --model_checkpoint outputs/vae/flxflow_final.safetensors \
  --train_diff_full \
  --n_epochs 200 \
  --batch_size 4 \
  --lr 1e-6 \
  --vae_dim 128 \
  --feature_maps_dim 128 \
  --sample_captions \
    "a photograph of a golden retriever in a park" \
    "an oil painting of a sailboat on the ocean" \
    "a digital illustration of a futuristic city" \
    "a watercolor painting of flowers in a vase" \
  --checkpoint_save_interval 200 \
  --use_fp16 \
  --workers 12
```

### Example 4: Using Config File

Create a config file `config.local.sh`:

```bash
#!/bin/bash

# Dataset
DATA_PATH="/data/my_images"
CAPTIONS_FILE="/data/my_captions.txt"
OUTPUT_PATH="outputs/my_training"

# Model
VAE_DIM=128
FEAT_DIM=128
FEAT_DIM_DISC=8

# Training
EPOCHS=50
BATCH_SIZE=2
LR=1e-5
LR_MIN=0.1
WORKERS=8

# Run training
fluxflow-train \
  --data_path "$DATA_PATH" \
  --captions_file "$CAPTIONS_FILE" \
  --output_path "$OUTPUT_PATH" \
  --train_vae \
  --train_spade \
  --n_epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --lr_min $LR_MIN \
  --vae_dim $VAE_DIM \
  --feature_maps_dim $FEAT_DIM \
  --feature_maps_dim_disc $FEAT_DIM_DISC \
  --workers $WORKERS \
  --use_fp16
```

Run with: `bash config.local.sh`

## Monitoring Training Progress

### Training Diagrams

Training metrics are automatically logged to `OUTPUT_FOLDER/graph/training_metrics.jsonl`.

**Automatic diagram generation** (on each checkpoint save):
```bash
python train.py --generate_diagrams ...
```

**Manual diagram generation**:
```bash
python src/fluxflow_training/scripts/generate_training_graphs.py outputs/
```

Generated diagrams in `outputs/graph/`:
- `training_losses.png` - VAE, Flow, Discriminator, Generator, LPIPS losses
- `kl_loss.png` - KL divergence with beta warmup schedule
- `learning_rates.png` - Learning rate schedules over time
- `batch_times.png` - Training speed (seconds/batch)
- `training_overview.png` - Combined overview with all metrics
- `training_summary.txt` - Statistical summary of training session

## Troubleshooting

### Out of Memory (OOM) Errors

**Solutions:**
1. Reduce batch size: `--batch_size 1`
2. Reduce dimensions: `--vae_dim 32 --feature_maps_dim 32`
3. Reduce image size: `--img_size 512`
4. Enable FP16: `--use_fp16`
5. Reduce workers: `--workers 1`
6. Use gradient accumulation: `--batch_size 1 --training_steps 4`

### NaN Losses

**Causes & Solutions:**
- **High learning rate**: Reduce `--lr` by 10x
- **Unstable GAN**: Reduce `--lambda_adv` or disable `--gan_training`
- **High KL beta**: Reduce `--kl_beta` or increase `--kl_warmup_steps`
- **Gradient explosion**: Reduce `--initial_clipping_norm` to 0.5

**Example fix:**
```bash
--lr 1e-6 --lambda_adv 0.3 --initial_clipping_norm 0.5 --kl_beta 0.0001
```

### Mode Collapse (GAN)

**Symptoms**: Generated images all look similar

**Solutions:**
1. Increase discriminator capacity: `--feature_maps_dim_disc 16`
2. Reduce GAN weight: `--lambda_adv 0.3`
3. Train without GAN first (omit `--gan_training`)

### Poor Image Quality

**Solutions:**
1. Train VAE longer: `--n_epochs 100`
2. Increase dimensions: `--vae_dim 256`
3. Increase GAN weight: `--lambda_adv 0.9`
4. Reduce KL beta: `--kl_beta 0.0001`
5. Check sample images during training

### Slow Training Speed

**Solutions:**
1. Enable FP16: `--use_fp16`
2. Increase workers: `--workers 16`
3. Increase batch size: `--batch_size 4`
4. Reduce sample frequency: `--checkpoint_save_interval 200` or `--samples_per_checkpoint 5`
5. Use TTI-2M streaming: `--use_tt2m`

### Text Conditioning Not Working

**Solutions:**
1. Check captions file format (tab-separated)
2. Verify tokenizer: `--tokenizer_name "distilbert-base-uncased"`
3. Train flow model longer: `--n_epochs 200`
4. Increase text embedding: `--text_embedding_dim 1024`

## Performance Benchmarks

### Training Times (NVIDIA RTX 3090, 24GB)

Approximate training times per epoch (10k images):

| Configuration | Batch Size | Time/Epoch (10k images) |
|--------------|-----------|------------------------|
| VAE only | 4 | ~30 min |
| VAE + GAN (with random packets) | 4 | ~50 min |
| Flow only | 2 | ~60 min |
| VAE + Flow | 2 | ~90 min |

Generation: ~2-5 seconds per image (512x512, 50 steps)

### Per-Batch Performance

#### RTX 3090 (24GB VRAM)

| Configuration | Batch Size | Speed | VRAM Usage |
|---------------|------------|-------|------------|
| VAE (dim=32) | 8 | ~2 sec/batch | ~8GB |
| VAE (dim=64) | 4 | ~3 sec/batch | ~12GB |
| VAE (dim=128) | 2 | ~5 sec/batch | ~18GB |
| VAE (dim=256) | 1 | ~10 sec/batch | ~22GB |
| Flow (dim=128) | 2 | ~8 sec/batch | ~20GB |

*With `--use_fp16`, speeds improve by ~40%*

#### RTX 4090 (24GB VRAM)

| Configuration | Batch Size | Speed | VRAM Usage |
|---------------|------------|-------|------------|
| VAE (dim=128, FP16) | 4 | ~2 sec/batch | ~16GB |
| VAE (dim=256, FP16) | 2 | ~4 sec/batch | ~22GB |
| Flow (dim=128, FP16) | 4 | ~4 sec/batch | ~20GB |

## Additional Resources

- **Example Config**: `config.example.sh`
- **UI Guide**: Use the web UI at `http://localhost:7860` for visual training configuration
- **Issues**: Report bugs at https://github.com/danny-mio/fluxflow-training/issues

## Summary

This guide covers all aspects of training FluxFlow models. Key takeaways:

1. **Start with VAE training** (Stage 1) for 50-100 epochs
2. **Then train the flow model** (Stage 2) for 100-200 epochs
3. **Optionally fine-tune jointly** (Stage 3) for 20-50 epochs
4. **Monitor losses and sample images** throughout training
5. **Adjust hyperparameters** based on your GPU and dataset
6. **Use the UI** for easier configuration and live monitoring

For questions, refer to the troubleshooting section or check the existing documentation.
