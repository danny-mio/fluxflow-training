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

## Quick Start

### 1. Prepare Your Data

**Option A: Local Dataset**
```bash
# Your directory structure should be:
# /path/to/images/
#   ├── image1.jpg
#   ├── image2.png
#   └── ...
# /path/to/captions.txt (tab-separated: filename\tcaption)

# Example captions.txt:
# image1.jpg	a photo of a cat sitting on a couch
# image2.png	an illustration of mountains at sunset
```

**Option B: TTI-2M Streaming Dataset**
```bash
# Use the TTI-2M dataset from HuggingFace
# Requires: HuggingFace token with access to the dataset
```

### 2. Basic Training Commands

**Train VAE (Stage 1 - Recommended First)**
```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/flux \
  --train_vae \
  --train_spade \
  --n_epochs 50 \
  --batch_size 2 \
  --lr 1e-5 \
  --vae_dim 128 \
  --feature_maps_dim 128 \
  --feature_maps_dim_disc 8
```

**Train Flow Model (Stage 2)**
```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/flux \
  --model_checkpoint outputs/flux/flxflow_final.safetensors \
  --train_diff_full \
  --n_epochs 100 \
  --batch_size 2 \
  --lr 5e-7 \
  --vae_dim 128 \
  --feature_maps_dim 128
```

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

### Optimizer Parameters Reference

#### Lion Optimizer

Memory-efficient optimizer that uses sign-based updates. Recommended for flow models.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "Lion" |
| `lr` | float | 1e-4 | Learning rate |
| `betas` | list[float, float] | [0.9, 0.99] | Coefficients for computing running averages |
| `weight_decay` | float | 0.0 | Weight decay coefficient |
| `decoupled_weight_decay` | bool | True | Use decoupled weight decay (recommended) |

**Example:**
```json
{
  "type": "Lion",
  "lr": 5e-7,
  "betas": [0.9, 0.95],
  "weight_decay": 0.01,
  "decoupled_weight_decay": true
}
```

**Best for:** Flow models, memory-constrained training
**Notes:** Uses less memory than Adam, often converges faster

#### AdamW Optimizer

Adam optimizer with decoupled weight decay. Excellent all-around optimizer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "AdamW" |
| `lr` | float | 1e-3 | Learning rate |
| `betas` | list[float, float] | [0.9, 0.999] | Coefficients for computing running averages |
| `weight_decay` | float | 0.0 | Weight decay coefficient (L2 penalty) |
| `eps` | float | 1e-8 | Term added to denominator for numerical stability |
| `amsgrad` | bool | False | Use AMSGrad variant for better convergence |

**Example:**
```json
{
  "type": "AdamW",
  "lr": 1e-5,
  "betas": [0.9, 0.95],
  "weight_decay": 0.01,
  "eps": 1e-8,
  "amsgrad": false
}
```

**Example with AMSGrad (for discriminator):**
```json
{
  "type": "AdamW",
  "lr": 5e-7,
  "betas": [0.0, 0.9],
  "weight_decay": 0.001,
  "amsgrad": true
}
```

**Best for:** VAE, text encoder, discriminator
**Notes:** More stable than Adam, handles weight decay correctly

#### Adam Optimizer

Standard Adam optimizer. Good baseline choice.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "Adam" |
| `lr` | float | 1e-3 | Learning rate |
| `betas` | list[float, float] | [0.9, 0.999] | Coefficients for computing running averages |
| `weight_decay` | float | 0.0 | Weight decay coefficient |
| `eps` | float | 1e-8 | Term added to denominator for numerical stability |

**Example:**
```json
{
  "type": "Adam",
  "lr": 1e-4,
  "betas": [0.9, 0.999],
  "weight_decay": 0.0,
  "eps": 1e-8
}
```

**Best for:** General purpose, quick experimentation
**Notes:** Prefer AdamW for better weight decay handling

#### SGD Optimizer

Stochastic gradient descent with momentum and Nesterov acceleration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "SGD" |
| `lr` | float | Required | Learning rate |
| `momentum` | float | 0.0 | Momentum factor (typically 0.9) |
| `weight_decay` | float | 0.0 | Weight decay coefficient |
| `dampening` | float | 0.0 | Dampening for momentum |
| `nesterov` | bool | False | Enable Nesterov momentum |

**Example:**
```json
{
  "type": "SGD",
  "lr": 0.01,
  "momentum": 0.9,
  "weight_decay": 1e-4,
  "nesterov": true
}
```

**Example (simple SGD without momentum):**
```json
{
  "type": "SGD",
  "lr": 0.001,
  "momentum": 0.0,
  "weight_decay": 0.0
}
```

**Best for:** Fine-tuning, transfer learning, some discriminator training
**Notes:** Requires careful learning rate tuning, benefits from momentum

#### RMSprop Optimizer

Root mean square propagation optimizer. Adapts learning rates per parameter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "RMSprop" |
| `lr` | float | 1e-2 | Learning rate |
| `alpha` | float | 0.99 | Smoothing constant |
| `eps` | float | 1e-8 | Term added to denominator for numerical stability |
| `weight_decay` | float | 0.0 | Weight decay coefficient |
| `momentum` | float | 0.0 | Momentum factor |
| `centered` | bool | False | Compute centered RMSprop |

**Example:**
```json
{
  "type": "RMSprop",
  "lr": 1e-4,
  "alpha": 0.99,
  "eps": 1e-8,
  "weight_decay": 0.0,
  "momentum": 0.0,
  "centered": false
}
```

**Example with momentum:**
```json
{
  "type": "RMSprop",
  "lr": 1e-3,
  "alpha": 0.95,
  "momentum": 0.9,
  "centered": true
}
```

**Best for:** RNNs, non-stationary objectives
**Notes:** Less commonly used for image generation, try Adam/Lion first

---

### Scheduler Parameters Reference

#### CosineAnnealingLR Scheduler

Cosine annealing learning rate schedule. Smoothly decreases LR following cosine curve.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "CosineAnnealingLR" |
| `eta_min_factor` | float | 0.1 | Minimum LR as fraction of initial LR |

**How it works:**
- LR starts at initial value
- Decreases following cosine curve over total training steps
- Minimum LR = initial_lr × eta_min_factor
- Smooth, gradual decay without sharp drops

**Example (standard):**
```json
{
  "type": "CosineAnnealingLR",
  "eta_min_factor": 0.1
}
```
*LR decays from initial to 10% of initial (e.g., 1e-5 → 1e-6)*

**Example (aggressive decay):**
```json
{
  "type": "CosineAnnealingLR",
  "eta_min_factor": 0.001
}
```
*LR decays from initial to 0.1% of initial (e.g., 1e-5 → 1e-8)*

**Example (minimal decay):**
```json
{
  "type": "CosineAnnealingLR",
  "eta_min_factor": 0.5
}
```
*LR decays from initial to 50% of initial (e.g., 1e-5 → 5e-6)*

**Best for:** Most training scenarios (default, recommended)
**Notes:** Smooth decay prevents training instability

#### LinearLR Scheduler

Linear learning rate decay from start_factor to end_factor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "LinearLR" |
| `start_factor` | float | 1.0 | Starting LR multiplier |
| `end_factor` | float | 0.1 | Ending LR multiplier |
| `total_iters` | int | auto | Number of steps for decay (defaults to total training steps) |

**How it works:**
- LR starts at initial_lr × start_factor
- Linearly decreases to initial_lr × end_factor
- Decay completes at total_iters steps

**Example (warmup then decay):**
```json
{
  "type": "LinearLR",
  "start_factor": 0.1,
  "end_factor": 1.0,
  "total_iters": 5000
}
```
*LR increases from 10% to 100% over 5000 steps (warmup)*

**Example (linear decay):**
```json
{
  "type": "LinearLR",
  "start_factor": 1.0,
  "end_factor": 0.0,
  "total_iters": 50000
}
```
*LR decreases from 100% to 0% over 50000 steps*

**Example (partial decay):**
```json
{
  "type": "LinearLR",
  "start_factor": 1.0,
  "end_factor": 0.25
}
```
*LR decreases from 100% to 25% over entire training*

**Best for:** Warmup schedules, simple linear decay
**Notes:** Less common than cosine, but useful for warmup

#### ExponentialLR Scheduler

Exponential learning rate decay. LR multiplied by gamma each step.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "ExponentialLR" |
| `gamma` | float | 0.95 | Multiplicative factor of LR decay |

**How it works:**
- Each step: new_lr = current_lr × gamma
- Exponential decay (fast initially, slower later)
- After N steps: lr = initial_lr × (gamma^N)

**Example (slow decay):**
```json
{
  "type": "ExponentialLR",
  "gamma": 0.9999
}
```
*Very gradual decay, LR halves after ~7000 steps*

**Example (medium decay):**
```json
{
  "type": "ExponentialLR",
  "gamma": 0.999
}
```
*Moderate decay, LR halves after ~700 steps*

**Example (fast decay):**
```json
{
  "type": "ExponentialLR",
  "gamma": 0.95
}
```
*Aggressive decay, LR halves after ~14 steps*

**Best for:** Fine-tuning, when you want faster initial decay
**Notes:** Gamma close to 1.0 = slow decay, far from 1.0 = fast decay

#### ConstantLR Scheduler

Constant learning rate with optional initial scaling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "ConstantLR" |
| `factor` | float | 1.0 | LR multiplication factor |
| `total_iters` | int | auto | Number of steps to apply factor (then returns to 1.0) |

**How it works:**
- For first total_iters steps: lr = initial_lr × factor
- After total_iters steps: lr = initial_lr × 1.0
- Useful for warmup with constant LR period

**Example (constant at 100%):**
```json
{
  "type": "ConstantLR",
  "factor": 1.0
}
```
*LR stays constant at initial value*

**Example (reduced constant LR):**
```json
{
  "type": "ConstantLR",
  "factor": 0.1,
  "total_iters": 10000
}
```
*LR is 10% of initial for first 10k steps, then jumps to 100%*

**Example (warmup):**
```json
{
  "type": "ConstantLR",
  "factor": 0.01,
  "total_iters": 1000
}
```
*LR is 1% of initial for first 1k steps (warmup), then jumps to 100%*

**Best for:** No LR scheduling, warmup periods
**Notes:** Simple but less flexible than other schedulers

#### StepLR Scheduler

Step-wise learning rate decay. Multiply LR by gamma every step_size steps.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "StepLR" |
| `step_size` | int | auto | Number of steps between LR decay |
| `gamma` | float | 0.1 | Multiplicative factor of LR decay |

**How it works:**
- Every step_size steps: lr = lr × gamma
- Piecewise constant LR with periodic drops
- Creates "staircase" LR schedule

**Example (decay every 10k steps):**
```json
{
  "type": "StepLR",
  "step_size": 10000,
  "gamma": 0.5
}
```
*Halve LR every 10,000 steps*

**Example (aggressive stepping):**
```json
{
  "type": "StepLR",
  "step_size": 5000,
  "gamma": 0.1
}
```
*Reduce LR to 10% every 5,000 steps*

**Example (gentle stepping):**
```json
{
  "type": "StepLR",
  "step_size": 20000,
  "gamma": 0.8
}
```
*Reduce LR to 80% every 20,000 steps*

**Best for:** Training with known plateaus, milestone-based decay
**Notes:** Can cause training instability at step boundaries

#### ReduceLROnPlateau Scheduler

Reduce learning rate when a metric plateaus. Requires metric monitoring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "ReduceLROnPlateau" |
| `mode` | str | "min" | "min" (lower is better) or "max" (higher is better) |
| `factor` | float | 0.1 | Factor by which LR is reduced: new_lr = lr × factor |
| `patience` | int | 10 | Number of steps with no improvement before reducing LR |
| `threshold` | float | 1e-4 | Threshold for measuring improvement |

**How it works:**
- Monitors validation metric (loss, accuracy, etc.)
- If no improvement for `patience` steps, reduce LR by `factor`
- Automatically adapts to training progress

**Example (reduce on loss plateau):**
```json
{
  "type": "ReduceLROnPlateau",
  "mode": "min",
  "factor": 0.5,
  "patience": 10,
  "threshold": 1e-4
}
```
*Halve LR if loss doesn't improve by 0.0001 for 10 steps*

**Example (reduce on metric plateau):**
```json
{
  "type": "ReduceLROnPlateau",
  "mode": "max",
  "factor": 0.1,
  "patience": 5,
  "threshold": 0.001
}
```
*Reduce LR to 10% if metric doesn't improve by 0.001 for 5 steps*

**Example (patient reduction):**
```json
{
  "type": "ReduceLROnPlateau",
  "mode": "min",
  "factor": 0.75,
  "patience": 20,
  "threshold": 1e-5
}
```
*Reduce LR to 75% if no improvement for 20 steps*

**Best for:** Validation metric-based training, uncertain convergence
**Notes:** Requires external metric tracking, not commonly used in standard training script

---

### Complete Configuration Examples

**Example 1: VAE Training (High Quality)**
```json
{
  "optimizers": {
    "vae": {
      "type": "AdamW",
      "lr": 2e-5,
      "betas": [0.9, 0.95],
      "weight_decay": 0.01,
      "eps": 1e-8
    },
    "discriminator": {
      "type": "AdamW",
      "lr": 2e-5,
      "betas": [0.0, 0.9],
      "weight_decay": 0.001,
      "amsgrad": true
    }
  },
  "schedulers": {
    "vae": {
      "type": "CosineAnnealingLR",
      "eta_min_factor": 0.1
    },
    "discriminator": {
      "type": "CosineAnnealingLR",
      "eta_min_factor": 0.1
    }
  }
}
```

**Example 2: Flow Training (Memory Efficient)**
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
    }
  }
}
```

**Example 3: Joint Training (Advanced)**
```json
{
  "optimizers": {
    "flow": {
      "type": "Lion",
      "lr": 1e-6,
      "betas": [0.9, 0.95],
      "weight_decay": 0.01,
      "decoupled_weight_decay": true
    },
    "vae": {
      "type": "AdamW",
      "lr": 5e-6,
      "betas": [0.9, 0.95],
      "weight_decay": 0.01
    },
    "text_encoder": {
      "type": "AdamW",
      "lr": 1e-7,
      "betas": [0.9, 0.99],
      "weight_decay": 0.01
    },
    "discriminator": {
      "type": "AdamW",
      "lr": 5e-6,
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

**Example 4: Experimental (SGD with Step Decay)**
```json
{
  "optimizers": {
    "vae": {
      "type": "SGD",
      "lr": 0.01,
      "momentum": 0.9,
      "weight_decay": 1e-4,
      "nesterov": true
    }
  },
  "schedulers": {
    "vae": {
      "type": "StepLR",
      "step_size": 10000,
      "gamma": 0.5
    }
  }
}
```

---

**Basic Usage:**
```bash
# Create config file
cat > optim_config.json << EOF
{
  "optimizers": {
    "flow": {"type": "Lion", "lr": 5e-7, "weight_decay": 0.01},
    "vae": {"type": "AdamW", "lr": 1e-5, "weight_decay": 0.01}
  },
  "schedulers": {
    "flow": {"type": "CosineAnnealingLR", "eta_min_factor": 0.1},
    "vae": {"type": "CosineAnnealingLR", "eta_min_factor": 0.1}
  }
}
EOF

# Use in training
fluxflow-train \
  --optim_sched_config optim_config.json \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --train_vae \
  --n_epochs 50
```

**Example:**
```bash
# VAE training with higher learning rate
--n_epochs 50 --batch_size 2 --lr 1e-5 --workers 8

# Flow training with gradient accumulation
--n_epochs 100 --lr 5e-7 --training_steps 4 --use_fp16

# Resume with preserved learning rate
--model_checkpoint checkpoint.safetensors --preserve_lr
```

#### Training Modes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--train_vae` | flag | False | Train VAE (compressor + expander) |
| `--gan_training` | flag | False | Enable GAN/discriminator training for VAE |
| `--train_spade` | flag | False | Use SPADE spatial conditioning (better quality) |
| `--train_diff` | flag | False | Train flow model with partial schedule |
| `--train_diff_full` | flag | False | Train flow model with full schedule (recommended) |

**Training Mode Combinations:**

| Mode | Command | Use Case |
|------|---------|----------|
| VAE Only (with GAN) | `--train_vae --gan_training --train_spade` | Stage 1: Train autoencoder |
| VAE Only (no GAN) | `--train_vae` | Fast VAE training, lower quality |
| Flow Only | `--train_diff_full` | Stage 2: Train diffusion model |
| Joint Training | `--train_vae --gan_training --train_diff_full --train_spade` | Advanced: Train both simultaneously |

**Example:**
```bash
# Stage 1: Train VAE with SPADE and GAN
--train_vae --train_spade

# Stage 2: Train flow model only
--train_diff_full

# Advanced: Joint training
--train_vae --train_diff_full --train_spade
```

#### KL Divergence

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--kl_beta` | float | 0.0001 | Final KL divergence weight (regularization) |
| `--kl_warmup_steps` | int | 5000 | Steps to linearly warm up KL weight from 0 to kl_beta |
| `--kl_free_bits` | float | 0.0 | Free bits (nats) - minimum KL before penalty applies |

**KL Divergence Tuning:**
- **Low KL Beta (0.0001)**: More detail, less regularization, potential overfitting
- **Medium KL Beta (0.001-0.01)**: Balanced detail and regularization
- **High KL Beta (0.1-1.0)**: Strong regularization, smoother latents, less detail
- **Free Bits**: Allows some KL divergence without penalty (e.g., 0.5 nats)

**Example:**
```bash
# Low regularization for maximum detail
--kl_beta 0.0001 --kl_warmup_steps 5000

# Balanced regularization
--kl_beta 0.01 --kl_warmup_steps 10000 --kl_free_bits 0.5

# Strong regularization for smooth latents
--kl_beta 1.0 --kl_warmup_steps 20000
```

#### Output & Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output_path` | str | "outputs" | Directory for saving checkpoints and samples |
| `--log_interval` | int | 10 | Print training metrics every N batches |
| `--checkpoint_save_interval` | int | 50 | Save checkpoints every N batches |
| `--samples_per_checkpoint` | int | 1 | Generate samples every N checkpoint saves |
| `--no_samples` | flag | False | Disable sample generation during training |
| `--test_image_address` | list | [] | List of test images for VAE reconstruction samples |
| `--sample_captions` | list | ["A sample caption"] | Captions for generating flow model samples |

**Example:**
```bash
# Standard logging and checkpointing
--output_path outputs/flux --log_interval 10 --checkpoint_save_interval 50

# Less frequent samples (every 5 checkpoints = every 250 batches)
--checkpoint_save_interval 50 --samples_per_checkpoint 5

# VAE reconstruction testing
--test_image_address test1.jpg test2.png --checkpoint_save_interval 100

# Disable samples to speed up training
--no_samples
```

#### Miscellaneous

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--tokenizer_name` | str | "distilbert-base-uncased" | HuggingFace tokenizer for text encoding |
| `--img_size` | int | 1024 | Target image size (images resized to this) |
| `--channels` | int | 3 | Number of image channels (3 for RGB) |
| `--lambda_adv` | float | 0.5 | Adversarial (GAN) loss weight |

**Example:**
```bash
# Use different tokenizer
--tokenizer_name "bert-base-uncased"

# Train on smaller images
--img_size 512

# Adjust GAN loss weight
--lambda_adv 0.9
```

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
- Watch `loss_kl` (KL divergence): Should stabilize around 10-50
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
- **Issues**: Report bugs at https://github.com/danny-mio/fluxflow/issues

## Summary

This guide covers all aspects of training FluxFlow models. Key takeaways:

1. **Start with VAE training** (Stage 1) for 50-100 epochs
2. **Then train the flow model** (Stage 2) for 100-200 epochs
3. **Optionally fine-tune jointly** (Stage 3) for 20-50 epochs
4. **Monitor losses and sample images** throughout training
5. **Adjust hyperparameters** based on your GPU and dataset
6. **Use the UI** for easier configuration and live monitoring

For questions, refer to the troubleshooting section or check the existing documentation.
