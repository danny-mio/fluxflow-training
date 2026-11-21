# FluxFlow Training Guide

A comprehensive guide to configuring and running training for FluxFlow text-to-image models.

## Table of Contents

- [Training workflow (v0.10.0)](#training-workflow-v0100)
- [v0.10.0 Loss Schedule](#v0100-loss-schedule)
- [Memory Requirements & OOM Prevention](#memory-requirements--oom-prevention)
- [Overview](#overview)
- [Configuration Methods](#configuration-methods)
- [Quick Start](#quick-start)
- [Training Command Reference](#training-command-reference)
- [Parameter Details](#parameter-details)
- [Pipeline Training Mode](#pipeline-training-mode)
- [Training Strategies](#training-strategies)
- [Configuration Examples](#configuration-examples)
- [Monitoring Training Progress](#monitoring-training-progress)
- [Troubleshooting](#troubleshooting)
- [Performance Benchmarks](#performance-benchmarks)
- [Additional Resources](#additional-resources)

## Training workflow (v0.10.0)

Train the VAE, calibrate its latent statistics, then train the flow.

v0.10.0 models are built from the YAML `model:` block (`model_type` set,
`model_version: "0.10.0"`). Without `model_type` (e.g. CLI-only runs) the trainer
builds the legacy v0.6.0 VAE.

**Pre-rework checkpoints are invalid.** Pre-rework v0.10.0 VAE checkpoints raise
`IncompatibleCheckpointError` (affine `ctx_final_norm`, or stage widths that differ
from the model config); flow checkpoints trained on pre-rework latents are invalid.
Retrain the VAE, calibrate, then retrain the flow.

1. **Train the VAE** (`train_vae` steps). VAE steps keep an EMA (`use_ema`, default
   `true`, decay `0.999`); samples and exported models use the EMA weights. EMA
   warmup: effective decay `min(decay, (1+n)/(10+n))` at update `n` (flow EMA:
   decay `0.9999`, same warmup). To reuse a trained discriminator without
   `model_checkpoint`, set top-level `discriminator_checkpoint` (or
   `--discriminator_checkpoint`): a `D_img` `.safetensors` (`D_img.` prefix
   optional) that overrides the checkpoint's D_img; a missing file raises, an
   incompatible one reinitialises D with a warning.
2. **Calibrate.** The flow's noise schedule assumes ~N(0, I) per channel. Calibration
   encodes training images, measures each latent channel's mean and std, and stores
   them as `latent_shift` / `latent_scale` on both compressor and expander. Encoding
   then normalizes and decoding denormalizes, so reconstruction is unchanged.
   - **In a pipeline**: at the transition to a VAE-frozen flow step, the VAE EMA
     parameters (if the VAE step kept an EMA) replace the live VAE weights; buffers
     (latent stats) stay live. The flow step therefore trains, calibrates and
     exports on the EMA VAE. If the VAE is
     then uncalibrated, the `train_diff` / `train_diff_full` step encodes
     `calibration_images` images (default `2000`), logs a per-channel table and
     exports the model checkpoint. Resume repeats the check.
   - **Standalone** (e.g. the VAE was trained elsewhere):

     ```bash
     fluxflow-calibrate-latents --checkpoint outputs/run/flxflow_final.safetensors \
         --config config.yaml --out outputs/run/flxflow_calibrated.safetensors
     ```

     It uses the config's `default_dataset` (or its first dataset) at its training
     resolutions. Options: `--num-images` (default `2000`), `--batch-size` (default
     `8`), `--force` (allow `--out` to equal `--checkpoint`).
3. **Train the flow** from the calibrated checkpoint. `FlowTrainer` raises
   `ValueError` on an uncalibrated v0.10.0 VAE, so set `calibration_images: 0` only
   when the checkpoint is already calibrated.

Automatic calibration only runs on an uncalibrated VAE. After further VAE training,
recalibrate with `fluxflow-calibrate-latents` and retrain the flow: flow checkpoints are
tied to the latent statistics they were trained on.

## v0.10.0 Loss Schedule

The v0.10.0 redesign restructures the VAE and flow losses around the
bezier-coupled architecture. The relevant schedules live in
`training/losses.py` and are wired into `VAETrainer` and `FlowTrainer`.

### KL warmup on the clean Gaussian latent

KL is annealed from `0` to `kl_z_weight` (pipeline default `1e-5`) with a cosine ramp over
`kl_z_warmup_steps` OPTIMIZER steps (x `gradient_accumulation_steps` micro-batches; `losses.cosine_warmup_weight`). `VAE` / `vae_loss` is the HF-frequency-weighted recon over the whole batch, including the z-only half (`ctx_dropout_frac`), and rises while `hf_ramp_steps` ramps alpha 0.3 to 1.2; read `L1ctx` (`recon_l1_ctx`) for the plain ctx-path L1. Legacy
`kl_beta` / `kl_warmup_steps` keys still load and emit a
`DeprecationWarning`; when both forms appear, the v0.10.0 form wins.

Free bits are per channel: each z channel's KL (mean over batch and spatial dims) is
floored at `kl_free_bits` nats/element (pipeline default `0.5`, `0` disables); a
channel below the floor gets no KL gradient.

### z-vs-ctx balance

The deterministic ctx half can let the decoder ignore z. Two training-only
knobs (v0.10.0 VAE; 0 disables each): `ctx_dropout_frac` (default 0.5) decodes
that fraction of each batch with `use_context=False` (z-only; GAN, latent-cycle
targets and the LR-plateau metric use the ctx-decoded half only), and
`ctx_noise_sigma` (default 0.5) adds `sigma * std(token)` noise to ctx before
decoding, so shrinking ctx cannot escape it (capacity ≈ 0.5·log2(1+1/σ²)
bits/dim). Logged as `L1ctx` / `L1z`; a large `L1z` gap means z carries little.

### CFG empty-prompt substitution

`FlowTrainer` routes CFG dropout through
`cfg_utils.apply_cfg_null_substitution`. The function caches the
`(null_seq, null_mask)` returned by
`fluxflow.utils.visualization.build_cfg_null_pair` on the text encoder, so
training and sampling see the *same* unconditional distribution. Set
`null_prompt` if you need a non-empty default.

### Minimal v0.10.0 YAML snippet

```yaml
training:
  pipeline:
    steps:
      - name: vae_warmup
        train_vae: true
        kl_z_weight: 1e-5
        kl_z_warmup_steps: 10000
        kl_free_bits: 0.5              # per-channel, nats/element
        lambda_ctx_aux: 0.01           # aux ctx vs z_tokens (VAE only)
        train_latent_cycle: false      # G.E idempotence; see config.example.yaml latent_cycle_*
        # logs: latent_cycle_loss (unweighted), latent_cycle_reject_frac

      - name: flow_cfg
        train_diff: true
        calibration_images: 2000       # auto-calibrate latent stats if uncalibrated
        null_prompt: ""                # cached empty CFG null context
        cfg_dropout_prob: 0.10
        ctx_loss_weight: 0.5           # ctx-dim vs VAE-dim v-prediction loss
        freeze:
          - compressor
          - expander
          - text_encoder_backbone      # keep DistilBERT frozen
        # text_encoder_projection stays trainable so the projection head can adapt
```

Datasets accept a matching `max_text_length` (default `128`); see
`data/datasets.py`.

---

## Memory Requirements & OOM Prevention

All figures in this section were measured in December 2025 on the pre-rework
(narrow) VAE; v0.10.0 wide-VAE numbers will differ.

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
```

**5. Reduce Widths**:
```yaml
model:
  vae_encoder_channels: [32, 64, 128, 128]   # default [32, 64, 128, 256]; lighter fallback (~50M params)
  feature_maps_dim: 64
  feature_maps_dim_disc: 8
```
v0.10.0 VAE widths are set by `vae_encoder_channels` / `vae_decoder_channels` /
`vae_spade_hidden`, not `vae_dim` (see [Understanding VAE Dimensions](#understanding-vae-dimensions)).

**6. Use mixed precision** (pre-rework estimate: saves ~20-30%):
```yaml
use_fp16: true
precision: bf16  # recommended for v0.10.0; default "fp16" is accepted
```

> SPADE is always-on in v0.10.0; the `train_spade` field is deprecated and ignored by the parser, so disabling it is no longer a memory lever.

### Hardware Recommendations

Pre-rework (narrow VAE); `dim` is `vae_dim`, which set the VAE widths then.
v0.10.0 numbers will differ.

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

FluxFlow uses a unified training script (`src/fluxflow_training/scripts/train.py`) that supports:
- **VAE Training**: Train the autoencoder (compressor + expander)
- **Flow Model Training**: Train the diffusion model for text-to-image generation
- **Joint Training**: Train both VAE and flow simultaneously (advanced)

The training process is highly configurable with parameters for data, model architecture, training behavior, and output.

## Configuration Methods

FluxFlow supports two configuration approaches:

### YAML Config Files (Recommended for Production)

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
        optimization:
          optimizers:
            vae:
              type: "AdamW"
              lr: 0.0001
              weight_decay: 0.01
          schedulers:
            vae:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1

output:
  output_path: "outputs/vae"
```

> SPADE is always-on in v0.10.0; the `train_spade` field is deprecated and ignored by the parser.

**Run:** `fluxflow-train --config config.yaml`

---

### CLI Arguments (For Quick Tests Only)

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
- No multi-step pipelines
- No per-step optimizer customization
- No inline YAML optimizer config
- Hard to version-control
- Complex commands become unwieldy

---

### Feature Comparison

YAML configs are the only supported path for multi-step pipelines, inline per-step optimizer/scheduler configuration, per-step freezing, and loss-based transitions. CLI args remain useful for one-off experiments but cannot express any of those, and version-controlling a long shell command is awkward in practice.

**Recommendation:**
- **Beginners / quick tests:** start with CLI args (see Quick Start below).
- **Production / serious training:** use a YAML config in pipeline mode (see "Pipeline Training Mode" below).

---

## Quick Start

Choose your path based on your needs:

### Path A: CLI Quick Test (5 Minutes)

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
  --vae_dim 64
```

When you're ready for serious training, switch to Path B (YAML Config).

---

### Path B: YAML Config (Production-Ready)

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
        gan_training: true

        optimization:
          optimizers:
            vae:
              type: "AdamW"
              lr: 0.00001  # Same as 1e-5
              betas: [0.9, 0.999]
              weight_decay: 0.01
            discriminator:
              type: "AdamW"
              lr: 0.00001
              betas: [0.0, 0.9]
              amsgrad: true

          schedulers:
            vae:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1
            discriminator:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1

output:
  output_path: "outputs/production"
```

**Step 3: Run Training**
```bash
fluxflow-train --config config.yaml
```

YAML configs are version-controllable, support multi-step pipelines with per-step optimizer customization, and avoid long CLI commands. See "Pipeline Training Mode" below for multi-stage training (VAE → Flow → Fine-tune).

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
--use_webdataset --webdataset_token YOUR_HF_TOKEN

# Custom WebDataset with specific field mappings
--use_webdataset --webdataset_token hf_token \
  --webdataset_url "hf://datasets/user/dataset/*.tar" \
  --webdataset_image_key "png" \
  --webdataset_caption_key "caption"

# Legacy (still works but deprecated):
# --use_tt2m --tt2m_token YOUR_HF_TOKEN
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
| `--discriminator_checkpoint` | str | None | `D_img` `.safetensors` to load (`D_img.` prefix optional); overrides `model_checkpoint`'s D_img. YAML: top-level `discriminator_checkpoint` |
| `--vae_dim` | int | 128 | Latent channels per token (v0.10.0: not a width lever, see below) |
| `--text_embedding_dim` | int | 1024 | Text embedding dimension from BERT |
| `--feature_maps_dim` | int | 128 | Flow model feature dimension |
| `--feature_maps_dim_disc` | int | 8 | Discriminator feature dimension |
| `--pretrained_bert_model` | str | - | Path to pretrained BERT checkpoint (optional) |

**Dimension Guidelines:** VAE widths: see [Understanding VAE Dimensions](#understanding-vae-dimensions).
`feature_maps_dim` / `feature_maps_dim_disc` set flow and discriminator capacity.

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
| `--use_fp16` | flag | False | Use mixed precision training |
| `--precision` | str | `fp16` | Mixed-precision dtype when `--use_fp16` is set: `fp16` or `bf16`. `bf16` recommended for v0.10.0 (fp16 can overflow, ~65504 max, now that Padé input clamps are removed); fp16 remains the default and is accepted (see [Mixed Precision Training](#mixed-precision-training)). |
| `--initial_clipping_norm` | float | 1.0 | Gradient clipping norm for stability |
| `--compile_activations` | flag | False | `torch.compile` the Padé activation math (`activation_type: pade`), fusing its elementwise ops into one kernel. YAML: `training.compile_activations`. One-time compile cost at startup. |

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
      "decoupled_weight_decay": false
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

FluxFlow supports per-model optimizer and scheduler configuration via the inline `optimization:` block on each pipeline step.

**Supported Optimizers:** AdamW, Adam, SGD, RMSprop, Lion

**Supported Schedulers:** CosineAnnealingLR, LinearLR, StepLR, MultiStepLR, ExponentialLR, ConstantLR, ReduceLROnPlateau

**Detailed References:**
- **[OPTIMIZERS.md](OPTIMIZERS.md)** - Complete optimizer parameter reference with examples
- **[SCHEDULERS.md](SCHEDULERS.md)** - Complete scheduler parameter reference with examples

**Quick Example:**

```yaml
training:
  pipeline:
    steps:
      - name: "flow"
        n_epochs: 100
        train_diff: true
        optimization:
          optimizers:
            flow:
              type: "Lion"
              lr: 5e-7
              betas: [0.9, 0.95]
              weight_decay: 0.01
          schedulers:
            flow:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1
```

---

## Parameter Details

### Understanding VAE Dimensions

`vae_dim` is the number of latent channels per token (z; the packed latent is
`2*vae_dim`, z‖ctx).

v0.10.0 VAE widths are separate `model:` keys (each 4 stages, every width >= 8 and
divisible by 8):

| Key | Default |
|-----|---------|
| `vae_encoder_channels` | `[32, 64, 128, 256]` |
| `vae_decoder_channels` | `[256, 128, 64, 32]` |
| `vae_spade_hidden` | `[128, 128, 64, 32]` |

A smaller encoder, `vae_encoder_channels: [32, 64, 128, 128]`, is a valid lighter
fallback (~50M params) when the defaults do not fit.
Checkpoints load at their own widths (inferred from
the weights). The legacy (non-v0.10.0) VAE still derives its widths from `vae_dim`.

### Understanding Feature Map Dimensions

- `feature_maps_dim`: Controls the flow model's capacity
- `feature_maps_dim_disc`: Controls the discriminator's capacity

### Mixed Precision Training

`--use_fp16` enables mixed precision; `--precision {fp16,bf16}` (YAML
`training.precision`, default `fp16`) selects the dtype. bf16 recommended for v0.10.0: it
shares fp32's exponent range, while fp16 (~65504 max) can overflow to Inf as
activations grow, now that the Padé input clamps are removed. fp16 remains the default
and is accepted.

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

### KL normalization (history)

Since v0.3.0 KL is a per-element mean (resolution-invariant), not a sum. Current KL
settings: [v0.10.0 Loss Schedule](#v0100-loss-schedule).

## Pipeline Training Mode

Multi-step pipeline training allows you to define sequential training phases with different configurations.

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

**Model creation**: all models are built once at startup and shared by every step;
steps do not create models. See [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md#model-creation).

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
        transition_on:
          mode: "loss_threshold"
          metric: "recon_loss"
          threshold: 0.01
          max_epochs: 60         # safety upper bound
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
      # Step 1: VAE warmup with AdamW
      - name: "vae_warmup"
        n_epochs: 10
        train_vae: true
        gan_training: false
        optimization:
          optimizers:
            vae:
              type: "AdamW"
              lr: 1e-5
              weight_decay: 0.01
          schedulers:
            vae:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1

      # Step 2: GAN training with Lion
      - name: "gan_training"
        n_epochs: 30
        train_vae: true
        gan_training: true
        optimization:
          optimizers:
            vae:
              type: "Lion"
              lr: 5e-6
              weight_decay: 0.01
            discriminator:
              type: "AdamW"
              lr: 5e-6
              betas: [0.0, 0.9]
              amsgrad: true
          schedulers:
            vae:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1
            discriminator:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1

      # Step 3: Flow training (freeze the VAE)
      - name: "flow_training"
        n_epochs: 100
        train_diff_full: true
        freeze:
          - compressor
          - expander
        optimization:
          optimizers:
            flow:
              type: "Lion"
              lr: 5e-7
              weight_decay: 0.01
          schedulers:
            flow:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1
```

### Pipeline-Specific Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `steps[].name` | str | Unique step identifier (used in logs, checkpoints) |
| `steps[].n_epochs` | int | Epochs for this step only |
| `steps[].max_steps` | int | Optional: max batches per epoch (for testing) |
| `steps[].freeze` | list[str] | Components to freeze (e.g., `[compressor, expander]` to freeze VAE, `[flow_processor]` to freeze flow, `[text_encoder_backbone, text_encoder_projection]` to freeze text encoder) |
| `steps[].unfreeze` | list[str] | Components to explicitly unfreeze (overrides `freeze`) |
| `steps[].transition_on.mode` | str | `epoch` (default) or `loss_threshold` |
| `steps[].transition_on.metric` | str | Metric to monitor when `mode: loss_threshold` (e.g., `recon_loss`, `vae_loss`, `flow_loss`) |
| `steps[].transition_on.threshold` | float | Exit step when metric < threshold |
| `steps[].transition_on.max_epochs` | int | Safety upper bound for `loss_threshold` mode |

Per-step optimizers are configured via the inline `optimization:` block (see Example 2 above); external JSON optimizer configs are not part of the v0.10.0 schema.

Valid `freeze` / `unfreeze` components: `compressor`, `expander`, `flow_processor`, `flow`, `text_encoder`, `text_encoder_backbone`, `text_encoder_projection`, `discriminator`, `D_img`, `context_branch`.

### Pipeline Features

#### Per-Step Checkpoints
- Each step saves its own checkpoints: `flxflow_step_vae_warmup_final.safetensors`
- Resume from any step: automatically loads the last completed step
- Step-specific metrics files: `outputs/graph/training_metrics_vae_warmup.jsonl`

#### Selective Freezing
- Freeze any combination of models between steps
- Example: Train VAE in step 1, freeze it in step 2 for flow training
- Gradients automatically disabled for frozen models

#### Loss-Threshold Transitions
- Automatically exit step when monitored metric drops below `threshold`
- Useful for adaptive training (exit VAE warmup when reconstruction is good enough)
- Example: `transition_on: {mode: "loss_threshold", metric: "recon_loss", threshold: 0.01, max_epochs: 50}`

#### Inline Optimizer/Scheduler Configs
- Different optimizers per step (e.g., Adam warmup → Lion training)
- Different schedulers per step
- Full control over per-model hyperparameters

#### GAN-Only Training Mode
- Set `train_vae: false` together with `gan_training: true` so the encoder/decoder learns from adversarial loss only
- No pixel-level reconstruction loss computed
- Use case: spatial conditioning without reconstruction overhead (saves ~8-12GB VRAM)
- Example:
  ```yaml
  - name: "gan_only"
    train_vae: false
    gan_training: true
  ```

#### Full Resume Support
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
        lr: 2e-5
        kl_z_weight: 1e-5
        kl_z_warmup_steps: 10000
        transition_on:
          mode: "loss_threshold"
          metric: "recon_loss"
          threshold: 0.02       # Exit when reconstruction is good
          max_epochs: 20        # safety upper bound

      # Step 2: Add GAN (SPADE is always-on in v0.10.0)
      - name: "vae_gan"
        n_epochs: 40
        train_vae: true
        gan_training: true
        lr: 1e-5
        lambda_adv: 0.9
        kl_z_weight: 1e-5
        optimization:
          optimizers:
            vae:
              type: "Lion"
              lr: 5e-6
              weight_decay: 0.01
            discriminator:
              type: "AdamW"
              lr: 5e-6
              betas: [0.0, 0.9]
              amsgrad: true
          schedulers:
            vae:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1
            discriminator:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1

      # Step 3: Flow training (freeze VAE)
      - name: "flow_training"
        n_epochs: 100
        train_diff_full: true
        train_vae: false
        freeze:
          - compressor
          - expander
        lr: 5e-7
        sample_captions:
          - "a photo of a cat sitting on a couch"
          - "an illustration of mountains at sunset"
        optimization:
          optimizers:
            flow:
              type: "Lion"
              lr: 5e-7
              weight_decay: 0.01
          schedulers:
            flow:
              type: "CosineAnnealingLR"
              eta_min_factor: 0.1
```

**Run:**
```bash
fluxflow-train --config config.yaml
```

**Output structure:**
```
outputs/full_pipeline/
├── flxflow_step_vae_warmup_final.safetensors
├── flxflow_step_vae_gan_final.safetensors
├── flxflow_step_flow_training_final.safetensors
├── graph/
│   ├── training_metrics_vae_warmup.jsonl
│   ├── training_metrics_vae_gan.jsonl
│   ├── training_metrics_flow_training.jsonl
│   ├── training_losses_vae_warmup.png
│   ├── training_losses_vae_gan.png
│   └── training_losses_flow_training.png
└── samples/
    ├── sample_vae_warmup_epoch_5_batch_100.png
    ├── sample_vae_gan_epoch_20_batch_500.png
    └── sample_flow_training_epoch_50_batch_1000.png
```

### Pipeline vs. Standard Training

| Feature | Standard Training | Pipeline Training |
|---------|------------------|------------------|
| Configuration | CLI args | YAML config |
| Stages | Single mode | Multiple sequential steps |
| Per-step checkpoints | No | Yes |
| Per-step optimizers | No | Yes |
| Selective freezing | Manual | Per-step config |
| Loss-based transitions | Manual | Automatic |
| Hypothesis testing | Requires multiple runs | Single run |
| Resume mid-pipeline | No | Yes |

**Recommendation**: Use pipeline mode for production training, standard mode for quick tests.

### Troubleshooting Pipeline Training

**Issue**: Pipeline doesn't start / "No pipeline steps defined"
- **Solution**: Ensure `training.pipeline.steps` exists in YAML config
- **Check**: `steps` must be a list with at least one entry

**Issue**: Step checkpoint not found when resuming
- **Solution**: Pipeline automatically loads last completed step
- **Check**: Look for `flxflow_step_<name>_final.safetensors` in output directory

**Issue**: Loss-based transition never triggers
- **Solution**: Check that `transition_on.metric` matches a valid metric name
- **Valid metrics**: `vae`, `vae_loss`, `recon`, `recon_loss`, `kl`, `kl_loss`, `discriminator`, `generator`, `lpips`, `flow`, `flow_loss`
- **Check logs**: See current loss values in console output

**Issue**: Optimizer config not loading
- **Solution**: Per-step optimizers must be configured via the inline `optimization:` block on the step itself (not an external JSON path). Check that `optimization.optimizers.<name>.type` is one of the supported optimizer types and that `<name>` matches the model component you intend to train.

**Issue**: Models not freezing
- **Solution**: Use the `freeze:` list with valid component names (e.g., `freeze: [compressor, expander]` to freeze the VAE, `freeze: [flow_processor]` to freeze the flow, `freeze: [text_encoder_backbone, text_encoder_projection]` for the text encoder)
- **Check logs**: Should see "Freezing <component>" in output

---

## Training Strategies

### Recommended 3-Stage Training

> CLI-only runs build the legacy v0.6.0 VAE (see [Training workflow (v0.10.0)](#training-workflow-v0100)). For v0.10.0 use a YAML `model:` block; the CLI flags below then override YAML values.

#### Stage 1: VAE Pretraining (50-100 epochs)

**Goal**: Train a high-quality autoencoder

```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/stage1_vae \
  --train_vae \
  --n_epochs 50 \
  --batch_size 2 \
  --lr 1e-5 \
  --lr_min 0.1 \
  --vae_dim 128 \
  --feature_maps_dim 128 \
  --feature_maps_dim_disc 8 \
  --lambda_adv 0.9 \
  --kl_z_weight 1e-5 \
  --kl_z_warmup_steps 10000 \
  --kl_free_bits 0.5 \
  --checkpoint_save_interval 50 \
  --gan_training \
  --workers 8
```

**Monitoring:**
- Watch `loss_recon` (reconstruction loss): Should decrease to < 0.01
- Watch `kl`: logged after the per-channel free-bits floor, so it never drops below
  `kl_free_bits` (0.5); a value at the floor means every z channel is at the floor
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
  --train_diff_full \
  --n_epochs 20 \
  --batch_size 2 \
  --lr 1e-6 \
  --lr_min 0.1 \
  --vae_dim 128 \
  --feature_maps_dim 128 \
  --feature_maps_dim_disc 8 \
  --lambda_adv 0.5 \
  --kl_z_weight 1e-5 \
  --kl_free_bits 0.5 \
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

Pre-rework sizing (narrow VAE; this CLI-only run builds the legacy v0.6.0 VAE, where
`vae_dim` sets widths). v0.10.0 VRAM will differ.

For GPUs with limited VRAM:

```bash
fluxflow-train \
  --data_path /path/to/images \
  --captions_file /path/to/captions.txt \
  --output_path outputs/low_vram \
  --train_vae \
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
  --webdataset_token YOUR_HF_TOKEN \
  --webdataset_image_key jpg \
  --webdataset_caption_key prompt \
  --output_path outputs/webdataset_training \
  --train_vae \
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
  --n_epochs 100 \
  --batch_size 4 \
  --lr 2e-5 \
  --lr_min 0.05 \
  --vae_dim 256 \
  --feature_maps_dim 256 \
  --feature_maps_dim_disc 16 \
  --lambda_adv 0.9 \
  --kl_z_weight 1e-5 \
  --kl_z_warmup_steps 10000 \
  --kl_free_bits 0.5 \
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
fluxflow-train --generate_diagrams ...
```

**Manual diagram generation**:
```bash
python src/fluxflow_training/scripts/generate_training_graphs.py outputs/
```

Generated diagrams in `outputs/graph/`:
- `training_losses.png` - VAE, Flow, Discriminator, Generator, LPIPS losses
- `kl_loss.png` - KL divergence and its weight (`kl_z_weight` cosine warmup; KL logged after the per-channel free-bits floor)
- `learning_rates.png` - Learning rate schedules over time
- `batch_times.png` - Training speed (seconds/batch)
- `training_overview.png` - Combined overview with all metrics
- `training_summary.txt` - Statistical summary of training session

## Troubleshooting

### Out of Memory (OOM) Errors

See [Memory Requirements & OOM Prevention](#memory-requirements--oom-prevention) above for the authoritative list of mitigations (reduce batch size, disable LPIPS, reduce image size, reduce dimensions, enable FP16) and the per-GPU recommended configurations.

### NaN Losses

**Causes & Solutions:**
- **High learning rate**: Reduce `--lr` by 10x
- **Unstable GAN**: Reduce `--lambda_adv` or disable `--gan_training`
- **High KL weight**: Reduce `kl_z_weight` or increase `kl_z_warmup_steps`
- **Gradient explosion**: Reduce `--initial_clipping_norm` to 0.5

**Example fix:**
```bash
--lr 1e-6 --lambda_adv 0.3 --initial_clipping_norm 0.5
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
2. Widen the VAE: `vae_encoder_channels` / `vae_decoder_channels` (see [Understanding VAE Dimensions](#understanding-vae-dimensions))
3. Increase GAN weight: `--lambda_adv 0.9`
4. Reduce KL weight: `kl_z_weight`
5. Check sample images during training

### Slow Training Speed

**Solutions:**
1. Enable mixed precision: `--use_fp16 --precision bf16` (bf16 recommended for v0.10.0; fp16 default accepted)
2. Increase workers: `--workers 16`
3. Increase batch size: `--batch_size 4`
4. Reduce sample frequency: `--checkpoint_save_interval 200` or `--samples_per_checkpoint 5`
5. Use WebDataset streaming: `--use_webdataset`

### Text Conditioning Not Working

**Solutions:**
1. Check captions file format (tab-separated)
2. Keep the dataset tokenizer in sync with the text encoder. `BertTextEncoder` uses `distilbert-base-uncased`, but `TextImageDataset` defaults to `gpt2` (`src/fluxflow_training/data/datasets.py`); pass `--tokenizer_name distilbert-base-uncased` (or set it in the dataset constructor) so the token IDs match what the encoder expects.
3. If using newer Transformers, ensure tokenizers support `__call__` (batch fallback)
4. Train flow model longer: `--n_epochs 200`
5. Increase text embedding: `--text_embedding_dim 1024`

## Performance Benchmarks

Pre-rework (narrow VAE; `dim` is `vae_dim`). v0.10.0 numbers will differ.

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

*With `--use_fp16`, speeds improved by ~40% (pre-rework)*

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

