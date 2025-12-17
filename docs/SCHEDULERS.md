# FluxFlow Scheduler Reference

This document provides detailed reference for all supported learning rate schedulers in FluxFlow training.

For training guide and usage examples, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

## Scheduler Configuration

Schedulers are configured in YAML or JSON format per model component:

```yaml
scheduler_config:
  vae_encoder:
    type: "CosineAnnealingLR"
    T_max: 100
    eta_min_factor: 0.1
  flow_processor:
    type: "LinearLR"
    start_factor: 1.0
    end_factor: 0.1
    total_iters: 1000
```

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

##### Sample Generation Behavior

**When samples are generated:**
- **VAE reconstruction samples** (`safe_vae_sample`): Generated when encoder/decoder is being trained
  - `train_vae=True`: VAE mode (reconstruction loss)
  - `gan_training=True`: GAN-only mode (adversarial loss without reconstruction)
  - `train_spade=True`: SPADE conditioning mode
- **Flow generation samples** (`save_sample_images`): Generated when flow model is being trained
  - `train_diff=True` or `train_diff_full=True`

**Sample frequency calculation:**
```
Samples generated every: checkpoint_save_interval × samples_per_checkpoint batches
```

**Example:**
- `checkpoint_save_interval=100` (checkpoint every 100 batches)
- `samples_per_checkpoint=50` (sample every 50 checkpoints)
- **Result**: Samples every **5,000 batches** (100 × 50)

**Sample filenames:**
- VAE: `vae_epoch_{global_step:04d}-{hash}-{suffix}.webp`
  - Example: `vae_epoch_05000-abc123-ctx.webp`, `vae_epoch_10000-def456-nr_o.webp`
  - Suffixes: `-ctx` (with context), `-nc` (no context), `_ns_i` (noise input), `_ns_o` (noise output), `-nr_o` (reconstructed output)
- Flow: `samples_epoch_{global_step:04d}_caption_{idx}-{size}.webp`
  - Example: `samples_epoch_05000_caption_0-0512.webp`, `samples_epoch_10000_caption_1-1024.webp`

**Note:** Filenames use `global_step` (total batch count) for unique, sequential numbering across entire training run.

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

