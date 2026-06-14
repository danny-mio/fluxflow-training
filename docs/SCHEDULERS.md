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
