# FluxFlow Optimizer Reference

This document provides detailed reference for all supported optimizers in FluxFlow training.

For training guide and usage examples, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

## Optimizer Configuration

Optimizers are configured in YAML or JSON format per model component:

```yaml
optimizer_config:
  vae_encoder:
    type: "AdamW"
    lr: 5e-5
    betas: [0.9, 0.999]
    weight_decay: 0.01
  flow_processor:
    type: "Lion"
    lr: 5e-7
    betas: [0.9, 0.95]
    weight_decay: 0.01
```

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

