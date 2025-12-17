# VAE High Contrast / Saturation Issue - Solutions

## Problem Description
VAE reconstructions show:
- Very high contrast (dark areas become black, bright areas become vivid)
- Oversaturated colors compared to input images
- Loss of detail in shadows and highlights

## Root Causes

### 1. **TrainableBezier RGB Activation** (Primary Cause)
**Location:** `fluxflow-core/src/fluxflow/models/vae.py:482-487`

```python
self.rgb_activation = TrainableBezier(
    shape=(3,),  # R, G, B channels only
    p0=-1.0,  # Output range start
    p3=1.0,  # Output range end
    channel_only=True,
)
```

The final output goes through a learnable Bezier curve that can become too steep, causing contrast expansion.

### 2. **Frequency-Weighted Loss Emphasis**
**Location:** `fluxflow-training/src/fluxflow_training/training/vae_trainer.py:485`

```python
recon_l1 = self._frequency_weighted_loss(out_imgs_rec, real_imgs, alpha=1.0)
```

The `alpha=1.0` puts equal weight on high-frequency details, which can encourage high contrast.

### 3. **Missing Perceptual Loss Early On**
LPIPS perceptual loss is only added in step 2, but the model learns aggressive contrast in step 1.

## Solutions (Apply One or More)

### Solution 1: Add Tanh Clamping (Recommended - Easiest)

**File:** `fluxflow-core/src/fluxflow/models/vae.py`

Replace line 527:
```python
# Before (causes high contrast)
return self.rgb_activation(rgb)

# After (smooth clamping)
return torch.tanh(self.rgb_activation(rgb))
```

**Why it works:**
- `tanh()` smoothly clamps outputs to [-1, 1]
- Prevents extreme values even if Bezier curve is steep
- Still allows Bezier to learn color correction, just bounded

**Trade-off:** Slightly reduces dynamic range, but much better color fidelity.

### Solution 2: Reduce High-Frequency Weight

**File:** `fluxflow-training/src/fluxflow_training/training/vae_trainer.py`

Change line 485:
```python
# Before (equal emphasis on high-freq details)
recon_l1 = self._frequency_weighted_loss(out_imgs_rec, real_imgs, alpha=1.0)

# After (reduce high-freq emphasis)
recon_l1 = self._frequency_weighted_loss(out_imgs_rec, real_imgs, alpha=0.3)
```

**Why it works:**
- Reduces penalty for not matching high-frequency details exactly
- Encourages smoother reconstructions
- Less aggressive sharpening

**Trade-off:** May lose some fine detail, but better tonal range.

### Solution 3: Initialize Bezier to Identity

**File:** `fluxflow-core/src/fluxflow/models/vae.py`

After line 487, add initialization:
```python
self.rgb_activation = TrainableBezier(
    shape=(3,),
    p0=-1.0,
    p3=1.0,
    channel_only=True,
)

# NEW: Initialize to linear (identity) curve
with torch.no_grad():
    # Set p1 and p2 to create a linear curve from p0 to p3
    self.rgb_activation.p1.fill_(-0.33)  # 1/3 of the way
    self.rgb_activation.p2.fill_(0.33)   # 2/3 of the way
```

**Why it works:**
- Starts with no color distortion (linear mapping)
- Model learns gentle adjustments gradually
- Prevents extreme curves early in training

**Trade-off:** None! This is just better initialization.

### Solution 4: Add Color Histogram Loss

**File:** `fluxflow-training/src/fluxflow_training/training/vae_trainer.py`

Add new method around line 240:
```python
def _histogram_loss(self, pred, target, bins=64):
    """
    Encourage matching color distribution between pred and target.
    
    Args:
        pred: Predicted images [B, C, H, W]
        target: Target images [B, C, H, W]
        bins: Number of histogram bins
        
    Returns:
        Histogram matching loss
    """
    loss = 0.0
    for c in range(3):  # R, G, B
        # Compute histograms
        pred_hist = torch.histc(pred[:, c], bins=bins, min=-1, max=1)
        target_hist = torch.histc(target[:, c], bins=bins, min=-1, max=1)
        
        # Normalize to distribution
        pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
        target_hist = target_hist / (target_hist.sum() + 1e-8)
        
        # KL divergence between distributions
        loss += F.kl_div(
            pred_hist.log(), 
            target_hist, 
            reduction='sum'
        )
    
    return loss / 3.0  # Average over channels
```

Then in line 487, add:
```python
recon_loss = recon_l1 + self.mse_weight * recon_mse

# NEW: Add histogram loss
hist_loss = self._histogram_loss(out_imgs_rec, real_imgs)
recon_loss = recon_loss + 0.1 * hist_loss  # Weight of 0.1
```

**Why it works:**
- Directly penalizes different color distributions
- Prevents contrast expansion and saturation
- Encourages matching the tonal range of input

**Trade-off:** Adds computational cost (~5-10% slower).

### Solution 5: Start LPIPS Earlier

**File:** Your pipeline config YAML

Move LPIPS to step 1:
```yaml
pipeline:
  steps:
    - name: "vae_warmup"
      epochs: 1
      train_vae: true
      train_reconstruction: true
      use_gan: true
      use_spade: false
      use_lpips: true  # CHANGED: was false, now true
```

**Why it works:**
- LPIPS perceptual loss from VGG penalizes unnatural colors
- Catches contrast/saturation issues early
- Provides better perceptual guidance

**Trade-off:** Slightly slower (LPIPS is expensive), but much better results.

## Recommended Combination

**For Best Results, Apply:**
1. ✅ Solution 1 (Tanh clamping) - Immediate fix
2. ✅ Solution 3 (Better Bezier init) - No downside
3. ✅ Solution 5 (LPIPS from start) - Better perceptual quality

**Optional Add-ons:**
- Solution 2 if still too contrasty
- Solution 4 for maximum color fidelity (at cost of speed)

## Implementation Priority

### Quick Fix (5 minutes)
```python
# In fluxflow-core/src/fluxflow/models/vae.py, line 527
return torch.tanh(self.rgb_activation(rgb))
```

### Better Fix (15 minutes)
1. Add tanh clamping (above)
2. Initialize Bezier to linear (Solution 3)
3. Enable LPIPS in step 1 (config change)

### Best Fix (30 minutes)
1. All of the above
2. Add histogram loss (Solution 4)
3. Tune `alpha` in frequency loss (Solution 2)

## Testing

After applying fixes, check reconstructions:
```python
# Generate samples during training
# Look for:
# - Preserved shadow detail (not crushed to black)
# - Natural color saturation (not oversaturated)
# - Smooth tonal transitions (not posterized)
```

Compare before/after:
- Save sample with old model
- Apply fixes and retrain
- Compare side-by-side

## Additional Notes

### Why This Happens
1. L1/MSE losses don't penalize contrast expansion
2. High-frequency loss encourages sharp edges → high contrast
3. Bezier activation learns to "stretch" dynamic range
4. GAN discriminator may reward vivid colors

### Prevention
- Use perceptual losses (LPIPS) from the start
- Regularize Bezier control points
- Monitor color statistics during training
- Use histogram matching or color augmentation

## Related Issues
- Posterization (solved by same fixes)
- Color shifts (add histogram loss)
- Loss of detail in shadows (reduce alpha in freq loss)
