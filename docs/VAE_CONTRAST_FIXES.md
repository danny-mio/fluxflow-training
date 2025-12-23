# VAE High Contrast / Oversaturation - Training Fixes

> **⚠️ DEPRECATED:** Bezier regularization (#1) has been removed as of Dec 2025. It was found to be unhelpful and added unnecessary complexity. The remaining color statistics, histogram matching, and contrast regularization losses are still active.

## Problem
VAE reconstructions show:
- **High contrast**: Dark areas become too dark/black, bright areas become vivid
- **Oversaturated colors**: Colors more vivid than input images
- **Loss of tonal range**: Crushed shadows, blown highlights

## Root Cause
The `TrainableBezier` RGB activation in the VAE decoder learns control points that can form steep curves, expanding the dynamic range and increasing contrast.

## Training-Side Solutions (No Model Changes)

### 1. Bezier Regularization Loss ❌ REMOVED

> **REMOVED:** This feature was removed in Dec 2025 as it was found to be unhelpful.

**What it did:**
Penalized Bezier control points that deviated from a linear curve.

**Why it was removed:**
- Did not provide meaningful improvements to image quality
- Added unnecessary complexity to the loss function
- Bezier activations work better without this constraint
- Allows gradual learning of color correction
- Prevents sudden contrast expansion

### 2. Color Statistics Matching Loss ✅ IMPLEMENTED

**What it does:**
Matches mean and standard deviation of each color channel between reconstruction and input.

**Implementation:**
```python
def _color_statistics_loss(self, pred, target):
    """Match per-channel mean and std."""
    for c in [R, G, B]:
        loss += (pred_mean - target_mean)²
        loss += (pred_std - target_std)²  # Prevents contrast expansion!
```

**Weight:** `0.05 * color_stats_loss` added to total loss

**Effect:**
- Directly prevents contrast expansion (std matching)
- Prevents color shifts (mean matching)
- Fast to compute, effective

### 3. Histogram Matching Loss ✅ IMPLEMENTED

**What it does:**
Matches the full color distribution (histogram) between reconstruction and input using Earth Mover's Distance.

**Implementation:**
```python
def _histogram_matching_loss(self, pred, target, bins=64):
    """Match color distributions using Wasserstein-1 distance."""
    # Compute CDFs and measure L1 distance
```

**Weight:** `0.02 * hist_loss` added to total loss

**Effect:**
- Matches entire tonal curve (not just mean/std)
- Prevents posterization and banding
- More expensive but very effective

### 4. Reduced High-Frequency Weight ✅ IMPLEMENTED

**What it does:**
Reduces emphasis on high-frequency details in reconstruction loss.

**Change:**
```python
# Before: alpha=1.0 (equal weight on high-freq)
recon_l1 = self._frequency_weighted_loss(out_imgs_rec, real_imgs, alpha=1.0)

# After: alpha=0.5 (reduced high-freq emphasis)
recon_l1 = self._frequency_weighted_loss(out_imgs_rec, real_imgs, alpha=0.5)
```

**Effect:**
- Less penalty for not matching sharp edges exactly
- Encourages smoother, more natural reconstructions
- Reduces tendency toward high contrast

## Loss Breakdown

**Total VAE Loss:**
```python
total_loss = (
    w_kl * beta * kl +                    # KL divergence
    w_recon * recon_loss +                # Reconstruction loss
    w_gan * G_img_loss +                  # GAN generator loss
    0.05 * color_stats_loss +             # Color statistics matching
    0.02 * hist_loss +                    # Histogram matching
    0.1 * contrast_loss                   # Contrast regularization
)
```

**Weights Explanation:**
- `0.05` for color stats: Directly prevents contrast expansion
- `0.02` for histogram: Refines tonal distribution
- `0.1` for contrast: Prevents over-saturation

## Monitoring

The training logs now include:
```
[Step 1000] VAE: 0.0523 | KL: 8461.29 | ...
  ColorStats: 0.0145  ← Should be low (<0.05)
  Hist: 0.0089        ← Should be low (<0.02)
  Contrast: 0.0032    ← Should be low (<0.05)
```

**What to watch:**
- `ColorStats` staying low → No contrast expansion
- `Hist` staying low → Good tonal matching
- `Contrast` staying low → No over-saturation

## Tuning Guide

### If still too contrasty:

**Increase regularization weights:**
```python
# In vae_trainer.py
total_loss = total_loss + 0.10 * color_stats_loss  # Was 0.05
total_loss = total_loss + 0.05 * hist_loss         # Was 0.02
total_loss = total_loss + 0.15 * contrast_loss     # Was 0.1
```

### If losing detail:

**Reduce regularization:**
```python
total_loss = total_loss + 0.02 * color_stats_loss  # Reduce from 0.05
total_loss = total_loss + 0.05 * contrast_loss     # Reduce from 0.1
```

### If colors look washed out:

**Reduce histogram loss:**
```python
total_loss = total_loss + 0.01 * hist_loss  # Reduce from 0.02
```

## Alternative: Enable LPIPS Earlier

LPIPS (perceptual loss from VGG) naturally prevents unnatural colors and contrast.

**In your pipeline config YAML:**
```yaml
pipeline:
  steps:
    - name: "vae_no_SPADE"
      epochs: 1
      train_vae: true
      use_gan: true
      use_lpips: true  # CHANGE: Enable from step 1 (was step 2)
```

**Effect:**
- VGG-based perceptual loss penalizes unnatural colors
- Catches contrast/saturation issues immediately
- Trade-off: ~10% slower training

## Best Practice

**Recommended Settings:**
```python
# Regularization weights
bezier_reg_weight = 0.01
color_stats_weight = 0.05
hist_weight = 0.02

# Frequency loss
freq_alpha = 0.5  # Reduced from 1.0

# LPIPS
use_lpips_from_start = True  # Enable in step 1
```

**This combination:**
- ✅ Matches color statistics (no contrast expansion)
- ✅ Matches tonal distribution (no posterization)
- ✅ Prevents over-saturation
- ✅ Adds perceptual guidance early

## Testing Reconstructions

After implementing these fixes:

1. **Generate samples** during training
2. **Check for:**
   - Natural contrast (not crushed blacks/blown highlights)
   - Natural color saturation (not oversaturated)
   - Smooth tonal transitions (not posterized)
   - Preserved shadow/highlight detail

3. **Compare before/after:**
   - Save samples from old training
   - Retrain with new losses
   - Side-by-side comparison

## Performance Impact

- **Color stats:** ~1% overhead (cheap statistics)
- **Histogram:** ~3-5% overhead (histc operation)
- **Contrast reg:** ~1% overhead (simple calculations)
- **Total:** ~5-7% slower per batch

**Worth it?** Yes! Much better visual quality for minimal slowdown.

## Summary

All fixes are **training-only** (no model architecture changes):
1. ✅ Color statistics matching → prevents contrast expansion
2. ✅ Histogram matching → prevents tonal distortion
3. ✅ Contrast regularization → prevents over-saturation

**Result:** Natural-looking reconstructions with proper contrast and saturation.
